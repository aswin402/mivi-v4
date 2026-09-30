//! Memory-only replay of actual prefill inputs; never compiled into inference.

use super::super::{
    packed_matmul, packed_matmul_with_kernel, PackedKernel, Q4_K_BLOCK_SIZE, Q4_K_BYTES,
};
use super::{
    positive_env, supported_shape, validate_sampled_error_bound, EvalResult, ProjectionError,
};
use crate::{dequantize_slice, quantized_matmul_rows, GgmlType};
use mivi_model::{LayerWeights, Model, PrefillStrategy, QuantizedTensor, TileActivations};

mod cumulative;
mod layer_trace;

#[derive(Clone, Copy, PartialEq, Eq)]
enum WalkMode {
    Replay,
    Exact,
    SingleDown(usize),
    DownOnly,
    FullFfn,
    FullFfnControl,
}

#[derive(Default)]
struct Coverage {
    packed: usize,
    fallback: usize,
}

#[test]
#[ignore = "requires MIVI_TEST_MODEL; captures real prefill activations"]
fn captured_prefill_projection_evaluation() -> EvalResult<()> {
    assert!(evaluate()? > 0, "no captured Q4 projections evaluated");
    Ok(())
}

fn evaluate() -> EvalResult<usize> {
    let path = std::env::var_os("MIVI_TEST_MODEL").ok_or("set MIVI_TEST_MODEL")?;
    let max_tokens = positive_env("MIVI_TEST_CAPTURE_TOKENS", 32)?;
    // One tile only: this diagnostic deliberately does not become a long-context run.
    if max_tokens > 64 {
        return Err("MIVI_TEST_CAPTURE_TOKENS must be at most 64".into());
    }
    let row_cap = positive_env("MIVI_TEST_MAX_ROWS", 1024)?;
    let mut model = Model::load_with_ctx(std::path::Path::new(&path), Some(max_tokens + 1))?;
    let prompt = match std::env::var("MIVI_TEST_CAPTURE_PROMPT") {
        Ok(prompt) => prompt,
        Err(std::env::VarError::NotPresent) => {
            "Explain how to safely read a file and handle an error in Rust.".to_owned()
        }
        Err(error) => return Err(error.into()),
    };
    let mut tokens = prompt_tokens(&model, &prompt)?;
    let original_len = tokens.len();
    tokens.truncate(max_tokens);
    if tokens.is_empty() {
        return Err("capture prompt encoded to no tokens".into());
    }
    let batch = tokens.len();
    println!("captured prefill: tokens={batch}, original tokens={original_len}, row cap={row_cap}, Rayon threads={}; prompt/activations are not printed or written", rayon::current_num_threads());
    model.set_prefill_strategy(PrefillStrategy::Chunked {
        tile_tokens: max_tokens,
    })?;
    model.generate_tokens_incremental(&tokens, 0, 0, |_, _| true)?;
    let production_logits = model.state.logits.to_vec();
    let (tested, residual) = walk(&mut model, &tokens, row_cap, WalkMode::Replay)?;
    let baseline = logits(&model, &residual)?;
    let control = ProjectionError::compare(&production_logits, &baseline)?;
    println!(
        "production chunked control: relative L2={:.8}, max abs={:.8}",
        control.relative_l2, control.max_abs
    );
    assert!(
        control.max_abs < 1e-3,
        "capture walker disagrees with production chunked prefill"
    );
    // Select by executed layer metadata and quantization format, not model family.
    let selected = model
        .weights
        .layers
        .iter()
        .enumerate()
        .rev()
        .find_map(|(index, layer)| {
            let down = match layer {
                LayerWeights::Attention(w) => &w.ffn.w_down,
                LayerWeights::Ssm(w) if model.config.ssm_conv_kernel > 0 => &w.ffn.w_down,
                LayerWeights::Ssm(_) => return None,
            };
            supported_shape(down.quant_type as u32, &[down.cols, down.rows]).then_some(index)
        });
    if let Some(index) = selected {
        let (_, changed) = walk(&mut model, &tokens, row_cap, WalkMode::SingleDown(index))?;
        let changed_logits = logits(&model, &changed)?;
        let error = ProjectionError::compare(&baseline, &changed_logits)?;
        println!("layer={index} down-only residual-delta logit sensitivity: relative L2={:.6}, max abs={:.6}, greedy token {} -> {}; NOT full packed inference or a quality acceptance gate", error.relative_l2, error.max_abs, top_token(&baseline)?, top_token(&changed_logits)?);
    } else {
        println!("no supported executed down projection; no logit perturbation evaluated");
    }
    Ok(tested.packed)
}

fn prompt_tokens(model: &Model, prompt: &str) -> EvalResult<Vec<u32>> {
    let mut tokens = model.tokenizer.encode(prompt);
    if matches!(
        model.gguf.metadata.get("tokenizer.ggml.add_bos_token"),
        Some(mivi_model::GgufValue::Bool(true))
    ) {
        let bos = model
            .gguf
            .metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|value| value.as_usize())
            .and_then(|value| u32::try_from(value).ok())
            .ok_or("GGUF requests BOS but has no valid BOS token ID")?;
        if tokens.first() != Some(&bos) {
            tokens.insert(0, bos);
        }
    }
    Ok(tokens)
}

fn walk(
    model: &mut Model,
    tokens: &[u32],
    row_cap: usize,
    mode: WalkMode,
) -> EvalResult<(Coverage, Vec<f32>)> {
    walk_observed(model, tokens, row_cap, mode, |_, _, _| Ok(()))
}

fn walk_observed<F>(
    model: &mut Model,
    tokens: &[u32],
    row_cap: usize,
    mode: WalkMode,
    mut observe: F,
) -> EvalResult<(Coverage, Vec<f32>)>
where
    F: FnMut(usize, &mut TileActivations, &[f32]) -> EvalResult<()>,
{
    model.reset_context();
    let batch = tokens.len();
    let cfg = &model.config;
    let mut tile = TileActivations::with_kv_dim(batch, cfg.dim, cfg.hidden_dim, cfg.kv_dim)?;
    let embedding = model.weights.token_embd;
    let kind = GgmlType::from_u32(embedding.quant_type as u32)?;
    let block = kind
        .block_size()
        .ok_or("unsupported embedding block size")?;
    let size = kind.type_size().ok_or("unsupported embedding type size")?;
    if embedding.cols != cfg.dim || !cfg.dim.is_multiple_of(block) {
        return Err("embedding dimensions are not supported".into());
    }
    let row_bytes = (cfg.dim / block)
        .checked_mul(size)
        .ok_or("embedding size overflow")?;
    let bytes = embedding
        .as_slice_checked(&model.gguf.mmap)
        .ok_or("embedding outside GGUF")?;
    for (row, &token) in tokens.iter().enumerate() {
        if token as usize >= embedding.rows {
            return Err("token outside embedding vocabulary".into());
        }
        let start = (token as usize)
            .checked_mul(row_bytes)
            .ok_or("embedding offset overflow")?;
        let end = start
            .checked_add(row_bytes)
            .ok_or("embedding end overflow")?;
        dequantize_slice(
            kind,
            bytes
                .get(start..end)
                .ok_or("embedding row outside tensor")?,
            tile.current_row_mut(row)?,
        )?;
    }
    let mut coverage = Coverage::default();
    for (index, layer) in model.weights.layers.iter().enumerate() {
        let ffn = match layer {
            LayerWeights::Attention(w) => {
                mivi_model::attention_forward_tile(
                    &mut tile,
                    &mut model.state,
                    &mut model.kv_cache,
                    &mivi_model::AttentionParams {
                        layer: index,
                        pos: 0,
                        weights: w,
                        mmap: &model.gguf.mmap,
                        config: cfg,
                        adapters: &model.active_adapters,
                        rope: &model.rope_cache,
                    },
                    0,
                    batch,
                )?;
                &w.ffn
            }
            LayerWeights::Ssm(w) => {
                mivi_model::ssm_forward_tile(
                    &mut tile,
                    &mut model.state,
                    &mivi_model::SsmParams {
                        layer: index,
                        weights: w,
                        mmap: &model.gguf.mmap,
                        config: cfg,
                        adapters: &model.active_adapters,
                    },
                    batch,
                )?;
                if cfg.ssm_conv_kernel == 0 {
                    println!("layer={index}: SSM branch skipped FFN; no stale scratch replay");
                    let before_delta = tile.final_current_row(batch)?.to_vec();
                    observe(index, &mut tile, &before_delta)?;
                    continue;
                }
                &w.ffn
            }
        };
        // Exact layer output for the CURRENT (possibly perturbed) layer input.
        // Observers can separate propagated error from this layer's delta injection.
        let before_delta = tile.final_current_row(batch)?.to_vec();
        for (name, weight, down_input) in [
            (&ffn.ffn_gate_name, &ffn.w_gate, false),
            (&ffn.ffn_up_name, &ffn.w_up, false),
            (&ffn.ffn_down_name, &ffn.w_down, true),
        ] {
            if mode != WalkMode::Replay {
                continue;
            }
            if !supported_shape(weight.quant_type as u32, &[weight.cols, weight.rows]) {
                coverage.fallback += 1;
                continue;
            }
            let input = if down_input {
                tile.gate()
            } else {
                tile.norm_mut()
            };
            replay(name, weight, &model.gguf.mmap, input, batch, row_cap)?;
            coverage.packed += 1;
        }
        // Isolate one down projection; subsequent layers remain F32-activation inference.
        // Perturb ALL token rows so subsequent attention/SSM sees coherent changed state.
        if mode == WalkMode::SingleDown(index) {
            let mut packed = vec![
                0.0;
                batch
                    .checked_mul(ffn.w_down.rows)
                    .ok_or("down shape overflow")?
            ];
            packed_matmul_with_kernel(
                &mut packed,
                ffn.w_down.as_slice(&model.gguf.mmap),
                tile.gate(),
                batch,
                ffn.w_down.rows,
                ffn.w_down.cols,
                PackedKernel::detected_tiled(),
            )?;
            if ffn.w_down.rows != cfg.dim {
                return Err("FFN down output does not match residual dimension".into());
            }
            let changed = residual_delta(tile.current(), tile.next(), &packed)?;
            tile.current_mut().copy_from_slice(&changed);
        }
        if matches!(
            mode,
            WalkMode::DownOnly | WalkMode::FullFfn | WalkMode::FullFfnControl
        ) {
            let full = mode != WalkMode::DownOnly;
            let packed = mode != WalkMode::FullFfnControl;
            let recomputed =
                cumulative::recompute(&mut tile, ffn, &model.gguf.mmap, batch, full, packed)?;
            coverage.packed += recomputed.packed;
            coverage.fallback += recomputed.fallback;
            let changed = residual_delta(tile.current(), tile.next(), &recomputed.output)?;
            tile.current_mut().copy_from_slice(&changed);
        }
        observe(index, &mut tile, &before_delta)?;
    }
    if mode == WalkMode::Replay {
        println!("captured Q4 projections={}, unsupported FFN projections={}; no timing/agent readiness claim", coverage.packed, coverage.fallback);
    }
    Ok((coverage, tile.final_current_row(batch)?.to_vec()))
}

fn replay(
    name: &str,
    weight: &QuantizedTensor,
    mmap: &[u8],
    input: &[f32],
    batch: usize,
    row_cap: usize,
) -> EvalResult<()> {
    let rows = weight.rows.min(row_cap);
    let row_bytes = (weight.cols / Q4_K_BLOCK_SIZE)
        .checked_mul(Q4_K_BYTES)
        .ok_or("Q4 row overflow")?;
    let length = rows
        .checked_mul(row_bytes)
        .ok_or("weight prefix overflow")?;
    let bytes = weight
        .as_slice_checked(mmap)
        .ok_or("weight outside GGUF")?
        .get(..length)
        .ok_or("weight prefix outside tensor")?;
    let mut reference = vec![0.0; batch.checked_mul(rows).ok_or("projection shape overflow")?];
    let mut scalar = reference.clone();
    let mut tiled = reference.clone();
    quantized_matmul_rows(
        &mut reference,
        GgmlType::Q4_K,
        bytes,
        input,
        batch,
        rows,
        weight.cols,
    )?;
    packed_matmul(&mut scalar, bytes, input, batch, rows, weight.cols)?;
    packed_matmul_with_kernel(
        &mut tiled,
        bytes,
        input,
        batch,
        rows,
        weight.cols,
        PackedKernel::detected_tiled(),
    )?;
    assert_eq!(scalar, tiled, "captured scalar/tiled mismatch for {name}");
    validate_sampled_error_bound(bytes, input, &reference, &tiled, batch, rows, weight.cols)?;
    let error = ProjectionError::compare(&reference, &tiled)?;
    println!(
        "  {name}: rows={rows}/{}, cols={}, relative L2={:.6}, max abs={:.6}",
        weight.rows, weight.cols, error.relative_l2, error.max_abs
    );
    Ok(())
}

fn logits(model: &Model, residual: &[f32]) -> EvalResult<Vec<f32>> {
    let mut normalized = residual.to_vec();
    if let Some(norm) = &model.weights.output_norm {
        mivi_model::rms_norm_rows(
            &mut normalized,
            residual,
            norm,
            1,
            model.config.dim,
            model.config.rms_norm_eps,
        )?;
    }
    let head = model
        .weights
        .output_proj
        .as_ref()
        .unwrap_or(&model.weights.token_embd);
    let mut out = vec![0.0; model.config.vocab_size];
    quantized_matmul_rows(
        &mut out,
        GgmlType::from_u32(head.quant_type as u32)?,
        head.as_slice(&model.gguf.mmap),
        &normalized,
        1,
        model.config.vocab_size,
        model.config.dim,
    )?;
    Ok(out)
}

fn residual_delta(residual: &[f32], exact: &[f32], packed: &[f32]) -> EvalResult<Vec<f32>> {
    if residual.len() != exact.len() || exact.len() != packed.len() {
        return Err("residual delta length mismatch".into());
    }
    // Rounding differs from replacing the projection before the residual add.
    let out: Vec<_> = residual
        .iter()
        .zip(exact)
        .zip(packed)
        .map(|((&x, &old), &new)| x + (new - old))
        .collect();
    ProjectionError::compare(residual, &out)?;
    Ok(out)
}

fn top_token(logits: &[f32]) -> EvalResult<usize> {
    if logits.is_empty() || logits.iter().any(|x| !x.is_finite()) {
        return Err("invalid logits for greedy token".into());
    }
    // Match production argmax tie behavior (first equal maximum).
    Ok(logits.iter().enumerate().fold(
        0,
        |best, (i, value)| if *value > logits[best] { i } else { best },
    ))
}

#[test]
fn residual_delta_identity_and_invalid_inputs() {
    assert_eq!(
        residual_delta(&[1.0, -2.0], &[3.0, 4.0], &[3.0, 4.0]).unwrap(),
        [1.0, -2.0]
    );
    assert!(residual_delta(&[1.0], &[], &[]).is_err());
    assert!(residual_delta(&[1.0], &[0.0], &[f32::INFINITY]).is_err());
    assert_eq!(top_token(&[1.0, 3.0, 3.0]).unwrap(), 1);
    assert!(top_token(&[]).is_err());
    assert!(top_token(&[f32::NAN]).is_err());
}
