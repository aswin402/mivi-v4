//! Memory-only actual prefill activation replay; never wired into inference.

use super::super::{quantized_matmul_rows_four_row, BatchProjectionScratch, GgmlType};
use super::{require_benchmark_environment, validate_exact, EvalResult};
use crate::batch_scratch::quantized_matmul_rows_with_scratch;
use mivi_model::{
    GgufValue, LayerWeights, Model, PrefillStrategy, QuantizedTensor, TileActivations,
};
use std::hint::black_box;
use std::time::Instant;

mod stages;

struct Capture {
    layer: usize,
    role: &'static str,
    tensor: QuantizedTensor,
    inputs: Vec<f32>,
}

fn eligible(tensor: &QuantizedTensor) -> bool {
    matches!(tensor.quant_type as u32, n if n == GgmlType::Q4_K as u32 || n == GgmlType::Q6_K as u32)
        && tensor.cols > 0
        && tensor.cols <= 16384
        && tensor.cols.is_multiple_of(256)
        && tensor.rows > 0
        && tensor.rows <= 8192
}

fn prompt_within_budget(prompt: &str) -> bool {
    // Diagnostic input-work cap, independent of model name/context capacity.
    prompt.len() <= 64 * 1024
}

#[test]
fn faithful_diagnostic_capture_prompt_budget() {
    assert!(prompt_within_budget(""));
    assert!(prompt_within_budget(&"x".repeat(64 * 1024)));
    assert!(!prompt_within_budget(&"x".repeat(64 * 1024 + 1)));
    assert!(!prompt_within_budget(&"é".repeat(32 * 1024 + 1)));
}

fn prepare_capture() -> EvalResult<(Model, Vec<u32>)> {
    require_benchmark_environment()?;
    let path = std::env::var_os("MIVI_TEST_MODEL").ok_or("set MIVI_TEST_MODEL")?;
    let model = Model::load_with_ctx(std::path::Path::new(&path), Some(65))?;
    if !model.active_adapters.active.is_empty() {
        return Err("capture rejects active adapters".into());
    }
    let prompt = match std::env::var("MIVI_TEST_CAPTURE_PROMPT") {
        Ok(prompt) => prompt,
        Err(std::env::VarError::NotPresent) => "Review this Rust workspace. Read the source files, explain the implementation, and propose safe changes with focused tests. Preserve unrelated changes and do not execute destructive commands. ".repeat(8),
        Err(_) => return Err("capture prompt must be valid Unicode".into()),
    };
    if !prompt_within_budget(&prompt) {
        return Err("capture prompt exceeds 64KiB UTF-8 tokenization budget".into());
    }
    let mut ids = model.tokenizer.encode(&prompt);
    if matches!(
        model.gguf.metadata.get("tokenizer.ggml.add_bos_token"),
        Some(GgufValue::Bool(true))
    ) {
        let bos = model
            .gguf
            .metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(GgufValue::as_usize)
            .and_then(|n| u32::try_from(n).ok())
            .ok_or("missing valid metadata BOS")?;
        if ids.first() != Some(&bos) {
            ids.insert(0, bos);
        }
    }
    if ids.len() < 64 {
        return Err("capture prompt must encode at least 64 effective IDs".into());
    }
    Ok((model, ids))
}

fn gated_captures(model: &mut Model, ids: &[u32], batch: usize) -> EvalResult<Vec<Capture>> {
    if !matches!(batch, 32 | 64) || ids.len() < batch {
        return Err("invalid capture origin batch".into());
    }
    model.reset_context();
    model.set_prefill_strategy(PrefillStrategy::Chunked { tile_tokens: batch })?;
    model.generate_tokens_incremental(&ids[..batch], 0, 0, |_, _| true)?;
    if model.current_pos() != batch {
        return Err("production prefill did not process complete effective prompt".into());
    }
    let production = model.state.logits.to_vec();
    let (walk_logits, captures) = capture_walk(model, &ids[..batch])?;
    validate_exact(&production, &walk_logits)?;
    if captures.len() != 2 {
        return Err("capture requires one executed eligible Q4_K and Q6_K projection".into());
    }
    println!("CAPTURE_GATE batch={batch} complete_logits={} bits=exact captured_formats=2 context=65 prompt_ids_activations=memory-only candidate_model_injection=false", production.len());
    Ok(captures)
}

#[test]
#[ignore = "requires MIVI_TEST_MODEL; exact baseline gate before captured timings"]
fn four_row_captured_activation() -> EvalResult<()> {
    let (mut model, ids) = prepare_capture()?;
    for batch in [32, 64] {
        let captures = gated_captures(&mut model, &ids, batch)?;
        for capture in &captures {
            replay(&model, capture, batch)?;
        }
    }
    Ok(())
}

fn capture_walk(model: &mut Model, ids: &[u32]) -> EvalResult<(Vec<f32>, Vec<Capture>)> {
    model.reset_context();
    let batch = ids.len();
    let cfg = &model.config;
    let mut tile = TileActivations::with_kv_dim(batch, cfg.dim, cfg.hidden_dim, cfg.kv_dim)?;
    let embedding = model.weights.token_embd;
    let kind = GgmlType::from_u32(embedding.quant_type as u32)?;
    let block = kind.block_size_checked()?;
    if embedding.cols != cfg.dim || !cfg.dim.is_multiple_of(block) {
        return Err("invalid embedding shape".into());
    }
    let row_bytes = (cfg.dim / block)
        .checked_mul(kind.type_size_checked()?)
        .ok_or("embedding overflow")?;
    let bytes = embedding
        .as_slice_checked(&model.gguf.mmap)
        .ok_or("embedding outside GGUF")?;
    for (row, &id) in ids.iter().enumerate() {
        if id as usize >= embedding.rows {
            return Err("embedding ID outside vocabulary".into());
        }
        let start = (id as usize)
            .checked_mul(row_bytes)
            .ok_or("embedding offset overflow")?;
        let end = start
            .checked_add(row_bytes)
            .ok_or("embedding end overflow")?;
        crate::dequantize_slice(
            kind,
            bytes.get(start..end).ok_or("short embedding storage")?,
            tile.current_row_mut(row)?,
        )?;
    }
    let mut captures: Vec<Capture> = Vec::new();
    for (layer, weights) in model.weights.layers.iter().enumerate() {
        let ffn = match weights {
            LayerWeights::Attention(w) => {
                mivi_model::attention_forward_tile(
                    &mut tile,
                    &mut model.state,
                    &mut model.kv_cache,
                    &mivi_model::AttentionParams {
                        layer,
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
                        layer,
                        weights: w,
                        mmap: &model.gguf.mmap,
                        config: cfg,
                        adapters: &model.active_adapters,
                    },
                    batch,
                )?;
                // Mirrors the production branch: no FFN occurred in this case.
                if cfg.ssm_conv_kernel == 0 {
                    continue;
                }
                &w.ffn
            }
        };
        for (role, tensor, down) in [
            ("gate", &ffn.w_gate, false),
            ("up", &ffn.w_up, false),
            ("down", &ffn.w_down, true),
        ] {
            if !eligible(tensor)
                || captures
                    .iter()
                    .any(|c| c.tensor.quant_type as u32 == tensor.quant_type as u32)
            {
                continue;
            }
            let input = if down { tile.gate() } else { tile.norm_mut() };
            let len = batch
                .checked_mul(tensor.cols)
                .ok_or("capture activation overflow")?;
            let input = input.get(..len).ok_or("short capture activation")?;
            if input.iter().any(|v| !v.is_finite()) {
                return Err("non-finite capture activation".into());
            }
            captures.push(Capture {
                layer,
                role,
                tensor: *tensor,
                inputs: input.to_vec(),
            });
        }
    }
    // Use precisely the production final norm and linear helper (no adapters).
    model
        .state
        .x
        .copy_from_slice(tile.final_current_row(batch)?);
    if let Some(norm) = &model.weights.output_norm {
        mivi_core::simd::rms_norm_simd(&mut model.state.xb, &model.state.x, norm, cfg.rms_norm_eps);
    } else {
        model.state.xb.copy_from_slice(&model.state.x);
    }
    let head = model
        .weights
        .output_proj
        .as_ref()
        .unwrap_or(&model.weights.token_embd);
    let mut logits = vec![0.0; cfg.vocab_size];
    mivi_model::linear_forward(
        &mut logits,
        &mivi_model::LinearParams {
            weight: head,
            input: &model.state.xb,
            rows: cfg.vocab_size,
            cols: cfg.dim,
            mmap: &model.gguf.mmap,
            adapters: &model.active_adapters,
            module_name: "output",
        },
        &mut model.state.lora_down,
    )?;
    Ok((logits, captures))
}

fn replay(model: &Model, capture: &Capture, batch: usize) -> EvalResult<()> {
    let t = &capture.tensor;
    let kind = GgmlType::from_u32(t.quant_type as u32)?;
    let weights = t
        .as_slice_checked(&model.gguf.mmap)
        .ok_or("capture weights outside GGUF")?;
    let mut scratch = BatchProjectionScratch::new(batch, t.rows, t.cols, 2)?;
    let mut four = BatchProjectionScratch::new_four_row(batch, t.rows, t.cols, 2)?;
    let mut out = std::array::from_fn::<_, 3, _>(|_| vec![0.0; batch * t.rows]);
    let mut reference = vec![0.0; batch * t.rows];
    crate::quantized_matmul_rows(
        &mut reference,
        kind,
        weights,
        &capture.inputs,
        batch,
        t.rows,
        t.cols,
    )?;
    validate_exact(&reference, &reference)?;
    let orders = [
        [0, 1, 2],
        [2, 1, 0],
        [1, 2, 0],
        [0, 2, 1],
        [2, 0, 1],
        [1, 0, 2],
    ];
    for round in 0..=6 {
        let order = if round == 0 {
            [0, 1, 2]
        } else {
            orders[round - 1]
        };
        let mut times = [0u128; 3];
        for route in order {
            let output = &mut out[route];
            let start = Instant::now();
            for _ in 0..3 {
                match route {
                    0 => crate::quantized_matmul_rows(
                        black_box(output),
                        kind,
                        black_box(weights),
                        black_box(&capture.inputs),
                        batch,
                        t.rows,
                        t.cols,
                    )?,
                    1 => quantized_matmul_rows_with_scratch(
                        black_box(output),
                        kind,
                        black_box(weights),
                        black_box(&capture.inputs),
                        batch,
                        t.rows,
                        t.cols,
                        &mut scratch,
                    )?,
                    2 => quantized_matmul_rows_four_row(
                        black_box(output),
                        kind,
                        black_box(weights),
                        black_box(&capture.inputs),
                        batch,
                        t.rows,
                        t.cols,
                        &mut four,
                    )?,
                    _ => unreachable!(),
                }
            }
            times[route] = start.elapsed().as_nanos();
            assert!(times[route] > 0);
            validate_exact(&reference, output)?;
        }
        println!("CAPTURE layer={} role={} kind={kind:?} rows={} cols={} batch={batch} round={round} order={order:?} times_ns={times:?}", capture.layer, capture.role, t.rows, t.cols);
    }
    Ok(())
}
