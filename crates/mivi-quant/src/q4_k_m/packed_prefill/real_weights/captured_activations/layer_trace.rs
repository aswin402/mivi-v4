//! Layer-wise sensitivity tracing; no production instrumentation or tensor dumps.

use super::super::super::ActivationCodec;
use super::{EvalResult, LayerWeights, Model, PrefillStrategy, WalkMode, Q4_K_BLOCK_SIZE};

struct ActivationStats {
    relative_l2: f64,
    peak_over_rms: f64,
}

fn activation_stats(
    input: &[f32],
    width: usize,
    codec: ActivationCodec,
) -> EvalResult<Option<ActivationStats>> {
    if width == 0
        || input.is_empty()
        || !input.len().is_multiple_of(width)
        || input.iter().any(|x| !x.is_finite())
    {
        return Err("invalid activation shape or non-finite input".into());
    }
    if !width.is_multiple_of(Q4_K_BLOCK_SIZE) {
        return Ok(None);
    }
    let mut reference_squared = 0.0;
    let mut error_squared = 0.0;
    let mut peak = 0.0f64;
    for block in input.chunks_exact(Q4_K_BLOCK_SIZE) {
        let reconstructed = codec.reconstruct(block)?;
        for (&original, reconstructed) in block.iter().zip(reconstructed) {
            let value = f64::from(original);
            let reconstructed = f64::from(reconstructed);
            reference_squared += value * value;
            error_squared += (reconstructed - value).powi(2);
            peak = peak.max(value.abs());
        }
    }
    let rms = (reference_squared / input.len() as f64).sqrt();
    let relative_l2 = (error_squared / reference_squared.max(f64::MIN_POSITIVE)).sqrt();
    let peak_over_rms = if rms > 0.0 { peak / rms } else { 0.0 };
    if !relative_l2.is_finite() || !peak_over_rms.is_finite() {
        return Err("non-finite activation metrics".into());
    }
    Ok(Some(ActivationStats {
        relative_l2,
        peak_over_rms,
    }))
}

#[test]
fn layer_trace_activation_stats_measure_actual_packing_error() {
    let width = super::Q4_K_BLOCK_SIZE;
    let mut input = vec![0.25; width];
    input[0] = 12.0;
    for codec in [ActivationCodec::Group256, ActivationCodec::Group32] {
        let stats = activation_stats(&input, width, codec).unwrap().unwrap();
        assert!(stats.relative_l2 > 0.0);
        assert!(stats.peak_over_rms > 10.0);
        let reconstructed = codec.reconstruct(&input).unwrap();
        let energy = input.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>();
        let error = input
            .iter()
            .zip(reconstructed)
            .map(|(&original, restored)| (f64::from(restored) - f64::from(original)).powi(2))
            .sum::<f64>();
        let expected_error = (error / energy).sqrt();
        assert!((stats.relative_l2 - expected_error).abs() < 1e-12);
        assert!((stats.peak_over_rms - 12.0 / (energy / width as f64).sqrt()).abs() < 1e-12);
        let zeros = activation_stats(&vec![0.0; width], width, codec)
            .unwrap()
            .unwrap();
        assert_eq!((zeros.relative_l2, zeros.peak_over_rms), (0.0, 0.0));
        assert!(activation_stats(&input[..width - 1], width, codec).is_err());
        assert!(activation_stats(&[f32::NAN; 256], width, codec).is_err());
        assert!(activation_stats(&[1.0; 3], 3, codec).unwrap().is_none());
    }
}

#[test]
#[ignore = "requires MIVI_TEST_MODEL; traces cumulative residual error by layer"]
fn layerwise_prefill_error_trace() -> EvalResult<()> {
    let codec = ActivationCodec::from_env()?;
    assert!(trace(codec)? > 0, "no layers traced");
    Ok(())
}

struct Snapshot {
    residual: Vec<f32>,
    normalized: Option<ActivationStats>,
    swiglu: Option<ActivationStats>,
}

fn validate_prompt_text(prompt: &str) -> EvalResult<()> {
    if prompt.trim().is_empty() {
        return Err("trace prompt must contain non-whitespace text before tokenization".into());
    }
    Ok(())
}

#[test]
fn layer_trace_rejects_blank_prompt_before_tokenization() {
    assert!(validate_prompt_text("").is_err());
    assert!(validate_prompt_text(" \n\t").is_err());
    assert!(validate_prompt_text("Inspect src/main.rs").is_ok());
}

fn trace(codec: ActivationCodec) -> EvalResult<usize> {
    let path = std::env::var_os("MIVI_TEST_MODEL").ok_or("set MIVI_TEST_MODEL")?;
    let max_tokens = super::positive_env("MIVI_TEST_CAPTURE_TOKENS", 64)?;
    if max_tokens > 64 {
        return Err("capture token limit must be at most 64".into());
    }
    let mut model = Model::load_with_ctx(std::path::Path::new(&path), Some(max_tokens + 1))?;
    let prompt = match std::env::var("MIVI_TEST_CAPTURE_PROMPT") {
        Ok(prompt) => prompt,
        Err(std::env::VarError::NotPresent) => "Tool available: read_file(path). Inspect src/main.rs before proposing a change. Return a JSON tool call.".to_owned(),
        Err(error) => return Err(error.into()),
    };
    validate_prompt_text(&prompt)?;
    let tokens = super::prompt_tokens(&model, &prompt)?;
    if tokens.is_empty() || tokens.len() > max_tokens {
        return Err(format!(
            "trace prompt needs {} tokens; refuses empty/truncated input (limit {max_tokens})",
            tokens.len()
        )
        .into());
    }
    let batch = tokens.len();
    let layer_metadata: Vec<_> = model
        .weights
        .layers
        .iter()
        .map(|layer| match layer {
            LayerWeights::Attention(_) => ("attention", true),
            LayerWeights::Ssm(_) => ("ssm", model.config.ssm_conv_kernel > 0),
        })
        .collect();
    model.set_prefill_strategy(PrefillStrategy::Chunked {
        tile_tokens: max_tokens,
    })?;
    model.generate_tokens_incremental(&tokens, 0, 0, |_, _| true)?;
    let production = model.state.logits.to_vec();
    let mut snapshots = Vec::new();
    let (_, residual) = super::walk_observed(
        &mut model,
        &tokens,
        1,
        WalkMode::Exact,
        codec,
        |index, tile, before_delta| {
            if index != snapshots.len() {
                return Err("out-of-order baseline layer observation".into());
            }
            let current = tile.final_current_row(batch)?;
            assert_eq!(current, before_delta, "exact path must not inject a delta");
            let residual = current.to_vec();
            let (_, executed_ffn) = layer_metadata[index];
            let dim = tile.dim();
            let hidden = tile.hidden_dim();
            let normalized = if executed_ffn {
                activation_stats(tile.norm_mut(), dim, codec)?
            } else {
                None
            };
            let swiglu = if executed_ffn {
                activation_stats(tile.gate(), hidden, codec)?
            } else {
                None
            };
            snapshots.push(Snapshot {
                residual,
                normalized,
                swiglu,
            });
            Ok(())
        },
    )?;
    assert_eq!(snapshots.len(), layer_metadata.len());
    assert_eq!(
        snapshots.last().ok_or("no model layers")?.residual,
        residual
    );
    let baseline = super::logits(&model, &residual)?;
    let baseline_control = super::cumulative::checked_logit_error(&production, &baseline)?;
    assert!(
        baseline_control.max_abs < 1e-3,
        "traced baseline differs from production"
    );
    let mut control_layers = 0;
    let mut max_control = 0.0f64;
    let (_, control_residual) = super::walk_observed(
        &mut model,
        &tokens,
        1,
        WalkMode::FullFfnControl,
        codec,
        |index, tile, _| {
            if index != control_layers {
                return Err("out-of-order control observation".into());
            }
            let error = super::cumulative::checked_logit_error(
                &snapshots[index].residual,
                tile.final_current_row(batch)?,
            )?;
            assert!(
                error.max_abs < 1e-3,
                "non-packed control differs at layer {index}"
            );
            max_control = max_control.max(error.max_abs);
            control_layers += 1;
            Ok(())
        },
    )?;
    assert_eq!(control_layers, snapshots.len());
    let control_logits = super::logits(&model, &control_residual)?;
    let control = super::cumulative::checked_logit_error(&production, &control_logits)?;
    assert!(
        control.max_abs < 1e-3,
        "traced recompute control differs from production"
    );
    println!("layer trace: codec={}, tokens={batch}, layers={}, baseline/production max abs={:.8}, per-layer recompute max abs={max_control:.8}, recompute/production max abs={:.8}; final-token residuals, activation stats over all token rows", codec.label(), snapshots.len(), baseline_control.max_abs, control.max_abs);
    for (index, snapshot) in snapshots.iter().enumerate() {
        println!(
            "baseline layer={index}, codec={}, kind={}, normalized={}, swiglu={}",
            codec.label(),
            layer_metadata[index].0,
            describe_stats(&snapshot.normalized),
            describe_stats(&snapshot.swiglu)
        );
    }
    for (mode, name) in [
        (WalkMode::DownOnly, "cumulative-down"),
        (WalkMode::GateOnly, "cumulative-gate"),
        (WalkMode::UpOnly, "cumulative-up"),
        (WalkMode::FullFfn, "cumulative-full-ffn"),
    ] {
        let mut seen = 0;
        let mut previous_error = 0.0;
        let mut largest_jump = (0usize, f64::NEG_INFINITY);
        let mut last_observed = Vec::new();
        let (coverage, residual) = super::walk_observed(
            &mut model,
            &tokens,
            1,
            mode,
            codec,
            |index, tile, before_delta| {
                if index != seen {
                    return Err("out-of-order packed observation".into());
                }
                let current = tile.final_current_row(batch)?;
                let propagated = super::cumulative::checked_logit_error(
                    &snapshots[index].residual,
                    before_delta,
                )?;
                let post =
                    super::cumulative::checked_logit_error(&snapshots[index].residual, current)?;
                let injection = super::cumulative::checked_logit_error(before_delta, current)?;
                let jump = post.relative_l2 - previous_error;
                if jump > largest_jump.1 {
                    largest_jump = (index, jump);
                }
                println!("mode={name}, codec={}, layer={index}, kind={}, propagated relative L2={:.6}, post relative L2={:.6}, injection relative L2={:.6}, post max abs={:.6}", codec.label(), layer_metadata[index].0, propagated.relative_l2, post.relative_l2, injection.relative_l2, post.max_abs);
                previous_error = post.relative_l2;
                last_observed = current.to_vec();
                seen += 1;
                Ok(())
            },
        )?;
        assert_eq!(seen, snapshots.len());
        assert_eq!(
            last_observed, residual,
            "observer must run AFTER cumulative injection"
        );
        assert!(coverage.packed > 0);
        let executed_layers = layer_metadata
            .iter()
            .filter(|(_, executed)| *executed)
            .count();
        assert_eq!(
            coverage.packed + coverage.fallback,
            executed_layers * if mode == WalkMode::DownOnly { 1 } else { 3 }
        );
        let changed_logits = super::logits(&model, &residual)?;
        let error = super::cumulative::checked_logit_error(&baseline, &changed_logits)?;
        println!("summary mode={name}, codec={}, largest relative-residual increase layer={} ({:.6}), final logit relative L2={:.6}, max abs={:.6}, greedy token {} -> {}; packed={}, fallback={}", codec.label(), largest_jump.0, largest_jump.1, error.relative_l2, error.max_abs, super::top_token(&baseline)?, super::top_token(&changed_logits)?, coverage.packed, coverage.fallback);
    }
    println!("No intermediate logits, tensor dumps, long-context, timing, or agent-quality result; growth localization is not causal attribution.");
    Ok(snapshots.len())
}

fn describe_stats(stats: &Option<ActivationStats>) -> String {
    match stats {
        Some(stats) => format!(
            "packing relative L2={:.6}, peak/RMS={:.3}",
            stats.relative_l2, stats.peak_over_rms
        ),
        None => "not applicable (skipped FFN or unsupported block width)".to_owned(),
    }
}
