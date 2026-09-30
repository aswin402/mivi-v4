//! Complete-row FFN recomputation for cumulative sensitivity diagnostics.

use super::{
    Coverage, EvalResult, GgmlType, Model, PrefillStrategy, ProjectionError, TileActivations,
    WalkMode,
};

struct Projection<'a> {
    kind: GgmlType,
    weights: &'a [u8],
    rows: usize,
    cols: usize,
}

fn project(
    out: &mut [f32],
    projection: &Projection<'_>,
    input: &[f32],
    batch: usize,
    packed: bool,
) -> EvalResult<bool> {
    if batch == 0
        || projection.rows == 0
        || projection.cols == 0
        || input.iter().any(|x| !x.is_finite())
    {
        return Err("invalid projection dimensions or non-finite activations".into());
    }
    let eligible = packed
        && super::supported_shape(projection.kind as u32, &[projection.cols, projection.rows]);
    if eligible {
        super::packed_matmul_with_kernel(
            out,
            projection.weights,
            input,
            batch,
            projection.rows,
            projection.cols,
            super::PackedKernel::detected_tiled(),
        )?;
    } else {
        super::quantized_matmul_rows(
            out,
            projection.kind,
            projection.weights,
            input,
            batch,
            projection.rows,
            projection.cols,
        )?;
    }
    if out.iter().any(|x| !x.is_finite()) {
        return Err("non-finite recomputed projection".into());
    }
    Ok(eligible)
}

impl<'a> Projection<'a> {
    fn from_tensor(tensor: &super::QuantizedTensor, mmap: &'a [u8]) -> EvalResult<Self> {
        Ok(Self {
            kind: GgmlType::from_u32(tensor.quant_type as u32)?,
            weights: tensor
                .as_slice_checked(mmap)
                .ok_or("projection outside GGUF")?,
            rows: tensor.rows,
            cols: tensor.cols,
        })
    }
}

pub(super) struct Recomputed {
    pub(super) output: Vec<f32>,
    pub(super) packed: usize,
    pub(super) fallback: usize,
}

pub(super) fn recompute(
    tile: &mut TileActivations,
    ffn: &mivi_model::weights::FfnLayerWeights,
    mmap: &[u8],
    batch: usize,
    full: bool,
    packed: bool,
) -> EvalResult<Recomputed> {
    let projections = FfnProjections {
        gate: Projection::from_tensor(&ffn.w_gate, mmap)?,
        up: Projection::from_tensor(&ffn.w_up, mmap)?,
        down: Projection::from_tensor(&ffn.w_down, mmap)?,
    };
    let normalized = tile.norm_mut().to_vec();
    projections.recompute(&normalized, tile.gate(), batch, full, packed)
}

struct FfnProjections<'a> {
    gate: Projection<'a>,
    up: Projection<'a>,
    down: Projection<'a>,
}

impl FfnProjections<'_> {
    fn recompute(
        &self,
        normalized: &[f32],
        original_gate: &[f32],
        batch: usize,
        full: bool,
        packed: bool,
    ) -> EvalResult<Recomputed> {
        if self.gate.cols != self.up.cols
            || self.gate.rows != self.up.rows
            || self.down.cols != self.gate.rows
            || self.down.rows != self.gate.cols
        {
            return Err("incompatible FFN gate/up/down shapes".into());
        }
        let hidden_len = batch
            .checked_mul(self.gate.rows)
            .ok_or("FFN hidden size overflow")?;
        let mut coverage = Coverage::default();
        let mut count = |used_packed: bool| {
            if used_packed {
                coverage.packed += 1;
            } else {
                coverage.fallback += 1;
            }
        };
        let gate = if full {
            let mut gate = vec![0.0; hidden_len];
            let mut up = vec![0.0; hidden_len];
            count(project(&mut gate, &self.gate, normalized, batch, packed)?);
            count(project(&mut up, &self.up, normalized, batch, packed)?);
            mivi_model::swiglu_rows(&mut gate, &up, batch, self.gate.rows)?;
            gate
        } else {
            if original_gate.len() != hidden_len {
                return Err("SwiGLU input length mismatch".into());
            }
            original_gate.to_vec()
        };
        let mut output = vec![
            0.0;
            batch
                .checked_mul(self.down.rows)
                .ok_or("FFN output size overflow")?
        ];
        count(project(&mut output, &self.down, &gate, batch, packed)?);
        Ok(Recomputed {
            output,
            packed: coverage.packed,
            fallback: coverage.fallback,
        })
    }
}

#[test]
fn cumulative_projection_f32_fallback_preserves_nonzero_outputs() {
    let values = [1.0f32, 2.0, -3.0, 4.0];
    let weights: Vec<_> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    let projection = Projection {
        kind: GgmlType::F32,
        weights: &weights,
        rows: 2,
        cols: 2,
    };
    let mut out = [0.0; 4];
    assert!(!project(&mut out, &projection, &[2.0, 1.0, -1.0, 3.0], 2, true).unwrap());
    assert_eq!(out, [4.0, -2.0, 5.0, 15.0]);
}

#[test]
fn cumulative_projection_uses_q4_and_rejects_invalid_inputs() {
    let weights = crate::q4_k_m::packed_prefill::fixture_weights(2, 256, 7);
    let projection = Projection {
        kind: GgmlType::Q4_K,
        weights: &weights,
        rows: 2,
        cols: 256,
    };
    let input = vec![0.5; 256];
    let mut out = [0.0; 2];
    assert!(project(&mut out, &projection, &input, 1, true).unwrap());
    let mut expected = [0.0; 2];
    super::packed_matmul_with_kernel(
        &mut expected,
        &weights,
        &input,
        1,
        2,
        256,
        super::PackedKernel::detected_tiled(),
    )
    .unwrap();
    assert_eq!(out, expected);
    assert!(project(&mut out, &projection, &input[..255], 1, true).is_err());
    assert!(project(&mut out, &projection, &vec![f32::NAN; 256], 1, true).is_err());
}

#[test]
#[ignore = "requires MIVI_TEST_MODEL; cumulative multi-prompt sensitivity"]
fn cumulative_prefill_evaluation() -> EvalResult<()> {
    assert!(
        evaluate_corpus()? >= 6,
        "both modes must evaluate at least three prompts"
    );
    Ok(())
}

fn checked_logit_error(expected: &[f32], actual: &[f32]) -> EvalResult<ProjectionError> {
    let error = ProjectionError::compare(expected, actual)?;
    if !error.relative_l2.is_finite() || !error.max_abs.is_finite() {
        return Err("non-finite derived logit error metric".into());
    }
    Ok(error)
}

fn validate_controls(
    production: &[f32],
    baseline: &[f32],
    recomputed: &[f32],
) -> EvalResult<(ProjectionError, ProjectionError)> {
    let baseline_error = checked_logit_error(production, baseline)?;
    let recomputed_error = checked_logit_error(production, recomputed)?;
    if baseline_error.max_abs >= 1e-3 || recomputed_error.max_abs >= 1e-3 {
        return Err("control logits disagree with production".into());
    }
    Ok((baseline_error, recomputed_error))
}

#[test]
fn cumulative_metrics_reject_nonfinite_derived_error() {
    assert!(checked_logit_error(&[0.0], &[3.0]).is_err());
    assert!(checked_logit_error(&[1.0], &[f32::NAN]).is_err());
}

#[test]
fn cumulative_controls_reject_combined_drift_from_production() {
    assert!(validate_controls(&[1.0], &[1.0008], &[1.0016]).is_err());
    assert!(validate_controls(&[1.0], &[1.0], &[1.0]).is_ok());
}

#[test]
fn cumulative_mixed_ffn_recomputes_swiglu_before_fallback_down() {
    let width = super::Q4_K_BLOCK_SIZE;
    let gate_weights = crate::q4_k_m::packed_prefill::fixture_weights(width, width, 5);
    let up_weights = crate::q4_k_m::packed_prefill::fixture_weights(width, width, 9);
    let down_weights: Vec<_> = (0..width * width)
        .flat_map(|i| {
            let value = if i / width == i % width { 1.0f32 } else { 0.0 };
            value.to_le_bytes()
        })
        .collect();
    let projections = FfnProjections {
        gate: Projection {
            kind: GgmlType::Q4_K,
            weights: &gate_weights,
            rows: width,
            cols: width,
        },
        up: Projection {
            kind: GgmlType::Q4_K,
            weights: &up_weights,
            rows: width,
            cols: width,
        },
        down: Projection {
            kind: GgmlType::F32,
            weights: &down_weights,
            rows: width,
            cols: width,
        },
    };
    let batch = 2;
    let input: Vec<_> = (0..batch * width)
        .map(|i| ((i % 31) as f32 - 15.0) / 16.0)
        .collect();
    for packed in [false, true] {
        let mut expected = vec![0.0; input.len()];
        let mut up = expected.clone();
        project(&mut expected, &projections.gate, &input, batch, packed).unwrap();
        project(&mut up, &projections.up, &input, batch, packed).unwrap();
        mivi_model::swiglu_rows(&mut expected, &up, batch, width).unwrap();
        let recomputed = projections
            .recompute(&input, &vec![0.0; input.len()], batch, true, packed)
            .unwrap();
        assert_eq!(
            recomputed.output, expected,
            "down identity must receive NEW SwiGLU, not retained original gate"
        );
        assert_eq!(
            (recomputed.packed, recomputed.fallback),
            if packed { (2, 1) } else { (0, 3) }
        );
        assert!(expected.iter().any(|x| *x != 0.0));
    }
    let original_gate = vec![0.25; input.len()];
    let down_only = projections
        .recompute(&input, &original_gate, batch, false, true)
        .unwrap();
    assert_eq!(down_only.output, original_gate);
    assert_eq!((down_only.packed, down_only.fallback), (0, 1));
    assert!(projections
        .recompute(&input, &[], batch, false, true)
        .is_err());
    assert!(projections
        .recompute(&input[..1], &original_gate, batch, true, true)
        .is_err());
}

fn evaluate_corpus() -> EvalResult<usize> {
    let path = std::env::var_os("MIVI_TEST_MODEL").ok_or("set MIVI_TEST_MODEL")?;
    let max_tokens = super::positive_env("MIVI_TEST_CAPTURE_TOKENS", 64)?;
    if max_tokens > 64 {
        return Err("capture token limit must be at most 64".into());
    }
    let mut model = Model::load_with_ctx(std::path::Path::new(&path), Some(max_tokens + 1))?;
    model.set_prefill_strategy(PrefillStrategy::Chunked {
        tile_tokens: max_tokens,
    })?;
    let fixtures = [
        ("coding", "Write Rust code that reads a UTF-8 file and returns an error instead of panicking."),
        ("tool-request", "Tool available: read_file(path). Inspect src/main.rs before proposing a change. Return a JSON tool call."),
        ("tool-result", "Tool result from read_file: fn main() { println!(\"hello\"); } Now add a unit test and explain the change."),
    ];
    let mut cases = 0;
    let executed_layers = model
        .weights
        .layers
        .iter()
        .filter(|layer| {
            matches!(layer, super::LayerWeights::Attention(_)) || model.config.ssm_conv_kernel > 0
        })
        .count();
    for (label, prompt) in fixtures {
        let tokens = super::prompt_tokens(&model, prompt)?;
        if tokens.is_empty() || tokens.len() > max_tokens {
            return Err(format!("fixture {label} needs {} tokens; refuses empty/truncated fixture (limit {max_tokens})", tokens.len()).into());
        }
        model.reset_context();
        model.prefix_cache.clear();
        model.generate_tokens_incremental(&tokens, 0, 0, |_, _| true)?;
        let production = model.state.logits.to_vec();
        let (_, residual) = super::walk(&mut model, &tokens, 1, WalkMode::Exact)?;
        let baseline = super::logits(&model, &residual)?;
        let (control_coverage, residual) =
            super::walk(&mut model, &tokens, 1, WalkMode::FullFfnControl)?;
        assert_eq!(control_coverage.packed, 0);
        assert_eq!(control_coverage.fallback, executed_layers * 3);
        let recomputed = super::logits(&model, &residual)?;
        let (control, recompute_control) = validate_controls(&production, &baseline, &recomputed)?;
        let walker_recompute = checked_logit_error(&baseline, &recomputed)?;
        println!("fixture={label}, tokens={}, baseline/production max abs={:.8}, recompute/production max abs={:.8}, recompute/walker max abs={:.8}", tokens.len(), control.max_abs, recompute_control.max_abs, walker_recompute.max_abs);
        for (mode, name) in [
            (WalkMode::DownOnly, "cumulative-down"),
            (WalkMode::FullFfn, "cumulative-full-ffn"),
        ] {
            let (coverage, residual) = super::walk(&mut model, &tokens, 1, mode)?;
            assert!(
                coverage.packed > 0,
                "no supported packed projections for {name}"
            );
            let projections_per_layer = if mode == WalkMode::DownOnly { 1 } else { 3 };
            assert_eq!(
                coverage.packed + coverage.fallback,
                executed_layers * projections_per_layer,
                "incomplete cumulative coverage for {name}"
            );
            let changed = super::logits(&model, &residual)?;
            let error = checked_logit_error(&baseline, &changed)?;
            println!("  mode={name}, packed={}, fallback={}, logit relative L2={:.6}, max abs={:.6}, greedy token {} -> {}", coverage.packed, coverage.fallback, error.relative_l2, error.max_abs, super::top_token(&baseline)?, super::top_token(&changed)?);
            cases += 1;
        }
    }
    println!("{cases} cumulative cases; complete matrices, raw fixtures, residual-delta rounding; no generation/tool-quality or timing claim");
    Ok(cases)
}
