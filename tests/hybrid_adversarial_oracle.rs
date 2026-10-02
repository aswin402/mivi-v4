use mivi_model::{BlockType, Model, PrefillStrategy};
use serde::Deserialize;
use std::path::{Path, PathBuf};

const MAX_SYNTHETIC_GGUF_BYTES: u64 = 4 * 1024 * 1024;

#[derive(Debug, Deserialize)]
struct OracleFixture {
    schema: u32,
    config: FixtureConfig,
    numerical_policy: NumericalPolicy,
    cases: Vec<OracleCase>,
}

#[derive(Debug, Deserialize)]
struct FixtureConfig {
    dim: usize,
    vocab_size: usize,
    context: usize,
    block_types: Vec<String>,
    n_heads: usize,
    n_kv_heads: usize,
    gqa_ratio: usize,
    dtype: String,
}

#[derive(Debug, Deserialize)]
struct NumericalPolicy {
    f32_atol: f64,
    f32_rtol: f64,
}

#[derive(Debug, Deserialize)]
struct OracleCase {
    name: String,
    token_ids: Vec<u32>,
    steps: Vec<OracleStep>,
}

#[derive(Debug, Deserialize)]
struct OracleStep {
    pos: usize,
    token_id: u32,
    logits: Vec<f64>,
    top_token: u32,
    top_margin: f64,
}

fn fixture_path() -> PathBuf {
    let path = std::env::var_os("MIVI_ADVERSARIAL_FIXTURE")
        .map(PathBuf::from)
        .expect("Set MIVI_ADVERSARIAL_FIXTURE to the absolute path of a privately generated synthetic GGUF");
    assert!(
        path.is_absolute(),
        "MIVI_ADVERSARIAL_FIXTURE must be absolute"
    );
    let metadata = std::fs::symlink_metadata(&path).unwrap_or_else(|error| {
        panic!("cannot inspect MIVI_ADVERSARIAL_FIXTURE {path:?}: {error}")
    });
    assert!(
        metadata.file_type().is_file(),
        "MIVI_ADVERSARIAL_FIXTURE must be a regular file, not a symlink or special file: {path:?}"
    );
    assert!(
        metadata.len() <= MAX_SYNTHETIC_GGUF_BYTES,
        "MIVI_ADVERSARIAL_FIXTURE exceeds the 4 MiB synthetic-fixture limit: {path:?}"
    );
    path
}

fn read_oracle() -> OracleFixture {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let trace_path = manifest.join("tests/fixtures/hybrid_adversarial.json");
    let trace: OracleFixture = serde_json::from_slice(
        &std::fs::read(&trace_path).expect("read committed synthetic oracle traces"),
    )
    .expect("parse committed synthetic oracle traces");
    assert_eq!(trace.schema, 1, "unsupported oracle schema");
    assert_eq!(trace.config.dim, 64);
    assert_eq!(trace.config.vocab_size, 64);
    assert_eq!(trace.config.context, 128);
    assert_eq!(
        trace.config.block_types,
        ["ssm", "attention", "ssm", "attention"]
    );
    assert_eq!(
        trace.config.n_heads / trace.config.n_kv_heads,
        trace.config.gqa_ratio
    );
    assert_eq!(trace.config.gqa_ratio, 2);
    assert_eq!(trace.config.dtype, "F32");
    assert_eq!(trace.numerical_policy.f32_atol, 1.0e-4);
    assert_eq!(trace.numerical_policy.f32_rtol, 1.0e-4);
    trace
}

fn argmax_and_margin(logits: &[f32]) -> (u32, f64) {
    assert_eq!(logits.len(), 64, "expected one logit per vocabulary entry");
    assert!(
        logits.iter().all(|value| value.is_finite()),
        "non-finite Rust logit"
    );
    let mut top = (0usize, f32::NEG_INFINITY);
    let mut second = f32::NEG_INFINITY;
    for (index, &value) in logits.iter().enumerate() {
        if value > top.1 {
            second = top.1;
            top = (index, value);
        } else if value > second {
            second = value;
        }
    }
    (top.0 as u32, f64::from(top.1 - second))
}

fn assert_step(actual: &[f32], expected: &OracleStep, atol: f64, rtol: f64, case: &str) {
    assert_eq!(
        actual.len(),
        expected.logits.len(),
        "case={case} pos={}",
        expected.pos
    );
    assert_eq!(
        expected.logits.len(),
        64,
        "fixture must include every vocabulary logit"
    );
    for (index, (&actual, &expected_logit)) in actual.iter().zip(&expected.logits).enumerate() {
        assert!(
            actual.is_finite() && expected_logit.is_finite(),
            "case={case} pos={} index={index} non-finite value",
            expected.pos
        );
        let actual = f64::from(actual);
        let limit = atol + rtol * expected_logit.abs();
        let error = (actual - expected_logit).abs();
        assert!(error <= limit, "case={case} pos={} index={index} actual={actual} expected={expected_logit} error={error} limit={limit}", expected.pos);
    }
    let (top_token, margin) = argmax_and_margin(actual);
    assert_eq!(
        top_token, expected.top_token,
        "case={case} pos={} top token",
        expected.pos
    );
    let margin_limit = atol + rtol * expected.top_margin.abs();
    assert!(
        (margin - expected.top_margin).abs() <= margin_limit,
        "case={case} pos={} top margin actual={margin} expected={} limit={margin_limit}",
        expected.pos,
        expected.top_margin
    );
}

fn clear_model(model: &mut Model) {
    model.reset_context();
    model.prefix_cache.clear();
    assert!(model.state.conv_states.iter().all(|value| *value == 0.0));
    assert!(model.state.ssm_states.iter().all(|value| *value == 0.0));
}

#[test]
fn hybrid_adversarial_oracle_matches_token_chunk_reset_and_continuation_paths() {
    let trace = read_oracle();
    let mut model = Model::load(&fixture_path()).expect("load explicit synthetic hybrid GGUF");
    assert_eq!(model.config.dim, trace.config.dim);
    assert_eq!(model.config.vocab_size, trace.config.vocab_size);
    assert_eq!(model.config.max_seq_len, trace.config.context);
    assert_eq!(model.config.n_heads, trace.config.n_heads);
    assert_eq!(model.config.n_kv_heads, trace.config.n_kv_heads);
    assert_eq!(
        model.config.block_types,
        [
            BlockType::SSM,
            BlockType::Attention,
            BlockType::SSM,
            BlockType::Attention,
        ]
    );

    let atol = trace.numerical_policy.f32_atol;
    let rtol = trace.numerical_policy.f32_rtol;
    let tile_sizes = [1, 2, 3, 8];

    for case in &trace.cases {
        assert_eq!(case.token_ids.len(), case.steps.len(), "case={}", case.name);
        clear_model(&mut model);
        model.set_prefill_strategy(PrefillStrategy::Token).unwrap();
        for step in &case.steps {
            assert_eq!(
                case.token_ids[step.pos], step.token_id,
                "case={} pos={}",
                case.name, step.pos
            );
            let logits = model
                .forward(step.token_id, step.pos)
                .expect("token-major forward")
                .to_vec();
            assert_step(&logits, step, atol, rtol, &case.name);
        }
        assert!(
            model.state.conv_states.iter().any(|value| *value != 0.0),
            "case={} must exercise nonzero convolution carry",
            case.name
        );

        for tile_tokens in tile_sizes {
            model
                .set_prefill_strategy(PrefillStrategy::Chunked { tile_tokens })
                .unwrap();
            for step in &case.steps {
                clear_model(&mut model);
                model
                    .set_prefill_strategy(PrefillStrategy::Chunked { tile_tokens })
                    .unwrap();
                model
                    .generate_tokens_incremental(&case.token_ids[..=step.pos], 0, 0, |_, _| true)
                    .expect("chunked prefix forward");
                assert_step(&model.state.logits, step, atol, rtol, &case.name);
            }
        }
    }

    // Reuse recurrent and KV state over a deliberate call boundary, then
    // compare every suffix point, not only the final output token.
    let continuation = trace
        .cases
        .iter()
        .find(|case| case.name == "teacher_a")
        .unwrap();
    let split = 2;
    clear_model(&mut model);
    model
        .set_prefill_strategy(PrefillStrategy::Chunked { tile_tokens: 3 })
        .unwrap();
    model
        .generate_tokens_incremental(&continuation.token_ids[..split], 0, 0, |_, _| true)
        .expect("split continuation prefix");
    assert_step(
        &model.state.logits,
        &continuation.steps[split - 1],
        atol,
        rtol,
        "split_prefix",
    );
    for step in &continuation.steps[split..] {
        model
            .generate_tokens_incremental(&[step.token_id], step.pos, 0, |_, _| true)
            .expect("split continuation suffix");
        assert_step(&model.state.logits, step, atol, rtol, "split_continuation");
    }
}
