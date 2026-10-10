//! Explicit local-fixture correctness gates and bounded live measurements.
use super::*;
use crate::fixture_diagnostics::{CaptureLimits, ModelOutcome, StageOutcome};

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[derive(Debug, PartialEq, Eq)]
struct Snapshot {
    logits: Vec<u32>,
    conv: Vec<u32>,
    ssm: Vec<u32>,
    keys: Vec<u32>,
    values: Vec<u32>,
    pos: usize,
}

fn snapshot(model: &Model) -> Snapshot {
    let (keys, values) = model.kv_cache.export_state(model.current_pos()).unwrap();
    assert!(model.state.logits.iter().all(|v| v.is_finite()));
    Snapshot {
        logits: bits(&model.state.logits),
        conv: bits(&model.state.conv_states),
        ssm: bits(&model.state.ssm_states),
        keys: bits(&keys),
        values: bits(&values),
        pos: model.current_pos(),
    }
}

fn load(context: usize) -> Model {
    let path = std::env::var("MIVI_TEST_MODEL").expect("set an absolute local fixture/model path");
    assert!(Path::new(&path).is_absolute());
    let mut model = Model::load_with_options(
        Path::new(&path),
        Some(context),
        Some(mivi_kv::KvPrecision::F32),
    )
    .unwrap();
    assert!(!model.state.parallel_attention_enabled);
    model
        .set_prefill_strategy(PrefillStrategy::Chunked { tile_tokens: 64 })
        .unwrap();
    model.sampler.config.temperature = 0.0;
    model.sampler.config.repetition_penalty = 1.0;
    model.sampler.set_seed(7);
    model
}

#[test]
#[ignore = "requires explicit MIVI_TEST_MODEL local hybrid fixture"]
fn parallel_attention_model_state_parity() {
    assert_eq!(rayon::current_num_threads(), 2);
    let mut model = load(512);
    assert!(model.config.max_seq_len >= 320);
    assert!(model.config.n_heads > 1);
    assert!(model.config.block_types.contains(&BlockType::Attention));
    assert!(model.config.block_types.contains(&BlockType::SSM));
    let prompt: Vec<_> = (0..273)
        .map(|i| ((i * 7 + 3) % model.config.vocab_size) as u32)
        .collect();
    for strategy in [
        PrefillStrategy::Token,
        PrefillStrategy::Chunked { tile_tokens: 64 },
    ] {
        model.set_prefill_strategy(strategy).unwrap();
        for warm in [false, true] {
            let mut expected = None;
            for enabled in [false, true] {
                model.prefix_cache.clear();
                model.reset_context();
                model.set_parallel_attention_experiment(enabled);
                if warm {
                    model
                        .generate_tokens_incremental(&prompt, 0, 0, |_, _| true)
                        .unwrap();
                    model.reset_context();
                }
                let (text, ids) = model
                    .generate_tokens_incremental(&prompt, 0, 2, |_, _| true)
                    .unwrap();
                let mut states = vec![snapshot(&model)];
                for id in [5, 7, 9] {
                    model
                        .forward(id % model.config.vocab_size as u32, model.current_pos())
                        .unwrap();
                    states.push(snapshot(&model));
                }
                assert_eq!(model.state.parallel_attention_calls > 0, enabled);
                let actual = (text, ids, states);
                if let Some(expected) = &expected {
                    assert!(
                        &actual == expected,
                        "hybrid state parity failed: strategy={strategy:?} warm={warm}"
                    );
                } else {
                    expected = Some(actual);
                }
            }
        }
    }
}

fn bounded_env(name: &str, allowed: &[usize]) -> usize {
    let value = std::env::var(name).unwrap().parse().unwrap();
    assert!(allowed.contains(&value), "invalid {name}");
    value
}

#[test]
#[ignore = "explicit bounded release pair; local real model required"]
fn parallel_attention_live_pair() {
    assert!(!cfg!(debug_assertions));
    assert_eq!(rayon::current_num_threads(), 2);
    let tokens = bounded_env("MIVI_ATTN_TOKENS", &[512, 2048]);
    let pair = bounded_env("MIVI_ATTN_PAIR", &[0, 1, 2]);
    let profile = bounded_env("MIVI_ATTN_PROFILE", &[0, 1]) == 1;
    let warm = bounded_env("MIVI_ATTN_WARM", &[0, 1]) == 1;
    let child_started = Instant::now();
    let mut model = load(4096);
    let text = "<workspace_context> Read workspace files and explain a parser error. Preserve unrelated changes and use tools carefully. </workspace_context> ".repeat(256);
    let prompt: Vec<_> = model
        .tokenizer
        .encode(&text)
        .into_iter()
        .take(tokens)
        .collect();
    assert_eq!(prompt.len(), tokens);
    let mut expected = None;
    for enabled in [pair % 2 != 0, pair % 2 == 0] {
        model.reset_context();
        model.prefix_cache.clear();
        model.set_parallel_attention_experiment(enabled);
        if warm {
            model
                .generate_tokens_incremental_with_cancel(
                    &prompt,
                    0,
                    0,
                    |_, _| true,
                    || child_started.elapsed() >= Duration::from_secs(160),
                )
                .unwrap();
            model.reset_context();
        }
        if profile {
            model.enable_forward_profile();
            model.reset_forward_profile();
        } else {
            model.disable_forward_profile();
        }
        model
            .start_fixture_capture(CaptureLimits {
                text_bytes: 4096,
                token_ids: 1,
            })
            .unwrap();
        let started = Instant::now();
        model.begin_fixture_capture_observation(started).unwrap();
        let mut first_text = None;
        let output = model
            .generate_tokens_incremental_with_cancel(
                &prompt,
                0,
                1,
                |_, chunk| {
                    if !chunk.is_empty() && first_text.is_none() {
                        first_text = Some(started.elapsed());
                    }
                    false
                },
                || child_started.elapsed() >= Duration::from_secs(160),
            )
            .unwrap();
        let wall = started.elapsed();
        model.finish_fixture_capture_observation(ModelOutcome::DeliveryStopped, Instant::now());
        let capture = model.take_fixture_capture().unwrap();
        assert_eq!(capture.prefill_outcome, Some(StageOutcome::Complete));
        assert_eq!(output.1.len(), 1);
        assert!(!output.0.is_empty());
        assert!(first_text.is_some(), "missing first text is failure");
        assert!(!capture.delivered.truncated && !capture.generated_ids.truncated);
        assert_eq!(
            capture.reused_tokens.unwrap() + capture.processed_tokens.unwrap(),
            capture.prompt_tokens.unwrap()
        );
        assert_eq!(capture.reused_tokens.unwrap() > 0, warm);
        assert_eq!(model.state.parallel_attention_calls > 0, enabled);
        let calls = model.state.parallel_attention_calls;
        let mut states = vec![snapshot(&model)];
        for &id in &prompt[..2] {
            model.forward(id, model.current_pos()).unwrap();
            states.push(snapshot(&model));
        }
        let actual = (output, states);
        if let Some(expected) = &expected {
            // Do not dump complete model-state vectors into logs on failure.
            assert!(
                &actual == expected,
                "full output/logit/KV/SSM parity failed"
            );
        } else {
            expected = Some(actual);
        }
        println!(
            "parallel_attention_live {}",
            serde_json::json!({
                "tokens_requested": tokens, "pair": pair, "profile": profile, "warm": warm,
                "candidate": enabled, "prefill_s": capture.prefill.unwrap().as_secs_f64(),
                "first_text_s": first_text.unwrap().as_secs_f64(), "wall_s": wall.as_secs_f64(),
                "prompt_tokens": capture.prompt_tokens, "computed_tokens": capture.processed_tokens,
                "reused_tokens": capture.reused_tokens, "candidate_calls": calls,
                "attention_region_s": capture.prefill_profile.map(|p| p.attention_stages.causal_attention.as_secs_f64())
            })
        );
    }
}
