//! Agent-sized, fixed-work controls. No HTTP or real agent quality is measured.

use super::*;
use crate::fixture_diagnostics::{
    CaptureLimits, ModelCapture, ModelOutcome, ModelRecorder, StageOutcome,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CacheCase {
    ColdPrefix,
    WarmPrefix,
}

#[derive(Clone, Copy, Debug)]
struct FirstOutputMeasurement {
    prefill: Duration,
    first_output: Duration,
    reused: usize,
    processed: usize,
}

fn validate_first_output(
    capture: &ModelCapture,
    case: CacheCase,
    physical_wall: Duration,
) -> std::result::Result<FirstOutputMeasurement, &'static str> {
    if capture.prefill_outcome != Some(StageOutcome::Complete)
        || capture.outcome != Some(ModelOutcome::DeliveryStopped)
        || capture.prefill_profile.is_some()
        || capture.progress_counter_overflow
        || capture.generated_ids.observed_tokens != 1
        || capture.generated_ids.ids.len() != 1
        || capture.generated_ids.truncated
        || capture.generated_ids.counter_overflow
        || capture.delivered.text.is_empty()
        || capture.delivered.truncated
        || capture.delivered.counter_overflow
    {
        return Err("incomplete or instrumented first-output observation");
    }
    let prefill = capture.prefill.ok_or("missing prefill time")?;
    let first_output = capture.first_delivered.ok_or("no nonempty delivery")?;
    let total = capture.prompt_tokens.ok_or("missing prompt count")?;
    let reused = capture.reused_tokens.ok_or("missing reused count")?;
    let processed = capture.processed_tokens.ok_or("missing processed count")?;
    if processed == 0
        || reused.checked_add(processed) != Some(total)
        || first_output < prefill
        || first_output > physical_wall
        || (case == CacheCase::ColdPrefix && reused != 0)
        || (case == CacheCase::WarmPrefix && reused == 0)
    {
        return Err("invalid cache state, work count or timer boundaries");
    }
    Ok(FirstOutputMeasurement {
        prefill,
        first_output,
        reused,
        processed,
    })
}

#[test]
fn agent_first_output_measurement_controls() {
    let entry = Instant::now();
    let mut recorder = ModelRecorder::new(CaptureLimits {
        text_bytes: 32,
        token_ids: 1,
    })
    .unwrap();
    recorder.begin(entry);
    recorder.prefill_begin(entry, 1558);
    recorder.prefill_end(
        entry + Duration::from_secs(2),
        0,
        1558,
        StageOutcome::Complete,
    );
    recorder.snapshot.generated_ids.push(7);
    recorder.delivered(entry + Duration::from_millis(2100), "answer");
    recorder.finish(
        entry + Duration::from_millis(2200),
        ModelOutcome::DeliveryStopped,
    );
    let capture = recorder.snapshot;
    let wall = Duration::from_millis(2200);
    let measurement = validate_first_output(&capture, CacheCase::ColdPrefix, wall).unwrap();
    assert_eq!(measurement.prefill, Duration::from_secs(2));
    assert_eq!(measurement.first_output, Duration::from_millis(2100));
    assert_eq!((measurement.reused, measurement.processed), (0, 1558));
    let mut warm = capture.clone();
    warm.reused_tokens = Some(1536);
    warm.processed_tokens = Some(22);
    assert!(validate_first_output(&warm, CacheCase::WarmPrefix, wall).is_ok());
    assert!(validate_first_output(&capture, CacheCase::WarmPrefix, wall).is_err());
    assert!(validate_first_output(&warm, CacheCase::ColdPrefix, wall).is_err());
    for invalid in 0..8 {
        let mut invalid_capture = capture.clone();
        match invalid {
            0 => invalid_capture.first_delivered = None,
            1 => invalid_capture.first_delivered = Some(Duration::from_secs(1)),
            2 => invalid_capture.first_delivered = Some(Duration::from_secs(3)),
            3 => invalid_capture.processed_tokens = Some(1557),
            4 => invalid_capture.prefill_outcome = Some(StageOutcome::Cancelled),
            5 => invalid_capture.generated_ids.push(8),
            6 => invalid_capture.prefill_profile = Some(Default::default()),
            _ => invalid_capture.progress_counter_overflow = true,
        }
        assert!(validate_first_output(&invalid_capture, CacheCase::ColdPrefix, wall).is_err());
    }
}

#[test]
#[ignore = "requires MIVI_TEST_MODEL and explicit release execution"]
fn q4_cached_sums_agent_prompt_parity_and_measurement() {
    assert!(!cfg!(debug_assertions), "run in release mode");
    assert_eq!(rayon::current_num_threads(), 2, "set RAYON_NUM_THREADS=2");
    let path = std::env::var("MIVI_TEST_MODEL").unwrap();
    let mut model = Model::load_with_ctx(Path::new(&path), Some(2048)).unwrap();
    model
        .set_prefill_strategy(PrefillStrategy::Chunked { tile_tokens: 64 })
        .unwrap();
    model.sampler.config.temperature = 0.0;
    model.disable_forward_profile();
    let eligible: usize = model
        .weights
        .layers
        .iter()
        .map(|layer| {
            let ffn = match layer {
                LayerWeights::Attention(w) => &w.ffn,
                LayerWeights::Ssm(w) => &w.ffn,
            };
            [&ffn.w_gate, &ffn.w_up, &ffn.w_down]
                .iter()
                .filter(|w| w.quant_type == mivi_quant::GgmlType::Q4_K)
                .count()
        })
        .sum();
    assert!(eligible > 0, "fixture must contain Q4 FFN tensors");
    let prompt = "<workspace_context> Read the workspace files, inspect the parser and explain how to handle an invalid input without changing unrelated files. Use tools carefully and preserve existing changes. </workspace_context> ".repeat(128);
    let prompt_ids: Vec<_> = model
        .tokenizer
        .encode(&prompt)
        .into_iter()
        .take(1557)
        .collect();
    let continuation = model.tokenizer.encode("Inspect the input, return a useful error, and preserve the original file contents before making changes.");
    assert!(prompt_ids.len() == 1557 && continuation.len() >= 16);
    let continuation = &continuation[..16];
    let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    for case in [CacheCase::ColdPrefix, CacheCase::WarmPrefix] {
        model.prefix_cache.clear();
        if case == CacheCase::WarmPrefix {
            model.state.q4_cached_sums_enabled = false;
            model.reset_context();
            model
                .generate_tokens_incremental(&prompt_ids, 0, 0, |_, _| true)
                .unwrap();
        }
        let mut baseline_samples = Vec::new();
        let mut candidate_samples = Vec::new();
        let mut paired_ratios = Vec::new();
        // One warmup pair, three measured pairs. A cold prefix means engine prefix
        // snapshots are cleared, NOT OS pages or mapped model weights evicted.
        for repetition in 0..4 {
            let mut expected = None;
            let mut expected_work = None;
            let mut pair = [[0.0f64; 3]; 2];
            for cached in [repetition % 2 == 0, repetition % 2 != 0] {
                model.reset_context();
                if case == CacheCase::ColdPrefix {
                    model.prefix_cache.clear();
                }
                model.state.q4_cached_sums_enabled = cached;
                model
                    .start_fixture_capture(CaptureLimits {
                        text_bytes: 512,
                        token_ids: 1,
                    })
                    .unwrap();
                println!("agent_cached_sums_begin: case={case:?} pair={repetition} warmup={} candidate={cached}", repetition == 0);
                let entry = Instant::now();
                model.begin_fixture_capture_observation(entry).unwrap();
                let mut callback_time = None;
                let (_, first_ids) = model
                    .generate_tokens_incremental(&prompt_ids, 0, 1, |_, chunk| {
                        assert!(!chunk.is_empty());
                        callback_time = Some(entry.elapsed());
                        false
                    })
                    .unwrap();
                let physical_wall = entry.elapsed();
                model.finish_fixture_capture_observation(
                    ModelOutcome::DeliveryStopped,
                    Instant::now(),
                );
                let capture = model.take_fixture_capture().unwrap();
                let measurement = validate_first_output(&capture, case, physical_wall).unwrap();
                let callback_time = callback_time.expect("must deliver nonempty text");
                assert!(
                    measurement.first_output <= callback_time && callback_time <= physical_wall
                );
                assert_eq!(first_ids, capture.generated_ids.ids);
                // Require this fixture's first callback to stop BEFORE feeding the
                // generated ID back into the model. Then both continuations start
                // at the same prompt state, not a route-dependent stopping point.
                let start_pos = model.current_pos();
                assert_eq!(Some(start_pos), capture.prompt_tokens);
                assert_eq!(
                    model.state.q4_cached_sums_calls, 0,
                    "chunked prefill must stay on baseline batch kernels"
                );
                let mut logits = vec![bits(&model.state.logits)];
                let work = (start_pos, measurement.reused, measurement.processed);
                if let Some(expected_work) = expected_work {
                    assert_eq!(work, expected_work);
                } else {
                    expected_work = Some(work);
                }
                let decode_start = Instant::now();
                for (offset, &id) in continuation.iter().enumerate() {
                    let values = model.forward(id, start_pos + offset).unwrap();
                    assert!(values.iter().all(|v| v.is_finite()));
                    logits.push(bits(values));
                }
                let decode = decode_start.elapsed();
                assert!(model.forward_profile().is_none());
                assert_eq!(
                    model.state.q4_cached_sums_calls,
                    if cached { eligible * 16 } else { 0 }
                );
                let (keys, values) = model.kv_cache.export_state(start_pos + 16).unwrap();
                let actual = (
                    logits,
                    first_ids,
                    capture.delivered.text,
                    bits(&model.state.conv_states),
                    bits(&model.state.ssm_states),
                    bits(&keys),
                    bits(&values),
                    model.current_pos(),
                );
                if let Some(expected) = &expected {
                    assert!(
                        &actual == expected,
                        "candidate changed first output, logits or KV/SSM state"
                    );
                } else {
                    expected = Some(actual);
                }
                let sample = [
                    measurement.prefill.as_secs_f64(),
                    measurement.first_output.as_secs_f64(),
                    decode.as_secs_f64(),
                ];
                pair[usize::from(cached)] = sample;
                println!("agent_cached_sums: case={case:?} pair={repetition} warmup={} candidate={cached} effective_prefix={start_pos} reused={} processed={} prefill_s={:.6} first_output_s={:.6} decode_forwards=16 decode_s={:.6} calls={}", repetition == 0, measurement.reused, measurement.processed, sample[0], sample[1], sample[2], model.state.q4_cached_sums_calls);
            }
            if repetition > 0 {
                baseline_samples.push(pair[0]);
                candidate_samples.push(pair[1]);
                paired_ratios.push(std::array::from_fn::<_, 3, _>(|i| pair[1][i] / pair[0][i]));
            }
        }
        let medians = |samples: Vec<[f64; 3]>| {
            std::array::from_fn::<_, 3, _>(|i| {
                let mut values: Vec<_> = samples.iter().map(|row| row[i]).collect();
                values.sort_by(f64::total_cmp);
                values[values.len() / 2]
            })
        };
        println!("agent_cached_sums_summary: case={case:?} measured_pairs=3 warmup_pairs=1 threads=2 parity=bit_exact baseline_medians_prefill_first_decode={:?} candidate_medians_prefill_first_decode={:?} paired_ratio_medians_prefill_first_decode={:?}", medians(baseline_samples), medians(candidate_samples), medians(paired_ratios));
    }
}
