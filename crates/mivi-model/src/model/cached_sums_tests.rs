//! Explicit, bounded real-model fixed-work experiment. Never in ordinary tests.

use super::*;

#[test]
#[ignore = "requires MIVI_TEST_MODEL and explicit release execution"]
fn q4_cached_sums_decode_parity_and_measurement() {
    assert!(!cfg!(debug_assertions), "run in release mode");
    assert_eq!(rayon::current_num_threads(), 2, "set RAYON_NUM_THREADS=2");
    let model_path = std::env::var("MIVI_TEST_MODEL").unwrap();
    let mut model = Model::load_with_ctx(std::path::Path::new(&model_path), Some(1024)).unwrap();
    model
        .set_prefill_strategy(PrefillStrategy::Chunked { tile_tokens: 64 })
        .unwrap();
    let mut tensor_counts = std::collections::BTreeMap::new();
    let mut eligible = 0;
    for layer in &model.weights.layers {
        let ffn = match layer {
            LayerWeights::Attention(w) => &w.ffn,
            LayerWeights::Ssm(w) => &w.ffn,
        };
        for weight in [&ffn.w_gate, &ffn.w_up, &ffn.w_down] {
            *tensor_counts
                .entry((weight.quant_type as u32, weight.rows, weight.cols))
                .or_insert(0usize) += 1;
            eligible += usize::from(weight.quant_type == mivi_quant::GgmlType::Q4_K);
        }
    }
    assert!(eligible > 0, "fixture must contain eligible Q4 FFN tensors");
    for ((format, rows, cols), count) in tensor_counts {
        println!(
            "ffn format={:?} rows={} cols={} count={}",
            mivi_quant::GgmlType::from_u32(format).unwrap(),
            rows,
            cols,
            count
        );
    }
    let prompt = "Read the workspace files and explain how to handle a parsing error. ".repeat(32);
    let prompt_ids: Vec<_> = model
        .tokenizer
        .encode(&prompt)
        .into_iter()
        .take(256)
        .collect();
    let continuation = model.tokenizer.encode("Inspect the input, return a useful error, and preserve the original file contents before making changes.");
    assert!(prompt_ids.len() == 256 && continuation.len() >= 16);
    let continuation = &continuation[..16];
    let bits = |values: &[f32]| values.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let mut baseline_walls = Vec::new();
    let mut candidate_walls = Vec::new();
    let mut ratios = Vec::new();
    model.disable_forward_profile();
    model.prefix_cache.clear();
    // A warmup pair then three alternating measured pairs. Prefix preparation is
    // identical baseline work outside the timer; cache is retained between members.
    for repetition in 0..4 {
        let mut expected = None;
        let mut pair_walls = [0.0f64; 2];
        for cached in [repetition % 2 == 0, repetition % 2 != 0] {
            model.state.q4_cached_sums_enabled = false;
            model.reset_context();
            model
                .generate_tokens_incremental(&prompt_ids, 0, 0, |_, _| true)
                .unwrap();
            assert_eq!(model.state.q4_cached_sums_calls, 0);
            let start_pos = model.current_pos();
            model.state.q4_cached_sums_enabled = cached;
            let started = Instant::now();
            let mut logits = Vec::new();
            for (offset, &id) in continuation.iter().enumerate() {
                let values = model.forward(id, start_pos + offset).unwrap();
                assert!(values.iter().all(|x| x.is_finite()));
                logits.push(bits(values));
            }
            let wall = started.elapsed().as_secs_f64();
            assert!(model.forward_profile().is_none());
            assert_eq!(
                model.state.q4_cached_sums_calls,
                if cached { eligible * 16 } else { 0 }
            );
            let (keys, values) = model.kv_cache.export_state(start_pos + 16).unwrap();
            let actual = (
                logits,
                bits(&model.state.conv_states),
                bits(&model.state.ssm_states),
                bits(&keys),
                bits(&values),
                model.current_pos(),
            );
            if let Some(expected) = &expected {
                assert_eq!(
                    &actual, expected,
                    "candidate changed logits or recurrent/KV state"
                );
            } else {
                expected = Some(actual);
            }
            pair_walls[usize::from(cached)] = wall;
            if repetition > 0 {
                println!("cached_sums pair={} candidate={} effective_prefix={} forwards=16 wall_s={:.6} calls={}", repetition, cached, start_pos, wall, model.state.q4_cached_sums_calls);
            }
        }
        if repetition > 0 {
            baseline_walls.push(pair_walls[0]);
            candidate_walls.push(pair_walls[1]);
            ratios.push(pair_walls[1] / pair_walls[0]);
            println!(
                "cached_sums pair={} candidate_over_baseline={:.6}",
                repetition,
                ratios.last().unwrap()
            );
        }
    }
    let median = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    println!("cached_sums_summary: prefix=256 continuation=16 pairs=3 warmup_pairs=1 threads=2 parity=bit_exact eligible_ffn_tensors={} scratch_bytes={} baseline_median_s={:.6} candidate_median_s={:.6} paired_ratio_median={:.6}", eligible, model.state.q4_activation_sums.len() * std::mem::size_of::<f32>(), median(baseline_walls), median(candidate_walls), median(ratios));
}
