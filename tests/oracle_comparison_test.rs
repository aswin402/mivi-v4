//! Oracle validation test comparing Rust inference engine against Python PyTorch oracle.

use mivi_model::Model;
use std::path::Path;

#[derive(serde::Deserialize)]
struct OracleTrace {
    pos: usize,
    token: u32,
    top_token: u32,
    logits_sample: Vec<f32>,
}

#[test]
fn test_rust_forward_matches_oracle() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let model_path = manifest_dir.join("models/mivi-tiny-test.gguf");
    assert!(
        model_path.exists(),
        "Test GGUF model must exist. Run python3 training/export/generate_fixture.py first."
    );

    let mut model = Model::load(&model_path).expect("Failed to load test model in Rust");
    assert_eq!(model.config.dim, 64);
    assert_eq!(model.config.n_layers, 2);
    assert_eq!(model.config.vocab_size, 64);

    let oracle_path = manifest_dir.join("tests/fixtures/oracle_output.json");
    let oracle_data_str =
        std::fs::read_to_string(&oracle_path).expect("Failed to read oracle_output.json");
    let oracle_traces: Vec<OracleTrace> =
        serde_json::from_str(&oracle_data_str).expect("Failed to parse oracle json");

    model.state.reset();
    model.kv_cache.reset();

    for trace in oracle_traces {
        let logits = model
            .forward(trace.token, trace.pos)
            .expect("Rust forward pass failed");

        // Find argmax token in Rust
        let mut rust_top_token = 0;
        let mut max_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > max_val {
                max_val = v;
                rust_top_token = i as u32;
            }
        }

        println!(
            "Pos {}: token={}, Rust top_token={}, Oracle top_token={}, logit_0={:.4}, oracle_0={:.4}",
            trace.pos, trace.token, rust_top_token, trace.top_token, logits[0], trace.logits_sample[0]
        );

        // Assert numerical match within reasonable quantization epsilon
        assert_eq!(
            rust_top_token, trace.top_token,
            "Top token mismatch at pos {}",
            trace.pos
        );

        for (idx, &expected_logit) in trace.logits_sample.iter().enumerate() {
            let diff = (logits[idx] - expected_logit).abs();
            assert!(
                diff < 0.05,
                "Logit mismatch at pos {}, index {}: rust={:.4}, oracle={:.4}, diff={:.4}",
                trace.pos,
                idx,
                logits[idx],
                expected_logit,
                diff
            );
        }
    }

    println!("✅ Rust engine matches Python Oracle ground-truth perfectly!");
}
