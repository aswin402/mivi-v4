#![cfg(feature = "projection-diagnostics")]

use mivi_quant::projection_diagnostics::quantized_matmul_rows_profiled;
use mivi_quant::{quantized_matmul_rows, GgmlType, QuantError, RAYON_PARALLEL_THRESHOLD};

#[test]
fn projection_profile_preserves_f32_bits() {
    let cols = 17;
    let rows = 3;
    let weights: Vec<u8> = (0..rows * cols)
        .flat_map(|i| (((i % 19) as f32 - 9.0) * 0.0625).to_le_bytes())
        .collect();
    for batch in [1, 2, 8, 9, 32, 64, 65] {
        let inputs: Vec<f32> = (0..batch * cols)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.125)
            .collect();
        let mut baseline = vec![f32::NAN; batch * rows];
        let mut profiled = baseline.clone();
        quantized_matmul_rows(
            &mut baseline,
            GgmlType::F32,
            &weights,
            &inputs,
            batch,
            rows,
            cols,
        )
        .unwrap();
        let p = quantized_matmul_rows_profiled(
            &mut profiled,
            GgmlType::F32,
            &weights,
            &inputs,
            batch,
            rows,
            cols,
        )
        .unwrap();
        assert_eq!(p.schema, 1);
        assert!(baseline.iter().chain(&profiled).all(|x| x.is_finite()));
        assert!(baseline
            .iter()
            .zip(&profiled)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
        if batch == 1 {
            assert!(p.delegated_matvec_ns.is_some());
            assert!(p.rows_wall_ns.is_none());
            assert!(p.workers.is_empty());
        } else {
            assert_eq!(p.workers.iter().map(|w| w.rows).sum::<usize>(), rows);
        }
    }
}

fn nonzero_weights(ggml_type: GgmlType, rows: usize, cols: usize) -> Vec<u8> {
    let row_bytes = cols / ggml_type.block_size().unwrap() * ggml_type.type_size().unwrap();
    let mut weights = vec![0; rows * row_bytes];
    match ggml_type {
        GgmlType::F32 => {
            for (i, bytes) in weights.chunks_exact_mut(4).enumerate() {
                bytes.copy_from_slice(&((i % 13) as f32 * 0.125 - 0.75).to_le_bytes());
            }
        }
        GgmlType::F16 | GgmlType::BF16 => {
            for (i, bytes) in weights.chunks_exact_mut(2).enumerate() {
                let value = (i % 13) as f32 * 0.125 - 0.75;
                let encoded = if ggml_type == GgmlType::F16 {
                    half::f16::from_f32(value).to_le_bytes()
                } else {
                    half::bf16::from_f32(value).to_le_bytes()
                };
                bytes.copy_from_slice(&encoded);
            }
        }
        _ => {
            for (i, block) in weights
                .chunks_exact_mut(ggml_type.type_size().unwrap())
                .enumerate()
            {
                for (j, byte) in block.iter_mut().enumerate() {
                    *byte = ((i + 7 * j) % 127) as u8;
                }
                match ggml_type {
                    GgmlType::Q8_0 => {
                        block[..2].copy_from_slice(&half::f16::from_f32(0.125).to_le_bytes());
                    }
                    GgmlType::Q4_K => {
                        block[..2].copy_from_slice(&half::f16::from_f32(0.125).to_le_bytes());
                        block[2..4].copy_from_slice(&half::f16::from_f32(0.0625).to_le_bytes());
                    }
                    GgmlType::Q6_K => {
                        let scale_start = block.len() - 2;
                        block[scale_start..]
                            .copy_from_slice(&half::f16::from_f32(0.125).to_le_bytes());
                    }
                    _ => unreachable!("fixture requested only supported matrix formats"),
                }
            }
        }
    }
    weights
}

#[test]
fn projection_profile_reports_exact_dispatch_branches_and_empty_work() {
    let weights = nonzero_weights(GgmlType::F32, 3, 17);
    let cases = [
        (0, 3, "empty"),
        (1, 3, "matvec"),
        (2, 3, "per_input_dot"),
        (8, 3, "per_input_dot"),
        (9, 3, "across_batch"),
        (32, 3, "across_batch_pair"),
        (3, 0, "empty"),
    ];
    for (batch, rows, branch) in cases {
        let inputs = vec![0.25; batch * 17];
        let mut out = vec![71.0; batch * rows];
        let p = quantized_matmul_rows_profiled(
            &mut out,
            GgmlType::F32,
            &weights,
            &inputs,
            batch,
            rows,
            17,
        )
        .unwrap();
        assert_eq!(p.branch, branch);
        if branch == "empty" {
            assert!(p.buffer_init_ns.is_none());
            assert!(p.input_transpose_ns.is_none());
            assert!(p.rows_wall_ns.is_none());
            assert!(p.output_layout_ns.is_none());
            assert!(p.delegated_matvec_ns.is_none());
            assert!(p.workers.is_empty());
        }
    }
}

#[test]
fn projection_profile_preserves_bits_for_supported_formats_and_parallel_tail() {
    let rows = RAYON_PARALLEL_THRESHOLD + 1;
    for ggml_type in [
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::BF16,
        GgmlType::Q8_0,
        GgmlType::Q4_K,
        GgmlType::Q6_K,
    ] {
        let cols = match ggml_type {
            GgmlType::Q8_0 => 32,
            GgmlType::Q4_K | GgmlType::Q6_K => 512,
            _ => 32,
        };
        let batch = 33;
        let weights = nonzero_weights(ggml_type, rows, cols);
        let inputs: Vec<f32> = (0..batch * cols)
            .map(|i| (i % 23) as f32 * 0.0625 - 0.75)
            .collect();

        for threads in [1, 2] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    let mut baseline = vec![f32::NAN; batch * rows];
                    let mut profiled = baseline.clone();
                    quantized_matmul_rows(
                        &mut baseline,
                        ggml_type,
                        &weights,
                        &inputs,
                        batch,
                        rows,
                        cols,
                    )
                    .unwrap();
                    let profile = quantized_matmul_rows_profiled(
                        &mut profiled,
                        ggml_type,
                        &weights,
                        &inputs,
                        batch,
                        rows,
                        cols,
                    )
                    .unwrap();
                    assert!(baseline.iter().chain(&profiled).all(|x| x.is_finite()));
                    assert!(baseline
                        .iter()
                        .zip(&profiled)
                        .all(|(a, b)| a.to_bits() == b.to_bits()));
                    assert_eq!(profile.workers.iter().map(|w| w.rows).sum::<usize>(), rows);
                    if threads == 1 {
                        assert_eq!(profile.workers.len(), 1);
                    } else {
                        assert!(profile.workers.len() > 1);
                    }
                });
        }
    }
}

#[test]
fn profiled_errors_leave_output_untouched() {
    let sentinel = 123.5;
    let mut out = vec![sentinel; 6];
    let f32_weights = vec![0; 3 * 4 * 4];
    let f32_inputs = vec![0.5; 2 * 4];
    let invalid_cases = [
        (&f32_weights[..], &f32_inputs[..], 2, 3, 4, 5),
        (&f32_weights[..], &f32_inputs[..3], 2, 3, 4, 6),
        (&f32_weights[..47], &f32_inputs[..], 2, 3, 4, 6),
    ];
    for (weights, inputs, batch, rows, cols, out_len) in invalid_cases {
        out.fill(sentinel);
        let result = quantized_matmul_rows_profiled(
            &mut out[..out_len],
            GgmlType::F32,
            weights,
            inputs,
            batch,
            rows,
            cols,
        );
        assert!(result.is_err());
        assert!(out.iter().all(|value| *value == sentinel));
    }

    out.fill(sentinel);
    let unsupported_weights = vec![0; 3 * 18];
    let unsupported_inputs = vec![0.5; 2 * 32];
    let unsupported = quantized_matmul_rows_profiled(
        &mut out,
        GgmlType::Q4_0,
        &unsupported_weights,
        &unsupported_inputs,
        2,
        3,
        32,
    );
    assert!(matches!(unsupported, Err(QuantError::UnsupportedType(_))));
    assert!(out.iter().all(|value| *value == sentinel));

    assert!(
        quantized_matmul_rows_profiled(&mut out, GgmlType::Q4_K, &[], &[], 2, 3, 255,).is_err()
    );
    assert!(out.iter().all(|value| *value == sentinel));

    assert!(
        quantized_matmul_rows_profiled(&mut out, GgmlType::F32, &[], &[], usize::MAX, 2, 0,)
            .is_err()
    );
    assert!(out.iter().all(|value| *value == sentinel));
}
