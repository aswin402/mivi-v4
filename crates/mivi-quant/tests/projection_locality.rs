#![cfg(feature = "projection-locality-experiment")]

use half::{bf16, f16};
use mivi_core::simd::ProjectionTokenTile::{Tokens128, Tokens32, Tokens64};
use mivi_quant::projection_diagnostics::quantized_matmul_rows_profiled_with_token_tile;
use mivi_quant::{
    quantized_matmul_rows, quantized_matmul_rows_with_token_tile, GgmlType, QuantError,
};

const TYPES: [GgmlType; 6] = [
    GgmlType::F32,
    GgmlType::F16,
    GgmlType::BF16,
    GgmlType::Q8_0,
    GgmlType::Q4_K,
    GgmlType::Q6_K,
];
const TILES: [mivi_core::simd::ProjectionTokenTile; 3] = [Tokens32, Tokens64, Tokens128];

fn encoded_weights(ggml_type: GgmlType, rows: usize, cols: usize) -> Vec<u8> {
    let row_bytes = cols / ggml_type.block_size().unwrap() * ggml_type.type_size().unwrap();
    let mut weights = vec![0; rows * row_bytes];
    match ggml_type {
        GgmlType::F32 => {
            for (i, chunk) in weights.chunks_exact_mut(4).enumerate() {
                chunk.copy_from_slice(&(((i % 17) as f32 - 8.0) * 0.0625).to_le_bytes());
            }
        }
        GgmlType::F16 | GgmlType::BF16 => {
            for (i, chunk) in weights.chunks_exact_mut(2).enumerate() {
                let value = ((i % 17) as f32 - 8.0) * 0.0625;
                let bytes = if ggml_type == GgmlType::F16 {
                    f16::from_f32(value).to_le_bytes()
                } else {
                    bf16::from_f32(value).to_le_bytes()
                };
                chunk.copy_from_slice(&bytes);
            }
        }
        _ => {
            let block_bytes = ggml_type.type_size().unwrap();
            for (block_index, block) in weights.chunks_exact_mut(block_bytes).enumerate() {
                for (i, byte) in block.iter_mut().enumerate() {
                    *byte = (block_index * 19 + i * 7 + 1) as u8;
                }
                match ggml_type {
                    GgmlType::Q8_0 => {
                        block[..2].copy_from_slice(&f16::from_f32(0.125).to_le_bytes());
                    }
                    GgmlType::Q4_K => {
                        block[..2].copy_from_slice(&f16::from_f32(0.125).to_le_bytes());
                        block[2..4].copy_from_slice(&f16::from_f32(0.0625).to_le_bytes());
                    }
                    GgmlType::Q6_K => {
                        let start = block.len() - 2;
                        block[start..].copy_from_slice(&f16::from_f32(0.125).to_le_bytes());
                    }
                    _ => unreachable!(),
                }
            }
        }
    }
    weights
}

fn assert_bits_equal(left: &[f32], right: &[f32], context: &str) {
    assert_eq!(left.len(), right.len(), "{context}: output length");
    for (index, (a, b)) in left.iter().zip(right).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "{context}: output[{index}]");
    }
}

#[test]
fn explicit_token_tiles_match_baseline_bits_for_supported_formats_and_batches() {
    let rows = 5;
    for ggml_type in TYPES {
        let block = ggml_type.block_size().unwrap();
        for cols in [block, block * 3] {
            let weights = encoded_weights(ggml_type, rows, cols);
            for batch in [1, 2, 8, 9, 31, 32, 33, 64, 65] {
                let inputs: Vec<f32> = (0..batch * cols)
                    .map(|i| ((i % 29) as f32 - 14.0) * 0.03125)
                    .collect();
                let mut baseline = vec![f32::NAN; batch * rows];
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
                for tile in TILES {
                    let mut actual = vec![f32::NAN; batch * rows];
                    quantized_matmul_rows_with_token_tile(
                        &mut actual,
                        ggml_type,
                        &weights,
                        &inputs,
                        batch,
                        rows,
                        cols,
                        tile,
                    )
                    .unwrap();
                    assert_bits_equal(
                        &baseline,
                        &actual,
                        &format!("{ggml_type:?}, batch={batch}, cols={cols}, tile={tile:?}"),
                    );
                }
            }
        }
    }
}

#[test]
fn explicit_token_tiles_preserve_checked_validation() {
    let tile = Tokens64;
    let weights = encoded_weights(GgmlType::F32, 2, 4);
    let inputs = vec![0.25; 32 * 4];
    let mut output = vec![0.0; 32 * 2];
    let call = |out: &mut [f32], weight: &[u8], input: &[f32], batch, rows, cols, kind| {
        quantized_matmul_rows_with_token_tile(out, kind, weight, input, batch, rows, cols, tile)
    };

    assert!(matches!(
        call(
            &mut output[..63],
            &weights,
            &inputs,
            32,
            2,
            1,
            GgmlType::F32
        ),
        Err(QuantError::BufferTooSmall { .. })
    ));
    assert!(matches!(
        call(
            &mut output,
            &weights,
            &inputs[..127],
            32,
            2,
            4,
            GgmlType::F32
        ),
        Err(QuantError::BufferTooSmall { .. })
    ));
    assert!(matches!(
        call(
            &mut output,
            &weights[..31],
            &inputs,
            32,
            2,
            4,
            GgmlType::F32
        ),
        Err(QuantError::BufferTooSmall { .. })
    ));
    assert!(matches!(
        call(&mut output, &weights, &inputs, 32, 2, 33, GgmlType::Q8_0),
        Err(QuantError::DimensionMisaligned {
            dim: 33,
            block_size: 32
        })
    ));
    assert!(matches!(
        call(&mut [], &[], &[], usize::MAX, 2, 4, GgmlType::F32),
        Err(QuantError::ArithmeticOverflow)
    ));
    assert!(matches!(
        call(&mut [], &[], &[], 1, 1, usize::MAX, GgmlType::F32),
        Err(QuantError::ArithmeticOverflow)
    ));
}

#[test]
fn profiled_token_tiles_match_baseline_and_report_across_batch_work() {
    let rows = 3;
    let cols = 16;
    let batch = 33;
    let weights = encoded_weights(GgmlType::F32, rows, cols);
    let inputs: Vec<f32> = (0..batch * cols)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.0625)
        .collect();
    let mut baseline = vec![f32::NAN; batch * rows];
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

    for tile in TILES {
        let mut profiled = vec![f32::NAN; batch * rows];
        let profile = quantized_matmul_rows_profiled_with_token_tile(
            &mut profiled,
            GgmlType::F32,
            &weights,
            &inputs,
            batch,
            rows,
            cols,
            tile,
        )
        .unwrap();

        assert_bits_equal(&baseline, &profiled, &format!("profiled tile={tile:?}"));
        assert_eq!(profile.branch, "across_batch_pair");
        assert!(profile.input_transpose_ns.is_some());
        assert!(profile.rows_wall_ns.is_some());
        assert!(!profile.workers.is_empty());
        assert!(profile
            .workers
            .iter()
            .any(|worker| worker.accumulate_ns > 0));
    }
}
