//! Benchmark-only packed activation experiment. Never used by inference.
//! Inspired by GGML Q8_K activation blocks and Q4_K x Q8_K CPU dot products.
//! Sources: llama.cpp `ggml/src/ggml-quants.c` and `ggml/src/ggml-cpu/ggml-cpu.c`.
//! Signed-scale activation rounding is not a bit-identical GGML serialization
//! contract. Synthetic error/performance results do not establish model quality.

use super::{dequantize_q4_k_m, get_scale_min_k4, Q4_K_BLOCK_SIZE, Q4_K_BYTES};
use rayon::prelude::*;

#[derive(Debug, thiserror::Error)]
enum PackedError {
    #[error(transparent)]
    Shape(#[from] crate::QuantError),
    #[error("packed activations require finite inputs")]
    NonFinite,
    #[error("packed activation scale underflows")]
    ScaleUnderflow,
}

struct PackedActivation {
    scale: f32,
    values: [i8; Q4_K_BLOCK_SIZE],
    sums: [i16; 16],
}

impl PackedActivation {
    fn pack(input: &[f32]) -> Result<Self, PackedError> {
        if input.len() != Q4_K_BLOCK_SIZE {
            return Err(crate::QuantError::BufferTooSmall {
                expected: Q4_K_BLOCK_SIZE,
                actual: input.len(),
            }
            .into());
        }
        let mut max = 0.0f32;
        for &value in input {
            if !value.is_finite() {
                return Err(PackedError::NonFinite);
            }
            if value.abs() > max.abs() {
                max = value;
            }
        }
        let mut packed = Self {
            scale: 0.0,
            values: [0; Q4_K_BLOCK_SIZE],
            sums: [0; 16],
        };
        if max == 0.0 {
            return Ok(packed);
        }
        packed.scale = max / -127.0;
        if packed.scale == 0.0 {
            return Err(PackedError::ScaleUnderflow);
        }
        // Normalize before scaling to avoid an infinite reciprocal for tiny
        // finite inputs. This is not a bit-identical GGML quantizer contract.
        for (out, &value) in packed.values.iter_mut().zip(input) {
            *out = ((value / max) * -127.0)
                .round_ties_even()
                .clamp(-127.0, 127.0) as i8;
        }
        for (sum, group) in packed.sums.iter_mut().zip(packed.values.chunks_exact(16)) {
            *sum = group.iter().map(|&v| i16::from(v)).sum();
        }
        Ok(packed)
    }
}

struct PackedWeights<'a> {
    scale: f32,
    min_scale: f32,
    scales: [i32; 8],
    mins: [i32; 8],
    quants: &'a [u8],
}

impl<'a> PackedWeights<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        let mut scales = [0; 8];
        let mut mins = [0; 8];
        for (i, (scale, min)) in scales.iter_mut().zip(&mut mins).enumerate() {
            let (s, m) = get_scale_min_k4(i, &bytes[4..16]);
            *scale = i32::from(s);
            *min = i32::from(m);
        }
        Self {
            scale: half::f16::from_le_bytes([bytes[0], bytes[1]]).to_f32(),
            min_scale: half::f16::from_le_bytes([bytes[2], bytes[3]]).to_f32(),
            scales,
            mins,
            quants: &bytes[16..Q4_K_BYTES],
        }
    }

    fn dot(&self, input: &PackedActivation) -> f32 {
        let mut weighted_dot = 0i32;
        let mut weighted_min = 0i32;
        for group in 0..4 {
            let bytes = &self.quants[group * 32..(group + 1) * 32];
            let values = &input.values[group * 64..(group + 1) * 64];
            let (low_values, high_values) = values.split_at(32);
            // A nibble times a signed activation fits in i16. Express that
            // bound so the compiler can choose narrower integer operations.
            let low: i32 = bytes
                .iter()
                .zip(low_values)
                .map(|(&q, &x)| i32::from(i16::from(q & 15) * i16::from(x)))
                .sum();
            let high: i32 = bytes
                .iter()
                .zip(high_values)
                .map(|(&q, &x)| i32::from(i16::from(q >> 4) * i16::from(x)))
                .sum();
            weighted_dot += self.scales[group * 2] * low + self.scales[group * 2 + 1] * high;
            for sub in [group * 2, group * 2 + 1] {
                let sum = i32::from(input.sums[sub * 2]) + i32::from(input.sums[sub * 2 + 1]);
                weighted_min += self.mins[sub] * sum;
            }
        }
        input.scale * (self.scale * weighted_dot as f32 - self.min_scale * weighted_min as f32)
    }
}

fn packed_matmul(
    out: &mut [f32],
    weights: &[u8],
    inputs: &[f32],
    batch: usize,
    rows: usize,
    cols: usize,
) -> Result<(), PackedError> {
    let blocks = cols / Q4_K_BLOCK_SIZE;
    let row_bytes = blocks
        .checked_mul(Q4_K_BYTES)
        .ok_or(crate::QuantError::ArithmeticOverflow)?;
    crate::validate_matmul_args(
        out,
        weights,
        inputs,
        batch,
        rows,
        cols,
        row_bytes,
        Q4_K_BLOCK_SIZE,
    )?;
    if batch == 0 || rows == 0 {
        return Ok(());
    }
    let activations: Vec<_> = inputs[..batch * cols]
        .chunks_exact(Q4_K_BLOCK_SIZE)
        .map(PackedActivation::pack)
        .collect::<Result<_, _>>()?;
    let mut row_major = vec![0.0; rows * batch];
    let compute = |row: usize, output: &mut [f32]| {
        for block_index in 0..blocks {
            let start = row * row_bytes + block_index * Q4_K_BYTES;
            let weight = PackedWeights::new(&weights[start..start + Q4_K_BYTES]);
            for (b, value) in output.iter_mut().enumerate() {
                *value += weight.dot(&activations[b * blocks + block_index]);
            }
        }
    };
    if rows >= crate::RAYON_PARALLEL_THRESHOLD && rayon::current_num_threads() > 1 {
        row_major
            .par_chunks_mut(batch)
            .enumerate()
            .for_each(|(row, output)| compute(row, output));
    } else {
        row_major
            .chunks_mut(batch)
            .enumerate()
            .for_each(|(row, output)| compute(row, output));
    }
    for row in 0..rows {
        for b in 0..batch {
            out[b * rows + row] = row_major[row * batch + b];
        }
    }
    Ok(())
}

#[test]
fn activation_pack_preserves_zero_and_bounds_rounding_error() {
    let zero = PackedActivation::pack(&[0.0; Q4_K_BLOCK_SIZE]).unwrap();
    assert_eq!(zero.scale, 0.0);
    assert_eq!(zero.values, [0; Q4_K_BLOCK_SIZE]);
    assert_eq!(zero.sums, [0; 16]);
    for amplitude in [-123.0, -1.0, -0.125, 0.125, 1.0, 123.0] {
        let input: Vec<f32> = (0..Q4_K_BLOCK_SIZE)
            .map(|i| ((i * 37 % 101) as f32 / 31.0 - 1.6) * amplitude)
            .collect();
        let packed = PackedActivation::pack(&input).unwrap();
        for (actual, expected) in packed.values.iter().zip(&input) {
            let restored = f32::from(*actual) * packed.scale;
            assert!(
                (restored - expected).abs()
                    <= packed.scale.abs() * 0.5001 + expected.abs() * f32::EPSILON
            );
        }
        for (group, sum) in packed.values.chunks_exact(16).zip(&packed.sums) {
            assert_eq!(group.iter().map(|&v| i16::from(v)).sum::<i16>(), *sum);
        }
    }
}

#[test]
fn activation_pack_rejects_nonfinite_and_short_inputs() {
    assert!(PackedActivation::pack(&[0.0; 255]).is_err());
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut input = [1.0; Q4_K_BLOCK_SIZE];
        input[123] = invalid;
        assert!(PackedActivation::pack(&input).is_err());
    }
    assert!(matches!(
        PackedActivation::pack(&[f32::from_bits(1); Q4_K_BLOCK_SIZE]),
        Err(PackedError::ScaleUnderflow)
    ));
}

#[test]
fn packed_matmul_covers_shapes_tails_and_quantization_error_bounds() {
    for (rows, batch, cols) in [
        (0, 3, 256),
        (3, 0, 256),
        (3, 32, 0),
        (1, 1, 256),
        (3, 9, 512),
        (257, 32, 256),
        (257, 64, 512),
    ] {
        let weights = fixture_weights(rows, cols, 47);
        let inputs: Vec<f32> = (0..batch * cols)
            .map(|i| (i * 37 % 101) as f32 / 31.0 - 1.6)
            .collect();
        let mut actual = vec![123.0; rows * batch + 1];
        packed_matmul(&mut actual, &weights, &inputs, batch, rows, cols).unwrap();
        assert_eq!(actual[rows * batch], 123.0);
        let mut decoded = vec![0.0; cols];
        for row in 0..rows {
            let start = row * (cols / Q4_K_BLOCK_SIZE) * Q4_K_BYTES;
            super::dequantize_q4_k_m_slice(
                &weights[start..start + (cols / Q4_K_BLOCK_SIZE) * Q4_K_BYTES],
                &mut decoded,
            );
            for b in 0..batch {
                let input = &inputs[b * cols..(b + 1) * cols];
                let expected: f64 = decoded
                    .iter()
                    .zip(input)
                    .map(|(&w, &x)| f64::from(w) * f64::from(x))
                    .sum();
                let magnitude: f64 = decoded
                    .iter()
                    .zip(input)
                    .map(|(&w, &x)| (f64::from(w) * f64::from(x)).abs())
                    .sum();
                let bound: f64 = decoded
                    .chunks_exact(Q4_K_BLOCK_SIZE)
                    .zip(input.chunks_exact(Q4_K_BLOCK_SIZE))
                    .map(|(w, x)| {
                        let packed = PackedActivation::pack(x).unwrap();
                        w.iter().map(|w| f64::from(w.abs())).sum::<f64>()
                            * f64::from(packed.scale.abs())
                            * 0.501
                    })
                    .sum();
                assert!(
                    (f64::from(actual[b * rows + row]) - expected).abs()
                        <= bound + magnitude * 1e-6 + 1e-6,
                    "rows={rows}, batch={batch}, cols={cols}, row={row}"
                );
            }
        }
    }
}

#[test]
fn packed_matmul_rejects_invalid_buffers_before_writing() {
    let weights = fixture_weights(3, 256, 47);
    let mut out = [123.0; 6];
    assert!(packed_matmul(
        &mut out,
        &weights[..weights.len() - 1],
        &[0.0; 512],
        2,
        3,
        256
    )
    .is_err());
    assert!(packed_matmul(&mut out[..5], &weights, &[0.0; 512], 2, 3, 256).is_err());
    assert!(packed_matmul(&mut out, &weights, &[0.0; 511], 2, 3, 256).is_err());
    assert!(packed_matmul(&mut out, &weights, &[0.0; 510], 2, 3, 255).is_err());
    assert!(packed_matmul(&mut out, &weights, &[f32::NAN; 512], 2, 3, 256).is_err());
    assert_eq!(out, [123.0; 6]);
}

#[test]
fn packed_dot_matches_dequantized_packed_activation() {
    for seed in [0, 1, 47, 255] {
        let weights = fixture_weights(1, Q4_K_BLOCK_SIZE, seed);
        let block = PackedWeights::new(&weights);
        let mut decoded = [0.0; Q4_K_BLOCK_SIZE];
        dequantize_q4_k_m(&weights, &mut decoded);
        let input: Vec<f32> = (0..Q4_K_BLOCK_SIZE)
            .map(|i| (i * 37 % 101) as f32 / 31.0 - 1.6)
            .collect();
        let packed = PackedActivation::pack(&input).unwrap();
        let expected: f64 = decoded
            .iter()
            .zip(&packed.values)
            .map(|(&w, &x)| f64::from(w) * f64::from(x) * f64::from(packed.scale))
            .sum();
        let magnitude: f64 = decoded
            .iter()
            .zip(&packed.values)
            .map(|(&w, &x)| (f64::from(w) * f64::from(x) * f64::from(packed.scale)).abs())
            .sum();
        assert!((f64::from(block.dot(&packed)) - expected).abs() <= magnitude * 1e-6 + 1e-6);

        let original: f64 = decoded
            .iter()
            .zip(&input)
            .map(|(&w, &x)| f64::from(w) * f64::from(x))
            .sum();
        let bound = decoded.iter().map(|w| f64::from(w.abs())).sum::<f64>()
            * f64::from(packed.scale.abs())
            * 0.501;
        assert!(
            (f64::from(block.dot(&packed)) - original).abs() <= bound + magnitude * 1e-6 + 1e-6
        );
    }
}

fn fixture_weights(rows: usize, cols: usize, seed: usize) -> Vec<u8> {
    let mut weights = vec![0; rows * (cols / Q4_K_BLOCK_SIZE) * Q4_K_BYTES];
    for (i, block) in weights.chunks_exact_mut(Q4_K_BYTES).enumerate() {
        for (j, byte) in block.iter_mut().enumerate() {
            *byte = ((i * 19 + j * 37 + seed) % 256) as u8;
        }
        block[..2].copy_from_slice(&half::f16::from_f32(0.03125).to_le_bytes());
        block[2..4].copy_from_slice(&half::f16::from_f32(0.015625).to_le_bytes());
    }
    weights
}

#[test]
#[ignore = "focused release-mode packed activation performance experiment"]
fn packed_prefill_benchmark() {
    use std::hint::black_box;
    use std::time::Instant;
    const ITERS: usize = 8;
    let rows = crate::RAYON_PARALLEL_THRESHOLD + 1;
    println!(
        "Q4_K benchmark only; Rayon threads={}",
        rayon::current_num_threads()
    );
    for cols in [2048, 8192] {
        for batch in [32, 64] {
            let weights = fixture_weights(rows, cols, 47);
            let inputs: Vec<f32> = (0..batch * cols)
                .map(|i| (i * 37 % 101) as f32 / 31.0 - 1.6)
                .collect();
            let mut reference = vec![0.0; batch * rows];
            let mut output = vec![0.0; batch * rows];
            crate::quantized_matmul_rows(
                &mut reference,
                crate::GgmlType::Q4_K,
                &weights,
                &inputs,
                batch,
                rows,
                cols,
            )
            .unwrap();
            packed_matmul(&mut output, &weights, &inputs, batch, rows, cols).unwrap();
            let mut error_squared = 0.0f64;
            let mut reference_squared = 0.0f64;
            let mut max_abs_error = 0.0f32;
            for (&actual, &expected) in output.iter().zip(&reference) {
                error_squared += f64::from(actual - expected).powi(2);
                reference_squared += f64::from(expected).powi(2);
                max_abs_error = max_abs_error.max((actual - expected).abs());
            }
            let relative_l2 = (error_squared / reference_squared.max(f64::MIN_POSITIVE)).sqrt();
            println!("rows={rows}, cols={cols}, batch={batch}: relative L2 error={relative_l2:.6}, max abs error={max_abs_error:.6}");
            assert!(
                relative_l2.is_finite() && relative_l2 < 0.05,
                "synthetic projection error exceeded the experiment's 5% relative L2 gate"
            );
            for packed in [false, true, false, true] {
                let start = Instant::now();
                for _ in 0..ITERS {
                    if packed {
                        packed_matmul(
                            black_box(&mut output),
                            black_box(&weights),
                            black_box(&inputs),
                            batch,
                            rows,
                            cols,
                        )
                        .unwrap();
                    } else {
                        crate::quantized_matmul_rows(
                            black_box(&mut output),
                            crate::GgmlType::Q4_K,
                            black_box(&weights),
                            black_box(&inputs),
                            batch,
                            rows,
                            cols,
                        )
                        .unwrap();
                    }
                    black_box(&output);
                }
                println!("  packed={packed}: {:.3} ms/matmul (includes activation packing or F32 transposition)", start.elapsed().as_secs_f64() * 1000.0 / ITERS as f64);
            }
        }
    }
}
