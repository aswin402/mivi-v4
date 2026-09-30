//! Benchmark-only packed activation experiment. Never used by inference.
//! Inspired by GGML Q8_K activation blocks and Q4_K x Q8_K CPU dot products.
//! Sources: llama.cpp `ggml/src/ggml-quants.c` and `ggml/src/ggml-cpu/ggml-cpu.c`.
//! Signed-scale activation rounding is not a bit-identical GGML serialization
//! contract. Synthetic error/performance results do not establish model quality.

use super::{dequantize_q4_k_m, get_scale_min_k4, Q4_K_BLOCK_SIZE, Q4_K_BYTES};
use rayon::prelude::*;

mod real_weights;
mod group32;

#[derive(Clone, Copy, Debug)]
enum PackedKernel {
    Scalar,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2(pulp::x86::V3),
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2Tiled(pulp::x86::V3),
}

impl PackedKernel {
    fn block_major_activations(self) -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if matches!(self, Self::Avx2Tiled(_)) {
            return true;
        }
        false
    }

    fn label(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Avx2(_) => "runtime AVX2",
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Avx2Tiled(_) => "runtime AVX2 tiled",
        }
    }

    fn detected() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if let Some(simd) = pulp::x86::V3::try_new() {
            return Self::Avx2(simd);
        }
        Self::Scalar
    }

    fn detected_tiled() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if let Some(simd) = pulp::x86::V3::try_new() {
            return Self::Avx2Tiled(simd);
        }
        Self::Scalar
    }
}

#[test]
fn packed_tiled_matmul_matches_scalar_batch_tails() {
    for batch in [1, 3, 4, 5, 7, 8, 9, 31, 32, 33, 63, 64, 65] {
        for seed in [0, 47, 255] {
            let (rows, cols) = (3, 512);
            let weights = fixture_weights(rows, cols, seed);
            let inputs: Vec<_> = (0..batch * cols)
                .map(|i| (i * 53 % 255) as f32 - 127.0)
                .collect();
            let mut expected = vec![123.0; rows * batch + 1];
            let mut actual = expected.clone();
            packed_matmul(&mut expected, &weights, &inputs, batch, rows, cols).unwrap();
            packed_matmul_with_kernel(
                &mut actual,
                &weights,
                &inputs,
                batch,
                rows,
                cols,
                PackedKernel::detected_tiled(),
            )
            .unwrap();
            assert_eq!(actual, expected, "batch={batch}, seed={seed}");
        }
    }
}

#[test]
fn packed_simd_matmul_matches_scalar_shapes() {
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
        let inputs: Vec<_> = (0..batch * cols)
            .map(|i| (i * 37 % 101) as f32 / 31.0 - 1.6)
            .collect();
        let mut scalar = vec![123.0; rows * batch + 1];
        let mut simd = scalar.clone();
        packed_matmul(&mut scalar, &weights, &inputs, batch, rows, cols).unwrap();
        packed_matmul_with_kernel(
            &mut simd,
            &weights,
            &inputs,
            batch,
            rows,
            cols,
            PackedKernel::detected(),
        )
        .unwrap();
        assert_eq!(scalar, simd, "rows={rows}, batch={batch}, cols={cols}");
        packed_matmul_with_kernel(
            &mut simd,
            &weights,
            &inputs,
            batch,
            rows,
            cols,
            PackedKernel::detected_tiled(),
        )
        .unwrap();
        assert_eq!(
            scalar, simd,
            "tiled: rows={rows}, batch={batch}, cols={cols}"
        );
    }
}

#[test]
fn packed_tiled_matmul_preserves_maximum_integer_range() {
    let (rows, batch, cols) = (3, 5, 512);
    let mut weights = vec![255; rows * (cols / Q4_K_BLOCK_SIZE) * Q4_K_BYTES];
    for block in weights.chunks_exact_mut(Q4_K_BYTES) {
        block[..2].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
        block[2..4].copy_from_slice(&half::f16::from_f32(0.25).to_le_bytes());
    }
    let inputs: Vec<_> = (0..batch * cols)
        .map(|i| match i / cols {
            0 => 127.0,
            1 => -127.0,
            2 => {
                if i % 2 == 0 {
                    127.0
                } else {
                    -127.0
                }
            }
            3 => 0.0,
            _ => 1e-20,
        })
        .collect();
    let mut scalar = vec![123.0; rows * batch + 1];
    let mut tiled = scalar.clone();
    packed_matmul(&mut scalar, &weights, &inputs, batch, rows, cols).unwrap();
    packed_matmul_with_kernel(
        &mut tiled,
        &weights,
        &inputs,
        batch,
        rows,
        cols,
        PackedKernel::detected_tiled(),
    )
    .unwrap();
    assert_eq!(scalar, tiled);
    assert!(scalar[0] > 0.0 && scalar[rows] < 0.0);
}

#[test]
fn packed_tiled_rejects_input_without_mutating_output() {
    let weights = fixture_weights(3, 512, 47);
    let mut inputs = vec![1.0; 5 * 512];
    let mut output = [123.0; 16];
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        // Even the last block must be validated before parallel output starts.
        inputs[5 * 512 - 1] = invalid;
        assert!(packed_matmul_with_kernel(
            &mut output,
            &weights,
            &inputs,
            5,
            3,
            512,
            PackedKernel::detected_tiled()
        )
        .is_err());
        assert_eq!(output, [123.0; 16]);
    }
    inputs.fill(f32::from_bits(1));
    assert!(packed_matmul_with_kernel(
        &mut output,
        &weights,
        &inputs,
        5,
        3,
        512,
        PackedKernel::detected_tiled()
    )
    .is_err());
    assert_eq!(output, [123.0; 16]);
    inputs.fill(1.0);
    assert!(packed_matmul_with_kernel(
        &mut output,
        &weights[..weights.len() - 1],
        &inputs,
        5,
        3,
        512,
        PackedKernel::detected_tiled()
    )
    .is_err());
    assert_eq!(output, [123.0; 16]);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn simd_nibble_dots(simd: pulp::x86::V3, bytes: &[u8], values: &[i8]) -> (i32, i32) {
    let bytes: [u8; 32] = bytes.try_into().expect("one nibble group");
    let raw: pulp::i8x32 = pulp::cast(bytes);
    let mask = simd.splat_i8x32(15);
    let low = simd.and_i8x32(raw, mask);
    let high = simd.and_i8x32(
        pulp::cast(simd.shr_const_u16x16::<4>(pulp::cast(raw))),
        mask,
    );
    let low_values: [i8; 32] = values[..32].try_into().expect("low activations");
    let high_values: [i8; 32] = values[32..].try_into().expect("high activations");
    let dot = |quants, activations| {
        // The first operand is interpreted as unsigned by maddubs. Masking
        // ensures 0..15; adjacent sums are at most 2*15*127=3810, so the
        // saturating i16 operation cannot saturate. Widen before reduction.
        let pairs = simd.multiply_saturating_add_adjacent_i8x32(quants, pulp::cast(activations));
        let lanes: [i32; 8] =
            pulp::cast(simd.multiply_wrapping_add_adjacent_i16x16(pairs, simd.splat_i16x16(1)));
        lanes.into_iter().sum()
    };
    (dot(low, low_values), dot(high, high_values))
}

#[test]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn simd_nibble_dots_match_scalar_extremes() {
    let Some(simd) = pulp::x86::V3::try_new() else {
        eprintln!("AVX2 unavailable; SIMD-specific test skipped");
        return;
    };
    for amplitude in [-127i8, 127] {
        let expected = 32 * 15 * i32::from(amplitude);
        assert_eq!(
            simd.vectorize(|| simd_nibble_dots(simd, &[255; 32], &[amplitude; 64])),
            (expected, expected),
            "reduction must widen before exceeding i16 range"
        );
    }
    for seed in 0..256 {
        let bytes: [u8; 32] = std::array::from_fn(|i| ((i * 37 + seed) % 256) as u8);
        for amplitude in [-127i8, -1, 0, 1, 127] {
            let values = [amplitude; 64];
            let expected_low: i32 = bytes
                .iter()
                .map(|&q| i32::from(q & 15) * i32::from(amplitude))
                .sum();
            let expected_high: i32 = bytes
                .iter()
                .map(|&q| i32::from(q >> 4) * i32::from(amplitude))
                .sum();
            assert_eq!(
                simd.vectorize(|| simd_nibble_dots(simd, &bytes, &values)),
                (expected_low, expected_high)
            );
        }
        let values: [i8; 64] = std::array::from_fn(|i| ((i * 53 + seed) % 255) as i8);
        let expected_low: i32 = bytes
            .iter()
            .zip(&values[..32])
            .map(|(&q, &x)| i32::from(q & 15) * i32::from(x))
            .sum();
        let expected_high: i32 = bytes
            .iter()
            .zip(&values[32..])
            .map(|(&q, &x)| i32::from(q >> 4) * i32::from(x))
            .sum();
        assert_eq!(
            simd.vectorize(|| simd_nibble_dots(simd, &bytes, &values)),
            (expected_low, expected_high)
        );
    }
}

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
        self.dot_with_kernel(input, PackedKernel::Scalar)
    }

    #[inline(always)]
    fn dot_with_kernel(&self, input: &PackedActivation, kernel: PackedKernel) -> f32 {
        let mut weighted_dot = 0i32;
        let mut weighted_min = 0i32;
        for group in 0..4 {
            let bytes = &self.quants[group * 32..(group + 1) * 32];
            let values = &input.values[group * 64..(group + 1) * 64];
            let (low_values, high_values) = values.split_at(32);
            // A nibble times a signed activation fits in i16. Express that
            // bound so the compiler can choose narrower integer operations.
            let (low, high) = match kernel {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                PackedKernel::Avx2(simd) | PackedKernel::Avx2Tiled(simd) => {
                    simd_nibble_dots(simd, bytes, values)
                }
                PackedKernel::Scalar => {
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
                    (low, high)
                }
            };
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
    packed_matmul_with_kernel(
        out,
        weights,
        inputs,
        batch,
        rows,
        cols,
        PackedKernel::Scalar,
    )
}

#[allow(clippy::too_many_arguments)]
fn packed_matmul_with_kernel(
    out: &mut [f32],
    weights: &[u8],
    inputs: &[f32],
    batch: usize,
    rows: usize,
    cols: usize,
    kernel: PackedKernel,
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
    let activations: Vec<_> = if kernel.block_major_activations() {
        // Pack directly into block-major order: tokens sharing a weight block
        // are adjacent. No additional copy of the packed activation matrix.
        (0..batch * blocks)
            .map(|i| {
                let start = (i % batch) * cols + (i / batch) * Q4_K_BLOCK_SIZE;
                PackedActivation::pack(&inputs[start..start + Q4_K_BLOCK_SIZE])
            })
            .collect::<Result<_, _>>()?
    } else {
        inputs[..batch * cols]
            .chunks_exact(Q4_K_BLOCK_SIZE)
            .map(PackedActivation::pack)
            .collect::<Result<_, _>>()?
    };
    let mut row_major = vec![0.0; rows * batch];
    // Enter the target-feature context once per row, not once per dot product.
    // Unsupported CPUs retain the scalar reference, with no global ISA flags.
    let dispatch = |row, output: &mut [f32]| match kernel {
        PackedKernel::Scalar => packed_row(
            row,
            output,
            weights,
            &activations,
            blocks,
            row_bytes,
            kernel,
        ),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        PackedKernel::Avx2(simd) => simd.vectorize(|| {
            packed_row(
                row,
                output,
                weights,
                &activations,
                blocks,
                row_bytes,
                kernel,
            )
        }),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        PackedKernel::Avx2Tiled(simd) => simd.vectorize(|| {
            packed_row_tiled(simd, row, output, weights, &activations, blocks, row_bytes)
        }),
    };
    if rows >= crate::RAYON_PARALLEL_THRESHOLD && rayon::current_num_threads() > 1 {
        row_major
            .par_chunks_mut(batch)
            .enumerate()
            .for_each(|(row, output)| dispatch(row, output));
    } else {
        row_major
            .chunks_mut(batch)
            .enumerate()
            .for_each(|(row, output)| dispatch(row, output));
    }
    for row in 0..rows {
        for b in 0..batch {
            out[b * rows + row] = row_major[row * batch + b];
        }
    }
    Ok(())
}

// Keep the row loop inside Pulp's target-feature context; an outlined closure
// otherwise calls SIMD wrappers across that boundary for every inner dot.
#[inline(always)]
fn packed_row(
    row: usize,
    output: &mut [f32],
    weights: &[u8],
    activations: &[PackedActivation],
    blocks: usize,
    row_bytes: usize,
    kernel: PackedKernel,
) {
    for block_index in 0..blocks {
        let start = row * row_bytes + block_index * Q4_K_BYTES;
        let weight = PackedWeights::new(&weights[start..start + Q4_K_BYTES]);
        for (b, value) in output.iter_mut().enumerate() {
            *value += weight.dot_with_kernel(&activations[b * blocks + block_index], kernel);
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn packed_row_tiled(
    simd: pulp::x86::V3,
    row: usize,
    output: &mut [f32],
    weights: &[u8],
    activations: &[PackedActivation],
    blocks: usize,
    row_bytes: usize,
) {
    let batch = output.len();
    for block_index in 0..blocks {
        let start = row * row_bytes + block_index * Q4_K_BYTES;
        let weight = PackedWeights::new(&weights[start..start + Q4_K_BYTES]);
        let prepared = PreparedWeights::new(simd, &weight);
        let remainder_start = output.len() / 4 * 4;
        let mut tiles = output.chunks_exact_mut(4);
        for (tile_index, tile) in tiles.by_ref().enumerate() {
            let inputs =
                std::array::from_fn(|b| &activations[block_index * batch + tile_index * 4 + b]);
            let dots = prepared.dot_tile::<4>(simd, inputs);
            for (value, dot) in tile.iter_mut().zip(dots) {
                *value += dot;
            }
        }
        for (b, value) in tiles.into_remainder().iter_mut().enumerate() {
            *value += prepared.dot_tile::<1>(
                simd,
                [&activations[block_index * batch + remainder_start + b]],
            )[0];
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
struct PreparedWeights {
    quants: [pulp::i8x32; 8],
    scales: [pulp::i16x16; 8],
    mins: pulp::i16x16,
    scale: f32,
    min_scale: f32,
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl PreparedWeights {
    #[inline(always)]
    fn new(simd: pulp::x86::V3, weight: &PackedWeights<'_>) -> Self {
        let mask = simd.splat_i8x32(15);
        let quants = std::array::from_fn(|sub| {
            let start = sub / 2 * 32;
            let bytes: [u8; 32] = weight.quants[start..start + 32]
                .try_into()
                .expect("nibble group");
            let raw: pulp::i8x32 = pulp::cast(bytes);
            if sub % 2 == 0 {
                simd.and_i8x32(raw, mask)
            } else {
                simd.and_i8x32(
                    pulp::cast(simd.shr_const_u16x16::<4>(pulp::cast(raw))),
                    mask,
                )
            }
        });
        let scales = std::array::from_fn(|sub| simd.splat_i16x16(weight.scales[sub] as i16));
        let mins: [i16; 16] = std::array::from_fn(|i| weight.mins[i / 2] as i16);
        Self {
            quants,
            scales,
            mins: pulp::cast(mins),
            scale: weight.scale,
            min_scale: weight.min_scale,
        }
    }

    #[inline(always)]
    fn dot_tile<const N: usize>(
        &self,
        simd: pulp::x86::V3,
        inputs: [&PackedActivation; N],
    ) -> [f32; N] {
        let mut accumulators = [simd.splat_i32x8(0); N];
        for sub in 0..8 {
            // Reuse this weight vector and its scale across the token tile.
            for (accumulator, input) in accumulators.iter_mut().zip(inputs) {
                let values: [i8; 32] = input.values[sub * 32..(sub + 1) * 32]
                    .try_into()
                    .expect("activation group");
                let pairs = simd
                    .multiply_saturating_add_adjacent_i8x32(self.quants[sub], pulp::cast(values));
                // Each pair is bounded by 3810, scale by 63. The next madd
                // widens to i32; the entire 256-element sum is <=30,723,840.
                let scaled = simd.multiply_wrapping_add_adjacent_i16x16(pairs, self.scales[sub]);
                *accumulator = simd.wrapping_add_i32x8(*accumulator, scaled);
            }
        }
        std::array::from_fn(|b| {
            // Reduce once per block/token rather than once per 32 values.
            let lanes: [i32; 8] = pulp::cast(accumulators[b]);
            let weighted_dot: i32 = lanes.into_iter().sum();
            let min_lanes: [i32; 8] = pulp::cast(
                simd.multiply_wrapping_add_adjacent_i16x16(pulp::cast(inputs[b].sums), self.mins),
            );
            let weighted_min: i32 = min_lanes.into_iter().sum();
            inputs[b].scale
                * (self.scale * weighted_dot as f32 - self.min_scale * weighted_min as f32)
        })
    }
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
    const ITERS: usize = 16;
    println!(
        "Q4_K benchmark only; Rayon threads={}",
        rayon::current_num_threads()
    );
    for rows in [crate::RAYON_PARALLEL_THRESHOLD + 1, 1024] {
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
                let simd_kernel = PackedKernel::detected();
                packed_matmul_with_kernel(
                    &mut reference,
                    &weights,
                    &inputs,
                    batch,
                    rows,
                    cols,
                    simd_kernel,
                )
                .unwrap();
                assert_eq!(
                    output, reference,
                    "SIMD must preserve packed reference results"
                );
                let tiled_kernel = PackedKernel::detected_tiled();
                packed_matmul_with_kernel(
                    &mut reference,
                    &weights,
                    &inputs,
                    batch,
                    rows,
                    cols,
                    tiled_kernel,
                )
                .unwrap();
                assert_eq!(
                    output, reference,
                    "tiled SIMD must preserve packed reference results"
                );
                for (label, kernel) in [
                    ("packed tiled", Some(tiled_kernel)),
                    ("packed SIMD", Some(simd_kernel)),
                    ("packed scalar", Some(PackedKernel::Scalar)),
                    ("F32", None),
                    ("F32", None),
                    ("packed scalar", Some(PackedKernel::Scalar)),
                    ("packed SIMD", Some(simd_kernel)),
                    ("packed tiled", Some(tiled_kernel)),
                ] {
                    let start = Instant::now();
                    for _ in 0..ITERS {
                        if let Some(kernel) = kernel {
                            packed_matmul_with_kernel(
                                black_box(&mut output),
                                black_box(&weights),
                                black_box(&inputs),
                                batch,
                                rows,
                                cols,
                                kernel,
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
                    println!("  {label} ({}): {:.3} ms/matmul (includes activation packing or F32 transposition)", kernel.map_or("existing production path", PackedKernel::label), start.elapsed().as_secs_f64() * 1000.0 / ITERS as f64);
                }
            }
        }
    }
}
