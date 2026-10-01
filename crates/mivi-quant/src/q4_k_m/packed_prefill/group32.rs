use super::{PackedError, PackedWeights, Q4_K_BLOCK_SIZE, Q4_K_BYTES};
use rayon::prelude::*;

pub(super) const GROUP_WIDTH: usize = 32;
pub(super) const GROUPS: usize = Q4_K_BLOCK_SIZE / GROUP_WIDTH;

const _: () = {
    assert!(Q4_K_BLOCK_SIZE % GROUP_WIDTH == 0);
    assert!(GROUPS == 8);
};

pub(super) struct Group32Activation {
    scales: [f32; GROUPS],
    values: [i8; Q4_K_BLOCK_SIZE],
    sums: [i16; GROUPS],
}

pub(super) fn packed_matmul(
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
    let used_weight_bytes = rows
        .checked_mul(row_bytes)
        .ok_or(crate::QuantError::ArithmeticOverflow)?;
    if batch == 0 || rows == 0 {
        return Ok(());
    }

    for block in weights[..used_weight_bytes].chunks_exact(Q4_K_BYTES) {
        let weight = PackedWeights::new(block);
        if !weight.scale.is_finite() || !weight.min_scale.is_finite() {
            return Err(PackedError::NonFiniteWeight);
        }
    }

    let activation_count = batch
        .checked_mul(blocks)
        .ok_or(crate::QuantError::ArithmeticOverflow)?;
    let activations = inputs[..batch * cols]
        .chunks_exact(Q4_K_BLOCK_SIZE)
        .map(Group32Activation::pack)
        .collect::<Result<Vec<_>, _>>()?;
    debug_assert_eq!(activations.len(), activation_count);

    let result_count = rows
        .checked_mul(batch)
        .ok_or(crate::QuantError::ArithmeticOverflow)?;
    let mut row_major = vec![0.0f32; result_count];
    let compute = |row: usize, output: &mut [f32]| {
        for block in 0..blocks {
            let start = row * row_bytes + block * Q4_K_BYTES;
            let weight = PackedWeights::new(&weights[start..start + Q4_K_BYTES]);
            for (b, value) in output.iter_mut().enumerate() {
                *value += activations[b * blocks + block].dot(&weight);
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
    if row_major.iter().any(|value| !value.is_finite()) {
        return Err(PackedError::NonFiniteOutput);
    }
    for row in 0..rows {
        for b in 0..batch {
            out[b * rows + row] = row_major[row * batch + b];
        }
    }
    Ok(())
}

impl Group32Activation {
    pub(super) fn pack(input: &[f32]) -> Result<Self, PackedError> {
        if input.len() != Q4_K_BLOCK_SIZE {
            return Err(crate::QuantError::BufferTooSmall {
                expected: Q4_K_BLOCK_SIZE,
                actual: input.len(),
            }
            .into());
        }
        if input.iter().any(|x| !x.is_finite()) {
            return Err(PackedError::NonFinite);
        }
        let mut packed = Self {
            scales: [0.0; GROUPS],
            values: [0; Q4_K_BLOCK_SIZE],
            sums: [0; GROUPS],
        };
        for (g, group) in input.chunks_exact(GROUP_WIDTH).enumerate() {
            let maximum = group
                .iter()
                .copied()
                .fold(0.0f32, |m, x| if x.abs() > m.abs() { x } else { m });
            if maximum == 0.0 {
                continue;
            }
            packed.scales[g] = maximum / -127.0;
            if packed.scales[g] == 0.0 {
                return Err(PackedError::ScaleUnderflow);
            }
            for (i, &value) in group.iter().enumerate() {
                packed.values[g * GROUP_WIDTH + i] = ((value / maximum) * -127.0)
                    .round_ties_even()
                    .clamp(-127.0, 127.0)
                    as i8;
            }
            packed.sums[g] = packed.values[g * GROUP_WIDTH..(g + 1) * GROUP_WIDTH]
                .iter()
                .map(|&x| i16::from(x))
                .sum();
        }
        Ok(packed)
    }

    pub(super) fn reconstructed(&self) -> [f32; Q4_K_BLOCK_SIZE] {
        std::array::from_fn(|i| f32::from(self.values[i]) * self.scales[i / GROUP_WIDTH])
    }

    pub(super) fn dot(&self, weight: &PackedWeights<'_>) -> f32 {
        let mut result = 0.0f32;
        // Ascending format-group order keeps the scalar combination deterministic.
        // Each integer dot is bounded by 32 * 15 * 127; each sum by 32 * 127.
        for g in 0..GROUPS {
            let bytes = &weight.quants[(g / 2) * GROUP_WIDTH..(g / 2 + 1) * GROUP_WIDTH];
            let values = &self.values[g * GROUP_WIDTH..(g + 1) * GROUP_WIDTH];
            let integer_dot: i32 = bytes
                .iter()
                .zip(values)
                .map(|(&byte, &x)| {
                    let q = if g % 2 == 0 { byte & 15 } else { byte >> 4 };
                    i32::from(q) * i32::from(x)
                })
                .sum();
            let term = self.scales[g]
                * (weight.scale * weight.scales[g] as f32 * integer_dot as f32
                    - weight.min_scale * weight.mins[g] as f32 * f32::from(self.sums[g]));
            result += term;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::q4_k_m::dequantize_q4_k_m;
    use crate::q4_k_m::packed_prefill::{fixture_weights, PackedActivation};

    fn assert_close_to_reference(rows: usize, batch: usize, cols: usize, seed: usize) {
        let blocks = cols / Q4_K_BLOCK_SIZE;
        let weights = fixture_weights(rows, cols, seed);
        let inputs: Vec<_> = (0..batch * cols)
            .map(|i| (i * 37 % 101) as f32 / 31.0 - 1.6)
            .collect();
        let mut out = vec![9876.5; rows * batch + 1];
        packed_matmul(&mut out, &weights, &inputs, batch, rows, cols).unwrap();

        for row in 0..rows {
            let row_weights = &weights[row * blocks * Q4_K_BYTES..(row + 1) * blocks * Q4_K_BYTES];
            for b in 0..batch {
                let mut expected = 0.0f64;
                let mut component_energy = 0.0f64;
                let mut block_magnitudes = 0.0f64;
                for block in 0..blocks {
                    let start = b * cols + block * Q4_K_BLOCK_SIZE;
                    let activation =
                        Group32Activation::pack(&inputs[start..start + Q4_K_BLOCK_SIZE]).unwrap();
                    let reconstructed = activation.reconstructed();
                    let weights_start = block * Q4_K_BYTES;
                    let mut decoded_block = [0.0; Q4_K_BLOCK_SIZE];
                    dequantize_q4_k_m(
                        &row_weights[weights_start..weights_start + Q4_K_BYTES],
                        &mut decoded_block,
                    );
                    let block_expected: f64 = decoded_block
                        .iter()
                        .zip(reconstructed)
                        .map(|(&w, x)| f64::from(w) * f64::from(x))
                        .sum();
                    expected += block_expected;
                    block_magnitudes += block_expected.abs();

                    let packed_weight =
                        PackedWeights::new(&row_weights[weights_start..weights_start + Q4_K_BYTES]);
                    component_energy += component_energy_bound(&activation, &packed_weight);
                }
                let internal = (blocks * 64) as f64 * f64::from(f32::EPSILON);
                let internal = internal / (1.0 - internal) * component_energy
                    + (blocks * 64) as f64 * f64::from(f32::from_bits(1));
                let interblock = if blocks > 1 {
                    (blocks as f64 * f64::from(f32::EPSILON))
                        / (1.0 - blocks as f64 * f64::from(f32::EPSILON))
                        * (block_magnitudes + internal)
                } else {
                    0.0
                };
                let actual = out[b * rows + row];
                assert!(
                    (f64::from(actual) - expected).abs() <= internal + interblock,
                    "row={row}, batch={b}, cols={cols}: actual={actual}, expected={expected}, bound={}",
                    internal + interblock
                );
            }
        }
        assert_eq!(out[rows * batch], 9876.5);
    }

    fn component_energy_bound(activation: &Group32Activation, weight: &PackedWeights<'_>) -> f64 {
        (0..GROUPS)
            .map(|group| {
                let bytes =
                    &weight.quants[(group / 2) * GROUP_WIDTH..(group / 2 + 1) * GROUP_WIDTH];
                let activations =
                    &activation.values[group * GROUP_WIDTH..(group + 1) * GROUP_WIDTH];
                let (mut weighted, mut activation_sum) = (0.0f64, 0.0f64);
                for (&byte, &x) in bytes.iter().zip(activations) {
                    let q = if group % 2 == 0 { byte & 15 } else { byte >> 4 };
                    weighted += f64::from(q) * f64::from(x).abs();
                    activation_sum += f64::from(x).abs();
                }
                f64::from(activation.scales[group].abs())
                    * (f64::from(weight.scale.abs())
                        * f64::from((weight.scales[group] as f32).abs())
                        * weighted
                        + f64::from(weight.min_scale.abs())
                            * f64::from((weight.mins[group] as f32).abs())
                            * activation_sum)
            })
            .sum()
    }

    fn original_input_reconstruction_bound(
        input: &[f32; Q4_K_BLOCK_SIZE],
        activation: &Group32Activation,
    ) -> [f64; Q4_K_BLOCK_SIZE] {
        let unit_roundoff = f64::from(f32::EPSILON) / 2.0;
        // One minimum subnormal safely bounds each operation's absolute rounding
        // error, including the underflow region where a relative bound is weak.
        let subnormal_floor = f64::from(f32::from_bits(1));
        let reconstructed = activation.reconstructed();
        std::array::from_fn(|i| {
            let group = i / GROUP_WIDTH;
            let values = &input[group * GROUP_WIDTH..(group + 1) * GROUP_WIDTH];
            let maximum =
                values
                    .iter()
                    .copied()
                    .fold(0.0f32, |m, x| if x.abs() > m.abs() { x } else { m });
            if maximum == 0.0 {
                return 0.0;
            }

            let magnitude = f64::from(maximum.abs());
            let normalized = input[i] / maximum;
            let quantized = f64::from(activation.values[i].abs());
            let ideal_scale_magnitude = magnitude / 127.0;

            // q is rounded from fl(fl(x / m) * -127). Relative to the
            // ideal integer coordinate, the two F32 operations contribute
            // 127 times the division error and the multiplication error;
            // integer rounding contributes at most one half.
            let division_error =
                unit_roundoff * (f64::from(input[i]).abs() / magnitude) + subnormal_floor;
            let multiplication_error =
                unit_roundoff * 127.0 * f64::from(normalized).abs() + subnormal_floor;
            let quantization_error =
                magnitude / 127.0 * (127.0 * division_error + multiplication_error + 0.5);

            // The stored scale rounds -m/127 to F32, then reconstruction
            // rounds q*scale to F32. Include both errors, including their
            // absolute subnormal floors.
            let scale_error = unit_roundoff * ideal_scale_magnitude + subnormal_floor;
            let scale_rounding_error = quantized * scale_error;
            let reconstruction_error = unit_roundoff
                * (quantized * f64::from(activation.scales[group]).abs())
                + subnormal_floor;
            let measured_reconstruction_rounding = (f64::from(reconstructed[i])
                - f64::from(activation.values[i]) * f64::from(activation.scales[group]))
            .abs();

            quantization_error
                + scale_rounding_error
                + reconstruction_error.max(measured_reconstruction_rounding)
        })
    }

    #[test]
    fn group32_matmul_matches_independent_f64_reference_and_preserves_tail() {
        for (rows, batch, cols) in [(3, 2, 512), (1, 1, 256), (257, 3, 256)] {
            assert_close_to_reference(rows, batch, cols, 47);
        }
    }

    #[test]
    fn group32_matmul_validates_zero_work_shapes_and_preserves_tail() {
        for (rows, batch, cols) in [(0, 3, 256), (3, 0, 256), (3, 2, 0)] {
            let weights = fixture_weights(rows, cols, 47);
            let inputs = vec![f32::NAN; batch * cols + 1];
            let mut out = vec![9876.5; rows * batch + 1];
            packed_matmul(&mut out, &weights, &inputs, batch, rows, cols).unwrap();
            assert!(out[..rows * batch].iter().all(|&x| x == 0.0));
            assert_eq!(out[rows * batch], 9876.5);
        }
    }

    #[test]
    fn group32_matmul_all_zero_inputs_produce_zero_and_ignore_unused_tails() {
        let rows = 3;
        let cols = 512;
        let mut weights = fixture_weights(rows, cols, 47);
        weights.extend_from_slice(&half::f16::INFINITY.to_le_bytes());
        weights.extend_from_slice(&[0; Q4_K_BYTES - 2]);
        let mut inputs = vec![0.0; 2 * cols + 1];
        inputs[2 * cols] = f32::NAN;
        let mut out = [9876.5; 7];
        packed_matmul(&mut out, &weights, &inputs, 2, rows, cols).unwrap();
        assert!(out[..rows * 2].iter().all(|&x| x == 0.0));
        assert_eq!(out[rows * 2..], [9876.5]);
    }

    #[test]
    fn group32_matmul_rejects_before_output_write() {
        let weights = fixture_weights(3, 512, 47);
        let mut input = vec![0.25; 2 * 512];
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            input[512 + 33] = invalid;
            let mut out = [123.0; 7];
            assert!(matches!(
                packed_matmul(&mut out, &weights, &input, 2, 3, 512),
                Err(PackedError::NonFinite)
            ));
            assert_eq!(out, [123.0; 7]);
        }
    }

    #[test]
    fn group32_matmul_rejects_malformed_inputs_and_late_block_underflow_transactionally() {
        let weights = fixture_weights(3, 512, 47);
        let mut out = [123.0; 7];
        let input = vec![1.0; 2 * 512];
        assert!(packed_matmul(&mut out[..5], &weights, &input, 2, 3, 512).is_err());
        assert_eq!(out, [123.0; 7]);
        assert!(packed_matmul(&mut out, &weights[..weights.len() - 1], &input, 2, 3, 512).is_err());
        assert_eq!(out, [123.0; 7]);
        assert!(packed_matmul(&mut out, &weights, &input[..input.len() - 1], 2, 3, 512).is_err());
        assert_eq!(out, [123.0; 7]);
        assert!(packed_matmul(&mut out, &weights, &input, 2, 3, 511).is_err());
        assert_eq!(out, [123.0; 7]);

        let mut tiny = vec![1.0; 2 * 512];
        tiny[256..512].fill(f32::from_bits(1));
        assert!(packed_matmul(&mut out, &weights, &tiny, 2, 3, 512).is_err());
        assert_eq!(out, [123.0; 7]);
    }

    #[test]
    fn group32_matmul_rejects_nonfinite_weight_headers_transactionally() {
        for header in [0, 2] {
            let mut weights = fixture_weights(1, 256, 47);
            weights[header..header + 2].copy_from_slice(&half::f16::INFINITY.to_le_bytes());
            let mut out = [123.0; 2];
            assert!(matches!(
                packed_matmul(&mut out, &weights, &[1.0; 256], 1, 1, 256),
                Err(PackedError::NonFiniteWeight)
            ));
            assert_eq!(out, [123.0; 2]);
        }
    }

    #[test]
    fn group32_matmul_rejects_nonfinite_derived_output_transactionally() {
        let mut weights = fixture_weights(1, 256, 47);
        weights[..2].copy_from_slice(&half::f16::from_f32(65504.0).to_le_bytes());
        weights[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
        let mut out = [123.0; 2];
        assert!(matches!(
            packed_matmul(&mut out, &weights, &[f32::MAX; 256], 1, 1, 256),
            Err(PackedError::NonFiniteOutput)
        ));
        assert_eq!(out, [123.0; 2]);
    }

    #[test]
    fn group32_matmul_rejects_checked_product_overflow_before_output_write() {
        let mut out = [123.0; 1];
        assert!(packed_matmul(&mut out, &[], &[], usize::MAX, 2, 0).is_err());
        assert_eq!(out, [123.0; 1]);
        assert!(packed_matmul(&mut out, &[], &[], 1, 1, usize::MAX).is_err());
        assert_eq!(out, [123.0; 1]);
    }

    #[test]
    fn group32_pack_limits_cross_group_outlier_error() {
        let mut input = [0.25f32; Q4_K_BLOCK_SIZE];
        input[0] = 12.0;
        let grouped = Group32Activation::pack(&input).unwrap().reconstructed();
        let old = PackedActivation::pack(&input).unwrap();
        let old_error: f64 = input
            .iter()
            .zip(&old.values)
            .map(|(&x, &q)| (f64::from(x) - f64::from(f32::from(q) * old.scale)).powi(2))
            .sum();
        let new_error: f64 = input
            .iter()
            .zip(grouped)
            .map(|(&x, y)| (f64::from(x) - f64::from(y)).powi(2))
            .sum();
        assert!(new_error < old_error);
        assert!(grouped[GROUP_WIDTH..].iter().all(|&x| x == 0.25));
    }

    #[test]
    fn group32_pack_handles_zeroes_signs_ties_and_tiny_inputs() {
        let zero = Group32Activation::pack(&[0.0; Q4_K_BLOCK_SIZE]).unwrap();
        assert_eq!(zero.reconstructed(), [0.0; Q4_K_BLOCK_SIZE]);
        assert_eq!(zero.scales, [0.0; GROUPS]);
        assert_eq!(zero.sums, [0; GROUPS]);

        let mut signs = [0.0; Q4_K_BLOCK_SIZE];
        signs[0] = 1.0;
        signs[GROUP_WIDTH] = -1.0;
        let packed = Group32Activation::pack(&signs).unwrap();
        assert_eq!(packed.scales[0], -1.0 / 127.0);
        assert_eq!(packed.scales[1], 1.0 / 127.0);
        assert_eq!(packed.values[0], -127);
        assert_eq!(packed.values[GROUP_WIDTH], -127);
        let reconstructed = packed.reconstructed();
        assert_eq!(reconstructed[0], 1.0);
        assert_eq!(reconstructed[GROUP_WIDTH], -1.0);

        let mut ties = [-127.0; Q4_K_BLOCK_SIZE];
        ties[0] = 0.5;
        ties[1] = 1.5;
        ties[2] = -0.5;
        ties[3] = -1.5;
        let tied = Group32Activation::pack(&ties).unwrap();
        assert_eq!(&tied.values[..4], &[0, 2, 0, -2]);
        assert_eq!(tied.scales[0], 1.0);

        let tiny = Group32Activation::pack(&[1.0e-30; Q4_K_BLOCK_SIZE]).unwrap();
        assert!(tiny.reconstructed().iter().all(|x| x.is_finite()));
    }

    #[test]
    fn group32_pack_rejects_malformed_nonfinite_and_underflow_inputs() {
        assert!(Group32Activation::pack(&[0.0; Q4_K_BLOCK_SIZE - 1]).is_err());
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut input = [1.0; Q4_K_BLOCK_SIZE];
            input[GROUP_WIDTH] = invalid;
            assert!(matches!(
                Group32Activation::pack(&input),
                Err(PackedError::NonFinite)
            ));
        }
        assert!(matches!(
            Group32Activation::pack(&[f32::from_bits(1); Q4_K_BLOCK_SIZE]),
            Err(PackedError::ScaleUnderflow)
        ));
    }

    #[test]
    fn group32_pack_original_input_bound_covers_signed_subnormal_scale_rounding() {
        for value in [
            f32::from_bits(100),
            -f32::from_bits(100),
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            1.25,
            -3.5,
            f32::MAX,
            -f32::MAX,
        ] {
            let input = [value; Q4_K_BLOCK_SIZE];
            let activation = Group32Activation::pack(&input).unwrap();
            let reconstructed = activation.reconstructed();
            let bounds = original_input_reconstruction_bound(&input, &activation);

            for i in 0..Q4_K_BLOCK_SIZE {
                let error = (f64::from(input[i]) - f64::from(reconstructed[i])).abs();
                assert!(
                    error <= bounds[i],
                    "input={:?}, index={i}: error={error:e}, bound={:e}",
                    input[i].to_bits(),
                    bounds[i]
                );
            }
        }
    }

    #[test]
    fn group32_dot_matches_independent_dequantized_reference_and_rounding_bound() {
        for seed in [0, 1, 47, 255] {
            let weights = fixture_weights(1, Q4_K_BLOCK_SIZE, seed);
            let packed_weights = PackedWeights::new(&weights);
            let mut decoded = [0.0; Q4_K_BLOCK_SIZE];
            dequantize_q4_k_m(&weights, &mut decoded);
            let mut input: [f32; Q4_K_BLOCK_SIZE] =
                std::array::from_fn(|i| (i * 37 % 101) as f32 / 31.0 - 1.6);
            input[0] = 12.0;
            input[GROUP_WIDTH] = -9.0;
            input[2 * GROUP_WIDTH] = 0.0;
            let activation = Group32Activation::pack(&input).unwrap();
            let reconstructed = activation.reconstructed();
            let expected: f64 = decoded
                .iter()
                .zip(reconstructed)
                .map(|(&w, x)| f64::from(w) * f64::from(x))
                .sum();
            let actual = activation.dot(&packed_weights);
            assert!(
                expected.abs() > 1.0,
                "fixture should distinguish a zero stub"
            );

            let component_energy = component_energy_bound(&activation, &packed_weights);
            // Conservative 64-operation F32 budget includes decoded-weight and
            // activation reconstruction rounding, affine terms and group summation.
            let gamma = 64.0 * f64::from(f32::EPSILON) / (1.0 - 64.0 * f64::from(f32::EPSILON));
            let arithmetic_bound = gamma * component_energy + 64.0 * f64::from(f32::from_bits(1));
            assert!(
                (f64::from(actual) - expected).abs() <= arithmetic_bound,
                "seed={seed}: actual={actual}, expected={expected}, bound={arithmetic_bound}"
            );
            let packing_bound: f64 = original_input_reconstruction_bound(&input, &activation)
                .iter()
                .zip(decoded)
                .map(|(&bound, weight)| bound * f64::from(weight.abs()))
                .sum();
            let original: f64 = decoded
                .iter()
                .zip(&input)
                .map(|(&w, &x)| f64::from(w) * f64::from(x))
                .sum();
            let original_error = (f64::from(actual) - original).abs();
            assert!(
                original_error
                    <= packing_bound
                        + arithmetic_bound
                        + original.abs() * 64.0 * f64::from(f32::EPSILON)
            );
        }
    }
}
