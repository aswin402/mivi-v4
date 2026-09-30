use super::{PackedError, PackedWeights, Q4_K_BLOCK_SIZE};

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

            let component_energy: f64 = (0..GROUPS)
                .map(|g| {
                    let bytes =
                        &packed_weights.quants[(g / 2) * GROUP_WIDTH..(g / 2 + 1) * GROUP_WIDTH];
                    let quantized_products = bytes
                        .iter()
                        .zip(&activation.values[g * GROUP_WIDTH..(g + 1) * GROUP_WIDTH])
                        .map(|(&byte, &x)| {
                            let q = if g % 2 == 0 { byte & 15 } else { byte >> 4 };
                            (f64::from(q), f64::from(x))
                        });
                    let mut weighted_magnitude = 0.0f64;
                    let mut activation_magnitude = 0.0f64;
                    for (q, x) in quantized_products {
                        weighted_magnitude += q * x.abs();
                        activation_magnitude += x.abs();
                    }
                    let scale_component = f64::from(activation.scales[g].abs())
                        * f64::from(packed_weights.scale.abs())
                        * f64::from((packed_weights.scales[g] as f32).abs())
                        * weighted_magnitude;
                    let min_component = f64::from(activation.scales[g].abs())
                        * f64::from(packed_weights.min_scale.abs())
                        * f64::from((packed_weights.mins[g] as f32).abs())
                        * activation_magnitude;
                    scale_component + min_component
                })
                .sum::<f64>();
            // Conservative 64-operation F32 budget includes decoded-weight and
            // activation reconstruction rounding, affine terms and group summation.
            let gamma = 64.0 * f64::from(f32::EPSILON) / (1.0 - 64.0 * f64::from(f32::EPSILON));
            let arithmetic_bound = gamma * component_energy + 64.0 * f64::from(f32::from_bits(1));
            assert!(
                (f64::from(actual) - expected).abs() <= arithmetic_bound,
                "seed={seed}: actual={actual}, expected={expected}, bound={arithmetic_bound}"
            );
            let packing_bound: f64 = (0..Q4_K_BLOCK_SIZE)
                .map(|i| {
                    let rounding = 0.501 * f64::from(activation.scales[i / GROUP_WIDTH].abs());
                    f64::from(decoded[i].abs())
                        * (rounding + f64::from(reconstructed[i].abs()) * f64::from(f32::EPSILON))
                })
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
