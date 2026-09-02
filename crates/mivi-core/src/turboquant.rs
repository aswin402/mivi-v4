//! TurboQuant 4-bit data-oblivious vector quantization with Fast Walsh-Hadamard orthogonal rotation.
//!
//! Based on Google Research & NYU's ICLR 2026 paper:
//! "TurboQuant: Data-Oblivious Vector Quantization with Near-Optimal Distortion" (arXiv:2504.19874).

const DEFAULT_ROTATION_SEED: u64 = 0x517cc1b727220a95;

/// Standard normal 4-bit Lloyd-Max centroids (in units of standard deviation sigma = 1/sqrt(dim)).
/// 16 optimal reconstruction levels for the near-Gaussian marginal distribution.
pub const LLOYD_MAX_4BIT_CENTROIDS: [f32; 16] = [
    -2.152, -1.603, -1.228, -0.923, -0.657, -0.412, -0.177, -0.058,
     0.058,  0.177,  0.412,  0.657,  0.923,  1.228,  1.603,  2.152,
];

/// Decision boundaries between consecutive 4-bit centroids (15 boundary thresholds).
pub const LLOYD_MAX_4BIT_BOUNDARIES: [f32; 15] = [
    -1.8775, -1.4155, -1.0755, -0.7900, -0.5345, -0.2945, -0.1175,
     0.0000,
     0.1175,  0.2945,  0.5345,  0.7900,  1.0755,  1.4155,  1.8775,
];

/// Simple deterministic SplitMix64 pseudo-random number generator for reproducible bit-for-bit rotations.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    #[inline]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }
}

/// In-place normalized Fast Walsh-Hadamard Transform (FWHT) for a slice whose length is a power of two.
#[inline]
pub fn fwht_in_place(buf: &mut [f32]) {
    let n = buf.len();
    if n <= 1 {
        return;
    }
    assert!(n.is_power_of_two(), "FWHT buffer length must be a power of two");

    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let x = buf[j];
                let y = buf[j + h];
                buf[j] = x + y;
                buf[j + h] = x - y;
            }
            i += h * 2;
        }
        h *= 2;
    }

    let inv_sqrt_n = 1.0 / (n as f32).sqrt();
    for val in buf.iter_mut() {
        *val *= inv_sqrt_n;
    }
}

/// Applies a 2-round deterministic orthogonal randomized block-Hadamard transform to `buf`.
///
/// Decorrelates coordinate dimensions so that each coordinate follows the analytic symmetric Beta/Gaussian
/// distribution without requiring any training data.
pub fn rotate_vector_in_place(buf: &mut [f32]) {
    let dim = buf.len();
    if dim < 8 {
        return;
    }

    // Determine largest power-of-two block size <= 64 dividing or fitting dim
    let mut block_size = 64;
    while block_size > 8 && (dim % block_size != 0) {
        block_size /= 2;
    }
    if dim % block_size != 0 {
        block_size = 8;
    }

    let mut rng = SplitMix64::new(DEFAULT_ROTATION_SEED ^ (dim as u64));

    // 2 rounds of permutation + sign-flip + block-Hadamard
    for _round in 0..2 {
        // 1. Deterministic Fisher-Yates coordinate permutation
        for i in (1..dim).rev() {
            let j = (rng.next_u32() as usize) % (i + 1);
            buf.swap(i, j);
        }

        // 2. Deterministic +/- 1 sign-flips
        for chunk in buf.chunks_mut(64) {
            let bits = rng.next_u64();
            for (idx, val) in chunk.iter_mut().enumerate() {
                if ((bits >> (idx % 64)) & 1) == 1 {
                    *val = -*val;
                }
            }
        }

        // 3. Block-wise Fast Walsh-Hadamard Transform
        for block in buf.chunks_exact_mut(block_size) {
            fwht_in_place(block);
        }
    }
}

/// 4-bit TurboQuant Vector Quantizer Engine.
#[derive(Debug, Clone)]
pub struct TurboQuant4Bit {
    dim: usize,
    sigma: f32,
    scaled_centroids: [f32; 16],
    scaled_boundaries: [f32; 15],
}

impl TurboQuant4Bit {
    /// Create a new 4-bit quantizer for vectors of dimension `dim`.
    pub fn new(dim: usize) -> Self {
        assert!(dim >= 8, "TurboQuant requires dimension >= 8");
        let sigma = 1.0 / (dim as f32).sqrt();

        let mut scaled_centroids = [0.0f32; 16];
        for i in 0..16 {
            scaled_centroids[i] = LLOYD_MAX_4BIT_CENTROIDS[i] * sigma;
        }

        let mut scaled_boundaries = [0.0f32; 15];
        for i in 0..15 {
            scaled_boundaries[i] = LLOYD_MAX_4BIT_BOUNDARIES[i] * sigma;
        }

        Self {
            dim,
            sigma,
            scaled_centroids,
            scaled_boundaries,
        }
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    #[inline]
    pub fn sigma(&self) -> f32 {
        self.sigma
    }

    /// Quantize an input float vector into a tuple of `(l2_norm, packed_4bit_bytes)`.
    ///
    /// Packs 2 coordinates per byte (low nibble = even index, high nibble = odd index).
    pub fn quantize(&self, vec: &[f32]) -> (f32, Vec<u8>) {
        assert_eq!(vec.len(), self.dim, "Input vector dimension mismatch");

        // 1. Calculate L2 norm
        let sum_sq: f32 = vec.iter().map(|&x| x * x).sum();
        let norm = sum_sq.sqrt();
        if norm == 0.0 {
            let packed_len = (self.dim + 1) / 2;
            return (0.0, vec![0x77u8; packed_len]); // Index 7 & 8 are near 0.0
        }

        // 2. Normalize and rotate in-place
        let mut rotated = vec.to_vec();
        let inv_norm = 1.0 / norm;
        for val in rotated.iter_mut() {
            *val *= inv_norm;
        }
        rotate_vector_in_place(&mut rotated);

        // 3. Map rotated coordinates to 4-bit centroid indices using boundary lookup
        let packed_len = (self.dim + 1) / 2;
        let mut packed = vec![0u8; packed_len];

        for (i, &coord) in rotated.iter().enumerate() {
            let code = match self.scaled_boundaries.binary_search_by(|b| b.partial_cmp(&coord).unwrap()) {
                Ok(idx) => (idx + 1).min(15) as u8,
                Err(idx) => idx.min(15) as u8,
            };

            let byte_idx = i / 2;
            if i % 2 == 0 {
                packed[byte_idx] |= code & 0x0F;
            } else {
                packed[byte_idx] |= (code & 0x0F) << 4;
            }
        }

        (norm, packed)
    }

    /// Precompute Query Look-Up Table (LUT) of size `dim * 16` for rapid asymmetric inner product search.
    pub fn build_query_lut(&self, query: &[f32]) -> Vec<f32> {
        assert_eq!(query.len(), self.dim, "Query vector dimension mismatch");

        // Rotate query vector with the same orthogonal transform
        let mut rotated_query = query.to_vec();
        rotate_vector_in_place(&mut rotated_query);

        let mut lut = vec![0.0f32; self.dim * 16];
        for (i, &q_val) in rotated_query.iter().enumerate() {
            let row_offset = i * 16;
            for c in 0..16 {
                lut[row_offset + c] = q_val * self.scaled_centroids[c];
            }
        }
        lut
    }

    /// Compute approximate dot product between a precomputed Query LUT and a packed 4-bit vector.
    #[inline]
    pub fn score_query_lut(&self, query_lut: &[f32], target_norm: f32, packed: &[u8]) -> f32 {
        if target_norm == 0.0 {
            return 0.0;
        }

        let mut dot = 0.0f32;
        let mut coord_idx = 0;

        for &byte in packed {
            // Low nibble
            let c0 = (byte & 0x0F) as usize;
            dot += query_lut[coord_idx * 16 + c0];
            coord_idx += 1;
            if coord_idx >= self.dim {
                break;
            }

            // High nibble
            let c1 = ((byte >> 4) & 0x0F) as usize;
            dot += query_lut[coord_idx * 16 + c1];
            coord_idx += 1;
            if coord_idx >= self.dim {
                break;
            }
        }

        dot * target_norm
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fwht_orthogonality_and_energy_conservation() {
        let mut vec = vec![1.0, 2.0, -1.0, 3.0, -2.0, 0.5, -0.5, 1.5];
        let original_norm_sq: f32 = vec.iter().map(|x| x * x).sum();

        fwht_in_place(&mut vec);

        let transformed_norm_sq: f32 = vec.iter().map(|x| x * x).sum();
        assert!(
            (original_norm_sq - transformed_norm_sq).abs() < 1e-4,
            "FWHT must conserve L2 norm"
        );
    }

    #[test]
    fn test_turboquant_4bit_quantize_and_lut_scoring() {
        let dim = 64;
        let quantizer = TurboQuant4Bit::new(dim);

        let mut v1 = vec![0.0f32; dim];
        let mut v2 = vec![0.0f32; dim];
        for i in 0..dim {
            v1[i] = ((i as f32 * 0.13).sin() * 2.0).clamp(-2.0, 2.0);
            v2[i] = ((i as f32 * 0.17).cos() * 1.5).clamp(-2.0, 2.0);
        }

        let (norm1, packed1) = quantizer.quantize(&v1);
        assert_eq!(packed1.len(), dim / 2);
        assert!(norm1 > 0.0);

        let lut = quantizer.build_query_lut(&v2);
        let approx_dot = quantizer.score_query_lut(&lut, norm1, &packed1);

        let true_dot: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        let diff = (approx_dot - true_dot).abs();
        let relative_err = diff / true_dot.abs().max(1e-4);

        assert!(
            relative_err < 0.15,
            "4-bit TurboQuant cosine approximation error must be < 15% (true: {true_dot}, approx: {approx_dot})"
        );
    }
}
