//! Precomputed Rotary Position Embedding (RoPE) frequency cache.
//! Eliminates per-token `powf`, `sin`, and `cos` runtime computation.

#[derive(Debug, Clone)]
pub struct RopeCache {
    pub head_dim: usize,
    pub half_dim: usize,
    pub max_seq_len: usize,
    pub cos_table: Box<[f32]>,
    pub sin_table: Box<[f32]>,
}

impl RopeCache {
    /// Precomputes the full sin/cos lookup table for sequence lengths up to `max_seq_len`.
    pub fn new(head_dim: usize, max_seq_len: usize, rope_base: f32) -> Self {
        assert_eq!(head_dim % 2, 0, "head_dim must be even for RoPE");
        let half_dim = head_dim / 2;

        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / rope_base.powf((2 * i) as f32 / head_dim as f32))
            .collect();

        let total_elements = max_seq_len * half_dim;
        let mut cos_table = vec![0.0f32; total_elements];
        let mut sin_table = vec![0.0f32; total_elements];

        for pos in 0..max_seq_len {
            let offset = pos * half_dim;
            for i in 0..half_dim {
                let angle = pos as f32 * inv_freq[i];
                let (sin, cos) = angle.sin_cos();
                cos_table[offset + i] = cos;
                sin_table[offset + i] = sin;
            }
        }

        Self {
            head_dim,
            half_dim,
            max_seq_len,
            cos_table: cos_table.into_boxed_slice(),
            sin_table: sin_table.into_boxed_slice(),
        }
    }

    /// Apply RoPE to query and key vectors in-place using precomputed lookup tables.
    #[inline]
    pub fn apply(
        &self,
        q: &mut [f32],
        k: &mut [f32],
        pos: usize,
        n_q_heads: usize,
        n_kv_heads: usize,
    ) {
        assert!(pos < self.max_seq_len, "Position exceeds precomputed RoPE cache size");
        let half_dim = self.half_dim;
        let head_dim = self.head_dim;
        let offset = pos * half_dim;
        let cos = &self.cos_table[offset..offset + half_dim];
        let sin = &self.sin_table[offset..offset + half_dim];

        // Rotate Query heads
        for h in 0..n_q_heads {
            let head = &mut q[h * head_dim..(h + 1) * head_dim];
            for i in 0..half_dim {
                let q0 = head[2 * i];
                let q1 = head[2 * i + 1];
                let c = cos[i];
                let s = sin[i];
                head[2 * i] = q0 * c - q1 * s;
                head[2 * i + 1] = q0 * s + q1 * c;
            }
        }

        // Rotate Key heads
        for h in 0..n_kv_heads {
            let head = &mut k[h * head_dim..(h + 1) * head_dim];
            for i in 0..half_dim {
                let k0 = head[2 * i];
                let k1 = head[2 * i + 1];
                let c = cos[i];
                let s = sin[i];
                head[2 * i] = k0 * c - k1 * s;
                head[2 * i + 1] = k0 * s + k1 * c;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::apply_rope;

    #[test]
    fn test_rope_cache_matches_math() {
        let head_dim = 64;
        let max_seq_len = 16;
        let rope_base = 10000.0;

        let cache = RopeCache::new(head_dim, max_seq_len, rope_base);

        let mut q1 = vec![1.0f32; head_dim];
        let mut k1 = vec![1.0f32; head_dim];
        let mut q2 = vec![1.0f32; head_dim];
        let mut k2 = vec![1.0f32; head_dim];

        let pos = 5;
        apply_rope(&mut q1, &mut k1, head_dim, pos, rope_base);
        cache.apply(&mut q2, &mut k2, pos, 1, 1);

        for i in 0..head_dim {
            assert!((q1[i] - q2[i]).abs() < 1e-5);
            assert!((k1[i] - k2[i]).abs() < 1e-5);
        }
    }
}
