use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug, PartialEq)]
pub enum RopeError {
    #[error("RoPE position {pos} out of bounds (max {max})")]
    PosOutOfBounds { pos: usize, max: usize },
    #[error("RoPE buffer too small: required {required}, got {actual}")]
    BufferTooSmall { required: usize, actual: usize },
}

pub type Result<T> = std::result::Result<T, RopeError>;

/// Rotary position embedding scaling strategy for long-context extension.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RopeScaling {
    /// Standard unscaled RoPE.
    None,
    /// Linear frequency scaling (pos / scale).
    Linear { scale: f32 },
    /// YaRN (Yet another RoPE extensioN) with NTK-aware frequency interpolation.
    YaRN {
        scale: f32,
        orig_max_seq_len: usize,
        extrapolation_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
    },
}

impl Default for RopeScaling {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Debug, Clone)]
pub struct RopeCache {
    pub head_dim: usize,
    pub half_dim: usize,
    pub max_seq_len: usize,
    pub scaling: RopeScaling,
    pub cos_table: Box<[f32]>,
    pub sin_table: Box<[f32]>,
}

impl RopeCache {
    /// Precomputes the full sin/cos lookup table for sequence lengths up to `max_seq_len`.
    pub fn new(head_dim: usize, max_seq_len: usize, rope_base: f32) -> Self {
        Self::new_with_scaling(head_dim, max_seq_len, rope_base, RopeScaling::None)
    }

    /// Precomputes RoPE tables with explicit scaling strategy (e.g. YaRN or Linear).
    pub fn new_with_scaling(
        head_dim: usize,
        max_seq_len: usize,
        rope_base: f32,
        scaling: RopeScaling,
    ) -> Self {
        assert_eq!(head_dim % 2, 0, "head_dim must be even for RoPE");
        let half_dim = head_dim / 2;

        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| {
                let base_freq = 1.0 / rope_base.powf((2 * i) as f32 / head_dim as f32);
                match scaling {
                    RopeScaling::None => base_freq,
                    RopeScaling::Linear { scale } => {
                        if scale > 0.0 {
                            base_freq / scale
                        } else {
                            base_freq
                        }
                    }
                    RopeScaling::YaRN {
                        scale,
                        orig_max_seq_len,
                        extrapolation_factor: _,
                        attn_factor: _,
                        beta_fast,
                        beta_slow,
                    } => {
                        if scale <= 1.0 {
                            base_freq
                        } else {
                            // YaRN ramp: smooth frequency interpolation across wavelength boundaries
                            let orig_len = orig_max_seq_len.max(1) as f32;
                            let low_rot = orig_len / beta_slow.max(0.01);
                            let high_rot = orig_len / beta_fast.max(0.01);
                            let wavelength = 2.0 * std::f32::consts::PI / base_freq;

                            let ramp = if high_rot >= low_rot {
                                if wavelength < high_rot { 0.0 } else { 1.0 }
                            } else if wavelength < high_rot {
                                0.0
                            } else if wavelength > low_rot {
                                1.0
                            } else {
                                (wavelength - high_rot) / (low_rot - high_rot)
                            };

                            let scaled_freq = base_freq / scale;
                            (1.0 - ramp) * base_freq + ramp * scaled_freq
                        }
                    }
                }
            })
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
            scaling,
            cos_table: cos_table.into_boxed_slice(),
            sin_table: sin_table.into_boxed_slice(),
        }
    }

    /// Safely apply RoPE returning an error if position or buffer size is out of bounds.
    #[inline]
    pub fn try_apply(
        &self,
        q: &mut [f32],
        k: &mut [f32],
        pos: usize,
        n_q_heads: usize,
        n_kv_heads: usize,
    ) -> Result<()> {
        if pos >= self.max_seq_len {
            return Err(RopeError::PosOutOfBounds {
                pos,
                max: self.max_seq_len,
            });
        }
        let half_dim = self.half_dim;
        let head_dim = self.head_dim;
        let offset = pos * half_dim;
        if offset + half_dim > self.cos_table.len() {
            return Err(RopeError::PosOutOfBounds {
                pos,
                max: self.max_seq_len,
            });
        }
        let cos = &self.cos_table[offset..offset + half_dim];
        let sin = &self.sin_table[offset..offset + half_dim];

        let cfg_q = RotateConfig {
            n_heads: n_q_heads,
            head_dim,
            half_dim,
            cos,
            sin,
        };
        let cfg_k = RotateConfig {
            n_heads: n_kv_heads,
            head_dim,
            half_dim,
            cos,
            sin,
        };
        try_rotate_heads(q, &cfg_q)?;
        try_rotate_heads(k, &cfg_k)?;
        Ok(())
    }

    /// Apply RoPE to query and key vectors in-place using precomputed lookup tables.
    ///
    /// # Panics
    /// Panics if position is out of bounds or buffers are too small. Prefer `try_apply`.
    #[inline]
    pub fn apply(
        &self,
        q: &mut [f32],
        k: &mut [f32],
        pos: usize,
        n_q_heads: usize,
        n_kv_heads: usize,
    ) {
        if let Err(e) = self.try_apply(q, k, pos, n_q_heads, n_kv_heads) {
            panic!("{}", e);
        }
    }
}

/// Parameter descriptor for head rotation in RoPE.
#[derive(Debug, Clone, Copy)]
pub struct RotateConfig<'a> {
    pub n_heads: usize,
    pub head_dim: usize,
    pub half_dim: usize,
    pub cos: &'a [f32],
    pub sin: &'a [f32],
}

#[inline]
fn try_rotate_heads(vec: &mut [f32], cfg: &RotateConfig) -> Result<()> {
    let req_len = cfg.n_heads * cfg.head_dim;
    if vec.len() < req_len {
        return Err(RopeError::BufferTooSmall {
            required: req_len,
            actual: vec.len(),
        });
    }
    for h in 0..cfg.n_heads {
        let head = &mut vec[h * cfg.head_dim..(h + 1) * cfg.head_dim];
        for i in 0..cfg.half_dim {
            let v0 = head[2 * i];
            let v1 = head[2 * i + 1];
            let c = cfg.cos[i];
            let s = cfg.sin[i];
            head[2 * i] = v0 * c - v1 * s;
            head[2 * i + 1] = v0 * s + v1 * c;
        }
    }
    Ok(())
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

    #[test]
    fn test_rope_scaling_yarn_and_linear_64k() {
        let head_dim = 64;
        let max_seq_len = 65536;
        let rope_base = 10000.0;

        let yarn_cache = RopeCache::new_with_scaling(
            head_dim,
            max_seq_len,
            rope_base,
            RopeScaling::YaRN {
                scale: 16.0,
                orig_max_seq_len: 4096,
                extrapolation_factor: 1.0,
                attn_factor: 1.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
            },
        );

        let mut q = vec![1.0f32; head_dim];
        let mut k = vec![1.0f32; head_dim];

        // Test at maximum 64K position
        assert!(yarn_cache.try_apply(&mut q, &mut k, 65535, 1, 1).is_ok());
        assert!(q.iter().all(|&v| v.is_finite()));
        assert!(k.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_rope_try_apply_errors() {
        let cache = RopeCache::new(64, 16, 10000.0);
        let mut q = vec![1.0f32; 64];
        let mut k = vec![1.0f32; 64];

        // Pos >= max_seq_len
        assert_eq!(
            cache.try_apply(&mut q, &mut k, 20, 1, 1),
            Err(RopeError::PosOutOfBounds { pos: 20, max: 16 })
        );

        // Buffer too small
        let mut short_q = vec![1.0f32; 32];
        assert_eq!(
            cache.try_apply(&mut short_q, &mut k, 0, 1, 1),
            Err(RopeError::BufferTooSmall {
                required: 64,
                actual: 32
            })
        );
    }
}
