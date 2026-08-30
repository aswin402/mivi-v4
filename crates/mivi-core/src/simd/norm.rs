//! SIMD-accelerated LayerNorm & RMSNorm kernels (AVX2 / ARM NEON / Fallback).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// High performance SIMD Root Mean Square Normalization (RMSNorm).
///
/// Computes: `out[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]`
pub fn rms_norm_simd(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    let len = x.len();
    if len == 0 {
        return;
    }
    assert_eq!(out.len(), len, "rms_norm_simd: out length mismatch");
    assert_eq!(weight.len(), len, "rms_norm_simd: weight length mismatch");

    #[cfg(target_arch = "x86_64")]
    {
        if *super::HAS_AVX2_FMA {
            // SAFETY: Verified HAS_AVX2_FMA feature detection at runtime and checked buffer lengths above.
            unsafe {
                let mut sum_sq_vec = _mm256_setzero_ps();
                let chunks = len / 8;

                for c in 0..chunks {
                    let xv = _mm256_loadu_ps(x.as_ptr().add(c * 8));
                    sum_sq_vec = _mm256_fmadd_ps(xv, xv, sum_sq_vec);
                }

                // Horizontal sum using shared avx2 helper
                let mut sum_sq = super::avx2::hsum256_ps(sum_sq_vec);

                for &val in x.iter().take(len).skip(chunks * 8) {
                    sum_sq += val * val;
                }

                let scale = 1.0 / (sum_sq / (len as f32) + eps).sqrt();
                let scale_vec = _mm256_set1_ps(scale);

                for c in 0..chunks {
                    let offset = c * 8;
                    let xv = _mm256_loadu_ps(x.as_ptr().add(offset));
                    let wv = _mm256_loadu_ps(weight.as_ptr().add(offset));
                    let res = _mm256_mul_ps(_mm256_mul_ps(xv, scale_vec), wv);
                    _mm256_storeu_ps(out.as_mut_ptr().add(offset), res);
                }

                for i in (chunks * 8)..len {
                    out[i] = x[i] * scale * weight[i];
                }
                return;
            }
        }
    }

    // Scalar fallback
    let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
    let scale = 1.0 / (sum_sq / (len as f32) + eps).sqrt();
    for i in 0..len {
        out[i] = x[i] * scale * weight[i];
    }
}
