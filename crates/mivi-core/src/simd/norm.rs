//! SIMD-accelerated LayerNorm & RMSNorm kernels (AVX2 / ARM NEON / Fallback).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
pub fn compute_rms_scale(x: &[f32], eps: f32) -> f32 {
    let len = x.len();
    if len == 0 {
        return 0.0;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if *super::HAS_AVX2_FMA {
            unsafe {
                let mut sum_sq_vec = _mm256_setzero_ps();
                let chunks = len / 8;
                for c in 0..chunks {
                    let xv = _mm256_loadu_ps(x.as_ptr().add(c * 8));
                    sum_sq_vec = _mm256_fmadd_ps(xv, xv, sum_sq_vec);
                }
                let mut sum_sq = super::avx2::hsum256_ps(sum_sq_vec);
                for &val in &x[chunks * 8..] {
                    sum_sq += val * val;
                }
                return 1.0 / (sum_sq / (len as f32) + eps).sqrt();
            }
        }
    }

    let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
    1.0 / (sum_sq / (len as f32) + eps).sqrt()
}

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

    let scale = compute_rms_scale(x, eps);

    #[cfg(target_arch = "x86_64")]
    {
        if *super::HAS_AVX2_FMA {
            unsafe {
                let chunks = len / 8;
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

    for i in 0..len {
        out[i] = x[i] * scale * weight[i];
    }
}

/// In-place high performance SIMD Root Mean Square Normalization (RMSNorm).
///
/// Computes: `x[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]`
#[inline]
pub fn rms_norm_in_place_simd(x: &mut [f32], weight: &[f32], eps: f32) {
    let len = x.len();
    if len == 0 {
        return;
    }
    assert_eq!(
        weight.len(),
        len,
        "rms_norm_in_place_simd: weight length mismatch"
    );

    let scale = compute_rms_scale(x, eps);

    #[cfg(target_arch = "x86_64")]
    {
        if *super::HAS_AVX2_FMA {
            unsafe {
                let chunks = len / 8;
                let scale_vec = _mm256_set1_ps(scale);
                for c in 0..chunks {
                    let offset = c * 8;
                    let x_ptr = x.as_mut_ptr().add(offset);
                    let xv = _mm256_loadu_ps(x_ptr);
                    let wv = _mm256_loadu_ps(weight.as_ptr().add(offset));
                    let res = _mm256_mul_ps(_mm256_mul_ps(xv, scale_vec), wv);
                    _mm256_storeu_ps(x_ptr, res);
                }
                for i in (chunks * 8)..len {
                    x[i] = x[i] * scale * weight[i];
                }
                return;
            }
        }
    }

    for i in 0..len {
        x[i] = x[i] * scale * weight[i];
    }
}
