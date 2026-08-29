//! Vectorized SIMD math kernels (RMSNorm, Softmax, Dot Product, SiLU) with AVX2 & NEON acceleration.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
static HAS_AVX2_FMA: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
});

/// Vectorized RMS Normalization: out = x * weight / sqrt(mean(x^2) + eps)
#[inline]
pub fn rms_norm_simd(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    assert!(out.len() >= x.len(), "Output buffer too small for norm: {} < {}", out.len(), x.len());
    assert!(weight.len() >= x.len(), "Weight vector too small for norm: {} < {}", weight.len(), x.len());

    let len = x.len();

    #[cfg(target_arch = "x86_64")]
    {
        if *HAS_AVX2_FMA {
            unsafe {
                let mut sum_sq_vec = _mm256_setzero_ps();
                let chunks = len / 8;

                for c in 0..chunks {
                    let xv = _mm256_loadu_ps(x.as_ptr().add(c * 8));
                    sum_sq_vec = _mm256_fmadd_ps(xv, xv, sum_sq_vec);
                }

                // Horizontal sum
                let lo = _mm256_castps256_ps128(sum_sq_vec);
                let hi = _mm256_extractf128_ps(sum_sq_vec, 1);
                let sum128 = _mm_add_ps(lo, hi);
                let shuf = _mm_movehl_ps(sum128, sum128);
                let sum64 = _mm_add_ps(sum128, shuf);
                let shuf2 = _mm_shuffle_ps(sum64, sum64, 1);
                let sum32 = _mm_add_ss(sum64, shuf2);
                let mut sum_sq = _mm_cvtss_f32(sum32);

                for i in (chunks * 8)..len {
                    sum_sq += x[i] * x[i];
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
