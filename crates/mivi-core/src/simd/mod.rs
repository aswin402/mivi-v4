//! SIMD acceleration dispatcher.

pub mod norm;
pub mod scalar;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(target_arch = "x86_64")]
static HAS_AVX2_FMA: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
});

pub use norm::rms_norm_simd;

/// Matrix-vector multiplication for dense f32 matrices: out[n] = w[n, d] * x[d]
#[inline]
pub fn matvec_f32(out: &mut [f32], w: &[f32], x: &[f32], n: usize, d: usize) {
    assert!(out.len() >= n, "Output buffer too small: {} < {}", out.len(), n);
    assert!(w.len() >= n * d, "Weight buffer too small: {} < {}", w.len(), n * d);
    assert!(x.len() >= d, "Input vector too small: {} < {}", x.len(), d);

    #[cfg(target_arch = "x86_64")]
    {
        if *HAS_AVX2_FMA {
            unsafe {
                return avx2::matvec_f32_avx2(out, w, x, n, d);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            return neon::matvec_f32_neon(out, w, x, n, d);
        }
    }

    scalar::matvec_f32_scalar(out, w, x, n, d);
}
