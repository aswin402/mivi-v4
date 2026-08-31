//! SIMD acceleration dispatcher.

pub mod norm;
pub mod scalar;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(target_arch = "x86_64")]
pub static HAS_AVX2_FMA: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
});

type MatvecFn = fn(&mut [f32], &[f32], &[f32], usize, usize);

static MATVEC_IMPL: std::sync::LazyLock<MatvecFn> = std::sync::LazyLock::new(|| {
    #[cfg(target_arch = "x86_64")]
    {
        if *HAS_AVX2_FMA {
            return |out, w, x, n, d| unsafe { avx2::matvec_f32_avx2(out, w, x, n, d) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return |out, w, x, n, d| unsafe { neon::matvec_f32_neon(out, w, x, n, d) };
    }

    #[allow(unreachable_code)]
    scalar::matvec_f32_scalar
});

#[cfg(target_arch = "x86_64")]
pub use avx2::hsum256_ps;

pub use norm::{rms_norm_in_place_simd, rms_norm_simd};

/// Vector dot product: sum(a[i] * b[i]) with SIMD acceleration.
#[inline]
pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if *HAS_AVX2_FMA {
            return unsafe { avx2::dot_product_avx2(a, b) };
        }
    }
    crate::math::dot_product_scalar(a, b)
}

/// Vector addition: out[i] += src[i] with SIMD acceleration.
#[inline]
pub fn vec_add_simd(out: &mut [f32], src: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if *HAS_AVX2_FMA {
            unsafe {
                avx2::vec_add_avx2(out, src);
                return;
            }
        }
    }
    crate::math::vec_add_scalar(out, src);
}

/// Vector fused multiply-add: out[i] += scale * src[i] with SIMD acceleration.
#[inline]
pub fn vec_fmadd_simd(out: &mut [f32], scale: f32, src: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if *HAS_AVX2_FMA {
            unsafe {
                avx2::vec_fmadd_avx2(out, scale, src);
                return;
            }
        }
    }
    crate::math::vec_fmadd_scalar(out, scale, src);
}

/// Matrix-vector multiplication for dense f32 matrices: out[n] = w[n, d] * x[d]
#[inline]
pub fn matvec_f32(out: &mut [f32], w: &[f32], x: &[f32], n: usize, d: usize) {
    assert!(
        out.len() >= n,
        "Output buffer too small: {} < {}",
        out.len(),
        n
    );
    assert!(
        w.len() >= n * d,
        "Weight buffer too small: {} < {}",
        w.len(),
        n * d
    );
    assert!(x.len() >= d, "Input vector too small: {} < {}", x.len(), d);

    (*MATVEC_IMPL)(out, w, x, n, d);
}
