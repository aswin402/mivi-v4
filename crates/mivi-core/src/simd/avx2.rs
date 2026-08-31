//! x86_64 AVX2 + FMA optimized kernels.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
/// Horizontal sum of 8 floats in an AVX2 `__m256` vector register.
///
/// # Safety
/// Caller must ensure AVX2 instructions are supported on the host CPU.
#[inline(always)]
pub unsafe fn hsum256_ps(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_shuffle_ps(sum64, sum64, 1);
    let sum32 = _mm_add_ss(sum64, shuf2);
    _mm_cvtss_f32(sum32)
}

/// Matrix-vector multiplication for dense F32 weights with AVX2 + FMA.
///
/// # Safety
/// Caller must ensure that the target CPU supports `avx2` and `fma` features,
/// and that slices `out`, `w`, and `x` have valid bounds ($out.len() \ge n$, $w.len() \ge n \times d$, $x.len() \ge d$).
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn matvec_f32_avx2(out: &mut [f32], w: &[f32], x: &[f32], n: usize, d: usize) {
    let chunks = d / 8;
    let remainder = d % 8;

    for (i, out_val) in out.iter_mut().enumerate().take(n) {
        let row_ptr = w.as_ptr().add(i * d);
        let x_ptr = x.as_ptr();

        let mut acc = _mm256_setzero_ps();

        for c in 0..chunks {
            let offset = c * 8;
            let wv = _mm256_loadu_ps(row_ptr.add(offset));
            let xv = _mm256_loadu_ps(x_ptr.add(offset));
            acc = _mm256_fmadd_ps(wv, xv, acc);
        }

        let mut sum = hsum256_ps(acc);

        // Process scalar remainder
        let rem_start = chunks * 8;
        for r in 0..remainder {
            sum += *row_ptr.add(rem_start + r) * *x_ptr.add(rem_start + r);
        }

        *out_val = sum;
    }
}

/// Vector dot product with AVX2 + FMA.
///
/// # Safety
/// Caller must ensure that the target CPU supports `avx2` and `fma` features and slices are at least `len` long.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let chunks = len / 8;
    let remainder = len % 8;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let mut acc = _mm256_setzero_ps();

    for c in 0..chunks {
        let offset = c * 8;
        let av = _mm256_loadu_ps(a_ptr.add(offset));
        let bv = _mm256_loadu_ps(b_ptr.add(offset));
        acc = _mm256_fmadd_ps(av, bv, acc);
    }

    let mut sum = hsum256_ps(acc);
    let rem_start = chunks * 8;
    for r in 0..remainder {
        sum += *a_ptr.add(rem_start + r) * *b_ptr.add(rem_start + r);
    }
    sum
}

/// Vector in-place addition `out[i] += src[i]` with AVX2.
///
/// # Safety
/// Caller must ensure that the target CPU supports `avx2` and slices are at least `len` long.
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn vec_add_avx2(out: &mut [f32], src: &[f32]) {
    let len = out.len().min(src.len());
    let chunks = len / 8;
    let remainder = len % 8;

    let out_ptr = out.as_mut_ptr();
    let src_ptr = src.as_ptr();

    for c in 0..chunks {
        let offset = c * 8;
        let ov = _mm256_loadu_ps(out_ptr.add(offset));
        let sv = _mm256_loadu_ps(src_ptr.add(offset));
        let res = _mm256_add_ps(ov, sv);
        _mm256_storeu_ps(out_ptr.add(offset), res);
    }

    let rem_start = chunks * 8;
    for r in 0..remainder {
        *out_ptr.add(rem_start + r) += *src_ptr.add(rem_start + r);
    }
}

/// Vector in-place fused multiply-add `out[i] += scale * src[i]` with AVX2 + FMA.
///
/// # Safety
/// Caller must ensure that the target CPU supports `avx2` and `fma` and slices are at least `len` long.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn vec_fmadd_avx2(out: &mut [f32], scale: f32, src: &[f32]) {
    let len = out.len().min(src.len());
    let chunks = len / 8;
    let remainder = len % 8;

    let scale_v = _mm256_set1_ps(scale);
    let out_ptr = out.as_mut_ptr();
    let src_ptr = src.as_ptr();

    for c in 0..chunks {
        let offset = c * 8;
        let ov = _mm256_loadu_ps(out_ptr.add(offset));
        let sv = _mm256_loadu_ps(src_ptr.add(offset));
        let res = _mm256_fmadd_ps(scale_v, sv, ov);
        _mm256_storeu_ps(out_ptr.add(offset), res);
    }

    let rem_start = chunks * 8;
    for r in 0..remainder {
        *out_ptr.add(rem_start + r) += scale * *src_ptr.add(rem_start + r);
    }
}
