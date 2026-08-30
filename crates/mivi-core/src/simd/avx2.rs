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
