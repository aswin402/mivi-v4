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

/// Accumulate two decoded rows within fixed-width column panels.
///
/// # Safety
/// Caller must ensure AVX2/FMA support, outputs of at least `batch`, weights
/// of at least `cols`, and inputs of at least `cols * batch` elements.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub(super) unsafe fn pair_panel_avx2<const COLUMN_TILE: usize>(
    out0: &mut [f32],
    out1: &mut [f32],
    weights0: &[f32],
    weights1: &[f32],
    transposed_inputs: &[f32],
    batch: usize,
    cols: usize,
) {
    let p0 = out0.as_mut_ptr();
    let p1 = out1.as_mut_ptr();
    let w0 = weights0.as_ptr();
    let w1 = weights1.as_ptr();
    let inputs = transposed_inputs.as_ptr();
    for column_start in (0..cols).step_by(COLUMN_TILE) {
        let column_end = (column_start + COLUMN_TILE).min(cols);
        let mut offset = 0;
        while offset + 32 <= batch {
            let mut a0 = _mm256_loadu_ps(p0.add(offset));
            let mut a1 = _mm256_loadu_ps(p0.add(offset + 8));
            let mut a2 = _mm256_loadu_ps(p0.add(offset + 16));
            let mut a3 = _mm256_loadu_ps(p0.add(offset + 24));
            let mut b0 = _mm256_loadu_ps(p1.add(offset));
            let mut b1 = _mm256_loadu_ps(p1.add(offset + 8));
            let mut b2 = _mm256_loadu_ps(p1.add(offset + 16));
            let mut b3 = _mm256_loadu_ps(p1.add(offset + 24));
            for col in column_start..column_end {
                let s0 = _mm256_set1_ps(*w0.add(col));
                let s1 = _mm256_set1_ps(*w1.add(col));
                let input = inputs.add(col * batch + offset);
                let x0 = _mm256_loadu_ps(input);
                a0 = _mm256_fmadd_ps(s0, x0, a0);
                b0 = _mm256_fmadd_ps(s1, x0, b0);
                let x1 = _mm256_loadu_ps(input.add(8));
                a1 = _mm256_fmadd_ps(s0, x1, a1);
                b1 = _mm256_fmadd_ps(s1, x1, b1);
                let x2 = _mm256_loadu_ps(input.add(16));
                a2 = _mm256_fmadd_ps(s0, x2, a2);
                b2 = _mm256_fmadd_ps(s1, x2, b2);
                let x3 = _mm256_loadu_ps(input.add(24));
                a3 = _mm256_fmadd_ps(s0, x3, a3);
                b3 = _mm256_fmadd_ps(s1, x3, b3);
            }
            _mm256_storeu_ps(p0.add(offset), a0);
            _mm256_storeu_ps(p0.add(offset + 8), a1);
            _mm256_storeu_ps(p0.add(offset + 16), a2);
            _mm256_storeu_ps(p0.add(offset + 24), a3);
            _mm256_storeu_ps(p1.add(offset), b0);
            _mm256_storeu_ps(p1.add(offset + 8), b1);
            _mm256_storeu_ps(p1.add(offset + 16), b2);
            _mm256_storeu_ps(p1.add(offset + 24), b3);
            offset += 32;
        }
        while offset + 8 <= batch {
            let mut a = _mm256_loadu_ps(p0.add(offset));
            let mut b = _mm256_loadu_ps(p1.add(offset));
            for col in column_start..column_end {
                let x = _mm256_loadu_ps(inputs.add(col * batch + offset));
                a = _mm256_fmadd_ps(_mm256_set1_ps(*w0.add(col)), x, a);
                b = _mm256_fmadd_ps(_mm256_set1_ps(*w1.add(col)), x, b);
            }
            _mm256_storeu_ps(p0.add(offset), a);
            _mm256_storeu_ps(p1.add(offset), b);
            offset += 8;
        }
        for index in offset..batch {
            let mut a = *p0.add(index);
            let mut b = *p1.add(index);
            for col in column_start..column_end {
                let x = *inputs.add(col * batch + index);
                a += *w0.add(col) * x;
                b += *w1.add(col) * x;
            }
            *p0.add(index) = a;
            *p1.add(index) = b;
        }
    }
}

/// Accumulate one decoded weight row against transposed batch inputs.
///
/// # Safety
/// Caller must ensure AVX2/FMA support and valid slice lengths.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn matmul_accumulate_transposed_avx2(
    out: &mut [f32],
    weights: &[f32],
    transposed_inputs: &[f32],
    batch: usize,
    cols: usize,
) {
    let out_ptr = out.as_mut_ptr();
    let weights_ptr = weights.as_ptr();
    let inputs_ptr = transposed_inputs.as_ptr();
    if batch < 32 {
        let vector_end = (batch / 8) * 8;
        for col in 0..cols {
            let scale = _mm256_set1_ps(*weights_ptr.add(col));
            let input = inputs_ptr.add(col * batch);
            for offset in (0..vector_end).step_by(8) {
                let accumulator = _mm256_loadu_ps(out_ptr.add(offset));
                let value = _mm256_loadu_ps(input.add(offset));
                _mm256_storeu_ps(
                    out_ptr.add(offset),
                    _mm256_fmadd_ps(scale, value, accumulator),
                );
            }
            for b in vector_end..batch {
                *out_ptr.add(b) += *weights_ptr.add(col) * *input.add(b);
            }
        }
        return;
    }
    let mut offset = 0;
    while offset + 64 <= batch {
        let mut a0 = _mm256_loadu_ps(out_ptr.add(offset));
        let mut a1 = _mm256_loadu_ps(out_ptr.add(offset + 8));
        let mut a2 = _mm256_loadu_ps(out_ptr.add(offset + 16));
        let mut a3 = _mm256_loadu_ps(out_ptr.add(offset + 24));
        let mut a4 = _mm256_loadu_ps(out_ptr.add(offset + 32));
        let mut a5 = _mm256_loadu_ps(out_ptr.add(offset + 40));
        let mut a6 = _mm256_loadu_ps(out_ptr.add(offset + 48));
        let mut a7 = _mm256_loadu_ps(out_ptr.add(offset + 56));
        for col in 0..cols {
            let scale = _mm256_set1_ps(*weights_ptr.add(col));
            let input = inputs_ptr.add(col * batch + offset);
            a0 = _mm256_fmadd_ps(scale, _mm256_loadu_ps(input), a0);
            a1 = _mm256_fmadd_ps(scale, _mm256_loadu_ps(input.add(8)), a1);
            a2 = _mm256_fmadd_ps(scale, _mm256_loadu_ps(input.add(16)), a2);
            a3 = _mm256_fmadd_ps(scale, _mm256_loadu_ps(input.add(24)), a3);
            a4 = _mm256_fmadd_ps(scale, _mm256_loadu_ps(input.add(32)), a4);
            a5 = _mm256_fmadd_ps(scale, _mm256_loadu_ps(input.add(40)), a5);
            a6 = _mm256_fmadd_ps(scale, _mm256_loadu_ps(input.add(48)), a6);
            a7 = _mm256_fmadd_ps(scale, _mm256_loadu_ps(input.add(56)), a7);
        }
        _mm256_storeu_ps(out_ptr.add(offset), a0);
        _mm256_storeu_ps(out_ptr.add(offset + 8), a1);
        _mm256_storeu_ps(out_ptr.add(offset + 16), a2);
        _mm256_storeu_ps(out_ptr.add(offset + 24), a3);
        _mm256_storeu_ps(out_ptr.add(offset + 32), a4);
        _mm256_storeu_ps(out_ptr.add(offset + 40), a5);
        _mm256_storeu_ps(out_ptr.add(offset + 48), a6);
        _mm256_storeu_ps(out_ptr.add(offset + 56), a7);
        offset += 64;
    }
    while offset + 32 <= batch {
        let mut a0 = _mm256_loadu_ps(out_ptr.add(offset));
        let mut a1 = _mm256_loadu_ps(out_ptr.add(offset + 8));
        let mut a2 = _mm256_loadu_ps(out_ptr.add(offset + 16));
        let mut a3 = _mm256_loadu_ps(out_ptr.add(offset + 24));
        for col in 0..cols {
            let scale = _mm256_set1_ps(*weights_ptr.add(col));
            let input = inputs_ptr.add(col * batch + offset);
            a0 = _mm256_fmadd_ps(scale, _mm256_loadu_ps(input), a0);
            a1 = _mm256_fmadd_ps(scale, _mm256_loadu_ps(input.add(8)), a1);
            a2 = _mm256_fmadd_ps(scale, _mm256_loadu_ps(input.add(16)), a2);
            a3 = _mm256_fmadd_ps(scale, _mm256_loadu_ps(input.add(24)), a3);
        }
        _mm256_storeu_ps(out_ptr.add(offset), a0);
        _mm256_storeu_ps(out_ptr.add(offset + 8), a1);
        _mm256_storeu_ps(out_ptr.add(offset + 16), a2);
        _mm256_storeu_ps(out_ptr.add(offset + 24), a3);
        offset += 32;
    }
    while offset + 8 <= batch {
        let mut accumulator = _mm256_loadu_ps(out_ptr.add(offset));
        for col in 0..cols {
            let scale = _mm256_set1_ps(*weights_ptr.add(col));
            let input = _mm256_loadu_ps(inputs_ptr.add(col * batch + offset));
            accumulator = _mm256_fmadd_ps(scale, input, accumulator);
        }
        _mm256_storeu_ps(out_ptr.add(offset), accumulator);
        offset += 8;
    }
    for batch_idx in offset..batch {
        let mut accumulator = *out_ptr.add(batch_idx);
        for col in 0..cols {
            accumulator += *weights_ptr.add(col) * *inputs_ptr.add(col * batch + batch_idx);
        }
        *out_ptr.add(batch_idx) = accumulator;
    }
}
