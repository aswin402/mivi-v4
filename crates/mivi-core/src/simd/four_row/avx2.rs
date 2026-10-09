//! Private register-blocked implementation; called only through checked dispatch.

use std::arch::x86_64::*;

/// # Safety
/// AVX2/FMA must be supported; slices must cover 4*batch outputs, 4*cols
/// decoded weights and cols*batch inputs, with nonoverflowing products.
#[target_feature(enable = "avx2", enable = "fma")]
pub(super) unsafe fn accumulate(
    out: &mut [f32],
    weights: &[f32],
    inputs: &[f32],
    batch: usize,
    cols: usize,
) {
    // SAFETY: the sole checked caller establishes the ISA and full slice bounds.
    // Row offsets stay within four disjoint rows. Vector loads/stores require
    // offset+16 (or +8) <= batch; column indices stay below cols. Unaligned
    // intrinsics permit arbitrary slice alignment. No pointer escapes this call.
    unsafe {
        let p0 = out.as_mut_ptr();
        let p1 = p0.add(batch);
        let p2 = p1.add(batch);
        let p3 = p2.add(batch);
        let w0 = weights.as_ptr();
        let w1 = w0.add(cols);
        let w2 = w1.add(cols);
        let w3 = w2.add(cols);
        let x = inputs.as_ptr();
        for start in (0..cols).step_by(128) {
            let end = start.saturating_add(128).min(cols);
            let mut offset = 0;
            while offset + 16 <= batch {
                let mut a0 = _mm256_loadu_ps(p0.add(offset));
                let mut a1 = _mm256_loadu_ps(p0.add(offset + 8));
                let mut b0 = _mm256_loadu_ps(p1.add(offset));
                let mut b1 = _mm256_loadu_ps(p1.add(offset + 8));
                let mut c0 = _mm256_loadu_ps(p2.add(offset));
                let mut c1 = _mm256_loadu_ps(p2.add(offset + 8));
                let mut d0 = _mm256_loadu_ps(p3.add(offset));
                let mut d1 = _mm256_loadu_ps(p3.add(offset + 8));
                for col in start..end {
                    let s0 = _mm256_set1_ps(*w0.add(col));
                    let s1 = _mm256_set1_ps(*w1.add(col));
                    let s2 = _mm256_set1_ps(*w2.add(col));
                    let s3 = _mm256_set1_ps(*w3.add(col));
                    let input = x.add(col * batch + offset);
                    let x0 = _mm256_loadu_ps(input);
                    let x1 = _mm256_loadu_ps(input.add(8));
                    a0 = _mm256_fmadd_ps(s0, x0, a0);
                    a1 = _mm256_fmadd_ps(s0, x1, a1);
                    b0 = _mm256_fmadd_ps(s1, x0, b0);
                    b1 = _mm256_fmadd_ps(s1, x1, b1);
                    c0 = _mm256_fmadd_ps(s2, x0, c0);
                    c1 = _mm256_fmadd_ps(s2, x1, c1);
                    d0 = _mm256_fmadd_ps(s3, x0, d0);
                    d1 = _mm256_fmadd_ps(s3, x1, d1);
                }
                _mm256_storeu_ps(p0.add(offset), a0);
                _mm256_storeu_ps(p0.add(offset + 8), a1);
                _mm256_storeu_ps(p1.add(offset), b0);
                _mm256_storeu_ps(p1.add(offset + 8), b1);
                _mm256_storeu_ps(p2.add(offset), c0);
                _mm256_storeu_ps(p2.add(offset + 8), c1);
                _mm256_storeu_ps(p3.add(offset), d0);
                _mm256_storeu_ps(p3.add(offset + 8), d1);
                offset += 16;
            }
            while offset + 8 <= batch {
                let mut a = _mm256_loadu_ps(p0.add(offset));
                let mut b = _mm256_loadu_ps(p1.add(offset));
                let mut c = _mm256_loadu_ps(p2.add(offset));
                let mut d = _mm256_loadu_ps(p3.add(offset));
                for col in start..end {
                    let value = _mm256_loadu_ps(x.add(col * batch + offset));
                    a = _mm256_fmadd_ps(_mm256_set1_ps(*w0.add(col)), value, a);
                    b = _mm256_fmadd_ps(_mm256_set1_ps(*w1.add(col)), value, b);
                    c = _mm256_fmadd_ps(_mm256_set1_ps(*w2.add(col)), value, c);
                    d = _mm256_fmadd_ps(_mm256_set1_ps(*w3.add(col)), value, d);
                }
                _mm256_storeu_ps(p0.add(offset), a);
                _mm256_storeu_ps(p1.add(offset), b);
                _mm256_storeu_ps(p2.add(offset), c);
                _mm256_storeu_ps(p3.add(offset), d);
                offset += 8;
            }
            for index in offset..batch {
                let mut a = *p0.add(index);
                let mut b = *p1.add(index);
                let mut c = *p2.add(index);
                let mut d = *p3.add(index);
                for col in start..end {
                    let value = *x.add(col * batch + index);
                    a += *w0.add(col) * value;
                    b += *w1.add(col) * value;
                    c += *w2.add(col) * value;
                    d += *w3.add(col) * value;
                }
                *p0.add(index) = a;
                *p1.add(index) = b;
                *p2.add(index) = c;
                *p3.add(index) = d;
            }
        }
    }
}
