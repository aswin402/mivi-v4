//! ARM64 NEON optimized kernels.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[inline]
pub unsafe fn matvec_f32_neon(out: &mut [f32], w: &[f32], x: &[f32], n: usize, d: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        let chunks = d / 4;
        let remainder = d % 4;

        for i in 0..n {
            let row_ptr = w.as_ptr().add(i * d);
            let x_ptr = x.as_ptr();

            let mut acc = vdupq_n_f32(0.0);

            for c in 0..chunks {
                let offset = c * 4;
                let wv = vld1q_f32(row_ptr.add(offset));
                let xv = vld1q_f32(x_ptr.add(offset));
                acc = vfmaq_f32(acc, wv, xv);
            }

            let mut sum = vaddvq_f32(acc);

            let rem_start = chunks * 4;
            for r in 0..remainder {
                sum += *row_ptr.add(rem_start + r) * *x_ptr.add(rem_start + r);
            }

            out[i] = sum;
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (out, w, x, n, d);
    }
}
