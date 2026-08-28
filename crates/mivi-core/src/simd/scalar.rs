//! Portable scalar fallback implementations for tensor operations.

#[inline]
pub fn matvec_f32_scalar(out: &mut [f32], w: &[f32], x: &[f32], n: usize, d: usize) {
    for i in 0..n {
        let row = &w[i * d..(i + 1) * d];
        let mut sum = 0.0f32;
        for j in 0..d {
            sum += row[j] * x[j];
        }
        out[i] = sum;
    }
}
