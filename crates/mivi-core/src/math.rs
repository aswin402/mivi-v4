//! Mathematical primitives for transformer and SSM operations.

#[inline(always)]
fn assert_same_len(a_name: &str, a_len: usize, b_name: &str, b_len: usize) {
    assert_eq!(
        a_len, b_len,
        "Length mismatch: {} ({}) != {} ({})",
        a_name, a_len, b_name, b_len
    );
}

/// RMS Normalization: y = x * weight / sqrt(mean(x^2) + eps)
#[inline]
pub fn rms_norm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    if x.is_empty() {
        return;
    }
    assert_same_len("out", out.len(), "x", x.len());
    assert_same_len("x", x.len(), "weight", weight.len());

    let len = x.len();
    let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
    let scale = 1.0 / (sum_sq / (len as f32) + eps).sqrt();

    for i in 0..len {
        out[i] = x[i] * scale * weight[i];
    }
}

/// In-place Softmax over a slice of f32 logits or attention weights.
#[inline]
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let mut max_val = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v.is_finite() && v > max_val {
            max_val = v;
        }
    }

    if !max_val.is_finite() {
        let uniform = 1.0 / x.len() as f32;
        for v in x.iter_mut() {
            *v = uniform;
        }
        return;
    }

    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        if v.is_finite() {
            *v = (*v - max_val).exp();
            sum += *v;
        } else {
            *v = 0.0;
        }
    }

    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for v in x.iter_mut() {
            *v *= inv_sum;
        }
    } else {
        let uniform = 1.0 / x.len() as f32;
        for v in x.iter_mut() {
            *v = uniform;
        }
    }
}

/// Scalar SiLU activation: x / (1.0 + exp(-x))
#[inline]
pub fn silu_scalar(x: f32) -> f32 {
    if x <= -50.0 {
        0.0
    } else {
        x / (1.0 + (-x).exp())
    }
}

/// In-place SiLU (Swish) activation: x = x * sigmoid(x)
#[inline]
pub fn silu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = silu_scalar(*v);
    }
}

/// SwiGLU in-place: gate[i] = silu(gate[i]) * up[i]
#[inline]
pub fn swiglu(gate: &mut [f32], up: &[f32]) {
    assert_same_len("gate", gate.len(), "up", up.len());

    for (g, &u) in gate.iter_mut().zip(up.iter()) {
        *g = silu_scalar(*g) * u;
    }
}

/// Scalar fallback for vector addition: out[i] += src[i]
#[inline]
pub fn vec_add_scalar(out: &mut [f32], src: &[f32]) {
    assert_same_len("out", out.len(), "src", src.len());
    for i in 0..out.len() {
        out[i] += src[i];
    }
}

/// Elementwise vector addition: out[i] += src[i] (SIMD accelerated)
#[inline]
pub fn vec_add(out: &mut [f32], src: &[f32]) {
    assert_same_len("out", out.len(), "src", src.len());
    crate::simd::vec_add_simd(out, src);
}

/// Scalar fallback for fused vector addition: out[i] += scale * src[i]
#[inline]
pub fn vec_fmadd_scalar(out: &mut [f32], scale: f32, src: &[f32]) {
    assert_same_len("out", out.len(), "src", src.len());
    for i in 0..out.len() {
        out[i] += scale * src[i];
    }
}

/// Elementwise fused multiply-add: out[i] += scale * src[i] (SIMD accelerated)
#[inline]
pub fn vec_fmadd(out: &mut [f32], scale: f32, src: &[f32]) {
    assert_same_len("out", out.len(), "src", src.len());
    crate::simd::vec_fmadd_simd(out, scale, src);
}

/// Scalar fallback for vector dot product.
#[inline]
pub fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    assert_same_len("a", a.len(), "b", b.len());
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Vector dot product (SIMD accelerated).
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_same_len("a", a.len(), "b", b.len());
    crate::simd::dot_product_simd(a, b)
}

/// Rotary Position Embedding (RoPE) applied to all query and key heads.
#[inline]
pub fn apply_rope(q: &mut [f32], k: &mut [f32], head_dim: usize, pos: usize, rope_base: f32) {
    assert!(head_dim.is_multiple_of(2));
    assert!(q.len() >= head_dim);
    assert!(k.len() >= head_dim);

    // Rotate all query heads
    for q_head in q.chunks_exact_mut(head_dim) {
        for i in (0..head_dim).step_by(2) {
            let freq = 1.0 / rope_base.powf(i as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            let (sin, cos) = angle.sin_cos();

            let q0 = q_head[i];
            let q1 = q_head[i + 1];
            q_head[i] = q0 * cos - q1 * sin;
            q_head[i + 1] = q0 * sin + q1 * cos;
        }
    }

    // Rotate all key heads
    for k_head in k.chunks_exact_mut(head_dim) {
        for i in (0..head_dim).step_by(2) {
            let freq = 1.0 / rope_base.powf(i as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            let (sin, cos) = angle.sin_cos();

            let k0 = k_head[i];
            let k1 = k_head[i + 1];
            k_head[i] = k0 * cos - k1 * sin;
            k_head[i + 1] = k0 * sin + k1 * cos;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let mut out = vec![0.0; 4];
        rms_norm(&mut out, &x, &weight, 1e-5);
        let sum_sq: f32 = out.iter().map(|v| v * v).sum();
        let rms = (sum_sq / 4.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_softmax() {
        let mut x = vec![1.0, 2.0, 3.0];
        softmax(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(x[2] > x[1] && x[1] > x[0]);
    }

    #[test]
    fn test_softmax_all_neg_infinity() {
        let mut x = vec![f32::NEG_INFINITY; 4];
        softmax(&mut x);
        assert!(!x[0].is_nan());
        assert_eq!(x[0], 0.25);
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_overflow_infinity_and_nan() {
        let mut x = vec![f32::INFINITY, f32::NAN, 5.0, 10.0];
        softmax(&mut x);
        assert!(x.iter().all(|v| v.is_finite()));
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_silu() {
        let mut x = vec![0.0, 2.0, -2.0];
        silu(&mut x);
        assert_eq!(x[0], 0.0);
        assert!(x[1] > 0.0);
        assert!(x[2] < 0.0);
    }

    #[test]
    fn test_swiglu() {
        let mut gate = vec![1.0, 2.0];
        let up = vec![2.0, 3.0];
        swiglu(&mut gate, &up);
        assert!(gate[0] > 0.0);
        assert!(gate[1] > 0.0);
    }
}
