//! Float16 / BFloat16 matrix operations.

use half::{bf16, f16};

/// Dequantize F16 slice to F32
pub fn dequantize_f16(input: &[u8], out: &mut [f32]) {
    let count = input.len() / 2;
    for i in 0..count {
        let val = f16::from_le_bytes([input[2 * i], input[2 * i + 1]]).to_f32();
        out[i] = val;
    }
}

/// Dequantize BF16 slice to F32
pub fn dequantize_bf16(input: &[u8], out: &mut [f32]) {
    let count = input.len() / 2;
    for i in 0..count {
        let val = bf16::from_le_bytes([input[2 * i], input[2 * i + 1]]).to_f32();
        out[i] = val;
    }
}

/// Matvec for F16 weights: out[n] = W[n, d] * x[d]
pub fn matvec_f16(out: &mut [f32], weights: &[u8], x: &[f32], n: usize, d: usize) {
    for i in 0..n {
        let row_offset = i * d * 2;
        let mut sum = 0.0f32;
        for j in 0..d {
            let offset = row_offset + j * 2;
            let w = f16::from_le_bytes([weights[offset], weights[offset + 1]]).to_f32();
            sum += w * x[j];
        }
        out[i] = sum;
    }
}
