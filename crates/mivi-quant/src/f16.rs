//! Float16 / BFloat16 matrix operations.

use half::{bf16, f16};

/// Dequantize F16 slice to F32
pub fn dequantize_f16(input: &[u8], out: &mut [f32]) {
    let count = input.len() / 2;
    assert!(out.len() >= count, "Output buffer too small for F16 dequant");
    for i in 0..count {
        let val = f16::from_le_bytes([input[2 * i], input[2 * i + 1]]).to_f32();
        out[i] = val;
    }
}

/// Dequantize BF16 slice to F32
pub fn dequantize_bf16(input: &[u8], out: &mut [f32]) {
    let count = input.len() / 2;
    assert!(out.len() >= count, "Output buffer too small for BF16 dequant");
    for i in 0..count {
        let val = bf16::from_le_bytes([input[2 * i], input[2 * i + 1]]).to_f32();
        out[i] = val;
    }
}

use rayon::prelude::*;

pub const PARALLEL_CHUNK_SIZE: usize = 16;

/// Matvec for F16 weights: out[n] = W[n, d] * x[d] with Rayon multithreading.
pub fn matvec_f16(out: &mut [f32], weights: &[u8], x: &[f32], n: usize, d: usize) {
    assert_eq!(out.len(), n);
    assert_eq!(x.len(), d);
    assert!(weights.len() >= n * d * 2);

    let row_bytes = d * 2;
    out.par_chunks_mut(PARALLEL_CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start_row = chunk_idx * PARALLEL_CHUNK_SIZE;
            for (offset, out_val) in out_chunk.iter_mut().enumerate() {
                let row = start_row + offset;
                let row_offset = row * row_bytes;
                let mut sum = 0.0f32;
                for j in 0..d {
                    let off = row_offset + j * 2;
                    let w = f16::from_le_bytes([weights[off], weights[off + 1]]).to_f32();
                    sum += w * x[j];
                }
                *out_val = sum;
            }
        });
}
