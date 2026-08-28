//! Q8_0 (8-bit quantization) implementation with AVX2 SIMD & Rayon parallelism.

use half::f16;
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub const Q8_0_BLOCK_SIZE: usize = 32;
pub const Q8_0_BYTES: usize = 34;
pub const PARALLEL_CHUNK_SIZE: usize = 16;

/// Dequantize one Q8_0 block (34 bytes) into 32 f32 outputs.
#[inline]
pub fn dequantize_q8_0(block: &[u8], out: &mut [f32]) {
    debug_assert!(block.len() >= Q8_0_BYTES);
    debug_assert!(out.len() >= Q8_0_BLOCK_SIZE);

    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
    for i in 0..32 {
        let q = block[2 + i] as i8;
        out[i] = (q as f32) * d;
    }
}

/// Dequantize multi-block Q8_0 buffer into f32 slice.
pub fn dequantize_q8_0_slice(bytes: &[u8], out: &mut [f32]) {
    let n_blocks = out.len() / Q8_0_BLOCK_SIZE;
    debug_assert!(bytes.len() >= n_blocks * Q8_0_BYTES);

    for b in 0..n_blocks {
        let block = &bytes[b * Q8_0_BYTES..(b + 1) * Q8_0_BYTES];
        let out_sub = &mut out[b * Q8_0_BLOCK_SIZE..(b + 1) * Q8_0_BLOCK_SIZE];
        dequantize_q8_0(block, out_sub);
    }
}

/// High-performance Quantized matrix-vector multiplication for Q8_0 weights: out[n] = W[n, d] * x[d]
pub fn matvec_q8_0(out: &mut [f32], weights: &[u8], x: &[f32], n: usize, d: usize) {
    debug_assert_eq!(d % Q8_0_BLOCK_SIZE, 0);
    let blocks_per_row = d / Q8_0_BLOCK_SIZE;
    let row_bytes = blocks_per_row * Q8_0_BYTES;

    // Use Rayon chunk parallelization for larger matrices
    if n >= 64 {
        out.par_chunks_mut(PARALLEL_CHUNK_SIZE)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let start_row = chunk_idx * PARALLEL_CHUNK_SIZE;
                for (r, out_val) in out_chunk.iter_mut().enumerate() {
                    let i = start_row + r;
                    let row_start = i * row_bytes;
                    *out_val = compute_single_row_q8_0(&weights[row_start..row_start + row_bytes], x, blocks_per_row);
                }
            });
    } else {
        for i in 0..n {
            let row_start = i * row_bytes;
            out[i] = compute_single_row_q8_0(&weights[row_start..row_start + row_bytes], x, blocks_per_row);
        }
    }
}

#[inline]
fn compute_single_row_q8_0(row_weights: &[u8], x: &[f32], blocks_per_row: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                return compute_single_row_q8_0_avx2(row_weights, x, blocks_per_row);
            }
        }
    }

    compute_single_row_q8_0_scalar(row_weights, x, blocks_per_row)
}

#[inline]
fn compute_single_row_q8_0_scalar(row_weights: &[u8], x: &[f32], blocks_per_row: usize) -> f32 {
    let mut row_sum = 0.0f32;
    for b in 0..blocks_per_row {
        let block_offset = b * Q8_0_BYTES;
        let d = f16::from_le_bytes([row_weights[block_offset], row_weights[block_offset + 1]]).to_f32();
        let qs = &row_weights[block_offset + 2..block_offset + 34];
        let x_sub = &x[b * Q8_0_BLOCK_SIZE..(b + 1) * Q8_0_BLOCK_SIZE];

        let mut block_acc = 0.0f32;
        for j in 0..32 {
            block_acc += (qs[j] as i8 as f32) * x_sub[j];
        }
        row_sum += block_acc * d;
    }
    row_sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn compute_single_row_q8_0_avx2(row_weights: &[u8], x: &[f32], blocks_per_row: usize) -> f32 {
    assert!(row_weights.len() >= blocks_per_row * Q8_0_BYTES);
    assert!(x.len() >= blocks_per_row * Q8_0_BLOCK_SIZE);
    let mut total_acc = 0.0f32;

    for b in 0..blocks_per_row {
        let block_offset = b * Q8_0_BYTES;
        let d = f16::from_le_bytes([row_weights[block_offset], row_weights[block_offset + 1]]).to_f32();
        let qs_ptr = row_weights.as_ptr().add(block_offset + 2) as *const i8;
        let x_ptr = x.as_ptr().add(b * Q8_0_BLOCK_SIZE);

        let mut block_acc_v = _mm256_setzero_ps();

        // Process 32 elements in 4 chunks of 8 floats
        for k in 0..4 {
            let offset = k * 8;
            // Load 8 i8 values and sign-extend to 8 i32 values
            let q_i8 = _mm_loadl_epi64(qs_ptr.add(offset) as *const __m128i);
            let q_i32 = _mm256_cvtepi8_epi32(q_i8);
            // Convert i32 to f32
            let q_f32 = _mm256_cvtepi32_ps(q_i32);
            // Load 8 x f32 values
            let x_v = _mm256_loadu_ps(x_ptr.add(offset));

            block_acc_v = _mm256_fmadd_ps(q_f32, x_v, block_acc_v);
        }

        // Horizontal sum
        let lo = _mm256_castps256_ps128(block_acc_v);
        let hi = _mm256_extractf128_ps(block_acc_v, 1);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehl_ps(sum128, sum128);
        let sum64 = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_shuffle_ps(sum64, sum64, 1);
        let sum32 = _mm_add_ss(sum64, shuf2);

        let block_sum = _mm_cvtss_f32(sum32);
        total_acc += block_sum * d;
    }

    total_acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q8_0_dequant() {
        let mut block = vec![0u8; Q8_0_BYTES];
        block[0] = 0x00;
        block[1] = 0x40; // d = 2.0
        block[2] = 10;
        block[3] = (-5i8) as u8;

        let mut out = [0.0f32; 32];
        dequantize_q8_0(&block, &mut out);

        assert_eq!(out[0], 20.0);
        assert_eq!(out[1], -10.0);
    }

    #[test]
    fn test_q8_0_matvec_simd() {
        let dim = 64;
        let n = 2;
        let row_bytes = (dim / 32) * Q8_0_BYTES;
        let mut weights = vec![0u8; n * row_bytes];

        // Row 0 block 0: d = 1.0, elements = 1
        weights[0] = 0x00;
        weights[1] = 0x3C; // d = 1.0 in f16
        for j in 0..32 {
            weights[2 + j] = 1;
        }

        let x = vec![2.0f32; dim];
        let mut out = vec![0.0f32; n];
        matvec_q8_0(&mut out, &weights, &x, n, dim);

        assert!((out[0] - 64.0).abs() < 1e-3);
    }
}
