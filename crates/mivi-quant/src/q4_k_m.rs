//! Q4_K_M (4-bit K-quant Medium) implementation with multi-threaded parallelization.

use half::f16;
use rayon::prelude::*;

pub const Q4_K_BLOCK_SIZE: usize = 256;
pub const Q4_K_BYTES: usize = 144;

/// Dequantize one Q4_K_M block (144 bytes) into 256 f32 outputs.
#[inline]
pub fn dequantize_q4_k_m(block: &[u8], out: &mut [f32]) {
    debug_assert!(block.len() >= Q4_K_BYTES);
    debug_assert!(out.len() >= Q4_K_BLOCK_SIZE);

    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
    let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();

    let scales = &block[4..16];
    let qs = &block[16..144];

    // Decode 8 6-bit scale and min factors from 12 bytes
    let mut sc = [0u8; 8];
    let mut m = [0u8; 8];

    // First 4 scale/min pairs
    sc[0] = scales[0] & 63;
    sc[1] = scales[1] & 63;
    sc[2] = scales[2] & 63;
    sc[3] = scales[3] & 63;

    m[0] = scales[4] & 63;
    m[1] = scales[5] & 63;
    m[2] = scales[6] & 63;
    m[3] = scales[7] & 63;

    // High 2 bits of scales/mins stored in scales[8..12]
    sc[4] = (scales[8] & 15) | ((scales[0] >> 6) << 4);
    sc[5] = ((scales[8] >> 4) & 15) | ((scales[1] >> 6) << 4);
    sc[6] = (scales[9] & 15) | ((scales[2] >> 6) << 4);
    sc[7] = ((scales[9] >> 4) & 15) | ((scales[3] >> 6) << 4);

    m[4] = (scales[10] & 15) | ((scales[4] >> 6) << 4);
    m[5] = ((scales[10] >> 4) & 15) | ((scales[5] >> 6) << 4);
    m[6] = (scales[11] & 15) | ((scales[6] >> 6) << 4);
    m[7] = ((scales[11] >> 4) & 15) | ((scales[7] >> 6) << 4);

    // Dequantize 8 sub-blocks of 32 values each
    for sb in 0..8 {
        let d_sb = d * (sc[sb] as f32);
        let m_sb = dmin * (m[sb] as f32);
        let out_sub = &mut out[sb * 32..(sb + 1) * 32];
        let q_sub = &qs[sb * 16..(sb + 1) * 16];

        for i in 0..16 {
            let byte = q_sub[i];
            let q0 = (byte & 0x0F) as f32;
            let q1 = (byte >> 4) as f32;

            out_sub[i] = d_sb * q0 - m_sb;
            out_sub[i + 16] = d_sb * q1 - m_sb;
        }
    }
}

/// Dequantize multi-block Q4_K_M buffer into f32 slice.
pub fn dequantize_q4_k_m_slice(bytes: &[u8], out: &mut [f32]) {
    let n_blocks = out.len() / Q4_K_BLOCK_SIZE;
    debug_assert!(bytes.len() >= n_blocks * Q4_K_BYTES);

    for b in 0..n_blocks {
        let block = &bytes[b * Q4_K_BYTES..(b + 1) * Q4_K_BYTES];
        let out_sub = &mut out[b * Q4_K_BLOCK_SIZE..(b + 1) * Q4_K_BLOCK_SIZE];
        dequantize_q4_k_m(block, out_sub);
    }
}

/// Quantized matrix-vector multiplication for Q4_K_M weights: out[n] = W[n, d] * x[d]
pub fn matvec_q4_k_m(out: &mut [f32], weights: &[u8], x: &[f32], n: usize, d: usize) {
    debug_assert_eq!(d % Q4_K_BLOCK_SIZE, 0);
    let blocks_per_row = d / Q4_K_BLOCK_SIZE;
    let row_bytes = blocks_per_row * Q4_K_BYTES;

    if n >= 64 {
        out.par_chunks_mut(16)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let start_row = chunk_idx * 16;
                let mut dequant_buf = [0.0f32; Q4_K_BLOCK_SIZE];

                for (r, out_val) in out_chunk.iter_mut().enumerate() {
                    let i = start_row + r;
                    let row_start = i * row_bytes;
                    let mut row_sum = 0.0f32;

                    for b in 0..blocks_per_row {
                        let block_offset = row_start + b * Q4_K_BYTES;
                        let block_bytes = &weights[block_offset..block_offset + Q4_K_BYTES];
                        dequantize_q4_k_m(block_bytes, &mut dequant_buf);

                        let x_sub = &x[b * Q4_K_BLOCK_SIZE..(b + 1) * Q4_K_BLOCK_SIZE];
                        for j in 0..Q4_K_BLOCK_SIZE {
                            row_sum += dequant_buf[j] * x_sub[j];
                        }
                    }
                    *out_val = row_sum;
                }
            });
    } else {
        let mut dequant_buf = [0.0f32; Q4_K_BLOCK_SIZE];
        for i in 0..n {
            let row_start = i * row_bytes;
            let mut row_sum = 0.0f32;

            for b in 0..blocks_per_row {
                let block_offset = row_start + b * Q4_K_BYTES;
                let block_bytes = &weights[block_offset..block_offset + Q4_K_BYTES];
                dequantize_q4_k_m(block_bytes, &mut dequant_buf);

                let x_sub = &x[b * Q4_K_BLOCK_SIZE..(b + 1) * Q4_K_BLOCK_SIZE];
                for j in 0..Q4_K_BLOCK_SIZE {
                    row_sum += dequant_buf[j] * x_sub[j];
                }
            }
            out[i] = row_sum;
        }
    }
}
