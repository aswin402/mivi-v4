//! Q4_K_M (4-bit K-quant Medium) implementation with multi-threaded parallelization.

use half::f16;

pub const Q4_K_BLOCK_SIZE: usize = 256;
pub const Q4_K_BYTES: usize = 144;

#[inline(always)]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        let d = q[j] & 63;
        let m = q[j + 4] & 63;
        (d, m)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// Dequantize one Q4_K_M block (144 bytes) into 256 f32 outputs.
#[inline]
pub fn dequantize_q4_k_m(block: &[u8], out: &mut [f32]) {
    assert!(block.len() >= Q4_K_BYTES, "Q4_K block buffer too small");
    assert!(out.len() >= Q4_K_BLOCK_SIZE, "Q4_K output buffer too small");

    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
    let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();

    let scales = &block[4..16];
    let qs = &block[16..144];

    // Dequantize 4 groups of 64 values (each group contains two 32-element sub-blocks)
    for j in 0..4 {
        let (sc0, m0) = get_scale_min_k4(2 * j, scales);
        let (sc1, m1) = get_scale_min_k4(2 * j + 1, scales);

        let d1 = d * (sc0 as f32);
        let m1_val = dmin * (m0 as f32);
        let d2 = d * (sc1 as f32);
        let m2_val = dmin * (m1 as f32);

        let q_sub = &qs[j * 32..(j + 1) * 32];
        let out_sub = &mut out[j * 64..(j + 1) * 64];

        for l in 0..32 {
            let byte = q_sub[l];
            let q0 = (byte & 0x0F) as f32;
            let q1 = (byte >> 4) as f32;

            out_sub[l] = d1 * q0 - m1_val;
            out_sub[l + 32] = d2 * q1 - m2_val;
        }
    }
}

/// Dequantize multi-block Q4_K_M buffer into f32 slice.
pub fn dequantize_q4_k_m_slice(bytes: &[u8], out: &mut [f32]) {
    crate::types::dequantize_blocks(bytes, out, Q4_K_BYTES, Q4_K_BLOCK_SIZE, dequantize_q4_k_m);
}

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
fn compute_single_row_q4_k_m(
    row_bytes: &[u8],
    x: &[f32],
    blocks_per_row: usize,
    use_avx2: bool,
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if use_avx2 {
            unsafe {
                return compute_single_row_q4_k_m_avx2(row_bytes, x, blocks_per_row);
            }
        }
    }

    let _ = use_avx2;
    compute_single_row_q4_k_m_scalar(row_bytes, x, blocks_per_row)
}

#[inline]
fn compute_single_row_q4_k_m_scalar(row_bytes: &[u8], x: &[f32], blocks_per_row: usize) -> f32 {
    let mut total_acc = 0.0f32;

    for b in 0..blocks_per_row {
        let block_offset = b * Q4_K_BYTES;
        let block = &row_bytes[block_offset..block_offset + Q4_K_BYTES];

        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();

        let scales = &block[4..16];
        let qs = &block[16..144];
        let x_block = &x[b * Q4_K_BLOCK_SIZE..(b + 1) * Q4_K_BLOCK_SIZE];

        for j in 0..4 {
            let (sc0, m0) = get_scale_min_k4(2 * j, scales);
            let (sc1, m1) = get_scale_min_k4(2 * j + 1, scales);

            let d1 = d * (sc0 as f32);
            let m1_val = dmin * (m0 as f32);
            let d2 = d * (sc1 as f32);
            let m2_val = dmin * (m1 as f32);

            let q_sub = &qs[j * 32..(j + 1) * 32];
            let x0 = &x_block[j * 64..j * 64 + 32];
            let x1 = &x_block[j * 64 + 32..j * 64 + 64];

            let mut sum_q0 = 0.0f32;
            let mut sum_x0 = 0.0f32;
            let mut sum_q1 = 0.0f32;
            let mut sum_x1 = 0.0f32;

            for l in 0..32 {
                let byte = q_sub[l];
                let q0 = (byte & 0x0F) as f32;
                let q1 = (byte >> 4) as f32;
                let x0_val = x0[l];
                let x1_val = x1[l];

                sum_q0 += q0 * x0_val;
                sum_x0 += x0_val;
                sum_q1 += q1 * x1_val;
                sum_x1 += x1_val;
            }

            total_acc += d1 * sum_q0 - m1_val * sum_x0 + d2 * sum_q1 - m2_val * sum_x1;
        }
    }

    total_acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn compute_single_row_q4_k_m_avx2(
    row_bytes: &[u8],
    x: &[f32],
    blocks_per_row: usize,
) -> f32 {
    let mut total_acc = 0.0f32;
    let mask_0f = _mm256_set1_epi32(0x0F);

    for b in 0..blocks_per_row {
        let block_offset = b * Q4_K_BYTES;
        let block = &row_bytes[block_offset..block_offset + Q4_K_BYTES];

        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();

        let scales = &block[4..16];
        let qs_ptr = block.as_ptr().add(16);
        let x_ptr = x.as_ptr().add(b * Q4_K_BLOCK_SIZE);

        for j in 0..4 {
            let (sc0, m0) = get_scale_min_k4(2 * j, scales);
            let (sc1, m1) = get_scale_min_k4(2 * j + 1, scales);

            let d1 = d * (sc0 as f32);
            let m1_val = dmin * (m0 as f32);
            let d2 = d * (sc1 as f32);
            let m2_val = dmin * (m1 as f32);

            let q_group_ptr = qs_ptr.add(j * 32);
            let x0_ptr = x_ptr.add(j * 64);
            let x1_ptr = x_ptr.add(j * 64 + 32);

            let mut acc_q0 = _mm256_setzero_ps();
            let mut acc_x0 = _mm256_setzero_ps();
            let mut acc_q1 = _mm256_setzero_ps();
            let mut acc_x1 = _mm256_setzero_ps();

            // Process 32 bytes in 4 chunks of 8 bytes
            for k in 0..4 {
                let offset = k * 8;
                let raw_8b = _mm_loadl_epi64(q_group_ptr.add(offset) as *const __m128i);
                let bytes_i32 = _mm256_cvtepu8_epi32(raw_8b);

                // Low nibbles (q0)
                let q0_i32 = _mm256_and_si256(bytes_i32, mask_0f);
                let q0_f32 = _mm256_cvtepi32_ps(q0_i32);
                let x0_v = _mm256_loadu_ps(x0_ptr.add(offset));

                acc_q0 = _mm256_fmadd_ps(q0_f32, x0_v, acc_q0);
                acc_x0 = _mm256_add_ps(acc_x0, x0_v);

                // High nibbles (q1)
                let q1_i32 = _mm256_srli_epi32(bytes_i32, 4);
                let q1_f32 = _mm256_cvtepi32_ps(q1_i32);
                let x1_v = _mm256_loadu_ps(x1_ptr.add(offset));

                acc_q1 = _mm256_fmadd_ps(q1_f32, x1_v, acc_q1);
                acc_x1 = _mm256_add_ps(acc_x1, x1_v);
            }

            let sum_q0 = mivi_core::simd::hsum256_ps(acc_q0);
            let sum_x0 = mivi_core::simd::hsum256_ps(acc_x0);
            let sum_q1 = mivi_core::simd::hsum256_ps(acc_q1);
            let sum_x1 = mivi_core::simd::hsum256_ps(acc_x1);

            total_acc += d1 * sum_q0 - m1_val * sum_x0 + d2 * sum_q1 - m2_val * sum_x1;
        }
    }

    total_acc
}

/// Checked matvec for Q4_K_M weights returning QuantError on dimension mismatch.
pub fn try_matvec_q4_k_m(
    out: &mut [f32],
    weights: &[u8],
    x: &[f32],
    n: usize,
    d: usize,
) -> crate::types::Result<()> {
    let blocks_per_row = d / Q4_K_BLOCK_SIZE;
    let row_bytes = blocks_per_row * Q4_K_BYTES;
    crate::types::validate_matvec_args(out, weights, x, n, d, row_bytes, Q4_K_BLOCK_SIZE)?;

    #[cfg(target_arch = "x86_64")]
    let use_avx2 = *mivi_core::simd::HAS_AVX2_FMA;
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;

    crate::types::parallel_row_matvec(out, weights, n, row_bytes, |row_slice, _| {
        compute_single_row_q4_k_m(row_slice, x, blocks_per_row, use_avx2)
    });
    Ok(())
}

/// Quantized matrix-vector multiplication for Q4_K_M weights: out[n] = W[n, d] * x[d]
///
/// # Panics
/// Panics if buffer lengths are insufficient or unaligned. Prefer `try_matvec_q4_k_m` in fallible contexts.
#[track_caller]
pub fn matvec_q4_k_m(out: &mut [f32], weights: &[u8], x: &[f32], n: usize, d: usize) {
    if let Err(e) = try_matvec_q4_k_m(out, weights, x, n, d) {
        panic!("{}", e);
    }
}
