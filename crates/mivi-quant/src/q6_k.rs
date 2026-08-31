//! Q6_K (6-bit K-quant) implementation with SIMD / multi-threaded parallelization.

use half::f16;

pub const Q6_K_BLOCK_SIZE: usize = 256;
pub const Q6_K_BYTES: usize = 210;

/// Dequantize one Q6_K block (210 bytes) into 256 f32 outputs.
#[inline]
pub fn dequantize_q6_k(block: &[u8], out: &mut [f32]) {
    assert!(block.len() >= Q6_K_BYTES, "Q6_K block buffer too small");
    assert!(out.len() >= Q6_K_BLOCK_SIZE, "Q6_K output buffer too small");

    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    let d = f16::from_le_bytes([block[208], block[209]]).to_f32();

    for half in 0..2 {
        let ql_half = &ql[half * 64..(half + 1) * 64];
        let qh_half = &qh[half * 32..(half + 1) * 32];
        let sc_half = &scales[half * 8..(half + 1) * 8];
        let y_half = &mut out[half * 128..(half + 1) * 128];

        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql_half[l] & 0x0F) | ((qh_half[l] & 3) << 4)) as i8 - 32;
            let q2 = ((ql_half[l + 32] & 0x0F) | (((qh_half[l] >> 2) & 3) << 4)) as i8 - 32;
            let q3 = ((ql_half[l] >> 4) | (((qh_half[l] >> 4) & 3) << 4)) as i8 - 32;
            let q4 = ((ql_half[l + 32] >> 4) | (((qh_half[l] >> 6) & 3) << 4)) as i8 - 32;

            y_half[l] = d * (sc_half[is] as i8 as f32) * (q1 as f32);
            y_half[l + 32] = d * (sc_half[is + 2] as i8 as f32) * (q2 as f32);
            y_half[l + 64] = d * (sc_half[is + 4] as i8 as f32) * (q3 as f32);
            y_half[l + 96] = d * (sc_half[is + 6] as i8 as f32) * (q4 as f32);
        }
    }
}

/// Dequantize multi-block Q6_K buffer into f32 slice.
pub fn dequantize_q6_k_slice(bytes: &[u8], out: &mut [f32]) {
    crate::types::dequantize_blocks(bytes, out, Q6_K_BYTES, Q6_K_BLOCK_SIZE, dequantize_q6_k);
}

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
fn compute_single_row_q6_k(
    row_bytes: &[u8],
    x: &[f32],
    blocks_per_row: usize,
    use_avx2: bool,
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if use_avx2 {
            unsafe {
                return compute_single_row_q6_k_avx2(row_bytes, x, blocks_per_row);
            }
        }
    }

    let _ = use_avx2;
    compute_single_row_q6_k_scalar(row_bytes, x, blocks_per_row)
}

#[inline]
fn compute_single_row_q6_k_scalar(row_bytes: &[u8], x: &[f32], blocks_per_row: usize) -> f32 {
    let mut total_acc = 0.0f32;

    for b in 0..blocks_per_row {
        let block_offset = b * Q6_K_BYTES;
        let block = &row_bytes[block_offset..block_offset + Q6_K_BYTES];

        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = f16::from_le_bytes([block[208], block[209]]).to_f32();
        let x_block = &x[b * Q6_K_BLOCK_SIZE..(b + 1) * Q6_K_BLOCK_SIZE];

        let mut block_sum = 0.0f32;

        for half in 0..2 {
            let ql_half = &ql[half * 64..(half + 1) * 64];
            let qh_half = &qh[half * 32..(half + 1) * 32];
            let sc_half = &scales[half * 8..(half + 1) * 8];
            let x_half = &x_block[half * 128..(half + 1) * 128];

            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql_half[l] & 0x0F) | ((qh_half[l] & 3) << 4)) as i8 - 32;
                let q2 = ((ql_half[l + 32] & 0x0F) | (((qh_half[l] >> 2) & 3) << 4)) as i8 - 32;
                let q3 = ((ql_half[l] >> 4) | (((qh_half[l] >> 4) & 3) << 4)) as i8 - 32;
                let q4 = ((ql_half[l + 32] >> 4) | (((qh_half[l] >> 6) & 3) << 4)) as i8 - 32;

                let sc0 = sc_half[is] as i8 as f32;
                let sc1 = sc_half[is + 2] as i8 as f32;
                let sc2 = sc_half[is + 4] as i8 as f32;
                let sc3 = sc_half[is + 6] as i8 as f32;

                block_sum += sc0 * (q1 as f32) * x_half[l]
                    + sc1 * (q2 as f32) * x_half[l + 32]
                    + sc2 * (q3 as f32) * x_half[l + 64]
                    + sc3 * (q4 as f32) * x_half[l + 96];
            }
        }
        total_acc += block_sum * d;
    }
    total_acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn compute_single_row_q6_k_avx2(row_bytes: &[u8], x: &[f32], blocks_per_row: usize) -> f32 {
    let mut total_acc = 0.0f32;
    let mask_0f = _mm256_set1_epi32(0x0F);
    let mask_03 = _mm256_set1_epi32(0x03);
    let c_32 = _mm256_set1_ps(32.0);

    for b in 0..blocks_per_row {
        let block_offset = b * Q6_K_BYTES;
        let block = &row_bytes[block_offset..block_offset + Q6_K_BYTES];

        let ql_ptr = block.as_ptr();
        let qh_ptr = block.as_ptr().add(128);
        let scales_ptr = block.as_ptr().add(192);
        let d = f16::from_le_bytes([block[208], block[209]]).to_f32();
        let x_ptr = x.as_ptr().add(b * Q6_K_BLOCK_SIZE);

        let mut block_acc_v = _mm256_setzero_ps();

        for half in 0..2 {
            let ql_half = ql_ptr.add(half * 64);
            let qh_half = qh_ptr.add(half * 32);
            let sc_half = scales_ptr.add(half * 8) as *const i8;
            let x_half = x_ptr.add(half * 128);

            for is in 0..2 {
                let sc0_v = _mm256_set1_ps(*sc_half.add(is) as f32);
                let sc1_v = _mm256_set1_ps(*sc_half.add(is + 2) as f32);
                let sc2_v = _mm256_set1_ps(*sc_half.add(is + 4) as f32);
                let sc3_v = _mm256_set1_ps(*sc_half.add(is + 6) as f32);

                for k in 0..2 {
                    let l = is * 16 + k * 8;

                    let ql_0 =
                        _mm256_cvtepu8_epi32(_mm_loadl_epi64(ql_half.add(l) as *const __m128i));
                    let ql_32 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                        ql_half.add(l + 32) as *const __m128i
                    ));
                    let qh_0 =
                        _mm256_cvtepu8_epi32(_mm_loadl_epi64(qh_half.add(l) as *const __m128i));

                    // q1 = (ql_0 & 0x0F) | ((qh_0 & 3) << 4) - 32
                    let q1_i = _mm256_or_si256(
                        _mm256_and_si256(ql_0, mask_0f),
                        _mm256_slli_epi32(_mm256_and_si256(qh_0, mask_03), 4),
                    );
                    let q1_f = _mm256_sub_ps(_mm256_cvtepi32_ps(q1_i), c_32);
                    let x0_v = _mm256_loadu_ps(x_half.add(l));
                    block_acc_v = _mm256_fmadd_ps(_mm256_mul_ps(sc0_v, q1_f), x0_v, block_acc_v);

                    // q2 = (ql_32 & 0x0F) | (((qh_0 >> 2) & 3) << 4) - 32
                    let q2_i = _mm256_or_si256(
                        _mm256_and_si256(ql_32, mask_0f),
                        _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(qh_0, 2), mask_03), 4),
                    );
                    let q2_f = _mm256_sub_ps(_mm256_cvtepi32_ps(q2_i), c_32);
                    let x1_v = _mm256_loadu_ps(x_half.add(l + 32));
                    block_acc_v = _mm256_fmadd_ps(_mm256_mul_ps(sc1_v, q2_f), x1_v, block_acc_v);

                    // q3 = (ql_0 >> 4) | (((qh_0 >> 4) & 3) << 4) - 32
                    let q3_i = _mm256_or_si256(
                        _mm256_srli_epi32(ql_0, 4),
                        _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(qh_0, 4), mask_03), 4),
                    );
                    let q3_f = _mm256_sub_ps(_mm256_cvtepi32_ps(q3_i), c_32);
                    let x2_v = _mm256_loadu_ps(x_half.add(l + 64));
                    block_acc_v = _mm256_fmadd_ps(_mm256_mul_ps(sc2_v, q3_f), x2_v, block_acc_v);

                    // q4 = (ql_32 >> 4) | (((qh_0 >> 6) & 3) << 4) - 32
                    let q4_i = _mm256_or_si256(
                        _mm256_srli_epi32(ql_32, 4),
                        _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(qh_0, 6), mask_03), 4),
                    );
                    let q4_f = _mm256_sub_ps(_mm256_cvtepi32_ps(q4_i), c_32);
                    let x3_v = _mm256_loadu_ps(x_half.add(l + 96));
                    block_acc_v = _mm256_fmadd_ps(_mm256_mul_ps(sc3_v, q4_f), x3_v, block_acc_v);
                }
            }
        }
        total_acc += mivi_core::simd::hsum256_ps(block_acc_v) * d;
    }

    total_acc
}

/// Checked matvec for Q6_K weights returning QuantError on dimension mismatch.
pub fn try_matvec_q6_k(
    out: &mut [f32],
    weights: &[u8],
    x: &[f32],
    n: usize,
    d: usize,
) -> crate::types::Result<()> {
    let blocks_per_row = d / Q6_K_BLOCK_SIZE;
    let row_bytes = blocks_per_row * Q6_K_BYTES;
    crate::types::validate_matvec_args(out, weights, x, n, d, row_bytes, Q6_K_BLOCK_SIZE)?;

    #[cfg(target_arch = "x86_64")]
    let use_avx2 = *mivi_core::simd::HAS_AVX2_FMA;
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;

    crate::types::parallel_row_matvec(out, weights, n, row_bytes, |row_slice, _| {
        compute_single_row_q6_k(row_slice, x, blocks_per_row, use_avx2)
    });
    Ok(())
}

/// Quantized matrix-vector multiplication for Q6_K weights: out[n] = W[n, d] * x[d]
#[track_caller]
pub fn matvec_q6_k(out: &mut [f32], weights: &[u8], x: &[f32], n: usize, d: usize) {
    if let Err(e) = try_matvec_q6_k(out, weights, x, n, d) {
        panic!("{}", e);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q6_k_dequant_roundtrip() {
        let mut block = vec![0u8; Q6_K_BYTES];
        // Set scale d = 1.0
        let d_bytes = f16::from_f32(1.0).to_le_bytes();
        block[208] = d_bytes[0];
        block[209] = d_bytes[1];

        // Set scale[0] = 2
        block[192] = 2;
        // Set ql[0] = 5, qh[0] = 0 -> q1 = 5 - 32 = -27, y = 1.0 * 2 * -27 = -54.0
        block[0] = 5;

        let mut out = vec![0.0f32; Q6_K_BLOCK_SIZE];
        dequantize_q6_k(&block, &mut out);
        assert!((out[0] - (-54.0)).abs() < 1e-3);
    }
}
