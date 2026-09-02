//! Q8_0 (8-bit quantization) implementation with AVX2 SIMD & Rayon parallelism.

use half::f16;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub const Q8_0_BLOCK_SIZE: usize = 32;
pub const Q8_0_BYTES: usize = 34;

/// Dequantize one Q8_0 block (34 bytes) into 32 f32 outputs.
#[inline]
pub fn dequantize_q8_0(block: &[u8], out: &mut [f32]) {
    assert!(block.len() >= Q8_0_BYTES, "Q8_0 block buffer too small");
    assert!(out.len() >= Q8_0_BLOCK_SIZE, "Q8_0 output buffer too small");

    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
    for i in 0..Q8_0_BLOCK_SIZE {
        let q = block[2 + i] as i8;
        out[i] = (q as f32) * d;
    }
}

/// Dequantize multi-block Q8_0 buffer into f32 slice.
pub fn dequantize_q8_0_slice(bytes: &[u8], out: &mut [f32]) {
    crate::types::dequantize_blocks(bytes, out, Q8_0_BYTES, Q8_0_BLOCK_SIZE, dequantize_q8_0);
}

/// Quantize 32 float elements into one Q8_0 block (34 bytes: 2 bytes f16 scale + 32 i8 values).
#[inline]
pub fn quantize_f32_to_q8_0_block(src: &[f32], dst_block: &mut [u8]) {
    assert!(src.len() >= Q8_0_BLOCK_SIZE, "Source float slice too small");
    assert!(dst_block.len() >= Q8_0_BYTES, "Destination block buffer too small");

    let mut amax = 0.0f32;
    for &val in &src[..Q8_0_BLOCK_SIZE] {
        let abs = val.abs();
        if abs > amax {
            amax = abs;
        }
    }

    let d = amax / 127.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };

    let d_f16 = f16::from_f32(d);
    let d_bytes = d_f16.to_le_bytes();
    dst_block[0] = d_bytes[0];
    dst_block[1] = d_bytes[1];

    for i in 0..Q8_0_BLOCK_SIZE {
        let q = (src[i] * id).round().clamp(-128.0, 127.0) as i8;
        dst_block[2 + i] = q as u8;
    }
}

/// Compute dot product between f32 Query block (length = 32) and Q8_0 Key block (34 bytes: f16 scale + 32 i8 values).
#[inline]
pub fn dot_q8_0_f32(q: &[f32], k_block: &[u8]) -> f32 {
    let d = f16::from_le_bytes([k_block[0], k_block[1]]).to_f32();

    #[cfg(target_arch = "x86_64")]
    if *mivi_core::simd::HAS_AVX2_FMA && q.len() >= Q8_0_BLOCK_SIZE && k_block.len() >= Q8_0_BYTES {
        unsafe {
            return dot_q8_0_f32_avx2(q, k_block, d);
        }
    }

    let mut sum = 0.0f32;
    for i in 0..Q8_0_BLOCK_SIZE.min(q.len()) {
        let k_val = (k_block[2 + i] as i8) as f32;
        sum += q[i] * k_val;
    }
    sum * d
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_q8_0_f32_avx2(q: &[f32], k_block: &[u8], d: f32) -> f32 {
    let q_ptr = q.as_ptr();
    let k_ptr = k_block.as_ptr().add(2); // Skip 2 bytes of f16 scale

    // Load 32 i8 elements into two 128-bit registers (16 bytes each)
    let k_vec_16_0 = _mm_loadu_si128(k_ptr as *const __m128i);
    let k_vec_16_1 = _mm_loadu_si128(k_ptr.add(16) as *const __m128i);

    // Sign extend i8 -> i32 in chunks of 8
    let k_i32_0 = _mm256_cvtepi8_epi32(k_vec_16_0);
    let k_i32_1 = _mm256_cvtepi8_epi32(_mm_srli_si128(k_vec_16_0, 8));
    let k_i32_2 = _mm256_cvtepi8_epi32(k_vec_16_1);
    let k_i32_3 = _mm256_cvtepi8_epi32(_mm_srli_si128(k_vec_16_1, 8));

    // Convert i32 -> f32
    let k_f32_0 = _mm256_cvtepi32_ps(k_i32_0);
    let k_f32_1 = _mm256_cvtepi32_ps(k_i32_1);
    let k_f32_2 = _mm256_cvtepi32_ps(k_i32_2);
    let k_f32_3 = _mm256_cvtepi32_ps(k_i32_3);

    // Load f32 query values
    let q_0 = _mm256_loadu_ps(q_ptr);
    let q_1 = _mm256_loadu_ps(q_ptr.add(8));
    let q_2 = _mm256_loadu_ps(q_ptr.add(16));
    let q_3 = _mm256_loadu_ps(q_ptr.add(24));

    // Fused multiply-add
    let mut acc = _mm256_mul_ps(q_0, k_f32_0);
    acc = _mm256_fmadd_ps(q_1, k_f32_1, acc);
    acc = _mm256_fmadd_ps(q_2, k_f32_2, acc);
    acc = _mm256_fmadd_ps(q_3, k_f32_3, acc);

    // Horizontal sum of acc
    let hi128 = _mm256_extractf128_ps(acc, 1);
    let lo128 = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_shuffle_ps(sum64, sum64, 1);
    let sum32 = _mm_add_ss(sum64, shuf2);

    _mm_cvtss_f32(sum32) * d
}

/// Checked matvec for Q8_0 weights returning QuantError on dimension mismatch.
pub fn try_matvec_q8_0(
    out: &mut [f32],
    weights: &[u8],
    x: &[f32],
    n: usize,
    d: usize,
) -> crate::types::Result<()> {
    let blocks_per_row = d / Q8_0_BLOCK_SIZE;
    let row_bytes = blocks_per_row * Q8_0_BYTES;
    crate::types::validate_matvec_args(out, weights, x, n, d, row_bytes, Q8_0_BLOCK_SIZE)?;

    #[cfg(target_arch = "x86_64")]
    let use_avx2 = *mivi_core::simd::HAS_AVX2_FMA;
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;

    crate::types::parallel_row_matvec(out, weights, n, row_bytes, |row_slice, _| {
        compute_single_row_q8_0(row_slice, x, blocks_per_row, use_avx2)
    });
    Ok(())
}

/// High-performance Quantized matrix-vector multiplication for Q8_0 weights: out[n] = W[n, d] * x[d]
///
/// # Panics
/// Panics if buffer lengths are insufficient or unaligned. Prefer `try_matvec_q8_0` in fallible contexts.
#[track_caller]
pub fn matvec_q8_0(out: &mut [f32], weights: &[u8], x: &[f32], n: usize, d: usize) {
    if let Err(e) = try_matvec_q8_0(out, weights, x, n, d) {
        panic!("{}", e);
    }
}

#[inline]
fn compute_single_row_q8_0(
    row_weights: &[u8],
    x: &[f32],
    blocks_per_row: usize,
    use_avx2: bool,
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if use_avx2 {
            unsafe {
                return compute_single_row_q8_0_avx2(row_weights, x, blocks_per_row);
            }
        }
    }

    let _ = use_avx2;
    compute_single_row_q8_0_scalar(row_weights, x, blocks_per_row)
}

#[inline]
fn compute_single_row_q8_0_scalar(row_weights: &[u8], x: &[f32], blocks_per_row: usize) -> f32 {
    let mut row_sum = 0.0f32;
    for b in 0..blocks_per_row {
        let block_offset = b * Q8_0_BYTES;
        let d =
            f16::from_le_bytes([row_weights[block_offset], row_weights[block_offset + 1]]).to_f32();
        let qs = &row_weights[block_offset + 2..block_offset + Q8_0_BYTES];
        let x_sub = &x[b * Q8_0_BLOCK_SIZE..(b + 1) * Q8_0_BLOCK_SIZE];

        let mut block_acc = 0.0f32;
        for j in 0..Q8_0_BLOCK_SIZE {
            block_acc += (qs[j] as i8 as f32) * x_sub[j];
        }
        row_sum += block_acc * d;
    }
    row_sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
/// # Safety
/// Caller must ensure AVX2 and FMA CPU features are enabled and row buffer lengths match `blocks_per_row * Q8_0_BYTES`.
unsafe fn compute_single_row_q8_0_avx2(
    row_weights: &[u8],
    x: &[f32],
    blocks_per_row: usize,
) -> f32 {
    let mut total_acc = 0.0f32;

    for b in 0..blocks_per_row {
        let block_offset = b * Q8_0_BYTES;
        let d =
            f16::from_le_bytes([row_weights[block_offset], row_weights[block_offset + 1]]).to_f32();
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

        // Horizontal sum using shared helper
        let block_sum = mivi_core::simd::hsum256_ps(block_acc_v);
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

    #[test]
    fn test_q8_0_quantize_and_dot_product() {
        let mut src = [0.0f32; 32];
        for i in 0..32 {
            src[i] = (i as f32) - 16.0;
        }

        let mut block = [0u8; Q8_0_BYTES];
        quantize_f32_to_q8_0_block(&src, &mut block);

        let mut dequant = [0.0f32; 32];
        dequantize_q8_0(&block, &mut dequant);

        // Check quantization error is low (< 0.2 error on max range 16.0)
        for i in 0..32 {
            assert!((src[i] - dequant[i]).abs() < 0.25);
        }

        // Test dot product against itself
        let dot = dot_q8_0_f32(&src, &block);
        let mut expected_dot = 0.0f32;
        for i in 0..32 {
            expected_dot += src[i] * dequant[i];
        }
        assert!((dot - expected_dot).abs() < 1e-3);
    }
}
