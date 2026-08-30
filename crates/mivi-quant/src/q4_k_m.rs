//! Q4_K_M (4-bit K-quant Medium) implementation with multi-threaded parallelization.

use half::f16;

pub const Q4_K_BLOCK_SIZE: usize = 256;
pub const Q4_K_BYTES: usize = 144;

/// Low 6 bits mask for scale index extraction.
const SCALE_INDEX_MASK: u8 = 0x3F; // 63
/// Low 4 bits mask for nibble extraction.
const NIBBLE_MASK: u8 = 0x0F; // 15
/// Bit shift for scale high bits.
const SCALE_HIGH_SHIFT: u32 = 6;
/// Bit shift for high nibble extraction.
const NIBBLE_SHIFT: u32 = 4;

/// Dequantize one Q4_K_M block (144 bytes) into 256 f32 outputs.
#[inline]
pub fn dequantize_q4_k_m(block: &[u8], out: &mut [f32]) {
    assert!(block.len() >= Q4_K_BYTES, "Q4_K block buffer too small");
    assert!(out.len() >= Q4_K_BLOCK_SIZE, "Q4_K output buffer too small");

    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
    let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();

    let scales = &block[4..16];
    let qs = &block[16..144];

    // Decode 8 6-bit scale and min factors from 12 bytes
    let mut sc = [0u8; 8];
    let mut m = [0u8; 8];

    // First 4 scale/min pairs
    sc[0] = scales[0] & SCALE_INDEX_MASK;
    sc[1] = scales[1] & SCALE_INDEX_MASK;
    sc[2] = scales[2] & SCALE_INDEX_MASK;
    sc[3] = scales[3] & SCALE_INDEX_MASK;

    m[0] = scales[4] & SCALE_INDEX_MASK;
    m[1] = scales[5] & SCALE_INDEX_MASK;
    m[2] = scales[6] & SCALE_INDEX_MASK;
    m[3] = scales[7] & SCALE_INDEX_MASK;

    // High 2 bits of scales/mins stored in scales[8..12]
    sc[4] = (scales[8] & NIBBLE_MASK) | ((scales[0] >> SCALE_HIGH_SHIFT) << NIBBLE_SHIFT);
    sc[5] = ((scales[8] >> NIBBLE_SHIFT) & NIBBLE_MASK)
        | ((scales[1] >> SCALE_HIGH_SHIFT) << NIBBLE_SHIFT);
    sc[6] = (scales[9] & NIBBLE_MASK) | ((scales[2] >> SCALE_HIGH_SHIFT) << NIBBLE_SHIFT);
    sc[7] = ((scales[9] >> NIBBLE_SHIFT) & NIBBLE_MASK)
        | ((scales[3] >> SCALE_HIGH_SHIFT) << NIBBLE_SHIFT);

    m[4] = (scales[10] & NIBBLE_MASK) | ((scales[4] >> SCALE_HIGH_SHIFT) << NIBBLE_SHIFT);
    m[5] = ((scales[10] >> NIBBLE_SHIFT) & NIBBLE_MASK)
        | ((scales[5] >> SCALE_HIGH_SHIFT) << NIBBLE_SHIFT);
    m[6] = (scales[11] & NIBBLE_MASK) | ((scales[6] >> SCALE_HIGH_SHIFT) << NIBBLE_SHIFT);
    m[7] = ((scales[11] >> NIBBLE_SHIFT) & NIBBLE_MASK)
        | ((scales[7] >> SCALE_HIGH_SHIFT) << NIBBLE_SHIFT);

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
    crate::types::dequantize_blocks(bytes, out, Q4_K_BYTES, Q4_K_BLOCK_SIZE, dequantize_q4_k_m);
}

#[inline(always)]
fn compute_single_row_q4_k_m(
    row_bytes: &[u8],
    x: &[f32],
    blocks_per_row: usize,
    dequant_buf: &mut [f32; Q4_K_BLOCK_SIZE],
) -> f32 {
    let mut row_sum = 0.0f32;
    for b in 0..blocks_per_row {
        let block_offset = b * Q4_K_BYTES;
        let block_bytes = &row_bytes[block_offset..block_offset + Q4_K_BYTES];
        dequantize_q4_k_m(block_bytes, dequant_buf);

        let x_sub = &x[b * Q4_K_BLOCK_SIZE..(b + 1) * Q4_K_BLOCK_SIZE];
        for j in 0..Q4_K_BLOCK_SIZE {
            row_sum += dequant_buf[j] * x_sub[j];
        }
    }
    row_sum
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

    crate::types::parallel_row_matvec(out, weights, n, row_bytes, |row_slice, _| {
        let mut dequant_buf = [0.0f32; Q4_K_BLOCK_SIZE];
        compute_single_row_q4_k_m(row_slice, x, blocks_per_row, &mut dequant_buf)
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
