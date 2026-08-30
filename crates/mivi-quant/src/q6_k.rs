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

#[inline(always)]
fn compute_single_row_q6_k(
    row_bytes: &[u8],
    x: &[f32],
    blocks_per_row: usize,
    dequant_buf: &mut [f32; Q6_K_BLOCK_SIZE],
) -> f32 {
    let mut row_sum = 0.0f32;
    for b in 0..blocks_per_row {
        let block_offset = b * Q6_K_BYTES;
        let block_bytes = &row_bytes[block_offset..block_offset + Q6_K_BYTES];
        dequantize_q6_k(block_bytes, dequant_buf);

        let x_sub = &x[b * Q6_K_BLOCK_SIZE..(b + 1) * Q6_K_BLOCK_SIZE];
        for j in 0..Q6_K_BLOCK_SIZE {
            row_sum += dequant_buf[j] * x_sub[j];
        }
    }
    row_sum
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

    crate::types::parallel_row_matvec(out, weights, n, row_bytes, |row_slice, _| {
        let mut dequant_buf = [0.0f32; Q6_K_BLOCK_SIZE];
        compute_single_row_q6_k(row_slice, x, blocks_per_row, &mut dequant_buf)
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
