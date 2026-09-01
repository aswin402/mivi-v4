//! Float16 / BFloat16 matrix operations.

use half::{bf16, f16};

pub const F16_BYTES: usize = 2;

/// Dequantize generic 16-bit float slice to F32 using converter function.
#[inline]
fn dequantize_half<F: Fn([u8; 2]) -> f32>(input: &[u8], out: &mut [f32], convert: F) {
    let count = input.len() / F16_BYTES;
    assert!(
        out.len() >= count,
        "Output buffer too small: expected {}, got {}",
        count,
        out.len()
    );
    for i in 0..count {
        let val = convert([input[F16_BYTES * i], input[F16_BYTES * i + 1]]);
        out[i] = val;
    }
}

/// Dequantize F16 slice to F32
pub fn dequantize_f16(input: &[u8], out: &mut [f32]) {
    dequantize_half(input, out, |b| f16::from_le_bytes(b).to_f32());
}

/// Dequantize BF16 slice to F32
pub fn dequantize_bf16(input: &[u8], out: &mut [f32]) {
    dequantize_half(input, out, |b| bf16::from_le_bytes(b).to_f32());
}

/// Checked matvec for F16 weights returning QuantError on dimension mismatch.
pub fn try_matvec_f16(
    out: &mut [f32],
    weights: &[u8],
    x: &[f32],
    n: usize,
    d: usize,
) -> crate::types::Result<()> {
    let row_bytes = d
        .checked_mul(F16_BYTES)
        .ok_or(crate::types::QuantError::ArithmeticOverflow)?;
    crate::types::validate_matvec_args(out, weights, x, n, d, row_bytes, 1)?;

    crate::types::parallel_row_matvec(out, weights, n, row_bytes, |row_slice, _| {
        let mut sum = 0.0f32;
        for (j, &xj) in x.iter().enumerate().take(d) {
            let off = j * F16_BYTES;
            let w = f16::from_le_bytes([row_slice[off], row_slice[off + 1]]).to_f32();
            sum += w * xj;
        }
        sum
    });
    Ok(())
}

/// Matvec for F16 weights: out[n] = W[n, d] * x[d] with Rayon multithreading.
#[track_caller]
pub fn matvec_f16(out: &mut [f32], weights: &[u8], x: &[f32], n: usize, d: usize) {
    if let Err(e) = try_matvec_f16(out, weights, x, n, d) {
        panic!("{}", e);
    }
}

/// Checked matvec for BF16 weights returning QuantError on dimension mismatch.
pub fn try_matvec_bf16(
    out: &mut [f32],
    weights: &[u8],
    x: &[f32],
    n: usize,
    d: usize,
) -> crate::types::Result<()> {
    let row_bytes = d
        .checked_mul(F16_BYTES)
        .ok_or(crate::types::QuantError::ArithmeticOverflow)?;
    crate::types::validate_matvec_args(out, weights, x, n, d, row_bytes, 1)?;

    crate::types::parallel_row_matvec(out, weights, n, row_bytes, |row_slice, _| {
        let mut sum = 0.0f32;
        for (j, &xj) in x.iter().enumerate().take(d) {
            let off = j * F16_BYTES;
            let w = bf16::from_le_bytes([row_slice[off], row_slice[off + 1]]).to_f32();
            sum += w * xj;
        }
        sum
    });
    Ok(())
}

/// Matvec for BF16 weights: out[n] = W[n, d] * x[d] with Rayon multithreading.
#[track_caller]
pub fn matvec_bf16(out: &mut [f32], weights: &[u8], x: &[f32], n: usize, d: usize) {
    if let Err(e) = try_matvec_bf16(out, weights, x, n, d) {
        panic!("{}", e);
    }
}
