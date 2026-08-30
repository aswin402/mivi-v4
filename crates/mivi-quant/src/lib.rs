//! Quantization definitions, dequantization routines, and quantized matrix-vector operations.

pub mod f16;
pub mod q4_k_m;
pub mod q6_k;
pub mod q8_0;
pub mod types;

pub use f16::{dequantize_bf16, dequantize_f16, matvec_f16, try_matvec_f16};
pub use q4_k_m::{
    dequantize_q4_k_m, dequantize_q4_k_m_slice, matvec_q4_k_m, try_matvec_q4_k_m, Q4_K_BLOCK_SIZE,
    Q4_K_BYTES,
};
pub use q6_k::{
    dequantize_q6_k, dequantize_q6_k_slice, matvec_q6_k, try_matvec_q6_k, Q6_K_BLOCK_SIZE,
    Q6_K_BYTES,
};
pub use q8_0::{
    dequantize_q8_0, dequantize_q8_0_slice, matvec_q8_0, try_matvec_q8_0, Q8_0_BLOCK_SIZE,
    Q8_0_BYTES,
};
pub use types::{
    parallel_row_matvec, validate_matvec_args, GgmlType, QuantError, Result, PARALLEL_CHUNK_SIZE,
    RAYON_PARALLEL_THRESHOLD,
};

pub const F32_BYTES: usize = 4;
pub const DEQUANT_STACK_CHUNK: usize = 256;

/// Dequantize arbitrary slice of quantized weights into f32 buffer.
pub fn dequantize_slice(ggml_type: GgmlType, bytes: &[u8], out: &mut [f32]) -> Result<()> {
    match ggml_type {
        GgmlType::Q8_0 => {
            dequantize_q8_0_slice(bytes, out);
            Ok(())
        }
        GgmlType::Q4_K => {
            dequantize_q4_k_m_slice(bytes, out);
            Ok(())
        }
        GgmlType::Q6_K => {
            dequantize_q6_k_slice(bytes, out);
            Ok(())
        }
        GgmlType::F16 => {
            dequantize_f16(bytes, out);
            Ok(())
        }
        GgmlType::BF16 => {
            dequantize_bf16(bytes, out);
            Ok(())
        }
        GgmlType::F32 => {
            let count = bytes.len() / F32_BYTES;
            if out.len() < count {
                return Err(QuantError::BufferTooSmall {
                    expected: count,
                    actual: out.len(),
                });
            }
            for (i, chunk) in bytes.chunks_exact(F32_BYTES).enumerate() {
                out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            Ok(())
        }
        other => Err(QuantError::UnsupportedType(other as u32)),
    }
}

/// Unified quantized matrix-vector multiplication dispatcher.
pub fn quantized_matvec(
    out: &mut [f32],
    ggml_type: GgmlType,
    weights: &[u8],
    x: &[f32],
    n: usize,
    d: usize,
) -> Result<()> {
    match ggml_type {
        GgmlType::Q4_K => {
            matvec_q4_k_m(out, weights, x, n, d);
            Ok(())
        }
        GgmlType::Q6_K => {
            matvec_q6_k(out, weights, x, n, d);
            Ok(())
        }
        GgmlType::Q8_0 => {
            matvec_q8_0(out, weights, x, n, d);
            Ok(())
        }
        GgmlType::F16 => {
            matvec_f16(out, weights, x, n, d);
            Ok(())
        }
        GgmlType::F32 => {
            if (weights.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>()) {
                // SAFETY: Pointer is non-null, valid for reads, memory-aligned to 4 bytes,
                // and the resulting slice lifetime is bounded by the input &weights reference.
                let float_slice = unsafe {
                    std::slice::from_raw_parts(
                        weights.as_ptr() as *const f32,
                        weights.len() / F32_BYTES,
                    )
                };
                mivi_core::simd::matvec_f32(out, float_slice, x, n, d);
            } else {
                // Zero-allocation fallback for unaligned F32 weights using stack buffer
                let mut stack_buf = [0.0f32; DEQUANT_STACK_CHUNK];
                for (row, out_val) in out.iter_mut().enumerate().take(n) {
                    let row_offset = row * d * F32_BYTES;
                    let mut sum = 0.0f32;
                    let mut col = 0;
                    while col < d {
                        let chunk_len = (d - col).min(DEQUANT_STACK_CHUNK);
                        let byte_start = row_offset + col * F32_BYTES;
                        let byte_end = byte_start + chunk_len * F32_BYTES;
                        let chunk_bytes = &weights[byte_start..byte_end];
                        for (i, chunk) in chunk_bytes.chunks_exact(F32_BYTES).enumerate() {
                            stack_buf[i] =
                                f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        }
                        let x_chunk = &x[col..col + chunk_len];
                        for k in 0..chunk_len {
                            sum += stack_buf[k] * x_chunk[k];
                        }
                        col += chunk_len;
                    }
                    *out_val = sum;
                }
            }
            Ok(())
        }
        other => Err(QuantError::UnsupportedType(other as u32)),
    }
}
