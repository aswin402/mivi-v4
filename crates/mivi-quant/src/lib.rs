//! Quantization definitions, dequantization routines, and quantized matrix-vector operations.

pub mod f16;
pub mod q4_k_m;
pub mod q8_0;
pub mod types;

pub use f16::{dequantize_bf16, dequantize_f16, matvec_f16};
pub use q4_k_m::{dequantize_q4_k_m, dequantize_q4_k_m_slice, matvec_q4_k_m, Q4_K_BLOCK_SIZE, Q4_K_BYTES};
pub use q8_0::{dequantize_q8_0, dequantize_q8_0_slice, matvec_q8_0, Q8_0_BLOCK_SIZE, Q8_0_BYTES};
pub use types::{GgmlType, QuantError, Result};

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
        GgmlType::F16 => {
            dequantize_f16(bytes, out);
            Ok(())
        }
        GgmlType::F32 => {
            let float_slice = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4)
            };
            out.copy_from_slice(float_slice);
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
        GgmlType::Q8_0 => {
            matvec_q8_0(out, weights, x, n, d);
            Ok(())
        }
        GgmlType::F16 => {
            matvec_f16(out, weights, x, n, d);
            Ok(())
        }
        GgmlType::F32 => {
            let float_slice = unsafe {
                std::slice::from_raw_parts(weights.as_ptr() as *const f32, weights.len() / 4)
            };
            mivi_core::simd::matvec_f32(out, float_slice, x, n, d);
            Ok(())
        }
        other => Err(QuantError::UnsupportedType(other as u32)),
    }
}
