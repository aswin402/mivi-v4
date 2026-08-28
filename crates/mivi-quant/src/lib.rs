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
            let count = bytes.len() / 4;
            assert!(out.len() >= count, "Output buffer too small for F32 dequant");
            for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                out[i] = f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
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
        GgmlType::Q8_0 => {
            matvec_q8_0(out, weights, x, n, d);
            Ok(())
        }
        GgmlType::F16 => {
            matvec_f16(out, weights, x, n, d);
            Ok(())
        }
        GgmlType::F32 => {
            if weights.as_ptr() as usize % std::mem::align_of::<f32>() == 0 {
                let float_slice = unsafe {
                    std::slice::from_raw_parts(weights.as_ptr() as *const f32, weights.len() / 4)
                };
                mivi_core::simd::matvec_f32(out, float_slice, x, n, d);
            } else {
                let mut aligned = vec![0.0f32; weights.len() / 4];
                for (i, chunk) in weights.chunks_exact(4).enumerate() {
                    aligned[i] = f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
                mivi_core::simd::matvec_f32(out, &aligned, x, n, d);
            }
            Ok(())
        }
        other => Err(QuantError::UnsupportedType(other as u32)),
    }
}
