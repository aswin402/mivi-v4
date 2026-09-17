//! Quantization definitions, dequantization routines, and quantized matrix-vector operations.

pub mod f16;
pub mod q4_k_m;
pub mod q6_k;
pub mod q8_0;
pub mod types;

pub use f16::{
    dequantize_bf16, dequantize_f16, matvec_bf16, matvec_f16, try_matvec_bf16, try_matvec_f16,
};
pub use q4_k_m::{
    dequantize_q4_k_m, dequantize_q4_k_m_slice, matvec_q4_k_m, try_matvec_q4_k_m, Q4_K_BLOCK_SIZE,
    Q4_K_BYTES,
};
pub use q6_k::{
    dequantize_q6_k, dequantize_q6_k_slice, matvec_q6_k, try_matvec_q6_k, Q6_K_BLOCK_SIZE,
    Q6_K_BYTES,
};
pub use q8_0::{
    dequantize_q8_0, dequantize_q8_0_slice, dot_q8_0_f32, matvec_q8_0, quantize_f32_to_q8_0_block,
    try_matvec_q8_0, Q8_0_BLOCK_SIZE, Q8_0_BYTES,
};
pub use types::{
    parallel_row_matvec, validate_matmul_args, validate_matvec_args, GgmlType, QuantError, Result,
    PARALLEL_CHUNK_SIZE, RAYON_PARALLEL_THRESHOLD,
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
        GgmlType::Q4_K => try_matvec_q4_k_m(out, weights, x, n, d),
        GgmlType::Q6_K => try_matvec_q6_k(out, weights, x, n, d),
        GgmlType::Q8_0 => try_matvec_q8_0(out, weights, x, n, d),
        GgmlType::F16 => try_matvec_f16(out, weights, x, n, d),
        GgmlType::BF16 => try_matvec_bf16(out, weights, x, n, d),
        GgmlType::F32 => {
            let row_bytes = d.checked_mul(F32_BYTES).ok_or(QuantError::BufferTooSmall {
                expected: usize::MAX,
                actual: weights.len(),
            })?;
            validate_matvec_args(out, weights, x, n, d, row_bytes, 1)?;
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

/// Row-major quantized matrix-matrix multiplication.
///
/// `weights` is `[rows, cols]`, `inputs` is `[batch, cols]`, and `out` is
/// `[batch, rows]`. Each quantized weight row is decoded once and reused for
/// every input row in the batch. This reference kernel provides the stable
/// batch API; format-specific tiled SIMD kernels can replace its inner loop
/// without changing callers.
pub fn quantized_matmul_rows(
    out: &mut [f32],
    ggml_type: GgmlType,
    weights: &[u8],
    inputs: &[f32],
    batch: usize,
    rows: usize,
    cols: usize,
) -> Result<()> {
    let block_size = ggml_type
        .block_size()
        .ok_or(QuantError::UnsupportedType(ggml_type as u32))?;
    let type_size = ggml_type
        .type_size()
        .ok_or(QuantError::UnsupportedType(ggml_type as u32))?;
    if !cols.is_multiple_of(block_size) {
        return Err(QuantError::DimensionMisaligned {
            dim: cols,
            block_size,
        });
    }
    let row_bytes = (cols / block_size)
        .checked_mul(type_size)
        .ok_or(QuantError::ArithmeticOverflow)?;
    validate_matmul_args(
        out, weights, inputs, batch, rows, cols, row_bytes, block_size,
    )?;

    // Preserve the exact established matvec result for the one-row case. Tiles
    // with more than one input row use the batch kernel below.
    if batch == 1 {
        return quantized_matvec(
            &mut out[..rows],
            ggml_type,
            weights,
            &inputs[..cols],
            rows,
            cols,
        );
    }

    if batch == 0 || rows == 0 {
        return Ok(());
    }

    let accumulation_mode = batch_accumulation_mode(batch);
    // Transpose once so the tile dimension is contiguous for the large-batch
    // kernel. Small batches use full-width input dot products instead: calling
    // a SIMD helper once per column over a two- or eight-element slice creates
    // thousands of tiny calls and defeats the established matvec kernels.
    let transposed_inputs = if accumulation_mode == BatchAccumulationMode::AcrossBatchFma {
        let mut transposed_inputs = vec![0.0f32; cols * batch];
        for batch_idx in 0..batch {
            for col in 0..cols {
                transposed_inputs[col * batch + batch_idx] = inputs[batch_idx * cols + col];
            }
        }
        transposed_inputs
    } else {
        Vec::new()
    };
    let mut row_major_output = vec![0.0f32; rows * batch];

    if rows >= types::RAYON_PARALLEL_THRESHOLD {
        let midpoint = rows / 2;
        let (left, right) = row_major_output.split_at_mut(midpoint * batch);
        let (left_result, right_result) = rayon::join(
            || {
                compute_batched_rows(
                    left,
                    0,
                    midpoint,
                    batch,
                    cols,
                    row_bytes,
                    ggml_type,
                    weights,
                    inputs,
                    &transposed_inputs,
                    accumulation_mode,
                )
            },
            || {
                compute_batched_rows(
                    right,
                    midpoint,
                    rows,
                    batch,
                    cols,
                    row_bytes,
                    ggml_type,
                    weights,
                    inputs,
                    &transposed_inputs,
                    accumulation_mode,
                )
            },
        );
        left_result?;
        right_result?;
    } else {
        compute_batched_rows(
            &mut row_major_output,
            0,
            rows,
            batch,
            cols,
            row_bytes,
            ggml_type,
            weights,
            inputs,
            &transposed_inputs,
            accumulation_mode,
        )?;
    }

    for row in 0..rows {
        for batch_idx in 0..batch {
            out[batch_idx * rows + row] = row_major_output[row * batch + batch_idx];
        }
    }
    Ok(())
}

fn compute_batched_rows(
    row_major_output: &mut [f32],
    row_start: usize,
    row_end: usize,
    batch: usize,
    cols: usize,
    row_bytes: usize,
    ggml_type: GgmlType,
    weights: &[u8],
    inputs: &[f32],
    transposed_inputs: &[f32],
    accumulation_mode: BatchAccumulationMode,
) -> Result<()> {
    let mut decoded_row = vec![0.0f32; cols];
    let mut row_output = vec![0.0f32; batch];
    for row_idx in row_start..row_end {
        let weight_start = row_idx * row_bytes;
        dequantize_slice(
            ggml_type,
            &weights[weight_start..weight_start + row_bytes],
            &mut decoded_row,
        )?;
        row_output.fill(0.0);
        match accumulation_mode {
            BatchAccumulationMode::PerInputDot => {
                for (batch_idx, output) in row_output.iter_mut().enumerate() {
                    let input_start = batch_idx * cols;
                    *output = mivi_core::simd::dot_product_simd(
                        &decoded_row,
                        &inputs[input_start..input_start + cols],
                    );
                }
            }
            BatchAccumulationMode::AcrossBatchFma => {
                for col in 0..cols {
                    mivi_core::simd::vec_fmadd_simd(
                        &mut row_output,
                        decoded_row[col],
                        &transposed_inputs[col * batch..(col + 1) * batch],
                    );
                }
            }
        }
        let output_start = (row_idx - row_start) * batch;
        row_major_output[output_start..output_start + batch].copy_from_slice(&row_output);
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BatchAccumulationMode {
    PerInputDot,
    AcrossBatchFma,
}

#[inline]
fn batch_accumulation_mode(batch: usize) -> BatchAccumulationMode {
    if batch <= 8 {
        BatchAccumulationMode::PerInputDot
    } else {
        BatchAccumulationMode::AcrossBatchFma
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row_bytes(ggml_type: GgmlType, cols: usize) -> usize {
        cols / ggml_type.block_size().unwrap() * ggml_type.type_size().unwrap()
    }

    fn assert_batched_matches_matvec(ggml_type: GgmlType, cols: usize) {
        let rows = 3;
        let weights = vec![0u8; rows * row_bytes(ggml_type, cols)];
        let inputs = (0..3 * cols)
            .map(|idx| (idx as f32 - 3.0) * 0.125)
            .collect::<Vec<_>>();

        for batch in [1, 2, 3] {
            let mut expected = vec![0.0f32; batch * rows];
            for input_idx in 0..batch {
                quantized_matvec(
                    &mut expected[input_idx * rows..(input_idx + 1) * rows],
                    ggml_type,
                    &weights,
                    &inputs[input_idx * cols..(input_idx + 1) * cols],
                    rows,
                    cols,
                )
                .unwrap();
            }

            let mut actual = vec![0.0f32; batch * rows];
            quantized_matmul_rows(
                &mut actual,
                ggml_type,
                &weights,
                &inputs[..batch * cols],
                batch,
                rows,
                cols,
            )
            .unwrap();

            assert_eq!(actual, expected, "batch={batch}, type={ggml_type:?}");
        }
    }

    #[test]
    fn batched_matmul_matches_matvec_for_supported_types() {
        assert_batched_matches_matvec(GgmlType::F32, 8);
        assert_batched_matches_matvec(GgmlType::F16, 8);
        assert_batched_matches_matvec(GgmlType::BF16, 8);
        assert_batched_matches_matvec(GgmlType::Q8_0, 32);
        assert_batched_matches_matvec(GgmlType::Q4_K, 256);
        assert_batched_matches_matvec(GgmlType::Q6_K, 256);
    }

    #[test]
    fn batched_matmul_rejects_short_buffers_and_misaligned_dimensions() {
        let weights = vec![0u8; 2 * 32];
        let inputs = vec![0.0f32; 2 * 32];
        let mut output = vec![0.0f32; 2 * 2];

        assert!(quantized_matmul_rows(
            &mut output[..3],
            GgmlType::Q8_0,
            &weights,
            &inputs,
            2,
            2,
            32,
        )
        .is_err());
        assert!(quantized_matmul_rows(
            &mut output,
            GgmlType::Q8_0,
            &weights,
            &inputs[..31],
            2,
            2,
            32,
        )
        .is_err());
        assert!(quantized_matmul_rows(
            &mut output,
            GgmlType::Q8_0,
            &weights[..31],
            &inputs,
            2,
            2,
            32,
        )
        .is_err());
        assert!(
            quantized_matmul_rows(&mut output, GgmlType::Q8_0, &weights, &inputs, 2, 2, 31,)
                .is_err()
        );
    }

    #[test]
    fn small_batches_use_full_input_dot_products() {
        assert_eq!(
            batch_accumulation_mode(2),
            BatchAccumulationMode::PerInputDot
        );
        assert_eq!(
            batch_accumulation_mode(8),
            BatchAccumulationMode::PerInputDot
        );
        assert_eq!(
            batch_accumulation_mode(16),
            BatchAccumulationMode::AcrossBatchFma
        );
    }
}
