//! Quantization definitions, dequantization routines, and quantized matrix-vector operations.

pub mod f16;
#[cfg(feature = "projection-diagnostics")]
pub mod projection_diagnostics;
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

use rayon::prelude::*;

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
    quantized_matmul_rows_impl::<false>(out, ggml_type, weights, inputs, batch, rows, cols)
        .map(|_| ())
}

/// Quantized matrix-matrix multiplication with an explicit projection token
/// traversal width for the large-batch accumulation kernel.
#[cfg(feature = "projection-locality-experiment")]
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_rows_with_token_tile(
    out: &mut [f32],
    ggml_type: GgmlType,
    weights: &[u8],
    inputs: &[f32],
    batch: usize,
    rows: usize,
    cols: usize,
    tile: mivi_core::simd::ProjectionTokenTile,
) -> Result<()> {
    quantized_matmul_rows_with_tile_impl::<false>(
        out,
        ggml_type,
        weights,
        inputs,
        batch,
        rows,
        cols,
        Some(tile),
    )
    .map(|_| ())
}

#[derive(Debug, Clone, Default)]
#[allow(dead_code)] // Diagnostic-only fields are intentionally inert in default builds.
pub(crate) struct WorkerWorkInternal {
    pub scratch_init_ns: u64,
    pub decode_ns: u64,
    pub accumulate_ns: u64,
    pub zero_copy_ns: u64,
    pub rows: usize,
}

#[derive(Debug, Clone, Default)]
#[allow(dead_code)] // Populated and read only by the opt-in feature instantiation.
pub(crate) struct ProjectionProfileInternal {
    pub schema: u32,
    pub branch: &'static str,
    pub call_wall_ns: u64,
    pub validation_ns: u64,
    pub buffer_init_ns: Option<u64>,
    pub input_transpose_ns: Option<u64>,
    pub rows_wall_ns: Option<u64>,
    pub output_layout_ns: Option<u64>,
    pub delegated_matvec_ns: Option<u64>,
    pub unclassified_wall_ns: u64,
    pub workers: Vec<WorkerWorkInternal>,
}

fn measured<const PROFILE: bool, T>(operation: impl FnOnce() -> T) -> (T, Option<u64>) {
    let started = if PROFILE {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let value = operation();
    let elapsed_ns = started.map(|t| u64::try_from(t.elapsed().as_nanos()).unwrap_or(u64::MAX));
    (value, elapsed_ns)
}

fn add_elapsed<const PROFILE: bool>(total: &mut u64, elapsed: Option<u64>) {
    if PROFILE {
        *total = total.saturating_add(elapsed.unwrap_or_default());
    }
}

fn finish_profile<const PROFILE: bool>(
    mut profile: Option<ProjectionProfileInternal>,
    call_started: Option<std::time::Instant>,
) -> Option<ProjectionProfileInternal> {
    if PROFILE {
        let profile = profile.as_mut().expect("profiled invocation has a profile");
        profile.call_wall_ns = u64::try_from(
            call_started
                .expect("profiled invocation has a call timer")
                .elapsed()
                .as_nanos(),
        )
        .unwrap_or(u64::MAX);
        let serial_stages = profile
            .validation_ns
            .saturating_add(profile.buffer_init_ns.unwrap_or_default())
            .saturating_add(profile.input_transpose_ns.unwrap_or_default())
            .saturating_add(profile.rows_wall_ns.unwrap_or_default())
            .saturating_add(profile.output_layout_ns.unwrap_or_default())
            .saturating_add(profile.delegated_matvec_ns.unwrap_or_default());
        profile.unclassified_wall_ns = profile.call_wall_ns.saturating_sub(serial_stages);
    }
    profile
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn quantized_matmul_rows_impl<const PROFILE: bool>(
    out: &mut [f32],
    ggml_type: GgmlType,
    weights: &[u8],
    inputs: &[f32],
    batch: usize,
    rows: usize,
    cols: usize,
) -> Result<Option<ProjectionProfileInternal>> {
    quantized_matmul_rows_with_tile_impl::<PROFILE>(
        out, ggml_type, weights, inputs, batch, rows, cols, None,
    )
}

#[allow(clippy::too_many_arguments)]
fn quantized_matmul_rows_with_tile_impl<const PROFILE: bool>(
    out: &mut [f32],
    ggml_type: GgmlType,
    weights: &[u8],
    inputs: &[f32],
    batch: usize,
    rows: usize,
    cols: usize,
    token_tile: Option<mivi_core::simd::ProjectionTokenTile>,
) -> Result<Option<ProjectionProfileInternal>> {
    let call_started = if PROFILE {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let profile = if PROFILE {
        Some(ProjectionProfileInternal {
            schema: 1,
            ..ProjectionProfileInternal::default()
        })
    } else {
        None
    };

    let (validation, validation_ns) = measured::<PROFILE, _>(|| {
        if !matches!(
            ggml_type,
            GgmlType::Q8_0
                | GgmlType::Q4_K
                | GgmlType::Q6_K
                | GgmlType::F16
                | GgmlType::BF16
                | GgmlType::F32
        ) {
            return Err(QuantError::UnsupportedType(ggml_type as u32));
        }
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
        let branch = if batch == 0 || rows == 0 {
            "empty"
        } else if batch == 1 {
            "matvec"
        } else if batch <= 8 {
            "per_input_dot"
        } else if batch >= 32 {
            "across_batch_pair"
        } else {
            "across_batch"
        };
        Ok((row_bytes, branch))
    });
    let (row_bytes, branch) = validation?;
    let mut profile = profile;
    if PROFILE {
        let current = profile.as_mut().expect("profiled invocation has a profile");
        current.validation_ns = validation_ns.unwrap_or_default();
        current.branch = branch;
    }

    if batch == 0 || rows == 0 {
        return Ok(finish_profile::<PROFILE>(profile, call_started));
    }

    // Preserve the exact established matvec result for the one-row case. Tiles
    // with more than one input row use the batch kernel below.
    if batch == 1 {
        let (result, delegated_matvec_ns) = measured::<PROFILE, _>(|| {
            quantized_matvec(
                &mut out[..rows],
                ggml_type,
                weights,
                &inputs[..cols],
                rows,
                cols,
            )
        });
        result?;
        if PROFILE {
            profile.as_mut().unwrap().delegated_matvec_ns = delegated_matvec_ns;
        }
        return Ok(finish_profile::<PROFILE>(profile, call_started));
    }

    let accumulation_mode = batch_accumulation_mode(batch);
    // Transpose once so the tile dimension is contiguous for the large-batch
    // kernel. Small batches use full-width input dot products instead: calling
    // a SIMD helper once per column over a two- or eight-element slice creates
    // thousands of tiny calls and defeats the established matvec kernels.
    let ((mut transposed_inputs, mut row_major_output), buffer_init_ns) =
        measured::<PROFILE, _>(|| {
            let transposed_inputs = if accumulation_mode == BatchAccumulationMode::AcrossBatchFma {
                vec![0.0f32; cols * batch]
            } else {
                Vec::new()
            };
            let row_major_output = vec![0.0f32; rows * batch];
            (transposed_inputs, row_major_output)
        });
    if PROFILE {
        profile.as_mut().unwrap().buffer_init_ns = buffer_init_ns;
    }
    if accumulation_mode == BatchAccumulationMode::AcrossBatchFma {
        let (_, input_transpose_ns) = measured::<PROFILE, _>(|| {
            for batch_idx in 0..batch {
                for col in 0..cols {
                    transposed_inputs[col * batch + batch_idx] = inputs[batch_idx * cols + col];
                }
            }
        });
        if PROFILE {
            profile.as_mut().unwrap().input_transpose_ns = input_transpose_ns;
        }
    }

    let num_threads = rayon::current_num_threads().max(1);
    let parallel = rows >= types::RAYON_PARALLEL_THRESHOLD && num_threads > 1;
    let execute_rows = || {
        if parallel {
            let chunk_rows = rows.div_ceil(num_threads);
            row_major_output
                .par_chunks_mut(chunk_rows * batch)
                .enumerate()
                .map(|(chunk_idx, slice)| {
                    let row_start = chunk_idx * chunk_rows;
                    let row_end = (row_start + chunk_rows).min(rows);
                    compute_batched_rows::<PROFILE>(
                        slice,
                        row_start,
                        row_end,
                        batch,
                        cols,
                        row_bytes,
                        ggml_type,
                        weights,
                        inputs,
                        &transposed_inputs,
                        accumulation_mode,
                        token_tile,
                    )
                })
                .collect::<Result<Vec<_>>>()
        } else {
            compute_batched_rows::<PROFILE>(
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
                token_tile,
            )
            .map(|worker| vec![worker])
        }
    };

    let worker_records = if PROFILE {
        let (result, rows_wall_ns) = measured::<PROFILE, _>(execute_rows);
        let records = result?;
        profile.as_mut().unwrap().rows_wall_ns = rows_wall_ns;
        records
    } else if parallel {
        let chunk_rows = rows.div_ceil(num_threads);
        row_major_output
            .par_chunks_mut(chunk_rows * batch)
            .enumerate()
            .try_for_each(|(chunk_idx, slice)| {
                let row_start = chunk_idx * chunk_rows;
                let row_end = (row_start + chunk_rows).min(rows);
                compute_batched_rows::<PROFILE>(
                    slice,
                    row_start,
                    row_end,
                    batch,
                    cols,
                    row_bytes,
                    ggml_type,
                    weights,
                    inputs,
                    &transposed_inputs,
                    accumulation_mode,
                    token_tile,
                )
                .map(|_| ())
            })?;
        Vec::new()
    } else {
        compute_batched_rows::<PROFILE>(
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
            token_tile,
        )?;
        Vec::new()
    };
    if PROFILE {
        profile.as_mut().unwrap().workers = worker_records;
    }

    let (_, output_layout_ns) = measured::<PROFILE, _>(|| {
        for row in 0..rows {
            for batch_idx in 0..batch {
                out[batch_idx * rows + row] = row_major_output[row * batch + batch_idx];
            }
        }
    });
    if PROFILE {
        profile.as_mut().unwrap().output_layout_ns = output_layout_ns;
    }
    Ok(finish_profile::<PROFILE>(profile, call_started))
}

#[allow(clippy::too_many_arguments)]
fn compute_batched_rows<const PROFILE: bool>(
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
    token_tile: Option<mivi_core::simd::ProjectionTokenTile>,
) -> Result<WorkerWorkInternal> {
    let mut work = WorkerWorkInternal {
        rows: row_end - row_start,
        ..WorkerWorkInternal::default()
    };
    if accumulation_mode == BatchAccumulationMode::AcrossBatchFma && batch >= 32 {
        let (mut decoded, scratch_init_ns) = measured::<PROFILE, _>(|| vec![0.0f32; 2 * cols]);
        add_elapsed::<PROFILE>(&mut work.scratch_init_ns, scratch_init_ns);
        for (pair_idx, output) in row_major_output.chunks_mut(2 * batch).enumerate() {
            let row_idx = row_start + pair_idx * 2;
            let weight_start = row_idx * row_bytes;
            let (first, second) = decoded.split_at_mut(cols);
            let (decode_result, decode_ns) = measured::<PROFILE, _>(|| {
                dequantize_slice(
                    ggml_type,
                    &weights[weight_start..weight_start + row_bytes],
                    first,
                )
            });
            decode_result?;
            add_elapsed::<PROFILE>(&mut work.decode_ns, decode_ns);
            let (_, zero_ns) = measured::<PROFILE, _>(|| output.fill(0.0));
            add_elapsed::<PROFILE>(&mut work.zero_copy_ns, zero_ns);
            if row_idx + 1 < row_end {
                let weight_start = weight_start + row_bytes;
                let (decode_result, decode_ns) = measured::<PROFILE, _>(|| {
                    dequantize_slice(
                        ggml_type,
                        &weights[weight_start..weight_start + row_bytes],
                        second,
                    )
                });
                decode_result?;
                add_elapsed::<PROFILE>(&mut work.decode_ns, decode_ns);
                let (out0, out1) = output.split_at_mut(batch);
                let (_, accumulate_ns) = measured::<PROFILE, _>(|| match token_tile {
                    Some(tile) => {
                        mivi_core::simd::matmul_accumulate_transposed_pair_with_tile_simd(
                            out0,
                            out1,
                            first,
                            second,
                            transposed_inputs,
                            batch,
                            cols,
                            tile,
                        )
                    }
                    None => mivi_core::simd::matmul_accumulate_transposed_pair_simd(
                        out0,
                        out1,
                        first,
                        second,
                        transposed_inputs,
                        batch,
                        cols,
                    ),
                });
                add_elapsed::<PROFILE>(&mut work.accumulate_ns, accumulate_ns);
            } else {
                let (_, accumulate_ns) = measured::<PROFILE, _>(|| match token_tile {
                    Some(tile) => mivi_core::simd::matmul_accumulate_transposed_with_tile_simd(
                        output,
                        first,
                        transposed_inputs,
                        batch,
                        cols,
                        tile,
                    ),
                    None => mivi_core::simd::matmul_accumulate_transposed_simd(
                        output,
                        first,
                        transposed_inputs,
                        batch,
                        cols,
                    ),
                });
                add_elapsed::<PROFILE>(&mut work.accumulate_ns, accumulate_ns);
            }
        }
        return Ok(work);
    }
    let (mut decoded_row, decoded_init_ns) = measured::<PROFILE, _>(|| vec![0.0f32; cols]);
    let (mut row_output, output_init_ns) = measured::<PROFILE, _>(|| vec![0.0f32; batch]);
    add_elapsed::<PROFILE>(&mut work.scratch_init_ns, decoded_init_ns);
    add_elapsed::<PROFILE>(&mut work.scratch_init_ns, output_init_ns);
    for row_idx in row_start..row_end {
        let weight_start = row_idx * row_bytes;
        let (_, zero_ns) = measured::<PROFILE, _>(|| row_output.fill(0.0));
        add_elapsed::<PROFILE>(&mut work.zero_copy_ns, zero_ns);
        let (decode_result, decode_ns) = measured::<PROFILE, _>(|| {
            dequantize_slice(
                ggml_type,
                &weights[weight_start..weight_start + row_bytes],
                &mut decoded_row,
            )
        });
        decode_result?;
        add_elapsed::<PROFILE>(&mut work.decode_ns, decode_ns);
        let (_, accumulate_ns) = measured::<PROFILE, _>(|| match accumulation_mode {
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
                match token_tile {
                    Some(tile) if batch >= 32 => {
                        mivi_core::simd::matmul_accumulate_transposed_with_tile_simd(
                            &mut row_output,
                            &decoded_row,
                            transposed_inputs,
                            batch,
                            cols,
                            tile,
                        )
                    }
                    None => mivi_core::simd::matmul_accumulate_transposed_simd(
                        &mut row_output,
                        &decoded_row,
                        transposed_inputs,
                        batch,
                        cols,
                    ),
                    Some(_) => mivi_core::simd::matmul_accumulate_transposed_simd(
                        &mut row_output,
                        &decoded_row,
                        transposed_inputs,
                        batch,
                        cols,
                    ),
                };
            }
        });
        add_elapsed::<PROFILE>(&mut work.accumulate_ns, accumulate_ns);
        let output_start = (row_idx - row_start) * batch;
        let (_, copy_ns) = measured::<PROFILE, _>(|| {
            row_major_output[output_start..output_start + batch].copy_from_slice(&row_output)
        });
        add_elapsed::<PROFILE>(&mut work.zero_copy_ns, copy_ns);
    }
    Ok(work)
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
    fn large_nonzero_quantized_batches_match_matvec() {
        let cols = 512;
        let rows = RAYON_PARALLEL_THRESHOLD + 1;
        for ggml_type in [GgmlType::Q4_K, GgmlType::Q8_0] {
            let block_bytes = ggml_type.type_size().unwrap();
            let mut weights = vec![0u8; rows * row_bytes(ggml_type, cols)];
            for (i, block) in weights.chunks_exact_mut(block_bytes).enumerate() {
                for (j, byte) in block.iter_mut().enumerate() {
                    *byte = ((i + j * 7) % 127) as u8;
                }
                block[..2].copy_from_slice(&half::f16::from_f32(0.125).to_le_bytes());
                if ggml_type == GgmlType::Q4_K {
                    block[2..4].copy_from_slice(&half::f16::from_f32(0.0625).to_le_bytes());
                }
            }
            for batch in [9, 32, 33, 64, 65] {
                let inputs: Vec<f32> = (0..batch * cols)
                    .map(|i| (i % 23) as f32 * 0.0625 - 0.75)
                    .collect();
                let mut actual = vec![0.0; batch * rows];
                quantized_matmul_rows(&mut actual, ggml_type, &weights, &inputs, batch, rows, cols)
                    .unwrap();
                let mut expected = vec![0.0; rows];
                for b in 0..batch {
                    quantized_matvec(
                        &mut expected,
                        ggml_type,
                        &weights,
                        &inputs[b * cols..(b + 1) * cols],
                        rows,
                        cols,
                    )
                    .unwrap();
                    for row in 0..rows {
                        assert!(
                            (actual[b * rows + row] - expected[row]).abs() < 1e-3,
                            "type={ggml_type:?}, batch={batch}, row={row}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn paired_batches_preserve_decoded_reference_for_all_formats() {
        let cols = 256;
        for ggml_type in [
            GgmlType::F32,
            GgmlType::F16,
            GgmlType::BF16,
            GgmlType::Q4_K,
            GgmlType::Q6_K,
            GgmlType::Q8_0,
        ] {
            // Three rows cover the serial odd-row tail; 257 also creates an
            // odd Rayon partition when the test pool has two workers.
            for rows in [3, RAYON_PARALLEL_THRESHOLD + 1] {
                let mut weights = vec![0u8; rows * row_bytes(ggml_type, cols)];
                match ggml_type {
                    GgmlType::F32 => {
                        for (i, value) in weights.chunks_exact_mut(4).enumerate() {
                            value.copy_from_slice(&((i % 13) as f32 * 0.125 - 0.75).to_le_bytes());
                        }
                    }
                    GgmlType::F16 | GgmlType::BF16 => {
                        for (i, value) in weights.chunks_exact_mut(2).enumerate() {
                            let f = (i % 13) as f32 * 0.125 - 0.75;
                            let bytes = if ggml_type == GgmlType::F16 {
                                half::f16::from_f32(f).to_le_bytes()
                            } else {
                                half::bf16::from_f32(f).to_le_bytes()
                            };
                            value.copy_from_slice(&bytes);
                        }
                    }
                    _ => {
                        for (i, block) in weights
                            .chunks_exact_mut(ggml_type.type_size().unwrap())
                            .enumerate()
                        {
                            for (j, byte) in block.iter_mut().enumerate() {
                                *byte = ((i + 7 * j) % 127) as u8;
                            }
                            let scale_start = if ggml_type == GgmlType::Q6_K {
                                block.len() - 2
                            } else {
                                0
                            };
                            block[scale_start..scale_start + 2]
                                .copy_from_slice(&half::f16::from_f32(0.125).to_le_bytes());
                            if ggml_type == GgmlType::Q4_K {
                                block[2..4]
                                    .copy_from_slice(&half::f16::from_f32(0.0625).to_le_bytes());
                            }
                        }
                    }
                }
                for batch in [32, 33, 64, 65] {
                    let inputs: Vec<f32> = (0..batch * cols)
                        .map(|i| (i % 23) as f32 * 0.0625 - 0.75)
                        .collect();
                    let mut transposed = vec![0.0; batch * cols];
                    for b in 0..batch {
                        for col in 0..cols {
                            transposed[col * batch + b] = inputs[b * cols + col];
                        }
                    }
                    let mut actual = vec![0.0; batch * rows + 1];
                    actual[batch * rows] = 123.0;
                    quantized_matmul_rows(
                        &mut actual,
                        ggml_type,
                        &weights,
                        &inputs,
                        batch,
                        rows,
                        cols,
                    )
                    .unwrap();
                    let mut decoded = vec![0.0; cols];
                    let mut reference = vec![0.0; batch];
                    for row in 0..rows {
                        let start = row * row_bytes(ggml_type, cols);
                        dequantize_slice(
                            ggml_type,
                            &weights[start..start + row_bytes(ggml_type, cols)],
                            &mut decoded,
                        )
                        .unwrap();
                        reference.fill(0.0);
                        mivi_core::simd::matmul_accumulate_transposed_simd(
                            &mut reference,
                            &decoded,
                            &transposed,
                            batch,
                            cols,
                        );
                        for b in 0..batch {
                            assert_eq!(
                                actual[b * rows + row],
                                reference[b],
                                "type={ggml_type:?}, rows={rows}, batch={batch}, row={row}"
                            );
                        }
                    }
                    assert_eq!(actual[batch * rows], 123.0);
                }
            }
        }
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
