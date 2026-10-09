//! Explicit reusable workspace experiment; ordinary batch dispatch is unchanged.

use crate::{GgmlType, QuantError, Result};
use rayon::prelude::*;

/// Caller-owned capacity, allocated once. Not used by default inference dispatch.
pub struct BatchProjectionScratch {
    batch: usize,
    rows: usize,
    cols: usize,
    transposed: Vec<f32>,
    output: Vec<f32>,
    workers: Vec<WorkerScratch>,
}

struct WorkerScratch {
    decoded: Vec<f32>,
    output: Vec<f32>,
}

fn require(dimension: &'static str, required: usize, available: usize) -> Result<()> {
    if required > available {
        return Err(QuantError::ScratchCapacity {
            dimension,
            required,
            available,
        });
    }
    Ok(())
}

fn product(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or(QuantError::ArithmeticOverflow)
}

fn buffer(len: usize) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| QuantError::ScratchAllocationFailed)?;
    values.resize(len, 0.0);
    Ok(values)
}

impl BatchProjectionScratch {
    /// Maximum shape and worker partitions. Returns errors rather than growing
    /// during calls. Reserve enough partitions for the largest intended Rayon pool.
    pub fn new(batch: usize, rows: usize, cols: usize, workers: usize) -> Result<Self> {
        require("workers", 1, workers)?;
        let input_len = product(batch, cols)?;
        let output_len = product(batch, rows)?;
        let decoded_len = product(2, cols)?;
        let per_worker = decoded_len
            .checked_add(batch)
            .ok_or(QuantError::ArithmeticOverflow)?;
        let floats = input_len
            .checked_add(output_len)
            .and_then(|n| n.checked_add(per_worker.checked_mul(workers)?))
            .ok_or(QuantError::ArithmeticOverflow)?;
        let bytes = product(floats, std::mem::size_of::<f32>())?
            .checked_add(product(workers, std::mem::size_of::<WorkerScratch>())?)
            .ok_or(QuantError::ArithmeticOverflow)?;
        if bytes > isize::MAX as usize {
            return Err(QuantError::ArithmeticOverflow);
        }
        let transposed = buffer(input_len)?;
        let output = buffer(output_len)?;
        let mut worker_buffers = Vec::new();
        worker_buffers
            .try_reserve_exact(workers)
            .map_err(|_| QuantError::ScratchAllocationFailed)?;
        for _ in 0..workers {
            worker_buffers.push(WorkerScratch {
                decoded: buffer(decoded_len)?,
                output: buffer(batch)?,
            });
        }
        Ok(Self {
            batch,
            rows,
            cols,
            transposed,
            output,
            workers: worker_buffers,
        })
    }
}

/// Same arithmetic as `quantized_matmul_rows`, with reusable projection buffers.
/// Invalid buffers/capacities fail before writes. Empty and batch-one operations
/// require no workspace capacity; batch one delegates to the established matvec.
/// This avoids projection-owned allocations for batch > 1, not allocations in
/// Rayon internals or delegated matvec implementations.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_rows_with_scratch(
    out: &mut [f32],
    kind: GgmlType,
    weights: &[u8],
    inputs: &[f32],
    batch: usize,
    rows: usize,
    cols: usize,
    scratch: &mut BatchProjectionScratch,
) -> Result<()> {
    if !matches!(
        kind,
        GgmlType::Q4_K
            | GgmlType::Q6_K
            | GgmlType::Q8_0
            | GgmlType::F16
            | GgmlType::BF16
            | GgmlType::F32
    ) {
        return Err(QuantError::UnsupportedType(kind as u32));
    }
    let block = kind.block_size_checked()?;
    if !cols.is_multiple_of(block) {
        return Err(QuantError::DimensionMisaligned {
            dim: cols,
            block_size: block,
        });
    }
    let row_bytes = product(cols / block, kind.type_size_checked()?)?;
    crate::validate_matmul_args(out, weights, inputs, batch, rows, cols, row_bytes, block)?;
    if batch == 0 || rows == 0 {
        return Ok(());
    }
    if batch == 1 {
        return crate::quantized_matvec(
            &mut out[..rows],
            kind,
            weights,
            &inputs[..cols],
            rows,
            cols,
        );
    }
    let threads = rayon::current_num_threads().max(1);
    let parallel = rows >= crate::RAYON_PARALLEL_THRESHOLD && threads > 1;
    let chunk_rows = if parallel {
        rows.div_ceil(threads)
    } else {
        rows
    };
    let partitions = rows.div_ceil(chunk_rows);
    require("batch", batch, scratch.batch)?;
    require("rows", rows, scratch.rows)?;
    require("columns", cols, scratch.cols)?;
    require("workers", partitions, scratch.workers.len())?;

    let transposed = &mut scratch.transposed[..batch * cols];
    if batch > 8 {
        for b in 0..batch {
            for col in 0..cols {
                transposed[col * batch + b] = inputs[b * cols + col];
            }
        }
    }
    let output = &mut scratch.output[..batch * rows];
    let workers = &mut scratch.workers[..partitions];
    if parallel {
        output
            .par_chunks_mut(chunk_rows * batch)
            .zip(workers.par_iter_mut())
            .enumerate()
            .try_for_each(|(idx, (output, worker))| {
                compute(
                    output,
                    idx * chunk_rows,
                    batch,
                    cols,
                    row_bytes,
                    kind,
                    weights,
                    inputs,
                    transposed,
                    worker,
                )
            })?;
    } else {
        compute(
            output,
            0,
            batch,
            cols,
            row_bytes,
            kind,
            weights,
            inputs,
            transposed,
            &mut workers[0],
        )?;
    }
    for row in 0..rows {
        for b in 0..batch {
            out[b * rows + row] = output[row * batch + b];
        }
    }
    Ok(())
}

// Deliberately mirror the baseline's branch/operation order. Sharing a refactored
// baseline helper would change the control being measured in the next experiment.
#[allow(clippy::too_many_arguments)]
fn compute(
    output: &mut [f32],
    row_start: usize,
    batch: usize,
    cols: usize,
    row_bytes: usize,
    kind: GgmlType,
    weights: &[u8],
    inputs: &[f32],
    transposed: &[f32],
    worker: &mut WorkerScratch,
) -> Result<()> {
    if batch >= 32 {
        let (first, second) = worker.decoded[..2 * cols].split_at_mut(cols);
        for (pair, out) in output.chunks_mut(2 * batch).enumerate() {
            let start = (row_start + pair * 2) * row_bytes;
            crate::dequantize_slice(kind, &weights[start..start + row_bytes], first)?;
            out.fill(0.0);
            if out.len() == 2 * batch {
                crate::dequantize_slice(
                    kind,
                    &weights[start + row_bytes..start + 2 * row_bytes],
                    second,
                )?;
                let (out0, out1) = out.split_at_mut(batch);
                mivi_core::simd::matmul_accumulate_transposed_pair_simd(
                    out0, out1, first, second, transposed, batch, cols,
                );
            } else {
                mivi_core::simd::matmul_accumulate_transposed_simd(
                    out, first, transposed, batch, cols,
                );
            }
        }
    } else {
        let decoded = &mut worker.decoded[..cols];
        let row_output = &mut worker.output[..batch];
        for (row, out) in output.chunks_exact_mut(batch).enumerate() {
            let start = (row_start + row) * row_bytes;
            crate::dequantize_slice(kind, &weights[start..start + row_bytes], decoded)?;
            row_output.fill(0.0);
            if batch <= 8 {
                for (b, value) in row_output.iter_mut().enumerate() {
                    *value = mivi_core::simd::dot_product_simd(
                        decoded,
                        &inputs[b * cols..(b + 1) * cols],
                    );
                }
            } else {
                mivi_core::simd::matmul_accumulate_transposed_simd(
                    row_output, decoded, transposed, batch, cols,
                );
            }
            out.copy_from_slice(row_output);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weights(kind: GgmlType, rows: usize, cols: usize) -> Vec<u8> {
        let block = kind.block_size_checked().unwrap();
        let size = kind.type_size_checked().unwrap();
        let mut bytes = vec![0u8; rows * (cols / block) * size];
        for (i, part) in bytes.chunks_exact_mut(size).enumerate() {
            match kind {
                GgmlType::F32 => {
                    part.copy_from_slice(&(((i % 23) as f32 - 11.0) / 13.0).to_le_bytes())
                }
                GgmlType::F16 => part.copy_from_slice(
                    &half::f16::from_f32(((i % 23) as f32 - 11.0) / 13.0).to_le_bytes(),
                ),
                GgmlType::BF16 => part.copy_from_slice(
                    &half::bf16::from_f32(((i % 23) as f32 - 11.0) / 13.0).to_le_bytes(),
                ),
                _ => {
                    for (j, byte) in part.iter_mut().enumerate() {
                        *byte = (i * 37 + j * 13 + 7) as u8;
                    }
                    let scale = half::f16::from_f32(0.03125).to_le_bytes();
                    if kind == GgmlType::Q6_K {
                        part[size - 2..].copy_from_slice(&scale);
                    } else {
                        part[..2].copy_from_slice(&scale);
                    }
                    if kind == GgmlType::Q4_K {
                        part[2..4].copy_from_slice(&scale);
                    }
                }
            }
        }
        bytes
    }

    fn storage(s: &BatchProjectionScratch) -> Vec<(usize, usize, usize)> {
        std::iter::once(&s.transposed)
            .chain(std::iter::once(&s.output))
            .chain(s.workers.iter().flat_map(|w| [&w.decoded, &w.output]))
            .map(|v| (v.as_ptr() as usize, v.len(), v.capacity()))
            .collect()
    }

    fn contents(s: &BatchProjectionScratch) -> Vec<u32> {
        s.transposed
            .iter()
            .chain(&s.output)
            .chain(
                s.workers
                    .iter()
                    .flat_map(|w| w.decoded.iter().chain(&w.output)),
            )
            .map(|v| v.to_bits())
            .collect()
    }

    fn poison(s: &mut BatchProjectionScratch) {
        s.transposed.fill(f32::NAN);
        s.output.fill(f32::NAN);
        for w in &mut s.workers {
            w.decoded.fill(f32::NAN);
            w.output.fill(f32::NAN);
        }
    }

    #[test]
    fn scratch_rejects_invalid_arguments_before_writes() {
        let mut s = BatchProjectionScratch::new(64, 3, 256, 2).unwrap();
        poison(&mut s);
        let saved = contents(&s);
        let w = weights(GgmlType::Q4_K, 3, 256);
        let x = vec![0.5; 65 * 512];
        let mut out = vec![17.0; 65 * 4];
        let saved_out = out.clone();
        // Output/input/weight length, unsupported format, block alignment,
        // each shape capacity, and checked arithmetic are distinct controls.
        for (kind, bytes, input, batch, rows, cols, out_len) in [
            (GgmlType::Q4_K, w.as_slice(), x.as_slice(), 64, 3, 256, 191),
            (
                GgmlType::Q4_K,
                &w[..w.len() - 1],
                x.as_slice(),
                64,
                3,
                256,
                192,
            ),
            (GgmlType::Q4_K, w.as_slice(), &x[..16383], 64, 3, 256, 192),
            (GgmlType::Q4_0, w.as_slice(), x.as_slice(), 64, 3, 256, 192),
            (GgmlType::Q4_K, w.as_slice(), x.as_slice(), 64, 3, 255, 192),
            (GgmlType::Q4_K, w.as_slice(), x.as_slice(), 65, 3, 256, 195),
            (GgmlType::F32, &[], &[], usize::MAX, 2, 0, 192),
        ] {
            assert!(quantized_matmul_rows_with_scratch(
                &mut out[..out_len],
                kind,
                bytes,
                input,
                batch,
                rows,
                cols,
                &mut s
            )
            .is_err());
            assert_eq!(out, saved_out);
            assert_eq!(contents(&s), saved);
        }
        for (rows, cols, dimension) in [(4, 256, "rows"), (3, 512, "columns")] {
            let w = weights(GgmlType::Q4_K, rows, cols);
            let err = quantized_matmul_rows_with_scratch(
                &mut out,
                GgmlType::Q4_K,
                &w,
                &x,
                64,
                rows,
                cols,
                &mut s,
            )
            .unwrap_err();
            assert!(
                matches!(err, QuantError::ScratchCapacity { dimension: d, .. } if d == dimension)
            );
            assert_eq!(out, saved_out);
            assert_eq!(contents(&s), saved);
        }
    }

    #[test]
    fn scratch_rejects_larger_pool_before_writes() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        pool.install(|| {
            let mut s = BatchProjectionScratch::new(32, 257, 256, 1).unwrap();
            poison(&mut s);
            let saved = contents(&s);
            let mut out = vec![17.0; 32 * 257];
            let w = weights(GgmlType::Q8_0, 257, 256);
            let x = vec![0.5; 32 * 256];
            assert!(matches!(
                quantized_matmul_rows_with_scratch(
                    &mut out,
                    GgmlType::Q8_0,
                    &w,
                    &x,
                    32,
                    257,
                    256,
                    &mut s
                ),
                Err(QuantError::ScratchCapacity {
                    dimension: "workers",
                    required: 2,
                    available: 1
                })
            ));
            assert!(out.iter().all(|v| *v == 17.0));
            assert_eq!(contents(&s), saved);
        });
    }

    #[test]
    fn scratch_constructor_checks_counts_and_byte_overflow() {
        for dims in [
            (usize::MAX, 2, 2, 1),
            (0, 0, usize::MAX, 1),
            (0, 0, 0, usize::MAX),
            (isize::MAX as usize / 4 + 1, 1, 0, 1),
        ] {
            assert!(matches!(
                BatchProjectionScratch::new(dims.0, dims.1, dims.2, dims.3),
                Err(QuantError::ArithmeticOverflow)
            ));
        }
        assert!(matches!(
            BatchProjectionScratch::new(1, 1, 1, 0),
            Err(QuantError::ScratchCapacity {
                dimension: "workers",
                ..
            })
        ));
    }

    #[test]
    fn scratch_empty_and_matvec_preserve_delegate_contract() {
        let mut s = BatchProjectionScratch::new(0, 0, 0, 1).unwrap();
        let w = weights(GgmlType::F32, 3, 7);
        let mut padded = vec![0u8; w.len() + 1];
        padded[1..].copy_from_slice(&w);
        let x = vec![0.25; 7];
        for weights in [w.as_slice(), &padded[1..]] {
            let mut old = vec![17.0; 4];
            let mut new = old.clone();
            crate::quantized_matmul_rows(&mut old, GgmlType::F32, weights, &x, 1, 3, 7).unwrap();
            quantized_matmul_rows_with_scratch(
                &mut new,
                GgmlType::F32,
                weights,
                &x,
                1,
                3,
                7,
                &mut s,
            )
            .unwrap();
            assert_eq!(old, new);
        }
        let mut out = [17.0];
        quantized_matmul_rows_with_scratch(&mut out, GgmlType::Q4_K, &[], &[], 0, 0, 256, &mut s)
            .unwrap();
        // Empty output does not exempt invalid external weight/input buffers.
        assert!(quantized_matmul_rows_with_scratch(
            &mut out,
            GgmlType::Q4_K,
            &[],
            &[],
            0,
            1,
            256,
            &mut s
        )
        .is_err());
        assert!(quantized_matmul_rows_with_scratch(
            &mut out,
            GgmlType::Q4_K,
            &[],
            &[],
            1,
            0,
            256,
            &mut s
        )
        .is_err());
        assert_eq!(out, [17.0]);
    }

    #[test]
    fn scratch_zero_columns_clear_active_output() {
        let mut s = BatchProjectionScratch::new(65, 257, 0, 2).unwrap();
        for batch in [2, 9, 32, 65] {
            let mut out = vec![17.0; batch * 257 + 1];
            quantized_matmul_rows_with_scratch(
                &mut out,
                GgmlType::Q4_K,
                &[],
                &[],
                batch,
                257,
                0,
                &mut s,
            )
            .unwrap();
            assert!(out[..batch * 257]
                .iter()
                .all(|v| v.to_bits() == 0.0f32.to_bits()));
            assert_eq!(out[batch * 257], 17.0);
        }
    }

    #[test]
    fn scratch_matches_established_batch_bits() {
        for threads in [1, 2] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut scratch = BatchProjectionScratch::new(65, 257, 512, threads).unwrap();
                let allocations = storage(&scratch);
                poison(&mut scratch);
                for kind in [GgmlType::Q4_K, GgmlType::Q6_K, GgmlType::Q8_0, GgmlType::F16, GgmlType::BF16, GgmlType::F32] {
                    for cols in [256, 512] {
                        for rows in [0, 1, 3, 257] {
                            let w = weights(kind, rows, cols);
                            for batch in [0, 1, 2, 8, 9, 32, 64, 65] {
                                let x: Vec<_> = (0..batch*cols).map(|i| ((i % 31) as f32 - 15.0) / 19.0).collect();
                                let mut old = vec![123.0; batch * rows + 3];
                                let mut new = old.clone();
                                crate::quantized_matmul_rows(&mut old, kind, &w, &x, batch, rows, cols).unwrap();
                                quantized_matmul_rows_with_scratch(&mut new, kind, &w, &x, batch, rows, cols, &mut scratch).unwrap();
                                assert_eq!(old.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), new.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), "{kind:?} batch={batch} rows={rows} cols={cols} threads={threads}");
                                assert_eq!(storage(&scratch), allocations);
                            }
                        }
                    }
                }
            });
        }
    }
}
