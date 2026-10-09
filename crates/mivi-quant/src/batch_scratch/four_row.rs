//! Explicit four-row experiment; the scratch-only control remains untouched.

use super::{buffer, product, require, WorkerScratch};
use super::{BatchProjectionScratch, GgmlType, QuantError, Result};
use rayon::prelude::*;

#[cfg(test)]
mod faithful_diagnostics;
#[cfg(test)]
mod measurement;

impl BatchProjectionScratch {
    /// Separate constructor retains the scratch-only control's two-row footprint.
    pub fn new_four_row(batch: usize, rows: usize, cols: usize, workers: usize) -> Result<Self> {
        require("workers", 1, workers)?;
        let input_len = product(batch, cols)?;
        let output_len = product(batch, rows)?;
        let decoded_len = product(4, cols)?;
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

/// Opt-in faithful four-row projection. Shapes, formats, external buffers and
/// active worker/decoded capacities are checked before any writes. No growth.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_rows_four_row(
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
    let decoded_rows = if batch >= 32 && chunk_rows >= 4 { 4 } else { 2 };
    let decoded_len = product(decoded_rows, cols)?;
    for worker in &scratch.workers[..partitions] {
        require("decoded elements", decoded_len, worker.decoded.len())?;
    }
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
    if batch < 32 || output.len() / batch < 4 {
        return super::compute(
            output, row_start, batch, cols, row_bytes, kind, weights, inputs, transposed, worker,
        );
    }
    let group_len = product(4, batch)?;
    let full_len = (output.len() / group_len) * group_len;
    let (full, tail) = output.split_at_mut(full_len);
    let decoded = &mut worker.decoded[..product(4, cols)?];
    for (group, out) in full.chunks_exact_mut(group_len).enumerate() {
        let row = row_start + group * 4;
        for slot in 0..4 {
            let start = (row + slot) * row_bytes;
            crate::dequantize_slice(
                kind,
                &weights[start..start + row_bytes],
                &mut decoded[slot * cols..(slot + 1) * cols],
            )?;
        }
        out.fill(0.0);
        mivi_core::simd::four_row::accumulate(out, decoded, transposed, batch, cols);
    }
    if !tail.is_empty() {
        super::compute(
            tail,
            row_start + full_len / batch,
            batch,
            cols,
            row_bytes,
            kind,
            weights,
            inputs,
            transposed,
            worker,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::tests::{contents, poison, storage, weights};
    use super::*;

    #[test]
    fn four_row_projection_preserves_all_baseline_bits() {
        for threads in [1, 2] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut workspace = BatchProjectionScratch::new_four_row(65, 259, 512, threads).unwrap();
                let allocations = storage(&workspace);
                poison(&mut workspace);
                for kind in [GgmlType::Q4_K, GgmlType::Q6_K, GgmlType::Q8_0, GgmlType::F16, GgmlType::BF16, GgmlType::F32] {
                    for cols in [0, 256, 512] {
                        for rows in [0, 1, 3, 4, 7, 257, 258, 259] {
                            let w = weights(kind, rows, cols);
                            for batch in [0, 1, 2, 8, 9, 31, 32, 33, 63, 64, 65] {
                                let x = (0..batch*cols).map(|i| ((i % 31) as f32 - 15.0) / 19.0).collect::<Vec<_>>();
                                let mut old = vec![17.0; batch*rows+3];
                                let mut new = old.clone();
                                crate::quantized_matmul_rows(&mut old, kind, &w, &x, batch, rows, cols).unwrap();
                                quantized_matmul_rows_four_row(&mut new, kind, &w, &x, batch, rows, cols, &mut workspace).unwrap();
                                assert_eq!(old.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), new.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "{kind:?} batch={batch} rows={rows} cols={cols} threads={threads}");
                                assert_eq!(storage(&workspace), allocations);
                            }
                        }
                    }
                }
            });
        }
    }

    #[test]
    fn four_row_constructor_checks_shape_and_byte_overflow() {
        for shape in [
            (usize::MAX, 2, 2, 1),
            (0, 0, usize::MAX / 2, 1),
            (0, 0, 0, usize::MAX),
            (isize::MAX as usize / 4 + 1, 1, 0, 1),
        ] {
            assert!(matches!(
                BatchProjectionScratch::new_four_row(shape.0, shape.1, shape.2, shape.3),
                Err(QuantError::ArithmeticOverflow)
            ));
        }
        assert!(matches!(
            BatchProjectionScratch::new_four_row(1, 1, 1, 0),
            Err(QuantError::ScratchCapacity {
                dimension: "workers",
                ..
            })
        ));
    }

    #[test]
    fn four_row_projection_rejects_short_decoded_workspace_before_writes() {
        let mut s = BatchProjectionScratch::new(32, 4, 256, 2).unwrap();
        poison(&mut s);
        let saved = contents(&s);
        let mut out = vec![17.0; 32 * 4];
        let w = weights(GgmlType::Q4_K, 4, 256);
        let x = vec![0.5; 32 * 256];
        assert!(matches!(
            quantized_matmul_rows_four_row(&mut out, GgmlType::Q4_K, &w, &x, 32, 4, 256, &mut s),
            Err(QuantError::ScratchCapacity {
                dimension: "decoded elements",
                required: 1024,
                available: 512
            })
        ));
        assert!(out.iter().all(|v| *v == 17.0));
        assert_eq!(contents(&s), saved);
    }

    #[test]
    fn four_row_projection_invalid_buffers_and_capacities_preserve_all_bits() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        pool.install(|| {
            let mut s = BatchProjectionScratch::new_four_row(32, 257, 256, 1).unwrap();
            poison(&mut s);
            let saved = contents(&s);
            let w = weights(GgmlType::Q4_K, 258, 512);
            let x = vec![0.5; 33 * 512];
            let mut out = vec![17.0; 33 * 258];
            let cases = [
                (GgmlType::Q4_K, w.len(), x.len(), 32, 4, 256, 127),
                (GgmlType::Q4_K, 575, x.len(), 32, 4, 256, out.len()),
                (GgmlType::Q4_K, w.len(), 8191, 32, 4, 256, out.len()),
                (GgmlType::Q4_0, w.len(), x.len(), 32, 4, 256, out.len()),
                (GgmlType::Q4_K, w.len(), x.len(), 32, 4, 255, out.len()),
                (GgmlType::Q4_K, w.len(), x.len(), 33, 4, 256, out.len()),
                (GgmlType::Q4_K, w.len(), x.len(), 32, 258, 256, out.len()),
                (GgmlType::Q4_K, w.len(), x.len(), 32, 4, 512, out.len()),
                (GgmlType::Q4_K, w.len(), x.len(), 32, 257, 256, out.len()),
                (GgmlType::F32, 0, 0, usize::MAX, 2, 0, out.len()),
            ];
            for (kind, wn, xn, batch, rows, cols, on) in cases {
                assert!(quantized_matmul_rows_four_row(
                    &mut out[..on],
                    kind,
                    &w[..wn],
                    &x[..xn],
                    batch,
                    rows,
                    cols,
                    &mut s
                )
                .is_err());
                assert!(out.iter().all(|v| *v == 17.0));
                assert_eq!(contents(&s), saved);
            }
        });
    }

    #[test]
    fn four_row_delegate_and_nonblock_float_columns_match_baseline() {
        let mut empty = BatchProjectionScratch::new_four_row(0, 0, 0, 1).unwrap();
        let w = weights(GgmlType::F32, 7, 7);
        let mut padded = vec![0u8; w.len() + 1];
        padded[1..].copy_from_slice(&w);
        let x = vec![0.25; 7];
        let mut old = vec![17.0; 10];
        let mut new = old.clone();
        crate::quantized_matmul_rows(&mut old, GgmlType::F32, &padded[1..], &x, 1, 7, 7).unwrap();
        quantized_matmul_rows_four_row(
            &mut new,
            GgmlType::F32,
            &padded[1..],
            &x,
            1,
            7,
            7,
            &mut empty,
        )
        .unwrap();
        assert_eq!(old, new);
        assert!(quantized_matmul_rows_four_row(
            &mut new,
            GgmlType::Q4_K,
            &[],
            &[],
            0,
            1,
            256,
            &mut empty
        )
        .is_err());
        assert!(quantized_matmul_rows_four_row(
            &mut new,
            GgmlType::Q4_K,
            &[],
            &[],
            1,
            0,
            256,
            &mut empty
        )
        .is_err());
        for kind in [GgmlType::F16, GgmlType::BF16, GgmlType::F32] {
            let w = weights(kind, 7, 7);
            let x = vec![0.25; 65 * 7];
            let mut s = BatchProjectionScratch::new_four_row(65, 7, 7, 2).unwrap();
            for batch in [2, 9, 31, 32, 33, 63, 64, 65] {
                let mut old = vec![17.0; batch * 7 + 1];
                let mut new = old.clone();
                crate::quantized_matmul_rows(&mut old, kind, &w, &x, batch, 7, 7).unwrap();
                quantized_matmul_rows_four_row(&mut new, kind, &w, &x, batch, 7, 7, &mut s)
                    .unwrap();
                assert_eq!(
                    old.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    new.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    }
}
