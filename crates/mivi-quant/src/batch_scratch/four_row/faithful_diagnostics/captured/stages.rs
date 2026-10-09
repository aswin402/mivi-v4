//! Test-only stage replay; unchanged quant APIs remain timing controls.

use super::{BatchProjectionScratch, EvalResult, GgmlType};
use crate::batch_scratch::{buffer, product};

fn orders() -> [[usize; 4]; 8] {
    [
        [0, 1, 2, 3],
        [3, 2, 1, 0],
        [1, 2, 3, 0],
        [0, 3, 2, 1],
        [2, 3, 0, 1],
        [1, 0, 3, 2],
        [3, 0, 1, 2],
        [2, 1, 0, 3],
    ]
}

#[test]
#[ignore = "requires local MIVI_TEST_MODEL; bounded exact-gated stage replay"]
fn four_row_stage_replay_measurement() -> EvalResult<()> {
    let (mut model, ids) = super::prepare_capture()?;
    println!("STAGE_CONTRACT routes=S,F,PS,PF calls_per_member=1 profile=replay-not-production cache=warm-resident worker_time=overlapping-not-wall");
    for batch in [32, 64] {
        let captures = super::gated_captures(&mut model, &ids, batch)?;
        for capture in captures {
            let shape = Shape::new(
                GgmlType::from_u32(capture.tensor.quant_type as u32)?,
                batch,
                capture.tensor.rows,
                capture.tensor.cols,
            )?;
            let weights = capture
                .tensor
                .as_slice_checked(&model.gguf.mmap)
                .ok_or("stage weights outside GGUF")?;
            let mut scratch = [
                BatchProjectionScratch::new(batch, shape.rows, shape.cols, 2)?,
                BatchProjectionScratch::new_four_row(batch, shape.rows, shape.cols, 2)?,
                BatchProjectionScratch::new(batch, shape.rows, shape.cols, 2)?,
                BatchProjectionScratch::new_four_row(batch, shape.rows, shape.cols, 2)?,
            ];
            let len = product(batch, shape.rows)?;
            let mut out = [buffer(len)?, buffer(len)?, buffer(len)?, buffer(len)?];
            let mut reference = buffer(batch * shape.rows)?;
            crate::quantized_matmul_rows(
                &mut reference,
                shape.kind,
                weights,
                &capture.inputs,
                batch,
                shape.rows,
                shape.cols,
            )?;
            super::validate_exact(&reference, &reference)?;
            for route in [2, 3] {
                replay(
                    &mut out[route],
                    weights,
                    &capture.inputs,
                    &shape,
                    &mut scratch[route],
                    if route == 2 { 2 } else { 4 },
                )?;
                super::validate_exact(&reference, &out[route])?;
            }
            println!(
                "STAGE_GATE kind={:?} batch={batch} complete_output={} replay_bits=exact",
                shape.kind,
                reference.len()
            );
            for round in 0..=8 {
                let order = if round == 0 {
                    [0, 1, 2, 3]
                } else {
                    orders()[round - 1]
                };
                for route in order {
                    let start = std::time::Instant::now();
                    let output = std::hint::black_box(&mut out[route]);
                    let input = std::hint::black_box(&capture.inputs);
                    let weights = std::hint::black_box(weights);
                    let profile = match route {
                        0 => {
                            super::quantized_matmul_rows_with_scratch(
                                output,
                                shape.kind,
                                weights,
                                input,
                                batch,
                                shape.rows,
                                shape.cols,
                                &mut scratch[route],
                            )?;
                            None
                        }
                        1 => {
                            super::quantized_matmul_rows_four_row(
                                output,
                                shape.kind,
                                weights,
                                input,
                                batch,
                                shape.rows,
                                shape.cols,
                                &mut scratch[route],
                            )?;
                            None
                        }
                        2 | 3 => Some(replay(
                            output,
                            weights,
                            input,
                            &shape,
                            &mut scratch[route],
                            if route == 2 { 2 } else { 4 },
                        )?),
                        _ => unreachable!(),
                    };
                    let wall = ns(start)?;
                    super::validate_exact(&reference, &out[route])?;
                    if let Some(ref p) = profile {
                        let group = if route == 2 { 2 } else { 4 };
                        for w in &p.workers {
                            if (w.rows, w.groups, w.decode_calls, w.helper_calls)
                                != (
                                    shape.chunk_rows,
                                    shape.chunk_rows / group,
                                    shape.chunk_rows,
                                    shape.chunk_rows / group,
                                )
                            {
                                return Err("stage worker counts inconsistent".into());
                            }
                        }
                        if p.call_wall_ns > wall {
                            return Err("stage inner call exceeds outer wall".into());
                        }
                    }
                    println!("STAGE kind={:?} layer={} role={} rows={} cols={} batch={batch} round={round} order={order:?} route={route} wall_ns={wall} profile={profile:?}",
                        shape.kind, capture.layer, capture.role, shape.rows, shape.cols);
                }
            }
        }
    }
    Ok(())
}

#[test]
fn stage_replay_orders_balance_position_and_pair_order() {
    let orders = orders();
    for route in 0..4 {
        for position in 0..4 {
            assert_eq!(
                orders
                    .iter()
                    .filter(|order| order[position] == route)
                    .count(),
                2
            );
        }
        for other in route + 1..4 {
            assert_eq!(
                orders
                    .iter()
                    .filter(|order| {
                        order.iter().position(|r| *r == route).unwrap()
                            < order.iter().position(|r| *r == other).unwrap()
                    })
                    .count(),
                4
            );
        }
    }
}

struct Shape {
    kind: GgmlType,
    batch: usize,
    rows: usize,
    cols: usize,
    chunk_rows: usize,
    row_bytes: usize,
}

impl Shape {
    fn new(kind: GgmlType, batch: usize, rows: usize, cols: usize) -> EvalResult<Self> {
        if !matches!(kind, GgmlType::Q4_K | GgmlType::Q6_K)
            || !matches!(batch, 32 | 64)
            || !(crate::RAYON_PARALLEL_THRESHOLD..=8192).contains(&rows)
            || cols == 0
            || cols > 16384
            || !cols.is_multiple_of(256)
        {
            return Err("unsupported bounded stage shape".into());
        }
        let chunk_rows = rows.div_ceil(2);
        if !chunk_rows.is_multiple_of(4) || !(rows - chunk_rows).is_multiple_of(4) {
            return Err("stage partitions must contain full four-row groups".into());
        }
        let row_bytes = product(cols / 256, kind.type_size_checked()?)?;
        product(rows, row_bytes)?;
        product(batch, cols)?;
        product(batch, rows)?;
        Ok(Self {
            kind,
            batch,
            rows,
            cols,
            chunk_rows,
            row_bytes,
        })
    }
}

fn residual(total: u64, stages: &[u64]) -> EvalResult<u64> {
    let used = stages
        .iter()
        .try_fold(0u64, |a, b| a.checked_add(*b))
        .ok_or("stage wall sum overflow")?;
    total
        .checked_sub(used)
        .ok_or_else(|| "stage wall exceeds call".into())
}

#[derive(Debug, Default, Clone, Copy)]
struct Work {
    rows: usize,
    groups: usize,
    decode_calls: usize,
    helper_calls: usize,
    decode_ns: u64,
    zero_ns: u64,
    accumulate_ns: u64,
}

#[derive(Debug, Default)]
struct Profile {
    validation_ns: u64,
    transpose_ns: u64,
    rows_wall_ns: u64,
    layout_ns: u64,
    call_wall_ns: u64,
    residual_ns: u64,
    workers: [Work; 2],
}

fn replay(
    out: &mut [f32],
    weights: &[u8],
    inputs: &[f32],
    shape: &Shape,
    scratch: &mut BatchProjectionScratch,
    group: usize,
) -> EvalResult<Profile> {
    use rayon::prelude::*;
    let call = std::time::Instant::now();
    let validation = std::time::Instant::now();
    if rayon::current_num_threads() != 2
        || !matches!(group, 2 | 4)
        || out.len() != product(shape.batch, shape.rows)?
        || inputs.len() != product(shape.batch, shape.cols)?
        || weights.len() != product(shape.rows, shape.row_bytes)?
    {
        return Err("invalid stage pool, group or external buffers".into());
    }
    crate::batch_scratch::require("batch", shape.batch, scratch.batch)?;
    crate::batch_scratch::require("rows", shape.rows, scratch.rows)?;
    crate::batch_scratch::require("columns", shape.cols, scratch.cols)?;
    crate::batch_scratch::require("workers", 2, scratch.workers.len())?;
    crate::batch_scratch::require(
        "transposed elements",
        product(shape.batch, shape.cols)?,
        scratch.transposed.len(),
    )?;
    crate::batch_scratch::require(
        "output elements",
        product(shape.batch, shape.rows)?,
        scratch.output.len(),
    )?;
    for w in &scratch.workers[..2] {
        crate::batch_scratch::require(
            "decoded elements",
            product(group, shape.cols)?,
            w.decoded.len(),
        )?;
    }
    let mut profile = Profile {
        validation_ns: ns(validation)?,
        ..Profile::default()
    };
    let start = std::time::Instant::now();
    let transposed = &mut scratch.transposed[..shape.batch * shape.cols];
    for b in 0..shape.batch {
        for col in 0..shape.cols {
            transposed[col * shape.batch + b] = inputs[b * shape.cols + col];
        }
    }
    profile.transpose_ns = ns(start)?;
    let start = std::time::Instant::now();
    let output = &mut scratch.output[..shape.batch * shape.rows];
    output
        .par_chunks_mut(shape.chunk_rows * shape.batch)
        .zip(scratch.workers[..2].par_iter_mut())
        .zip(profile.workers.par_iter_mut())
        .enumerate()
        .try_for_each(|(idx, ((output, worker), report))| -> crate::Result<()> {
            *report = worker_replay(
                output,
                &mut worker.decoded[..group * shape.cols],
                weights,
                transposed,
                shape,
                idx * shape.chunk_rows,
                group,
            )?;
            Ok(())
        })?;
    profile.rows_wall_ns = ns(start)?;
    let start = std::time::Instant::now();
    for row in 0..shape.rows {
        for b in 0..shape.batch {
            out[b * shape.rows + row] = output[row * shape.batch + b];
        }
    }
    profile.layout_ns = ns(start)?;
    profile.call_wall_ns = ns(call)?;
    profile.residual_ns = residual(
        profile.call_wall_ns,
        &[
            profile.validation_ns,
            profile.transpose_ns,
            profile.rows_wall_ns,
            profile.layout_ns,
        ],
    )?;
    Ok(profile)
}

fn ns(start: std::time::Instant) -> crate::Result<u64> {
    u64::try_from(start.elapsed().as_nanos()).map_err(|_| crate::QuantError::ArithmeticOverflow)
}

fn add_time(total: &mut u64, start: std::time::Instant) -> crate::Result<()> {
    *total = total
        .checked_add(ns(start)?)
        .ok_or(crate::QuantError::ArithmeticOverflow)?;
    Ok(())
}

fn worker_replay(
    out: &mut [f32],
    decoded: &mut [f32],
    weights: &[u8],
    inputs: &[f32],
    shape: &Shape,
    row_start: usize,
    group: usize,
) -> crate::Result<Work> {
    let mut work = Work {
        rows: out.len() / shape.batch,
        ..Work::default()
    };
    for (idx, output) in out.chunks_exact_mut(group * shape.batch).enumerate() {
        for slot in 0..group {
            let start = std::time::Instant::now();
            let offset = (row_start + idx * group + slot) * shape.row_bytes;
            crate::dequantize_slice(
                shape.kind,
                &weights[offset..offset + shape.row_bytes],
                &mut decoded[slot * shape.cols..(slot + 1) * shape.cols],
            )?;
            add_time(&mut work.decode_ns, start)?;
            work.decode_calls += 1;
            if (group == 2 && slot == 0) || (group == 4 && slot == 3) {
                let start = std::time::Instant::now();
                output.fill(0.0);
                add_time(&mut work.zero_ns, start)?;
            }
        }
        let start = std::time::Instant::now();
        if group == 4 {
            mivi_core::simd::four_row::accumulate(output, decoded, inputs, shape.batch, shape.cols);
        } else {
            let (a, b) = output.split_at_mut(shape.batch);
            let (w0, w1) = decoded.split_at(shape.cols);
            mivi_core::simd::matmul_accumulate_transposed_pair_simd(
                a,
                b,
                w0,
                w1,
                inputs,
                shape.batch,
                shape.cols,
            );
        }
        add_time(&mut work.accumulate_ns, start)?;
        work.helper_calls += 1;
        work.groups += 1;
    }
    Ok(work)
}

#[test]
fn stage_replay_complete_bits_and_worker_counts() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    pool.install(|| {
        for kind in [GgmlType::Q4_K, GgmlType::Q6_K] {
            let weights = crate::batch_scratch::tests::weights(kind, 256, 256);
            for batch in [32, 64] {
                let shape = Shape::new(kind, batch, 256, 256).unwrap();
                let input: Vec<_> = (0..batch * 256)
                    .map(|i| ((i % 31) as f32 - 15.0) / 19.0)
                    .collect();
                let mut reference = buffer(batch * 256).unwrap();
                crate::quantized_matmul_rows(
                    &mut reference,
                    kind,
                    &weights,
                    &input,
                    batch,
                    256,
                    256,
                )
                .unwrap();
                for group in [2, 4] {
                    let mut scratch =
                        BatchProjectionScratch::new_four_row(64, 256, 256, 2).unwrap();
                    let storage = crate::batch_scratch::tests::storage(&scratch);
                    for _ in 0..2 {
                        crate::batch_scratch::tests::poison(&mut scratch);
                        let mut out = vec![f32::NAN; batch * 256];
                        let report =
                            replay(&mut out, &weights, &input, &shape, &mut scratch, group)
                                .unwrap();
                        super::validate_exact(&reference, &out).unwrap();
                        assert_eq!(crate::batch_scratch::tests::storage(&scratch), storage);
                        for w in report.workers {
                            assert_eq!(
                                (w.rows, w.groups, w.decode_calls, w.helper_calls),
                                (128, 128 / group, 128, 128 / group)
                            );
                        }
                        assert_eq!(
                            report.residual_ns,
                            residual(
                                report.call_wall_ns,
                                &[
                                    report.validation_ns,
                                    report.transpose_ns,
                                    report.rows_wall_ns,
                                    report.layout_ns
                                ]
                            )
                            .unwrap()
                        );
                    }
                }
            }
        }
    });
}

#[test]
fn stage_replay_shape_is_bounded() {
    for kind in [GgmlType::Q4_K, GgmlType::Q6_K] {
        for batch in [32, 64] {
            assert!(Shape::new(kind, batch, 256, 256).is_ok());
        }
    }
    for (kind, batch, rows, cols) in [
        (GgmlType::F32, 32, 256, 256),
        (GgmlType::Q4_K, 31, 256, 256),
        (GgmlType::Q4_K, 32, 0, 256),
        (GgmlType::Q4_K, 32, 248, 256),
        (GgmlType::Q4_K, 32, 260, 256),
        (GgmlType::Q4_K, 32, 8200, 256),
        (GgmlType::Q4_K, 32, 256, 0),
        (GgmlType::Q4_K, 32, 256, 257),
        (GgmlType::Q4_K, 32, 256, 16640),
        (GgmlType::Q4_K, usize::MAX, usize::MAX, usize::MAX),
    ] {
        assert!(Shape::new(kind, batch, rows, cols).is_err());
    }
}

#[test]
fn stage_replay_wall_accounting_checks_overlap_and_overflow() {
    assert_eq!(residual(100, &[10, 20, 30]).unwrap(), 40);
    assert_eq!(residual(0, &[0, 0]).unwrap(), 0);
    assert!(residual(10, &[11]).is_err());
    assert!(residual(u64::MAX, &[u64::MAX, 1]).is_err());
}

#[test]
fn stage_replay_invalid_inputs_preserve_all_bits() {
    let shape = Shape::new(GgmlType::Q4_K, 32, 256, 256).unwrap();
    let weights = crate::batch_scratch::tests::weights(shape.kind, 256, 256);
    let inputs = vec![0.25; 32 * 256];
    for threads in [1, 2] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            for case in 0..11 {
                let mut scratch = BatchProjectionScratch::new_four_row(32, 256, 256, 2).unwrap();
                if case == 4 {
                    scratch.workers[1].decoded.truncate(3 * 256);
                }
                if case == 5 {
                    scratch.transposed.pop();
                }
                if case == 6 {
                    scratch.output.pop();
                }
                if case == 7 {
                    scratch.workers.pop();
                }
                if case == 8 {
                    scratch.batch = 31;
                }
                if case == 9 {
                    scratch.rows = 255;
                }
                if case == 10 {
                    scratch.cols = 255;
                }
                crate::batch_scratch::tests::poison(&mut scratch);
                let before = crate::batch_scratch::tests::contents(&scratch);
                let mut out = vec![17.0; 32 * 256];
                let output_len = if case == 0 { out.len() - 1 } else { out.len() };
                let weight_len = if case == 1 {
                    weights.len() - 1
                } else {
                    weights.len()
                };
                let input_len = if case == 2 {
                    inputs.len() - 1
                } else {
                    inputs.len()
                };
                let group = if case == 3 { 3 } else { 4 };
                assert!(replay(
                    &mut out[..output_len],
                    &weights[..weight_len],
                    &inputs[..input_len],
                    &shape,
                    &mut scratch,
                    group
                )
                .is_err());
                assert!(out.iter().all(|v| v.to_bits() == 17.0f32.to_bits()));
                assert_eq!(crate::batch_scratch::tests::contents(&scratch), before);
            }
        });
    }
}
