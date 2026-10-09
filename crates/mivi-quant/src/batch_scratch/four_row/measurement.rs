//! Bounded opt-in operator diagnostic, not model/agent performance evidence.

use super::{quantized_matmul_rows_four_row, BatchProjectionScratch, GgmlType};
use crate::batch_scratch::quantized_matmul_rows_with_scratch;
use std::hint::black_box;
use std::time::Instant;

fn shape_allowed(wire_type: u32, dims: &[usize]) -> bool {
    matches!(wire_type, n if n == GgmlType::Q4_K as u32 || n == GgmlType::Q6_K as u32)
        && dims.len() == 2
        && dims[0] > 0
        && dims[0] <= 16384
        && dims[0].is_multiple_of(256)
        && dims[1] > 0
        && dims[1] <= 8192
}

type EvalResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[test]
#[ignore = "requires MIVI_TEST_MODEL; bounded operator timing, not model inference"]
fn four_row_operator_measurement() -> EvalResult<()> {
    if rayon::current_num_threads() != 2 {
        return Err("this bounded diagnostic requires RAYON_NUM_THREADS=2".into());
    }
    let path = std::env::var_os("MIVI_TEST_MODEL").ok_or("set MIVI_TEST_MODEL to local GGUF")?;
    let gguf = mivi_model::GgufFile::open(std::path::Path::new(&path))?;
    #[cfg(target_arch = "x86_64")]
    println!("FOUR_ROW avx2_fma={} threads=2 profile=release-required activations=generated pages=warm-process calls_per_member=3 measured_triples=6", *mivi_core::simd::HAS_AVX2_FMA);
    if cfg!(debug_assertions) {
        return Err("operator timing requires --release".into());
    }
    let synthetic = crate::batch_scratch::tests::weights(GgmlType::Q8_0, 257, 256);
    measure("synthetic", GgmlType::Q8_0, &synthetic, 257, 256)?;
    // Sort metadata deterministically, then select one full eligible matrix per
    // format. Compare wire types across the dev-dependency's crate artifact.
    for kind in [GgmlType::Q4_K, GgmlType::Q6_K] {
        let mut candidates = gguf
            .tensors
            .values()
            .filter(|t| {
                shape_allowed(t.ggml_type as u32, &t.dims) && t.ggml_type as u32 == kind as u32
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|a, b| a.name.cmp(&b.name));
        let tensor = candidates
            .first()
            .ok_or("GGUF missing eligible Q4_K/Q6_K full matrix")?;
        let cols = tensor.dims[0];
        let rows = tensor.dims[1];
        let (_, weights) = gguf.get_tensor_data(&tensor.name)?;
        let row_bytes = (cols / kind.block_size_checked()?)
            .checked_mul(kind.type_size_checked()?)
            .ok_or("row byte overflow")?;
        let byte_len = rows.checked_mul(row_bytes).ok_or("weight byte overflow")?;
        let weights = weights.get(..byte_len).ok_or("short tensor storage")?;
        measure("real", kind, weights, rows, cols)?;
    }
    Ok(())
}

fn measure(
    source: &str,
    kind: GgmlType,
    weights: &[u8],
    rows: usize,
    cols: usize,
) -> EvalResult<()> {
    // Every measured triple contains each control once. All six order
    // permutations balance each control's first/middle/last position.
    let orders = [
        [0, 1, 2],
        [2, 1, 0],
        [1, 2, 0],
        [0, 2, 1],
        [2, 0, 1],
        [1, 0, 2],
    ];
    for batch in [8, 9, 32, 64, 65] {
        let inputs = (0..batch * cols)
            .map(|i| ((i % 31) as f32 - 15.0) / 19.0)
            .collect::<Vec<_>>();
        let mut scratch = BatchProjectionScratch::new(batch, rows, cols, 2)?;
        let mut four = BatchProjectionScratch::new_four_row(batch, rows, cols, 2)?;
        let mut outputs = std::array::from_fn::<_, 3, _>(|_| vec![0.0; batch * rows]);
        let mut reference = vec![0.0; batch * rows];
        crate::quantized_matmul_rows(&mut reference, kind, weights, &inputs, batch, rows, cols)?;
        assert!(reference.iter().all(|v| v.is_finite()));
        for round in 0..=6 {
            let order = if round == 0 {
                [0, 1, 2]
            } else {
                orders[round - 1]
            };
            let mut times = [0u128; 3];
            for variant in order {
                let out = &mut outputs[variant];
                let start = Instant::now();
                for _ in 0..3 {
                    match variant {
                        0 => crate::quantized_matmul_rows(
                            black_box(out),
                            kind,
                            black_box(weights),
                            black_box(&inputs),
                            batch,
                            rows,
                            cols,
                        )?,
                        1 => quantized_matmul_rows_with_scratch(
                            black_box(out),
                            kind,
                            black_box(weights),
                            black_box(&inputs),
                            batch,
                            rows,
                            cols,
                            &mut scratch,
                        )?,
                        2 => quantized_matmul_rows_four_row(
                            black_box(out),
                            kind,
                            black_box(weights),
                            black_box(&inputs),
                            batch,
                            rows,
                            cols,
                            &mut four,
                        )?,
                        _ => unreachable!(),
                    }
                }
                times[variant] = start.elapsed().as_nanos();
                assert!(times[variant] > 0);
                assert!(
                    out.iter()
                        .zip(&reference)
                        .all(|(a, b)| a.is_finite() && a.to_bits() == b.to_bits()),
                    "{source} {kind:?} batch={batch} variant={variant} complete output mismatch"
                );
            }
            println!("FOUR_ROW source={source} kind={kind:?} rows={rows} cols={cols} batch={batch} round={round} order={order:?} times_ns={times:?}");
        }
    }
    Ok(())
}

#[test]
fn four_row_measurement_rejects_unbounded_or_unsupported_shapes() {
    assert!(shape_allowed(crate::GgmlType::Q4_K as u32, &[2048, 8192]));
    assert!(shape_allowed(crate::GgmlType::Q6_K as u32, &[8192, 2048]));
    for dims in [
        &[][..],
        &[256][..],
        &[256, 3, 2][..],
        &[0, 3][..],
        &[256, 0][..],
        &[255, 3][..],
        &[16640, 3][..],
        &[256, 8193][..],
    ] {
        assert!(!shape_allowed(crate::GgmlType::Q4_K as u32, dims));
    }
    assert!(!shape_allowed(crate::GgmlType::F32 as u32, &[256, 3]));
}
