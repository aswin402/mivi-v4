//! Test-only faithful replay controls, with no production dispatch changes.

type EvalResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

mod captured;

use std::hint::black_box;
use std::time::Instant;

fn validate_exact(expected: &[f32], actual: &[f32]) -> EvalResult<()> {
    if expected.is_empty() || expected.len() != actual.len() {
        return Err("exact gate rejected empty or differing lengths".into());
    }
    if expected
        .iter()
        .zip(actual)
        .any(|(a, b)| !a.is_finite() || !b.is_finite() || a.to_bits() != b.to_bits())
    {
        return Err(
            "exact gate rejected non-finite or differing output bits; values not logged".into(),
        );
    }
    Ok(())
}

fn isolation_shape(batch: usize, cols: usize) -> bool {
    matches!(batch, 32 | 64 | 65) && matches!(cols, 2048 | 8192)
}

fn require_benchmark_environment() -> EvalResult<()> {
    if cfg!(debug_assertions) || rayon::current_num_threads() != 2 {
        return Err("diagnostic requires --release and RAYON_NUM_THREADS=2".into());
    }
    Ok(())
}

#[test]
#[ignore = "bounded accumulation-only timing; release and two Rayon threads required"]
fn four_row_accumulation_isolation() -> EvalResult<()> {
    require_benchmark_environment()?;
    println!("ISOLATION rows=4 weights=generated-decoded-f32 transpose=outside-timers cache=warm-small-group iterations=200");
    for cols in [2048, 8192] {
        let weights = (0..4 * cols)
            .map(|i| ((i % 23) as f32 - 11.0) / 13.0)
            .collect::<Vec<_>>();
        for batch in [32, 64, 65] {
            assert!(isolation_shape(batch, cols));
            // Column-major inputs are prepared outside every timer; no decoding.
            let inputs = (0..batch * cols)
                .map(|i| ((i % 31) as f32 - 15.0) / 19.0)
                .collect::<Vec<_>>();
            let mut outputs = [vec![0.0; 4 * batch], vec![0.0; 4 * batch]];
            for round in 0..=6 {
                let order = if round % 2 == 0 { [1, 0] } else { [0, 1] };
                let mut times = [0u128; 2];
                for route in order {
                    let out = &mut outputs[route];
                    let start = Instant::now();
                    for _ in 0..200 {
                        out.fill(0.0);
                        if route == 1 {
                            mivi_core::simd::four_row::accumulate(
                                black_box(out),
                                black_box(&weights),
                                black_box(&inputs),
                                batch,
                                cols,
                            );
                        } else {
                            for pair in 0..2 {
                                let (a, b) = out[pair * 2 * batch..(pair + 1) * 2 * batch]
                                    .split_at_mut(batch);
                                mivi_core::simd::matmul_accumulate_transposed_pair_simd(
                                    black_box(a),
                                    black_box(b),
                                    black_box(&weights[pair * 2 * cols..(pair * 2 + 1) * cols]),
                                    black_box(
                                        &weights[(pair * 2 + 1) * cols..(pair * 2 + 2) * cols],
                                    ),
                                    black_box(&inputs),
                                    batch,
                                    cols,
                                );
                            }
                        }
                    }
                    times[route] = start.elapsed().as_nanos();
                    assert!(times[route] > 0);
                }
                validate_exact(&outputs[0], &outputs[1])?;
                println!("ISOLATION cols={cols} batch={batch} round={round} order={order:?} times_ns={times:?}");
            }
        }
    }
    Ok(())
}

#[test]
fn faithful_diagnostic_exact_gate_checks_every_bit_and_finiteness() {
    assert!(validate_exact(&[0.0, 1.0], &[0.0, 1.0]).is_ok());
    assert!(validate_exact(&[0.0], &[-0.0]).is_err());
    assert!(validate_exact(&[1.0], &[f32::from_bits(1.0f32.to_bits() + 1)]).is_err());
    assert!(validate_exact(&[1.0, 2.0], &[1.0]).is_err());
    assert!(validate_exact(&[], &[]).is_err());
    assert!(validate_exact(&[f32::NAN], &[f32::NAN]).is_err());
    assert!(validate_exact(&[f32::INFINITY], &[f32::INFINITY]).is_err());
}

#[test]
fn faithful_diagnostic_isolation_shape_is_bounded() {
    for batch in [32, 64, 65] {
        for cols in [2048, 8192] {
            assert!(isolation_shape(batch, cols));
        }
    }
    for (batch, cols) in [
        (0, 2048),
        (31, 2048),
        (66, 2048),
        (32, 0),
        (32, 2049),
        (32, usize::MAX),
        (usize::MAX, 2048),
    ] {
        assert!(!isolation_shape(batch, cols));
    }
}
