//! Default-off four-output-row accumulation experiment.

#[cfg(target_arch = "x86_64")]
mod avx2;

/// Accumulate four contiguous decoded rows against `[cols, batch]` inputs.
/// Output and weight layouts are `[4, batch]` and `[4, cols]` respectively.
///
/// # Panics
/// Panics before writes on dimension overflow or insufficient slices.
pub fn accumulate(out: &mut [f32], weights: &[f32], inputs: &[f32], batch: usize, cols: usize) {
    let output_len = batch
        .checked_mul(4)
        .expect("four-row output dimensions overflow");
    let weight_len = cols
        .checked_mul(4)
        .expect("four-row weight dimensions overflow");
    let input_len = cols
        .checked_mul(batch)
        .expect("four-row input dimensions overflow");
    let out = &mut out[..output_len];
    let weights = &weights[..weight_len];
    let inputs = &inputs[..input_len];
    if batch == 0 || cols == 0 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if batch >= 32 && *super::HAS_AVX2_FMA {
        // SAFETY: runtime detection guarantees AVX2/FMA. Checked products and
        // bounded slices above cover four rows and all input lanes. Mutable
        // output cannot alias either immutable input through safe Rust callers.
        unsafe {
            avx2::accumulate(out, weights, inputs, batch, cols);
        }
        return;
    }
    fallback(out, weights, inputs, batch, cols);
}

fn fallback(out: &mut [f32], weights: &[f32], inputs: &[f32], batch: usize, cols: usize) {
    for pair in 0..2 {
        let (a, b) = out[pair * 2 * batch..(pair + 1) * 2 * batch].split_at_mut(batch);
        super::matmul_accumulate_transposed_pair_simd(
            a,
            b,
            &weights[pair * 2 * cols..(pair * 2 + 1) * cols],
            &weights[(pair * 2 + 1) * cols..(pair * 2 + 2) * cols],
            inputs,
            batch,
            cols,
        );
    }
}

#[cfg(test)]
mod tests {
    fn reference(out: &mut [f32], weights: &[f32], inputs: &[f32], batch: usize, cols: usize) {
        for pair in 0..2 {
            let (a, b) = out[pair * 2 * batch..(pair + 1) * 2 * batch].split_at_mut(batch);
            super::super::matmul_accumulate_transposed_pair_simd(
                a,
                b,
                &weights[pair * 2 * cols..(pair * 2 + 1) * cols],
                &weights[(pair * 2 + 1) * cols..(pair * 2 + 2) * cols],
                inputs,
                batch,
                cols,
            );
        }
    }

    #[test]
    fn four_row_preserves_pair_bits_and_output_tails() {
        for batch in [0, 1, 2, 8, 9, 31, 32, 33, 63, 64, 65] {
            for cols in [0, 1, 7, 127, 128, 129, 257] {
                let weights: Vec<_> = (0..4 * cols)
                    .map(|i| ((i % 23) as f32 - 11.0) / 13.0)
                    .collect();
                let inputs: Vec<_> = (0..batch * cols)
                    .map(|i| ((i % 31) as f32 - 15.0) / 19.0)
                    .collect();
                let mut old: Vec<_> = (0..4 * batch + 3)
                    .map(|i| ((i % 7) as f32 - 3.0) / 17.0)
                    .collect();
                let mut new = old.clone();
                reference(&mut old, &weights, &inputs, batch, cols);
                super::accumulate(&mut new, &weights, &inputs, batch, cols);
                assert_eq!(
                    old.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    new.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "batch={batch} cols={cols}"
                );
                let mut fallback = (0..4 * batch + 3)
                    .map(|i| ((i % 7) as f32 - 3.0) / 17.0)
                    .collect::<Vec<_>>();
                super::fallback(&mut fallback, &weights, &inputs, batch, cols);
                assert_eq!(
                    old.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    fallback.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn four_row_invalid_dimensions_panic_before_writes() {
        use std::panic::{catch_unwind, AssertUnwindSafe};
        let mut out = vec![17.0; 4 * 33];
        let weights = vec![0.5; 4 * 129];
        let inputs = vec![0.25; 33 * 129];
        for (output_len, weight_len, input_len, batch, cols) in [
            (out.len() - 1, weights.len(), inputs.len(), 33, 129),
            (out.len(), weights.len() - 1, inputs.len(), 33, 129),
            (out.len(), weights.len(), inputs.len() - 1, 33, 129),
            (out.len(), 0, 0, usize::MAX, 0),
            (out.len(), 0, 0, 0, usize::MAX),
            (out.len(), 0, 0, usize::MAX / 4, 5),
        ] {
            assert!(catch_unwind(AssertUnwindSafe(|| super::accumulate(
                &mut out[..output_len],
                &weights[..weight_len],
                &inputs[..input_len],
                batch,
                cols
            )))
            .is_err());
            assert!(out.iter().all(|v| *v == 17.0));
        }
    }

    #[test]
    fn four_row_preserves_cancellation_rounding() {
        for batch in [32, 33, 63, 64, 65] {
            let cols = 259;
            let weights: Vec<_> = (0..4 * cols)
                .map(|i| [1.0000001, -1.0, 0.125, -0.125, 0.00003125][i % 5])
                .collect();
            let inputs: Vec<_> = (0..batch * cols)
                .map(|i| [1e7, 1.0, -1e7, 0.00001, -0.375][i % 5])
                .collect();
            let mut old = vec![0.0; 4 * batch];
            let mut new = old.clone();
            reference(&mut old, &weights, &inputs, batch, cols);
            super::accumulate(&mut new, &weights, &inputs, batch, cols);
            assert_eq!(
                old.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                new.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }
}
