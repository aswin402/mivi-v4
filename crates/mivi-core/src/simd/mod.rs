//! SIMD acceleration dispatcher.

pub mod norm;
pub mod scalar;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(target_arch = "x86_64")]
pub static HAS_AVX2_FMA: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
});

type MatvecFn = fn(&mut [f32], &[f32], &[f32], usize, usize);

static MATVEC_IMPL: std::sync::LazyLock<MatvecFn> = std::sync::LazyLock::new(|| {
    #[cfg(target_arch = "x86_64")]
    {
        if *HAS_AVX2_FMA {
            return |out, w, x, n, d| unsafe { avx2::matvec_f32_avx2(out, w, x, n, d) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return |out, w, x, n, d| unsafe { neon::matvec_f32_neon(out, w, x, n, d) };
    }

    #[allow(unreachable_code)]
    scalar::matvec_f32_scalar
});

#[cfg(target_arch = "x86_64")]
pub use avx2::hsum256_ps;

pub use norm::{rms_norm_in_place_simd, rms_norm_simd};

/// Token traversal width for the projection locality experiment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionTokenTile {
    Tokens32,
    Tokens64,
    Tokens128,
}

impl ProjectionTokenTile {
    #[inline]
    const fn width(self) -> usize {
        match self {
            Self::Tokens32 => 32,
            Self::Tokens64 => 64,
            Self::Tokens128 => 128,
        }
    }
}

/// Column panel width for paired projection accumulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionColumnTile {
    Columns32,
    Columns64,
    Columns128,
}

/// Vector dot product: sum(a[i] * b[i]) with SIMD acceleration.
#[inline]
pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if *HAS_AVX2_FMA {
            return unsafe { avx2::dot_product_avx2(a, b) };
        }
    }
    crate::math::dot_product_scalar(a, b)
}

/// Vector addition: out[i] += src[i] with SIMD acceleration.
#[inline]
pub fn vec_add_simd(out: &mut [f32], src: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if *HAS_AVX2_FMA {
            unsafe {
                avx2::vec_add_avx2(out, src);
                return;
            }
        }
    }
    crate::math::vec_add_scalar(out, src);
}

/// Vector fused multiply-add: out[i] += scale * src[i] with SIMD acceleration.
#[inline]
pub fn vec_fmadd_simd(out: &mut [f32], scale: f32, src: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if *HAS_AVX2_FMA {
            unsafe {
                avx2::vec_fmadd_avx2(out, scale, src);
                return;
            }
        }
    }
    crate::math::vec_fmadd_scalar(out, scale, src);
}

/// Accumulate a decoded row against column-major batches in one SIMD dispatch.
///
/// `transposed_inputs` is laid out as `[cols, batch]`, while `out` is `[batch]`.
#[inline]
pub fn matmul_accumulate_transposed_simd(
    out: &mut [f32],
    weights: &[f32],
    transposed_inputs: &[f32],
    batch: usize,
    cols: usize,
) {
    debug_assert!(out.len() >= batch);
    debug_assert!(weights.len() >= cols);
    debug_assert!(transposed_inputs.len() >= cols * batch);

    #[cfg(target_arch = "x86_64")]
    {
        if *HAS_AVX2_FMA {
            // SAFETY: the feature flag guarantees AVX2/FMA support and the debug
            // assertions document the slice requirements for this internal kernel.
            unsafe {
                avx2::matmul_accumulate_transposed_avx2(
                    &mut out[..batch],
                    &weights[..cols],
                    &transposed_inputs[..cols * batch],
                    batch,
                    cols,
                );
            }
            return;
        }
    }

    for (col, &scale) in weights.iter().take(cols).enumerate() {
        let input_start = col * batch;
        for batch_idx in 0..batch {
            out[batch_idx] += scale * transposed_inputs[input_start + batch_idx];
        }
    }
}

/// Accumulate a decoded row against column-major batches using an explicit
/// token traversal width.
///
/// `transposed_inputs` is laid out as `[cols, batch]`, while `out` is `[batch]`.
#[inline]
pub fn matmul_accumulate_transposed_with_tile_simd(
    out: &mut [f32],
    weights: &[f32],
    transposed_inputs: &[f32],
    batch: usize,
    cols: usize,
    tile: ProjectionTokenTile,
) {
    let inputs_len = cols.checked_mul(batch).expect("batch dimensions overflow");
    let out = &mut out[..batch];
    let weights = &weights[..cols];
    let inputs = &transposed_inputs[..inputs_len];
    matmul_accumulate_transposed_with_tile_impl(out, weights, None, inputs, batch, cols, tile);
}

/// Accumulate two decoded rows while sharing the transposed input loads.
/// Output slices are `[batch]`; inputs are `[cols, batch]`.
///
/// # Panics
/// Panics if either output/weight slice is too short, the input dimensions
/// overflow, or the transposed input slice does not cover those dimensions.
#[inline]
pub fn matmul_accumulate_transposed_pair_simd(
    out0: &mut [f32],
    out1: &mut [f32],
    weights0: &[f32],
    weights1: &[f32],
    transposed_inputs: &[f32],
    batch: usize,
    cols: usize,
) {
    matmul_accumulate_transposed_pair_with_column_tile_simd(
        out0,
        out1,
        weights0,
        weights1,
        transposed_inputs,
        batch,
        cols,
        ProjectionColumnTile::Columns128,
    );
}

/// Accumulate two decoded rows with a fixed column panel width.
/// Output slices are `[batch]`; inputs are `[cols, batch]`.
///
/// # Panics
/// Panics if either output/weight slice is too short, the input dimensions
/// overflow, or the transposed input slice does not cover those dimensions.
#[inline]
pub fn matmul_accumulate_transposed_pair_with_column_tile_simd(
    out0: &mut [f32],
    out1: &mut [f32],
    weights0: &[f32],
    weights1: &[f32],
    transposed_inputs: &[f32],
    batch: usize,
    cols: usize,
    tile: ProjectionColumnTile,
) {
    let out0 = &mut out0[..batch];
    let out1 = &mut out1[..batch];
    let weights0 = &weights0[..cols];
    let weights1 = &weights1[..cols];
    let inputs_len = cols.checked_mul(batch).expect("batch dimensions overflow");
    let inputs = &transposed_inputs[..inputs_len];
    #[cfg(target_arch = "x86_64")]
    if batch >= 32 && *HAS_AVX2_FMA {
        // SAFETY: runtime feature detection and the public entry points' slices
        // guarantee AVX2/FMA support and valid bounds for the entire kernel.
        unsafe {
            match tile {
                ProjectionColumnTile::Columns32 => {
                    avx2::pair_panel_avx2::<32>(out0, out1, weights0, weights1, inputs, batch, cols)
                }
                ProjectionColumnTile::Columns64 => {
                    avx2::pair_panel_avx2::<64>(out0, out1, weights0, weights1, inputs, batch, cols)
                }
                ProjectionColumnTile::Columns128 => avx2::pair_panel_avx2::<128>(
                    out0, out1, weights0, weights1, inputs, batch, cols,
                ),
            }
        }
        return;
    }
    matmul_accumulate_transposed_simd(out0, weights0, inputs, batch, cols);
    matmul_accumulate_transposed_simd(out1, weights1, inputs, batch, cols);
}

/// Accumulate two decoded rows with an explicit token traversal width.
/// Output slices are `[batch]`; inputs are `[cols, batch]`.
///
/// # Panics
/// Panics if either output/weight slice is too short, the input dimensions
/// overflow, or the transposed input slice does not cover those dimensions.
#[inline]
pub fn matmul_accumulate_transposed_pair_with_tile_simd(
    out0: &mut [f32],
    out1: &mut [f32],
    weights0: &[f32],
    weights1: &[f32],
    transposed_inputs: &[f32],
    batch: usize,
    cols: usize,
    tile: ProjectionTokenTile,
) {
    let out0 = &mut out0[..batch];
    let out1 = &mut out1[..batch];
    let weights0 = &weights0[..cols];
    let weights1 = &weights1[..cols];
    let inputs_len = cols.checked_mul(batch).expect("batch dimensions overflow");
    let inputs = &transposed_inputs[..inputs_len];
    matmul_accumulate_transposed_pair_with_tile_impl(
        out0, out1, weights0, weights1, inputs, batch, cols, tile,
    );
}

#[inline]
fn matmul_accumulate_transposed_with_tile_impl(
    out0: &mut [f32],
    weights0: &[f32],
    mut second_row: Option<(&mut [f32], &[f32])>,
    inputs: &[f32],
    batch: usize,
    cols: usize,
    tile: ProjectionTokenTile,
) {
    #[cfg(target_arch = "x86_64")]
    if *HAS_AVX2_FMA {
        // SAFETY: runtime feature detection and the public entry points' slices
        // guarantee AVX2/FMA support and valid bounds for the entire kernel.
        unsafe {
            avx2::matmul_accumulate_transposed_with_tile_avx2(
                out0,
                weights0,
                second_row,
                inputs,
                batch,
                cols,
                tile.width(),
            );
        }
        return;
    }

    let width = tile.width();
    for token_start in (0..batch).step_by(width) {
        let token_end = (token_start + width).min(batch);
        for token in token_start..token_end {
            let mut accumulator0 = out0[token];
            let mut accumulator1 = second_row.as_ref().map(|(out, _)| out[token]);
            for col in 0..cols {
                let input = inputs[col * batch + token];
                accumulator0 += weights0[col] * input;
                if let (Some(accumulator1), Some((_, weights1))) =
                    (accumulator1.as_mut(), second_row.as_ref())
                {
                    *accumulator1 += weights1[col] * input;
                }
            }
            out0[token] = accumulator0;
            if let (Some(accumulator1), Some((out1, _))) = (accumulator1, second_row.as_mut()) {
                out1[token] = accumulator1;
            }
        }
    }
}

#[inline]
fn matmul_accumulate_transposed_pair_with_tile_impl(
    out0: &mut [f32],
    out1: &mut [f32],
    weights0: &[f32],
    weights1: &[f32],
    inputs: &[f32],
    batch: usize,
    cols: usize,
    tile: ProjectionTokenTile,
) {
    matmul_accumulate_transposed_with_tile_impl(
        out0,
        weights0,
        Some((out1, weights1)),
        inputs,
        batch,
        cols,
        tile,
    );
}

#[cfg(test)]
mod tests {
    #[test]
    fn projection_token_tile_compatibility_matches_default_simd() {
        for tile in [
            super::ProjectionTokenTile::Tokens32,
            super::ProjectionTokenTile::Tokens64,
            super::ProjectionTokenTile::Tokens128,
        ] {
            for batch in [31, 33, 65] {
                for cols in [1, 17, 257] {
                    let weights0: Vec<f32> =
                        (0..cols).map(|i| (i % 13) as f32 * 0.125 - 0.75).collect();
                    let weights1: Vec<f32> =
                        (0..cols).map(|i| (i % 17) as f32 * 0.0625 - 0.5).collect();
                    let inputs: Vec<f32> = (0..cols * batch)
                        .map(|i| (i % 19) as f32 * 0.0625 - 0.5)
                        .collect();
                    let mut expected0 = vec![0.25_f32; batch];
                    let mut expected1 = vec![-0.5_f32; batch];
                    let mut actual0 = expected0.clone();
                    let mut actual1 = expected1.clone();

                    super::matmul_accumulate_transposed_simd(
                        &mut expected0,
                        &weights0,
                        &inputs,
                        batch,
                        cols,
                    );
                    super::matmul_accumulate_transposed_simd(
                        &mut expected1,
                        &weights1,
                        &inputs,
                        batch,
                        cols,
                    );
                    super::matmul_accumulate_transposed_with_tile_simd(
                        &mut actual0,
                        &weights0,
                        &inputs,
                        batch,
                        cols,
                        tile,
                    );
                    assert_eq!(
                        actual0
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>(),
                        expected0
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>(),
                        "single: tile={tile:?}, batch={batch}, cols={cols}"
                    );

                    actual0.fill(0.25);
                    super::matmul_accumulate_transposed_pair_with_tile_simd(
                        &mut actual0,
                        &mut actual1,
                        &weights0,
                        &weights1,
                        &inputs,
                        batch,
                        cols,
                        tile,
                    );
                    for (expected, actual) in [(&expected0, &actual0), (&expected1, &actual1)] {
                        assert_eq!(
                            expected
                                .iter()
                                .map(|value| value.to_bits())
                                .collect::<Vec<_>>(),
                            actual
                                .iter()
                                .map(|value| value.to_bits())
                                .collect::<Vec<_>>(),
                            "pair: tile={tile:?}, batch={batch}, cols={cols}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn projection_column_tile_matches_default_pair_bits_and_preserves_sentinels() {
        for tile in [
            super::ProjectionColumnTile::Columns32,
            super::ProjectionColumnTile::Columns64,
            super::ProjectionColumnTile::Columns128,
        ] {
            for batch in [32, 33, 63, 64, 65, 127, 128, 129] {
                for cols in [1, 17, 257] {
                    let weights0: Vec<f32> =
                        (0..cols).map(|i| (i % 13) as f32 * 0.125 - 0.75).collect();
                    let weights1: Vec<f32> =
                        (0..cols).map(|i| (i % 17) as f32 * 0.0625 - 0.5).collect();
                    let inputs: Vec<f32> = (0..cols * batch)
                        .map(|i| (i % 19) as f32 * 0.0625 - 0.5)
                        .collect();

                    let mut expected0 = vec![0.25_f32; batch + 1];
                    let mut expected1 = vec![-0.5_f32; batch + 1];
                    let mut actual0 = expected0.clone();
                    let mut actual1 = expected1.clone();
                    super::matmul_accumulate_transposed_pair_simd(
                        &mut expected0,
                        &mut expected1,
                        &weights0,
                        &weights1,
                        &inputs,
                        batch,
                        cols,
                    );
                    super::matmul_accumulate_transposed_pair_with_column_tile_simd(
                        &mut actual0,
                        &mut actual1,
                        &weights0,
                        &weights1,
                        &inputs,
                        batch,
                        cols,
                        tile,
                    );
                    for (expected, actual) in [(&expected0, &actual0), (&expected1, &actual1)] {
                        for index in 0..batch {
                            assert_eq!(expected[index].to_bits(), actual[index].to_bits());
                        }
                        assert_eq!(expected[batch].to_bits(), actual[batch].to_bits());
                    }
                }
            }
        }
    }

    #[test]
    fn paired_transposed_accumulation_matches_independent_rows() {
        for batch in [0, 1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 128] {
            for cols in [0, 1, 17, 257] {
                let first: Vec<f32> = (0..cols).map(|i| (i % 13) as f32 * 0.125 - 0.75).collect();
                let second: Vec<f32> = (0..cols).map(|i| (i % 17) as f32 * 0.0625 - 0.5).collect();
                let inputs: Vec<f32> = (0..cols * batch)
                    .map(|i| (i % 19) as f32 * 0.0625 - 0.5)
                    .collect();
                let mut actual0 = vec![0.25; batch + 1];
                let mut actual1 = vec![-0.5; batch + 1];
                let mut expected0 = actual0.clone();
                let mut expected1 = actual1.clone();
                super::matmul_accumulate_transposed_simd(
                    &mut expected0,
                    &first,
                    &inputs,
                    batch,
                    cols,
                );
                super::matmul_accumulate_transposed_simd(
                    &mut expected1,
                    &second,
                    &inputs,
                    batch,
                    cols,
                );
                super::matmul_accumulate_transposed_pair_simd(
                    &mut actual0,
                    &mut actual1,
                    &first,
                    &second,
                    &inputs,
                    batch,
                    cols,
                );
                assert_eq!(actual0, expected0, "first row: batch={batch}, cols={cols}");
                assert_eq!(actual1, expected1, "second row: batch={batch}, cols={cols}");
            }
        }
    }

    #[test]
    fn paired_transposed_accumulation_preserves_rounding_order() {
        let cols = 1025;
        let first: Vec<f32> = (0..cols).map(|i| (i % 13) as f32 / 7.0 - 0.75).collect();
        let second: Vec<f32> = (0..cols).map(|i| (i % 17) as f32 / 11.0 - 0.5).collect();
        for batch in [32, 33, 64, 65, 128] {
            let inputs: Vec<f32> = (0..cols * batch)
                .map(|i| (i % 19) as f32 / 19.0 - 0.5)
                .collect();
            let mut actual0 = vec![0.125; batch];
            let mut actual1 = vec![-0.75; batch];
            let mut expected0 = actual0.clone();
            let mut expected1 = actual1.clone();
            super::matmul_accumulate_transposed_simd(&mut expected0, &first, &inputs, batch, cols);
            super::matmul_accumulate_transposed_simd(&mut expected1, &second, &inputs, batch, cols);
            super::matmul_accumulate_transposed_pair_simd(
                &mut actual0,
                &mut actual1,
                &first,
                &second,
                &inputs,
                batch,
                cols,
            );
            assert_eq!(actual0, expected0, "first row: batch={batch}");
            assert_eq!(actual1, expected1, "second row: batch={batch}");
        }
    }

    #[test]
    #[should_panic]
    fn paired_transposed_accumulation_rejects_short_inputs() {
        super::matmul_accumulate_transposed_pair_simd(
            &mut [0.0; 32],
            &mut [0.0; 32],
            &[1.0; 2],
            &[1.0; 2],
            &[0.0; 63],
            32,
            2,
        );
    }

    #[test]
    #[should_panic]
    fn paired_transposed_accumulation_rejects_short_second_row() {
        super::matmul_accumulate_transposed_pair_simd(
            &mut [0.0; 32],
            &mut [0.0; 31],
            &[1.0; 2],
            &[1.0; 2],
            &[0.0; 64],
            32,
            2,
        );
    }

    #[test]
    #[should_panic]
    fn paired_transposed_accumulation_rejects_short_second_weights() {
        super::matmul_accumulate_transposed_pair_simd(
            &mut [0.0; 32],
            &mut [0.0; 32],
            &[1.0; 2],
            &[1.0],
            &[0.0; 64],
            32,
            2,
        );
    }

    #[test]
    #[ignore = "focused release-mode performance measurement"]
    fn paired_transposed_accumulation_benchmark() {
        use std::hint::black_box;
        use std::time::Instant;
        for batch in [32, 64, 128] {
            let cols = 8192;
            let first: Vec<f32> = (0..cols).map(|i| (i % 13) as f32 * 0.125).collect();
            let second: Vec<f32> = (0..cols).map(|i| (i % 17) as f32 * 0.0625).collect();
            let inputs: Vec<f32> = (0..cols * batch)
                .map(|i| (i % 19) as f32 * 0.0625)
                .collect();
            let mut out0 = vec![0.0; batch];
            let mut out1 = vec![0.0; batch];
            for paired in [false, true, false, true] {
                let started = Instant::now();
                for _ in 0..512 {
                    out0.fill(0.0);
                    out1.fill(0.0);
                    if paired {
                        super::matmul_accumulate_transposed_pair_simd(
                            black_box(&mut out0),
                            black_box(&mut out1),
                            black_box(&first),
                            black_box(&second),
                            black_box(&inputs),
                            batch,
                            cols,
                        );
                    } else {
                        super::matmul_accumulate_transposed_simd(
                            black_box(&mut out0),
                            black_box(&first),
                            black_box(&inputs),
                            batch,
                            cols,
                        );
                        super::matmul_accumulate_transposed_simd(
                            black_box(&mut out1),
                            black_box(&second),
                            black_box(&inputs),
                            batch,
                            cols,
                        );
                    }
                    black_box((&out0, &out1));
                }
                println!(
                    "paired={paired}, batch={batch}: {:.3} ms/two rows",
                    started.elapsed().as_secs_f64() * 1000.0 / 512.0
                );
            }
        }
    }

    #[test]
    fn transposed_batch_accumulation_covers_vector_blocks_and_tails() {
        for batch in [0, 1, 7, 8, 9, 31, 32, 33, 64, 65, 128] {
            for cols in [0, 1, 17, 257] {
                let weights: Vec<f32> = (0..cols).map(|i| (i % 13) as f32 * 0.125 - 0.75).collect();
                let inputs: Vec<f32> = (0..cols * batch)
                    .map(|i| (i % 19) as f32 * 0.0625 - 0.5)
                    .collect();
                // Nonzero initial values check accumulation, and the sentinel
                // checks that the kernel touches only the requested batch.
                let mut actual = vec![0.25; batch + 1];
                let mut expected = actual.clone();
                for col in 0..cols {
                    for b in 0..batch {
                        expected[b] += weights[col] * inputs[col * batch + b];
                    }
                }
                super::matmul_accumulate_transposed_simd(
                    &mut actual,
                    &weights,
                    &inputs,
                    batch,
                    cols,
                );
                assert_eq!(actual, expected, "batch={batch}, cols={cols}");
            }
        }
    }

    #[test]
    #[ignore = "focused release-mode performance measurement"]
    fn transposed_batch_accumulation_benchmark() {
        use std::hint::black_box;
        use std::time::Instant;
        for batch in [32, 64, 128] {
            let cols = 8192;
            let weights: Vec<f32> = (0..cols).map(|i| (i % 13) as f32 * 0.125).collect();
            let inputs: Vec<f32> = (0..cols * batch)
                .map(|i| (i % 19) as f32 * 0.0625)
                .collect();
            let mut output = vec![0.0; batch];
            let started = Instant::now();
            for _ in 0..512 {
                output.fill(0.0);
                super::matmul_accumulate_transposed_simd(
                    black_box(&mut output),
                    black_box(&weights),
                    black_box(&inputs),
                    batch,
                    cols,
                );
                black_box(&output);
            }
            println!(
                "batch={batch}, cols={cols}: {:.3} ms/row",
                started.elapsed().as_secs_f64() * 1000.0 / 512.0
            );
        }
    }

    #[test]
    fn transposed_batch_accumulation_matches_scalar_reference() {
        let weights = [0.5f32, -1.0, 2.0];
        let inputs = [
            1.0f32, 2.0, 3.0, 4.0, // column 0
            -2.0, 0.5, 1.5, 3.0, // column 1
            4.0, -1.0, 2.0, 0.25, // column 2
        ];
        let mut actual = [0.0f32; 4];

        super::matmul_accumulate_transposed_simd(&mut actual, &weights, &inputs, 4, weights.len());

        let expected = [
            0.5 * 1.0 + -1.0 * -2.0 + 2.0 * 4.0,
            0.5 * 2.0 + -1.0 * 0.5 + 2.0 * -1.0,
            0.5 * 3.0 + -1.0 * 1.5 + 2.0 * 2.0,
            0.5 * 4.0 + -1.0 * 3.0 + 2.0 * 0.25,
        ];

        assert_eq!(actual, expected);
    }
}

/// Matrix-vector multiplication for dense f32 matrices: out[n] = w[n, d] * x[d]
#[inline]
pub fn matvec_f32(out: &mut [f32], w: &[f32], x: &[f32], n: usize, d: usize) {
    assert!(
        out.len() >= n,
        "Output buffer too small: {} < {}",
        out.len(),
        n
    );
    assert!(
        w.len() >= n * d,
        "Weight buffer too small: {} < {}",
        w.len(),
        n * d
    );
    assert!(x.len() >= d, "Input vector too small: {} < {}", x.len(), d);

    (*MATVEC_IMPL)(out, w, x, n, d);
}
