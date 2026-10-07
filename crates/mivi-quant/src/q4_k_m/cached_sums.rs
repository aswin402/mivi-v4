//! Opt-in faithful cached activation sums. Ordinary dispatch remains unchanged.

use super::*;

fn prepare_sums(x: &[f32], sums: &mut [f32], use_avx2: bool) {
    #[cfg(target_arch = "x86_64")]
    if use_avx2 {
        // SAFETY: the caller selected HAS_AVX2_FMA and validated 32 inputs per sum.
        unsafe {
            prepare_sums_avx2(x, sums);
        }
        return;
    }
    let _ = use_avx2;
    for (index, sum) in sums.iter_mut().enumerate() {
        *sum = x[index * 32..(index + 1) * 32]
            .iter()
            .fold(0.0f32, |a, &b| a + b);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn prepare_sums_avx2(x: &[f32], sums: &mut [f32]) {
    for (index, sum) in sums.iter_mut().enumerate() {
        let ptr = x.as_ptr().add(index * 32);
        let mut acc = _mm256_setzero_ps();
        for chunk in 0..4 {
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(ptr.add(chunk * 8)));
        }
        *sum = mivi_core::simd::hsum256_ps(acc);
    }
}

fn cached_row(row: &[u8], x: &[f32], blocks: usize, sums: &[f32], use_avx2: bool) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if use_avx2 {
        // SAFETY: runtime CPU dispatch and validated weights, input and sum bounds.
        return unsafe { cached_row_avx2(row, x, blocks, sums) };
    }
    let _ = use_avx2;
    cached_row_scalar(row, x, blocks, sums)
}
#[inline]
fn cached_row_scalar(row_bytes: &[u8], x: &[f32], blocks_per_row: usize, sums: &[f32]) -> f32 {
    let mut total_acc = 0.0f32;

    for b in 0..blocks_per_row {
        let block_offset = b * Q4_K_BYTES;
        let block = &row_bytes[block_offset..block_offset + Q4_K_BYTES];

        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();

        let scales = &block[4..16];
        let qs = &block[16..144];
        let x_block = &x[b * Q4_K_BLOCK_SIZE..(b + 1) * Q4_K_BLOCK_SIZE];

        for j in 0..4 {
            let (sc0, m0) = get_scale_min_k4(2 * j, scales);
            let (sc1, m1) = get_scale_min_k4(2 * j + 1, scales);

            let d1 = d * (sc0 as f32);
            let m1_val = dmin * (m0 as f32);
            let d2 = d * (sc1 as f32);
            let m2_val = dmin * (m1 as f32);

            let q_sub = &qs[j * 32..(j + 1) * 32];
            let x0 = &x_block[j * 64..j * 64 + 32];
            let x1 = &x_block[j * 64 + 32..j * 64 + 64];

            let mut sum_q0 = 0.0f32;
            let sum_x0 = sums[b * 8 + j * 2];
            let mut sum_q1 = 0.0f32;
            let sum_x1 = sums[b * 8 + j * 2 + 1];

            for l in 0..32 {
                let byte = q_sub[l];
                let q0 = (byte & 0x0F) as f32;
                let q1 = (byte >> 4) as f32;
                let x0_val = x0[l];
                let x1_val = x1[l];

                sum_q0 += q0 * x0_val;
                sum_q1 += q1 * x1_val;
            }

            total_acc += d1 * sum_q0 - m1_val * sum_x0 + d2 * sum_q1 - m2_val * sum_x1;
        }
    }

    total_acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn cached_row_avx2(row_bytes: &[u8], x: &[f32], blocks_per_row: usize, sums: &[f32]) -> f32 {
    let mut total_acc = 0.0f32;
    let mask_0f = _mm256_set1_epi32(0x0F);

    for b in 0..blocks_per_row {
        let block_offset = b * Q4_K_BYTES;
        let block = &row_bytes[block_offset..block_offset + Q4_K_BYTES];

        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();

        let scales = &block[4..16];
        let qs_ptr = block.as_ptr().add(16);
        let x_ptr = x.as_ptr().add(b * Q4_K_BLOCK_SIZE);

        for j in 0..4 {
            let (sc0, m0) = get_scale_min_k4(2 * j, scales);
            let (sc1, m1) = get_scale_min_k4(2 * j + 1, scales);

            let d1 = d * (sc0 as f32);
            let m1_val = dmin * (m0 as f32);
            let d2 = d * (sc1 as f32);
            let m2_val = dmin * (m1 as f32);

            let q_group_ptr = qs_ptr.add(j * 32);
            let x0_ptr = x_ptr.add(j * 64);
            let x1_ptr = x_ptr.add(j * 64 + 32);

            let mut acc_q0 = _mm256_setzero_ps();
            let mut acc_q1 = _mm256_setzero_ps();

            // Process 32 bytes in 4 chunks of 8 bytes
            for k in 0..4 {
                let offset = k * 8;
                let raw_8b = _mm_loadl_epi64(q_group_ptr.add(offset) as *const __m128i);
                let bytes_i32 = _mm256_cvtepu8_epi32(raw_8b);

                // Low nibbles (q0)
                let q0_i32 = _mm256_and_si256(bytes_i32, mask_0f);
                let q0_f32 = _mm256_cvtepi32_ps(q0_i32);
                let x0_v = _mm256_loadu_ps(x0_ptr.add(offset));

                acc_q0 = _mm256_fmadd_ps(q0_f32, x0_v, acc_q0);

                // High nibbles (q1)
                let q1_i32 = _mm256_srli_epi32(bytes_i32, 4);
                let q1_f32 = _mm256_cvtepi32_ps(q1_i32);
                let x1_v = _mm256_loadu_ps(x1_ptr.add(offset));

                acc_q1 = _mm256_fmadd_ps(q1_f32, x1_v, acc_q1);
            }

            let sum_q0 = mivi_core::simd::hsum256_ps(acc_q0);
            let sum_x0 = sums[b * 8 + j * 2];
            let sum_q1 = mivi_core::simd::hsum256_ps(acc_q1);
            let sum_x1 = sums[b * 8 + j * 2 + 1];

            total_acc += d1 * sum_q0 - m1_val * sum_x0 + d2 * sum_q1 - m2_val * sum_x1;
        }
    }

    total_acc
}

/// Caller-owned scratch needs d/32 floats. Errors leave both destinations intact.
pub fn try_matvec_q4_k_m_cached(
    out: &mut [f32],
    weights: &[u8],
    x: &[f32],
    n: usize,
    d: usize,
    scratch: &mut [f32],
) -> crate::types::Result<()> {
    let blocks = d / Q4_K_BLOCK_SIZE;
    let row_bytes = blocks
        .checked_mul(Q4_K_BYTES)
        .ok_or(crate::types::QuantError::ArithmeticOverflow)?;
    crate::types::validate_matvec_args(out, weights, x, n, d, row_bytes, Q4_K_BLOCK_SIZE)?;
    let required = d / 32;
    if scratch.len() < required {
        return Err(crate::types::QuantError::BufferTooSmall {
            expected: required,
            actual: scratch.len(),
        });
    }
    if n == 0 {
        return Ok(());
    }
    #[cfg(target_arch = "x86_64")]
    let use_avx2 = *mivi_core::simd::HAS_AVX2_FMA;
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;
    let sums = &mut scratch[..required];
    prepare_sums(x, sums, use_avx2);
    crate::types::parallel_row_matvec(out, weights, n, row_bytes, |row, _| {
        cached_row(row, x, blocks, sums, use_avx2)
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weights(n: usize, d: usize) -> Vec<u8> {
        (0..n * (d / 256))
            .flat_map(|seed| {
                let mut block = vec![0; Q4_K_BYTES];
                block[..2].copy_from_slice(&f16::from_f32(0.0625).to_le_bytes());
                block[2..4].copy_from_slice(&f16::from_f32(0.03125).to_le_bytes());
                for (i, byte) in block[4..].iter_mut().enumerate() {
                    *byte = seed.wrapping_mul(41).wrapping_add(i * 53 + 17) as u8;
                }
                block
            })
            .collect()
    }

    #[test]
    fn cached_sums_runtime_exact_and_reusable() {
        for n in [0, 1, 3, 257] {
            for d in [0, 256, 512, 2048] {
                let weights = weights(n, d);
                let mut scratch = vec![f32::NAN; d / 32 + 1];
                for seed in [3, 19] {
                    let x: Vec<_> = (0..d)
                        .map(|i| ((i * 37 + seed) % 113) as f32 / 17.0 - 3.0)
                        .collect();
                    let mut baseline = vec![123.0; n + 1];
                    let mut candidate = baseline.clone();
                    try_matvec_q4_k_m(&mut baseline, &weights, &x, n, d).unwrap();
                    try_matvec_q4_k_m_cached(&mut candidate, &weights, &x, n, d, &mut scratch)
                        .unwrap();
                    assert_eq!(
                        candidate.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        baseline.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                    );
                    assert!(scratch[d / 32].is_nan(), "scratch tail changed");
                }
            }
        }
    }

    #[test]
    fn cached_sums_each_route_exact() {
        for use_avx2 in [false, true] {
            if use_avx2 {
                #[cfg(target_arch = "x86_64")]
                if !*mivi_core::simd::HAS_AVX2_FMA {
                    continue;
                }
                #[cfg(not(target_arch = "x86_64"))]
                continue;
            }
            for d in [256, 512, 2048] {
                let row = weights(1, d);
                let x: Vec<_> = (0..d)
                    .map(|i| ((i * 37 + 3) % 113) as f32 / 17.0 - 3.0)
                    .collect();
                let mut sums = vec![f32::NAN; d / 32];
                prepare_sums(&x, &mut sums, use_avx2);
                assert_eq!(
                    cached_row(&row, &x, d / 256, &sums, use_avx2).to_bits(),
                    compute_single_row_q4_k_m(&row, &x, d / 256, use_avx2).to_bits()
                );
            }
        }
    }

    #[test]
    fn cached_sums_rejects_before_writes() {
        let weights = weights(3, 256);
        for (n, d, weight_len, input_len, scratch_len) in [
            (3, 255, weights.len(), 256, 8),
            (4, 256, weights.len(), 256, 8),
            (3, 256, weights.len() - 1, 256, 8),
            (3, 256, weights.len(), 255, 8),
            (3, 256, weights.len(), 256, 7),
            (usize::MAX, 256, weights.len(), 256, 8),
        ] {
            let mut out = vec![42.0; 3];
            let mut scratch = vec![73.0; scratch_len];
            assert!(try_matvec_q4_k_m_cached(
                &mut out,
                &weights[..weight_len],
                &vec![1.0; input_len],
                n,
                d,
                &mut scratch
            )
            .is_err());
            assert_eq!(out, vec![42.0; 3]);
            assert_eq!(scratch, vec![73.0; scratch_len]);
        }
    }
}
