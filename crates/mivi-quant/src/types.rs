//! Quantization format descriptors and block constants.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum QuantError {
    #[error("Unsupported GGML quantization type: {0}")]
    UnsupportedType(u32),
    #[error("Buffer too small: expected at least {expected} bytes, got {actual}")]
    BufferTooSmall { expected: usize, actual: usize },
    #[error("Dimension not a multiple of block size {block_size}: {dim}")]
    DimensionMisaligned { dim: usize, block_size: usize },
}

pub type Result<T> = std::result::Result<T, QuantError>;

/// GGML Quantization types matching GGUF specification.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12, // Q4_K_M
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
}

impl GgmlType {
    pub fn from_u32(val: u32) -> Result<Self> {
        match val {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2_K),
            11 => Ok(Self::Q3_K),
            12 => Ok(Self::Q4_K),
            13 => Ok(Self::Q5_K),
            14 => Ok(Self::Q6_K),
            15 => Ok(Self::Q8_K),
            16 => Ok(Self::IQ2_XXS),
            17 => Ok(Self::IQ2_XS),
            18 => Ok(Self::IQ3_XXS),
            19 => Ok(Self::IQ1_S),
            20 => Ok(Self::IQ4_NL),
            21 => Ok(Self::IQ3_S),
            22 => Ok(Self::IQ2_S),
            23 => Ok(Self::IQ4_XS),
            24 => Ok(Self::I8),
            25 => Ok(Self::I16),
            26 => Ok(Self::I32),
            27 => Ok(Self::I64),
            28 => Ok(Self::F64),
            29 => Ok(Self::IQ1_M),
            30 => Ok(Self::BF16),
            _ => Err(QuantError::UnsupportedType(val)),
        }
    }

    /// Block size (number of elements represented per quantization block).
    pub fn block_size(&self) -> Option<usize> {
        match self {
            Self::F32 | Self::I32 | Self::F64 | Self::I64 => Some(1),
            Self::F16 | Self::BF16 | Self::I16 | Self::I8 => Some(1),
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => Some(32),
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::Q8_K => {
                Some(256)
            }
            _ => None,
        }
    }

    /// Checked block size per block, returning QuantError on unsupported types.
    pub fn block_size_checked(&self) -> Result<usize> {
        self.block_size()
            .ok_or(QuantError::UnsupportedType(*self as u32))
    }

    /// Byte size per block.
    pub fn type_size(&self) -> Option<usize> {
        match self {
            Self::F32 | Self::I32 => Some(4),
            Self::F16 | Self::BF16 | Self::I16 => Some(2),
            Self::I8 => Some(1),
            Self::I64 | Self::F64 => Some(8),
            Self::Q4_0 => Some(18), // 2 + 16
            Self::Q4_1 => Some(20), // 2 + 2 + 16
            Self::Q5_0 => Some(22), // 2 + 4 + 16
            Self::Q5_1 => Some(24), // 2 + 2 + 4 + 16
            Self::Q8_0 => Some(34), // 2 + 32
            Self::Q8_1 => Some(36), // 4 + 32
            Self::Q2_K => Some(84),
            Self::Q3_K => Some(110),
            Self::Q4_K => Some(144), // Q4_K super-block of 256 elements
            Self::Q5_K => Some(176),
            Self::Q6_K => Some(210),
            Self::Q8_K => Some(292),
            _ => None,
        }
    }

    /// Checked byte size per block, returning QuantError on unsupported types.
    pub fn type_size_checked(&self) -> Result<usize> {
        self.type_size()
            .ok_or(QuantError::UnsupportedType(*self as u32))
    }
}

/// Chunk size for parallel rayon processing in quantized matrix-vector kernels.
pub const PARALLEL_CHUNK_SIZE: usize = 16;

/// Minimum row count threshold to trigger multi-threaded Rayon execution.
pub const RAYON_PARALLEL_THRESHOLD: usize = 64;

/// Validate input/output dimensions and buffer sizes for matrix-vector operations.
pub fn validate_matvec_args(
    out: &[f32],
    weights: &[u8],
    x: &[f32],
    n: usize,
    d: usize,
    row_bytes: usize,
    block_size: usize,
) -> Result<()> {
    if !d.is_multiple_of(block_size) {
        return Err(QuantError::DimensionMisaligned { dim: d, block_size });
    }
    if out.len() < n {
        return Err(QuantError::BufferTooSmall {
            expected: n,
            actual: out.len(),
        });
    }
    if x.len() < d {
        return Err(QuantError::BufferTooSmall {
            expected: d,
            actual: x.len(),
        });
    }
    let min_weights = n.checked_mul(row_bytes).ok_or(QuantError::BufferTooSmall {
        expected: usize::MAX,
        actual: weights.len(),
    })?;
    if weights.len() < min_weights {
        return Err(QuantError::BufferTooSmall {
            expected: min_weights,
            actual: weights.len(),
        });
    }
    Ok(())
}

use rayon::prelude::*;

/// Generic helper for parallel or serial row matrix-vector multiplication.
pub fn parallel_row_matvec<F>(
    out: &mut [f32],
    weights: &[u8],
    n: usize,
    row_bytes: usize,
    compute_row: F,
) where
    F: Fn(&[u8], usize) -> f32 + Send + Sync,
{
    if n >= RAYON_PARALLEL_THRESHOLD {
        out.par_chunks_mut(PARALLEL_CHUNK_SIZE)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let start_row = chunk_idx * PARALLEL_CHUNK_SIZE;
                for (r, out_val) in out_chunk.iter_mut().enumerate() {
                    let i = start_row + r;
                    let row_start = i * row_bytes;
                    let row_slice = &weights[row_start..row_start + row_bytes];
                    *out_val = compute_row(row_slice, i);
                }
            });
    } else {
        for (i, out_val) in out.iter_mut().enumerate().take(n) {
            let row_start = i * row_bytes;
            let row_slice = &weights[row_start..row_start + row_bytes];
            *out_val = compute_row(row_slice, i);
        }
    }
}

/// Generic helper for block-wise dequantization of fixed-size blocks.
pub fn dequantize_blocks<F>(
    bytes: &[u8],
    out: &mut [f32],
    block_bytes: usize,
    block_elems: usize,
    mut decode_block: F,
) where
    F: FnMut(&[u8], &mut [f32]),
{
    let n_blocks = out.len() / block_elems;
    assert!(
        bytes.len() >= n_blocks * block_bytes,
        "Dequantization slice buffer too small"
    );

    for b in 0..n_blocks {
        let block = &bytes[b * block_bytes..(b + 1) * block_bytes];
        let out_sub = &mut out[b * block_elems..(b + 1) * block_elems];
        decode_block(block, out_sub);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_size_known_types() {
        assert_eq!(GgmlType::F32.block_size(), Some(1));
        assert_eq!(GgmlType::Q8_0.block_size(), Some(32));
        assert_eq!(GgmlType::Q4_K.block_size(), Some(256));
    }

    #[test]
    fn test_block_size_unknown_returns_none() {
        assert_eq!(GgmlType::IQ2_XS.block_size(), None);
        assert!(GgmlType::IQ2_XS.block_size_checked().is_err());
    }
}
