//! Reusable activation storage and row-wise helpers for chunked prompt prefill.

use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum TileError {
    #[error("tile dimensions must be greater than zero")]
    InvalidDimensions,
    #[error("tile allocation size overflow")]
    SizeOverflow,
    #[error("tile row {row} is outside the allocated range of {rows} rows")]
    RowOutOfBounds { row: usize, rows: usize },
    #[error("tile requires at least one processed row")]
    EmptyRows,
    #[error("tile buffer is too small: expected {expected}, got {actual}")]
    BufferTooSmall { expected: usize, actual: usize },
}

pub type Result<T> = std::result::Result<T, TileError>;

/// Reusable row-major buffers for one prompt tile.
///
/// `current` and `next` are ping-pong activation buffers. The remaining
/// buffers are tile-sized scratch and are intentionally separate from the
/// single-token recurrent state in `RunState`.
#[derive(Debug)]
pub struct TileActivations {
    tile_tokens: usize,
    dim: usize,
    hidden_dim: usize,
    pub(crate) current: Vec<f32>,
    pub(crate) next: Vec<f32>,
    pub(crate) norm: Vec<f32>,
    pub(crate) projection: Vec<f32>,
    pub(crate) gate: Vec<f32>,
    pub(crate) up: Vec<f32>,
    pub(crate) q: Vec<f32>,
    pub(crate) k: Vec<f32>,
    pub(crate) v: Vec<f32>,
}

impl TileActivations {
    pub fn new(tile_tokens: usize, dim: usize, hidden_dim: usize) -> Result<Self> {
        Self::with_kv_dim(tile_tokens, dim, hidden_dim, dim)
    }

    pub fn with_kv_dim(
        tile_tokens: usize,
        dim: usize,
        hidden_dim: usize,
        kv_dim: usize,
    ) -> Result<Self> {
        if tile_tokens == 0 || dim == 0 || hidden_dim == 0 || kv_dim == 0 {
            return Err(TileError::InvalidDimensions);
        }

        let dim_len = checked_len(tile_tokens, dim)?;
        let projection_len = checked_len(
            tile_tokens,
            dim.checked_mul(3).ok_or(TileError::SizeOverflow)?,
        )?;
        let hidden_len = checked_len(tile_tokens, hidden_dim)?;
        let kv_len = checked_len(tile_tokens, kv_dim)?;
        Ok(Self {
            tile_tokens,
            dim,
            hidden_dim,
            current: vec![0.0; dim_len],
            next: vec![0.0; dim_len],
            norm: vec![0.0; dim_len],
            projection: vec![0.0; projection_len],
            gate: vec![0.0; hidden_len],
            up: vec![0.0; hidden_len],
            q: vec![0.0; dim_len],
            k: vec![0.0; kv_len],
            v: vec![0.0; kv_len],
        })
    }

    #[inline]
    pub fn tile_tokens(&self) -> usize {
        self.tile_tokens
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    #[inline]
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    #[inline]
    pub fn current(&self) -> &[f32] {
        &self.current
    }

    #[inline]
    pub fn current_mut(&mut self) -> &mut [f32] {
        &mut self.current
    }

    #[inline]
    pub fn next(&self) -> &[f32] {
        &self.next
    }

    #[inline]
    pub fn next_mut(&mut self) -> &mut [f32] {
        &mut self.next
    }

    #[inline]
    pub fn norm_mut(&mut self) -> &mut [f32] {
        &mut self.norm
    }

    #[inline]
    pub fn projection_mut(&mut self) -> &mut [f32] {
        &mut self.projection
    }

    #[inline]
    pub fn gate(&self) -> &[f32] {
        &self.gate
    }

    #[inline]
    pub fn gate_mut(&mut self) -> &mut [f32] {
        &mut self.gate
    }

    #[inline]
    pub fn up(&self) -> &[f32] {
        &self.up
    }

    #[inline]
    pub fn up_mut(&mut self) -> &mut [f32] {
        &mut self.up
    }

    pub fn current_row(&self, row: usize) -> Result<&[f32]> {
        self.row_bounds(row)?;
        let start = row * self.dim;
        Ok(&self.current[start..start + self.dim])
    }

    pub fn current_row_mut(&mut self, row: usize) -> Result<&mut [f32]> {
        self.row_bounds(row)?;
        let start = row * self.dim;
        Ok(&mut self.current[start..start + self.dim])
    }

    pub fn final_current_row(&self, rows: usize) -> Result<&[f32]> {
        self.validate_rows(rows)?;
        self.current_row(rows - 1)
    }

    #[inline]
    pub fn swap_current_buffers(&mut self) {
        std::mem::swap(&mut self.current, &mut self.next);
    }

    fn row_bounds(&self, row: usize) -> Result<()> {
        if row >= self.tile_tokens {
            return Err(TileError::RowOutOfBounds {
                row,
                rows: self.tile_tokens,
            });
        }
        Ok(())
    }

    fn validate_rows(&self, rows: usize) -> Result<()> {
        if rows == 0 {
            return Err(TileError::EmptyRows);
        }
        if rows > self.tile_tokens {
            return Err(TileError::RowOutOfBounds {
                row: rows - 1,
                rows: self.tile_tokens,
            });
        }
        Ok(())
    }
}

fn checked_len(rows: usize, width: usize) -> Result<usize> {
    rows.checked_mul(width).ok_or(TileError::SizeOverflow)
}

fn validate_row_buffers(output: &[f32], input: &[f32], rows: usize, width: usize) -> Result<usize> {
    let expected = checked_len(rows, width)?;
    if output.len() < expected {
        return Err(TileError::BufferTooSmall {
            expected,
            actual: output.len(),
        });
    }
    if input.len() < expected {
        return Err(TileError::BufferTooSmall {
            expected,
            actual: input.len(),
        });
    }
    Ok(expected)
}

/// Apply RMSNorm independently to each row of a tile.
pub fn rms_norm_rows(
    output: &mut [f32],
    input: &[f32],
    weight: &[f32],
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<()> {
    let expected = validate_row_buffers(output, input, rows, dim)?;
    if weight.len() < dim {
        return Err(TileError::BufferTooSmall {
            expected: dim,
            actual: weight.len(),
        });
    }
    for row in 0..rows {
        let start = row * dim;
        mivi_core::simd::rms_norm_simd(
            &mut output[start..start + dim],
            &input[start..start + dim],
            &weight[..dim],
            eps,
        );
    }
    debug_assert_eq!(expected, rows * dim);
    Ok(())
}

/// Add one tile of residual rows into another tile in place.
pub fn add_rows_in_place(
    output: &mut [f32],
    residual: &[f32],
    rows: usize,
    dim: usize,
) -> Result<()> {
    let expected = validate_row_buffers(output, residual, rows, dim)?;
    for (out, residual) in output[..expected]
        .iter_mut()
        .zip(residual[..expected].iter())
    {
        *out += residual;
    }
    Ok(())
}

/// Apply SwiGLU independently to each row, storing the result in `gate`.
pub fn swiglu_rows(gate: &mut [f32], up: &[f32], rows: usize, hidden_dim: usize) -> Result<()> {
    let expected = validate_row_buffers(gate, up, rows, hidden_dim)?;
    for row in 0..rows {
        let start = row * hidden_dim;
        mivi_core::math::swiglu(
            &mut gate[start..start + hidden_dim],
            &up[start..start + hidden_dim],
        );
    }
    debug_assert_eq!(expected, rows * hidden_dim);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_rows_use_row_major_offsets() {
        let mut tile = TileActivations::new(4, 3, 5).unwrap();
        tile.current_mut()[3..6].copy_from_slice(&[1.0, 2.0, 3.0]);

        assert_eq!(tile.current_row(1).unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(tile.current_row(2).unwrap(), &[0.0, 0.0, 0.0]);
        assert!(tile.current_row(4).is_err());
    }

    #[test]
    fn tile_supports_required_lengths_and_final_row_selection() {
        for tile_tokens in [1, 2, 64] {
            let tile = TileActivations::new(tile_tokens, 4, 8).unwrap();
            assert_eq!(tile.current().len(), tile_tokens * 4);
            assert_eq!(tile.gate().len(), tile_tokens * 8);
            assert_eq!(tile.final_current_row(tile_tokens).unwrap().len(), 4);
        }
    }

    #[test]
    fn tile_rejects_invalid_dimensions_and_rows() {
        assert!(TileActivations::new(0, 4, 8).is_err());
        assert!(TileActivations::new(4, 0, 8).is_err());
        assert!(TileActivations::new(4, 4, 0).is_err());

        let tile = TileActivations::new(2, 4, 8).unwrap();
        assert!(tile.final_current_row(0).is_err());
        assert!(tile.final_current_row(3).is_err());
    }

    #[test]
    fn row_helpers_apply_norm_residual_and_swiglu() {
        let input = [3.0, 4.0, 0.0, 0.0];
        let mut normalized = [0.0; 4];
        rms_norm_rows(&mut normalized, &input, &[1.0; 4], 1, 4, 1e-5).unwrap();
        assert!((normalized[0] - 1.2).abs() < 0.01);
        assert!((normalized[1] - 1.6).abs() < 0.01);

        add_rows_in_place(&mut normalized, &[1.0; 4], 1, 4).unwrap();
        assert!((normalized[0] - 2.2).abs() < 0.01);

        let mut gate = [1.0, -1.0];
        swiglu_rows(&mut gate, &[2.0, 3.0], 1, 2).unwrap();
        assert!(gate[0] > 1.4 && gate[0] < 1.5);
        assert!(gate[1] < -0.8 && gate[1] > -1.0);
    }
}
