//! Contiguous preallocated Key-Value Cache with strict validation.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum KvError {
    #[error("Context length exceeded: attempted pos {pos} >= max_seq_len {max}")]
    ContextOverflow { pos: usize, max: usize },
    #[error("Layer index out of bounds: {layer} >= {max}")]
    InvalidLayer { layer: usize, max: usize },
    #[error("Dimension mismatch: expected kv_dim {expected}, got {got}")]
    DimMismatch { expected: usize, got: usize },
    #[error(
        "KV cache allocation overflow: {n_layers} × {max_seq_len} × {kv_dim} exceeds usize limits"
    )]
    AllocationOverflow {
        n_layers: usize,
        max_seq_len: usize,
        kv_dim: usize,
    },
}

pub type Result<T> = std::result::Result<T, KvError>;

/// Contiguous flat KV cache preallocated for all attention layers up to `max_seq_len`.
#[derive(Debug)]
pub struct KvCache {
    n_layers: usize,
    max_seq_len: usize,
    kv_dim: usize,
    // Flat buffer: [n_layers, max_seq_len, kv_dim]
    k_cache: Box<[f32]>,
    // Flat buffer: [n_layers, max_seq_len, kv_dim]
    v_cache: Box<[f32]>,
    current_pos: usize,
}

impl KvCache {
    pub fn try_new(n_layers: usize, max_seq_len: usize, kv_dim: usize) -> Result<Self> {
        let total_elements = n_layers
            .checked_mul(max_seq_len)
            .and_then(|v| v.checked_mul(kv_dim))
            .ok_or(KvError::AllocationOverflow {
                n_layers,
                max_seq_len,
                kv_dim,
            })?;
        Ok(Self {
            n_layers,
            max_seq_len,
            kv_dim,
            k_cache: vec![0.0f32; total_elements].into_boxed_slice(),
            v_cache: vec![0.0f32; total_elements].into_boxed_slice(),
            current_pos: 0,
        })
    }

    /// Allocates a new KV cache.
    ///
    /// # Panics
    /// Panics if total cache elements overflow `usize`. Prefer `try_new` in fallible contexts.
    #[track_caller]
    pub fn new(n_layers: usize, max_seq_len: usize, kv_dim: usize) -> Self {
        Self::try_new(n_layers, max_seq_len, kv_dim).expect("Failed to allocate KV cache")
    }

    #[inline]
    pub fn current_pos(&self) -> usize {
        self.current_pos
    }

    #[inline]
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Reset cache position to start of sequence without expensive memory clearing.
    pub fn reset(&mut self) {
        self.current_pos = 0;
    }

    #[inline]
    fn checked_offset(&self, layer: usize, pos: usize) -> Result<usize> {
        if layer >= self.n_layers {
            return Err(KvError::InvalidLayer {
                layer,
                max: self.n_layers,
            });
        }
        if pos >= self.max_seq_len {
            return Err(KvError::ContextOverflow {
                pos,
                max: self.max_seq_len,
            });
        }
        Ok((layer * self.max_seq_len + pos) * self.kv_dim)
    }

    #[inline]
    fn validate_dim(&self, slice: &[f32]) -> Result<()> {
        if slice.len() != self.kv_dim {
            return Err(KvError::DimMismatch {
                expected: self.kv_dim,
                got: slice.len(),
            });
        }
        Ok(())
    }

    /// Store Key and Value vectors for a given layer at current position.
    #[inline]
    pub fn store(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) -> Result<()> {
        let offset = self.checked_offset(layer, pos)?;
        self.validate_dim(k)?;
        self.validate_dim(v)?;

        self.k_cache[offset..offset + self.kv_dim].copy_from_slice(k);
        self.v_cache[offset..offset + self.kv_dim].copy_from_slice(v);

        if pos >= self.current_pos {
            self.current_pos = pos + 1;
        }

        Ok(())
    }

    /// Read Key vector slice for a given layer and position.
    #[inline]
    pub fn get_k(&self, layer: usize, pos: usize) -> Result<&[f32]> {
        let offset = self.checked_offset(layer, pos)?;
        Ok(&self.k_cache[offset..offset + self.kv_dim])
    }

    /// Read Value vector slice for a given layer and position.
    #[inline]
    pub fn get_v(&self, layer: usize, pos: usize) -> Result<&[f32]> {
        let offset = self.checked_offset(layer, pos)?;
        Ok(&self.v_cache[offset..offset + self.kv_dim])
    }

    /// Read Key vector slice without runtime bounds checks.
    ///
    /// # Safety
    /// Caller must ensure `layer < n_layers` and `pos < max_seq_len`.
    #[inline]
    pub unsafe fn get_k_unchecked(&self, layer: usize, pos: usize) -> &[f32] {
        let offset = (layer * self.max_seq_len + pos) * self.kv_dim;
        &self.k_cache[offset..offset + self.kv_dim]
    }

    /// Read Value vector slice without runtime bounds checks.
    ///
    /// # Safety
    /// Caller must ensure `layer < n_layers` and `pos < max_seq_len`.
    #[inline]
    pub unsafe fn get_v_unchecked(&self, layer: usize, pos: usize) -> &[f32] {
        let offset = (layer * self.max_seq_len + pos) * self.kv_dim;
        &self.v_cache[offset..offset + self.kv_dim]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_overflow() {
        let mut kv = KvCache::new(2, 2, 4);
        let k = [1.0, 2.0, 3.0, 4.0];
        let v = [5.0, 6.0, 7.0, 8.0];

        assert!(kv.store(0, 0, &k, &v).is_ok());
        assert!(kv.store(0, 1, &k, &v).is_ok());

        let err = kv.store(0, 2, &k, &v);
        assert!(matches!(
            err,
            Err(KvError::ContextOverflow { pos: 2, max: 2 })
        ));
    }

    #[test]
    fn test_kv_cache_try_new_overflow() {
        let err = KvCache::try_new(usize::MAX / 2, usize::MAX / 2, 4);
        assert!(matches!(err, Err(KvError::AllocationOverflow { .. })));
    }
}
