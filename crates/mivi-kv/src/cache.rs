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
    pub fn new(n_layers: usize, max_seq_len: usize, kv_dim: usize) -> Self {
        let total_elements = n_layers * max_seq_len * kv_dim;
        Self {
            n_layers,
            max_seq_len,
            kv_dim,
            k_cache: vec![0.0f32; total_elements].into_boxed_slice(),
            v_cache: vec![0.0f32; total_elements].into_boxed_slice(),
            current_pos: 0,
        }
    }

    #[inline]
    pub fn current_pos(&self) -> usize {
        self.current_pos
    }

    #[inline]
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Reset cache position to start of sequence.
    pub fn reset(&mut self) {
        self.current_pos = 0;
        self.k_cache.fill(0.0);
        self.v_cache.fill(0.0);
    }

    /// Store Key and Value vectors for a given layer at current position.
    #[inline]
    pub fn store(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) -> Result<()> {
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
        if k.len() != self.kv_dim {
            return Err(KvError::DimMismatch {
                expected: self.kv_dim,
                got: k.len(),
            });
        }
        if v.len() != self.kv_dim {
            return Err(KvError::DimMismatch {
                expected: self.kv_dim,
                got: v.len(),
            });
        }

        let offset = (layer * self.max_seq_len + pos) * self.kv_dim;
        self.k_cache[offset..offset + self.kv_dim].copy_from_slice(k);
        self.v_cache[offset..offset + self.kv_dim].copy_from_slice(v);

        if pos >= self.current_pos {
            self.current_pos = pos + 1;
        }

        Ok(())
    }

    /// Read Key vector slice for a given layer and position.
    #[inline]
    pub fn get_k(&self, layer: usize, pos: usize) -> &[f32] {
        assert!(layer < self.n_layers, "Layer index out of bounds");
        assert!(pos < self.max_seq_len, "Position index out of bounds");
        let offset = (layer * self.max_seq_len + pos) * self.kv_dim;
        &self.k_cache[offset..offset + self.kv_dim]
    }

    /// Read Value vector slice for a given layer and position.
    #[inline]
    pub fn get_v(&self, layer: usize, pos: usize) -> &[f32] {
        assert!(layer < self.n_layers, "Layer index out of bounds");
        assert!(pos < self.max_seq_len, "Position index out of bounds");
        let offset = (layer * self.max_seq_len + pos) * self.kv_dim;
        &self.v_cache[offset..offset + self.kv_dim]
    }
}
