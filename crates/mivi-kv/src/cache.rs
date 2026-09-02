use mivi_quant::q8_0::{quantize_f32_to_q8_0_block, Q8_0_BLOCK_SIZE, Q8_0_BYTES};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvPrecision {
    /// Full 32-bit floating point precision (4.0 bytes per element).
    F32,
    /// 8-bit block-quantized precision with f16 scale (34 bytes per 32 elements = 1.0625 bytes per element).
    Q8_0,
}

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
    #[error("Unsupported operation for KV precision {0:?}")]
    UnsupportedPrecision(KvPrecision),
}

pub type Result<T> = std::result::Result<T, KvError>;

/// Contiguous flat KV cache preallocated for attention layers up to `max_seq_len`.
#[derive(Debug)]
pub struct KvCache {
    n_layers: usize,
    max_seq_len: usize,
    kv_dim: usize,
    precision: KvPrecision,
    layer_map: Vec<usize>,
    // Flat buffer for F32: [n_allocated_layers, max_seq_len, kv_dim]
    k_cache: Box<[f32]>,
    v_cache: Box<[f32]>,
    // Flat buffer for Q8_0: [n_allocated_layers, max_seq_len, blocks_per_token * 34]
    k_q8_cache: Box<[u8]>,
    v_q8_cache: Box<[u8]>,
    current_pos: usize,
}

impl KvCache {
    /// Allocate KV cache for all layers 0..n_layers with default FP32 precision.
    pub fn try_new(n_layers: usize, max_seq_len: usize, kv_dim: usize) -> Result<Self> {
        if n_layers > 65536 {
            return Err(KvError::AllocationOverflow {
                n_layers,
                max_seq_len,
                kv_dim,
            });
        }
        let all_layers: Vec<usize> = (0..n_layers).collect();
        Self::try_new_selective_with_precision(
            n_layers,
            max_seq_len,
            kv_dim,
            &all_layers,
            KvPrecision::F32,
        )
    }

    /// Allocate KV cache only for the specified attention layers with FP32 precision.
    pub fn try_new_selective(
        n_layers: usize,
        max_seq_len: usize,
        kv_dim: usize,
        attention_layers: &[usize],
    ) -> Result<Self> {
        Self::try_new_selective_with_precision(
            n_layers,
            max_seq_len,
            kv_dim,
            attention_layers,
            KvPrecision::F32,
        )
    }

    /// Allocate KV cache for the specified attention layers with explicit precision (F32 or Q8_0).
    pub fn try_new_selective_with_precision(
        n_layers: usize,
        max_seq_len: usize,
        kv_dim: usize,
        attention_layers: &[usize],
        precision: KvPrecision,
    ) -> Result<Self> {
        if n_layers > 65536 {
            return Err(KvError::AllocationOverflow {
                n_layers,
                max_seq_len,
                kv_dim,
            });
        }
        let n_attn = attention_layers.len();
        let mut layer_map = vec![usize::MAX; n_layers];
        for (cache_idx, &layer_idx) in attention_layers.iter().enumerate() {
            if layer_idx < n_layers {
                layer_map[layer_idx] = cache_idx;
            }
        }

        match precision {
            KvPrecision::F32 => {
                let total_elements = n_attn
                    .checked_mul(max_seq_len)
                    .and_then(|v| v.checked_mul(kv_dim))
                    .ok_or(KvError::AllocationOverflow {
                        n_layers: n_attn,
                        max_seq_len,
                        kv_dim,
                    })?;
                Ok(Self {
                    n_layers,
                    max_seq_len,
                    kv_dim,
                    precision,
                    layer_map,
                    k_cache: vec![0.0f32; total_elements].into_boxed_slice(),
                    v_cache: vec![0.0f32; total_elements].into_boxed_slice(),
                    k_q8_cache: vec![0u8; 0].into_boxed_slice(),
                    v_q8_cache: vec![0u8; 0].into_boxed_slice(),
                    current_pos: 0,
                })
            }
            KvPrecision::Q8_0 => {
                let blocks_per_token = (kv_dim + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
                let bytes_per_token = blocks_per_token * Q8_0_BYTES;
                let total_bytes = n_attn
                    .checked_mul(max_seq_len)
                    .and_then(|v| v.checked_mul(bytes_per_token))
                    .ok_or(KvError::AllocationOverflow {
                        n_layers: n_attn,
                        max_seq_len,
                        kv_dim,
                    })?;
                Ok(Self {
                    n_layers,
                    max_seq_len,
                    kv_dim,
                    precision,
                    layer_map,
                    k_cache: vec![0.0f32; 0].into_boxed_slice(),
                    v_cache: vec![0.0f32; 0].into_boxed_slice(),
                    k_q8_cache: vec![0u8; total_bytes].into_boxed_slice(),
                    v_q8_cache: vec![0u8; total_bytes].into_boxed_slice(),
                    current_pos: 0,
                })
            }
        }
    }

    /// Allocates a new KV cache.
    #[track_caller]
    pub fn new(n_layers: usize, max_seq_len: usize, kv_dim: usize) -> Self {
        Self::try_new(n_layers, max_seq_len, kv_dim).expect("Failed to allocate KV cache")
    }

    #[inline]
    pub fn precision(&self) -> KvPrecision {
        self.precision
    }

    #[inline]
    pub fn current_pos(&self) -> usize {
        self.current_pos
    }

    #[inline]
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Returns the maximum sequence capacity in tokens.
    #[inline]
    pub fn capacity_tokens(&self) -> usize {
        self.max_seq_len
    }

    /// Returns the exact memory footprint in bytes allocated for this KV cache.
    #[inline]
    pub fn memory_bytes(&self) -> usize {
        match self.precision {
            KvPrecision::F32 => {
                (self.k_cache.len() + self.v_cache.len()) * std::mem::size_of::<f32>()
            }
            KvPrecision::Q8_0 => self.k_q8_cache.len() + self.v_q8_cache.len(),
        }
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
        let cache_layer = if layer < self.layer_map.len() {
            self.layer_map[layer]
        } else {
            layer
        };
        if cache_layer == usize::MAX {
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
        Ok((cache_layer * self.max_seq_len + pos) * self.kv_dim)
    }

    #[inline]
    fn checked_q8_offset(&self, layer: usize, pos: usize) -> Result<usize> {
        if layer >= self.n_layers {
            return Err(KvError::InvalidLayer {
                layer,
                max: self.n_layers,
            });
        }
        let cache_layer = if layer < self.layer_map.len() {
            self.layer_map[layer]
        } else {
            layer
        };
        if cache_layer == usize::MAX {
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
        let blocks_per_token = (self.kv_dim + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
        let bytes_per_token = blocks_per_token * Q8_0_BYTES;
        Ok((cache_layer * self.max_seq_len + pos) * bytes_per_token)
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
        self.validate_dim(k)?;
        self.validate_dim(v)?;

        match self.precision {
            KvPrecision::F32 => {
                let offset = self.checked_offset(layer, pos)?;
                self.k_cache[offset..offset + self.kv_dim].copy_from_slice(k);
                self.v_cache[offset..offset + self.kv_dim].copy_from_slice(v);
            }
            KvPrecision::Q8_0 => {
                let offset = self.checked_q8_offset(layer, pos)?;
                let blocks = (self.kv_dim + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
                for b in 0..blocks {
                    let f32_start = b * Q8_0_BLOCK_SIZE;
                    let f32_end = (f32_start + Q8_0_BLOCK_SIZE).min(self.kv_dim);
                    let block_offset = offset + b * Q8_0_BYTES;

                    let mut k_buf = [0.0f32; Q8_0_BLOCK_SIZE];
                    k_buf[..f32_end - f32_start].copy_from_slice(&k[f32_start..f32_end]);
                    quantize_f32_to_q8_0_block(
                        &k_buf,
                        &mut self.k_q8_cache[block_offset..block_offset + Q8_0_BYTES],
                    );

                    let mut v_buf = [0.0f32; Q8_0_BLOCK_SIZE];
                    v_buf[..f32_end - f32_start].copy_from_slice(&v[f32_start..f32_end]);
                    quantize_f32_to_q8_0_block(
                        &v_buf,
                        &mut self.v_q8_cache[block_offset..block_offset + Q8_0_BYTES],
                    );
                }
            }
        }

        if pos >= self.current_pos {
            self.current_pos = pos + 1;
        }

        Ok(())
    }

    /// Read Key vector slice for a given layer and position (only available for FP32 precision).
    #[inline]
    pub fn get_k(&self, layer: usize, pos: usize) -> Result<&[f32]> {
        if self.precision != KvPrecision::F32 {
            return Err(KvError::UnsupportedPrecision(self.precision));
        }
        let offset = self.checked_offset(layer, pos)?;
        Ok(&self.k_cache[offset..offset + self.kv_dim])
    }

    /// Read Value vector slice for a given layer and position (only available for FP32 precision).
    #[inline]
    pub fn get_v(&self, layer: usize, pos: usize) -> Result<&[f32]> {
        if self.precision != KvPrecision::F32 {
            return Err(KvError::UnsupportedPrecision(self.precision));
        }
        let offset = self.checked_offset(layer, pos)?;
        Ok(&self.v_cache[offset..offset + self.kv_dim])
    }

    /// Read Key vector slice without runtime bounds checks (FP32).
    ///
    /// # Safety
    /// Caller must ensure `self.precision == KvPrecision::F32`, `layer < n_layers`, `layer` is an attention layer, and `pos < max_seq_len`.
    #[inline]
    pub unsafe fn get_k_unchecked(&self, layer: usize, pos: usize) -> &[f32] {
        let cache_layer = *self.layer_map.get_unchecked(layer);
        let offset = (cache_layer * self.max_seq_len + pos) * self.kv_dim;
        std::slice::from_raw_parts(self.k_cache.as_ptr().add(offset), self.kv_dim)
    }

    /// Read Value vector slice without runtime bounds checks (FP32).
    ///
    /// # Safety
    /// Caller must ensure `self.precision == KvPrecision::F32`, `layer < n_layers`, `layer` is an attention layer, and `pos < max_seq_len`.
    #[inline]
    pub unsafe fn get_v_unchecked(&self, layer: usize, pos: usize) -> &[f32] {
        let cache_layer = *self.layer_map.get_unchecked(layer);
        let offset = (cache_layer * self.max_seq_len + pos) * self.kv_dim;
        std::slice::from_raw_parts(self.v_cache.as_ptr().add(offset), self.kv_dim)
    }

    /// Read Q8_0 Key block slice for a given layer, token position, and block index.
    #[inline]
    pub fn get_k_q8_block(&self, layer: usize, pos: usize, block_idx: usize) -> Result<&[u8]> {
        if self.precision != KvPrecision::Q8_0 {
            return Err(KvError::UnsupportedPrecision(self.precision));
        }
        let offset = self.checked_q8_offset(layer, pos)?;
        let block_offset = offset + block_idx * Q8_0_BYTES;
        if block_offset + Q8_0_BYTES <= self.k_q8_cache.len() {
            Ok(&self.k_q8_cache[block_offset..block_offset + Q8_0_BYTES])
        } else {
            Err(KvError::ContextOverflow {
                pos,
                max: self.max_seq_len,
            })
        }
    }

    /// Read Q8_0 Key block slice without runtime bounds checks.
    ///
    /// # Safety
    /// Caller must ensure `self.precision == KvPrecision::Q8_0`, `layer` is a valid attention layer, `pos < max_seq_len`, and `block_idx < blocks_per_token`.
    #[inline]
    pub unsafe fn get_k_q8_block_unchecked(
        &self,
        layer: usize,
        pos: usize,
        block_idx: usize,
    ) -> &[u8] {
        let cache_layer = *self.layer_map.get_unchecked(layer);
        let blocks_per_token = (self.kv_dim + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
        let bytes_per_token = blocks_per_token * Q8_0_BYTES;
        let offset =
            (cache_layer * self.max_seq_len + pos) * bytes_per_token + block_idx * Q8_0_BYTES;
        std::slice::from_raw_parts(self.k_q8_cache.as_ptr().add(offset), Q8_0_BYTES)
    }

    /// Read Q8_0 Value block slice without runtime bounds checks.
    ///
    /// # Safety
    /// Caller must ensure `self.precision == KvPrecision::Q8_0`, `layer` is a valid attention layer, `pos < max_seq_len`, and `block_idx < blocks_per_token`.
    #[inline]
    pub unsafe fn get_v_q8_block_unchecked(
        &self,
        layer: usize,
        pos: usize,
        block_idx: usize,
    ) -> &[u8] {
        let cache_layer = *self.layer_map.get_unchecked(layer);
        let blocks_per_token = (self.kv_dim + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
        let bytes_per_token = blocks_per_token * Q8_0_BYTES;
        let offset =
            (cache_layer * self.max_seq_len + pos) * bytes_per_token + block_idx * Q8_0_BYTES;
        std::slice::from_raw_parts(self.v_q8_cache.as_ptr().add(offset), Q8_0_BYTES)
    }

    /// Returns the number of layers allocated in this KV cache.
    #[inline]
    pub fn n_allocated_layers(&self) -> usize {
        let mut max_idx = 0;
        let mut count = 0;
        for &idx in &self.layer_map {
            if idx != usize::MAX {
                count += 1;
                max_idx = max_idx.max(idx + 1);
            }
        }
        max_idx.max(count)
    }

    /// Export the KV cache memory up to `pos` tokens for state snapshotting.
    pub fn export_state(&self, pos: usize) -> (Vec<f32>, Vec<f32>) {
        let n_alloc = self.n_allocated_layers();
        let target_pos = pos.min(self.max_seq_len);
        let mut k_out = Vec::with_capacity(n_alloc * target_pos * self.kv_dim);
        let mut v_out = Vec::with_capacity(n_alloc * target_pos * self.kv_dim);

        for cache_layer in 0..n_alloc {
            let start = cache_layer * self.max_seq_len * self.kv_dim;
            let end = start + target_pos * self.kv_dim;
            if end <= self.k_cache.len() {
                k_out.extend_from_slice(&self.k_cache[start..end]);
                v_out.extend_from_slice(&self.v_cache[start..end]);
            }
        }

        (k_out, v_out)
    }

    /// Import a previously exported KV cache memory slice up to `pos` tokens.
    pub fn import_state(&mut self, pos: usize, k_data: &[f32], v_data: &[f32]) -> Result<()> {
        let n_alloc = self.n_allocated_layers();
        let target_pos = pos.min(self.max_seq_len);
        let expected_elements = n_alloc * target_pos * self.kv_dim;

        if k_data.len() != expected_elements || v_data.len() != expected_elements {
            return Err(KvError::DimMismatch {
                expected: expected_elements,
                got: k_data.len(),
            });
        }

        for cache_layer in 0..n_alloc {
            let start = cache_layer * self.max_seq_len * self.kv_dim;
            let end = start + target_pos * self.kv_dim;
            let src_start = cache_layer * target_pos * self.kv_dim;
            let src_end = src_start + target_pos * self.kv_dim;

            self.k_cache[start..end].copy_from_slice(&k_data[src_start..src_end]);
            self.v_cache[start..end].copy_from_slice(&v_data[src_start..src_end]);
        }

        self.current_pos = target_pos;
        Ok(())
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

    #[test]
    fn test_kv_cache_selective_layers() {
        let mut kv = KvCache::try_new_selective(16, 128, 64, &[2, 5, 8]).unwrap();
        let k = vec![1.0; 64];
        let v = vec![2.0; 64];

        // Attention layers succeed
        assert!(kv.store(2, 0, &k, &v).is_ok());
        assert!(kv.store(5, 0, &k, &v).is_ok());
        assert!(kv.store(8, 0, &k, &v).is_ok());

        // SSM layers fail (no KV allocated)
        assert!(kv.store(0, 0, &k, &v).is_err());
        assert!(kv.store(1, 0, &k, &v).is_err());
        assert!(kv.store(3, 0, &k, &v).is_err());
    }

    #[test]
    fn test_kv_cache_64k_memory_calculation() {
        // 16 layers (6 attention layers), 65,536 tokens (64k), kv_dim=512
        let attention_layers = vec![2, 5, 8, 11, 13, 15];
        let kv = KvCache::try_new_selective(16, 65536, 512, &attention_layers).unwrap();

        assert_eq!(kv.capacity_tokens(), 65536);
        assert_eq!(kv.n_allocated_layers(), 6);
        // 6 layers * 65536 tokens * 512 dim * 4 bytes * 2 (K+V) = 1,610,612,736 bytes (~1.61 GB in F32)
        assert_eq!(kv.memory_bytes(), 6 * 65536 * 512 * 4 * 2);
    }

    #[test]
    fn test_kv_cache_q8_0_memory_calculation_and_store() {
        // 16 layers (6 attention layers), 65,536 tokens (64k), kv_dim=512
        let attention_layers = vec![2, 5, 8, 11, 13, 15];
        let mut kv = KvCache::try_new_selective_with_precision(
            16,
            65536,
            512,
            &attention_layers,
            KvPrecision::Q8_0,
        )
        .unwrap();

        assert_eq!(kv.precision(), KvPrecision::Q8_0);
        assert_eq!(kv.capacity_tokens(), 65536);
        // 512 dim / 32 = 16 blocks * 34 bytes = 544 bytes per token
        // 6 layers * 65536 tokens * 544 bytes * 2 (K+V) = 427,819,008 bytes (~427 MB in Q8_0 vs 1.61 GB in F32)
        assert_eq!(kv.memory_bytes(), 6 * 65536 * (16 * 34) * 2);

        let k = vec![1.5f32; 512];
        let v = vec![-0.75f32; 512];
        assert!(kv.store(2, 0, &k, &v).is_ok());

        let k_block = kv.get_k_q8_block(2, 0, 0).expect("Should get k block 0");
        assert_eq!(k_block.len(), 34);

        let dot = mivi_quant::q8_0::dot_q8_0_f32(&k[0..32], k_block);
        let expected_dot: f32 = k[0..32].iter().map(|x| x * x).sum();
        assert!((dot - expected_dot).abs() < 0.5);
    }
}
