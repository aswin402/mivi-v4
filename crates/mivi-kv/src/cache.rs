use mivi_core::{TurboQuant2Bit, TurboQuant4Bit};
use mivi_quant::q8_0::{quantize_f32_to_q8_0_block, Q8_0_BLOCK_SIZE, Q8_0_BYTES};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvPrecision {
    /// Full 32-bit floating point precision (4.0 bytes per element).
    F32,
    /// 8-bit block-quantized precision with f16 scale (34 bytes per 32 elements = 1.0625 bytes per element).
    Q8_0,
    /// TurboQuant 4-bit data-oblivious quantization with orthogonal rotation (0.5 bytes per element + 4 bytes norm).
    TurboQuant4,
    /// TurboQuant 2-bit data-oblivious quantization with orthogonal rotation (0.25 bytes per element + 4 bytes norm).
    TurboQuant2,
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
    // Flat buffer for Q8_0 / TurboQuant: [n_allocated_layers, max_seq_len, bytes_per_token]
    k_q8_cache: Box<[u8]>,
    v_q8_cache: Box<[u8]>,
    tq4: Option<TurboQuant4Bit>,
    tq2: Option<TurboQuant2Bit>,
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
                    tq4: None,
                    tq2: None,
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
                    tq4: None,
                    tq2: None,
                    current_pos: 0,
                })
            }
            KvPrecision::TurboQuant4 => {
                let bytes_per_token = 4 + (kv_dim + 1) / 2;
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
                    tq4: Some(TurboQuant4Bit::new(kv_dim)),
                    tq2: None,
                    current_pos: 0,
                })
            }
            KvPrecision::TurboQuant2 => {
                let bytes_per_token = 4 + (kv_dim + 3) / 4;
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
                    tq4: None,
                    tq2: Some(TurboQuant2Bit::new(kv_dim)),
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
    pub fn tq4(&self) -> Option<&TurboQuant4Bit> {
        self.tq4.as_ref()
    }

    #[inline]
    pub fn tq2(&self) -> Option<&TurboQuant2Bit> {
        self.tq2.as_ref()
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
            KvPrecision::Q8_0 | KvPrecision::TurboQuant4 | KvPrecision::TurboQuant2 => {
                self.k_q8_cache.len() + self.v_q8_cache.len()
            }
        }
    }

    /// Returns the storage size in bytes allocated per token for this precision mode.
    #[inline]
    pub fn bytes_per_token(&self) -> usize {
        match self.precision {
            KvPrecision::F32 => self.kv_dim * std::mem::size_of::<f32>(),
            KvPrecision::Q8_0 => {
                let blocks_per_head = 32;
                let num_blocks = (self.kv_dim + blocks_per_head - 1) / blocks_per_head;
                num_blocks * 34
            }
            KvPrecision::TurboQuant4 => 4 + (self.kv_dim + 1) / 2,
            KvPrecision::TurboQuant2 => 4 + (self.kv_dim + 3) / 4,
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
    fn checked_tq_offset(&self, layer: usize, pos: usize) -> Result<usize> {
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
        let bytes_per_token = match self.precision {
            KvPrecision::TurboQuant4 => 4 + (self.kv_dim + 1) / 2,
            KvPrecision::TurboQuant2 => 4 + (self.kv_dim + 3) / 4,
            _ => return Err(KvError::UnsupportedPrecision(self.precision)),
        };
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
            KvPrecision::TurboQuant4 => {
                let offset = self.checked_tq_offset(layer, pos)?;
                let bytes_per_token = 4 + (self.kv_dim + 1) / 2;
                let tq = self.tq4.as_ref().unwrap();

                let (norm_k, packed_k) = tq.quantize(k);
                self.k_q8_cache[offset..offset + 4].copy_from_slice(&norm_k.to_le_bytes());
                self.k_q8_cache[offset + 4..offset + bytes_per_token].copy_from_slice(&packed_k);

                let (norm_v, packed_v) = tq.quantize(v);
                self.v_q8_cache[offset..offset + 4].copy_from_slice(&norm_v.to_le_bytes());
                self.v_q8_cache[offset + 4..offset + bytes_per_token].copy_from_slice(&packed_v);
            }
            KvPrecision::TurboQuant2 => {
                let offset = self.checked_tq_offset(layer, pos)?;
                let bytes_per_token = 4 + (self.kv_dim + 3) / 4;
                let tq = self.tq2.as_ref().unwrap();

                let (norm_k, packed_k) = tq.quantize(k);
                self.k_q8_cache[offset..offset + 4].copy_from_slice(&norm_k.to_le_bytes());
                self.k_q8_cache[offset + 4..offset + bytes_per_token].copy_from_slice(&packed_k);

                let (norm_v, packed_v) = tq.quantize(v);
                self.v_q8_cache[offset..offset + 4].copy_from_slice(&norm_v.to_le_bytes());
                self.v_q8_cache[offset + 4..offset + bytes_per_token].copy_from_slice(&packed_v);
            }
        }

        if pos >= self.current_pos {
            self.current_pos = pos + 1;
        }

        Ok(())
    }

    /// Read TurboQuant4 Key packed slice without runtime bounds checks.
    ///
    /// # Safety
    /// Caller must ensure `self.precision == KvPrecision::TurboQuant4`, `layer` is a valid attention layer, and `pos < max_seq_len`.
    #[inline]
    pub unsafe fn get_k_tq4_packed_unchecked(&self, layer: usize, pos: usize) -> (f32, &[u8]) {
        let cache_layer = *self.layer_map.get_unchecked(layer);
        let bytes_per_token = 4 + (self.kv_dim + 1) / 2;
        let offset = (cache_layer * self.max_seq_len + pos) * bytes_per_token;
        let mut norm_bytes = [0u8; 4];
        std::ptr::copy_nonoverlapping(
            self.k_q8_cache.as_ptr().add(offset),
            norm_bytes.as_mut_ptr(),
            4,
        );
        let norm = f32::from_le_bytes(norm_bytes);
        let packed = std::slice::from_raw_parts(
            self.k_q8_cache.as_ptr().add(offset + 4),
            (self.kv_dim + 1) / 2,
        );
        (norm, packed)
    }

    /// Read and dequantize TurboQuant4 Value vector into `out_v` without runtime bounds checks.
    ///
    /// # Safety
    /// Caller must ensure `self.precision == KvPrecision::TurboQuant4`, `layer` is a valid attention layer, and `pos < max_seq_len`.
    #[inline]
    pub unsafe fn get_v_tq4_dequantized_unchecked(
        &self,
        layer: usize,
        pos: usize,
        out_v: &mut [f32],
    ) {
        let cache_layer = *self.layer_map.get_unchecked(layer);
        let bytes_per_token = 4 + (self.kv_dim + 1) / 2;
        let offset = (cache_layer * self.max_seq_len + pos) * bytes_per_token;
        let mut norm_bytes = [0u8; 4];
        std::ptr::copy_nonoverlapping(
            self.v_q8_cache.as_ptr().add(offset),
            norm_bytes.as_mut_ptr(),
            4,
        );
        let norm = f32::from_le_bytes(norm_bytes);
        let packed = std::slice::from_raw_parts(
            self.v_q8_cache.as_ptr().add(offset + 4),
            (self.kv_dim + 1) / 2,
        );
        if let Some(ref tq) = self.tq4 {
            tq.dequantize(norm, packed, out_v);
        }
    }

    /// Read TurboQuant2 Key packed slice without runtime bounds checks.
    ///
    /// # Safety
    /// Caller must ensure `self.precision == KvPrecision::TurboQuant2`, `layer` is a valid attention layer, and `pos < max_seq_len`.
    #[inline]
    pub unsafe fn get_k_tq2_packed_unchecked(&self, layer: usize, pos: usize) -> (f32, &[u8]) {
        let cache_layer = *self.layer_map.get_unchecked(layer);
        let bytes_per_token = 4 + (self.kv_dim + 3) / 4;
        let offset = (cache_layer * self.max_seq_len + pos) * bytes_per_token;
        let mut norm_bytes = [0u8; 4];
        std::ptr::copy_nonoverlapping(
            self.k_q8_cache.as_ptr().add(offset),
            norm_bytes.as_mut_ptr(),
            4,
        );
        let norm = f32::from_le_bytes(norm_bytes);
        let packed = std::slice::from_raw_parts(
            self.k_q8_cache.as_ptr().add(offset + 4),
            (self.kv_dim + 3) / 4,
        );
        (norm, packed)
    }

    /// Read and dequantize TurboQuant2 Value vector into `out_v` without runtime bounds checks.
    ///
    /// # Safety
    /// Caller must ensure `self.precision == KvPrecision::TurboQuant2`, `layer` is a valid attention layer, and `pos < max_seq_len`.
    #[inline]
    pub unsafe fn get_v_tq2_dequantized_unchecked(
        &self,
        layer: usize,
        pos: usize,
        out_v: &mut [f32],
    ) {
        let cache_layer = *self.layer_map.get_unchecked(layer);
        let bytes_per_token = 4 + (self.kv_dim + 3) / 4;
        let offset = (cache_layer * self.max_seq_len + pos) * bytes_per_token;
        let mut norm_bytes = [0u8; 4];
        std::ptr::copy_nonoverlapping(
            self.v_q8_cache.as_ptr().add(offset),
            norm_bytes.as_mut_ptr(),
            4,
        );
        let norm = f32::from_le_bytes(norm_bytes);
        let packed = std::slice::from_raw_parts(
            self.v_q8_cache.as_ptr().add(offset + 4),
            (self.kv_dim + 3) / 4,
        );
        if let Some(ref tq) = self.tq2 {
            tq.dequantize(norm, packed, out_v);
        }
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
    pub fn export_state(&self, pos: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let n_alloc = self.n_allocated_layers();
        let target_pos = pos.min(self.max_seq_len);

        match self.precision {
            KvPrecision::F32 => {
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
                Ok((k_out, v_out))
            }
            KvPrecision::Q8_0 | KvPrecision::TurboQuant4 | KvPrecision::TurboQuant2 => {
                let bpt = self.bytes_per_token();
                let f32_per_token = (bpt + 3) / 4;
                let mut k_out = vec![0.0f32; n_alloc * target_pos * f32_per_token];
                let mut v_out = vec![0.0f32; n_alloc * target_pos * f32_per_token];

                let k_out_bytes: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(k_out.as_mut_ptr() as *mut u8, k_out.len() * 4)
                };
                let v_out_bytes: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(v_out.as_mut_ptr() as *mut u8, v_out.len() * 4)
                };

                for cache_layer in 0..n_alloc {
                    let start = cache_layer * self.max_seq_len * bpt;
                    let end = start + target_pos * bpt;
                    let dst_start = cache_layer * target_pos * bpt;
                    let dst_end = dst_start + target_pos * bpt;

                    if end <= self.k_q8_cache.len() {
                        k_out_bytes[dst_start..dst_end].copy_from_slice(&self.k_q8_cache[start..end]);
                        v_out_bytes[dst_start..dst_end].copy_from_slice(&self.v_q8_cache[start..end]);
                    }
                }
                Ok((k_out, v_out))
            }
        }
    }

    /// Import a previously exported KV cache memory slice up to `pos` tokens.
    pub fn import_state(&mut self, pos: usize, k_data: &[f32], v_data: &[f32]) -> Result<()> {
        let n_alloc = self.n_allocated_layers();
        let target_pos = pos.min(self.max_seq_len);

        match self.precision {
            KvPrecision::F32 => {
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
            KvPrecision::Q8_0 | KvPrecision::TurboQuant4 | KvPrecision::TurboQuant2 => {
                let bpt = self.bytes_per_token();
                let f32_per_token = (bpt + 3) / 4;
                let expected_elements = n_alloc * target_pos * f32_per_token;

                if k_data.len() != expected_elements || v_data.len() != expected_elements {
                    return Err(KvError::DimMismatch {
                        expected: expected_elements,
                        got: k_data.len(),
                    });
                }

                let k_src_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(k_data.as_ptr() as *const u8, k_data.len() * 4)
                };
                let v_src_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(v_data.as_ptr() as *const u8, v_data.len() * 4)
                };

                for cache_layer in 0..n_alloc {
                    let start = cache_layer * self.max_seq_len * bpt;
                    let end = start + target_pos * bpt;
                    let src_start = cache_layer * target_pos * bpt;
                    let src_end = src_start + target_pos * bpt;

                    if end <= self.k_q8_cache.len() {
                        self.k_q8_cache[start..end].copy_from_slice(&k_src_bytes[src_start..src_end]);
                        self.v_q8_cache[start..end].copy_from_slice(&v_src_bytes[src_start..src_end]);
                    }
                }
                self.current_pos = target_pos;
                Ok(())
            }
        }
    }


    /// Import KV cache data at a specific absolute position without resetting current_pos.
    ///
    /// Copies `pos` tokens of cached KV data into the active KV cache at `target_pos`.
    /// Does NOT update current_pos. Used for continuation KV reuse across agent steps.
    pub fn import_state_at(
        &mut self,
        target_pos: usize,
        pos: usize,
        k_data: &[f32],
        v_data: &[f32],
    ) -> Result<()> {
        let n_alloc = self.n_allocated_layers();

        if target_pos > self.max_seq_len || pos > self.max_seq_len - target_pos {
            return Err(KvError::ContextOverflow {
                pos: target_pos.saturating_add(pos),
                max: self.max_seq_len,
            });
        }

        match self.precision {
            KvPrecision::F32 => {
                let expected_elements = n_alloc
                    .checked_mul(pos)
                    .and_then(|elements| elements.checked_mul(self.kv_dim))
                    .ok_or(KvError::AllocationOverflow {
                        n_layers: n_alloc,
                        max_seq_len: pos,
                        kv_dim: self.kv_dim,
                    })?;
                if k_data.len() != expected_elements || v_data.len() != expected_elements {
                    return Err(KvError::DimMismatch {
                        expected: expected_elements,
                        got: k_data.len(),
                    });
                }

                for cache_layer in 0..n_alloc {
                    let dst_start = cache_layer * self.max_seq_len * self.kv_dim + target_pos * self.kv_dim;
                    let dst_end = dst_start + pos * self.kv_dim;
                    let src_start = cache_layer * pos * self.kv_dim;
                    let src_end = src_start + pos * self.kv_dim;

                    self.k_cache[dst_start..dst_end].copy_from_slice(&k_data[src_start..src_end]);
                    self.v_cache[dst_start..dst_end].copy_from_slice(&v_data[src_start..src_end]);
                }
                Ok(())
            }
            KvPrecision::Q8_0 | KvPrecision::TurboQuant4 | KvPrecision::TurboQuant2 => {
                let bpt = self.bytes_per_token();
                let f32_per_token = (bpt + 3) / 4;
                let expected_elements = n_alloc
                    .checked_mul(pos)
                    .and_then(|elements| elements.checked_mul(f32_per_token))
                    .ok_or(KvError::AllocationOverflow {
                        n_layers: n_alloc,
                        max_seq_len: pos,
                        kv_dim: bpt,
                    })?;

                if k_data.len() != expected_elements || v_data.len() != expected_elements {
                    return Err(KvError::DimMismatch {
                        expected: expected_elements,
                        got: k_data.len(),
                    });
                }

                let k_src_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(k_data.as_ptr() as *const u8, k_data.len() * 4)
                };
                let v_src_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(v_data.as_ptr() as *const u8, v_data.len() * 4)
                };

                for cache_layer in 0..n_alloc {
                    let dst_start = cache_layer * self.max_seq_len * bpt + target_pos * bpt;
                    let dst_end = dst_start + pos * bpt;
                    let src_start = cache_layer * pos * bpt;
                    let src_end = src_start + pos * bpt;

                    if dst_end <= self.k_q8_cache.len() {
                        self.k_q8_cache[dst_start..dst_end].copy_from_slice(&k_src_bytes[src_start..src_end]);
                        self.v_q8_cache[dst_start..dst_end].copy_from_slice(&v_src_bytes[src_start..src_end]);
                    }
                }
                Ok(())
            }
        }
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
    fn test_import_state_at_rejects_destination_overflow() {
        let mut kv = KvCache::new(1, 4, 2);
        let k = vec![1.0; 2 * 2];
        let v = vec![2.0; 2 * 2];

        let result = kv.import_state_at(3, 2, &k, &v);

        assert!(matches!(result, Err(KvError::ContextOverflow { .. })));
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

    #[test]
    fn test_kv_cache_turboquant4_memory_and_store() {
        let attention_layers = vec![2, 5, 8, 11, 13, 15];
        let mut kv = KvCache::try_new_selective_with_precision(
            16,
            65536,
            512,
            &attention_layers,
            KvPrecision::TurboQuant4,
        )
        .unwrap();

        assert_eq!(kv.precision(), KvPrecision::TurboQuant4);
        assert_eq!(kv.capacity_tokens(), 65536);
        // 512 dim / 2 = 256 bytes + 4 bytes norm = 260 bytes per token
        // 6 layers * 65536 tokens * 260 bytes * 2 (K+V) = 204,472,320 bytes (~204.4 MB vs 1.61 GB in F32)
        assert_eq!(kv.memory_bytes(), 6 * 65536 * 260 * 2);

        let k = vec![1.2f32; 512];
        let v = vec![-0.8f32; 512];
        assert!(kv.store(2, 0, &k, &v).is_ok());

        let (norm_k, packed_k) = unsafe { kv.get_k_tq4_packed_unchecked(2, 0) };
        assert!(norm_k > 0.0);
        assert_eq!(packed_k.len(), 256);

        let mut v_dequant = vec![0.0f32; 512];
        unsafe { kv.get_v_tq4_dequantized_unchecked(2, 0, &mut v_dequant) };
        let diff = (v_dequant[0] - v[0]).abs();
        assert!(diff < 0.3, "Dequantized V value diff must be small (got: {diff})");
    }

    #[test]
    fn test_kv_cache_turboquant2_memory_and_store() {
        let attention_layers = vec![2, 5, 8, 11, 13, 15];
        let mut kv = KvCache::try_new_selective_with_precision(
            16,
            65536,
            512,
            &attention_layers,
            KvPrecision::TurboQuant2,
        )
        .unwrap();

        assert_eq!(kv.precision(), KvPrecision::TurboQuant2);
        assert_eq!(kv.capacity_tokens(), 65536);
        // 512 dim / 4 = 128 bytes + 4 bytes norm = 132 bytes per token
        // 6 layers * 65536 tokens * 132 bytes * 2 (K+V) = 103,809,024 bytes (~103.8 MB vs 1.61 GB in F32)
        assert_eq!(kv.memory_bytes(), 6 * 65536 * 132 * 2);

        let k = vec![1.2f32; 512];
        let v = vec![-0.8f32; 512];
        assert!(kv.store(2, 0, &k, &v).is_ok());

        let (norm_k, packed_k) = unsafe { kv.get_k_tq2_packed_unchecked(2, 0) };
        assert!(norm_k > 0.0);
        assert_eq!(packed_k.len(), 128);

        let mut v_dequant = vec![0.0f32; 512];
        unsafe { kv.get_v_tq2_dequantized_unchecked(2, 0, &mut v_dequant) };
        let diff = (v_dequant[0] - v[0]).abs();
        assert!(diff < 0.5, "Dequantized V value diff must be reasonable (got: {diff})");
    }
}
