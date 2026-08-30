//! Pre-resolved ModelWeights structure mapping GGUF tensor memory layout at load time.
//! Eliminates all runtime string allocations and hash table lookups during token decoding.

use mivi_quant::GgmlType;

#[derive(Debug, Clone, Copy)]
pub struct QuantizedTensor {
    pub quant_type: GgmlType,
    pub offset: usize,
    pub len: usize,
    pub rows: usize,
    pub cols: usize,
}

impl QuantizedTensor {
    /// Hot-path slice accessor.
    ///
    /// # Panics
    /// Panics if offset + len exceeds mmap bounds. Use `as_slice_checked` if input bounds are untrusted.
    #[inline(always)]
    #[track_caller]
    pub fn as_slice<'a>(&self, mmap: &'a [u8]) -> &'a [u8] {
        let end = self.offset.saturating_add(self.len);
        assert!(
            end <= mmap.len(),
            "QuantizedTensor slice out of bounds: offset={}, len={}, mmap_len={}",
            self.offset,
            self.len,
            mmap.len()
        );
        &mmap[self.offset..end]
    }

    #[inline(always)]
    pub fn as_slice_checked<'a>(&self, mmap: &'a [u8]) -> Option<&'a [u8]> {
        let end = self.offset.checked_add(self.len)?;
        if end <= mmap.len() {
            Some(&mmap[self.offset..end])
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct FfnLayerWeights {
    pub ffn_norm: Box<[f32]>,
    pub w_gate: QuantizedTensor,
    pub w_up: QuantizedTensor,
    pub w_down: QuantizedTensor,
    pub ffn_gate_name: String,
    pub ffn_up_name: String,
    pub ffn_down_name: String,
}

#[derive(Debug, Clone)]
pub struct AttentionLayerWeights {
    pub attn_norm: Box<[f32]>,
    pub wq: QuantizedTensor,
    pub wk: QuantizedTensor,
    pub wv: QuantizedTensor,
    pub wo: QuantizedTensor,
    pub ffn: FfnLayerWeights,

    // Pre-formatted module paths for zero-alloc LoRA dynamic dispatch
    pub q_name: String,
    pub k_name: String,
    pub v_name: String,
    pub o_name: String,
}

#[derive(Debug, Clone)]
pub struct SsmLayerWeights {
    pub ssm_norm: Box<[f32]>,
    pub in_proj: QuantizedTensor,
    pub ssm_a: Box<[f32]>,
    pub ssm_conv: Box<[f32]>,
    pub out_proj: QuantizedTensor,
    pub ffn: FfnLayerWeights,

    // Pre-formatted module paths for zero-alloc LoRA dynamic dispatch
    pub in_name: String,
    pub out_name: String,
}

#[derive(Debug, Clone)]
pub enum LayerWeights {
    Attention(AttentionLayerWeights),
    Ssm(SsmLayerWeights),
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    pub token_embd: QuantizedTensor,
    pub layers: Vec<LayerWeights>,
    pub output_norm: Option<Box<[f32]>>,
    pub output_proj: Option<QuantizedTensor>,
}
