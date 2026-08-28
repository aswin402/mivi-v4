//! Model configuration descriptors for LFM2.5 hybrid architectures.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockType {
    SSM,
    Attention,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub kv_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rope_base: f32,
    pub ssm_state_dim: usize,
    pub ssm_conv_kernel: usize,
    pub block_types: Vec<BlockType>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        // Default configuration for LFM2.5-350M
        let dim = 1024;
        let n_heads = 16;
        let n_kv_heads = 4;
        let head_dim = dim / n_heads; // 64
        let kv_dim = n_kv_heads * head_dim; // 256
        let n_layers = 16;

        // 10 Conv/SSM blocks + 6 GQA blocks
        let mut block_types = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            if i % 3 == 2 {
                block_types.push(BlockType::Attention);
            } else {
                block_types.push(BlockType::SSM);
            }
        }
        // Ensure total is 16
        while block_types.len() < n_layers {
            block_types.push(BlockType::Attention);
        }

        Self {
            name: "mivi-v4-lfm2.5-350m".to_string(),
            dim,
            hidden_dim: 2816,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            kv_dim,
            vocab_size: 65536,
            max_seq_len: 32768,
            rope_base: 1_000_000.0,
            ssm_state_dim: 512,
            ssm_conv_kernel: 4,
            block_types,
        }
    }
}
