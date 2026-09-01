//! Model configuration descriptors for LFM2.5 hybrid architectures.

use serde::{Deserialize, Serialize};

pub const DEFAULT_RMS_NORM_EPS: f32 = 1e-5;
pub const DEFAULT_SSM_A_VAL: f32 = 0.95;
pub const DEFAULT_ROPE_BASE: f32 = 1_000_000.0;
pub const DEFAULT_MAX_LORA_RANK: usize = 64;
pub const DEFAULT_N_EXPERTS: usize = 6;
pub const DEFAULT_HEAD_DIM: usize = 64;
pub const RECENT_TOKENS_WINDOW: usize = 64;
pub const DEFAULT_STOP_TOKEN_IM_END: &str = "<|im_end|>";
pub const DEFAULT_STOP_TOKEN_ENDOFTEXT: &str = "<|endoftext|>";

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
    pub rms_norm_eps: f32,
    pub ssm_state_dim: usize,
    pub ssm_conv_kernel: usize,
    pub block_types: Vec<BlockType>,
}

impl ModelConfig {
    /// Validates internal consistency of hyperparameters.
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.dim == 0 {
            return Err("Model dimension (dim) must be > 0".to_string());
        }
        if self.n_heads == 0 {
            return Err("Number of attention heads (n_heads) must be > 0".to_string());
        }
        if self.n_kv_heads == 0 {
            return Err("Number of KV heads (n_kv_heads) must be > 0".to_string());
        }
        if !self.dim.is_multiple_of(self.n_heads) {
            return Err(format!(
                "Model dim ({}) must be divisible by n_heads ({})",
                self.dim, self.n_heads
            ));
        }
        if !self.n_heads.is_multiple_of(self.n_kv_heads) {
            return Err(format!(
                "n_heads ({}) must be divisible by n_kv_heads ({})",
                self.n_heads, self.n_kv_heads
            ));
        }
        if self.n_layers == 0 {
            return Err("Number of layers (n_layers) must be > 0".to_string());
        }
        if self.block_types.len() != self.n_layers {
            return Err(format!(
                "block_types length ({}) does not match n_layers ({})",
                self.block_types.len(),
                self.n_layers
            ));
        }
        Ok(())
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        let dim = 1024;
        let n_heads = 16;
        let n_kv_heads = 8;
        let head_dim = dim / n_heads; // 64
        let kv_dim = n_kv_heads * head_dim; // 512
        let n_layers = 16;

        let mut block_types = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            if i % 3 == 2 {
                block_types.push(BlockType::Attention);
            } else {
                block_types.push(BlockType::SSM);
            }
        }

        Self {
            name: "mivi-v4-lfm2.5-350m".to_string(),
            dim,
            hidden_dim: 4608,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            kv_dim,
            vocab_size: 65536,
            max_seq_len: 128000,
            rope_base: DEFAULT_ROPE_BASE,
            rms_norm_eps: DEFAULT_RMS_NORM_EPS,
            ssm_state_dim: 128,
            ssm_conv_kernel: 3,
            block_types,
        }
    }
}

/// Generation runtime configuration parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub seed: Option<u64>,
    pub stop_tokens: Vec<String>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.2,
            top_p: 0.9,
            top_k: 40,
            min_p: 0.0,
            repetition_penalty: 1.05,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: None,
            stop_tokens: vec![
                DEFAULT_STOP_TOKEN_IM_END.to_string(),
                DEFAULT_STOP_TOKEN_ENDOFTEXT.to_string(),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validate_success() {
        let config = ModelConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validate_rejects_indivisible_heads() {
        let config = ModelConfig {
            n_heads: 16,
            n_kv_heads: 3, // 16 % 3 != 0
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }
}
