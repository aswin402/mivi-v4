//! Complete MiviModel forward pass, dynamic LoRA adapters, and generation engine.

use crate::config::{BlockType, ModelConfig};
use crate::gguf::{GgufFile, GgufValue};
use crate::lora::ActiveAdapters;
use crate::sampler::{Sampler, SamplerConfig};
use crate::ssm::{ssm_forward, SsmWeights};
use crate::transformer::{attention_forward, AttentionWeights};
use mivi_core::arena::{ArenaConfig, RunState};
use mivi_core::math::rms_norm;
use mivi_kv::KvCache;
use mivi_quant::quantized_matvec;
use mivi_tokenizer::Tokenizer;
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("GGUF loading error: {0}")]
    Gguf(#[from] crate::gguf::GgufError),
    #[error("Tokenizer error: {0}")]
    Tokenizer(#[from] mivi_tokenizer::TokenizerError),
    #[error("Missing model weight: {0}")]
    MissingWeight(String),
    #[error("Invalid token ID: {0}")]
    InvalidToken(u32),
    #[error("Context overflow: current pos {pos} >= max_seq_len {max}")]
    ContextOverflow { pos: usize, max: usize },
}

pub type Result<T> = std::result::Result<T, ModelError>;

pub struct Model {
    pub config: ModelConfig,
    pub gguf: GgufFile,
    pub state: RunState,
    pub kv_cache: KvCache,
    pub tokenizer: Tokenizer,
    pub sampler: Sampler,
    pub active_adapters: ActiveAdapters,
    pub rope_cache: mivi_core::RopeCache,
}

#[inline]
fn safe_f32_slice(raw: &[u8]) -> &[f32] {
    let ptr = raw.as_ptr();
    assert_eq!(
        ptr as usize % std::mem::align_of::<f32>(),
        0,
        "GGUF tensor buffer is not 4-byte aligned"
    );
    unsafe { std::slice::from_raw_parts(ptr as *const f32, raw.len() / 4) }
}

impl Model {
    pub fn load(path: &Path) -> Result<Self> {
        let gguf = GgufFile::open(path)?;

        // Extract metadata dynamically from GGUF
        let mut config = ModelConfig::default();
        if let Some(s) = gguf.metadata.get("general.name").and_then(|v| v.as_str()) {
            config.name = s.to_string();
        }
        if let Some(v) = gguf.metadata.get("lfm.context_length").and_then(|v| v.as_usize()) {
            config.max_seq_len = v;
        }
        if let Some(v) = gguf.metadata.get("lfm.embedding_length").and_then(|v| v.as_usize()) {
            config.dim = v;
        }
        if let Some(v) = gguf.metadata.get("lfm.feed_forward_length").and_then(|v| v.as_usize()) {
            config.hidden_dim = v;
        }
        if let Some(v) = gguf.metadata.get("lfm.block_count").and_then(|v| v.as_usize()) {
            config.n_layers = v;
        }
        if let Some(v) = gguf.metadata.get("lfm.attention.head_count").and_then(|v| v.as_usize()) {
            config.n_heads = v;
        }
        if let Some(v) = gguf.metadata.get("lfm.attention.head_count_kv").and_then(|v| v.as_usize()) {
            config.n_kv_heads = v;
        }
        if let Some(v) = gguf.metadata.get("lfm.rope.freq_base").and_then(|v| v.as_f32()) {
            config.rope_base = v;
        }

        config.head_dim = if config.n_heads > 0 {
            config.dim / config.n_heads
        } else {
            64
        };
        config.kv_dim = config.n_kv_heads * config.head_dim;

        // Discover block types per layer
        let mut block_types = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let ssm_key = format!("blk.{}.ssm_in.weight", i);
            if gguf.tensors.contains_key(&ssm_key) {
                block_types.push(BlockType::SSM);
            } else {
                block_types.push(BlockType::Attention);
            }
        }
        config.block_types = block_types;

        // Load vocabulary from GGUF metadata
        let mut tokens = Vec::new();
        if let Some(GgufValue::Array(arr)) = gguf.metadata.get("tokenizer.ggml.tokens") {
            for val in arr {
                if let GgufValue::String(s) = val {
                    tokens.push(s.clone());
                }
            }
        }
        if tokens.is_empty() {
            for i in 0..config.vocab_size {
                tokens.push(format!("<tok_{}>", i));
            }
        } else {
            config.vocab_size = tokens.len();
        }

        let arena_cfg = ArenaConfig {
            dim: config.dim,
            hidden_dim: config.hidden_dim,
            n_layers: config.n_layers,
            n_heads: config.n_heads,
            n_kv_heads: config.n_kv_heads,
            head_dim: config.head_dim,
            kv_dim: config.kv_dim,
            vocab_size: config.vocab_size,
            max_seq_len: config.max_seq_len,
            ssm_state_dim: config.ssm_state_dim,
            max_lora_rank: 64,
            n_experts: 6,
        };

        let state = RunState::new(&arena_cfg);
        let kv_cache = KvCache::new(config.n_layers, config.max_seq_len, config.kv_dim);

        let vocab = mivi_tokenizer::Vocab::new(tokens);
        let tokenizer = Tokenizer::new(vocab, std::collections::HashMap::new());
        let sampler = Sampler::new(SamplerConfig::default());
        let active_adapters = ActiveAdapters::new();
        let rope_cache = mivi_core::RopeCache::new(
            config.head_dim,
            config.max_seq_len,
            config.rope_base,
        );

        Ok(Self {
            config,
            gguf,
            state,
            kv_cache,
            tokenizer,
            sampler,
            active_adapters,
            rope_cache,
        })
    }

    /// Returns slice of unnormalized logits over vocabulary.
    pub fn forward(&mut self, token_id: u32, pos: usize) -> Result<&[f32]> {
        if (token_id as usize) >= self.config.vocab_size {
            return Err(ModelError::InvalidToken(token_id));
        }
        if pos >= self.config.max_seq_len {
            return Err(ModelError::ContextOverflow {
                pos,
                max: self.config.max_seq_len,
            });
        }

        let dim = self.config.dim;

        // 1. Embedding lookup: token_embd.weight
        let (emb_info, emb_data) = self
            .gguf
            .get_tensor_data("token_embd.weight")
            .or_else(|_| self.gguf.get_tensor_data("model.embed_tokens.weight"))
            .map_err(|_| ModelError::MissingWeight("token_embd.weight".into()))?;

        // Dequantize / copy token embedding to state.x
        let row_bytes_len =
            (dim * emb_info.ggml_type.type_size()) / emb_info.ggml_type.block_size();
        let row_offset = (token_id as usize) * row_bytes_len;
        if row_offset + row_bytes_len > emb_data.len() {
            return Err(ModelError::InvalidToken(token_id));
        }
        let row_bytes = &emb_data[row_offset..row_offset + row_bytes_len];
        let _ = mivi_quant::dequantize_slice(emb_info.ggml_type, row_bytes, &mut self.state.x);

        // 2. Iterate through layers
        for layer_idx in 0..self.config.n_layers {
            let block_type = self.config.block_types[layer_idx];

            match block_type {
                BlockType::Attention => {
                    let norm_name = format!("blk.{}.attn_norm.weight", layer_idx);
                    let q_name = format!("blk.{}.attn_q.weight", layer_idx);
                    let k_name = format!("blk.{}.attn_k.weight", layer_idx);
                    let v_name = format!("blk.{}.attn_v.weight", layer_idx);
                    let o_name = format!("blk.{}.attn_output.weight", layer_idx);

                    let ffn_norm_name = format!("blk.{}.ffn_norm.weight", layer_idx);
                    let ffn_gate_name = format!("blk.{}.ffn_gate.weight", layer_idx);
                    let ffn_up_name = format!("blk.{}.ffn_up.weight", layer_idx);
                    let ffn_down_name = format!("blk.{}.ffn_down.weight", layer_idx);

                    if let (
                        Ok((_, attn_norm_raw)),
                        Ok((q_info, q_raw)),
                        Ok((k_info, k_raw)),
                        Ok((v_info, v_raw)),
                        Ok((o_info, o_raw)),
                        Ok((_, ffn_norm_raw)),
                        Ok((gate_info, gate_raw)),
                        Ok((up_info, up_raw)),
                        Ok((down_info, down_raw)),
                    ) = (
                        self.gguf.get_tensor_data(&norm_name),
                        self.gguf.get_tensor_data(&q_name),
                        self.gguf.get_tensor_data(&k_name),
                        self.gguf.get_tensor_data(&v_name),
                        self.gguf.get_tensor_data(&o_name),
                        self.gguf.get_tensor_data(&ffn_norm_name),
                        self.gguf.get_tensor_data(&ffn_gate_name),
                        self.gguf.get_tensor_data(&ffn_up_name),
                        self.gguf.get_tensor_data(&ffn_down_name),
                    ) {
                        let attn_norm = safe_f32_slice(attn_norm_raw);
                        let ffn_norm = safe_f32_slice(ffn_norm_raw);

                        let w = AttentionWeights {
                            attn_norm,
                            q_weight: (q_info.ggml_type, q_raw),
                            k_weight: (k_info.ggml_type, k_raw),
                            v_weight: (v_info.ggml_type, v_raw),
                            o_weight: (o_info.ggml_type, o_raw),
                            ffn_norm,
                            ffn_gate: (gate_info.ggml_type, gate_raw),
                            ffn_up: (up_info.ggml_type, up_raw),
                            ffn_down: (down_info.ggml_type, down_raw),
                        };

                        attention_forward(
                            layer_idx,
                            pos,
                            &mut self.state,
                            &mut self.kv_cache,
                            &w,
                            &self.config,
                            &self.active_adapters,
                            &self.rope_cache,
                        );
                    }
                }
                BlockType::SSM => {
                    let norm_name = format!("blk.{}.ssm_norm.weight", layer_idx);
                    let in_proj_name = format!("blk.{}.ssm_in.weight", layer_idx);
                    let out_proj_name = format!("blk.{}.ssm_out.weight", layer_idx);
                    let ffn_norm_name = format!("blk.{}.ffn_norm.weight", layer_idx);
                    let ffn_gate_name = format!("blk.{}.ffn_gate.weight", layer_idx);
                    let ffn_up_name = format!("blk.{}.ffn_up.weight", layer_idx);
                    let ffn_down_name = format!("blk.{}.ffn_down.weight", layer_idx);

                    let a_name = format!("blk.{}.ssm_a.weight", layer_idx);
                    let conv_name = format!("blk.{}.ssm_conv.weight", layer_idx);

                    if let (
                        Ok((_, ssm_norm_raw)),
                        Ok((in_info, in_raw)),
                        Ok((out_info, out_raw)),
                        Ok((_, ffn_norm_raw)),
                        Ok((gate_info, gate_raw)),
                        Ok((up_info, up_raw)),
                        Ok((down_info, down_raw)),
                    ) = (
                        self.gguf.get_tensor_data(&norm_name),
                        self.gguf.get_tensor_data(&in_proj_name),
                        self.gguf.get_tensor_data(&out_proj_name),
                        self.gguf.get_tensor_data(&ffn_norm_name),
                        self.gguf.get_tensor_data(&ffn_gate_name),
                        self.gguf.get_tensor_data(&ffn_up_name),
                        self.gguf.get_tensor_data(&ffn_down_name),
                    ) {
                        let ssm_norm = safe_f32_slice(ssm_norm_raw);
                        let ffn_norm = safe_f32_slice(ffn_norm_raw);

                        let default_ssm_a = [0.95f32; 512];
                        let default_conv_weight = [0.25f32; 4];

                        let ssm_a: &[f32] = if let Ok((_, raw)) = self.gguf.get_tensor_data(&a_name) {
                            safe_f32_slice(raw)
                        } else {
                            &default_ssm_a
                        };

                        let conv_weight: &[f32] = if let Ok((_, raw)) = self.gguf.get_tensor_data(&conv_name) {
                            safe_f32_slice(raw)
                        } else {
                            &default_conv_weight
                        };

                        let w = SsmWeights {
                            ssm_norm,
                            in_proj: (in_info.ggml_type, in_raw),
                            conv_weight,
                            ssm_a,
                            ssm_b: (in_info.ggml_type, in_raw),
                            ssm_c: (out_info.ggml_type, out_raw),
                            out_proj: (out_info.ggml_type, out_raw),
                            ffn_norm,
                            ffn_gate: (gate_info.ggml_type, gate_raw),
                            ffn_up: (up_info.ggml_type, up_raw),
                            ffn_down: (down_info.ggml_type, down_raw),
                        };

                        ssm_forward(layer_idx, &mut self.state, &w, &self.config, &self.active_adapters);
                    }
                }
            }
        }

        // 3. Final RMSNorm
        if let Ok((_, norm_raw)) = self.gguf.get_tensor_data("output_norm.weight") {
            let final_norm = safe_f32_slice(norm_raw);
            rms_norm(&mut self.state.xb, &self.state.x, final_norm, 1e-5);
        } else {
            self.state.xb.copy_from_slice(&self.state.x);
        }

        // 4. Output projection to vocabulary logits (output.weight / lm_head.weight)
        if let Ok((head_info, head_raw)) = self.gguf.get_tensor_data("output.weight") {
            let _ = quantized_matvec(
                &mut self.state.logits,
                head_info.ggml_type,
                head_raw,
                &self.state.xb,
                self.config.vocab_size,
                dim,
            );
            self.active_adapters.apply_module(
                "output",
                &self.state.xb,
                &mut self.state.lora_down,
                &mut self.state.logits,
            );
        }

        Ok(&self.state.logits)
    }

    /// Prefill prompt tokens and generate up to `max_tokens`.
    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        self.state.reset();
        self.kv_cache.reset();

        let token_ids = self.tokenizer.encode(prompt);
        let mut generated_ids = Vec::new();
        let mut recent_tokens = Vec::new();

        let eos_token_id = match self.gguf.metadata.get("tokenizer.ggml.eos_token_id") {
            Some(GgufValue::U32(id)) => *id,
            _ => 2,
        };

        let mut pos = 0;
        // Prefill prompt
        for &tok in &token_ids {
            let _ = self.forward(tok, pos)?;
            pos += 1;
        }

        let mut current_token = if token_ids.is_empty() {
            1
        } else {
            *token_ids.last().unwrap()
        };

        // Generation loop
        for _ in 0..max_tokens {
            let logits = self.forward(current_token, pos)?;
            let mut logits_clone = logits.to_vec();
            let next_token = self.sampler.sample(&mut logits_clone, &recent_tokens);

            if next_token == eos_token_id {
                break;
            }

            generated_ids.push(next_token);
            recent_tokens.push(next_token);
            if recent_tokens.len() > 64 {
                recent_tokens.remove(0);
            }

            // Check for EOS token strings
            if let Some(tok_str) = self.tokenizer.decode_token(next_token) {
                if tok_str == "<|im_end|>" || tok_str == "<|endoftext|>" {
                    break;
                }
            }

            current_token = next_token;
            pos += 1;
        }

        Ok(self.tokenizer.decode(&generated_ids))
    }
}
