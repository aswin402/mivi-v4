use crate::config::{
    GenerationConfig, ModelConfig, DEFAULT_MAX_LORA_RANK, DEFAULT_N_EXPERTS, DEFAULT_RMS_NORM_EPS,
    RECENT_TOKENS_WINDOW,
};
use crate::gguf::{GgufFile, GgufValue};
use crate::loader::{extract_merges, extract_model_config, extract_vocab, resolve_model_weights};
use crate::lora::ActiveAdapters;
use crate::sampler::Sampler;
use crate::ssm::ssm_forward;
use crate::transformer::attention_forward;
use crate::weights::{LayerWeights, ModelWeights};
use mivi_core::arena::{ArenaConfig, RunState};
use mivi_kv::KvCache;
use mivi_tokenizer::{Tokenizer, EOS_TOKEN_ID};
use std::collections::VecDeque;
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("GGUF loading error: {0}")]
    Gguf(#[from] crate::gguf::GgufError),
    #[error("Tokenizer error: {0}")]
    Tokenizer(#[from] mivi_tokenizer::TokenizerError),
    #[error("Quantization error: {0}")]
    Quant(#[from] mivi_quant::QuantError),
    #[error("KV cache error: {0}")]
    KvCache(#[from] mivi_kv::KvError),
    #[error("Missing model weight: {0}")]
    MissingWeight(String),
    #[error("Invalid model configuration: {0}")]
    InvalidConfig(String),
    #[error("Malformed tensor alignment: {0}")]
    MalformedTensorAlignment(String),
    #[error("Invalid token ID: {0}")]
    InvalidToken(u32),
    #[error("Context overflow: current pos {pos} >= max_seq_len {max}")]
    ContextOverflow { pos: usize, max: usize },
    #[error("Dimension mismatch: {0}")]
    DimMismatch(String),
}

pub type Result<T> = std::result::Result<T, ModelError>;

pub struct Model {
    pub config: ModelConfig,
    pub gguf: GgufFile,
    pub weights: ModelWeights,
    pub state: RunState,
    pub kv_cache: KvCache,
    pub tokenizer: Tokenizer,
    pub sampler: Sampler,
    pub active_adapters: ActiveAdapters,
    pub rope_cache: mivi_core::RopeCache,
}

impl Model {
    pub fn load(path: &Path) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        let mut config = extract_model_config(&gguf)?;
        let tokens = extract_vocab(&gguf, config.vocab_size);
        config.vocab_size = tokens.len();

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
            ssm_conv_kernel: config.ssm_conv_kernel,
            max_lora_rank: DEFAULT_MAX_LORA_RANK,
            n_experts: DEFAULT_N_EXPERTS,
        };

        let state = RunState::new(&arena_cfg);
        let kv_cache = KvCache::new(config.n_layers, config.max_seq_len, config.kv_dim);
        let weights = resolve_model_weights(&gguf, &config)?;

        let vocab = mivi_tokenizer::Vocab::new(tokens);
        let merges = extract_merges(&gguf);
        let mut gen_config = GenerationConfig::default();
        if let Some(GgufValue::U32(eos_id)) = gguf.metadata.get("tokenizer.ggml.eos_token_id") {
            if let Some(eos_str) = vocab.get_token(*eos_id) {
                if !gen_config.stop_tokens.contains(&eos_str.to_string()) {
                    gen_config.stop_tokens.push(eos_str.to_string());
                }
            }
        }
        let tokenizer = Tokenizer::new(vocab, merges);
        let sampler = Sampler::new(gen_config);
        let active_adapters = ActiveAdapters::new();
        let rope_cache =
            mivi_core::RopeCache::new(config.head_dim, config.max_seq_len, config.rope_base);

        Ok(Self {
            config,
            gguf,
            weights,
            state,
            kv_cache,
            tokenizer,
            sampler,
            active_adapters,
            rope_cache,
        })
    }

    /// Returns slice of unnormalized logits over vocabulary with ZERO heap allocations.
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

        // 1. Embedding lookup: token_embd
        let emb = &self.weights.token_embd;
        let type_size = emb.quant_type.type_size().unwrap_or(mivi_quant::F32_BYTES);
        let block_size = emb.quant_type.block_size().unwrap_or(1);
        let row_bytes_len = (dim * type_size) / block_size;
        let row_offset = emb.offset + (token_id as usize) * row_bytes_len;
        if row_offset + row_bytes_len > self.gguf.mmap.len() {
            return Err(ModelError::InvalidToken(token_id));
        }
        let row_bytes = &self.gguf.mmap[row_offset..row_offset + row_bytes_len];
        mivi_quant::dequantize_slice(emb.quant_type, row_bytes, &mut self.state.x)?;

        // 2. Iterate through layers
        for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
            match layer {
                LayerWeights::Attention(w) => {
                    let params = crate::transformer::AttentionParams {
                        layer: layer_idx,
                        pos,
                        weights: w,
                        mmap: &self.gguf.mmap,
                        config: &self.config,
                        adapters: &self.active_adapters,
                        rope: &self.rope_cache,
                    };
                    attention_forward(&mut self.state, &mut self.kv_cache, &params)?;
                }
                LayerWeights::Ssm(w) => {
                    let params = crate::ssm::SsmParams {
                        layer: layer_idx,
                        weights: w,
                        mmap: &self.gguf.mmap,
                        config: &self.config,
                        adapters: &self.active_adapters,
                    };
                    ssm_forward(&mut self.state, &params)?;
                }
            }
        }

        // 3. Final RMSNorm (SIMD accelerated)
        if let Some(ref final_norm) = self.weights.output_norm {
            if final_norm.len() != dim {
                return Err(ModelError::DimMismatch(format!(
                    "output_norm length mismatch: expected {}, got {}",
                    dim,
                    final_norm.len()
                )));
            }
            mivi_core::simd::rms_norm_simd(
                &mut self.state.xb,
                &self.state.x,
                final_norm,
                DEFAULT_RMS_NORM_EPS,
            );
        } else {
            self.state.xb.copy_from_slice(&self.state.x);
        }

        // 4. Output projection to vocabulary logits (falls back to tied token_embd)
        let head = self
            .weights
            .output_proj
            .as_ref()
            .unwrap_or(&self.weights.token_embd);
        let out_params = crate::ffn::LinearParams {
            weight: head,
            input: &self.state.xb,
            rows: self.config.vocab_size,
            cols: dim,
            mmap: &self.gguf.mmap,
            adapters: &self.active_adapters,
            module_name: "output",
        };
        crate::ffn::linear_forward(
            &mut self.state.logits,
            &out_params,
            &mut self.state.lora_down,
        )?;

        #[cfg(debug_assertions)]
        {
            if let Some(nan_idx) = self.state.logits.iter().position(|v| v.is_nan()) {
                tracing::warn!(
                    "NaN detected in logits at index {} during forward pass",
                    nan_idx
                );
            }
        }

        Ok(&self.state.logits)
    }

    /// Prefill prompt tokens and generate tokens incrementally via a streaming callback.
    pub fn generate_streaming<F>(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        mut on_token: F,
    ) -> Result<String>
    where
        F: FnMut(u32, &str) -> bool,
    {
        self.state.reset();
        self.kv_cache.reset();

        let token_ids = self.tokenizer.encode(prompt);
        if token_ids.is_empty() {
            return Ok(String::new());
        }

        let mut generated_ids = Vec::new();
        let mut recent_tokens = VecDeque::with_capacity(RECENT_TOKENS_WINDOW + 1);

        let eos_token_id = match self.gguf.metadata.get("tokenizer.ggml.eos_token_id") {
            Some(GgufValue::U32(id)) => *id,
            _ => EOS_TOKEN_ID,
        };

        // Prefill prompt
        for (i, &tok) in token_ids.iter().enumerate() {
            let _ = self.forward(tok, i)?;
        }
        let mut pos = token_ids.len();

        // Generation loop
        for _ in 0..max_tokens {
            if pos >= self.config.max_seq_len {
                // Gracefully stop at context window limit
                break;
            }

            self.state
                .logits_scratch
                .copy_from_slice(&self.state.logits);
            let recent_slice = recent_tokens.make_contiguous();
            let next_token = self
                .sampler
                .sample(&mut self.state.logits_scratch, recent_slice);

            if next_token == eos_token_id {
                break;
            }

            // Decode token string
            let tok_str = self.tokenizer.decode_token(next_token).unwrap_or_default();
            if self
                .sampler
                .config
                .stop_tokens
                .iter()
                .any(|st| st == tok_str)
            {
                break;
            }

            generated_ids.push(next_token);
            recent_tokens.push_back(next_token);
            if recent_tokens.len() > RECENT_TOKENS_WINDOW {
                recent_tokens.pop_front();
            }

            if !on_token(next_token, tok_str) {
                break;
            }

            let _ = self.forward(next_token, pos)?;
            pos += 1;
        }

        Ok(self.tokenizer.decode(&generated_ids))
    }

    /// Prefill prompt tokens and generate full string up to `max_tokens`.
    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        self.generate_streaming(prompt, max_tokens, |_, _| true)
    }
}
