use crate::config::{
    BlockType, GenerationConfig, ModelConfig, DEFAULT_MAX_LORA_RANK, DEFAULT_N_EXPERTS,
    DEFAULT_RMS_NORM_EPS, RECENT_TOKENS_WINDOW,
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
    #[error("Inference execution error: {0}")]
    ExecutionFailed(String),
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
    /// Load model with default context length ceiling (4096).
    pub fn load(path: &Path) -> Result<Self> {
        Self::load_with_ctx(path, None)
    }

    /// Load model with custom working context length.
    pub fn load_with_ctx(path: &Path, max_ctx: Option<usize>) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        let mut config = extract_model_config(&gguf)?;
        let tokens = extract_vocab(&gguf, config.vocab_size);
        config.vocab_size = tokens.len();

        let attn_count = config
            .block_types
            .iter()
            .filter(|b| **b == BlockType::Attention)
            .count();
        let ssm_count = config
            .block_types
            .iter()
            .filter(|b| **b == BlockType::SSM)
            .count();
        eprintln!(
            "[mivi] Config: dim={}, hidden={}, heads={}, kv_heads={}, kv_dim={}, layers={} ({} attn + {} ssm)",
            config.dim, config.hidden_dim, config.n_heads, config.n_kv_heads, config.kv_dim,
            config.n_layers, attn_count, ssm_count
        );

        // Cap working context length to limit memory usage on low-resource systems.
        const DEFAULT_WORKING_CTX: usize = 4096;
        let ctx_cap = max_ctx.unwrap_or(DEFAULT_WORKING_CTX);
        let working_seq_len = config.max_seq_len.min(ctx_cap);

        let arena_cfg = ArenaConfig {
            dim: config.dim,
            hidden_dim: config.hidden_dim,
            n_layers: config.n_layers,
            n_heads: config.n_heads,
            n_kv_heads: config.n_kv_heads,
            head_dim: config.head_dim,
            kv_dim: config.kv_dim,
            vocab_size: config.vocab_size,
            max_seq_len: working_seq_len,
            ssm_state_dim: config.ssm_state_dim,
            ssm_conv_kernel: config.ssm_conv_kernel,
            max_lora_rank: DEFAULT_MAX_LORA_RANK,
            n_experts: DEFAULT_N_EXPERTS,
        };

        let state = RunState::new(&arena_cfg);
        let attn_layers: Vec<usize> = config
            .block_types
            .iter()
            .enumerate()
            .filter_map(|(idx, &bt)| {
                if bt == BlockType::Attention {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();
        let kv_cache = KvCache::try_new_selective(
            config.n_layers,
            working_seq_len,
            config.kv_dim,
            &attn_layers,
        )?;
        let weights = resolve_model_weights(&gguf, &config)?;

        let vocab = mivi_tokenizer::Vocab::new(tokens);
        let merges = extract_merges(&gguf);
        let mut gen_config = GenerationConfig::default();
        let eos_id = gguf
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_usize().map(|u| u as u32))
            .unwrap_or(EOS_TOKEN_ID);
        if let Some(eos_str) = vocab.get_token(eos_id) {
            if !gen_config.stop_tokens.contains(&eos_str.to_string()) {
                gen_config.stop_tokens.push(eos_str.to_string());
            }
        }
        let tokenizer = Tokenizer::new(vocab, merges);
        let sampler = Sampler::new(gen_config);
        let active_adapters = ActiveAdapters::new();
        let rope_cache =
            mivi_core::RopeCache::new(config.head_dim, working_seq_len, config.rope_base);

        Ok(Self {
            config: ModelConfig {
                max_seq_len: working_seq_len,
                ..config
            },
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

    /// Core forward pass for a single token.
    /// If `compute_logits` is false, skips final RMSNorm and output linear projection,
    /// saving substantial compute during prompt prefill.
    pub fn forward_step(
        &mut self,
        token_id: u32,
        pos: usize,
        compute_logits: bool,
    ) -> Result<Option<&[f32]>> {
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

        if !compute_logits {
            return Ok(None);
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

        Ok(Some(&self.state.logits))
    }

    /// Returns slice of unnormalized logits over vocabulary with ZERO heap allocations.
    pub fn forward(&mut self, token_id: u32, pos: usize) -> Result<&[f32]> {
        self.forward_step(token_id, pos, true)?
            .ok_or_else(|| ModelError::ExecutionFailed("Expected logits from forward step".into()))
    }

    /// Reset internal KV cache and recurrent state buffers.
    pub fn reset_context(&mut self) {
        self.state.reset();
        self.kv_cache.reset();
    }

    /// Get the current position in the KV cache.
    #[inline]
    pub fn current_pos(&self) -> usize {
        self.kv_cache.current_pos()
    }

    /// Prefill prompt tokens and generate tokens incrementally via a streaming callback from position 0.
    pub fn generate_streaming<F>(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        on_token: F,
    ) -> Result<String>
    where
        F: FnMut(u32, &str) -> bool,
    {
        self.reset_context();
        self.generate_streaming_incremental(prompt, 0, max_tokens, on_token)
    }

    /// Prefill prompt tokens starting from `start_pos` without resetting KV cache or recurrent states.
    pub fn generate_streaming_incremental<F>(
        &mut self,
        prompt: &str,
        start_pos: usize,
        max_tokens: usize,
        mut on_token: F,
    ) -> Result<String>
    where
        F: FnMut(u32, &str) -> bool,
    {
        let mut token_ids = self.tokenizer.encode(prompt);
        if token_ids.is_empty() {
            return Ok(String::new());
        }

        // Prepend BOS token only for raw/completion prompts at start_pos 0, NOT for ChatML prompts.
        let add_bos = match self.gguf.metadata.get("tokenizer.ggml.add_bos_token") {
            Some(GgufValue::Bool(b)) => *b,
            _ => false,
        };
        if add_bos && start_pos == 0 {
            let bos_id = match self.gguf.metadata.get("tokenizer.ggml.bos_token_id") {
                Some(v) => v.as_usize().unwrap_or(1) as u32,
                None => 1,
            };
            if token_ids.first() != Some(&bos_id) {
                token_ids.insert(0, bos_id);
            }
        }

        let n_prompt = token_ids.len();
        // Prefill prompt: skip logits computation for all except the last prompt token
        for (i, &tok) in token_ids.iter().enumerate() {
            let cur_pos = start_pos + i;
            let is_last = i + 1 == n_prompt;
            let _ = self.forward_step(tok, cur_pos, is_last)?;
        }
        let mut pos = start_pos + n_prompt;

        let mut generated_ids = Vec::new();
        let mut recent_tokens = VecDeque::with_capacity(RECENT_TOKENS_WINDOW + 1);
        let mut stream_decoder = mivi_tokenizer::Utf8StreamDecoder::new();
        let mut pending_text = String::new();
        let mut raw_bytes = Vec::new();

        let eos_token_id = self
            .gguf
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_usize().map(|u| u as u32))
            .unwrap_or(EOS_TOKEN_ID);

        // Generation loop
        for _ in 0..max_tokens {
            if pos >= self.config.max_seq_len {
                break;
            }

            let recent_slice = recent_tokens.make_contiguous();
            // Preserve raw logits by copying into logits_scratch for sampling
            self.state
                .logits_scratch
                .copy_from_slice(&self.state.logits);
            let next_token = self
                .sampler
                .sample(&mut self.state.logits_scratch, recent_slice);

            if next_token == eos_token_id {
                break;
            }

            generated_ids.push(next_token);
            recent_tokens.push_back(next_token);
            if recent_tokens.len() > RECENT_TOKENS_WINDOW {
                recent_tokens.pop_front();
            }

            // Decode token bytes using the streaming UTF-8 decoder
            raw_bytes.clear();
            self.tokenizer
                .decode_token_bytes(next_token, &mut raw_bytes);
            let decoded_chunk = stream_decoder.feed(&raw_bytes);
            pending_text.push_str(&decoded_chunk);

            // Check for full stop sequence matches
            if let Some(matched_len) =
                matches_any_stop_suffix(&pending_text, &self.sampler.config.stop_tokens)
            {
                let keep_len = pending_text.len().saturating_sub(matched_len);
                pending_text.truncate(keep_len);
                if !pending_text.is_empty() {
                    let _ = on_token(next_token, &pending_text);
                    pending_text.clear();
                }
                break;
            }

            // Hold back any partial prefix of a stop sequence
            let hold_back =
                longest_stop_prefix_len(&pending_text, &self.sampler.config.stop_tokens);
            if hold_back < pending_text.len() {
                let emit_len = pending_text.len() - hold_back;
                let emit_str: String = pending_text.drain(..emit_len).collect();
                if !emit_str.is_empty() && !on_token(next_token, &emit_str) {
                    break;
                }
            }

            let _ = self.forward_step(next_token, pos, true)?;
            pos += 1;
        }

        // Flush remaining decoder bytes
        let flushed = stream_decoder.flush();
        pending_text.push_str(&flushed);
        if !pending_text.is_empty() {
            if let Some(matched_len) =
                matches_any_stop_suffix(&pending_text, &self.sampler.config.stop_tokens)
            {
                let keep_len = pending_text.len().saturating_sub(matched_len);
                pending_text.truncate(keep_len);
            }
            if !pending_text.is_empty() {
                let last_id = generated_ids.last().copied().unwrap_or(0);
                let _ = on_token(last_id, &pending_text);
            }
        }

        let full_decoded = self.tokenizer.decode(&generated_ids);
        let mut clean_result = full_decoded;
        for st in &self.sampler.config.stop_tokens {
            if !st.is_empty() && clean_result.ends_with(st.as_str()) {
                clean_result.truncate(clean_result.len() - st.len());
            }
        }
        Ok(clean_result)
    }

    /// Prefill prompt tokens and generate full string up to `max_tokens`.
    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        self.generate_streaming(prompt, max_tokens, |_, _| true)
    }
}

/// Helper to find the maximum length of a stop token prefix matching the tail of `text`.
fn longest_stop_prefix_len(text: &str, stop_tokens: &[String]) -> usize {
    let mut max_match = 0;
    for st in stop_tokens {
        if st.is_empty() {
            continue;
        }
        let check_len = (st.len() - 1).min(text.len());
        for len in (1..=check_len).rev() {
            if text.ends_with(&st[..len]) {
                max_match = max_match.max(len);
                break;
            }
        }
    }
    max_match
}

/// Helper to check if `text` ends with any full stop sequence, returning the matched length.
fn matches_any_stop_suffix(text: &str, stop_tokens: &[String]) -> Option<usize> {
    for st in stop_tokens {
        if !st.is_empty() && text.ends_with(st.as_str()) {
            return Some(st.len());
        }
    }
    None
}
