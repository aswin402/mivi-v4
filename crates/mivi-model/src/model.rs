use crate::config::{
    BlockType, GenerationConfig, ModelConfig, PrefillStrategy, DEFAULT_MAX_LORA_RANK,
    DEFAULT_N_EXPERTS, RECENT_TOKENS_WINDOW,
};
#[cfg(feature = "fixture-diagnostics")]
use crate::fixture_diagnostics::ModelRecorder;
use crate::gguf::{GgufFile, GgufValue};
use crate::loader::{extract_merges, extract_model_config, extract_vocab, resolve_model_weights};
use crate::lora::ActiveAdapters;
use crate::prefill::TileActivations;
use crate::sampler::Sampler;
use crate::ssm::{
    ssm_forward_profiled, ssm_forward_tile, ssm_forward_tile_profiled, SsmStageProfile,
};
use crate::transformer::{
    attention_forward_profiled, attention_forward_tile_profiled, AttentionStageProfile,
};
use crate::weights::{LayerWeights, ModelWeights};

#[cfg(all(test, feature = "q4-cached-sums-experiment"))]
#[path = "model/cached_sums_tests.rs"]
mod cached_sums_tests;

#[cfg(all(
    test,
    feature = "q4-cached-sums-experiment",
    feature = "fixture-diagnostics"
))]
#[path = "model/cached_sums_agent_tests.rs"]
mod cached_sums_agent_tests;
use mivi_core::arena::{ArenaConfig, RunState};
use mivi_kv::{compute_chunk_hash, KvCache};
use mivi_tokenizer::{Tokenizer, EOS_TOKEN_ID};
use std::collections::VecDeque;
use std::path::Path;
use std::time::{Duration, Instant};
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

/// Aggregate timings for the forward stages of a model run.
///
/// This is opt-in diagnostic data. It is disabled by default so normal inference does not
/// create timing timestamps on every layer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ForwardProfileSnapshot {
    pub tokens: usize,
    pub embedding: Duration,
    pub attention: Duration,
    pub attention_stages: AttentionStageProfile,
    pub ssm: Duration,
    pub ssm_stages: SsmStageProfile,
    pub logits: Duration,
}

impl ForwardProfileSnapshot {
    /// Return the time spent in the measured forward stages.
    #[inline]
    pub fn total_stage_time(self) -> Duration {
        self.embedding + self.attention + self.ssm + self.logits
    }
}

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
    pub prefix_cache: mivi_kv::PrefixCache,
    prefill_strategy: PrefillStrategy,
    forward_profile: Option<ForwardProfileSnapshot>,
    last_prefill_profile: Option<ForwardProfileSnapshot>,
    #[cfg(feature = "fixture-diagnostics")]
    fixture_recorder: Option<ModelRecorder>,
}

impl Model {
    /// Load model with default context length ceiling (4096).
    pub fn load(path: &Path) -> Result<Self> {
        Self::load_with_options(path, None, None)
    }

    /// Load model with custom working context length.
    pub fn load_with_ctx(path: &Path, max_ctx: Option<usize>) -> Result<Self> {
        Self::load_with_options(path, max_ctx, None)
    }

    /// Load model with custom working context length and KV cache precision.
    pub fn load_with_options(
        path: &Path,
        max_ctx: Option<usize>,
        kv_precision: Option<mivi_kv::KvPrecision>,
    ) -> Result<Self> {
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
        let precision = kv_precision.unwrap_or(mivi_kv::KvPrecision::F32);
        eprintln!(
            "[mivi] Config: dim={}, hidden={}, heads={}, kv_heads={}, kv_dim={}, layers={} ({} attn + {} ssm), kv_precision={:?}",
            config.dim, config.hidden_dim, config.n_heads, config.n_kv_heads, config.kv_dim,
            config.n_layers, attn_count, ssm_count, precision
        );

        // Working context length configuration (defaults to 16k, capped at 64k max).
        pub const DEFAULT_WORKING_CTX: usize = 16384;
        pub const MAX_SUPPORTED_CTX: usize = 65536;
        let ctx_cap = max_ctx
            .unwrap_or(DEFAULT_WORKING_CTX)
            .min(MAX_SUPPORTED_CTX);
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
        let kv_cache = KvCache::try_new_selective_with_precision(
            config.n_layers,
            working_seq_len,
            config.kv_dim,
            &attn_layers,
            precision,
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
        let rope_scaling = if working_seq_len > 4096 {
            let scale = working_seq_len as f32 / 4096.0;
            mivi_core::rope::RopeScaling::YaRN {
                scale,
                orig_max_seq_len: 4096,
                extrapolation_factor: 1.0,
                attn_factor: 1.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
            }
        } else {
            mivi_core::rope::RopeScaling::None
        };
        let rope_cache = mivi_core::RopeCache::new_with_scaling(
            config.head_dim,
            working_seq_len,
            config.rope_base,
            rope_scaling,
        );

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
            prefix_cache: mivi_kv::PrefixCache::default(),
            prefill_strategy: PrefillStrategy::default(),
            forward_profile: None,
            last_prefill_profile: None,
            #[cfg(feature = "fixture-diagnostics")]
            fixture_recorder: None,
        })
    }

    /// Arm a bounded observation of the next streaming generation.
    ///
    /// # Errors
    /// Returns an error if capture is already armed or its limits are invalid.
    #[cfg(feature = "fixture-diagnostics")]
    pub fn start_fixture_capture(
        &mut self,
        limits: crate::fixture_diagnostics::CaptureLimits,
    ) -> std::result::Result<(), &'static str> {
        if self.fixture_recorder.is_some() {
            return Err("fixture capture already active");
        }
        self.fixture_recorder = Some(ModelRecorder::new(limits)?);
        Ok(())
    }

    /// Take the armed recorder's snapshot and disarm observation.
    #[cfg(feature = "fixture-diagnostics")]
    pub fn take_fixture_capture(&mut self) -> Option<crate::fixture_diagnostics::ModelCapture> {
        self.fixture_recorder
            .take()
            .map(|recorder| recorder.snapshot)
    }

    /// Begin timing an armed direct-token replay capture.
    #[cfg(feature = "fixture-diagnostics")]
    pub fn begin_fixture_capture_observation(
        &mut self,
        entry: Instant,
    ) -> std::result::Result<(), &'static str> {
        let recorder = self
            .fixture_recorder
            .as_mut()
            .ok_or("fixture capture is not active")?;
        recorder.begin(entry);
        Ok(())
    }

    /// Finish an armed direct-token replay capture at physical model return.
    #[cfg(feature = "fixture-diagnostics")]
    pub fn finish_fixture_capture_observation(
        &mut self,
        outcome: crate::fixture_diagnostics::ModelOutcome,
        finished: Instant,
    ) {
        if let Some(recorder) = self.fixture_recorder.as_mut() {
            recorder.finish(finished, outcome);
        }
    }

    /// Set the prompt-prefill execution strategy.
    pub fn set_prefill_strategy(&mut self, strategy: PrefillStrategy) -> Result<()> {
        strategy.validate().map_err(ModelError::InvalidConfig)?;
        self.prefill_strategy = strategy;
        Ok(())
    }

    /// Return the configured prompt-prefill execution strategy.
    #[inline]
    pub fn prefill_strategy(&self) -> PrefillStrategy {
        self.prefill_strategy
    }

    /// Enable aggregate forward-stage timing for diagnostic benchmarks.
    pub fn enable_forward_profile(&mut self) {
        self.forward_profile = Some(ForwardProfileSnapshot::default());
    }

    /// Disable aggregate forward-stage timing and release its diagnostic state.
    pub fn disable_forward_profile(&mut self) {
        self.forward_profile = None;
    }

    /// Clear the current aggregate forward-stage timings while keeping profiling enabled.
    pub fn reset_forward_profile(&mut self) {
        if let Some(profile) = self.forward_profile.as_mut() {
            *profile = ForwardProfileSnapshot::default();
        }
    }

    /// Return the current aggregate forward-stage timings, if profiling is enabled.
    #[inline]
    pub fn forward_profile(&self) -> Option<ForwardProfileSnapshot> {
        self.forward_profile
    }

    /// Return the cumulative forward-stage profile captured immediately after the most recent prefill.
    ///
    /// This is available only when forward profiling was enabled. Reset the profile before
    /// generation when the snapshot must be isolated to one call.
    #[inline]
    pub fn last_prefill_profile(&self) -> Option<ForwardProfileSnapshot> {
        self.last_prefill_profile
    }

    #[inline]
    fn record_forward_profile(
        &mut self,
        embedding: Duration,
        attention: Duration,
        ssm: Duration,
        logits: Duration,
    ) {
        self.record_forward_profile_batch(
            1,
            embedding,
            attention,
            ssm,
            SsmStageProfile::default(),
            logits,
        );
    }

    #[inline]
    fn record_forward_profile_batch(
        &mut self,
        tokens: usize,
        embedding: Duration,
        attention: Duration,
        ssm: Duration,
        ssm_stages: SsmStageProfile,
        logits: Duration,
    ) {
        if let Some(profile) = self.forward_profile.as_mut() {
            profile.tokens += tokens;
            profile.embedding += embedding;
            profile.attention += attention;
            profile.ssm += ssm;
            profile.ssm_stages.add_assign(ssm_stages);
            profile.logits += logits;
        }
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
        let profile_enabled = self.forward_profile.is_some();

        // 1. Embedding lookup: token_embd
        let embedding_start = profile_enabled.then(Instant::now);
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
        let embedding_elapsed = embedding_start
            .map(|start| start.elapsed())
            .unwrap_or_default();

        // 2. Iterate through layers
        let mut attention_elapsed = Duration::ZERO;
        let mut ssm_elapsed = Duration::ZERO;
        let mut attention_stages = AttentionStageProfile::default();
        let mut ssm_stages = SsmStageProfile::default();
        for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
            match layer {
                LayerWeights::Attention(w) => {
                    let stage_start = profile_enabled.then(Instant::now);
                    let params = crate::transformer::AttentionParams {
                        layer: layer_idx,
                        pos,
                        weights: w,
                        mmap: &self.gguf.mmap,
                        config: &self.config,
                        adapters: &self.active_adapters,
                        rope: &self.rope_cache,
                    };
                    attention_forward_profiled(
                        &mut self.state,
                        &mut self.kv_cache,
                        &params,
                        profile_enabled.then_some(&mut attention_stages),
                    )?;
                    if let Some(start) = stage_start {
                        attention_elapsed += start.elapsed();
                    }
                }
                LayerWeights::Ssm(w) => {
                    let stage_start = profile_enabled.then(Instant::now);
                    let params = crate::ssm::SsmParams {
                        layer: layer_idx,
                        weights: w,
                        mmap: &self.gguf.mmap,
                        config: &self.config,
                        adapters: &self.active_adapters,
                    };
                    ssm_forward_profiled(
                        &mut self.state,
                        &params,
                        profile_enabled.then_some(&mut ssm_stages),
                    )?;
                    if let Some(start) = stage_start {
                        ssm_elapsed += start.elapsed();
                    }
                }
            }
        }

        if let Some(profile) = self.forward_profile.as_mut() {
            profile.attention_stages.add_assign(attention_stages);
            profile.ssm_stages.add_assign(ssm_stages);
        }

        if !compute_logits {
            self.record_forward_profile(
                embedding_elapsed,
                attention_elapsed,
                ssm_elapsed,
                Duration::ZERO,
            );
            return Ok(None);
        }

        // 3. Final RMSNorm and output projection (SIMD accelerated)
        let logits_start = profile_enabled.then(Instant::now);
        self.compute_logits_from_state()?;

        self.record_forward_profile(
            embedding_elapsed,
            attention_elapsed,
            ssm_elapsed,
            logits_start
                .map(|start| start.elapsed())
                .unwrap_or_default(),
        );

        Ok(Some(&self.state.logits))
    }

    fn compute_logits_from_state(&mut self) -> Result<()> {
        let dim = self.config.dim;
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
                self.config.rms_norm_eps,
            );
        } else {
            self.state.xb.copy_from_slice(&self.state.x);
        }

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
        Ok(())
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

    fn run_prefill<C>(
        &mut self,
        prompt_tokens: &[u32],
        start_pos: usize,
        start_prefill_idx: usize,
        chained_hash: &mut u64,
        should_cancel: &mut C,
    ) -> Result<bool>
    where
        C: FnMut() -> bool,
    {
        let strategy = self.prefill_strategy;
        match strategy {
            PrefillStrategy::Token => self.run_token_prefill(
                prompt_tokens,
                start_pos,
                start_prefill_idx,
                chained_hash,
                should_cancel,
            ),
            PrefillStrategy::Chunked { tile_tokens: _ }
                if !self.active_adapters.active.is_empty() =>
            {
                self.run_token_prefill(
                    prompt_tokens,
                    start_pos,
                    start_prefill_idx,
                    chained_hash,
                    should_cancel,
                )
            }
            PrefillStrategy::Chunked { tile_tokens } => self
                .run_chunked_prefill(
                    prompt_tokens,
                    start_pos,
                    start_prefill_idx,
                    *chained_hash,
                    tile_tokens,
                    should_cancel,
                )
                .map(|(completed, next_hash)| {
                    *chained_hash = next_hash;
                    completed
                }),
        }
    }

    fn run_token_prefill<C>(
        &mut self,
        prompt_tokens: &[u32],
        start_pos: usize,
        start_prefill_idx: usize,
        chained_hash: &mut u64,
        should_cancel: &mut C,
    ) -> Result<bool>
    where
        C: FnMut() -> bool,
    {
        let n_prompt = prompt_tokens.len();
        for i in start_prefill_idx..n_prompt {
            if should_cancel() {
                return Ok(false);
            }
            let tok = prompt_tokens[i];
            let cur_pos = start_pos + i;
            let is_last = i + 1 == n_prompt;
            let _ = self.forward_step(tok, cur_pos, is_last)?;
            #[cfg(feature = "fixture-diagnostics")]
            if let Some(recorder) = self.fixture_recorder.as_mut() {
                recorder.processed(1);
            }

            if n_prompt >= 500 && (i + 1) % 500 == 0 {
                use std::io::Write;
                let pct = ((i + 1) * 100) / n_prompt;
                println!(
                    "    \x1b[2m│  ⏳ prefill progress: [{}/{}] ({}%)\x1b[0m",
                    i + 1,
                    n_prompt,
                    pct
                );
                let _ = std::io::stdout().flush();
            }

            if start_pos == 0 && (cur_pos + 1).is_multiple_of(self.prefix_cache.chunk_size()) {
                *chained_hash =
                    self.cache_prefill_boundary(prompt_tokens, cur_pos + 1, *chained_hash);
            }
        }
        Ok(true)
    }

    fn cache_prefill_boundary(&mut self, tokens: &[u32], end: usize, prev_hash: u64) -> u64 {
        let size = self.prefix_cache.chunk_size();
        let start = end - size;
        let (chunk_tokens, standalone_hash) = cache_chunk_tokens(&tokens[start..end]);
        let hash = compute_chunk_hash(prev_hash, &chunk_tokens);
        if let Ok((k, v)) = self.kv_cache.export_state_range(start, size) {
            let (conv, hidden) = self.state.export_ssm_states();
            let state = mivi_kv::HybridStateSnapshot::new(end, standalone_hash, k, v, conv, hidden);
            // A missing ancestor or budget eviction simply leaves this prefix
            // uncached. Continue hashing the real preceding token sequence.
            let _ =
                self.prefix_cache
                    .insert_delta_chunk(prev_hash, &chunk_tokens, start / size, state);
        }
        hash
    }

    fn embed_prefill_tile(
        &self,
        prompt_tokens: &[u32],
        start: usize,
        end: usize,
        tile: &mut TileActivations,
    ) -> Result<()> {
        let dim = self.config.dim;
        let embedding = &self.weights.token_embd;
        let type_size = embedding
            .quant_type
            .type_size()
            .unwrap_or(mivi_quant::F32_BYTES);
        let block_size = embedding.quant_type.block_size().unwrap_or(1);
        let row_bytes =
            (dim / block_size)
                .checked_mul(type_size)
                .ok_or(ModelError::ExecutionFailed(
                    "embedding row size overflow".to_string(),
                ))?;
        let rows = end.saturating_sub(start);
        let output_len = rows.checked_mul(dim).ok_or(ModelError::ExecutionFailed(
            "embedding tile size overflow".to_string(),
        ))?;
        if rows == 0 || rows > tile.tile_tokens() {
            return Err(ModelError::DimMismatch(format!(
                "invalid embedding tile row count: {rows}"
            )));
        }

        for (row, &token_id) in prompt_tokens[start..end].iter().enumerate() {
            if (token_id as usize) >= self.config.vocab_size {
                return Err(ModelError::InvalidToken(token_id));
            }
            let row_offset = embedding
                .offset
                .checked_add((token_id as usize).checked_mul(row_bytes).ok_or(
                    ModelError::ExecutionFailed("embedding offset overflow".to_string()),
                )?)
                .ok_or(ModelError::ExecutionFailed(
                    "embedding offset overflow".to_string(),
                ))?;
            let row_end = row_offset
                .checked_add(row_bytes)
                .ok_or(ModelError::ExecutionFailed(
                    "embedding row end overflow".to_string(),
                ))?;
            if row_end > self.gguf.mmap.len() {
                return Err(ModelError::InvalidToken(token_id));
            }
            let row_bytes = &self.gguf.mmap[row_offset..row_end];
            let output_start = row * dim;
            mivi_quant::dequantize_slice(
                embedding.quant_type,
                row_bytes,
                &mut tile.current[output_start..output_start + dim],
            )?;
        }
        debug_assert_eq!(output_len, rows * dim);
        Ok(())
    }

    fn run_chunked_prefill<C>(
        &mut self,
        prompt_tokens: &[u32],
        start_pos: usize,
        start_prefill_idx: usize,
        mut chained_hash: u64,
        tile_tokens: usize,
        should_cancel: &mut C,
    ) -> Result<(bool, u64)>
    where
        C: FnMut() -> bool,
    {
        if tile_tokens == 0 {
            return Err(ModelError::InvalidConfig(
                "prefill tile size must be greater than zero".to_string(),
            ));
        }
        let n_prompt = prompt_tokens.len();
        if start_prefill_idx >= n_prompt {
            return Ok((true, chained_hash));
        }

        let tile_capacity = tile_tokens.min(n_prompt - start_prefill_idx);
        let mut tile = TileActivations::with_kv_dim(
            tile_capacity,
            self.config.dim,
            self.config.hidden_dim,
            self.config.kv_dim,
        )
        .map_err(|error| ModelError::ExecutionFailed(error.to_string()))?;
        let mut cursor = start_prefill_idx;
        let mut last_progress_report = start_prefill_idx;

        while cursor < n_prompt {
            if should_cancel() {
                return Ok((false, chained_hash));
            }

            let mut end = cursor.saturating_add(tile_capacity).min(n_prompt);
            if start_pos == 0 {
                let absolute_start = start_pos + cursor;
                let chunk_size = self.prefix_cache.chunk_size();
                let next_boundary = ((absolute_start / chunk_size) + 1) * chunk_size;
                let boundary_index = next_boundary.saturating_sub(start_pos).min(n_prompt);
                end = end.min(boundary_index.max(cursor + 1));
            }
            let rows = end - cursor;
            let profile_enabled = self.forward_profile.is_some();
            let embedding_start = profile_enabled.then(Instant::now);
            self.embed_prefill_tile(prompt_tokens, cursor, end, &mut tile)?;
            let embedding_elapsed = embedding_start
                .map(|start| start.elapsed())
                .unwrap_or_default();
            let absolute_start = start_pos + cursor;
            let mut attention_elapsed = Duration::ZERO;
            let mut ssm_elapsed = Duration::ZERO;
            let mut ssm_stage_profile = SsmStageProfile::default();
            let mut attention_stage_profile = AttentionStageProfile::default();

            for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
                match layer {
                    LayerWeights::Attention(w) => {
                        let stage_start = profile_enabled.then(Instant::now);
                        let params = crate::transformer::AttentionParams {
                            layer: layer_idx,
                            pos: absolute_start,
                            weights: w,
                            mmap: &self.gguf.mmap,
                            config: &self.config,
                            adapters: &self.active_adapters,
                            rope: &self.rope_cache,
                        };
                        attention_forward_tile_profiled(
                            &mut tile,
                            &mut self.state,
                            &mut self.kv_cache,
                            &params,
                            absolute_start,
                            rows,
                            profile_enabled.then_some(&mut attention_stage_profile),
                        )?;
                        if let Some(start) = stage_start {
                            attention_elapsed += start.elapsed();
                        }
                    }
                    LayerWeights::Ssm(w) => {
                        let stage_start = profile_enabled.then(Instant::now);
                        let params = crate::ssm::SsmParams {
                            layer: layer_idx,
                            weights: w,
                            mmap: &self.gguf.mmap,
                            config: &self.config,
                            adapters: &self.active_adapters,
                        };
                        if profile_enabled {
                            ssm_forward_tile_profiled(
                                &mut tile,
                                &mut self.state,
                                &params,
                                rows,
                                Some(&mut ssm_stage_profile),
                            )?;
                        } else {
                            ssm_forward_tile(&mut tile, &mut self.state, &params, rows)?;
                        }
                        if let Some(start) = stage_start {
                            ssm_elapsed += start.elapsed();
                        }
                    }
                }
            }

            let logits_start = profile_enabled.then(Instant::now);
            if end == n_prompt {
                let final_row = tile
                    .final_current_row(rows)
                    .map_err(|error| ModelError::ExecutionFailed(error.to_string()))?;
                self.state.x.copy_from_slice(final_row);
                self.compute_logits_from_state()?;
            }
            #[cfg(feature = "fixture-diagnostics")]
            if let Some(recorder) = self.fixture_recorder.as_mut() {
                recorder.processed(rows);
            }
            let logits_elapsed = logits_start
                .map(|start| start.elapsed())
                .unwrap_or_default();
            self.record_forward_profile_batch(
                rows,
                embedding_elapsed,
                attention_elapsed,
                ssm_elapsed,
                ssm_stage_profile,
                logits_elapsed,
            );
            if let Some(profile) = self.forward_profile.as_mut() {
                profile.attention_stages.add_assign(attention_stage_profile);
            }

            if n_prompt >= 500
                && (end.saturating_sub(last_progress_report) >= 500 || end == n_prompt)
            {
                last_progress_report = end;
                use std::io::Write;
                let pct = (end * 100) / n_prompt;
                println!(
                    "    \x1b[2m│  ⏳ prefill progress: [{}/{}] ({}%)\x1b[0m",
                    end, n_prompt, pct
                );
                let _ = std::io::stdout().flush();
            }

            let boundary_pos = start_pos + end;
            if start_pos == 0 && boundary_pos.is_multiple_of(self.prefix_cache.chunk_size()) {
                chained_hash =
                    self.cache_prefill_boundary(prompt_tokens, boundary_pos, chained_hash);
            }
            cursor = end;
        }

        Ok((true, chained_hash))
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
        self.generate_streaming_with_cancel(prompt, max_tokens, on_token, || false)
    }

    /// Prefill prompt tokens and generate tokens incrementally while allowing a caller to
    /// cooperatively stop work during both prefill and token generation.
    pub fn generate_streaming_with_cancel<F, C>(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        on_token: F,
        should_cancel: C,
    ) -> Result<String>
    where
        F: FnMut(u32, &str) -> bool,
        C: FnMut() -> bool,
    {
        #[cfg(feature = "fixture-diagnostics")]
        if let Some(recorder) = self.fixture_recorder.as_mut() {
            recorder.begin(Instant::now());
        }
        self.reset_context();
        #[cfg(feature = "fixture-diagnostics")]
        let cancellation_observed = std::cell::Cell::new(false);
        #[cfg(feature = "fixture-diagnostics")]
        let delivery_stopped = std::cell::Cell::new(false);
        #[cfg(feature = "fixture-diagnostics")]
        let result = {
            let mut on_token = on_token;
            let mut should_cancel = should_cancel;
            self.generate_streaming_incremental_with_cancel(
                prompt,
                0,
                max_tokens,
                |id, text| {
                    let keep_going = on_token(id, text);
                    if !keep_going {
                        delivery_stopped.set(true);
                    }
                    keep_going
                },
                || {
                    let cancelled = should_cancel();
                    if cancelled {
                        cancellation_observed.set(true);
                    }
                    cancelled
                },
            )
        };
        #[cfg(not(feature = "fixture-diagnostics"))]
        let result = self.generate_streaming_incremental_with_cancel(
            prompt,
            0,
            max_tokens,
            on_token,
            should_cancel,
        );
        #[cfg(feature = "fixture-diagnostics")]
        if let Some(recorder) = self.fixture_recorder.as_mut() {
            use crate::fixture_diagnostics::ModelOutcome;
            let outcome = if result.is_err() {
                ModelOutcome::ModelError
            } else if cancellation_observed.get() {
                ModelOutcome::Cancelled
            } else if delivery_stopped.get() {
                ModelOutcome::DeliveryStopped
            } else {
                ModelOutcome::Complete
            };
            recorder.finish(Instant::now(), outcome);
        }
        result
    }

    /// Prefill prompt tokens starting from `start_pos` without resetting KV cache or recurrent states.
    pub fn generate_streaming_incremental<F>(
        &mut self,
        prompt: &str,
        start_pos: usize,
        max_tokens: usize,
        on_token: F,
    ) -> Result<String>
    where
        F: FnMut(u32, &str) -> bool,
    {
        self.generate_streaming_incremental_with_cancel(
            prompt,
            start_pos,
            max_tokens,
            on_token,
            || false,
        )
    }

    /// Prefill prompt tokens starting from `start_pos` and generate tokens incrementally with
    /// cooperative cancellation checks.
    pub fn generate_streaming_incremental_with_cancel<F, C>(
        &mut self,
        prompt: &str,
        start_pos: usize,
        max_tokens: usize,
        on_token: F,
        should_cancel: C,
    ) -> Result<String>
    where
        F: FnMut(u32, &str) -> bool,
        C: FnMut() -> bool,
    {
        let mut token_ids = self.tokenizer.encode(prompt);
        if token_ids.is_empty() {
            #[cfg(feature = "fixture-diagnostics")]
            if let Some(recorder) = self.fixture_recorder.as_mut() {
                recorder.tokenization_done(Instant::now());
            }
            return Ok(String::new());
        }

        let add_bos = match self.gguf.metadata.get("tokenizer.ggml.add_bos_token") {
            Some(GgufValue::Bool(b)) => *b,
            _ => false,
        };
        let bos_id = self
            .gguf
            .metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|value| value.as_usize().map(|id| id as u32))
            .unwrap_or(1);
        if should_prepend_bos(add_bos, start_pos, token_ids.first(), bos_id) {
            token_ids.insert(0, bos_id);
        }
        #[cfg(feature = "fixture-diagnostics")]
        if let Some(recorder) = self.fixture_recorder.as_mut() {
            recorder.tokenization_done(Instant::now());
        }

        self.generate_tokens_incremental_with_cancel(
            &token_ids,
            start_pos,
            max_tokens,
            on_token,
            should_cancel,
        )
        .map(|(text, _)| text)
    }

    /// Prefill given token IDs directly starting from `start_pos` (skipping re-tokenization)
    /// and generate output tokens up to `max_tokens`.
    /// Returns a tuple of (generated_text, generated_token_ids).
    pub fn generate_tokens_incremental<F>(
        &mut self,
        prompt_tokens: &[u32],
        start_pos: usize,
        max_tokens: usize,
        on_token: F,
    ) -> Result<(String, Vec<u32>)>
    where
        F: FnMut(u32, &str) -> bool,
    {
        self.generate_tokens_incremental_with_cancel(
            prompt_tokens,
            start_pos,
            max_tokens,
            on_token,
            || false,
        )
    }

    /// Prefill given token IDs directly and generate tokens with cooperative cancellation checks.
    pub fn generate_tokens_incremental_with_cancel<F, C>(
        &mut self,
        prompt_tokens: &[u32],
        start_pos: usize,
        max_tokens: usize,
        mut on_token: F,
        mut should_cancel: C,
    ) -> Result<(String, Vec<u32>)>
    where
        F: FnMut(u32, &str) -> bool,
        C: FnMut() -> bool,
    {
        self.last_prefill_profile = None;
        if prompt_tokens.is_empty() && start_pos == 0 {
            return Ok((String::new(), Vec::new()));
        }
        if should_cancel() {
            return Ok((String::new(), Vec::new()));
        }

        let mut tokens_buf: Vec<u32>;
        let prompt_tokens: &[u32] = if start_pos == 0 {
            let add_bos = match self.gguf.metadata.get("tokenizer.ggml.add_bos_token") {
                Some(GgufValue::Bool(b)) => *b,
                _ => false,
            };
            let bos_id = self
                .gguf
                .metadata
                .get("tokenizer.ggml.bos_token_id")
                .and_then(|v| v.as_usize().map(|u| u as u32))
                .unwrap_or(1);
            if should_prepend_bos(add_bos, start_pos, prompt_tokens.first(), bos_id) {
                tokens_buf = Vec::with_capacity(prompt_tokens.len() + 1);
                tokens_buf.push(bos_id);
                tokens_buf.extend_from_slice(prompt_tokens);
                &tokens_buf
            } else {
                prompt_tokens
            }
        } else {
            prompt_tokens
        };

        let n_prompt = prompt_tokens.len();
        let context_end = checked_context_end(start_pos, n_prompt, self.config.max_seq_len)?;
        #[cfg(feature = "fixture-diagnostics")]
        if let Some(recorder) = self.fixture_recorder.as_mut() {
            recorder.prefill_begin(Instant::now(), n_prompt);
        }

        // 1. Check hierarchical prefix cache if starting from sequence position 0
        let mut start_prefill_idx = 0;
        let mut chained_hash = 0u64;

        if start_pos == 0 {
            // Always leave the final token unprocessed so logits are rebuilt
            // from the state BEFORE that token, especially for recurrent models.
            let reusable_tokens = &prompt_tokens[..n_prompt.saturating_sub(1)];
            if let Ok(Some(restored)) = self
                .prefix_cache
                .restore_longest_prefix(reusable_tokens, &mut self.kv_cache)
            {
                self.state
                    .import_ssm_states(&restored.ssm_conv_states, &restored.ssm_hidden_states);
                start_prefill_idx = restored.matched_tokens;
                chained_hash = restored.hash;
            }
        }

        // Suffix snapshots are not reused here. A hybrid KV/SSM state depends on the
        // complete preceding context, so matching token bytes alone cannot establish
        // that a snapshot is causally valid for this continuation.

        // 2. Prefill new prompt tokens (skipping already-cached prefix tokens)
        let prefill_result = self.run_prefill(
            prompt_tokens,
            start_pos,
            start_prefill_idx,
            &mut chained_hash,
            &mut should_cancel,
        );
        self.last_prefill_profile = self.forward_profile();
        #[cfg(feature = "fixture-diagnostics")]
        let prefill_profile = self.last_prefill_profile;
        #[cfg(feature = "fixture-diagnostics")]
        if let Some(recorder) = self.fixture_recorder.as_mut() {
            use crate::fixture_diagnostics::StageOutcome;
            recorder.prefill_profile(prefill_profile);
            let stage_outcome = match &prefill_result {
                Ok(true) => StageOutcome::Complete,
                Ok(false) => StageOutcome::Cancelled,
                Err(_) => StageOutcome::ModelError,
            };
            let processed = recorder.snapshot.processed_tokens.unwrap_or(0);
            recorder.prefill_end(Instant::now(), start_prefill_idx, processed, stage_outcome);
        }
        if !prefill_result? {
            return Ok((String::new(), Vec::new()));
        }
        let mut pos = context_end;

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

        let im_end_id = self.tokenizer.vocab().get_id("<|im_end|>");
        let endoftext_id = self.tokenizer.vocab().get_id("<|endoftext|>");

        // Generation loop
        for step in 0..max_tokens {
            if should_cancel() {
                break;
            }
            if pos >= self.config.max_seq_len {
                break;
            }

            let recent_slice = recent_tokens.make_contiguous();
            // Preserve raw logits by copying into logits_scratch for sampling
            self.state
                .logits_scratch
                .copy_from_slice(&self.state.logits);

            // Suppress EOS and end-of-sequence tokens on step 0 to prevent empty turn dropouts
            if step == 0 {
                if (eos_token_id as usize) < self.state.logits_scratch.len() {
                    self.state.logits_scratch[eos_token_id as usize] = f32::NEG_INFINITY;
                }
                if let Some(id) = im_end_id {
                    if (id as usize) < self.state.logits_scratch.len() {
                        self.state.logits_scratch[id as usize] = f32::NEG_INFINITY;
                    }
                }
                if let Some(id) = endoftext_id {
                    if (id as usize) < self.state.logits_scratch.len() {
                        self.state.logits_scratch[id as usize] = f32::NEG_INFINITY;
                    }
                }
            }

            let next_token = self
                .sampler
                .sample(&mut self.state.logits_scratch, recent_slice);

            let terminal_reason = if next_token == eos_token_id {
                Some("eos")
            } else if Some(next_token) == im_end_id {
                Some("im_end")
            } else if Some(next_token) == endoftext_id {
                Some("endoftext")
            } else {
                None
            };
            if let Some(_reason) = terminal_reason {
                #[cfg(feature = "fixture-diagnostics")]
                if let Some(recorder) = self.fixture_recorder.as_mut() {
                    recorder.terminal_token(next_token);
                    recorder.stopping_reason(_reason);
                }
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
            #[cfg(feature = "fixture-diagnostics")]
            if let Some(recorder) = self.fixture_recorder.as_mut() {
                recorder.raw(Instant::now(), &decoded_chunk);
                recorder.snapshot.generated_ids.push(next_token);
            }
            pending_text.push_str(&decoded_chunk);

            // Check for full stop sequence matches
            if let Some(matched_len) =
                matches_any_stop_suffix(&pending_text, &self.sampler.config.stop_tokens)
            {
                #[cfg(feature = "fixture-diagnostics")]
                if let Some(recorder) = self.fixture_recorder.as_mut() {
                    recorder.stopping_reason("stop_sequence");
                }
                let keep_len = pending_text.len().saturating_sub(matched_len);
                pending_text.truncate(keep_len);
                if !pending_text.is_empty() {
                    #[cfg(feature = "fixture-diagnostics")]
                    if let Some(recorder) = self.fixture_recorder.as_mut() {
                        recorder.delivered(Instant::now(), &pending_text);
                    }
                    // Preserve this site's existing ignored callback result.
                    let _keep_going = on_token(next_token, &pending_text);
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
                if !emit_str.is_empty() {
                    #[cfg(feature = "fixture-diagnostics")]
                    if let Some(recorder) = self.fixture_recorder.as_mut() {
                        recorder.delivered(Instant::now(), &emit_str);
                    }
                    let keep_going = on_token(next_token, &emit_str);
                    if !keep_going {
                        break;
                    }
                }
            }

            let _ = self.forward_step(next_token, pos, true)?;

            pos += 1;
        }

        // Flush remaining decoder bytes
        let flushed = stream_decoder.flush();
        #[cfg(feature = "fixture-diagnostics")]
        if let Some(recorder) = self.fixture_recorder.as_mut() {
            recorder.raw(Instant::now(), &flushed);
        }
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
                #[cfg(feature = "fixture-diagnostics")]
                if let Some(recorder) = self.fixture_recorder.as_mut() {
                    recorder.delivered(Instant::now(), &pending_text);
                }
                // Preserve this site's existing ignored callback result.
                let _keep_going = on_token(last_id, &pending_text);
            }
        }

        let full_decoded = self.tokenizer.decode(&generated_ids);
        let mut clean_result = full_decoded;
        for st in &self.sampler.config.stop_tokens {
            if !st.is_empty() && clean_result.ends_with(st.as_str()) {
                clean_result.truncate(clean_result.len() - st.len());
            }
        }
        Ok((clean_result, generated_ids))
    }

    /// Prefill prompt tokens and generate full string up to `max_tokens`.
    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        self.generate_streaming(prompt, max_tokens, |_, _| true)
    }

    /// Generate text strictly constrained to valid JSON syntax via logit masking.
    pub fn generate_with_json_grammar(
        &mut self,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<String> {
        self.generate_with_json_grammar_and_cancel(prompt, max_tokens, || false)
    }

    /// Generate syntax-constrained JSON with cancellation before each model forward.
    /// Cancellation may return partial JSON; callers must not publish it as success.
    pub fn generate_with_json_grammar_and_cancel<C>(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        mut should_cancel: C,
    ) -> Result<String>
    where
        C: FnMut() -> bool,
    {
        if should_cancel() {
            return Ok(String::new());
        }
        let mut grammar = crate::grammar::JsonGrammar::new();
        let token_ids = self.tokenizer.encode(prompt);
        if token_ids.is_empty() {
            return Ok(String::new());
        }

        self.reset_context();
        let mut generated_ids = Vec::new();
        let mut recent_tokens = VecDeque::with_capacity(RECENT_TOKENS_WINDOW + 1);
        let mut stream_decoder = mivi_tokenizer::Utf8StreamDecoder::new();
        let mut full_output = String::new();
        let mut raw_bytes = Vec::new();

        // 1. Prefill
        for (i, &tok) in token_ids.iter().enumerate() {
            if should_cancel() {
                return Ok(String::new());
            }
            let is_last = i + 1 == token_ids.len();
            let _ = self.forward_step(tok, i, is_last)?;
        }
        let mut pos = token_ids.len();

        let eos_token_id = self
            .gguf
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_usize().map(|u| u as u32))
            .unwrap_or(EOS_TOKEN_ID);

        // If prompt ended with '{', feed it into the grammar
        if prompt.ends_with('{') {
            grammar.feed("{");
        }

        // 2. Generation with grammar logit masking
        for _ in 0..max_tokens {
            if should_cancel() || pos >= self.config.max_seq_len || grammar.completed {
                break;
            }

            self.state
                .logits_scratch
                .copy_from_slice(&self.state.logits);

            // Apply grammar mask
            let mask = grammar.compute_mask(self.tokenizer.vocab());
            mask.apply_to_logits(&mut self.state.logits_scratch);

            let recent_slice = recent_tokens.make_contiguous();
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

            raw_bytes.clear();
            self.tokenizer
                .decode_token_bytes(next_token, &mut raw_bytes);
            let decoded_chunk = stream_decoder.feed(&raw_bytes);
            if !decoded_chunk.is_empty() {
                grammar.feed(&decoded_chunk);
                full_output.push_str(&decoded_chunk);
                if grammar.completed {
                    break;
                }
            }

            if should_cancel() {
                break;
            }
            let _ = self.forward_step(next_token, pos, true)?;
            pos += 1;
        }

        Ok(full_output)
    }
}

/// Helper to find the maximum length of a stop token prefix matching the tail of `text`.
fn longest_stop_prefix_len(text: &str, stop_tokens: &[String]) -> usize {
    let mut max_match = 0;
    for st in stop_tokens {
        if st.is_empty() {
            continue;
        }
        let mut char_indices = st.char_indices();
        let _ = char_indices.next();
        while let Some((len, _)) = char_indices.next_back() {
            if len <= text.len() && text.ends_with(&st[..len]) {
                max_match = max_match.max(len);
                break;
            }
        }
    }
    max_match
}

#[inline]
fn cache_chunk_tokens(tokens: &[u32]) -> (Vec<u32>, u64) {
    (tokens.to_vec(), compute_chunk_hash(0, tokens))
}

fn checked_context_end(start_pos: usize, token_count: usize, max_seq_len: usize) -> Result<usize> {
    let end = start_pos
        .checked_add(token_count)
        .ok_or(ModelError::ContextOverflow {
            pos: usize::MAX,
            max: max_seq_len,
        })?;
    if end > max_seq_len {
        return Err(ModelError::ContextOverflow {
            pos: end,
            max: max_seq_len,
        });
    }
    Ok(end)
}

#[inline]
fn should_prepend_bos(
    add_bos: bool,
    start_pos: usize,
    first_token: Option<&u32>,
    bos_id: u32,
) -> bool {
    add_bos && start_pos == 0 && first_token != Some(&bos_id)
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

#[cfg(test)]
mod prefix_cache_integration_tests {
    #[test]
    #[ignore = "requires explicit MIVI_TEST_MODEL; observes real streaming hooks"]
    #[cfg(feature = "fixture-diagnostics")]
    fn fixture_model_observation() -> std::result::Result<(), Box<dyn std::error::Error>> {
        use crate::fixture_diagnostics::{CaptureLimits, ModelOutcome};

        let path = std::env::var("MIVI_TEST_MODEL")?;
        let mut model = Model::load_with_ctx(std::path::Path::new(&path), Some(512))?;
        model.sampler.config.temperature = 0.0;
        model.sampler.set_seed(7);
        model
            .start_fixture_capture(CaptureLimits {
                text_bytes: 4096,
                token_ids: 32,
            })
            .map_err(|message| std::io::Error::new(std::io::ErrorKind::InvalidInput, message))?;
        let mut delivered = String::new();
        model.generate_streaming("Return the word hello.", 16, |_, text| {
            delivered.push_str(text);
            true
        })?;
        assert!(
            !delivered.is_empty(),
            "observation fixture did not exercise decoded output"
        );
        let capture = model
            .take_fixture_capture()
            .ok_or("missing fixture capture")?;
        assert!(
            !capture.raw_decoded.text.is_empty(),
            "raw observation missing"
        );
        assert!(
            capture.delivered.text == delivered,
            "delivered observation mismatch"
        );
        assert_eq!(capture.outcome, Some(ModelOutcome::Complete));
        assert!(model.take_fixture_capture().is_none());
        Ok(())
    }

    #[test]
    #[ignore = "requires explicit MIVI_TEST_MODEL; verifies bounded real-model lifecycle and parity"]
    #[cfg(feature = "fixture-diagnostics")]
    fn fixture_model_parity_and_lifecycle() -> std::result::Result<(), Box<dyn std::error::Error>> {
        use crate::fixture_diagnostics::{CaptureLimits, ModelOutcome, StageOutcome};

        const PROMPT: &str = "Return the word hello.";
        const BUDGET: usize = 4;
        #[derive(Clone, Copy)]
        enum Gate {
            Initial,
            AfterPrefillUnit,
            AfterDelivery,
        }

        // The worker cannot physically return until the test releases this gate.
        // Count only the existing cancellation callback evaluations.
        fn gated_cancel(model: &mut Model, gate: Gate) -> Result<(String, String, usize, u64)> {
            use std::sync::mpsc::{channel, sync_channel, TryRecvError};

            let (entered_tx, entered_rx) = channel();
            let (cancel_tx, cancel_rx) = sync_channel(0);
            let (returned_tx, returned_rx) = channel();
            std::thread::scope(|scope| {
                let worker = scope.spawn(move || {
                    let delivered_once = std::cell::Cell::new(false);
                    let mut delivered = String::new();
                    let mut checks = 0;
                    let result = model.generate_streaming_with_cancel(
                        PROMPT,
                        BUDGET,
                        |_, text| {
                            delivered.push_str(text);
                            delivered_once.set(true);
                            true
                        },
                        || {
                            checks += 1;
                            let at_gate = match gate {
                                Gate::Initial => checks == 1,
                                Gate::AfterPrefillUnit => checks == 3,
                                Gate::AfterDelivery => delivered_once.get(),
                            };
                            if at_gate {
                                entered_tx.send(()).expect("gate receiver disconnected");
                                cancel_rx
                                    .recv_timeout(Duration::from_secs(120))
                                    .expect("cancellation gate disconnected or timed out")
                            } else {
                                false
                            }
                        },
                    );
                    returned_tx.send(()).expect("return receiver disconnected");
                    result.map(|output| (output, delivered, checks, model.sampler.rng_state()))
                });
                entered_rx
                    .recv_timeout(Duration::from_secs(120))
                    .expect("bounded cancellation gate was not reached");
                assert!(
                    matches!(returned_rx.try_recv(), Err(TryRecvError::Empty)),
                    "generation returned before gate release"
                );
                cancel_tx
                    .send(true)
                    .expect("generation left cancellation gate");
                let result = worker.join().expect("generation worker panicked");
                assert!(
                    returned_rx.try_recv().is_ok(),
                    "physical return was not signalled"
                );
                result
            })
        }

        let limits = CaptureLimits {
            text_bytes: 4096,
            token_ids: 32,
        };
        let arm = |model: &mut Model| {
            model
                .start_fixture_capture(limits)
                .map_err(|message| std::io::Error::new(std::io::ErrorKind::InvalidInput, message))
        };
        let strategies = [
            PrefillStrategy::Token,
            PrefillStrategy::Chunked { tile_tokens: 2 },
        ];
        let gates = [Gate::Initial, Gate::AfterPrefillUnit, Gate::AfterDelivery];
        let path = std::env::var("MIVI_TEST_MODEL")?;

        // Drop the entire unarmed model before loading the observed one.
        let (
            baseline_output,
            baseline_delivered,
            baseline_chunks,
            baseline_ids,
            baseline_rng,
            baseline_cancelled,
        ) = {
            let mut model = Model::load_with_ctx(std::path::Path::new(&path), Some(512))?;
            model.set_prefill_strategy(PrefillStrategy::Token)?;
            model.sampler.config.temperature = 0.0;
            model.sampler.set_seed(7);
            assert!(model.fixture_recorder.is_none());
            let mut delivered = String::new();
            let mut chunks = Vec::new();
            let mut ids = Vec::new();
            let output = model.generate_streaming(PROMPT, BUDGET, |id, text| {
                delivered.push_str(text);
                chunks.push(text.to_owned());
                ids.push(id);
                true
            })?;
            assert!(
                !delivered.is_empty(),
                "parity fixture did not exercise delivery"
            );
            let rng = model.sampler.rng_state();
            let mut cancelled = Vec::new();
            for strategy in strategies {
                model.set_prefill_strategy(strategy)?;
                for gate in gates {
                    model.prefix_cache.clear();
                    model.sampler.set_seed(7);
                    cancelled.push(gated_cancel(&mut model, gate)?);
                    assert!(model.fixture_recorder.is_none());
                    assert!(model.take_fixture_capture().is_none());
                }
            }
            (output, delivered, chunks, ids, rng, cancelled)
        };

        let mut model = Model::load_with_ctx(std::path::Path::new(&path), Some(512))?;
        model.set_prefill_strategy(PrefillStrategy::Token)?;
        model.sampler.config.temperature = 0.0;
        model.sampler.set_seed(7);
        assert!(model.fixture_recorder.is_none());
        assert!(model
            .start_fixture_capture(CaptureLimits {
                text_bytes: 0,
                token_ids: 32
            })
            .is_err());
        assert!(model.fixture_recorder.is_none());
        arm(&mut model)?;
        assert!(model.start_fixture_capture(limits).is_err());
        let mut delivered = String::new();
        let mut ids = Vec::new();
        let output = model.generate_streaming(PROMPT, BUDGET, |id, text| {
            delivered.push_str(text);
            ids.push(id);
            true
        })?;
        assert!(
            output == baseline_output,
            "observer changed generated output"
        );
        assert!(
            delivered == baseline_delivered,
            "observer changed delivered output"
        );
        assert!(ids == baseline_ids, "observer changed delivery token IDs");
        assert_eq!(model.sampler.rng_state(), baseline_rng);
        let capture = model
            .take_fixture_capture()
            .ok_or("missing parity capture")?;
        assert!(
            capture.delivered.text == delivered,
            "delivery hook mismatch"
        );
        assert!(
            !capture.raw_decoded.text.is_empty(),
            "raw hook was not exercised"
        );
        assert_eq!(capture.outcome, Some(ModelOutcome::Complete));
        assert_eq!(capture.prefill_outcome, Some(StageOutcome::Complete));
        assert_eq!(capture.processed_tokens, capture.prompt_tokens);
        assert_eq!(capture.reused_tokens, Some(0));
        assert!(
            capture.tokenization.is_some() && capture.prefill.is_some() && capture.decode.is_some()
        );
        assert!(capture.first_raw.is_some() && capture.first_delivered.is_some());
        assert!(capture.first_raw <= capture.first_delivered);
        assert!(model.take_fixture_capture().is_none());
        let generated_ids = capture.generated_ids.ids;
        let eos = model
            .gguf
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(GgufValue::as_usize)
            .unwrap_or(EOS_TOKEN_ID as usize) as u32;
        assert!(
            generated_ids.iter().all(|&id| id != eos
                && Some(id) != model.tokenizer.vocab().get_id("<|im_end|>")
                && Some(id) != model.tokenizer.vocab().get_id("<|endoftext|>")),
            "termination ID was captured"
        );

        let mut baseline_cases = baseline_cancelled.into_iter();
        for strategy in strategies {
            model.set_prefill_strategy(strategy)?;
            for gate in gates {
                model.prefix_cache.clear();
                model.sampler.set_seed(7);
                arm(&mut model)?;
                let actual = gated_cancel(&mut model, gate)?;
                let expected = baseline_cases
                    .next()
                    .ok_or("missing baseline cancellation case")?;
                assert!(
                    actual == expected,
                    "observer changed gated cancellation behavior"
                );
                let capture = model
                    .take_fixture_capture()
                    .ok_or("missing cancellation capture")?;
                assert_eq!(capture.outcome, Some(ModelOutcome::Cancelled));
                assert!(capture.tokenization.is_some());
                assert!(
                    capture.delivered.text == actual.1,
                    "cancelled delivery hook mismatch"
                );
                match gate {
                    Gate::Initial => {
                        assert_eq!(actual.2, 1);
                        assert_eq!(capture.prefill, None);
                        assert_eq!(capture.decode, None);
                        assert_eq!(capture.processed_tokens, None);
                    }
                    Gate::AfterPrefillUnit => {
                        assert_eq!(actual.2, 3);
                        assert_eq!(capture.prefill_outcome, Some(StageOutcome::Cancelled));
                        let completed = match strategy {
                            PrefillStrategy::Token => 1,
                            PrefillStrategy::Chunked { tile_tokens } => tile_tokens,
                        };
                        assert_eq!(capture.processed_tokens, Some(completed));
                        assert!(capture.prompt_tokens.unwrap() > completed);
                        assert_eq!(capture.reused_tokens, Some(0));
                        assert_eq!(capture.decode, None);
                        assert!(capture.generated_ids.ids.is_empty());
                    }
                    Gate::AfterDelivery => {
                        assert_eq!(capture.prefill_outcome, Some(StageOutcome::Complete));
                        assert_eq!(capture.processed_tokens, capture.prompt_tokens);
                        assert!(capture.decode.is_some());
                        assert!(!capture.generated_ids.ids.is_empty());
                        assert!(
                            !actual.1.is_empty(),
                            "decode cancellation gate did not exercise delivery"
                        );
                    }
                }
            }
        }

        // The remaining probes reuse the observed model, clearing its prefix cache.
        model.set_prefill_strategy(PrefillStrategy::Token)?;
        arm(&mut model)?;
        let mut checks = 0;
        let empty = model.generate_streaming_with_cancel(
            "",
            1,
            |_, _| panic!("empty encoding delivered text"),
            || {
                checks += 1;
                true
            },
        )?;
        assert!(empty.is_empty());
        assert_eq!(checks, 0);
        let empty = model
            .take_fixture_capture()
            .ok_or("missing empty encoding capture")?;
        assert!(empty.tokenization.is_some());
        assert_eq!(empty.outcome, Some(ModelOutcome::Complete));
        assert_eq!(empty.prefill, None);
        assert_eq!(empty.decode, None);
        assert_eq!(empty.processed_tokens, None);
        assert_eq!(empty.first_raw, None);
        assert_eq!(empty.first_delivered, None);
        assert!(empty.generated_ids.ids.is_empty());

        let context_limit = model.config.max_seq_len;
        model.config.max_seq_len = 1;
        arm(&mut model)?;
        let result = model.generate_streaming(PROMPT, 1, |_, _| true);
        assert!(matches!(result, Err(ModelError::ContextOverflow { .. })));
        model.config.max_seq_len = context_limit;
        let error = model
            .take_fixture_capture()
            .ok_or("missing context error capture")?;
        assert_eq!(error.outcome, Some(ModelOutcome::ModelError));
        assert!(error.tokenization.is_some());
        assert_eq!(error.prefill, None);
        assert_eq!(error.decode, None);

        let output_norm = model.weights.output_norm.take();
        model.weights.output_norm = Some(Box::default());
        for strategy in strategies {
            model.prefix_cache.clear();
            model.set_prefill_strategy(strategy)?;
            arm(&mut model)?;
            let result = model.generate_streaming(PROMPT, 1, |_, _| true);
            assert!(matches!(result, Err(ModelError::DimMismatch(_))));
            let error = model
                .take_fixture_capture()
                .ok_or("missing prefill error capture")?;
            assert_eq!(error.outcome, Some(ModelOutcome::ModelError));
            assert_eq!(error.prefill_outcome, Some(StageOutcome::ModelError));
            assert!(error.prefill.is_some());
            let prompt_tokens = error.prompt_tokens.ok_or("missing prefill token count")?;
            let completed = match strategy {
                PrefillStrategy::Token => prompt_tokens - 1,
                PrefillStrategy::Chunked { tile_tokens } => {
                    ((prompt_tokens - 1) / tile_tokens) * tile_tokens
                }
            };
            assert_eq!(error.processed_tokens, Some(completed));
            assert_eq!(error.reused_tokens, Some(0));
            assert_eq!(error.decode, None);
            assert!(error.generated_ids.ids.is_empty());
        }
        model.weights.output_norm = output_norm;
        model.set_prefill_strategy(PrefillStrategy::Token)?;
        model.prefix_cache.clear();
        model.sampler.set_seed(7);
        arm(&mut model)?;
        let mut delivered = String::new();
        model.generate_streaming(PROMPT, 1, |_, text| {
            delivered.push_str(text);
            false
        })?;
        let stopped = model
            .take_fixture_capture()
            .ok_or("missing delivery stop capture")?;
        assert_eq!(stopped.outcome, Some(ModelOutcome::DeliveryStopped));
        assert!(
            stopped.delivered.text == delivered,
            "delivery stop hook mismatch"
        );
        assert!(
            !delivered.is_empty(),
            "one-token delivery fixture was unexercised"
        );

        // A complete stop trims raw bytes; an incomplete stop prefix is emitted
        // only at the existing final pending-text callback, whose false is ignored.
        let first_chunk = baseline_chunks
            .first()
            .ok_or("missing first delivery chunk")?;
        model.prefix_cache.clear();
        model.sampler.set_seed(7);
        model.sampler.config.stop_tokens = vec![first_chunk.clone()];
        arm(&mut model)?;
        let output =
            model.generate_streaming(PROMPT, 1, |_, _| panic!("complete stop delivered text"))?;
        let stopped = model
            .take_fixture_capture()
            .ok_or("missing text stop capture")?;
        assert!(output.is_empty());
        assert!(
            stopped.raw_decoded.text == *first_chunk,
            "raw stop hook mismatch"
        );
        assert!(stopped.delivered.text.is_empty());
        assert_eq!(stopped.outcome, Some(ModelOutcome::Complete));
        assert_eq!(stopped.generated_ids.observed_tokens, 1);
        assert!(stopped.first_raw.is_some());
        assert_eq!(stopped.first_delivered, None);

        if let Some((keep, _)) = first_chunk
            .char_indices()
            .last()
            .filter(|(keep, _)| *keep > 0)
        {
            model.prefix_cache.clear();
            model.sampler.set_seed(7);
            model.sampler.config.stop_tokens = vec![first_chunk[keep..].to_owned()];
            arm(&mut model)?;
            let mut calls = 0;
            let output = model.generate_streaming(PROMPT, 1, |_, text| {
                calls += 1;
                assert!(
                    text == &first_chunk[..keep],
                    "matched-stop delivery mismatch"
                );
                false
            })?;
            let stopped = model
                .take_fixture_capture()
                .ok_or("missing matched-stop delivery capture")?;
            assert!(
                output == first_chunk[..keep],
                "ignored stop callback changed output"
            );
            assert!(
                stopped.raw_decoded.text == *first_chunk,
                "matched-stop raw mismatch"
            );
            assert!(
                stopped.delivered.text == first_chunk[..keep],
                "matched-stop hook mismatch"
            );
            assert_eq!(stopped.outcome, Some(ModelOutcome::DeliveryStopped));
            assert_eq!(calls, 1);
            eprintln!("[fixture] matched-stop callback with retained text: exercised");
        } else {
            eprintln!("[fixture] matched-stop callback with retained text: unexercised (single-character first chunk)");
        }

        model.prefix_cache.clear();
        model.sampler.set_seed(7);
        model.sampler.config.stop_tokens = vec![format!("{first_chunk}\u{10ffff}")];
        arm(&mut model)?;
        let mut calls = 0;
        let output = model.generate_streaming(PROMPT, 1, |_, text| {
            calls += 1;
            assert!(text == first_chunk, "pending flush delivery mismatch");
            false
        })?;
        let flushed = model
            .take_fixture_capture()
            .ok_or("missing pending flush capture")?;
        assert!(
            output == *first_chunk,
            "ignored final callback changed output"
        );
        assert!(
            flushed.raw_decoded.text == *first_chunk,
            "flush raw hook mismatch"
        );
        assert!(
            flushed.delivered.text == *first_chunk,
            "flush delivery hook mismatch"
        );
        assert_eq!(calls, 1);
        assert_eq!(flushed.generated_ids.observed_tokens, 1);
        assert_eq!(flushed.outcome, Some(ModelOutcome::DeliveryStopped));

        // Use an already observed later token as EOS so the real termination
        // branch is exercised without increasing the generation budget.
        let (stop_at, &termination_id) = generated_ids
            .iter()
            .enumerate()
            .skip(1)
            .find(|(_, id)| **id != generated_ids[0])
            .ok_or("bounded EOS fixture lacks a distinct later token")?;
        model.prefix_cache.clear();
        model.sampler.set_seed(7);
        model.sampler.config.stop_tokens.clear();
        model.gguf.metadata.insert(
            "tokenizer.ggml.eos_token_id".to_owned(),
            GgufValue::U32(termination_id),
        );
        arm(&mut model)?;
        model.generate_streaming(PROMPT, BUDGET, |_, _| true)?;
        let eos_capture = model.take_fixture_capture().ok_or("missing EOS capture")?;
        assert_eq!(eos_capture.generated_ids.observed_tokens, stop_at);
        assert!(
            eos_capture.generated_ids.ids == generated_ids[..stop_at],
            "EOS branch changed preceding token IDs"
        );
        assert!(
            eos_capture
                .generated_ids
                .ids
                .iter()
                .all(|&id| id != termination_id),
            "EOS ID was captured"
        );
        assert_eq!(eos_capture.outcome, Some(ModelOutcome::Complete));
        assert!(model.take_fixture_capture().is_none());
        Ok(())
    }

    #[test]
    #[ignore = "requires MIVI_TEST_MODEL; compares profiling on/off"]
    fn attention_tile_profile_is_opt_in_and_resets() {
        let model_path = std::env::var("MIVI_TEST_MODEL").unwrap();
        let mut model = super::Model::load(std::path::Path::new(&model_path)).unwrap();
        model
            .set_prefill_strategy(super::PrefillStrategy::Chunked { tile_tokens: 64 })
            .unwrap();
        let tokens: Vec<u32> = (0..64)
            .map(|i| ((i % (model.config.vocab_size - 1)) + 1) as u32)
            .collect();
        assert!(model.forward_profile().is_none());
        model
            .generate_tokens_incremental(&tokens, 0, 0, |_, _| true)
            .unwrap();
        let logits = model.state.logits.clone();
        let conv = model.state.conv_states.clone();
        let kv = model.kv_cache.export_state(tokens.len()).unwrap();

        model.prefix_cache.clear();
        model.reset_context();
        model.enable_forward_profile();
        model
            .generate_tokens_incremental(&tokens, 0, 0, |_, _| true)
            .unwrap();
        assert_eq!(model.state.logits, logits);
        assert_eq!(model.state.conv_states, conv);
        assert_eq!(model.kv_cache.export_state(tokens.len()).unwrap(), kv);
        let profile = model.forward_profile().unwrap();
        assert_eq!(profile.tokens, tokens.len());
        assert!(profile.attention_stages.total() <= profile.attention);
        if model
            .config
            .block_types
            .contains(&super::BlockType::Attention)
        {
            assert!(!profile.attention_stages.total().is_zero());
        }
        model.reset_forward_profile();
        assert_eq!(
            model.forward_profile(),
            Some(super::ForwardProfileSnapshot::default())
        );
        model.disable_forward_profile();
        assert!(model.forward_profile().is_none());
    }

    #[test]
    #[ignore = "requires MIVI_TEST_MODEL; compares cold and restored model states"]
    fn test_delta_prefix_reuse_matches_cold_inference() {
        let model_path = std::env::var("MIVI_TEST_MODEL").unwrap();
        let mut model = super::Model::load(std::path::Path::new(&model_path)).unwrap();
        model
            .set_prefill_strategy(super::PrefillStrategy::Chunked { tile_tokens: 64 })
            .unwrap();
        model.sampler.config.temperature = 0.0;
        let lengths = std::env::var("MIVI_TEST_PROMPT_LEN")
            .map(|value| vec![value.parse::<usize>().unwrap()])
            .unwrap_or_else(|_| vec![64, 128, 129, 192]);
        for len in lengths {
            let tokens: Vec<u32> = (0..len)
                .map(|i| ((i % (model.config.vocab_size - 1)) + 1) as u32)
                .collect();
            model.prefix_cache.clear();
            model.reset_context();
            model
                .generate_tokens_incremental(&tokens, 0, 0, |_, _| true)
                .unwrap();
            let expected_logits = model.state.logits.clone();
            let expected_conv = model.state.conv_states.clone();
            let expected_kv = model.kv_cache.export_state(len).unwrap();
            model.reset_context();
            model
                .generate_tokens_incremental(&tokens, 0, 0, |_, _| true)
                .unwrap();
            assert_eq!(model.current_pos(), len);
            assert!(
                model.kv_cache.export_state(len).unwrap() == expected_kv,
                "KV mismatch at {len}"
            );
            assert!(
                model.state.conv_states == expected_conv,
                "SSM mismatch at {len}"
            );
            assert!(
                model.state.logits == expected_logits,
                "logits mismatch at {len}"
            );
        }
    }
    use super::*;

    #[test]
    fn prefix_cache_chunk_hash_uses_the_complete_token_chunk() {
        let chunk = vec![1u32, 10, 20, 30];
        let (stored_tokens, stored_hash) = cache_chunk_tokens(&chunk);

        assert_eq!(stored_tokens, chunk);
        assert_eq!(stored_hash, compute_chunk_hash(0, &chunk));
    }

    /// Direct unit test: suffix matching using only public PrefixCache APIs.
    #[test]
    fn test_suffix_match_directly() {
        use mivi_kv::{compute_chunk_hash, HybridStateSnapshot, PrefixCache};

        let mut cache = PrefixCache::new(4, 64);

        // Use insert_chunk (public API) to store a synthetic chunk.
        // chained_hash=0 makes the stored key equal to standalone_hash.
        let chunk_tokens: Vec<u32> = (1000u32..1064).collect();
        let standalone_hash = compute_chunk_hash(0, &chunk_tokens);
        let state = HybridStateSnapshot::new(64, standalone_hash, vec![], vec![], vec![], vec![]);
        cache.insert_chunk(0, &chunk_tokens, 0, state);

        // Simulate agent step 1: 64 history + same 64-token chunk suffix
        let history: Vec<u32> = (2000u32..2064).collect();
        let step1_input: Vec<u32> = history.iter().chain(chunk_tokens.iter()).cloned().collect();

        let result = cache.find_longest_suffix_match(64, &step1_input);
        assert!(result.is_some(), "suffix match should find cached chunk");
        let (skip, pos, _) = result.unwrap();
        assert_eq!(skip, 64, "should skip 64 tokens");
        assert_eq!(pos, 64, "match at position 64");
    }

    #[test]
    fn incremental_context_position_overflow_is_rejected() {
        assert!(checked_context_end(usize::MAX, 1, usize::MAX).is_err());
        assert_eq!(checked_context_end(4, 3, 8).unwrap(), 7);
    }

    #[test]
    fn unicode_stop_prefix_does_not_panic() {
        let stop_tokens = vec!["💥".to_string()];

        assert_eq!(longest_stop_prefix_len("x", &stop_tokens), 0);
    }

    #[test]
    fn bos_insertion_uses_token_identity_instead_of_prompt_markup() {
        assert!(!should_prepend_bos(true, 0, Some(&7), 7));
        assert!(should_prepend_bos(true, 0, Some(&8), 7));
        assert!(!should_prepend_bos(true, 1, Some(&8), 7));
        assert!(!should_prepend_bos(false, 0, Some(&8), 7));
    }

    #[test]
    fn forward_profile_snapshot_sums_stage_durations() {
        let profile = ForwardProfileSnapshot {
            tokens: 3,
            embedding: std::time::Duration::from_millis(1),
            attention: std::time::Duration::from_millis(2),
            attention_stages: AttentionStageProfile {
                causal_attention: std::time::Duration::from_millis(2),
                ..AttentionStageProfile::default()
            },
            ssm: std::time::Duration::from_millis(3),
            ssm_stages: SsmStageProfile::default(),
            logits: std::time::Duration::from_millis(4),
        };

        assert_eq!(
            profile.total_stage_time(),
            std::time::Duration::from_millis(10)
        );
    }

    /// Bounded teacher-forced decode diagnostics; never runs in the default test suite.
    #[test]
    #[ignore = "requires MIVI_TEST_MODEL and explicit release execution"]
    fn decode_substage_profile_parity_and_measurement() {
        assert!(
            !cfg!(debug_assertions),
            "run this diagnostic in release mode"
        );
        assert_eq!(rayon::current_num_threads(), 2, "set RAYON_NUM_THREADS=2");
        let model_path = std::env::var("MIVI_TEST_MODEL").unwrap();
        let mut model =
            Model::load_with_ctx(std::path::Path::new(&model_path), Some(1024)).unwrap();
        model
            .set_prefill_strategy(PrefillStrategy::Chunked { tile_tokens: 64 })
            .unwrap();
        let prompt =
            "Read the workspace files and explain how to handle a parsing error. ".repeat(32);
        let prompt_ids: Vec<_> = model
            .tokenizer
            .encode(&prompt)
            .into_iter()
            .take(256)
            .collect();
        let continuation = model.tokenizer.encode(
            "Inspect the input, return a useful error, and preserve the original file contents before making changes.",
        );
        assert!(prompt_ids.len() == 256 && continuation.len() >= 16);
        let continuation = &continuation[..16];
        let mut profiles = Vec::new();
        let mut profiled_walls = Vec::new();
        let mut unprofiled_walls = Vec::new();
        model.prefix_cache.clear();

        // One warmup pair followed by three pairs; alternate order to reduce order bias.
        for repetition in 0..4 {
            let mut expected = None;
            for profile_enabled in [repetition % 2 == 0, repetition % 2 != 0] {
                model.disable_forward_profile();
                model.reset_context();
                // Prefill is outside the timed region. Retain its prefix cache to avoid
                // repeating expensive cold prefill for every fixed-work decode member.
                model
                    .generate_tokens_incremental(&prompt_ids, 0, 0, |_, _| true)
                    .unwrap();
                let start_pos = model.current_pos();
                if profile_enabled {
                    model.enable_forward_profile();
                }
                let started = Instant::now();
                let mut logits = Vec::new();
                for (offset, &id) in continuation.iter().enumerate() {
                    let values = model.forward(id, start_pos + offset).unwrap().to_vec();
                    assert!(values.iter().all(|value| value.is_finite()));
                    logits.push(values);
                }
                let wall = started.elapsed();
                if repetition > 0 {
                    println!(
                        "decode pair={} profile={} effective_prefix={} forwards=16 wall_s={:.4}",
                        repetition,
                        profile_enabled,
                        start_pos,
                        wall.as_secs_f64(),
                    );
                }
                let actual = (
                    logits,
                    model.state.conv_states.clone(),
                    model.state.ssm_states.clone(),
                    model
                        .kv_cache
                        .export_state(start_pos + continuation.len())
                        .unwrap(),
                    model.current_pos(),
                );
                if let Some(expected) = &expected {
                    assert_eq!(
                        &actual, expected,
                        "profiling changed logits or recurrent/KV state"
                    );
                } else {
                    expected = Some(actual);
                }
                if profile_enabled {
                    let profile = model.forward_profile().unwrap();
                    assert_eq!(profile.tokens, continuation.len());
                    assert!(profile.attention_stages.total() <= profile.attention);
                    assert!(profile.ssm_stages.total() <= profile.ssm);
                    if model.config.block_types.contains(&BlockType::Attention) {
                        assert!(!profile.attention_stages.ffn.is_zero());
                    }
                    if model.config.block_types.contains(&BlockType::SSM) {
                        assert!(!profile.ssm_stages.ffn.is_zero());
                    }
                    if repetition > 0 {
                        profiles.push(profile);
                        profiled_walls.push(wall.as_secs_f64());
                    }
                } else {
                    assert!(model.forward_profile().is_none());
                    if repetition > 0 {
                        unprofiled_walls.push(wall.as_secs_f64());
                    }
                }
            }
        }
        let median = |mut values: Vec<f64>| {
            values.sort_by(f64::total_cmp);
            values[values.len() / 2]
        };
        let stage_ms = |select: fn(&ForwardProfileSnapshot) -> Duration| {
            median(
                profiles
                    .iter()
                    .map(|p| select(p).as_secs_f64() * 1000.0 / 16.0)
                    .collect(),
            )
        };
        println!(
            "decode_substage: prefix=256 continuation=16 pairs=3 warmup_pairs=1 threads=2 parity=exact profiled_wall_median_s={:.4} unprofiled_wall_median_s={:.4}",
            median(profiled_walls), median(unprofiled_walls),
        );
        println!(
            "decode median ms/forward: attention(norm/qkv/causal/out/ffn)={:.4}/{:.4}/{:.4}/{:.4}/{:.4} SSM(norm/in/conv/out/ffn)={:.4}/{:.4}/{:.4}/{:.4}/{:.4} logits={:.4}",
            stage_ms(|p| p.attention_stages.norm), stage_ms(|p| p.attention_stages.qkv_projection),
            stage_ms(|p| p.attention_stages.causal_attention), stage_ms(|p| p.attention_stages.output_projection),
            stage_ms(|p| p.attention_stages.ffn), stage_ms(|p| p.ssm_stages.norm),
            stage_ms(|p| p.ssm_stages.input_projection), stage_ms(|p| p.ssm_stages.convolution),
            stage_ms(|p| p.ssm_stages.output_projection), stage_ms(|p| p.ssm_stages.ffn), stage_ms(|p| p.logits),
        );
        println!(
            "decode FFN share median={:.2}%",
            median(
                profiles
                    .iter()
                    .map(|p| {
                        (p.attention_stages.ffn + p.ssm_stages.ffn).as_secs_f64()
                            / p.total_stage_time().as_secs_f64()
                            * 100.0
                    })
                    .collect()
            )
        );
    }

    /// Compare real-model token and chunked prefill across cache boundaries.
    /// Run with:
    ///   MIVI_TEST_MODEL=models/mivi-tiny-test.gguf cargo test -p mivi-model test_chunked_prefill_matches_token_path --lib --jobs 1 -- --ignored
    #[test]
    #[ignore]
    fn test_chunked_prefill_matches_token_path() {
        let model_path = std::env::var("MIVI_TEST_MODEL").unwrap();
        for prompt_len in [63usize, 64, 65] {
            for tile_tokens in [1usize, 2, 8, 64] {
                let mut token_model =
                    Model::load_with_options(std::path::Path::new(&model_path), None, None)
                        .expect("failed to load token model");
                let mut chunked_model =
                    Model::load_with_options(std::path::Path::new(&model_path), None, None)
                        .expect("failed to load chunked model");
                chunked_model
                    .set_prefill_strategy(PrefillStrategy::Chunked { tile_tokens })
                    .unwrap();
                token_model.sampler.config.temperature = 0.0;
                chunked_model.sampler.config.temperature = 0.0;

                let vocab_limit = token_model.config.vocab_size.max(2) - 1;
                let prompt_tokens = (0..prompt_len)
                    .map(|idx| (idx % vocab_limit) as u32)
                    .collect::<Vec<_>>();
                let _ = token_model
                    .generate_tokens_incremental(&prompt_tokens, 0, 0, |_, _| true)
                    .unwrap();
                let _ = chunked_model
                    .generate_tokens_incremental(&prompt_tokens, 0, 0, |_, _| true)
                    .unwrap();

                assert_eq!(token_model.current_pos(), chunked_model.current_pos());
                assert_eq!(token_model.current_pos(), prompt_len);
                for (actual, expected) in chunked_model
                    .state
                    .logits
                    .iter()
                    .zip(token_model.state.logits.iter())
                {
                    assert!((actual - expected).abs() < 1e-3);
                }
                let (token_k, token_v) = token_model.kv_cache.export_state(prompt_len).unwrap();
                let (chunked_k, chunked_v) =
                    chunked_model.kv_cache.export_state(prompt_len).unwrap();
                assert_eq!(token_k.len(), chunked_k.len());
                assert_eq!(token_v.len(), chunked_v.len());
                for (actual, expected) in chunked_k.iter().zip(token_k.iter()) {
                    assert!((actual - expected).abs() < 1e-3);
                }
                for (actual, expected) in chunked_v.iter().zip(token_v.iter()) {
                    assert!((actual - expected).abs() < 1e-3);
                }
                for (actual, expected) in chunked_model
                    .state
                    .conv_states
                    .iter()
                    .zip(token_model.state.conv_states.iter())
                {
                    assert!((actual - expected).abs() < 1e-3);
                }
                assert_eq!(
                    token_model.prefix_cache.len(),
                    chunked_model.prefix_cache.len()
                );

                let (_, token_ids) = token_model
                    .generate_tokens_incremental(&[], prompt_len, 3, |_, _| true)
                    .unwrap();
                let (_, chunked_ids) = chunked_model
                    .generate_tokens_incremental(&[], prompt_len, 3, |_, _| true)
                    .unwrap();
                assert_eq!(token_ids, chunked_ids);
            }
        }
    }

    /// Integration test with real model: verify continuation preserves the prefix cache.
    /// Suffix snapshots are intentionally not restored because token-byte matching
    /// alone cannot prove that the cached hybrid state has the same causal context.
    /// Requires GGUF model. Run with:
    ///   MIVI_TEST_MODEL=/path/to/model.gguf cargo test -p mivi-model test_incremental_continuation_preserves_prefix_cache -- --ignored
    #[test]
    #[ignore]
    fn test_incremental_continuation_preserves_prefix_cache() {
        let model_path = std::env::var("MIVI_TEST_MODEL").unwrap();
        let mut model = Model::load_with_options(std::path::Path::new(&model_path), None, None)
            .expect("failed to load model");

        let system = "You are a helpful assistant.";
        let prompt = format!("{}\nUser: Tell me a very long and detailed story.", system);

        // Step 0: generate enough tokens to fill at least one 64-token chunk.
        let _ = model.generate_streaming_incremental(&prompt, 0, 512, |_tok, _text| false);
        let step0_pos = model.current_pos();
        let cached = model.prefix_cache.len();
        println!("Step 0: {} tokens, {} chunks", step0_pos, cached);

        if step0_pos < 64 {
            println!("SKIP: EOS at {} tokens (< 64)", step0_pos);
            return;
        }
        assert!(cached > 0, "expected >=1 chunk after step 0");

        // Step 1: continuation must remain correct while retaining the cache.
        let step1_pos_before = model.current_pos();
        let step1 = format!("{}\nUser: hello", system);
        let _ =
            model.generate_streaming_incremental(&step1, step1_pos_before, 5, |_tok, _text| false);

        println!(
            "Step 1: {}->{} tokens",
            step1_pos_before,
            model.current_pos()
        );
        assert!(model.prefix_cache.len() >= cached, "cache should persist");
    }
}
