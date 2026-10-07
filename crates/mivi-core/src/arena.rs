//! Zero-heap RunState arena.
//! Preallocates all activation, residual, logits, attention, and recurrence buffers
//! at startup, ensuring 0 heap allocations during token decoding.

/// Parameters required to configure and size the RunState arena.
#[derive(Debug, Clone)]
pub struct ArenaConfig {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub kv_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub ssm_state_dim: usize,
    pub ssm_conv_kernel: usize,
    pub max_lora_rank: usize,
    pub n_experts: usize,
}

/// RunState holds all temporary activation arrays for single-token forward pass.
#[derive(Debug)]
pub struct RunState {
    // Current token hidden representation (dim)
    pub x: Box<[f32]>,
    // Ping-pong double buffer for zero-copy layer handoff (dim)
    pub x_pong: Box<[f32]>,
    // Branch / residual buffer 1 (dim)
    pub xb: Box<[f32]>,
    // Branch / residual buffer 2 (dim)
    pub xb2: Box<[f32]>,
    // FFN intermediate buffer 1 (hidden_dim)
    pub hb: Box<[f32]>,
    // FFN intermediate buffer 2 (hidden_dim)
    pub hb2: Box<[f32]>,
    // Query vector (dim)
    pub q: Box<[f32]>,
    // Key vector (kv_dim)
    pub k: Box<[f32]>,
    // Value vector (kv_dim)
    pub v: Box<[f32]>,
    // Attention scores buffer (n_heads * max_seq_len)
    pub att: Box<[f32]>,
    // Attention output buffer (dim)
    pub attn_out: Box<[f32]>,
    // Logits over entire vocabulary (vocab_size)
    pub logits: Box<[f32]>,
    // Logits scratchpad for non-destructive temperature/top-p sampling
    pub logits_scratch: Box<[f32]>,

    // SSM state (recurrent state per SSM block)
    pub ssm_states: Box<[f32]>,
    // SSM depthwise 1D convolution state buffer: [n_layers, dim, ssm_conv_kernel]
    pub conv_states: Box<[f32]>,
    // ShortConv expanded projection buffer (3 * dim)
    pub shortconv_in: Box<[f32]>,

    // LoRA intermediate buffer
    pub lora_down: Box<[f32]>,

    /// Transient per-projection sums; never part of cached recurrent state.
    #[cfg(feature = "q4-cached-sums-experiment")]
    pub q4_activation_sums: Box<[f32]>,
    /// Explicit experimental selector, off even when the feature is compiled.
    #[cfg(feature = "q4-cached-sums-experiment")]
    pub q4_cached_sums_enabled: bool,
    /// Eligibility evidence for explicit diagnostic runs only.
    #[cfg(feature = "q4-cached-sums-experiment")]
    pub q4_cached_sums_calls: usize,
}

impl RunState {
    /// Allocates all fixed-size arrays once at engine initialization.
    pub fn new(cfg: &ArenaConfig) -> Self {
        Self {
            x: vec![0.0f32; cfg.dim].into_boxed_slice(),
            x_pong: vec![0.0f32; cfg.dim].into_boxed_slice(),
            xb: vec![0.0f32; cfg.dim].into_boxed_slice(),
            xb2: vec![0.0f32; cfg.dim].into_boxed_slice(),
            hb: vec![0.0f32; cfg.hidden_dim].into_boxed_slice(),
            hb2: vec![0.0f32; cfg.hidden_dim].into_boxed_slice(),
            q: vec![0.0f32; cfg.dim].into_boxed_slice(),
            k: vec![0.0f32; cfg.kv_dim].into_boxed_slice(),
            v: vec![0.0f32; cfg.kv_dim].into_boxed_slice(),
            att: vec![0.0f32; cfg.n_heads * cfg.max_seq_len].into_boxed_slice(),
            attn_out: vec![0.0f32; cfg.dim].into_boxed_slice(),
            logits: vec![0.0f32; cfg.vocab_size].into_boxed_slice(),
            logits_scratch: vec![0.0f32; cfg.vocab_size].into_boxed_slice(),
            ssm_states: vec![0.0f32; cfg.n_layers * cfg.ssm_state_dim].into_boxed_slice(),
            conv_states: vec![0.0f32; cfg.n_layers * cfg.dim * cfg.ssm_conv_kernel]
                .into_boxed_slice(),
            shortconv_in: vec![0.0f32; 3 * cfg.dim].into_boxed_slice(),
            lora_down: vec![0.0f32; cfg.max_lora_rank].into_boxed_slice(),
            #[cfg(feature = "q4-cached-sums-experiment")]
            q4_activation_sums: vec![0.0; cfg.dim.max(cfg.hidden_dim).div_ceil(32)]
                .into_boxed_slice(),
            #[cfg(feature = "q4-cached-sums-experiment")]
            q4_cached_sums_enabled: false,
            #[cfg(feature = "q4-cached-sums-experiment")]
            q4_cached_sums_calls: 0,
        }
    }

    /// Reset recurrent states and working buffers between independent sequences.
    pub fn reset(&mut self) {
        self.x.fill(0.0);
        self.x_pong.fill(0.0);
        self.xb.fill(0.0);
        self.xb2.fill(0.0);
        self.hb.fill(0.0);
        self.hb2.fill(0.0);
        self.q.fill(0.0);
        self.k.fill(0.0);
        self.v.fill(0.0);
        self.att.fill(0.0);
        self.attn_out.fill(0.0);
        self.logits.fill(0.0);
        self.logits_scratch.fill(0.0);
        self.ssm_states.fill(0.0);
        self.conv_states.fill(0.0);
        self.shortconv_in.fill(0.0);
        self.lora_down.fill(0.0);
        #[cfg(feature = "q4-cached-sums-experiment")]
        {
            self.q4_activation_sums.fill(0.0);
            self.q4_cached_sums_calls = 0;
        }
    }

    /// Export SSM recurrent state buffers for hybrid prefix caching and snapshotting.
    pub fn export_ssm_states(&self) -> (Vec<f32>, Vec<f32>) {
        (self.conv_states.to_vec(), self.ssm_states.to_vec())
    }

    /// Import previously saved SSM recurrent state buffers.
    pub fn import_ssm_states(&mut self, conv: &[f32], ssm: &[f32]) {
        if conv.len() == self.conv_states.len() {
            self.conv_states.copy_from_slice(conv);
        }
        if ssm.len() == self.ssm_states.len() {
            self.ssm_states.copy_from_slice(ssm);
        }
    }
}

#[cfg(all(test, feature = "q4-cached-sums-experiment"))]
mod cached_sums_tests {
    use super::*;

    #[test]
    fn cached_sums_arena_capacity_and_reset() {
        let cfg = ArenaConfig {
            dim: 256,
            hidden_dim: 513,
            n_layers: 1,
            n_heads: 1,
            n_kv_heads: 1,
            head_dim: 256,
            kv_dim: 256,
            vocab_size: 8,
            max_seq_len: 4,
            ssm_state_dim: 0,
            ssm_conv_kernel: 0,
            max_lora_rank: 0,
            n_experts: 0,
        };
        let mut state = RunState::new(&cfg);
        assert_eq!(state.q4_activation_sums.len(), 17);
        assert!(!state.q4_cached_sums_enabled);
        state.q4_activation_sums.fill(7.0);
        state.q4_cached_sums_enabled = true;
        state.q4_cached_sums_calls = 9;
        let ptr = state.q4_activation_sums.as_ptr();
        state.reset();
        assert!(state.q4_activation_sums.iter().all(|&x| x == 0.0));
        assert_eq!(state.q4_cached_sums_calls, 0);
        assert!(
            state.q4_cached_sums_enabled,
            "reset must retain explicit selector"
        );
        assert_eq!(state.q4_activation_sums.as_ptr(), ptr);
    }
}
