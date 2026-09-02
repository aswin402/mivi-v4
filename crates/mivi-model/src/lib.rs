//! Model loading, tensor forward pass, and sampling for mivi-v4.

pub mod config;
pub mod expert_cache;
pub mod ffn;
pub mod gguf;
pub mod grammar;
pub mod loader;
pub mod lora;
pub mod model;
pub mod pld;
pub mod sampler;
pub mod ssm;
pub mod transformer;
pub mod weights;

pub use config::{
    BlockType, GenerationConfig, ModelConfig, DEFAULT_MAX_LORA_RANK, DEFAULT_N_EXPERTS,
    DEFAULT_RMS_NORM_EPS, DEFAULT_ROPE_BASE, DEFAULT_SSM_A_VAL, RECENT_TOKENS_WINDOW,
};
pub use expert_cache::{
    ExpertHeatStat, ExpertHeatTracker, ExpertKey, ExpertPinningManager, ExpertPinningStrategy,
    DEFAULT_EXPERT_HEAT_FILE, DEFAULT_HEAT_DECAY_FACTOR,
};
pub use ffn::{ffn_swiglu_forward, linear_forward, FfnSwigluParams, LinearParams};
pub use gguf::{GgufError, GgufFile, GgufValue, TensorInfo};
pub use grammar::{JsonGrammar, JsonScope, TokenBitMask, ToolCallGrammar};
pub use loader::safe_f32_slice;
pub use lora::{ActiveAdapters, LoraAdapter, LoraWeightPair};
pub use model::{Model, ModelError, Result};
pub use pld::{
    PromptLookupProposer, ReasoningSpecRouter, SpeculativeMode, TreeDraftCandidate,
    TreePldProposer, TreeVerifier, DEFAULT_PLD_DRAFT_SIZE, DEFAULT_PLD_NGRAM_SIZE,
    MAX_TREE_DEPTH, REASONING_DRAFT_DEPTH,
};
pub use sampler::{Sampler, SamplerConfig};
pub use ssm::{ssm_forward, SsmParams};
pub use transformer::{attention_forward, AttentionParams};
pub use weights::{LayerWeights, ModelWeights, QuantizedTensor};
