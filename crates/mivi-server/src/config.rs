use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub max_body_bytes: usize,
    pub max_messages: usize,
    pub max_allowed_tokens: usize,
    pub default_max_tokens: usize,
    pub default_max_agent_steps: usize,
    pub channel_capacity: usize,
    pub agent_gen_tokens: usize,
}

pub const DEFAULT_MAX_BODY_BYTES: usize = 2 * 1024 * 1024; // 2MB
pub const DEFAULT_MAX_MESSAGES: usize = 128;
pub const DEFAULT_MAX_ALLOWED_TOKENS: usize = 8192;
pub const DEFAULT_MAX_TOKENS: usize = 256;
pub const DEFAULT_MAX_AGENT_STEPS: usize = 10;
pub const DEFAULT_CHANNEL_CAPACITY: usize = 64;
pub const DEFAULT_AGENT_GEN_TOKENS: usize = 512;

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
            max_messages: DEFAULT_MAX_MESSAGES,
            max_allowed_tokens: DEFAULT_MAX_ALLOWED_TOKENS,
            default_max_tokens: DEFAULT_MAX_TOKENS,
            default_max_agent_steps: DEFAULT_MAX_AGENT_STEPS,
            channel_capacity: DEFAULT_CHANNEL_CAPACITY,
            agent_gen_tokens: DEFAULT_AGENT_GEN_TOKENS,
        }
    }
}
