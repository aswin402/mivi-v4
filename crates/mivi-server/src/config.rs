use serde::{Deserialize, Serialize};

use crate::model_profile::ModelProfileConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Optional explicit model protocol profile. When absent, the loaded model metadata is used.
    #[serde(default)]
    pub model_profile: Option<ModelProfileConfig>,
    /// Prompt-prefill execution strategy applied to the loaded model.
    #[serde(default)]
    pub prefill_strategy: mivi_model::PrefillStrategy,
    /// Explicit browser origins allowed to access the HTTP API. Empty means disabled.
    pub cors_allowed_origins: Vec<String>,
    pub max_body_bytes: usize,
    pub max_messages: usize,
    pub max_allowed_tokens: usize,
    pub default_max_tokens: usize,
    pub default_max_agent_steps: usize,
    pub channel_capacity: usize,
    /// Maximum number of requests allowed to hold an inference slot at once.
    pub max_concurrent_requests: usize,
    /// Maximum number of blocking tool handlers allowed to run at once.
    pub max_concurrent_tool_executions: usize,
    /// Maximum wall-clock time for one inference request, including streaming generation.
    #[serde(default = "default_request_timeout_secs")]
    pub request_timeout_secs: u64,
    /// Maximum wall-clock time before the first model output is produced.
    #[serde(default = "default_first_token_timeout_secs")]
    pub first_token_timeout_secs: u64,
    pub agent_gen_tokens: usize,
    pub max_port_attempts: u16,
}

pub const DEFAULT_MAX_BODY_BYTES: usize = 2 * 1024 * 1024; // 2MB
pub const DEFAULT_MAX_MESSAGES: usize = 128;
pub const DEFAULT_MAX_ALLOWED_TOKENS: usize = 8192;
pub const DEFAULT_MAX_TOKENS: usize = 256;
pub const DEFAULT_MAX_AGENT_STEPS: usize = 10;
pub const DEFAULT_CHANNEL_CAPACITY: usize = 64;
pub const DEFAULT_MAX_CONCURRENT_REQUESTS: usize = 1;
pub const DEFAULT_MAX_CONCURRENT_TOOL_EXECUTIONS: usize =
    mivi_tools::DEFAULT_MAX_CONCURRENT_TOOL_EXECUTIONS;
pub const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 300;
pub const DEFAULT_FIRST_TOKEN_TIMEOUT_SECS: u64 = 120;
pub const DEFAULT_AGENT_GEN_TOKENS: usize = 512;
pub const DEFAULT_MAX_PORT_ATTEMPTS: u16 = 20;

fn default_request_timeout_secs() -> u64 {
    DEFAULT_REQUEST_TIMEOUT_SECS
}

fn default_first_token_timeout_secs() -> u64 {
    DEFAULT_FIRST_TOKEN_TIMEOUT_SECS
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            model_profile: None,
            prefill_strategy: mivi_model::PrefillStrategy::Chunked { tile_tokens: 64 },
            cors_allowed_origins: Vec::new(),
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
            max_messages: DEFAULT_MAX_MESSAGES,
            max_allowed_tokens: DEFAULT_MAX_ALLOWED_TOKENS,
            default_max_tokens: DEFAULT_MAX_TOKENS,
            default_max_agent_steps: DEFAULT_MAX_AGENT_STEPS,
            channel_capacity: DEFAULT_CHANNEL_CAPACITY,
            max_concurrent_requests: DEFAULT_MAX_CONCURRENT_REQUESTS,
            max_concurrent_tool_executions: DEFAULT_MAX_CONCURRENT_TOOL_EXECUTIONS,
            request_timeout_secs: DEFAULT_REQUEST_TIMEOUT_SECS,
            first_token_timeout_secs: DEFAULT_FIRST_TOKEN_TIMEOUT_SECS,
            agent_gen_tokens: DEFAULT_AGENT_GEN_TOKENS,
            max_port_attempts: DEFAULT_MAX_PORT_ATTEMPTS,
        }
    }
}

impl ServerConfig {
    /// Normalize values that Tokio requires to be non-zero.
    pub fn normalized(mut self) -> Self {
        self.channel_capacity = self.channel_capacity.max(1);
        self.max_concurrent_requests = self.max_concurrent_requests.max(1);
        self.max_concurrent_tool_executions = self.max_concurrent_tool_executions.max(1);
        self.request_timeout_secs = self.request_timeout_secs.max(1);
        self.first_token_timeout_secs = self.first_token_timeout_secs.max(1);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::ServerConfig;

    #[test]
    fn zero_capacity_settings_are_normalized() {
        let config = ServerConfig {
            channel_capacity: 0,
            max_concurrent_requests: 0,
            max_concurrent_tool_executions: 0,
            request_timeout_secs: 0,
            first_token_timeout_secs: 0,
            ..ServerConfig::default()
        }
        .normalized();

        assert_eq!(config.channel_capacity, 1);
        assert_eq!(config.max_concurrent_requests, 1);
        assert_eq!(config.max_concurrent_tool_executions, 1);
        assert_eq!(config.request_timeout_secs, 1);
        assert_eq!(config.first_token_timeout_secs, 1);
    }

    #[test]
    fn prefill_strategy_defaults_to_chunked_agent_prefill() {
        assert_eq!(
            ServerConfig::default().prefill_strategy,
            mivi_model::PrefillStrategy::Chunked { tile_tokens: 64 }
        );
    }
}
