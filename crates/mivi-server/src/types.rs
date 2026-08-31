//! OpenAI-compatible API request and response data types + Mivi extended endpoints.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<MessageDto>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageDto {
    pub role: String,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
}

impl From<&MessageDto> for mivi_tokenizer::ChatMessage {
    fn from(m: &MessageDto) -> Self {
        let role = m.role.parse().unwrap_or_else(|_| {
            tracing::warn!("Unrecognized role '{}', defaulting to User", m.role);
            mivi_tokenizer::Role::User
        });
        Self {
            role,
            content: m.content.clone(),
            name: m.name.clone(),
        }
    }
}

impl From<MessageDto> for mivi_tokenizer::ChatMessage {
    fn from(m: MessageDto) -> Self {
        let role = m.role.parse().unwrap_or(mivi_tokenizer::Role::User);
        Self {
            role,
            content: m.content,
            name: m.name,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChoiceDto>,
    pub usage: UsageDto,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChoiceDto {
    pub index: usize,
    pub message: MessageDto,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageDto {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Request for executing a full autonomous agent task loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRunRequest {
    pub task: String,
    #[serde(default = "default_max_steps")]
    pub max_steps: usize,
    #[serde(default)]
    pub allowed_tools: Option<Vec<String>>,
    #[serde(default)]
    pub context_docs: Option<Vec<String>>,
}

fn default_max_steps() -> usize {
    crate::config::ServerConfig::default().default_max_agent_steps
}

/// Telemetry status response for /v1/mivi/status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiviStatusResponse {
    pub engine: String,
    pub version: String,
    pub model: String,
    pub memory_rss_mb: f32,
    pub active_tools_count: usize,
    pub status: String,
    pub uptime_seconds: u64,
}

/// Standard OpenAI-compatible error response wrapper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiErrorResponse {
    pub error: OpenAiErrorDetail,
}

/// Standard OpenAI-compatible error details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiErrorDetail {
    pub message: String,
    pub r#type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("Unauthorized: {0}")]
    Unauthorized(String),
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
    #[error("Inference failed: {0}")]
    InferenceError(String),
    #[error("Internal server error: {0}")]
    Internal(String),
}

impl axum::response::IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        let (status, err_type, code, msg) = match self {
            AppError::Unauthorized(m) => (
                axum::http::StatusCode::UNAUTHORIZED,
                "invalid_request_error",
                Some("invalid_api_key"),
                m,
            ),
            AppError::InvalidRequest(m) => (
                axum::http::StatusCode::BAD_REQUEST,
                "invalid_request_error",
                Some("invalid_request"),
                m,
            ),
            AppError::InferenceError(m) => (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                Some("inference_error"),
                m,
            ),
            AppError::Internal(m) => (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                Some("internal_server_error"),
                m,
            ),
        };

        let body = axum::Json(OpenAiErrorResponse {
            error: OpenAiErrorDetail {
                message: msg,
                r#type: err_type.to_string(),
                param: None,
                code: code.map(ToString::to_string),
            },
        });

        (status, body).into_response()
    }
}
