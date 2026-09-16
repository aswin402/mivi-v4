//! OpenAI-compatible API request and response data types + Mivi extended endpoints.

use serde::{Deserialize, Serialize};
use serde_json::Value;

fn render_tool_calls(tool_calls: Option<&[serde_json::Value]>) -> Option<String> {
    let tool_calls = tool_calls?;
    if tool_calls.is_empty() {
        return None;
    }

    let mut rendered = String::new();
    for call in tool_calls {
        let payload = call
            .get("function")
            .and_then(|function| {
                let name = function.get("name")?.as_str()?;
                let arguments = function.get("arguments")?;
                let arguments = match arguments {
                    serde_json::Value::String(raw) => {
                        serde_json::from_str(raw).unwrap_or_else(|_| arguments.clone())
                    }
                    value => value.clone(),
                };
                Some(serde_json::json!({"name": name, "arguments": arguments}))
            })
            .unwrap_or_else(|| call.clone());

        if !rendered.is_empty() {
            rendered.push('\n');
        }
        rendered.push_str("<tool_call>");
        rendered.push_str(&payload.to_string());
        rendered.push_str("</tool_call>");
    }
    Some(rendered)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<MessageDto>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default, alias = "max_completion_tokens")]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub stop: Option<serde_json::Value>,
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
}

impl ChatCompletionRequest {
    /// Normalize OpenAI function wrappers into model-neutral tool definitions.
    pub fn to_canonical_tools(&self) -> Result<Vec<mivi_protocol::ToolDefinition>, String> {
        canonical_tools_from_values(self.tools.as_deref())
    }
}

/// Normalize an optional filtered OpenAI tool list into model-neutral definitions.
pub fn canonical_tools_from_values(
    tools: Option<&[Value]>,
) -> Result<Vec<mivi_protocol::ToolDefinition>, String> {
    tools
        .unwrap_or_default()
        .iter()
        .enumerate()
        .map(|(index, tool)| normalize_openai_tool(index, tool))
        .collect()
}

fn normalize_openai_tool(
    index: usize,
    value: &Value,
) -> Result<mivi_protocol::ToolDefinition, String> {
    let tool_object = value
        .as_object()
        .ok_or_else(|| format!("tools[{index}] must be an object"))?;
    if tool_object.get("type").and_then(Value::as_str) != Some("function") {
        return Err(format!("tools[{index}].type must be 'function'"));
    }

    let function = tool_object
        .get("function")
        .and_then(Value::as_object)
        .ok_or_else(|| format!("tools[{index}].function must be an object"))?;
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| format!("tools[{index}].function.name must be a non-empty string"))?;
    let parameters = function
        .get("parameters")
        .cloned()
        .unwrap_or_else(|| Value::Object(serde_json::Map::new()));
    if !parameters.is_object() {
        return Err(format!(
            "tools[{index}].function.parameters must be a JSON object"
        ));
    }

    Ok(mivi_protocol::ToolDefinition {
        name: name.to_string(),
        description: function
            .get("description")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        parameters,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageDto {
    pub role: String,
    #[serde(default, deserialize_with = "deserialize_polymorphic_content")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl MessageDto {
    /// Convert an external API message without losing tool-call identity or null content.
    ///
    /// Model-specific rendering is intentionally deferred to a model adapter. The existing
    /// `From<&MessageDto> for mivi_tokenizer::ChatMessage` implementation remains temporarily
    /// available for the legacy formatter during the adapter migration.
    pub fn to_canonical_message(&self) -> Result<mivi_protocol::Message, String> {
        let tool_calls = self
            .tool_calls
            .as_deref()
            .unwrap_or_default()
            .iter()
            .map(parse_openai_tool_call)
            .collect::<Result<Vec<_>, _>>()?;

        Ok(mivi_protocol::Message {
            role: self.role.clone(),
            content: self.content.clone(),
            name: self.name.clone(),
            tool_call_id: self.tool_call_id.clone(),
            tool_calls,
            reasoning: self.thinking.clone(),
        })
    }
}

fn parse_openai_tool_call(value: &Value) -> Result<mivi_protocol::ToolCall, String> {
    let id = value
        .get("id")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let function = value
        .get("function")
        .and_then(Value::as_object)
        .ok_or_else(|| "tool call must contain a function object".to_string())?;
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| "tool call function.name must be a non-empty string".to_string())?;
    let raw_arguments = function
        .get("arguments")
        .ok_or_else(|| "tool call function.arguments is required".to_string())?;
    let arguments = match raw_arguments {
        Value::String(raw) => serde_json::from_str(raw)
            .map_err(|error| format!("tool call arguments must be valid JSON: {error}"))?,
        value => value.clone(),
    };

    Ok(mivi_protocol::ToolCall {
        id,
        name: name.to_string(),
        arguments,
    })
}

fn deserialize_polymorphic_content<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let opt = Option::<serde_json::Value>::deserialize(deserializer)?;
    match opt {
        Some(serde_json::Value::String(s)) => Ok(Some(s)),
        Some(serde_json::Value::Array(blocks)) => {
            let mut acc = String::new();
            for b in blocks {
                if let Some(t) = b.get("text").and_then(|v| v.as_str()) {
                    acc.push_str(t);
                }
            }
            Ok(Some(acc))
        }
        Some(serde_json::Value::Null) | None => Ok(None),
        Some(other) => Ok(Some(other.to_string())),
    }
}

impl From<&MessageDto> for mivi_tokenizer::ChatMessage {
    fn from(m: &MessageDto) -> Self {
        let role = m.role.parse().unwrap_or_else(|_| {
            tracing::warn!("Unrecognized role '{}', defaulting to User", m.role);
            mivi_tokenizer::Role::User
        });
        let tool_calls = render_tool_calls(m.tool_calls.as_deref());
        let content = match (&m.content, tool_calls) {
            (Some(content), Some(tool_calls)) => Some(format!("{content}\n{tool_calls}")),
            (None, Some(tool_calls)) => Some(tool_calls),
            (content, None) => content.clone(),
        };
        Self {
            role,
            content,
            name: m.name.clone(),
        }
    }
}

#[cfg(test)]
mod canonical_message_tests {
    use super::*;

    #[test]
    fn canonical_message_preserves_tool_call_identity_and_null_content() {
        let message = MessageDto {
            role: "assistant".to_string(),
            content: None,
            name: None,
            thinking: None,
            tool_calls: Some(vec![serde_json::json!({
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": "{\"path\":\"README.md\"}"
                }
            })]),
            tool_call_id: None,
        };

        let canonical = message.to_canonical_message().expect("valid tool call");

        assert_eq!(canonical.role, "assistant");
        assert_eq!(canonical.content, None);
        assert_eq!(canonical.tool_calls.len(), 1);
        assert_eq!(canonical.tool_calls[0].id.as_deref(), Some("call_1"));
        assert_eq!(canonical.tool_calls[0].name, "read_file");
        assert_eq!(canonical.tool_calls[0].arguments["path"], "README.md");
    }

    #[test]
    fn canonical_tools_remove_openai_wrapper_without_losing_schema() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "messages": [],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"]
                    }
                }
            }]
        }))
        .expect("valid OpenAI request");

        let tools = request
            .to_canonical_tools()
            .expect("valid tool definitions");

        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "read_file");
        assert_eq!(tools[0].description.as_deref(), Some("Read a file"));
        assert_eq!(tools[0].parameters["required"][0], "path");
    }
}

impl From<MessageDto> for mivi_tokenizer::ChatMessage {
    fn from(m: MessageDto) -> Self {
        (&m).into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_tool_choice_defaults_to_auto_and_accepts_required() {
        let automatic: AgentRunRequest = serde_json::from_value(serde_json::json!({
            "task": "Inspect the project"
        }))
        .expect("default agent tool choice should deserialize");
        assert_eq!(automatic.tool_choice, AgentToolChoice::Auto);

        let required: AgentRunRequest = serde_json::from_value(serde_json::json!({
            "task": "Calculate 2 + 2",
            "tool_choice": "required"
        }))
        .expect("required agent tool choice should deserialize");
        assert_eq!(required.tool_choice, AgentToolChoice::Required);
    }

    #[test]
    fn agent_tool_call_retries_round_trip_and_default_to_one() {
        let automatic: AgentRunRequest = serde_json::from_value(serde_json::json!({
            "task": "Calculate 2 + 2"
        }))
        .expect("default agent retry policy should deserialize");
        assert_eq!(automatic.tool_call_retries, 1);

        let configured: AgentRunRequest = serde_json::from_value(serde_json::json!({
            "task": "Calculate 2 + 2",
            "tool_call_retries": 2
        }))
        .expect("configured agent retry policy should deserialize");
        let encoded = serde_json::to_value(configured).expect("retry policy should serialize");
        assert_eq!(encoded["tool_call_retries"], 2);
    }

    #[test]
    fn agent_sampling_controls_round_trip() {
        let configured: AgentRunRequest = serde_json::from_value(serde_json::json!({
            "task": "Calculate 2 + 2",
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 50,
            "min_p": 0.05,
            "repetition_penalty": 1.1,
            "presence_penalty": 0.2,
            "frequency_penalty": 0.3,
            "seed": 42
        }))
        .expect("agent sampling controls should deserialize");
        let encoded = serde_json::to_value(configured).expect("sampling controls should serialize");
        assert!((encoded["temperature"].as_f64().unwrap() - 0.1).abs() < 1e-6);
        assert!((encoded["top_p"].as_f64().unwrap() - 0.9).abs() < 1e-6);
        assert_eq!(encoded["top_k"], 50);
        assert!((encoded["min_p"].as_f64().unwrap() - 0.05).abs() < 1e-6);
        assert!((encoded["repetition_penalty"].as_f64().unwrap() - 1.1).abs() < 1e-6);
        assert!((encoded["presence_penalty"].as_f64().unwrap() - 0.2).abs() < 1e-6);
        assert!((encoded["frequency_penalty"].as_f64().unwrap() - 0.3).abs() < 1e-6);
        assert_eq!(encoded["seed"], 42);
    }

    #[test]
    fn assistant_tool_calls_are_preserved_as_model_markup() {
        let message = MessageDto {
            role: "assistant".to_string(),
            content: None,
            name: None,
            thinking: None,
            tool_calls: Some(vec![serde_json::json!({
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": "{\"path\":\"README.md\"}"
                }
            })]),
            tool_call_id: None,
        };

        let chat_message: mivi_tokenizer::ChatMessage = (&message).into();

        let content = chat_message.content.expect("tool call content");
        assert!(content.starts_with("<tool_call>") && content.ends_with("</tool_call>"));
        let payload = &content["<tool_call>".len()..content.len() - "</tool_call>".len()];
        let payload: serde_json::Value = serde_json::from_str(payload).unwrap();
        assert_eq!(payload["name"], "read_file");
        assert_eq!(payload["arguments"]["path"], "README.md");
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

/// Controls whether an internal agent may finish with plain text or must call a tool first.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentToolChoice {
    #[default]
    Auto,
    Required,
}

/// Request for executing a full autonomous agent task loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRunRequest {
    pub task: String,
    #[serde(default = "default_max_steps")]
    pub max_steps: usize,
    #[serde(default)]
    pub tool_choice: AgentToolChoice,
    #[serde(default = "default_tool_call_retries")]
    pub tool_call_retries: usize,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub allowed_tools: Option<Vec<String>>,
    #[serde(default)]
    pub context_docs: Option<Vec<String>>,
}

fn default_max_steps() -> usize {
    crate::config::ServerConfig::default().default_max_agent_steps
}

fn default_tool_call_retries() -> usize {
    1
}

/// Runtime capabilities selected for the loaded model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCapabilityReport {
    pub profile: Option<String>,
    pub tool_codec: Option<String>,
    pub supports_tools: bool,
    pub supports_streaming: bool,
}

/// Telemetry status response for /v1/mivi/status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiviStatusResponse {
    pub engine: String,
    pub version: String,
    pub model: String,
    pub memory_rss_mb: f32,
    pub active_tools_count: usize,
    pub context_length: Option<usize>,
    pub capabilities: ModelCapabilityReport,
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
    #[error(
        "Context length exceeded: prompt uses {prompt_tokens} tokens and requested output reserves {requested_output_tokens}, but model context limit is {context_length}"
    )]
    ContextLengthExceeded {
        prompt_tokens: usize,
        requested_output_tokens: usize,
        context_length: usize,
    },
    #[error("Inference failed: {0}")]
    InferenceError(String),
    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),
    #[error("Server is busy: {0}")]
    TooManyRequests(String),
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
            AppError::ContextLengthExceeded {
                prompt_tokens,
                requested_output_tokens,
                context_length,
            } => (
                axum::http::StatusCode::BAD_REQUEST,
                "invalid_request_error",
                Some("context_length_exceeded"),
                format!(
                    "Context length exceeded: prompt uses {prompt_tokens} tokens and requested output reserves {requested_output_tokens}, but model context limit is {context_length}"
                ),
            ),
            AppError::InferenceError(m) => (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                Some("inference_error"),
                m,
            ),
            AppError::ServiceUnavailable(m) => (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                "api_error",
                Some("service_unavailable"),
                m,
            ),
            AppError::TooManyRequests(m) => (
                axum::http::StatusCode::TOO_MANY_REQUESTS,
                "api_error",
                Some("server_busy"),
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
