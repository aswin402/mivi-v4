use crate::state::AppState;
use axum::{
    extract::State,
    http::StatusCode,
    response::{sse::Event, IntoResponse, Response, Sse},
    Json,
};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

/// Anthropic Message input.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: serde_json::Value,
}

/// Anthropic Tool definition.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnthropicTool {
    pub name: String,
    pub description: Option<String>,
    #[serde(default)]
    pub input_schema: serde_json::Value,
}

/// Anthropic Messages Request body (/v1/messages).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnthropicRequest {
    #[serde(default)]
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    pub system: Option<String>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: bool,
    pub tools: Option<Vec<AnthropicTool>>,
}

/// Anthropic Content Block.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

/// Anthropic Usage statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

/// Anthropic Messages Response body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub msg_type: String,
    pub role: String,
    pub content: Vec<AnthropicContentBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub usage: AnthropicUsage,
}

/// Convert an Anthropic request into ChatML prompt string.
pub fn convert_anthropic_to_chatml(req: &AnthropicRequest) -> String {
    let mut prompt = String::new();

    // 1. System prompt
    let mut system_text = req.system.clone().unwrap_or_default();
    if let Some(ref tools) = req.tools {
        if !tools.is_empty() {
            if !system_text.is_empty() {
                system_text.push_str("\n\n");
            }
            system_text.push_str("Available tools:\n");
            for tool in tools {
                system_text.push_str(&format!(
                    "- {}: {}\n",
                    tool.name,
                    tool.description.as_deref().unwrap_or("")
                ));
            }
            system_text.push_str("\nTo invoke a tool, output: <tool_call>{\"name\": \"tool_name\", \"arguments\": {...}}</tool_call>");
        }
    }

    if !system_text.is_empty() {
        prompt.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", system_text));
    }

    // 2. Messages
    for msg in &req.messages {
        let content_str = match &msg.content {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Array(blocks) => {
                let mut acc = String::new();
                for block in blocks {
                    let b_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
                    match b_type {
                        "text" => {
                            if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                                acc.push_str(text);
                            }
                        }
                        "tool_use" => {
                            let name = block.get("name").and_then(|n| n.as_str()).unwrap_or("");
                            let input = block.get("input").cloned().unwrap_or(serde_json::json!({}));
                            acc.push_str(&format!(
                                "<tool_call>{{\"name\":\"{}\",\"arguments\":{}}}</tool_call>",
                                name, input
                            ));
                        }
                        "tool_result" => {
                            let content_str = block.get("content")
                                .map(|c| if let Some(s) = c.as_str() { s.to_string() } else { c.to_string() })
                                .unwrap_or_default();
                            acc.push_str(&format!("<tool_result>{}</tool_result>", content_str));
                        }
                        _ => {
                            if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                                acc.push_str(text);
                            }
                        }
                    }
                }
                acc
            }
            other => other.to_string(),
        };

        prompt.push_str(&format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            msg.role, content_str
        ));
    }

    // 3. Assistant generation suffix
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

/// POST /v1/messages route handler.
pub async fn anthropic_messages_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AnthropicRequest>,
) -> Response {
    if req.messages.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "type": "error",
                "error": { "type": "invalid_request_error", "message": "messages array cannot be empty" }
            })),
        )
            .into_response();
    }

    let prompt = convert_anthropic_to_chatml(&req);
    let max_tokens = req.max_tokens.unwrap_or(2048);
    let temperature = req.temperature;
    let top_p = req.top_p;
    let model_name = if req.model.is_empty() {
        state.model_name.clone()
    } else {
        req.model.clone()
    };
    let message_id = format!("msg_{}", uuid::Uuid::new_v4().simple());

    if req.stream {
        // SSE streaming response
        let stream_rx = match state
            .engine
            .generate_stream_with_params(&prompt, max_tokens, temperature, top_p)
            .await
        {
            Ok(rx) => rx,
            Err(e) => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(serde_json::json!({
                        "type": "error",
                        "error": { "type": "api_error", "message": format!("Engine error: {}", e) }
                    })),
                )
                    .into_response();
            }
        };

        let mid = message_id.clone();
        let mname = model_name.clone();

        // 1. message_start and content_block_start
        let msg_start_event = Event::default().event("message_start").data(
            serde_json::json!({
                "type": "message_start",
                "message": {
                    "id": mid,
                    "type": "message",
                    "role": "assistant",
                    "model": mname,
                    "content": [],
                    "stop_reason": null,
                    "usage": { "input_tokens": 15, "output_tokens": 0 }
                }
            })
            .to_string(),
        );

        let block_start_event = Event::default().event("content_block_start").data(
            serde_json::json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": { "type": "text", "text": "" }
            })
            .to_string(),
        );

        // 2. content_block_delta stream
        let stream = ReceiverStream::new(stream_rx).map(move |chunk_res| {
            let chunk = chunk_res.unwrap_or_default();
            let data = serde_json::json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "text_delta",
                    "text": chunk
                }
            });
            Ok::<Event, Infallible>(Event::default().event("content_block_delta").data(data.to_string()))
        });

        // 3. content_block_stop, message_delta, and message_stop
        let block_stop_event = Event::default().event("content_block_stop").data(
            serde_json::json!({
                "type": "content_block_stop",
                "index": 0
            })
            .to_string(),
        );

        let msg_delta_event = Event::default().event("message_delta").data(
            serde_json::json!({
                "type": "message_delta",
                "delta": { "stop_reason": "end_turn", "stop_sequence": null },
                "usage": { "output_tokens": 10 }
            })
            .to_string(),
        );

        let msg_stop_event = Event::default().event("message_stop").data(
            serde_json::json!({
                "type": "message_stop"
            })
            .to_string(),
        );

        let full_stream = futures::stream::iter(vec![
            Ok(msg_start_event),
            Ok(block_start_event),
        ])
        .chain(stream)
        .chain(futures::stream::iter(vec![
            Ok(block_stop_event),
            Ok(msg_delta_event),
            Ok(msg_stop_event),
        ]));

        Sse::new(full_stream).into_response()
    } else {
        // Non-streaming JSON response
        match state
            .engine
            .generate_with_params(&prompt, max_tokens, temperature, top_p)
            .await
        {
            Ok((output_text, prompt_tokens, completion_tokens)) => {
                let parsed_tools = mivi_tools::extract_tool_calls(&output_text);
                let has_tools = !parsed_tools.is_empty();
                let content_blocks = if has_tools {
                    let mut blocks = Vec::new();
                    for tool in parsed_tools {
                        blocks.push(AnthropicContentBlock::ToolUse {
                            id: format!("toolu_{}", uuid::Uuid::new_v4().simple()),
                            name: tool.name,
                            input: tool.arguments,
                        });
                    }
                    blocks
                } else {
                    vec![AnthropicContentBlock::Text {
                        text: output_text.clone(),
                    }]
                };

                let stop_reason = if has_tools {
                    Some("tool_use".to_string())
                } else {
                    Some("end_turn".to_string())
                };

                let resp = AnthropicResponse {
                    id: message_id,
                    msg_type: "message".to_string(),
                    role: "assistant".to_string(),
                    content: content_blocks,
                    model: model_name,
                    stop_reason,
                    usage: AnthropicUsage {
                        input_tokens: prompt_tokens,
                        output_tokens: completion_tokens,
                    },
                };

                Json(resp).into_response()
            }
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "type": "error",
                    "error": { "type": "api_error", "message": e }
                })),
            )
                .into_response(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_anthropic_to_chatml() {
        let req = AnthropicRequest {
            model: "mivi-v4".to_string(),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: serde_json::json!("Hello Mivi"),
            }],
            system: Some("You are a helpful assistant.".to_string()),
            max_tokens: Some(128),
            temperature: Some(0.7),
            top_p: None,
            stream: false,
            tools: None,
        };

        let chatml = convert_anthropic_to_chatml(&req);
        assert!(chatml.contains("<|im_start|>system\nYou are a helpful assistant.<|im_end|>"));
        assert!(chatml.contains("<|im_start|>user\nHello Mivi<|im_end|>"));
        assert!(chatml.ends_with("<|im_start|>assistant\n"));
    }
}
