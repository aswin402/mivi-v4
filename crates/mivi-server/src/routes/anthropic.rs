use crate::generation::{
    validate_sampling_parameters, validate_stop_sequences, GenerationOptions, ResponseMode,
};
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
    #[serde(default)]
    pub system: Option<serde_json::Value>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
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
    let mut system_text = match &req.system {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Array(blocks)) => {
            let mut acc = String::new();
            for b in blocks {
                if let Some(t) = b.get("text").and_then(|v| v.as_str()) {
                    acc.push_str(t);
                }
            }
            acc
        }
        _ => String::new(),
    };
    if let Some(ref tools) = req.tools {
        if !tools.is_empty() {
            if !system_text.is_empty() {
                system_text.push_str("\n\n");
            }
            system_text.push_str("Available tools:\n");
            for tool in tools {
                let schema_str = serde_json::to_string(&tool.input_schema).unwrap_or_else(|_| "{}".to_string());
                system_text.push_str(&format!(
                    "- {}: {} | parameters: {}\n",
                    tool.name,
                    tool.description.as_deref().unwrap_or(""),
                    schema_str
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
pub const DEFAULT_ANTHROPIC_MAX_TOKENS: usize = 2048;

fn bounded_max_tokens(requested: Option<usize>, max_allowed: usize) -> usize {
    requested
        .unwrap_or(DEFAULT_ANTHROPIC_MAX_TOKENS)
        .min(max_allowed)
}

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
    if !state.engine.has_model() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "type": "error",
                "error": { "type": "api_error", "message": "No model is loaded" }
            })),
        )
            .into_response();
    }
    if !crate::model_matches(Some(req.model.as_str()), &state.model_name) {
        return anthropic_invalid_request(format!(
            "Unknown model '{}'; loaded model is '{}'",
            req.model, state.model_name
        ));
    }

    if let Err(message) = validate_sampling_parameters(req.temperature, req.top_p, None, None) {
        return anthropic_invalid_request(message);
    }
    if let Some(stop_sequences) = req.stop_sequences.as_deref() {
        if let Err(message) = validate_stop_sequences(stop_sequences) {
            return anthropic_invalid_request(message);
        }
    }

    let prompt = convert_anthropic_to_chatml(&req);
    let max_tokens = bounded_max_tokens(req.max_tokens, state.config.max_allowed_tokens);
    let options = GenerationOptions {
        temperature: req.temperature,
        top_p: req.top_p,
        stop_tokens: req.stop_sequences.clone(),
        response_mode: ResponseMode::Text,
        ..GenerationOptions::default()
    };
    let tools_enabled = req.tools.as_ref().is_some_and(|tools| !tools.is_empty());
    let model_name = state.model_name.clone();
    let message_id = format!("msg_{}", uuid::Uuid::new_v4().simple());

    let last_user_prompt = req
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .and_then(|m| match &m.content {
            serde_json::Value::String(s) => Some(s.clone()),
            serde_json::Value::Array(arr) => arr.iter().find_map(|item| {
                if item.get("type").and_then(|t| t.as_str()) == Some("text") {
                    item.get("text").and_then(|t| t.as_str()).map(ToString::to_string)
                } else {
                    None
                }
            }),
            _ => None,
        });

    if let Some(prompt_text) = &last_user_prompt {
        let approx_tokens = (prompt.len() / 4).max(1);
        crate::logging::print_incoming_prompt(prompt_text, Some(approx_tokens), false);
    }

    if req.stream {
        // SSE streaming response
        let stream_rx = match state
            .engine
            .generate_stream_with_options(&prompt, max_tokens, options.clone())
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
        let input_tokens = state.engine.encode(&prompt).await.len().max(1);

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
                    "usage": { "input_tokens": input_tokens, "output_tokens": 0 }
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

        // 2. content_block_delta stream with dynamic token counting
        let assembled_text = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let text_accum = assembled_text.clone();

        let stream = ReceiverStream::new(stream_rx).filter_map(move |chunk_res| {
            let chunk = match chunk_res {
                Ok(chunk) => chunk,
                Err(error) => {
                    let data = serde_json::json!({
                        "type": "error",
                        "error": { "type": "api_error", "message": error }
                    })
                    .to_string();
                    return futures::future::ready(Some(Ok::<Event, Infallible>(
                        Event::default().event("error").data(data),
                    )));
                }
            };
            if let Ok(mut guard) = text_accum.lock() {
                guard.push_str(&chunk);
            }
            futures::future::ready(if tools_enabled {
                None
            } else {
                let data = serde_json::json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "text_delta",
                        "text": chunk
                    }
                })
                .to_string();
                Some(Ok::<Event, Infallible>(
                    Event::default().event("content_block_delta").data(data),
                ))
            })
        });

        // 3. Close the text block, optionally emit structured tool-use blocks, and finish.
        let token_engine = state.engine.clone();
        let prompt_log_clone = last_user_prompt.clone();
        let text_final = assembled_text.clone();
        let post_generation_stream = futures::stream::once(async move {
            let output = text_final
                .lock()
                .map(|guard| guard.clone())
                .unwrap_or_default();
            let out_tokens = token_engine.encode(&output).await.len().max(1);
            let thinking = mivi_tools::extract_thinking(&output);
            let clean = if tools_enabled {
                mivi_tools::strip_tool_calls(&mivi_tools::strip_thinking(&output))
            } else {
                mivi_tools::strip_thinking(&output)
            };
            let parsed_tools = if tools_enabled {
                mivi_tools::extract_tool_calls(&output)
            } else {
                Vec::new()
            };
            let has_tools = !parsed_tools.is_empty();
            let mut events = Vec::new();

            if tools_enabled && !clean.is_empty() {
                let data = serde_json::json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": { "type": "text_delta", "text": clean }
                })
                .to_string();
                events.push(Ok::<Event, Infallible>(
                    Event::default().event("content_block_delta").data(data),
                ));
            }
            events.push(Ok(Event::default().event("content_block_stop").data(
                serde_json::json!({ "type": "content_block_stop", "index": 0 }).to_string(),
            )));

            if tools_enabled {
                for (index, tool) in parsed_tools.iter().enumerate() {
                    let tool_index = index + 1;
                    let tool_id = format!("toolu_{}", uuid::Uuid::new_v4().simple());
                    events.push(Ok(Event::default().event("content_block_start").data(
                        serde_json::json!({
                            "type": "content_block_start",
                            "index": tool_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": tool_id,
                                "name": tool.name,
                                "input": {}
                            }
                        })
                        .to_string(),
                    )));
                    events.push(Ok(Event::default().event("content_block_delta").data(
                        serde_json::json!({
                            "type": "content_block_delta",
                            "index": tool_index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": tool.arguments.to_string()
                            }
                        })
                        .to_string(),
                    )));
                    events.push(Ok(Event::default().event("content_block_stop").data(
                        serde_json::json!({ "type": "content_block_stop", "index": tool_index })
                            .to_string(),
                    )));
                }
            }

            if !output.is_empty() {
                crate::logging::print_interaction_box(
                    prompt_log_clone.as_deref(),
                    thinking.as_deref(),
                    None,
                    Some(&clean),
                    false,
                );
            }
            events.push(Ok(Event::default().event("message_delta").data(
                serde_json::json!({
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": if has_tools { "tool_use" } else { "end_turn" },
                        "stop_sequence": null
                    },
                    "usage": { "output_tokens": out_tokens }
                })
                .to_string(),
            )));
            events
        });

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
        .chain(post_generation_stream.flat_map(futures::stream::iter))
        .chain(futures::stream::iter(vec![
            Ok(msg_stop_event),
        ]));

        let mut resp = Sse::new(full_stream)
            .keep_alive(axum::response::sse::KeepAlive::default())
            .into_response();
        let log_meta = crate::logging::LogMetadata {
            prompt_summary: last_user_prompt,
            is_streaming: true,
            ..Default::default()
        };
        resp.extensions_mut().insert(log_meta);
        resp
    } else {
        // Non-streaming JSON response
        match state
            .engine
            .generate_with_options(&prompt, max_tokens, options)
            .await
        {
            Ok((output_text, prompt_tokens, completion_tokens)) => {
                let parsed_tools = if tools_enabled {
                    mivi_tools::extract_tool_calls(&output_text)
                } else {
                    Vec::new()
                };
                let has_tools = !parsed_tools.is_empty();
                let clean_text = if tools_enabled {
                    mivi_tools::strip_tool_calls(&output_text)
                } else {
                    output_text.clone()
                };
                let thinking = mivi_tools::extract_thinking(&output_text);
                let content_blocks = if has_tools {
                    let mut blocks = Vec::new();
                    if !clean_text.is_empty() {
                        blocks.push(AnthropicContentBlock::Text { text: clean_text.clone() });
                    }
                    for tool in &parsed_tools {
                        blocks.push(AnthropicContentBlock::ToolUse {
                            id: format!("toolu_{}", uuid::Uuid::new_v4().simple()),
                            name: tool.name.clone(),
                            input: tool.arguments.clone(),
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
                    stop_reason: stop_reason.clone(),
                    usage: AnthropicUsage {
                        input_tokens: prompt_tokens,
                        output_tokens: completion_tokens,
                    },
                };

                let mut resp_obj = Json(resp).into_response();
                let reply_preview = if clean_text.is_empty() { output_text.clone() } else { clean_text };
                let tc_names: Vec<String> = parsed_tools.iter().map(|t| format!("{}(...)", t.name)).collect();
                crate::logging::print_completion_response_box(
                    thinking.as_deref(),
                    if tc_names.is_empty() { None } else { Some(&tc_names) },
                    Some(&reply_preview),
                );
                let log_meta = crate::logging::LogMetadata {
                    prompt_summary: None,
                    response_summary: Some(reply_preview),
                    thinking_summary: thinking,
                    tokens_prompt: Some(prompt_tokens),
                    tokens_completion: Some(completion_tokens),
                    finish_reason: stop_reason,
                    tool_calls: if tc_names.is_empty() { None } else { Some(tc_names) },
                    ..Default::default()
                };
                resp_obj.extensions_mut().insert(log_meta);
                resp_obj
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

fn anthropic_invalid_request(message: String) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({
            "type": "error",
            "error": { "type": "invalid_request_error", "message": message }
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anthropic_max_tokens_is_capped_by_server_limit() {
        assert_eq!(bounded_max_tokens(Some(100), 32), 32);
        assert_eq!(bounded_max_tokens(None, 32), 32);
        assert_eq!(bounded_max_tokens(Some(16), 32), 16);
    }

    #[test]
    fn test_convert_anthropic_to_chatml() {
        let req = AnthropicRequest {
            model: "mivi-v4".to_string(),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: serde_json::json!("Hello Mivi"),
            }],
            system: Some(serde_json::json!("You are a helpful assistant.")),
            max_tokens: Some(128),
            temperature: Some(0.7),
            top_p: None,
            stop_sequences: None,
            stream: false,
            tools: None,
        };

        let chatml = convert_anthropic_to_chatml(&req);
        assert!(chatml.contains("<|im_start|>system\nYou are a helpful assistant.<|im_end|>"));
        assert!(chatml.contains("<|im_start|>user\nHello Mivi<|im_end|>"));

        // Test polymorphic system block array
        let mut req_blocks = req.clone();
        req_blocks.system = Some(serde_json::json!([
            {"type": "text", "text": "System block text."}
        ]));
        let chatml_blocks = convert_anthropic_to_chatml(&req_blocks);
        assert!(chatml_blocks.contains("<|im_start|>system\nSystem block text.<|im_end|>"));
        assert!(chatml.ends_with("<|im_start|>assistant\n"));
    }
}
