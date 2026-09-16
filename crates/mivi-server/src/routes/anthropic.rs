use crate::engine_actor::GenerationCancellation;
use crate::generation::{
    validate_sampling_parameters, validate_stop_sequences, GenerationOptions, ResponseMode,
};
use crate::model_profile::{ModelProfile, ModelProfileConfig};
use crate::state::AppState;
use axum::{
    extract::State,
    http::StatusCode,
    response::{sse::Event, IntoResponse, Response, Sse},
    Json,
};
use futures::StreamExt;
use mivi_protocol::{Message, ToolCall, ToolDefinition};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;
use std::time::Instant;

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

struct CancellationOnDrop(GenerationCancellation);

impl CancellationOnDrop {
    fn cancel(&self) {
        self.0.cancel();
    }
}

impl Drop for CancellationOnDrop {
    fn drop(&mut self) {
        self.cancel();
    }
}

struct AnthropicChunkStreamState {
    receiver: tokio::sync::mpsc::Receiver<Result<String, String>>,
    cancellation: CancellationOnDrop,
    deadline: Pin<Box<tokio::time::Sleep>>,
    first_token_deadline: Pin<Box<tokio::time::Sleep>>,
    first_token_received: bool,
    failed: bool,
    stream_failed: Arc<AtomicBool>,
    assembled_text: Arc<std::sync::Mutex<String>>,
    tools_enabled: bool,
}

fn anthropic_chunk_stream(
    receiver: tokio::sync::mpsc::Receiver<Result<String, String>>,
    cancellation: GenerationCancellation,
    request_timeout: Duration,
    first_token_timeout: Duration,
    tools_enabled: bool,
    assembled_text: Arc<std::sync::Mutex<String>>,
    stream_failed: Arc<AtomicBool>,
) -> impl futures::Stream<Item = Result<Event, Infallible>> {
    futures::stream::unfold(
        AnthropicChunkStreamState {
            receiver,
            cancellation: CancellationOnDrop(cancellation),
            deadline: Box::pin(tokio::time::sleep(request_timeout)),
            first_token_deadline: Box::pin(tokio::time::sleep(first_token_timeout)),
            first_token_received: false,
            failed: false,
            stream_failed,
            assembled_text,
            tools_enabled,
        },
        |mut state| async move {
            loop {
                if state.failed {
                    return None;
                }

                tokio::select! {
                    _ = &mut state.first_token_deadline, if !state.first_token_received => {
                        state.cancellation.cancel();
                        state.failed = true;
                        state.stream_failed.store(true, Ordering::Release);
                        let data = serde_json::json!({
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": "Model did not produce a first token before the configured deadline."
                            }
                        }).to_string();
                        return Some((Ok(Event::default().event("error").data(data)), state));
                    }
                    _ = &mut state.deadline => {
                        state.cancellation.cancel();
                        state.failed = true;
                        state.stream_failed.store(true, Ordering::Release);
                        let data = serde_json::json!({
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": "Inference request timed out."
                            }
                        }).to_string();
                        return Some((Ok(Event::default().event("error").data(data)), state));
                    }
                    chunk = state.receiver.recv() => {
                        match chunk {
                            Some(Ok(chunk)) => {
                                if !chunk.is_empty() {
                                    state.first_token_received = true;
                                }
                                if let Ok(mut guard) = state.assembled_text.lock() {
                                    guard.push_str(&chunk);
                                }
                                if state.tools_enabled {
                                    continue;
                                }
                                let data = serde_json::json!({
                                    "type": "content_block_delta",
                                    "index": 0,
                                    "delta": { "type": "text_delta", "text": chunk }
                                }).to_string();
                                return Some((
                                    Ok(Event::default().event("content_block_delta").data(data)),
                                    state,
                                ));
                            }
                            Some(Err(error)) => {
                                state.failed = true;
                                state.stream_failed.store(true, Ordering::Release);
                                let data = serde_json::json!({
                                    "type": "error",
                                    "error": { "type": "api_error", "message": error }
                                }).to_string();
                                return Some((Ok(Event::default().event("error").data(data)), state));
                            }
                            None => return None,
                        }
                    }
                }
            }
        },
    )
}

fn anthropic_value_to_text(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::Null => None,
        serde_json::Value::String(text) => Some(text.clone()),
        serde_json::Value::Array(blocks) => {
            let mut text = String::new();
            for block in blocks {
                if let Some(block_text) = block.get("text").and_then(|value| value.as_str()) {
                    text.push_str(block_text);
                } else if let Some(block_content) = block.get("content") {
                    if let Some(block_text) = anthropic_value_to_text(block_content) {
                        text.push_str(&block_text);
                    }
                }
            }
            Some(text)
        }
        value => Some(value.to_string()),
    }
}

fn canonical_anthropic_messages(req: &AnthropicRequest) -> Vec<Message> {
    let mut messages = Vec::new();
    if let Some(content) = req.system.as_ref().and_then(anthropic_value_to_text) {
        if !content.is_empty() {
            messages.push(Message {
                role: "system".to_string(),
                content: Some(content),
                name: None,
                tool_call_id: None,
                tool_calls: Vec::new(),
                reasoning: None,
            });
        }
    }

    for message in &req.messages {
        let role = if message.role.eq_ignore_ascii_case("assistant") {
            "assistant"
        } else if message.role.eq_ignore_ascii_case("tool") {
            "tool"
        } else {
            "user"
        };

        let Some(blocks) = message.content.as_array() else {
            messages.push(Message {
                role: role.to_string(),
                content: anthropic_value_to_text(&message.content),
                name: None,
                tool_call_id: None,
                tool_calls: Vec::new(),
                reasoning: None,
            });
            continue;
        };

        let mut text = String::new();
        let mut reasoning = None;
        let mut tool_calls = Vec::new();
        let mut tool_results = Vec::new();

        for block in blocks {
            match block.get("type").and_then(|value| value.as_str()) {
                Some("text") => {
                    if let Some(value) = block.get("text").and_then(|value| value.as_str()) {
                        text.push_str(value);
                    }
                }
                Some("thinking") => {
                    reasoning = block
                        .get("thinking")
                        .and_then(|value| value.as_str())
                        .map(ToOwned::to_owned);
                }
                Some("tool_use") => {
                    let Some(name) = block.get("name").and_then(|value| value.as_str()) else {
                        continue;
                    };
                    if name.is_empty() {
                        continue;
                    }
                    tool_calls.push(ToolCall {
                        id: block
                            .get("id")
                            .and_then(|value| value.as_str())
                            .map(ToOwned::to_owned),
                        name: name.to_string(),
                        arguments: block
                            .get("input")
                            .cloned()
                            .unwrap_or_else(|| serde_json::json!({})),
                    });
                }
                Some("tool_result") => {
                    tool_results.push(Message {
                        role: "tool".to_string(),
                        content: block.get("content").and_then(anthropic_value_to_text),
                        name: None,
                        tool_call_id: block
                            .get("tool_use_id")
                            .and_then(|value| value.as_str())
                            .map(ToOwned::to_owned),
                        tool_calls: Vec::new(),
                        reasoning: None,
                    });
                }
                _ => {
                    if let Some(value) = block.get("text").and_then(|value| value.as_str()) {
                        text.push_str(value);
                    }
                }
            }
        }

        if role == "assistant" {
            if !text.is_empty() || !tool_calls.is_empty() || reasoning.is_some() {
                messages.push(Message {
                    role: role.to_string(),
                    content: if text.is_empty() { None } else { Some(text) },
                    name: None,
                    tool_call_id: None,
                    tool_calls,
                    reasoning,
                });
            }
        } else {
            if !text.is_empty() {
                messages.push(Message {
                    role: role.to_string(),
                    content: Some(text),
                    name: None,
                    tool_call_id: None,
                    tool_calls: Vec::new(),
                    reasoning,
                });
            }
            messages.extend(tool_results);
        }
    }

    messages
}

fn canonical_anthropic_tools(req: &AnthropicRequest) -> Vec<ToolDefinition> {
    req.tools
        .as_deref()
        .unwrap_or_default()
        .iter()
        .map(|tool| ToolDefinition {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: tool.input_schema.clone(),
        })
        .collect()
}

/// Convert an Anthropic request into the legacy ChatML profile for compatibility callers.
pub fn convert_anthropic_to_chatml(req: &AnthropicRequest) -> String {
    let profile = ModelProfile::from_config(&ModelProfileConfig::Legacy)
        .expect("legacy model profile must always be valid");
    profile
        .render_prompt(
            &canonical_anthropic_messages(req),
            &canonical_anthropic_tools(req),
            false,
        )
        .unwrap_or_default()
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
    if let Some(tools) = req.tools.as_deref() {
        for (index, tool) in tools.iter().enumerate() {
            if tool.name.trim().is_empty() {
                return anthropic_invalid_request(format!(
                    "tools[{index}].name must be a non-empty string"
                ));
            }
            if !tool.input_schema.is_object() {
                return anthropic_invalid_request(format!(
                    "tools[{index}].input_schema must be a JSON object"
                ));
            }
        }
    }

    let profile = match ModelProfile::resolve(
        state.engine.model_metadata(),
        state.config.model_profile.as_ref(),
    ) {
        Ok(profile) => profile,
        Err(message) => return anthropic_invalid_request(message),
    };
    let canonical_messages = canonical_anthropic_messages(&req);
    if !profile.supports_tools()
        && (req.tools.as_ref().is_some_and(|items| !items.is_empty())
            || canonical_messages.iter().any(|message| {
                !message.tool_calls.is_empty() || message.role.eq_ignore_ascii_case("tool")
            }))
    {
        return anthropic_invalid_request(
            "selected model profile does not support tool calls".to_string(),
        );
    }
    let prompt =
        match profile.render_prompt(&canonical_messages, &canonical_anthropic_tools(&req), false) {
            Ok(prompt) => prompt,
            Err(message) => return anthropic_invalid_request(message),
        };
    let max_tokens = bounded_max_tokens(req.max_tokens, state.config.max_allowed_tokens);
    let options = GenerationOptions {
        temperature: req.temperature,
        top_p: req.top_p,
        stop_tokens: req.stop_sequences.clone(),
        response_mode: ResponseMode::Text,
        ..GenerationOptions::default()
    };
    let tools_enabled = req.tools.as_ref().is_some_and(|tools| !tools.is_empty());
    let tool_definitions = req.tools.clone().unwrap_or_default();
    let slot_wait_started = Instant::now();
    let inference_permit = match state.try_acquire_inference_slot() {
        Ok(permit) => permit,
        Err(_) => {
            state.metrics.record_inference_rejected();
            return anthropic_overloaded_response();
        }
    };
    state
        .metrics
        .record_inference_accepted(slot_wait_started.elapsed());
    let admitted_prompt_tokens =
        match crate::routes::admit_model_context(&state.engine, &prompt, max_tokens).await {
            Ok(prompt_tokens) => prompt_tokens,
            Err(error) => return anthropic_invalid_request(error.to_string()),
        };
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
                    item.get("text")
                        .and_then(|t| t.as_str())
                        .map(ToString::to_string)
                } else {
                    None
                }
            }),
            _ => None,
        });

    if let Some(prompt_text) = &last_user_prompt {
        crate::logging::print_incoming_prompt(prompt_text, admitted_prompt_tokens, false);
    }

    if req.stream {
        // SSE streaming response
        // Encode before starting generation so the engine actor cannot block on
        // the bounded stream buffer before it processes this usage request.
        let input_tokens = match admitted_prompt_tokens {
            Some(prompt_tokens) => prompt_tokens,
            None => state.engine.encode(&prompt).await.len(),
        };
        let metrics = state.metrics.clone();
        let generation_started = Instant::now();
        let (stream_rx, cancellation) = match state
            .engine
            .generate_stream_with_options_cancelable(&prompt, max_tokens, options.clone())
            .await
        {
            Ok(stream) => stream,
            Err(e) => {
                metrics.record_generation(generation_started.elapsed());
                state.metrics.record_inference_error();
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
        let stream_failed = Arc::new(AtomicBool::new(false));
        let stream = anthropic_chunk_stream(
            stream_rx,
            cancellation,
            Duration::from_secs(state.config.request_timeout_secs),
            Duration::from_secs(state.config.first_token_timeout_secs),
            tools_enabled,
            assembled_text.clone(),
            stream_failed.clone(),
        );

        // 3. Close the text block, optionally emit structured tool-use blocks, and finish.
        let token_engine = state.engine.clone();
        let prompt_log_clone = last_user_prompt.clone();
        let text_final = assembled_text.clone();
        let validation_tools = tool_definitions.clone();
        let output_profile = profile.clone();
        let stream_failed_final = stream_failed.clone();
        let post_generation_stream = futures::stream::once(async move {
            let _inference_permit = inference_permit;
            if stream_failed_final.load(Ordering::Acquire) {
                metrics.record_generation(generation_started.elapsed());
                metrics.record_tokens(input_tokens, 0);
                metrics.record_inference_error();
                return vec![Ok::<Event, Infallible>(
                    Event::default().event("content_block_stop").data(
                        serde_json::json!({ "type": "content_block_stop", "index": 0 }).to_string(),
                    ),
                )];
            }
            let output = text_final
                .lock()
                .map(|guard| guard.clone())
                .unwrap_or_default();
            let out_tokens = token_engine.encode(&output).await.len();
            metrics.record_generation(generation_started.elapsed());
            metrics.record_tokens(input_tokens, out_tokens);
            let thinking = mivi_tools::extract_thinking(&output);
            let clean = if tools_enabled {
                output_profile
                    .tool_codec()
                    .strip(&mivi_tools::strip_thinking(&output))
            } else {
                mivi_tools::strip_thinking(&output)
            };
            let (parsed_tools, parse_error) = if tools_enabled {
                match output_profile.tool_codec().extract(&output) {
                    Ok(tool_calls) => (tool_calls, None),
                    Err(error) => (Vec::new(), Some(error)),
                }
            } else {
                (Vec::new(), None)
            };
            let validation_error = parse_error.or_else(|| {
                if tools_enabled {
                    validate_anthropic_tool_calls(&parsed_tools, &validation_tools).err()
                } else {
                    None
                }
            });
            let has_tools = validation_error.is_none() && !parsed_tools.is_empty();
            let mut events = Vec::new();

            if let Some(error) = validation_error {
                metrics.record_inference_error();
                tracing::error!(%error, "Model generated invalid Anthropic tool call");
                events.push(Ok::<Event, Infallible>(
                    Event::default().event("error").data(
                        serde_json::json!({
                            "type": "error",
                            "error": { "type": "api_error", "message": error }
                        })
                        .to_string(),
                    ),
                ));
            }

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

            if has_tools {
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

        let full_stream = futures::stream::iter(vec![Ok(msg_start_event), Ok(block_start_event)])
            .chain(stream)
            .chain(post_generation_stream.flat_map(futures::stream::iter))
            .chain(futures::stream::iter(vec![Ok(msg_stop_event)]));

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
        let _inference_permit = inference_permit;
        let generation_started = Instant::now();
        let generation_result = state
            .engine
            .generate_with_options(&prompt, max_tokens, options)
            .await;
        state
            .metrics
            .record_generation(generation_started.elapsed());
        match generation_result {
            Ok((output_text, prompt_tokens, completion_tokens)) => {
                state
                    .metrics
                    .record_tokens(prompt_tokens, completion_tokens);
                let parsed_tools = if tools_enabled {
                    match profile.tool_codec().extract(&output_text) {
                        Ok(tool_calls) => tool_calls,
                        Err(error) => {
                            state.metrics.record_inference_error();
                            return anthropic_inference_error(error);
                        }
                    }
                } else {
                    Vec::new()
                };
                if let Err(error) = validate_anthropic_tool_calls(&parsed_tools, &tool_definitions)
                {
                    state.metrics.record_inference_error();
                    return anthropic_inference_error(error);
                }
                let has_tools = !parsed_tools.is_empty();
                let clean_text = if tools_enabled {
                    profile
                        .tool_codec()
                        .strip(&mivi_tools::strip_thinking(&output_text))
                } else {
                    output_text.clone()
                };
                let thinking = mivi_tools::extract_thinking(&output_text);
                let content_blocks = if has_tools {
                    let mut blocks = Vec::new();
                    if !clean_text.is_empty() {
                        blocks.push(AnthropicContentBlock::Text {
                            text: clean_text.clone(),
                        });
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
                let reply_preview = if clean_text.is_empty() {
                    output_text.clone()
                } else {
                    clean_text
                };
                let tc_names: Vec<String> = parsed_tools
                    .iter()
                    .map(|t| format!("{}(...)", t.name))
                    .collect();
                crate::logging::print_completion_response_box(
                    thinking.as_deref(),
                    if tc_names.is_empty() {
                        None
                    } else {
                        Some(&tc_names)
                    },
                    Some(&reply_preview),
                );
                let log_meta = crate::logging::LogMetadata {
                    prompt_summary: None,
                    response_summary: Some(reply_preview),
                    thinking_summary: thinking,
                    tokens_prompt: Some(prompt_tokens),
                    tokens_completion: Some(completion_tokens),
                    finish_reason: stop_reason,
                    tool_calls: if tc_names.is_empty() {
                        None
                    } else {
                        Some(tc_names)
                    },
                    ..Default::default()
                };
                resp_obj.extensions_mut().insert(log_meta);
                resp_obj
            }
            Err(e) => {
                state.metrics.record_inference_error();
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "type": "error",
                        "error": { "type": "api_error", "message": e }
                    })),
                )
                    .into_response()
            }
        }
    }
}

fn validate_anthropic_tool_calls(
    calls: &[mivi_tools::ToolCall],
    tools: &[AnthropicTool],
) -> Result<(), String> {
    for call in calls {
        if !call.arguments.is_object() {
            return Err(format!(
                "Arguments for tool '{}' must be a JSON object",
                call.name
            ));
        }
        let Some(tool) = tools.iter().find(|tool| tool.name == call.name) else {
            return Err(format!(
                "Model generated a call to undeclared tool '{}'",
                call.name
            ));
        };
        mivi_tools::validate_tool_arguments(&tool.input_schema, &call.arguments)
            .map_err(|error| format!("Invalid arguments for tool '{}': {error}", call.name))?;
    }
    Ok(())
}

fn anthropic_inference_error(message: String) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({
            "type": "error",
            "error": { "type": "api_error", "message": message }
        })),
    )
        .into_response()
}

fn anthropic_overloaded_response() -> Response {
    (
        StatusCode::TOO_MANY_REQUESTS,
        Json(serde_json::json!({
            "type": "error",
            "error": {
                "type": "api_error",
                "message": "The maximum number of concurrent inference requests is active"
            }
        })),
    )
        .into_response()
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
