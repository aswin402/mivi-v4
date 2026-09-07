//! Chat completions HTTP endpoint with blocking and SSE streaming modes.

use crate::engine_actor::EngineHandle;
use crate::generation::{
    filter_tools_for_choice, parse_response_mode, parse_stop_sequences, parse_tool_choice,
    validate_additional_sampling_parameters, validate_openai_tool_definitions,
    validate_sampling_parameters, validate_tool_calls_against_tools, GenerationOptions,
    ResponseMode, ToolChoice,
};
use crate::state::AppState;
use crate::streaming::*;
use crate::types::*;
use axum::{
    extract::{Json, State},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{mpsc, OwnedSemaphorePermit};
use tokio_stream::wrappers::ReceiverStream;

#[inline]
pub(crate) fn sse_response(
    rx: mpsc::Receiver<std::result::Result<Event, std::convert::Infallible>>,
) -> Response {
    let stream = ReceiverStream::new(rx);
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    if req.messages.is_empty() {
        return AppError::InvalidRequest("messages array cannot be empty".to_string())
            .into_response();
    }
    if req.messages.len() > state.config.max_messages {
        return AppError::InvalidRequest(format!(
            "messages array exceeds limit of {} items",
            state.config.max_messages
        ))
        .into_response();
    }
    if !state.engine.has_model() {
        return AppError::ServiceUnavailable("No model is loaded".to_string()).into_response();
    }
    if !crate::model_matches(req.model.as_deref(), &state.model_name) {
        return AppError::InvalidRequest(format!(
            "Unknown model '{}'; loaded model is '{}'",
            req.model.as_deref().unwrap_or_default(),
            state.model_name
        ))
        .into_response();
    }

    if let Err(message) = validate_sampling_parameters(
        req.temperature,
        req.top_p,
        req.presence_penalty,
        req.frequency_penalty,
    ) {
        return AppError::InvalidRequest(message).into_response();
    }
    if let Err(message) =
        validate_additional_sampling_parameters(req.top_k, req.min_p, req.repetition_penalty)
    {
        return AppError::InvalidRequest(message).into_response();
    }
    let response_mode = match parse_response_mode(req.response_format.as_ref()) {
        Ok(mode) => mode,
        Err(message) => return AppError::InvalidRequest(message).into_response(),
    };
    let is_streaming = req.stream.unwrap_or(false);
    if is_streaming && response_mode == ResponseMode::JsonObject {
        return AppError::InvalidRequest(
            "json_object response format is not supported for streaming".to_string(),
        )
        .into_response();
    }
    if let Err(message) = validate_openai_tool_definitions(req.tools.as_deref()) {
        return AppError::InvalidRequest(message).into_response();
    }
    let tool_choice = match parse_tool_choice(req.tools.as_deref(), req.tool_choice.as_ref()) {
        Ok(choice) => choice,
        Err(message) => return AppError::InvalidRequest(message).into_response(),
    };
    let tools = filter_tools_for_choice(req.tools.clone(), &tool_choice);
    let tool_calls_enabled = tool_choice.allows_tool_calls(tools.as_deref());
    let stop_tokens = match parse_stop_sequences(req.stop.as_ref()) {
        Ok(stops) => stops,
        Err(message) => return AppError::InvalidRequest(message).into_response(),
    };
    let options = GenerationOptions {
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        min_p: req.min_p,
        repetition_penalty: req.repetition_penalty,
        presence_penalty: req.presence_penalty,
        frequency_penalty: req.frequency_penalty,
        seed: req.seed,
        stop_tokens,
        response_mode,
    };

    let max_tokens = req
        .max_tokens
        .unwrap_or(state.config.default_max_tokens)
        .min(state.config.max_allowed_tokens);
    let slot_wait_started = Instant::now();
    let inference_permit = match state.try_acquire_inference_slot() {
        Ok(permit) => permit,
        Err(_) => {
            state.metrics.record_inference_rejected();
            return AppError::TooManyRequests(
                "The maximum number of concurrent inference requests is active".to_string(),
            )
            .into_response();
        }
    };
    state
        .metrics
        .record_inference_accepted(slot_wait_started.elapsed());
    let completion_id = format!("{}{}", mivi_core::CHATCMPL_ID_PREFIX, uuid::Uuid::new_v4());
    let last_user_prompt = req
        .messages
        .iter()
        .rev()
        .find(|m| m.role.eq_ignore_ascii_case("user") || m.role.eq_ignore_ascii_case("developer"))
        .and_then(|m| m.content.as_deref())
        .map(|s| crate::logging::summarize_prompt(s, 140));

    let model_name = state.model_name.clone();

    let chat_messages: Vec<mivi_tokenizer::ChatMessage> =
        req.messages.iter().map(Into::into).collect();

    let tools_json = tools.as_ref().and_then(|t| match serde_json::to_string(t) {
        Ok(json) => Some(json),
        Err(e) => {
            tracing::error!("Failed to serialize tools list: {}", e);
            None
        }
    });

    let enable_thinking = req
        .reasoning_effort
        .as_deref()
        .map(|r| r != "none")
        .unwrap_or(false);
    let prompt =
        mivi_tokenizer::format_chatml(&chat_messages, tools_json.as_deref(), enable_thinking);

    if let Some(prompt_text) = &last_user_prompt {
        let approx_tokens = (prompt.len() / 4).max(1);
        crate::logging::print_incoming_prompt(prompt_text, Some(approx_tokens), false);
    }

    if is_streaming {
        let ctx = ChatStreamContext {
            prompt,
            max_tokens,
            options,
            tool_choice,
            tool_calls_enabled,
            tool_definitions: tools.clone(),
            inference_permit,
            completion_id,
            model_name,
            engine: state.engine.clone(),
            channel_capacity: state.config.channel_capacity,
            metrics: state.metrics.clone(),
        };
        let mut resp = handle_chat_streaming(ctx);
        let log_meta = crate::logging::LogMetadata {
            prompt_summary: last_user_prompt,
            is_streaming: true,
            ..Default::default()
        };
        resp.extensions_mut().insert(log_meta);
        resp
    } else {
        let ctx = ChatBlockingContext {
            prompt: &prompt,
            max_tokens,
            options,
            tool_choice,
            tool_calls_enabled,
            tool_definitions: tools.clone(),
            inference_permit,
            completion_id,
            model_name,
            engine: &state.engine,
            metrics: state.metrics.clone(),
        };
        handle_chat_blocking(ctx).await
    }
}

struct ChatStreamContext {
    prompt: String,
    max_tokens: usize,
    options: GenerationOptions,
    tool_choice: ToolChoice,
    tool_calls_enabled: bool,
    tool_definitions: Option<Vec<serde_json::Value>>,
    inference_permit: OwnedSemaphorePermit,
    completion_id: String,
    model_name: String,
    engine: EngineHandle,
    channel_capacity: usize,
    metrics: Arc<crate::state::ServerMetrics>,
}

struct ChatBlockingContext<'a> {
    prompt: &'a str,
    max_tokens: usize,
    options: GenerationOptions,
    tool_choice: ToolChoice,
    tool_calls_enabled: bool,
    tool_definitions: Option<Vec<serde_json::Value>>,
    inference_permit: OwnedSemaphorePermit,
    completion_id: String,
    model_name: String,
    engine: &'a EngineHandle,
    metrics: Arc<crate::state::ServerMetrics>,
}

pub const THINKING_INIT_MSG: &str = "Generating completion with Mivi engine...";

fn handle_chat_streaming(ctx: ChatStreamContext) -> Response {
    let (tx, rx) =
        mpsc::channel::<std::result::Result<Event, std::convert::Infallible>>(ctx.channel_capacity);
    let cid = ctx.completion_id;
    let mname = ctx.model_name;
    let engine = ctx.engine;
    let metrics = ctx.metrics;
    let prompt = ctx.prompt;
    let max_tokens = ctx.max_tokens;
    let options = ctx.options;
    let tool_choice = ctx.tool_choice;
    let tool_calls_enabled = ctx.tool_calls_enabled;
    let tool_definitions = ctx.tool_definitions;
    let inference_permit = ctx.inference_permit;

    tokio::spawn(async move {
        let _inference_permit = inference_permit;
        send_sse_sequence_with_finish(&tx, &cid, &mname, None, || async {
            let generation_started = Instant::now();
            match engine
                .generate_stream_with_options(&prompt, max_tokens, options)
                .await
            {
                Ok(mut stream_rx) => {
                    let mut assembled = String::new();
                    let mut stream_failed = false;
                    let mut client_disconnected = false;
                    let mut keepalive = tokio::time::interval(std::time::Duration::from_secs(2));
                    keepalive.tick().await;

                    loop {
                        tokio::select! {
                            chunk_res = stream_rx.recv() => {
                                match chunk_res {
                                    Some(Ok(word)) => {
                                        assembled.push_str(&word);
                                        if !tool_calls_enabled
                                            && tx
                                                .send(Ok(create_content_chunk_event(
                                                    &cid, &mname, &word,
                                                )))
                                                .await
                                                .is_err()
                                        {
                                            client_disconnected = true;
                                            break;
                                        }
                                    }
                                    Some(Err(err_msg)) => {
                                        stream_failed = true;
                                        tracing::error!("Inference stream error: {}", err_msg);
                                        let _ = tx
                                            .send(Ok(create_error_chunk_event(
                                                &cid,
                                                &mname,
                                                &format!("Inference error: {err_msg}"),
                                            )))
                                            .await;
                                        break;
                                    }
                                    None => break,
                                }
                            }
                            _ = keepalive.tick() => {
                                // Emit SSE comment heartbeat to keep client socket active during CPU prefill
                                if tx.send(Ok(create_keepalive_event())).await.is_err() {
                                    client_disconnected = true;
                                    break;
                                }
                            }
                        }
                    }
                    if client_disconnected {
                        drop(stream_rx);
                        metrics.record_generation(generation_started.elapsed());
                        return "error";
                    }
                    metrics.record_generation(generation_started.elapsed());
                    if stream_failed {
                        metrics.record_inference_error();
                        return "error";
                    }
                    if !assembled.is_empty() {
                        let thinking = mivi_tools::extract_thinking(&assembled);
                        let clean_reply = mivi_tools::strip_thinking(&assembled);
                        let tool_calls = extract_tool_calls_for_choice(&assembled, &tool_choice);
                        if let Err(error) = validate_tool_calls_against_tools(
                            &tool_calls,
                            tool_definitions.as_deref().unwrap_or_default(),
                        ) {
                            tracing::error!(%error, "Model generated invalid tool call");
                            metrics.record_inference_error();
                            let _ = tx
                                .send(Ok(create_error_chunk_event(&cid, &mname, &error)))
                                .await;
                            return "error";
                        }
                        let tc_names: Vec<String> = tool_calls
                            .into_iter()
                            .map(|tc| format!("{}(...)", tc.name))
                            .collect();
                        if tool_calls_enabled {
                            if !tc_names.is_empty() {
                                let values = tool_call_values(&assembled, &tool_choice);
                                let _ = tx
                                    .send(Ok(create_tool_calls_chunk_event(&cid, &mname, &values)))
                                    .await;
                            } else {
                                let clean_reply = mivi_tools::strip_tool_calls(&clean_reply);
                                if !clean_reply.is_empty() {
                                    let _ = tx
                                        .send(Ok(create_content_chunk_event(
                                            &cid,
                                            &mname,
                                            &clean_reply,
                                        )))
                                        .await;
                                }
                            }
                        }
                        crate::logging::print_completion_response_box(
                            thinking.as_deref(),
                            if tc_names.is_empty() {
                                None
                            } else {
                                Some(&tc_names)
                            },
                            Some(&clean_reply),
                        );
                        return if tool_calls_enabled && !tc_names.is_empty() {
                            "tool_calls"
                        } else {
                            "stop"
                        };
                    }
                }
                Err(e) => {
                    metrics.record_generation(generation_started.elapsed());
                    metrics.record_inference_error();
                    tracing::error!("Failed to start stream: {}", e);
                    let _ = tx
                        .send(Ok(create_error_chunk_event(
                            &cid,
                            &mname,
                            "Failed to initialize streaming.",
                        )))
                        .await;
                    return "error";
                }
            }
            "stop"
        })
        .await;
    });

    sse_response(rx)
}

async fn handle_chat_blocking(ctx: ChatBlockingContext<'_>) -> Response {
    let _inference_permit = ctx.inference_permit;
    let generation_started = Instant::now();
    let generation_result = ctx
        .engine
        .generate_with_options(ctx.prompt, ctx.max_tokens, ctx.options.clone())
        .await;
    ctx.metrics.record_generation(generation_started.elapsed());
    match generation_result {
        Ok((output, p_tokens, c_tokens)) => {
            ctx.metrics.record_tokens(p_tokens, c_tokens);
            let thinking = mivi_tools::extract_thinking(&output);
            let tool_calls_extracted = extract_tool_calls_for_choice(&output, &ctx.tool_choice);
            if let Err(error) = validate_tool_calls_against_tools(
                &tool_calls_extracted,
                ctx.tool_definitions.as_deref().unwrap_or_default(),
            ) {
                ctx.metrics.record_inference_error();
                return AppError::InferenceError(error).into_response();
            }

            let (tool_calls, finish_reason) = if !tool_calls_extracted.is_empty() {
                let tc_vals: Vec<serde_json::Value> = tool_calls_extracted
                    .iter()
                    .enumerate()
                    .map(|(i, tc)| {
                        serde_json::json!({
                            "id": format!("call_{}_{}", i, uuid::Uuid::new_v4().simple()),
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments.to_string(),
                            }
                        })
                    })
                    .collect();
                (Some(tc_vals), "tool_calls")
            } else if c_tokens >= ctx.max_tokens {
                (None, "length")
            } else {
                (None, "stop")
            };

            let content = if tool_calls.is_some() || !ctx.tool_calls_enabled {
                let cleaned = mivi_tools::strip_tool_calls(&output);
                if cleaned.is_empty() {
                    None
                } else {
                    Some(cleaned)
                }
            } else {
                Some(output.clone())
            };

            let response = ChatCompletionResponse {
                id: ctx.completion_id,
                object: OPENAI_COMPLETION_OBJECT.to_string(),
                created: chrono::Utc::now().timestamp() as u64,
                model: ctx.model_name,
                choices: vec![ChoiceDto {
                    index: 0,
                    message: MessageDto {
                        role: ROLE_ASSISTANT.to_string(),
                        content: content.clone(),
                        name: None,
                        thinking: thinking.clone(),
                        tool_calls,
                    },
                    finish_reason: Some(finish_reason.to_string()),
                }],
                usage: UsageDto {
                    prompt_tokens: p_tokens,
                    completion_tokens: c_tokens,
                    total_tokens: p_tokens + c_tokens,
                },
            };

            let mut resp = Json(response).into_response();
            let reply_preview = mivi_tools::strip_thinking(&output);
            let tc_names: Vec<String> = tool_calls_extracted
                .into_iter()
                .map(|tc| format!("{}(...)", tc.name))
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
                tokens_prompt: Some(p_tokens),
                tokens_completion: Some(c_tokens),
                finish_reason: Some(finish_reason.to_string()),
                tool_calls: if tc_names.is_empty() {
                    None
                } else {
                    Some(tc_names)
                },
                ..Default::default()
            };
            resp.extensions_mut().insert(log_meta);
            resp
        }
        Err(e) => {
            ctx.metrics.record_inference_error();
            AppError::InferenceError(e).into_response()
        }
    }
}

fn extract_tool_calls_for_choice(output: &str, choice: &ToolChoice) -> Vec<mivi_tools::ToolCall> {
    if matches!(choice, ToolChoice::Disabled) {
        return Vec::new();
    }
    mivi_tools::extract_tool_calls(output)
        .into_iter()
        .filter(|call| match choice {
            ToolChoice::Named(name) => {
                call.name == name.as_str() || call.name == mivi_tools::PARSE_ERROR_TOOL_NAME
            }
            ToolChoice::Auto => true,
            ToolChoice::Disabled => false,
        })
        .collect()
}

fn tool_call_values(output: &str, choice: &ToolChoice) -> Vec<serde_json::Value> {
    extract_tool_calls_for_choice(output, choice)
        .into_iter()
        .enumerate()
        .map(|(index, call)| {
            serde_json::json!({
                "index": index,
                "id": format!("call_{}", uuid::Uuid::new_v4().simple()),
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.arguments.to_string(),
                }
            })
        })
        .collect()
}
