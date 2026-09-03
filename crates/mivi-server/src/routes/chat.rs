//! Chat completions HTTP endpoint with blocking and SSE streaming modes.

use crate::engine_actor::EngineHandle;
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
use tokio::sync::mpsc;
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

    let max_tokens = req
        .max_tokens
        .unwrap_or(state.config.default_max_tokens)
        .min(state.config.max_allowed_tokens);
    let completion_id = format!("{}{}", mivi_core::CHATCMPL_ID_PREFIX, uuid::Uuid::new_v4());
    let is_streaming = req.stream.unwrap_or(false);
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

    let tools_json = req
        .tools
        .as_ref()
        .and_then(|t| match serde_json::to_string(t) {
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
    let prompt = mivi_tokenizer::format_chatml(&chat_messages, tools_json.as_deref(), enable_thinking);

    if let Some(prompt_text) = &last_user_prompt {
        let approx_tokens = (prompt.len() / 4).max(1);
        crate::logging::print_incoming_prompt(prompt_text, Some(approx_tokens), false);
    }

    if is_streaming {
        let ctx = ChatStreamContext {
            prompt,
            max_tokens,
            temperature: req.temperature,
            top_p: req.top_p,
            completion_id,
            model_name,
            engine: state.engine.clone(),
            channel_capacity: state.config.channel_capacity,
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
            temperature: req.temperature,
            top_p: req.top_p,
            completion_id,
            model_name,
            engine: &state.engine,
        };
        handle_chat_blocking(ctx).await
    }
}

struct ChatStreamContext {
    prompt: String,
    max_tokens: usize,
    temperature: Option<f32>,
    top_p: Option<f32>,
    completion_id: String,
    model_name: String,
    engine: EngineHandle,
    channel_capacity: usize,
}

struct ChatBlockingContext<'a> {
    prompt: &'a str,
    max_tokens: usize,
    temperature: Option<f32>,
    top_p: Option<f32>,
    completion_id: String,
    model_name: String,
    engine: &'a EngineHandle,
}

pub const THINKING_INIT_MSG: &str = "Generating completion with Mivi engine...";

fn handle_chat_streaming(ctx: ChatStreamContext) -> Response {
    let (tx, rx) =
        mpsc::channel::<std::result::Result<Event, std::convert::Infallible>>(ctx.channel_capacity);
    let cid = ctx.completion_id;
    let mname = ctx.model_name;
    let engine = ctx.engine;
    let prompt = ctx.prompt;
    let max_tokens = ctx.max_tokens;
    let temperature = ctx.temperature;
    let top_p = ctx.top_p;

    tokio::spawn(async move {
        send_sse_sequence(&tx, &cid, &mname, None, || async {
            match engine
                .generate_stream_with_params(&prompt, max_tokens, temperature, top_p)
                .await
            {
                Ok(mut stream_rx) => {
                    let mut assembled = String::new();
                    while let Some(chunk_res) = stream_rx.recv().await {
                        match chunk_res {
                            Ok(word) => {
                                assembled.push_str(&word);
                                if tx
                                    .send(Ok(create_content_chunk_event(&cid, &mname, &word)))
                                    .await
                                    .is_err()
                                {
                                    break;
                                }
                            }
                            Err(err_msg) => {
                                tracing::error!("Inference stream error: {}", err_msg);
                                let _ = tx
                                    .send(Ok(create_error_chunk_event(
                                        &cid,
                                        &mname,
                                        "Internal inference error occurred.",
                                    )))
                                    .await;
                                break;
                            }
                        }
                    }
                    if !assembled.is_empty() {
                        let thinking = mivi_tools::extract_thinking(&assembled);
                        let clean_reply = mivi_tools::strip_thinking(&assembled);
                        let tool_calls = mivi_tools::extract_tool_calls(&assembled);
                        let tc_names: Vec<String> = tool_calls.into_iter().map(|tc| format!("{}(...)", tc.name)).collect();
                        crate::logging::print_completion_response_box(
                            thinking.as_deref(),
                            if tc_names.is_empty() { None } else { Some(&tc_names) },
                            Some(&clean_reply),
                        );
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to start stream: {}", e);
                    let _ = tx
                        .send(Ok(create_error_chunk_event(
                            &cid,
                            &mname,
                            "Failed to initialize streaming.",
                        )))
                        .await;
                }
            }
        })
        .await;
    });

    sse_response(rx)
}

async fn handle_chat_blocking(ctx: ChatBlockingContext<'_>) -> Response {
    match ctx
        .engine
        .generate_with_params(ctx.prompt, ctx.max_tokens, ctx.temperature, ctx.top_p)
        .await
    {
        Ok((output, p_tokens, c_tokens)) => {
            let thinking = mivi_tools::extract_thinking(&output);
            let tool_calls_extracted = mivi_tools::extract_tool_calls(&output);

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

            let content = if tool_calls.is_some() {
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
                if tc_names.is_empty() { None } else { Some(&tc_names) },
                Some(&reply_preview),
            );
            let log_meta = crate::logging::LogMetadata {
                prompt_summary: None,
                response_summary: Some(reply_preview),
                thinking_summary: thinking,
                tokens_prompt: Some(p_tokens),
                tokens_completion: Some(c_tokens),
                finish_reason: Some(finish_reason.to_string()),
                tool_calls: if tc_names.is_empty() { None } else { Some(tc_names) },
                ..Default::default()
            };
            resp.extensions_mut().insert(log_meta);
            resp
        }
        Err(e) => AppError::InferenceError(e).into_response(),
    }
}
