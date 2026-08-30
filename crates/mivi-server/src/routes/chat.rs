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
    let prompt = mivi_tokenizer::format_chatml(&chat_messages, tools_json.as_deref(), true);

    if is_streaming {
        let ctx = ChatStreamContext {
            prompt,
            max_tokens,
            completion_id,
            model_name,
            engine: state.engine.clone(),
            channel_capacity: state.config.channel_capacity,
        };
        handle_chat_streaming(ctx)
    } else {
        handle_chat_blocking(
            &prompt,
            max_tokens,
            completion_id,
            model_name,
            &state.engine,
        )
        .await
    }
}

struct ChatStreamContext {
    prompt: String,
    max_tokens: usize,
    completion_id: String,
    model_name: String,
    engine: EngineHandle,
    channel_capacity: usize,
}

pub const THINKING_INIT_MSG: &str = "Generating completion with Mivi engine...";
pub const VERIFIED_OUTPUT_MSG: &str = "Verified output tokens.";

fn handle_chat_streaming(ctx: ChatStreamContext) -> Response {
    let (tx, rx) =
        mpsc::channel::<std::result::Result<Event, std::convert::Infallible>>(ctx.channel_capacity);
    let cid = ctx.completion_id;
    let mname = ctx.model_name;
    let engine = ctx.engine;
    let prompt = ctx.prompt;
    let max_tokens = ctx.max_tokens;

    tokio::spawn(async move {
        send_sse_sequence(&tx, &cid, &mname, Some(THINKING_INIT_MSG), || async {
            match engine.generate_stream(&prompt, max_tokens).await {
                Ok(mut stream_rx) => {
                    while let Some(chunk_res) = stream_rx.recv().await {
                        match chunk_res {
                            Ok(word) => {
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

async fn handle_chat_blocking(
    prompt: &str,
    max_tokens: usize,
    completion_id: String,
    model_name: String,
    engine: &EngineHandle,
) -> Response {
    match engine.generate(prompt, max_tokens).await {
        Ok((output, p_tokens, c_tokens)) => {
            let response = ChatCompletionResponse {
                id: completion_id,
                object: OPENAI_COMPLETION_OBJECT.to_string(),
                created: chrono::Utc::now().timestamp() as u64,
                model: model_name,
                choices: vec![ChoiceDto {
                    index: 0,
                    message: MessageDto {
                        role: ROLE_ASSISTANT.to_string(),
                        content: Some(output),
                        name: None,
                        thinking: Some(VERIFIED_OUTPUT_MSG.to_string()),
                        tool_calls: None,
                    },
                    finish_reason: Some(FINISH_REASON_STOP.to_string()),
                }],
                usage: UsageDto {
                    prompt_tokens: p_tokens,
                    completion_tokens: c_tokens,
                    total_tokens: p_tokens + c_tokens,
                },
            };
            Json(response).into_response()
        }
        Err(e) => AppError::InferenceError(e).into_response(),
    }
}
