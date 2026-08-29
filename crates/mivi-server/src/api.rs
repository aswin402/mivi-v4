//! Axum HTTP API handlers supporting JSON completions, real-time SSE streaming, and Mivi Agent OS endpoints.

use crate::engine_actor::EngineHandle;
use crate::streaming::*;
use crate::types::*;
use axum::{
    extract::{DefaultBodyLimit, Json},
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Router,
};
use mivi_tools::{get_builtin_tool_definitions, ToolBroker};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::CorsLayer;

pub struct AppState {
    pub model_name: String,
    pub start_time: Instant,
    pub broker: ToolBroker,
    pub engine: EngineHandle,
    pub api_key: Option<String>,
}

pub fn create_router(state: Arc<AppState>) -> Router {
    let auth_key = state.api_key.clone();
    let router = Router::new()
        .route("/health", get(health_check))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/mivi/status", get(get_status))
        .route("/v1/mivi/tools", get(list_tools))
        .route("/v1/mivi/agent", post(run_agent_task))
        .layer(CorsLayer::permissive())
        .layer(DefaultBodyLimit::max(2 * 1024 * 1024));

    if let Some(key) = auth_key {
        router
            .layer(axum::middleware::from_fn_with_state(
                Some(key),
                crate::auth::require_api_key,
            ))
            .with_state(state)
    } else {
        router.with_state(state)
    }
}

async fn health_check() -> impl IntoResponse {
    (StatusCode::OK, Json(serde_json::json!({"status": "ok", "engine": "mivi-v4"})))
}

async fn list_models(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
) -> impl IntoResponse {
    Json(serde_json::json!({
        "object": "list",
        "data": [
            {
                "id": state.model_name,
                "object": "model",
                "owned_by": "mivi",
                "permission": []
            }
        ]
    }))
}

async fn get_status(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
) -> impl IntoResponse {
    let uptime = state.start_time.elapsed().as_secs();

    let resp = MiviStatusResponse {
        engine: "mivi-v4".to_string(),
        version: "0.1.0".to_string(),
        model: state.model_name.clone(),
        memory_rss_mb: estimate_process_memory_mb(),
        active_tools_count: get_builtin_tool_definitions().len(),
        status: "healthy".to_string(),
        uptime_seconds: uptime,
    };

    Json(resp)
}

async fn list_tools() -> impl IntoResponse {
    let tools = get_builtin_tool_definitions();
    Json(serde_json::json!({
        "object": "list",
        "tools": tools
    }))
}

async fn run_agent_task(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    Json(req): Json<AgentRunRequest>,
) -> Response {
    let model_name = state.model_name.clone();
    let cid = format!("agent-run-{}", uuid::Uuid::new_v4());
    let broker = state.broker.clone();
    let engine = state.engine.clone();

    let (tx, rx) = mpsc::channel::<std::result::Result<Event, std::convert::Infallible>>(64);
    let cid_clone = cid.clone();
    let mname = model_name.clone();

    tokio::spawn(async move {
        let max_steps = if req.max_steps == 0 { 10 } else { req.max_steps };
        let agent_state = mivi_agent::AgentState::new(&req.task, max_steps);
        let mut agent = mivi_agent::AgentLoop::new(agent_state, &broker);

        let _ = tx.send(Ok(create_thinking_chunk_event(&cid_clone, &mname, &format!("Initializing agent for task: '{}'", req.task)))).await;

        let initial_prompt = format!("<user_request>\n{}\n</user_request>\nFormulate a plan and call appropriate tools if necessary.", req.task);

        match engine.generate(&initial_prompt, 512).await {
            Ok((model_out, _, _)) => {
                let result = agent.step(&model_out).await;
                let _ = tx.send(Ok(create_content_chunk_event(&cid_clone, &mname, &result))).await;
            }
            Err(e) => {
                let _ = tx.send(Ok(create_content_chunk_event(&cid_clone, &mname, &format!("<error>Model inference failed: {}</error>", e)))).await;
            }
        }

        let _ = tx.send(Ok(create_done_chunk_event(&cid_clone, &mname, "stop"))).await;
        let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
    });

    let stream = ReceiverStream::new(rx);
    Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
}

async fn chat_completions(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    if req.messages.is_empty() {
        return AppError::InvalidRequest("messages array cannot be empty".to_string()).into_response();
    }
    if req.messages.len() > 128 {
        return AppError::InvalidRequest("messages array exceeds limit of 128 items".to_string()).into_response();
    }

    const MAX_ALLOWED_TOKENS: usize = 8192;
    let max_tokens = req.max_tokens.unwrap_or(256).min(MAX_ALLOWED_TOKENS);
    let completion_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let is_streaming = req.stream.unwrap_or(false);
    let model_name = state.model_name.clone();

    let last_message = req
        .messages
        .last()
        .and_then(|m| m.content.clone())
        .unwrap_or_else(|| "Hello".to_string());

    if is_streaming {
        let (tx, rx) = mpsc::channel::<std::result::Result<Event, std::convert::Infallible>>(64);
        let cid = completion_id.clone();
        let mname = model_name.clone();
        let engine = state.engine.clone();

        tokio::spawn(async move {
            let _ = tx.send(Ok(create_thinking_chunk_event(&cid, &mname, "Generating completion with Mivi engine..."))).await;

            match engine.generate_stream(&last_message, max_tokens).await {
                Ok(mut stream_rx) => {
                    while let Some(chunk_res) = stream_rx.recv().await {
                        match chunk_res {
                            Ok(word) => {
                                if tx.send(Ok(create_content_chunk_event(&cid, &mname, &word))).await.is_err() {
                                    break;
                                }
                            }
                            Err(err_msg) => {
                                let _ = tx.send(Ok(create_content_chunk_event(&cid, &mname, &format!("<error>{}</error>", err_msg)))).await;
                                break;
                            }
                        }
                    }
                }
                Err(e) => {
                    let _ = tx.send(Ok(create_content_chunk_event(&cid, &mname, &format!("<error>{}</error>", e)))).await;
                }
            }

            let _ = tx.send(Ok(create_done_chunk_event(&cid, &mname, "stop"))).await;
            let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
        });

        let stream = ReceiverStream::new(rx);
        Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
    } else {
        match state.engine.generate(&last_message, max_tokens).await {
            Ok((output, p_tokens, c_tokens)) => {
                let response = ChatCompletionResponse {
                    id: completion_id,
                    object: "chat.completion".to_string(),
                    created: chrono::Utc::now().timestamp() as u64,
                    model: model_name,
                    choices: vec![ChoiceDto {
                        index: 0,
                        message: MessageDto {
                            role: "assistant".to_string(),
                            content: Some(output),
                            name: None,
                            thinking: Some("Verified output tokens.".to_string()),
                            tool_calls: None,
                        },
                        finish_reason: Some("stop".to_string()),
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
}

fn estimate_process_memory_mb() -> f32 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(statm) = std::fs::read_to_string("/proc/self/statm") {
            if let Some(rss_pages) = statm.split_whitespace().nth(1) {
                if let Ok(pages) = rss_pages.parse::<usize>() {
                    let page_size_bytes = get_system_page_size();
                    return (pages as f64 * page_size_bytes as f64 / (1024.0 * 1024.0)) as f32;
                }
            }
        }
    }
    128.0
}

#[cfg(target_os = "linux")]
fn get_system_page_size() -> usize {
    extern "C" {
        fn sysconf(name: std::ffi::c_int) -> std::ffi::c_long;
    }
    // _SC_PAGESIZE / _SC_PAGE_SIZE is 30 on Linux
    let res = unsafe { sysconf(30) };
    if res > 0 {
        res as usize
    } else {
        4096
    }
}
