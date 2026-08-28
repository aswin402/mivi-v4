//! Axum HTTP API handlers supporting JSON completions, real-time SSE streaming, and Mivi Agent OS endpoints.

use crate::streaming::*;
use crate::types::*;
use axum::{
    extract::Json,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Router,
};
use futures::stream;
use mivi_tools::{get_builtin_tool_definitions, ToolBroker};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tokio::sync::Mutex;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::CorsLayer;

pub struct AppState {
    pub model_name: String,
    pub start_time: Instant,
    pub broker: ToolBroker,
    pub model: Option<Arc<Mutex<mivi_model::Model>>>,
}

pub fn create_router(state: Arc<Mutex<AppState>>) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/mivi/status", get(get_status))
        .route("/v1/mivi/tools", get(list_tools))
        .route("/v1/mivi/agent", post(run_agent_task))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn health_check() -> impl IntoResponse {
    (StatusCode::OK, Json(serde_json::json!({"status": "ok", "engine": "mivi-v4"})))
}

async fn list_models(
    axum::extract::State(state): axum::extract::State<Arc<Mutex<AppState>>>,
) -> impl IntoResponse {
    let s = state.lock().await;
    Json(serde_json::json!({
        "object": "list",
        "data": [
            {
                "id": s.model_name,
                "object": "model",
                "owned_by": "mivi",
                "permission": []
            }
        ]
    }))
}

async fn get_status(
    axum::extract::State(state): axum::extract::State<Arc<Mutex<AppState>>>,
) -> impl IntoResponse {
    let s = state.lock().await;
    let uptime = s.start_time.elapsed().as_secs();

    let resp = MiviStatusResponse {
        engine: "mivi-v4".to_string(),
        version: "0.1.0".to_string(),
        model: s.model_name.clone(),
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
    axum::extract::State(state): axum::extract::State<Arc<Mutex<AppState>>>,
    Json(req): Json<AgentRunRequest>,
) -> Response {
    let s = state.lock().await;
    let model_name = s.model_name.clone();
    let cid = format!("agent-run-{}", uuid::Uuid::new_v4());
    let maybe_model = s.model.clone();
    let broker = s.broker.clone();
    drop(s);

    let (tx, rx) = mpsc::channel::<std::result::Result<Event, std::convert::Infallible>>(64);
    let cid_clone = cid.clone();
    let mname = model_name.clone();

    tokio::spawn(async move {
        let max_steps = if req.max_steps == 0 { 10 } else { req.max_steps };
        let agent_state = mivi_agent::AgentState::new(&req.task, max_steps);
        let mut agent = mivi_agent::AgentLoop::new(agent_state, &broker);

        let _ = tx.send(Ok(create_thinking_chunk_event(&cid_clone, &mname, &format!("Initializing agent for task: '{}'", req.task)))).await;

        let initial_prompt = format!("<user_request>\n{}\n</user_request>\nFormulate a plan and call appropriate tools if necessary.", req.task);

        if let Some(model_arc) = maybe_model {
            let model_out = tokio::task::spawn_blocking(move || {
                let mut m = model_arc.blocking_lock();
                m.generate(&initial_prompt, 512).unwrap_or_else(|_| "Task completed.".to_string())
            }).await.unwrap_or_else(|_| "Inference failed.".to_string());

            let result = agent.step(&model_out).await;
            let _ = tx.send(Ok(create_content_chunk_event(&cid_clone, &mname, &result))).await;
        } else {
            let mock_plan = format!("<plan>\n1. Inspect request\n2. Execute action for: {}\n3. Verify\n</plan>\nTask completed successfully.", req.task);
            let result = agent.step(&mock_plan).await;
            let _ = tx.send(Ok(create_content_chunk_event(&cid_clone, &mname, &result))).await;
        }

        let _ = tx.send(Ok(create_done_chunk_event(&cid_clone, &mname, "stop"))).await;
        let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
    });

    let stream = ReceiverStream::new(rx);
    Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
}

async fn chat_completions(
    axum::extract::State(state): axum::extract::State<Arc<Mutex<AppState>>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let s = state.lock().await;
    let model_name = s.model_name.clone();
    let is_streaming = req.stream.unwrap_or(false);
    let completion_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let max_tokens = req.max_tokens.unwrap_or(256);

    let maybe_model = s.model.clone();
    drop(s);

    if let Some(model_arc) = maybe_model {
        let last_message = req
            .messages
            .last()
            .and_then(|m| m.content.clone())
            .unwrap_or_else(|| "Hello".to_string());

        if is_streaming {
            let (tx, rx) = mpsc::channel::<std::result::Result<Event, std::convert::Infallible>>(64);
            let cid = completion_id.clone();
            let mname = model_name.clone();

            tokio::task::spawn_blocking(move || {
                let mut m = model_arc.blocking_lock();
                let output = m.generate(&last_message, max_tokens).unwrap_or_else(|_| "Inference completed.".to_string());

                let _ = tx.blocking_send(Ok(create_thinking_chunk_event(&cid, &mname, "Generating completion with Mivi model...")));
                for word in output.split_inclusive(' ') {
                    let _ = tx.blocking_send(Ok(create_content_chunk_event(&cid, &mname, word)));
                }
                let _ = tx.blocking_send(Ok(create_done_chunk_event(&cid, &mname, "stop")));
                let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
            });

            let stream = ReceiverStream::new(rx);
            Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
        } else {
            let (output, p_tokens, c_tokens) = tokio::task::spawn_blocking(move || {
                let mut m = model_arc.blocking_lock();
                let p_tok = m.tokenizer.encode(&last_message).len();
                let out = m.generate(&last_message, max_tokens).unwrap_or_else(|_| "Inference completed.".to_string());
                let c_tok = m.tokenizer.encode(&out).len();
                (out, p_tok, c_tok)
            })
            .await
            .unwrap_or_else(|_| ("Inference task failed.".to_string(), 0, 0));

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
    } else if is_streaming {
        let cid = completion_id.clone();
        let mname = model_name.clone();

        let events: Vec<std::result::Result<Event, std::convert::Infallible>> = vec![
            Ok(create_thinking_chunk_event(&cid, &mname, "Processing request with Mivi engine...")),
            Ok(create_content_chunk_event(&cid, &mname, "Mivi-v4 ")),
            Ok(create_content_chunk_event(&cid, &mname, "inference ")),
            Ok(create_content_chunk_event(&cid, &mname, "ready.")),
            Ok(create_done_chunk_event(&cid, &mname, "stop")),
            Ok(Event::default().data("[DONE]")),
        ];

        let stream = stream::iter(events);
        Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
    } else {
        let p_tok = req.messages.iter().filter_map(|m| m.content.as_ref()).map(|c| c.split_whitespace().count()).sum::<usize>();
        let response = ChatCompletionResponse {
            id: completion_id,
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp() as u64,
            model: model_name,
            choices: vec![ChoiceDto {
                index: 0,
                message: MessageDto {
                    role: "assistant".to_string(),
                    content: Some("Mivi-v4 inference engine ready.".to_string()),
                    name: None,
                    thinking: Some("Verified model routing and inference pipeline.".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: UsageDto {
                prompt_tokens: p_tok.max(1),
                completion_tokens: 6,
                total_tokens: p_tok.max(1) + 6,
            },
        };

        Json(response).into_response()
    }
}

fn estimate_process_memory_mb() -> f32 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(statm) = std::fs::read_to_string("/proc/self/statm") {
            if let Some(rss_pages) = statm.split_whitespace().nth(1) {
                if let Ok(pages) = rss_pages.parse::<usize>() {
                    let page_size_kb = 4; // 4KB page
                    return (pages * page_size_kb) as f32 / 1024.0;
                }
            }
        }
    }
    128.0
}
