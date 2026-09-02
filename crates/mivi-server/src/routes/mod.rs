//! HTTP route handlers and Axum router builder for mivi-server.

pub mod agent;
pub mod anthropic;
pub mod chat;

use crate::state::AppState;
use crate::types::MiviStatusResponse;
use axum::{
    extract::{DefaultBodyLimit, Json, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use mivi_tools::get_builtin_tool_definitions;
use std::sync::Arc;
use std::time::Duration;
use tower_http::cors::CorsLayer;
use tower_http::timeout::TimeoutLayer;

pub const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 300;

pub fn create_router(state: Arc<AppState>) -> Router {
    let auth_key = state.api_key.clone();
    let max_body = state.config.max_body_bytes;
    let public_routes = Router::new()
        .route("/", get(crate::ui::serve_embedded_ui))
        .route("/web", get(crate::ui::serve_embedded_ui))
        .route("/health", get(health_check))
        // Base API check endpoints for AI agent frameworks (e.g. baseURL = http://localhost:8080/v1)
        .route("/v1", get(v1_root))
        .route("/v1/", get(v1_root))
        .route("/v1/models", get(list_models))
        .route("/v1/models/:model_id", get(get_model_info))
        .route("/models", get(list_models))
        .route("/models/:model_id", get(get_model_info))
        .route("/v1/mivi/status", get(get_status))
        .route("/v1/mivi/tools", get(list_tools))
        // Ollama API compatibility routes
        .route("/api/tags", get(list_models_ollama))
        .route("/api/version", get(ollama_version));

    let protected_routes = Router::new()
        .route("/v1/chat/completions", post(chat::chat_completions))
        .route("/chat/completions", post(chat::chat_completions))
        .route("/v1/messages", post(anthropic::anthropic_messages_handler))
        .route("/messages", post(anthropic::anthropic_messages_handler))
        .route("/v1/mivi/agent", post(agent::run_agent_task));

    let protected_routes = if let Some(key) = auth_key {
        protected_routes.layer(axum::middleware::from_fn_with_state(
            Some(key),
            crate::auth::require_api_key,
        ))
    } else {
        protected_routes
    };

    public_routes
        .merge(protected_routes)
        .layer(axum::middleware::from_fn(
            crate::logging::mivi_log_middleware,
        ))
        .layer(TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(DEFAULT_REQUEST_TIMEOUT_SECS),
        ))
        .layer(CorsLayer::permissive())
        .layer(DefaultBodyLimit::max(max_body))
        .with_state(state)
}

async fn v1_root(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "engine": mivi_core::ENGINE_NAME,
        "model": state.model_name,
        "version": env!("CARGO_PKG_VERSION"),
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "messages": "/v1/messages",
            "models": "/v1/models",
            "agent": "/v1/mivi/agent"
        }
    }))
}

async fn health_check(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let engine_alive = !state.engine.is_closed();
    let status = if engine_alive { "ok" } else { "degraded" };
    let code = if engine_alive {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        code,
        Json(serde_json::json!({
            "status": status,
            "engine": mivi_core::ENGINE_NAME,
            "engine_alive": engine_alive,
        })),
    )
}

async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(serde_json::json!({
        "object": "list",
        "data": [
            {
                "id": state.model_name,
                "object": "model",
                "owned_by": mivi_core::ENGINE_OWNER,
                "permission": []
            }
        ]
    }))
}

async fn get_model_info(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(model_id): axum::extract::Path<String>,
) -> impl IntoResponse {
    let id = if model_id.is_empty() {
        state.model_name.clone()
    } else {
        model_id
    };
    Json(serde_json::json!({
        "id": id,
        "object": "model",
        "owned_by": mivi_core::ENGINE_OWNER,
        "permission": []
    }))
}

async fn list_models_ollama(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(serde_json::json!({
        "models": [
            {
                "name": state.model_name,
                "model": state.model_name,
                "modified_at": chrono::Utc::now().to_rfc3339(),
                "size": 0,
                "digest": "",
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "hybrid-slm",
                    "parameter_size": "1.2B",
                    "quantization_level": "Q4_K_M"
                }
            }
        ]
    }))
}

async fn ollama_version() -> impl IntoResponse {
    Json(serde_json::json!({
        "version": env!("CARGO_PKG_VERSION")
    }))
}

async fn get_status(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let uptime = state.start_time.elapsed().as_secs();

    let resp = MiviStatusResponse {
        engine: mivi_core::ENGINE_NAME.to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        model: state.model_name.clone(),
        memory_rss_mb: mivi_core::estimate_process_memory_mb(),
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
