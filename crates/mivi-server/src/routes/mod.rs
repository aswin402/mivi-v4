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
use tower_http::cors::{AllowOrigin, CorsLayer};
use tower_http::timeout::TimeoutLayer;

pub const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 300;

pub fn create_router(state: Arc<AppState>) -> Router {
    let auth_key = state.api_key.clone();
    let max_body = state.config.max_body_bytes;
    let public_routes = Router::new()
        .route("/", get(crate::ui::serve_embedded_ui))
        .route("/web", get(crate::ui::serve_embedded_ui))
        .route("/health", get(health_check))
        .route("/v1/models", get(list_models))
        .route("/v1/mivi/status", get(get_status))
        .route("/v1/mivi/tools", get(list_tools));

    let protected_routes = Router::new()
        .route("/v1/chat/completions", post(chat::chat_completions))
        .route("/v1/messages", post(anthropic::anthropic_messages_handler))
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
        .layer(
            CorsLayer::new()
                .allow_origin(AllowOrigin::predicate(|origin, _| {
                    if let Ok(s) = origin.to_str() {
                        s.starts_with("http://localhost")
                            || s.starts_with("https://localhost")
                            || s.starts_with("http://127.0.0.1")
                            || s.starts_with("https://127.0.0.1")
                            || s.starts_with("http://[::1]")
                            || s.starts_with("https://[::1]")
                            || s.starts_with("http://0.0.0.0")
                            || s.starts_with("http://192.168.")
                            || s.starts_with("http://10.")
                            || s.starts_with("http://172.")
                    } else {
                        false
                    }
                }))
                .allow_methods([axum::http::Method::GET, axum::http::Method::POST])
                .allow_headers([
                    axum::http::header::CONTENT_TYPE,
                    axum::http::header::AUTHORIZATION,
                    axum::http::HeaderName::from_static("x-api-key"),
                    axum::http::HeaderName::from_static("anthropic-version"),
                    axum::http::HeaderName::from_static("anthropic-beta"),
                ]),
        )
        .layer(DefaultBodyLimit::max(max_body))
        .with_state(state)
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
