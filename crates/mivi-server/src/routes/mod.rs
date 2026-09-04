//! HTTP route handlers and Axum router builder for mivi-server.

pub mod agent;
pub mod anthropic;
pub mod chat;

use crate::state::AppState;
use crate::types::MiviStatusResponse;
use axum::{
    extract::{DefaultBodyLimit, Json, State},
    http::{HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use mivi_tools::get_builtin_tool_definitions;
use std::sync::Arc;
use std::time::Duration;
use tower_http::cors::{AllowOrigin, Any, CorsLayer};
use tower_http::timeout::TimeoutLayer;

pub const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 300;

fn configured_cors_layer(origins: &[String]) -> CorsLayer {
    let allowed_origins = origins
        .iter()
        .filter_map(|origin| {
            if origin == "*" {
                tracing::warn!("Ignoring wildcard CORS origin; use an explicit origin instead");
                return None;
            }
            match HeaderValue::try_from(origin.as_str()) {
                Ok(value) => Some(value),
                Err(error) => {
                    tracing::warn!(origin, %error, "Ignoring invalid CORS origin");
                    None
                }
            }
        })
        .collect::<Vec<_>>();

    if origins.is_empty() {
        CorsLayer::new()
    } else {
        CorsLayer::new()
            .allow_origin(AllowOrigin::list(allowed_origins))
            .allow_methods(Any)
            .allow_headers(Any)
    }
}

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
        // Do not expose the unauthenticated loopback API to arbitrary browser origins.
        // Browser clients can opt into CORS through a separately configured deployment
        // layer; the default server is intended for local/native clients.
        .layer(configured_cors_layer(&state.config.cors_allowed_origins))
        .layer(DefaultBodyLimit::max(max_body))
        .with_state(state)
}

async fn v1_root(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "engine": mivi_core::ENGINE_NAME,
        "model": if state.engine.has_model() {
            serde_json::Value::String(state.model_name.clone())
        } else {
            serde_json::Value::Null
        },
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
    let model_loaded = state.engine.has_model();
    let status = if !engine_alive {
        "degraded"
    } else if !model_loaded {
        "no_model"
    } else {
        "ok"
    };
    let code = if engine_alive && model_loaded {
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
            "model_loaded": model_loaded,
        })),
    )
}

async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(serde_json::json!({
        "object": "list",
        "data": if state.engine.has_model() { serde_json::json!([
            {
                "id": state.model_name,
                "object": "model",
                "owned_by": mivi_core::ENGINE_OWNER,
                "permission": []
            }
        ]) } else { serde_json::json!([]) }
    }))
}

async fn get_model_info(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(model_id): axum::extract::Path<String>,
) -> Response {
    if !state.engine.has_model() {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": { "message": "No model is loaded", "type": "model_not_found" }
            })),
        )
            .into_response();
    }
    let id = if model_id.is_empty() {
        state.model_name.clone()
    } else {
        model_id
    };
    if id != state.model_name && id != mivi_core::DEFAULT_MODEL_ID {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": { "message": "Model not found", "type": "model_not_found" }
            })),
        )
            .into_response();
    }
    Json(serde_json::json!({
        "id": id,
        "object": "model",
        "owned_by": mivi_core::ENGINE_OWNER,
        "permission": []
    }))
    .into_response()
}

async fn list_models_ollama(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(serde_json::json!({
        "models": if state.engine.has_model() { serde_json::json!([
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
                    "parameter_size": "unknown",
                    "quantization_level": "unknown"
                }
            }
        ]) } else { serde_json::json!([]) }
    }))
}

async fn ollama_version() -> impl IntoResponse {
    Json(serde_json::json!({
        "version": env!("CARGO_PKG_VERSION")
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request};
    use mivi_tools::ToolBroker;
    use std::sync::Arc;
    use tower::ServiceExt;

    #[tokio::test]
    async fn default_router_does_not_allow_cross_origin_requests() {
        let engine = crate::EngineActor::spawn(None);
        let state = Arc::new(AppState::new("test-model", ToolBroker::new(), engine, None));
        let app = create_router(state);
        let request = Request::builder()
            .method("OPTIONS")
            .uri("/v1/mivi/agent")
            .header("origin", "https://untrusted.example")
            .header("access-control-request-method", "POST")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert!(response.headers().get("access-control-allow-origin").is_none());
    }

    #[tokio::test]
    async fn no_model_is_not_reported_as_ready_or_available() {
        let engine = crate::EngineActor::spawn(None);
        let state = Arc::new(AppState::new("test-model", ToolBroker::new(), engine, None));
        let app = create_router(state);

        let health = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(health.status(), StatusCode::SERVICE_UNAVAILABLE);

        let model = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/v1/models/test-model")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(model.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn model_validation_accepts_loaded_name_and_default_alias_only() {
        assert!(crate::model_matches(None, "custom-model"));
        assert!(crate::model_matches(Some("custom-model"), "custom-model"));
        assert!(crate::model_matches(Some("mivi"), "custom-model"));
        assert!(!crate::model_matches(Some("other-model"), "custom-model"));
    }

    #[tokio::test]
    async fn configured_cors_allows_only_listed_origins() {
        let engine = crate::EngineActor::spawn(None);
        let mut config = crate::ServerConfig::default();
        config.cors_allowed_origins = vec!["https://trusted.example".to_string()];
        let state = Arc::new(AppState::with_config(
            "test-model",
            ToolBroker::new(),
            engine,
            None,
            config,
        ));
        let app = create_router(state);

        let trusted = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("OPTIONS")
                    .uri("/v1/mivi/agent")
                    .header("origin", "https://trusted.example")
                    .header("access-control-request-method", "POST")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            trusted
                .headers()
                .get("access-control-allow-origin")
                .and_then(|value| value.to_str().ok()),
            Some("https://trusted.example")
        );

        let untrusted = app
            .oneshot(
                Request::builder()
                    .method("OPTIONS")
                    .uri("/v1/mivi/agent")
                    .header("origin", "https://untrusted.example")
                    .header("access-control-request-method", "POST")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(untrusted
            .headers()
            .get("access-control-allow-origin")
            .is_none());
    }
}

async fn get_status(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let uptime = state.start_time.elapsed().as_secs();

    let resp = MiviStatusResponse {
        engine: mivi_core::ENGINE_NAME.to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        model: state.model_name.clone(),
        memory_rss_mb: mivi_core::estimate_process_memory_mb(),
        active_tools_count: get_builtin_tool_definitions().len(),
        status: if !state.engine.is_closed() && state.engine.has_model() {
            "healthy"
        } else if !state.engine.is_closed() {
            "no_model"
        } else {
            "degraded"
        }
        .to_string(),
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
