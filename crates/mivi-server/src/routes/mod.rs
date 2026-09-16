//! HTTP route handlers and Axum router builder for mivi-server.

pub mod agent;
pub mod anthropic;
pub mod chat;

use crate::engine_actor::EngineHandle;
use crate::model_profile::ModelProfile;
use crate::state::AppState;
use crate::types::{AppError, MiviStatusResponse, ModelCapabilityReport};
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

pub use crate::config::DEFAULT_REQUEST_TIMEOUT_SECS;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ContextBudgetError {
    pub(crate) prompt_tokens: usize,
    pub(crate) requested_output_tokens: usize,
    pub(crate) context_length: usize,
}

impl std::fmt::Display for ContextBudgetError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "Context length exceeded: prompt uses {} tokens and requested output reserves {}, but model context limit is {}",
            self.prompt_tokens, self.requested_output_tokens, self.context_length
        )
    }
}

impl ContextBudgetError {
    pub(crate) fn into_openai_error(self) -> AppError {
        AppError::ContextLengthExceeded {
            prompt_tokens: self.prompt_tokens,
            requested_output_tokens: self.requested_output_tokens,
            context_length: self.context_length,
        }
    }
}

pub(crate) fn validate_context_budget(
    prompt_tokens: usize,
    requested_output_tokens: usize,
    context_length: usize,
) -> Result<(), ContextBudgetError> {
    let required_tokens =
        prompt_tokens
            .checked_add(requested_output_tokens)
            .ok_or(ContextBudgetError {
                prompt_tokens,
                requested_output_tokens,
                context_length,
            })?;
    if required_tokens > context_length {
        return Err(ContextBudgetError {
            prompt_tokens,
            requested_output_tokens,
            context_length,
        });
    }
    Ok(())
}

pub(crate) async fn admit_model_context(
    engine: &EngineHandle,
    prompt: &str,
    requested_output_tokens: usize,
) -> Result<Option<usize>, ContextBudgetError> {
    let Some(context_length) = engine
        .model_metadata()
        .and_then(|metadata| metadata.context_length)
    else {
        return Ok(None);
    };

    let prompt_tokens = engine.encode(prompt).await.len();
    validate_context_budget(prompt_tokens, requested_output_tokens, context_length)?;
    Ok(Some(prompt_tokens))
}

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
        .route("/health", get(health_check));

    let protected_routes = Router::new()
        // Base API check endpoints for AI agent frameworks (e.g. baseURL = http://localhost:8080/v1)
        .route("/v1", get(v1_root))
        .route("/v1/", get(v1_root))
        .route("/v1/models", get(list_models))
        .route("/v1/models/:model_id", get(get_model_info))
        .route("/models", get(list_models))
        .route("/models/:model_id", get(get_model_info))
        .route("/metrics", get(metrics))
        .route("/v1/mivi/status", get(get_status))
        .route("/v1/mivi/tools", get(list_tools))
        // Ollama API compatibility routes
        .route("/api/tags", get(list_models_ollama))
        .route("/api/version", get(ollama_version))
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
            Duration::from_secs(state.config.request_timeout_secs),
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
            "agent": "/v1/mivi/agent",
            "metrics": "/metrics"
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
                "permission": [],
                "context_length": state
                    .engine
                    .model_metadata()
                    .and_then(|metadata| metadata.context_length)
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
        "permission": [],
        "context_length": state
            .engine
            .model_metadata()
            .and_then(|metadata| metadata.context_length)
    }))
    .into_response()
}

async fn list_models_ollama(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let models = if state.engine.has_model() {
        let mut model = serde_json::json!({
            "name": state.model_name,
            "model": state.model_name,
            "modified_at": chrono::Utc::now().to_rfc3339(),
            "details": {
                "format": "gguf"
            }
        });

        if let Some(metadata) = state.engine.model_metadata() {
            model["size"] = serde_json::json!(metadata.size_bytes);

            if let Some(family) = &metadata.family {
                model["details"]["family"] = serde_json::json!(family);
            }
            if let Some(parameter_count) = metadata.parameter_count {
                model["details"]["parameter_count"] = serde_json::json!(parameter_count);
                model["details"]["parameter_size"] =
                    serde_json::json!(format_parameter_size(parameter_count));
            }
            if let Some(quantization_level) = &metadata.quantization_level {
                model["details"]["quantization_level"] = serde_json::json!(quantization_level);
            }
        }

        serde_json::json!([model])
    } else {
        serde_json::json!([])
    };

    Json(serde_json::json!({ "models": models }))
}

fn format_parameter_size(parameter_count: u64) -> String {
    const THOUSAND: f64 = 1_000.0;
    const MILLION: f64 = 1_000_000.0;
    const BILLION: f64 = 1_000_000_000.0;

    let count = parameter_count as f64;
    if count >= BILLION {
        format_scaled_parameter_size(count / BILLION, "B")
    } else if count >= MILLION {
        format_scaled_parameter_size(count / MILLION, "M")
    } else if count >= THOUSAND {
        format_scaled_parameter_size(count / THOUSAND, "K")
    } else {
        format!("{parameter_count}")
    }
}

fn format_scaled_parameter_size(value: f64, suffix: &str) -> String {
    if value >= 10.0 {
        format!("{value:.0}{suffix}")
    } else {
        format!("{value:.1}{suffix}")
    }
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
        assert!(response
            .headers()
            .get("access-control-allow-origin")
            .is_none());
    }

    #[tokio::test]
    async fn api_key_protects_api_metadata_routes() {
        let engine = crate::EngineActor::spawn(None);
        let state = Arc::new(AppState::new(
            "test-model",
            ToolBroker::new(),
            engine,
            Some("test-key".to_string()),
        ));
        let app = create_router(state);

        for path in [
            "/v1",
            "/v1/models",
            "/models",
            "/metrics",
            "/v1/mivi/status",
            "/v1/mivi/tools",
            "/api/tags",
            "/api/version",
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("GET")
                        .uri(path)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED, "{path}");
        }

        let authenticated = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/v1")
                    .header("Authorization", "Bearer test-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(authenticated.status(), StatusCode::OK);
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

    #[tokio::test]
    async fn ollama_tags_do_not_return_fake_model_metadata() {
        let engine = crate::EngineActor::spawn_mock();
        let state = Arc::new(AppState::new("test-model", ToolBroker::new(), engine, None));
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/api/tags")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let model = &payload["models"][0];
        assert!(model.get("size").is_none());
        assert!(model.get("digest").is_none());
        assert_eq!(model["details"]["format"], "gguf");
        assert!(model["details"].get("family").is_none());
        assert!(model["details"].get("parameter_size").is_none());
        assert!(model["details"].get("quantization_level").is_none());
    }

    #[tokio::test]
    async fn metrics_endpoint_reports_empty_counters() {
        let engine = crate::EngineActor::spawn_mock();
        let state = Arc::new(AppState::new("test-model", ToolBroker::new(), engine, None));
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["inference_requests_total"], 0);
        assert_eq!(payload["inference_requests_rejected_total"], 0);
        assert_eq!(payload["generation_count"], 0);
        assert_eq!(payload["tool_timeouts_total"], 0);
    }

    #[tokio::test]
    async fn metrics_endpoint_counts_rejected_inference_requests() {
        let engine = crate::EngineActor::spawn_mock();
        let state = Arc::new(AppState::new("test-model", ToolBroker::new(), engine, None));
        let permit = state.try_acquire_inference_slot().unwrap();
        let app = create_router(state.clone());

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"messages":[{"role":"user","content":"hello"}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        drop(permit);

        let response = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["inference_requests_total"], 0);
        assert_eq!(payload["inference_requests_rejected_total"], 1);
    }

    #[tokio::test]
    async fn metrics_endpoint_records_successful_generation() {
        let engine = crate::EngineActor::spawn_mock();
        let state = Arc::new(AppState::new("test-model", ToolBroker::new(), engine, None));
        let app = create_router(state);

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"messages":[{"role":"user","content":"hello"}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let response = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["inference_requests_total"], 1);
        assert_eq!(payload["inference_requests_rejected_total"], 0);
        assert_eq!(payload["inference_errors_total"], 0);
        assert_eq!(payload["generation_count"], 1);
        assert!(payload["prompt_tokens_total"].as_u64().unwrap() > 0);
        assert!(payload["completion_tokens_total"].as_u64().unwrap() > 0);
    }

    #[test]
    fn ollama_parameter_size_uses_compact_units() {
        assert_eq!(format_parameter_size(350_000_000), "350M");
        assert_eq!(format_parameter_size(1_200_000_000), "1.2B");
        assert_eq!(format_parameter_size(999), "999");
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

    #[tokio::test]
    async fn model_metadata_reports_context_length_to_clients() {
        let (command_tx, _command_rx) = tokio::sync::mpsc::channel(1);
        let engine = crate::EngineHandle::with_capacity_and_metadata(
            command_tx,
            true,
            1,
            Some(crate::EngineModelMetadata {
                context_length: Some(4096),
                ..crate::EngineModelMetadata::default()
            }),
        );
        let state = Arc::new(AppState::new("test-model", ToolBroker::new(), engine, None));
        let app = create_router(state);

        let models = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let models_body = axum::body::to_bytes(models.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let models_json: serde_json::Value = serde_json::from_slice(&models_body).unwrap();
        assert_eq!(models_json["data"][0]["context_length"], 4096);

        let status = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/v1/mivi/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let status_body = axum::body::to_bytes(status.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let status_json: serde_json::Value = serde_json::from_slice(&status_body).unwrap();
        assert_eq!(status_json["context_length"], 4096);
        assert_eq!(status_json["capabilities"]["profile"], "legacy");
        assert_eq!(status_json["capabilities"]["tool_codec"], "legacy_json_xml");
        assert_eq!(status_json["capabilities"]["supports_tools"], true);
        assert_eq!(status_json["capabilities"]["supports_streaming"], true);
    }

    #[tokio::test]
    async fn text_only_profile_reports_disabled_tool_capability() {
        let (command_tx, _command_rx) = tokio::sync::mpsc::channel(1);
        let engine = crate::EngineHandle::new(command_tx, true);
        let state = Arc::new(AppState::with_config(
            "test-model",
            ToolBroker::new(),
            engine,
            None,
            crate::ServerConfig {
                model_profile: Some(crate::model_profile::ModelProfileConfig::TextOnly {
                    start_of_text: "<BOS>".to_string(),
                    message_start: "<MSG>".to_string(),
                    message_end: "</MSG>".to_string(),
                }),
                ..crate::ServerConfig::default()
            },
        ));
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/v1/mivi/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let status_json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(status_json["capabilities"]["profile"], "text_only");
        assert_eq!(status_json["capabilities"]["tool_codec"], "none");
        assert_eq!(status_json["capabilities"]["supports_tools"], false);
        assert_eq!(status_json["capabilities"]["supports_streaming"], true);
    }
}

async fn get_status(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let uptime = state.start_time.elapsed().as_secs();
    let capabilities = if state.engine.has_model() {
        ModelProfile::resolve(
            state.engine.model_metadata(),
            state.config.model_profile.as_ref(),
        )
        .map(|profile| ModelCapabilityReport {
            profile: Some(profile.capability_name().to_string()),
            tool_codec: Some(profile.tool_codec_name().to_string()),
            supports_tools: profile.supports_tools(),
            supports_streaming: true,
        })
        .unwrap_or(ModelCapabilityReport {
            profile: None,
            tool_codec: None,
            supports_tools: false,
            supports_streaming: false,
        })
    } else {
        ModelCapabilityReport {
            profile: None,
            tool_codec: None,
            supports_tools: false,
            supports_streaming: false,
        }
    };

    let resp = MiviStatusResponse {
        engine: mivi_core::ENGINE_NAME.to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        model: state.model_name.clone(),
        memory_rss_mb: mivi_core::estimate_process_memory_mb(),
        active_tools_count: get_builtin_tool_definitions().len(),
        context_length: state
            .engine
            .model_metadata()
            .and_then(|metadata| metadata.context_length),
        capabilities,
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

async fn metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(state.metrics.snapshot())
}

async fn list_tools() -> impl IntoResponse {
    let tools = get_builtin_tool_definitions();
    Json(serde_json::json!({
        "object": "list",
        "tools": tools
    }))
}
