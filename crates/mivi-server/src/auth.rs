//! Optional API Key Authentication Middleware for Mivi server routes.

use crate::types::{OpenAiErrorDetail, OpenAiErrorResponse};
use axum::{
    extract::Request,
    http::{header, HeaderMap, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};

pub async fn require_api_key(
    axum::extract::State(expected_key): axum::extract::State<Option<String>>,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Result<Response, Response> {
    if let Some(ref expected) = expected_key {
        let auth_header = headers
            .get(header::AUTHORIZATION)
            .and_then(|val| val.to_str().ok());

        let token = match auth_header {
            Some(h) if h.starts_with("Bearer ") => &h[7..],
            Some(h) if h.starts_with("bearer ") => &h[7..],
            _ => {
                let resp = OpenAiErrorResponse {
                    error: OpenAiErrorDetail {
                        message: "Missing Authorization header with Bearer token.".to_string(),
                        r#type: "invalid_request_error".to_string(),
                        param: None,
                        code: Some("invalid_api_key".to_string()),
                    },
                };
                return Err((StatusCode::UNAUTHORIZED, Json(resp)).into_response());
            }
        };

        if token != expected {
            let resp = OpenAiErrorResponse {
                error: OpenAiErrorDetail {
                    message: "Invalid API key provided.".to_string(),
                    r#type: "invalid_request_error".to_string(),
                    param: None,
                    code: Some("invalid_api_key".to_string()),
                },
            };
            return Err((StatusCode::UNAUTHORIZED, Json(resp)).into_response());
        }
    }

    Ok(next.run(request).await)
}
