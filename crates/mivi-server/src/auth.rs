use crate::types::AppError;
use axum::{
    extract::Request,
    http::{header, HeaderMap},
    middleware::Next,
    response::{IntoResponse, Response},
};

use subtle::ConstantTimeEq;

pub const AUTH_MISSING_HEADER: &str = "Missing Authorization header with Bearer token.";
pub const AUTH_INVALID_KEY: &str = "Invalid API key provided.";

#[inline]
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.ct_eq(b).into()
}

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
                return Err(AppError::Unauthorized(AUTH_MISSING_HEADER.to_string()).into_response());
            }
        };

        if !constant_time_eq(token.as_bytes(), expected.as_bytes()) {
            return Err(AppError::Unauthorized(AUTH_INVALID_KEY.to_string()).into_response());
        }
    }

    Ok(next.run(request).await)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_time_eq() {
        assert!(constant_time_eq(b"secret-key", b"secret-key"));
        assert!(!constant_time_eq(b"secret-key", b"secret-kex"));
        assert!(!constant_time_eq(b"secret-key", b"secret"));
        assert!(!constant_time_eq(b"secret", b"secret-key"));
        assert!(constant_time_eq(b"", b""));
    }
}
