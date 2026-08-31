//! OpenAI-compatible HTTP API server for mivi-v4.

pub mod auth;
pub mod config;
pub mod engine_actor;
pub mod grammar;
pub mod logging;
pub mod routes;
pub mod state;
pub mod streaming;
pub mod types;
pub mod watchdog;

pub use auth::require_api_key;
pub use config::ServerConfig;
pub use engine_actor::{EngineActor, EngineHandle};
pub use grammar::{JsonConstraintState, ResponseFormat};
pub use logging::{mivi_log_middleware, summarize_prompt, LogMetadata};
pub use routes::create_router;
pub use state::AppState;
pub use streaming::{
    create_chunk_event, create_content_chunk_event, create_done_chunk_event, create_done_event,
    create_thinking_chunk_event, send_sse_sequence, ChatCompletionChunk, ChunkChoice, ChunkDelta,
};
pub use types::{
    AgentRunRequest, AppError, ChatCompletionRequest, ChatCompletionResponse, ChoiceDto,
    MessageDto, MiviStatusResponse, OpenAiErrorDetail, OpenAiErrorResponse, UsageDto,
};
pub use watchdog::{ResourceWatchdog, WatchdogConfig};

/// Attempts to bind a TcpListener to the given IP and port.
///
/// If the requested port is already in use (`std::io::ErrorKind::AddrInUse`), this function
/// will automatically attempt successive ports (`port + 1`, `port + 2`, ...) up to `max_attempts` times.
pub async fn bind_with_fallback(
    ip: std::net::IpAddr,
    initial_port: u16,
    max_attempts: u16,
) -> std::io::Result<(tokio::net::TcpListener, std::net::SocketAddr)> {
    let max_attempts = max_attempts.max(1);
    for offset in 0..max_attempts {
        let current_port = initial_port.checked_add(offset).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Port number overflowed u16 range during fallback search",
            )
        })?;
        let addr = std::net::SocketAddr::new(ip, current_port);

        match tokio::net::TcpListener::bind(addr).await {
            Ok(listener) => {
                let actual_addr = listener.local_addr()?;
                if offset > 0 {
                    tracing::warn!(
                        "Preferred port {} was in use. Successfully bound fallback port http://{}",
                        initial_port,
                        actual_addr
                    );
                }
                return Ok((listener, actual_addr));
            }
            Err(err) if err.kind() == std::io::ErrorKind::AddrInUse => {
                tracing::warn!(
                    "Port {} is currently in use (AddrInUse). Trying fallback port {}...",
                    current_port,
                    current_port.saturating_add(1)
                );
                continue;
            }
            Err(err) => return Err(err),
        }
    }

    Err(std::io::Error::new(
        std::io::ErrorKind::AddrInUse,
        format!(
            "Failed to bind to any port in range {}..={} after {} attempts (all ports in use)",
            initial_port,
            initial_port.saturating_add(max_attempts - 1),
            max_attempts
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    #[tokio::test]
    async fn test_bind_with_fallback_success() {
        let ip = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1));
        // First bind a random listener to occupy a port
        let initial_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let occupied_port = initial_listener.local_addr().unwrap().port();

        // Attempting to bind on occupied_port with fallback should bind to next available port
        let (fallback_listener, fallback_addr) =
            bind_with_fallback(ip, occupied_port, 10).await.unwrap();

        assert_ne!(fallback_addr.port(), occupied_port);
        assert!(fallback_addr.port() > occupied_port);
        drop(initial_listener);
        drop(fallback_listener);
    }
}
