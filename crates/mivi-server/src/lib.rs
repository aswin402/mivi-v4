//! OpenAI-compatible HTTP API server for mivi-v4.

pub mod auth;
pub mod config;
pub mod engine_actor;
pub mod grammar;
pub mod routes;
pub mod state;
pub mod streaming;
pub mod types;

pub use auth::require_api_key;
pub use config::ServerConfig;
pub use engine_actor::{EngineActor, EngineHandle};
pub use grammar::{JsonConstraintState, ResponseFormat};
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
