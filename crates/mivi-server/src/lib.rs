//! OpenAI-compatible HTTP API server for mivi-v4.

pub mod api;
pub mod auth;
pub mod engine_actor;
pub mod streaming;
pub mod types;

pub use api::{create_router, AppState};
pub use auth::require_api_key;
pub use engine_actor::{EngineActor, EngineHandle};
pub use streaming::*;
pub use types::*;
