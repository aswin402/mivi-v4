//! Agent state machine and loop executor.

pub mod engine;
pub mod state;

pub use engine::AgentLoop;
pub use state::{AgentPhase, AgentState};
