//! Agent state machine and loop executor.

pub mod engine;
pub mod state;
pub mod xml_utils;

pub use engine::AgentLoop;
pub use state::{AgentPhase, AgentState};
pub use xml_utils::{escape_xml_attr, escape_xml_content};
