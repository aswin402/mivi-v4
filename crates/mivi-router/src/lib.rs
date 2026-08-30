//! Fast regex-based intent classification and semantic routing for tasks and tools.
//!
//! Provides two-tier classification:
//! - Level 1: Primary Task Family (Chat, Agent, Code, Debug, Research, General)
//! - Level 2: Secondary Specialization (Implementation, Debugging, Testing, Frontend, Backend, Architecture)

pub mod classifier;

pub use classifier::{
    CodeSpecialization, IntentClassifier, RouteDecision, TaskFamily, CONFIDENCE_CODE,
    CONFIDENCE_DEBUG, CONFIDENCE_DEFAULT_AGENT, CONFIDENCE_RESEARCH, CONFIDENCE_SHORT_CHAT,
    CONFIDENCE_TEST, SHORT_PROMPT_CHAR_LIMIT,
};
