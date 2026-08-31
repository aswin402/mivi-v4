//! Canonical agent state machine definition.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentPhase {
    Observing,
    Thinking,
    Planning,
    Acting,
    Verifying,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    pub task: String,
    pub phase: AgentPhase,
    pub step_count: usize,
    pub max_steps: usize,
    pub plan: Vec<String>,
    pub memory: VecDeque<String>,
}

pub const DEFAULT_MAX_AGENT_STEPS: usize = 10;
pub const DEFAULT_MAX_MEMORY_ITEMS: usize = 100;

impl AgentState {
    pub fn new(task: &str, max_steps: usize) -> Self {
        Self {
            task: task.to_string(),
            phase: AgentPhase::Observing,
            step_count: 0,
            max_steps,
            plan: Vec::new(),
            memory: VecDeque::new(),
        }
    }
}

impl Default for AgentState {
    fn default() -> Self {
        Self::new("", DEFAULT_MAX_AGENT_STEPS)
    }
}
