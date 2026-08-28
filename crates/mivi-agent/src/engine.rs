//! Canonical agent execution loop.

use crate::state::{AgentPhase, AgentState};
use mivi_tools::{extract_thinking, extract_tool_calls, ToolBroker};

pub struct AgentLoop<'a> {
    pub state: AgentState,
    pub broker: &'a ToolBroker,
}

impl<'a> AgentLoop<'a> {
    pub fn new(state: AgentState, broker: &'a ToolBroker) -> Self {
        Self { state, broker }
    }

    /// Process step output from model and execute requested actions.
    pub async fn step(&mut self, model_output: &str) -> String {
        self.state.step_count += 1;

        if let Some(think) = extract_thinking(model_output) {
            self.state.memory.push(format!("Thinking: {}", think));
        }

        let calls = extract_tool_calls(model_output);
        if calls.is_empty() {
            self.state.phase = AgentPhase::Completed;
            return model_output.to_string();
        }

        self.state.phase = AgentPhase::Acting;
        let mut results_str = String::new();

        for call in &calls {
            let res = self.broker.execute(call).await;
            results_str.push_str(&format!(
                "<tool_result name=\"{}\">{}</tool_result>\n",
                res.name,
                if res.success { res.output } else { res.error.unwrap_or_default() }
            ));
        }

        self.state.phase = AgentPhase::Verifying;
        results_str
    }
}
