use crate::state::{AgentPhase, AgentState};
use crate::xml_utils::{escape_xml_attr, escape_xml_content};
use mivi_tools::{extract_thinking, extract_tool_calls, ToolBroker, ToolResult};
use std::collections::VecDeque;
use std::time::Duration;

pub const DEFAULT_TOOL_TIMEOUT_SECS: u64 = 30;
pub const STAGNATION_WINDOW_SIZE: usize = 3;
pub const SENTINEL_FINISH: &str = "finish";
pub const SENTINEL_COMPLETE: &str = "complete_task";
pub const THINKING_PREFIX: &str = "Thinking: ";
pub const TASK_COMPLETED_MSG: &str = "Task explicitly marked completed.";

pub struct AgentLoop<'a> {
    pub state: AgentState,
    pub broker: &'a ToolBroker,
    pub tool_timeout: Duration,
    recent_actions: VecDeque<String>,
}

impl<'a> AgentLoop<'a> {
    pub fn new(state: AgentState, broker: &'a ToolBroker) -> Self {
        Self {
            state,
            broker,
            tool_timeout: Duration::from_secs(DEFAULT_TOOL_TIMEOUT_SECS),
            recent_actions: VecDeque::with_capacity(STAGNATION_WINDOW_SIZE + 1),
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.tool_timeout = timeout;
        self
    }

    /// Process step output from model and execute requested actions.
    pub async fn step(&mut self, model_output: &str) -> String {
        if self.state.step_count >= self.state.max_steps {
            self.state.phase = AgentPhase::Failed;
            return format!(
                "<error>Agent exceeded maximum step limit of {}. Terminating loop.</error>",
                self.state.max_steps
            );
        }
        self.state.step_count += 1;

        if let Some(think) = extract_thinking(model_output) {
            self.state
                .memory
                .push_back(format!("{}{}", THINKING_PREFIX, think));
            if self.state.memory.len() > crate::state::DEFAULT_MAX_MEMORY_ITEMS {
                self.state.memory.pop_front();
            }
        }

        let calls = extract_tool_calls(model_output);
        if calls.is_empty() {
            // No tool call emitted -> complete or return thought
            self.state.phase = AgentPhase::Completed;
            return model_output.to_string();
        }

        // Check if explicitly calling finish
        if calls
            .iter()
            .any(|c| c.name == SENTINEL_FINISH || c.name == SENTINEL_COMPLETE)
        {
            self.state.phase = AgentPhase::Completed;
            return TASK_COMPLETED_MSG.to_string();
        }

        // Stagnation detection: track repeated single actions and 2-cycle oscillations
        let call_signature = format!("{:?}", calls);
        self.recent_actions.push_back(call_signature.clone());
        if self.recent_actions.len() > 6 {
            self.recent_actions.pop_front();
        }

        let is_1_cycle = self.recent_actions.len() >= STAGNATION_WINDOW_SIZE
            && self
                .recent_actions
                .iter()
                .rev()
                .take(STAGNATION_WINDOW_SIZE)
                .all(|a| a == &call_signature);

        let is_2_cycle = self.recent_actions.len() == 6
            && self.recent_actions[0] == self.recent_actions[2]
            && self.recent_actions[2] == self.recent_actions[4]
            && self.recent_actions[1] == self.recent_actions[3]
            && self.recent_actions[3] == self.recent_actions[5]
            && self.recent_actions[0] != self.recent_actions[1];

        if is_1_cycle || is_2_cycle {
            self.state.phase = AgentPhase::Failed;
            return "<warning>Agent stagnation detected: repeated or oscillating tool calls detected. Terminating loop safely.</warning>".to_string();
        }

        self.state.phase = AgentPhase::Acting;
        let mut results_str = String::new();

        for call in &calls {
            let res = match tokio::time::timeout(self.tool_timeout, self.broker.execute(call)).await
            {
                Ok(result) => result,
                Err(_) => ToolResult::err(
                    call.name.clone(),
                    format!(
                        "Tool '{}' timed out after {:?}",
                        call.name, self.tool_timeout
                    ),
                ),
            };

            let body = if res.success {
                res.output
            } else {
                res.error.unwrap_or_default()
            };

            let status_attr = if res.success { "" } else { " status=\"error\"" };
            results_str.push_str(&format!(
                "<tool_result name=\"{}\"{}>{}</tool_result>\n",
                escape_xml_attr(&res.name),
                status_attr,
                escape_xml_content(&body)
            ));
        }

        self.state.phase = AgentPhase::Observing;
        results_str
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_agent_memory_bounding() {
        let broker = ToolBroker::new();
        let state = AgentState::new("test task", 200);
        let mut agent = AgentLoop::new(state, &broker);

        for i in 0..150 {
            let msg = format!("<think>Thought {}</think>", i);
            agent.step(&msg).await;
        }

        assert_eq!(
            agent.state.memory.len(),
            crate::state::DEFAULT_MAX_MEMORY_ITEMS
        );
        assert_eq!(
            agent.state.memory.back().unwrap(),
            &format!("{}{}", THINKING_PREFIX, "Thought 149")
        );
    }
}
