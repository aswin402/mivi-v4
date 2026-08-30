use crate::state::{AgentPhase, AgentState};
use crate::xml_utils::{escape_xml_attr, escape_xml_content};
use mivi_tools::{extract_thinking, extract_tool_calls, ToolBroker, ToolResult};
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
    recent_actions: Vec<String>,
}

impl<'a> AgentLoop<'a> {
    pub fn new(state: AgentState, broker: &'a ToolBroker) -> Self {
        Self {
            state,
            broker,
            tool_timeout: Duration::from_secs(DEFAULT_TOOL_TIMEOUT_SECS),
            recent_actions: Vec::new(),
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
                .push(format!("{}{}", THINKING_PREFIX, think));
            if self.state.memory.len() > crate::state::DEFAULT_MAX_MEMORY_ITEMS {
                self.state.memory.remove(0);
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

        // Stagnation detection: track repeated identical calls
        let call_signature = format!("{:?}", calls);
        self.recent_actions.push(call_signature.clone());
        if self.recent_actions.len() > STAGNATION_WINDOW_SIZE {
            self.recent_actions.remove(0);
        }
        if self.recent_actions.len() == STAGNATION_WINDOW_SIZE
            && self.recent_actions.iter().all(|a| a == &call_signature)
        {
            self.state.phase = AgentPhase::Completed;
            return format!(
                "<warning>Agent stagnation detected: repeated same tool call {} times. Terminating loop safely.</warning>",
                STAGNATION_WINDOW_SIZE
            );
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

            results_str.push_str(&format!(
                "<tool_result name=\"{}\">{}</tool_result>\n",
                escape_xml_attr(&res.name),
                escape_xml_content(&body)
            ));
        }

        self.state.phase = AgentPhase::Verifying;
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
            agent.state.memory.last().unwrap(),
            &format!("{}{}", THINKING_PREFIX, "Thought 149")
        );
    }
}
