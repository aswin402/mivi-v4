use crate::state::{AgentPhase, AgentState};
use crate::xml_utils::{escape_xml_attr, escape_xml_content};
use mivi_tools::{extract_thinking, extract_tool_calls, ToolBroker, ToolResult};
use std::collections::{HashSet, VecDeque};
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
    allowed_tools: Option<HashSet<String>>,
    recent_actions: VecDeque<String>,
}

impl<'a> AgentLoop<'a> {
    pub fn new(state: AgentState, broker: &'a ToolBroker) -> Self {
        Self {
            state,
            broker,
            tool_timeout: Duration::from_secs(DEFAULT_TOOL_TIMEOUT_SECS),
            allowed_tools: None,
            recent_actions: VecDeque::with_capacity(STAGNATION_WINDOW_SIZE + 1),
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.tool_timeout = timeout;
        self
    }

    /// Restrict this run to the named tools. `None` keeps the broker's registered tools available.
    pub fn with_allowed_tools(mut self, allowed_tools: Option<Vec<String>>) -> Self {
        self.allowed_tools = allowed_tools.map(|tools| tools.into_iter().collect());
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
        let mut timed_out = false;

        for call in &calls {
            let res = if self
                .allowed_tools
                .as_ref()
                .is_some_and(|allowed| !allowed.contains(&call.name))
            {
                ToolResult::err(
                    call.name.clone(),
                    format!("Tool '{}' is not allowed for this agent run", call.name),
                )
            } else {
                match tokio::time::timeout(self.tool_timeout, self.broker.execute(call)).await {
                    Ok(result) => result,
                    Err(_) => {
                        timed_out = true;
                        ToolResult::err(
                            call.name.clone(),
                            format!(
                                "Tool '{}' timed out after {:?}; side-effect status is unknown",
                                call.name, self.tool_timeout
                            ),
                        )
                    }
                }
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

        self.state.phase = if timed_out {
            AgentPhase::Failed
        } else {
            AgentPhase::Observing
        };
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

    #[tokio::test]
    async fn disallowed_tools_are_not_executed() {
        use std::sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        };

        let broker = ToolBroker::new();
        let executed = Arc::new(AtomicBool::new(false));
        let executed_by_tool = Arc::clone(&executed);
        broker
            .register(
                "secret_tool",
                Arc::new(move |_| {
                    executed_by_tool.store(true, Ordering::SeqCst);
                    ToolResult::ok("secret_tool", "should not run")
                }),
            )
            .await;

        let state = AgentState::new("test task", 3);
        let mut agent =
            AgentLoop::new(state, &broker).with_allowed_tools(Some(vec!["safe_tool".to_string()]));
        let result = agent
            .step(r#"<tool_call>{"name":"secret_tool","arguments":{}}</tool_call>"#)
            .await;

        assert!(!executed.load(Ordering::SeqCst));
        assert!(result.contains("not allowed"));
    }

    #[tokio::test]
    async fn timed_out_tool_fails_agent_to_prevent_ambiguous_retries() {
        let broker = ToolBroker::new();
        broker
            .register(
                "slow_tool",
                std::sync::Arc::new(|_| {
                    std::thread::sleep(Duration::from_millis(25));
                    ToolResult::ok("slow_tool", "finished")
                }),
            )
            .await;

        let state = AgentState::new("test task", 3);
        let mut agent = AgentLoop::new(state, &broker).with_timeout(Duration::from_millis(1));
        let result = agent
            .step(r#"<tool_call>{"name":"slow_tool","arguments":{}}</tool_call>"#)
            .await;

        assert!(result.contains("timed out"));
        assert_eq!(agent.state.phase, AgentPhase::Failed);
    }
}
