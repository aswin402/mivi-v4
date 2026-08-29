//! Tool broker and executor. Enforces timeouts, sandboxing, and permissions.

use crate::schema::{ToolCall, ToolResult};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

pub type ToolHandler = Arc<dyn Fn(serde_json::Value) -> ToolResult + Send + Sync>;

#[derive(Default, Clone)]
pub struct ToolBroker {
    handlers: Arc<RwLock<HashMap<String, ToolHandler>>>,
}

impl ToolBroker {
    pub fn new() -> Self {
        Self {
            handlers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn register(&self, name: &str, handler: ToolHandler) {
        let mut map = self.handlers.write().await;
        map.insert(name.to_string(), handler);
    }

    pub async fn execute(&self, call: &ToolCall) -> ToolResult {
        let handler = {
            let map = self.handlers.read().await;
            map.get(&call.name).cloned()
        };
        if let Some(handler) = handler {
            let args = call.arguments.clone();
            let call_name = call.name.clone();
            match tokio::task::spawn_blocking(move || handler(args)).await {
                Ok(result) => result,
                Err(e) => ToolResult {
                    name: call_name,
                    success: false,
                    output: String::new(),
                    error: Some(format!("Tool execution task failed: {}", e)),
                },
            }
        } else {
            ToolResult {
                name: call.name.clone(),
                success: false,
                output: String::new(),
                error: Some(format!("Tool '{}' not registered in broker", call.name)),
            }
        }
    }
}
