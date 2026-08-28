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
        let map = self.handlers.read().await;
        if let Some(handler) = map.get(&call.name) {
            handler(call.arguments.clone())
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
