//! Tool schema representations matching JSON Schema & OpenAI tools spec.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub r#type: String, // "function"
    pub function: FunctionDefinition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

pub const PARSE_ERROR_TOOL_NAME: &str = "__parse_error";

impl ToolCall {
    pub fn new(name: impl Into<String>, arguments: serde_json::Value) -> Self {
        Self {
            name: name.into(),
            arguments,
        }
    }

    pub fn parse_error(err_msg: &str, raw_str: &str) -> Self {
        Self {
            name: PARSE_ERROR_TOOL_NAME.to_string(),
            arguments: serde_json::json!({
                "error": err_msg,
                "raw": raw_str,
            }),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub name: String,
    pub success: bool,
    pub output: String,
    pub error: Option<String>,
}

impl ToolResult {
    pub fn ok(name: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            success: true,
            output: output.into(),
            error: None,
        }
    }

    pub fn err(name: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            success: false,
            output: String::new(),
            error: Some(error.into()),
        }
    }
}
