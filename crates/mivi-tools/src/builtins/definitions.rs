//! Tool definitions and registration routines for default built-in tools.

use super::calc::handle_calculator_cancellable;
use super::fs::{
    handle_list_dir_cancellable, handle_read_file_cancellable, handle_write_file_cancellable,
};
use crate::broker::ToolBroker;
use crate::schema::{FunctionDefinition, ToolDefinition};
use std::path::Path;
use std::sync::Arc;

pub const TOOL_READ_FILE: &str = "read_file";
pub const TOOL_WRITE_FILE: &str = "write_file";
pub const TOOL_LIST_DIR: &str = "list_dir";
pub const TOOL_CALCULATOR: &str = "calculator";

/// Register all default built-in tools into a ToolBroker.
pub async fn register_builtin_tools(broker: &ToolBroker, workspace_root: &Path) {
    let ws_path = workspace_root.to_path_buf();

    let ws1 = ws_path.clone();
    broker
        .register_cancellable(
            TOOL_READ_FILE,
            Arc::new(move |args, cancellation| {
                handle_read_file_cancellable(args, &ws1, cancellation)
            }),
        )
        .await;

    let ws2 = ws_path.clone();
    broker
        .register_cancellable(
            TOOL_WRITE_FILE,
            Arc::new(move |args, cancellation| {
                handle_write_file_cancellable(args, &ws2, cancellation)
            }),
        )
        .await;

    let ws3 = ws_path.clone();
    broker
        .register_cancellable(
            TOOL_LIST_DIR,
            Arc::new(move |args, cancellation| {
                handle_list_dir_cancellable(args, &ws3, cancellation)
            }),
        )
        .await;

    broker
        .register_cancellable(TOOL_CALCULATOR, Arc::new(handle_calculator_cancellable))
        .await;
}

/// Helper function to return ToolDefinitions for the built-in tools.
pub fn get_builtin_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            r#type: "function".to_string(),
            function: FunctionDefinition {
                name: TOOL_READ_FILE.to_string(),
                description: Some(
                    "Reads and returns the complete text content of a file in workspace"
                        .to_string(),
                ),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Relative path to file" }
                    },
                    "required": ["path"]
                }),
            },
        },
        ToolDefinition {
            r#type: "function".to_string(),
            function: FunctionDefinition {
                name: TOOL_WRITE_FILE.to_string(),
                description: Some(
                    "Writes text content to a specified file in workspace".to_string(),
                ),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Relative path to file" },
                        "content": { "type": "string", "description": "File content to write" }
                    },
                    "required": ["path", "content"]
                }),
            },
        },
        ToolDefinition {
            r#type: "function".to_string(),
            function: FunctionDefinition {
                name: TOOL_LIST_DIR.to_string(),
                description: Some("Lists contents of a directory in workspace".to_string()),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Relative directory path" }
                    }
                }),
            },
        },
        ToolDefinition {
            r#type: "function".to_string(),
            function: FunctionDefinition {
                name: TOOL_CALCULATOR.to_string(),
                description: Some(
                    "Evaluates simple arithmetic expressions (+, -, *, /)".to_string(),
                ),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "expression": { "type": "string", "description": "Math expression (e.g. '125 * 8 + 40')" }
                    },
                    "required": ["expression"]
                }),
            },
        },
    ]
}
