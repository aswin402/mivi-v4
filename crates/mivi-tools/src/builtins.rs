//! Built-in standard tools for mivi-v4 agent workflows.

use crate::broker::ToolBroker;
use crate::schema::{FunctionDefinition, ToolDefinition, ToolResult};
use std::path::Path;
use std::sync::Arc;

/// Register all default built-in tools into a ToolBroker.
pub async fn register_builtin_tools(broker: &ToolBroker, workspace_root: &Path) {
    let ws_path = workspace_root.to_path_buf();

    // 1. Filesystem: read_file
    let ws_clone = ws_path.clone();
    broker
        .register(
            "read_file",
            Arc::new(move |args| {
                let path_str = match args.get("path").and_then(|v| v.as_str()) {
                    Some(p) => p,
                    None => {
                        return ToolResult {
                            name: "read_file".to_string(),
                            success: false,
                            output: String::new(),
                            error: Some("Missing required parameter 'path'".to_string()),
                        }
                    }
                };

                let target = ws_clone.join(path_str);
                match std::fs::read_to_string(&target) {
                    Ok(content) => ToolResult {
                        name: "read_file".to_string(),
                        success: true,
                        output: content,
                        error: None,
                    },
                    Err(e) => ToolResult {
                        name: "read_file".to_string(),
                        success: false,
                        output: String::new(),
                        error: Some(format!("Failed to read file '{}': {}", path_str, e)),
                    },
                }
            }),
        )
        .await;

    // 2. Filesystem: write_file
    let ws_clone = ws_path.clone();
    broker
        .register(
            "write_file",
            Arc::new(move |args| {
                let path_str = match args.get("path").and_then(|v| v.as_str()) {
                    Some(p) => p,
                    None => {
                        return ToolResult {
                            name: "write_file".to_string(),
                            success: false,
                            output: String::new(),
                            error: Some("Missing required parameter 'path'".to_string()),
                        }
                    }
                };
                let content = match args.get("content").and_then(|v| v.as_str()) {
                    Some(c) => c,
                    None => {
                        return ToolResult {
                            name: "write_file".to_string(),
                            success: false,
                            output: String::new(),
                            error: Some("Missing required parameter 'content'".to_string()),
                        }
                    }
                };

                let target = ws_clone.join(path_str);
                if let Some(parent) = target.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                match std::fs::write(&target, content) {
                    Ok(_) => ToolResult {
                        name: "write_file".to_string(),
                        success: true,
                        output: format!("Successfully wrote {} bytes to '{}'", content.len(), path_str),
                        error: None,
                    },
                    Err(e) => ToolResult {
                        name: "write_file".to_string(),
                        success: false,
                        output: String::new(),
                        error: Some(format!("Failed to write file '{}': {}", path_str, e)),
                    },
                }
            }),
        )
        .await;

    // 3. Filesystem: list_dir
    let ws_clone = ws_path.clone();
    broker
        .register(
            "list_dir",
            Arc::new(move |args| {
                let path_str = args
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or(".");
                let target = ws_clone.join(path_str);

                match std::fs::read_dir(&target) {
                    Ok(entries) => {
                        let mut names = Vec::new();
                        for entry in entries.flatten() {
                            if let Ok(file_type) = entry.file_type() {
                                let kind = if file_type.is_dir() { "dir" } else { "file" };
                                names.push(format!("{}: {}", kind, entry.file_name().to_string_lossy()));
                            }
                        }
                        ToolResult {
                            name: "list_dir".to_string(),
                            success: true,
                            output: names.join("\n"),
                            error: None,
                        }
                    }
                    Err(e) => ToolResult {
                        name: "list_dir".to_string(),
                        success: false,
                        output: String::new(),
                        error: Some(format!("Failed to list dir '{}': {}", path_str, e)),
                    },
                }
            }),
        )
        .await;

    // 4. Calculator tool
    broker
        .register(
            "calculator",
            Arc::new(|args| {
                let expr = match args.get("expression").and_then(|v| v.as_str()) {
                    Some(e) => e,
                    None => {
                        return ToolResult {
                            name: "calculator".to_string(),
                            success: false,
                            output: String::new(),
                            error: Some("Missing required parameter 'expression'".to_string()),
                        }
                    }
                };

                // Simple arithmetic evaluation support (+, -, *, /)
                let result = evaluate_simple_expression(expr);
                match result {
                    Ok(val) => ToolResult {
                        name: "calculator".to_string(),
                        success: true,
                        output: val.to_string(),
                        error: None,
                    },
                    Err(err) => ToolResult {
                        name: "calculator".to_string(),
                        success: false,
                        output: String::new(),
                        error: Some(err),
                    },
                }
            }),
        )
        .await;
}

/// Helper function to return ToolDefinitions for the built-in tools.
pub fn get_builtin_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            r#type: "function".to_string(),
            function: FunctionDefinition {
                name: "read_file".to_string(),
                description: Some("Reads and returns the complete text content of a file in workspace".to_string()),
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
                name: "write_file".to_string(),
                description: Some("Writes text content to a specified file in workspace".to_string()),
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
                name: "list_dir".to_string(),
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
                name: "calculator".to_string(),
                description: Some("Evaluates simple arithmetic expressions (+, -, *, /)".to_string()),
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

fn evaluate_simple_expression(expr: &str) -> std::result::Result<f64, String> {
    let clean: String = expr.chars().filter(|c| !c.is_whitespace()).collect();
    if clean.is_empty() {
        return Err("Empty expression".to_string());
    }

    // Split on addition/subtraction
    if let Some(pos) = clean.rfind('+') {
        let left = evaluate_simple_expression(&clean[..pos])?;
        let right = evaluate_simple_expression(&clean[pos + 1..])?;
        return Ok(left + right);
    }
    if let Some(pos) = clean.rfind('-') {
        if pos > 0 {
            let left = evaluate_simple_expression(&clean[..pos])?;
            let right = evaluate_simple_expression(&clean[pos + 1..])?;
            return Ok(left - right);
        }
    }
    if let Some(pos) = clean.rfind('*') {
        let left = evaluate_simple_expression(&clean[..pos])?;
        let right = evaluate_simple_expression(&clean[pos + 1..])?;
        return Ok(left * right);
    }
    if let Some(pos) = clean.rfind('/') {
        let left = evaluate_simple_expression(&clean[..pos])?;
        let right = evaluate_simple_expression(&clean[pos + 1..])?;
        if right == 0.0 {
            return Err("Division by zero".to_string());
        }
        return Ok(left / right);
    }

    clean.parse::<f64>().map_err(|e| format!("Invalid number '{}': {}", clean, e))
}
