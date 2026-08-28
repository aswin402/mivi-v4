//! Built-in standard tools for mivi-v4 agent workflows.

use crate::broker::ToolBroker;
use crate::schema::{FunctionDefinition, ToolDefinition, ToolResult};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

/// Safely resolve an untrusted relative path against a trusted workspace root.
/// Rejects '..', absolute paths, and Windows UNC prefixes.
pub fn safe_join(base: &Path, untrusted: &str) -> Result<PathBuf, String> {
    let rel = Path::new(untrusted);
    for component in rel.components() {
        match component {
            Component::Normal(_) | Component::CurDir => {}
            Component::ParentDir => return Err(format!("Path traversal blocked: '..' is forbidden in '{}'", untrusted)),
            Component::RootDir | Component::Prefix(_) => {
                return Err(format!("Absolute or prefixed paths are forbidden in '{}'", untrusted))
            }
        }
    }
    Ok(base.join(rel))
}

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

                let target = match safe_join(&ws_clone, path_str) {
                    Ok(t) => t,
                    Err(e) => {
                        return ToolResult {
                            name: "read_file".to_string(),
                            success: false,
                            output: String::new(),
                            error: Some(e),
                        }
                    }
                };

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

                let target = match safe_join(&ws_clone, path_str) {
                    Ok(t) => t,
                    Err(e) => {
                        return ToolResult {
                            name: "write_file".to_string(),
                            success: false,
                            output: String::new(),
                            error: Some(e),
                        }
                    }
                };

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
                let target = match safe_join(&ws_clone, path_str) {
                    Ok(t) => t,
                    Err(e) => {
                        return ToolResult {
                            name: "list_dir".to_string(),
                            success: false,
                            output: String::new(),
                            error: Some(e),
                        }
                    }
                };

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

#[derive(Debug, PartialEq, Clone)]
enum MathToken {
    Number(f64),
    Plus,
    Minus,
    Star,
    Slash,
    LParen,
    RParen,
}

fn tokenize_expr(input: &str) -> std::result::Result<Vec<MathToken>, String> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();

    while let Some(&c) = chars.peek() {
        match c {
            ' ' | '\t' | '\r' | '\n' => {
                chars.next();
            }
            '+' => {
                tokens.push(MathToken::Plus);
                chars.next();
            }
            '-' => {
                tokens.push(MathToken::Minus);
                chars.next();
            }
            '*' => {
                tokens.push(MathToken::Star);
                chars.next();
            }
            '/' => {
                tokens.push(MathToken::Slash);
                chars.next();
            }
            '(' => {
                tokens.push(MathToken::LParen);
                chars.next();
            }
            ')' => {
                tokens.push(MathToken::RParen);
                chars.next();
            }
            '0'..='9' | '.' => {
                let mut num_str = String::new();
                while let Some(&nc) = chars.peek() {
                    if nc.is_ascii_digit() || nc == '.' {
                        num_str.push(nc);
                        chars.next();
                    } else {
                        break;
                    }
                }
                let val = num_str
                    .parse::<f64>()
                    .map_err(|e| format!("Invalid number '{}': {}", num_str, e))?;
                tokens.push(MathToken::Number(val));
            }
            _ => return Err(format!("Unexpected character in math expression: '{}'", c)),
        }
    }
    Ok(tokens)
}

struct PrattParser<'a> {
    tokens: &'a [MathToken],
    pos: usize,
}

impl<'a> PrattParser<'a> {
    fn new(tokens: &'a [MathToken]) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&MathToken> {
        self.tokens.get(self.pos)
    }

    fn next(&mut self) -> Option<&MathToken> {
        let tok = self.tokens.get(self.pos);
        if tok.is_some() {
            self.pos += 1;
        }
        tok
    }

    fn infix_binding_power(op: &MathToken) -> Option<(u8, u8)> {
        match op {
            MathToken::Plus | MathToken::Minus => Some((1, 2)),
            MathToken::Star | MathToken::Slash => Some((3, 4)),
            _ => None,
        }
    }

    fn prefix_binding_power(op: &MathToken) -> Option<u8> {
        match op {
            MathToken::Plus | MathToken::Minus => Some(5),
            _ => None,
        }
    }

    fn parse_expr(&mut self, min_bp: u8) -> std::result::Result<f64, String> {
        let mut lhs = match self.next() {
            Some(MathToken::Number(n)) => *n,
            Some(MathToken::Minus) => {
                let bp = Self::prefix_binding_power(&MathToken::Minus).unwrap();
                let rhs = self.parse_expr(bp)?;
                -rhs
            }
            Some(MathToken::Plus) => {
                let bp = Self::prefix_binding_power(&MathToken::Plus).unwrap();
                self.parse_expr(bp)?
            }
            Some(MathToken::LParen) => {
                let val = self.parse_expr(0)?;
                if self.next() != Some(&MathToken::RParen) {
                    return Err("Expected closing parenthesis ')'".to_string());
                }
                val
            }
            Some(tok) => return Err(format!("Unexpected token in prefix position: {:?}", tok)),
            None => return Err("Unexpected end of expression".to_string()),
        };

        loop {
            let op = match self.peek() {
                Some(op) => op,
                None => break,
            };

            if let Some((l_bp, r_bp)) = Self::infix_binding_power(op) {
                if l_bp < min_bp {
                    break;
                }
                let op = self.next().unwrap().clone();
                let rhs = self.parse_expr(r_bp)?;

                lhs = match op {
                    MathToken::Plus => lhs + rhs,
                    MathToken::Minus => lhs - rhs,
                    MathToken::Star => lhs * rhs,
                    MathToken::Slash => {
                        if rhs == 0.0 {
                            return Err("Division by zero".to_string());
                        }
                        lhs / rhs
                    }
                    _ => unreachable!(),
                };
                continue;
            }
            break;
        }

        Ok(lhs)
    }
}

pub fn evaluate_expression(expr: &str) -> std::result::Result<f64, String> {
    let tokens = tokenize_expr(expr)?;
    if tokens.is_empty() {
        return Err("Empty expression".to_string());
    }
    let mut parser = PrattParser::new(&tokens);
    let res = parser.parse_expr(0)?;
    if parser.pos < tokens.len() {
        return Err("Unparsed trailing tokens in expression".to_string());
    }
    Ok(res)
}

fn evaluate_simple_expression(expr: &str) -> std::result::Result<f64, String> {
    evaluate_expression(expr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_join_blocks_traversal() {
        let base = Path::new("/workspace");
        assert!(safe_join(base, "../../etc/passwd").is_err());
        assert!(safe_join(base, "/etc/shadow").is_err());
        assert!(safe_join(base, "valid/sub/path.txt").is_ok());
        assert!(safe_join(base, "./local_file.rs").is_ok());
    }

    #[test]
    fn test_calculator_pratt_parser() {
        assert_eq!(evaluate_expression("3 + 4 * 2").unwrap(), 11.0);
        assert_eq!(evaluate_expression("(3 + 4) * 2").unwrap(), 14.0);
        assert_eq!(evaluate_expression("-5 + 10").unwrap(), 5.0);
        assert_eq!(evaluate_expression("3 - -5").unwrap(), 8.0);
        assert_eq!(evaluate_expression("-(3 * 2) + -4").unwrap(), -10.0);
        assert_eq!(evaluate_expression("100 / 4 / 5").unwrap(), 5.0);
    }
}
