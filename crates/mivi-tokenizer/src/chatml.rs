//! ChatML formatting utility for OpenAI messages and agent schemas.

use crate::special::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::System => write!(f, "system"),
            Self::User => write!(f, "user"),
            Self::Assistant => write!(f, "assistant"),
            Self::Tool => write!(f, "tool"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: Option<String>,
    pub name: Option<String>,
}

/// Formats OpenAI-style messages into ChatML string for model prompting.
pub fn format_chatml(
    messages: &[ChatMessage],
    tools_json: Option<&str>,
    enable_thinking: bool,
) -> String {
    let mut out = String::new();

    // Check if system message exists
    let has_system = messages.iter().any(|m| m.role == Role::System);

    if !has_system && (tools_json.is_some() || enable_thinking) {
        out.push_str(&format!("{}system\n", BOS_TOKEN));
        out.push_str("You are Mivi-v4, a fast, reliable, tool-using AI agent.");
        if let Some(tools) = tools_json {
            out.push_str(&format!("\n{}\n{}\n{}\n", TOOLS_DEF_START, tools, TOOLS_DEF_END));
            out.push_str("To invoke a tool, output:\n<tool_call>{\"name\":\"tool_name\",\"arguments\":{...}}</tool_call>\n");
        }
        if enable_thinking {
            out.push_str("\nThink concisely inside <think>...</think> before taking actions or answering.\n");
        }
        out.push_str(&format!("{}\n", EOS_TOKEN));
    }

    for msg in messages {
        out.push_str(&format!("{}{}\n", BOS_TOKEN, msg.role));
        if msg.role == Role::System {
            if let Some(content) = &msg.content {
                out.push_str(content);
            }
            if let Some(tools) = tools_json {
                out.push_str(&format!("\n{}\n{}\n{}\n", TOOLS_DEF_START, tools, TOOLS_DEF_END));
                out.push_str("To invoke a tool, output:\n<tool_call>{\"name\":\"tool_name\",\"arguments\":{...}}</tool_call>\n");
            }
            if enable_thinking {
                out.push_str("\nThink concisely inside <think>...</think> before taking actions or answering.\n");
            }
        } else {
            if let Some(content) = &msg.content {
                out.push_str(content);
            }
        }
        out.push_str(&format!("{}\n", EOS_TOKEN));
    }

    // Append assistant prompt header
    out.push_str(&format!("{}assistant\n", BOS_TOKEN));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_chatml_with_tools_and_thinking() {
        let messages = vec![ChatMessage {
            role: Role::User,
            content: Some("Search the web".to_string()),
            name: None,
        }];

        let tools = r#"[{"type":"function","function":{"name":"web_search"}}]"#;
        let formatted = format_chatml(&messages, Some(tools), true);

        assert!(formatted.contains("<|im_start|>system"));
        assert!(formatted.contains("<tools>"));
        assert!(formatted.contains("<think>"));
        assert!(formatted.contains("<|im_start|>user\nSearch the web"));
        assert!(formatted.ends_with("<|im_start|>assistant\n"));
    }
}
