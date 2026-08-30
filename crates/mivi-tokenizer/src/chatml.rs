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

impl std::str::FromStr for Role {
    type Err = std::convert::Infallible;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "system" => Ok(Self::System),
            "assistant" => Ok(Self::Assistant),
            "tool" => Ok(Self::Tool),
            _ => Ok(Self::User),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: Option<String>,
    pub name: Option<String>,
}

pub use mivi_core::DEFAULT_SYSTEM_PROMPT;

pub const CHATML_MSG_SIZE_ESTIMATE: usize = 128;
pub const CHATML_OVERHEAD_ESTIMATE: usize = 256;

/// Formats OpenAI-style messages into ChatML string for model prompting.
pub fn format_chatml(
    messages: &[ChatMessage],
    tools_json: Option<&str>,
    enable_thinking: bool,
) -> String {
    use std::fmt::Write;

    let estimated_cap = messages.len() * CHATML_MSG_SIZE_ESTIMATE + CHATML_OVERHEAD_ESTIMATE;
    let mut out = String::with_capacity(estimated_cap);

    // Check if system message exists
    let has_system = messages.iter().any(|m| m.role == Role::System);

    if !has_system && (tools_json.is_some() || enable_thinking) {
        writeln!(out, "{}system", BOS_TOKEN).unwrap();
        out.push_str(DEFAULT_SYSTEM_PROMPT);
        append_tool_and_thinking_instructions(&mut out, tools_json, enable_thinking);
        writeln!(out, "{}", EOS_TOKEN).unwrap();
    }

    for msg in messages {
        if let Some(name) = &msg.name {
            writeln!(out, "{}{}:{name}", BOS_TOKEN, msg.role).unwrap();
        } else {
            writeln!(out, "{}{}", BOS_TOKEN, msg.role).unwrap();
        }

        if let Some(content) = &msg.content {
            out.push_str(content);
        }

        if msg.role == Role::System {
            append_tool_and_thinking_instructions(&mut out, tools_json, enable_thinking);
        }

        writeln!(out, "{}", EOS_TOKEN).unwrap();
    }

    // Append assistant prompt header
    writeln!(out, "{}assistant", BOS_TOKEN).unwrap();
    out
}

fn append_tool_and_thinking_instructions(
    out: &mut String,
    tools_json: Option<&str>,
    enable_thinking: bool,
) {
    use std::fmt::Write;

    if let Some(tools) = tools_json {
        writeln!(out, "\n{}\n{}\n{}", TOOLS_DEF_START, tools, TOOLS_DEF_END).unwrap();
        writeln!(
            out,
            "To invoke a tool, output:\n{}{{\"name\":\"tool_name\",\"arguments\":{{...}}}}{}",
            TOOL_CALL_START, TOOL_CALL_END
        )
        .unwrap();
    }
    if enable_thinking {
        writeln!(
            out,
            "\nThink concisely inside {}...{} before taking actions or answering.",
            THINK_START, THINK_END
        )
        .unwrap();
    }
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
