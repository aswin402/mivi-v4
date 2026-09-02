//! Parser for <tool_call> and <think> markup in generated tokens using zero-allocation LazyLock regexes.

use crate::schema::ToolCall;
use regex::Regex;
use std::sync::LazyLock;

// Compile-time verified regex patterns for agent markup extraction
static TOOL_CALL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"<tool_call>([\s\S]*?)</tool_call>")
        .expect("Valid regex literal for tool call parser")
});

static THINKING_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"<think>([\s\S]*?)</think>").expect("Valid regex literal for thinking parser")
});

pub fn extract_tool_calls(text: &str) -> Vec<ToolCall> {
    let mut tool_calls = Vec::new();

    for cap in TOOL_CALL_RE.captures_iter(text) {
        if let Some(json_match) = cap.get(1) {
            let raw_str = json_match.as_str().trim();
            match serde_json::from_str::<serde_json::Value>(raw_str) {
                Ok(tc) => {
                    if let (Some(name), Some(args)) = (tc.get("name"), tc.get("arguments")) {
                        if let Some(name_str) = name.as_str() {
                            tool_calls.push(ToolCall::new(name_str, args.clone()));
                        } else {
                            tool_calls.push(ToolCall::parse_error(
                                "Tool call 'name' field must be a string",
                                raw_str,
                            ));
                        }
                    } else {
                        tool_calls.push(ToolCall::parse_error(
                            "Tool call JSON must contain 'name' and 'arguments' fields",
                            raw_str,
                        ));
                    }
                }
                Err(e) => {
                    tool_calls.push(ToolCall::parse_error(
                        &format!("Invalid JSON in <tool_call>: {}", e),
                        raw_str,
                    ));
                }
            }
        }
    }
    tool_calls
}

pub fn extract_thinking(text: &str) -> Option<String> {
    if let Some(cap) = THINKING_RE.captures(text) {
        cap.get(1).map(|m| m.as_str().trim().to_string())
    } else if let Some(idx) = text.find("<think>") {
        let after = &text[idx + 7..];
        if !after.trim().is_empty() {
            Some(after.trim().to_string())
        } else {
            None
        }
    } else {
        None
    }
}

/// Strip <tool_call>...</tool_call> tags from text.
pub fn strip_tool_calls(text: &str) -> String {
    let without_calls = TOOL_CALL_RE.replace_all(text, "");
    without_calls.trim().to_string()
}

/// Strip <think>...</think> tags (including unclosed <think>) from text.
pub fn strip_thinking(text: &str) -> String {
    let without_think = THINKING_RE.replace_all(text, "");
    if without_think.contains("<think>") {
        if let Some(idx) = without_think.find("<think>") {
            without_think[..idx].trim().to_string()
        } else {
            without_think.trim().to_string()
        }
    } else {
        without_think.trim().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_tool_calls() {
        let text = r#"I will search the web.
<tool_call>
{"name": "web_search", "arguments": {"query": "Rust 2026"}}
</tool_call>
Finished."#;

        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "web_search");
        assert_eq!(calls[0].arguments["query"], "Rust 2026");
    }

    #[test]
    fn test_extract_thinking() {
        let text = "<think>I should inspect the repository first.</think>\nHello world!";
        let think = extract_thinking(text);
        assert_eq!(
            think.as_deref(),
            Some("I should inspect the repository first.")
        );
    }
}
