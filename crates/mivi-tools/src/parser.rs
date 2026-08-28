//! Parser for <tool_call> and <think> markup in generated tokens using zero-allocation LazyLock regexes.

use crate::schema::ToolCall;
use regex::Regex;
use std::sync::LazyLock;

static TOOL_CALL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"<tool_call>([\s\S]*?)</tool_call>").expect("Invalid tool call regex pattern")
});

static THINKING_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"<think>([\s\S]*?)</think>").expect("Invalid thinking regex pattern")
});

pub fn extract_tool_calls(text: &str) -> Vec<ToolCall> {
    let mut tool_calls = Vec::new();

    for cap in TOOL_CALL_RE.captures_iter(text) {
        if let Some(json_str) = cap.get(1) {
            if let Ok(tc) = serde_json::from_str::<serde_json::Value>(json_str.as_str().trim()) {
                if let (Some(name), Some(args)) = (tc.get("name"), tc.get("arguments")) {
                    if let Some(name_str) = name.as_str() {
                        tool_calls.push(ToolCall {
                            name: name_str.to_string(),
                            arguments: args.clone(),
                        });
                    }
                }
            }
        }
    }
    tool_calls
}

pub fn extract_thinking(text: &str) -> Option<String> {
    THINKING_RE
        .captures(text)
        .and_then(|cap| cap.get(1))
        .map(|m| m.as_str().trim().to_string())
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
