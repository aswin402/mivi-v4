//! Parser for <tool_call> and <think> markup in generated tokens using zero-allocation LazyLock regexes.

use crate::schema::ToolCall;
use regex::Regex;
use serde_json::{Map, Value};
use std::sync::LazyLock;

// Compile-time verified regex patterns for agent markup extraction
static TOOL_CALL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"<tool_call>([\s\S]*?)</tool_call>")
        .expect("Valid regex literal for tool call parser")
});

static THINKING_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"<think>([\s\S]*?)</think>").expect("Valid regex literal for thinking parser")
});

/// A model-specific decoder for structured tool calls.
///
/// The server selects a codec through the loaded model profile. Keeping this interface in the
/// tool layer prevents model-specific delimiters from leaking into HTTP routes or the agent loop.
pub trait ToolCallCodec: Send + Sync {
    fn extract(&self, text: &str) -> Result<Vec<ToolCall>, String>;
    fn strip(&self, text: &str) -> String;

    /// Return the opening marker when output can be classified incrementally.
    ///
    /// Codecs without a stable marker return `None`; callers must then wait for the complete
    /// generation before deciding whether the output is text or a tool call.
    fn opening_delimiter(&self) -> Option<&str> {
        None
    }

    /// Return the currently parseable prefix of a structured tool call.
    fn stream_update(&self, _text: &str) -> Result<Option<ToolCallStreamUpdate>, String> {
        Ok(None)
    }
}

/// A codec-neutral snapshot used to produce OpenAI tool-call deltas.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallStreamUpdate {
    pub index: usize,
    pub name: String,
    /// A JSON object prefix that only grows as more model output becomes parseable.
    pub arguments: String,
    pub complete: bool,
}

/// Decode the legacy Mivi XML/JSON tool-call format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LegacyJsonXmlToolCallCodec;

impl ToolCallCodec for LegacyJsonXmlToolCallCodec {
    fn extract(&self, text: &str) -> Result<Vec<ToolCall>, String> {
        Ok(extract_tool_calls(text))
    }

    fn strip(&self, text: &str) -> String {
        strip_tool_calls(text)
    }

    fn opening_delimiter(&self) -> Option<&str> {
        Some("<tool_call>")
    }
}

/// Decode a Python-style list of keyword-argument calls between configured delimiters.
///
/// Example input: `<call>[read_file(path="README.md")]</call>`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DelimitedPythonToolCallCodec {
    start: String,
    end: String,
}

impl DelimitedPythonToolCallCodec {
    pub fn new(start: impl Into<String>, end: impl Into<String>) -> Self {
        Self {
            start: start.into(),
            end: end.into(),
        }
    }
}

impl ToolCallCodec for DelimitedPythonToolCallCodec {
    fn extract(&self, text: &str) -> Result<Vec<ToolCall>, String> {
        if self.start.is_empty() || self.end.is_empty() {
            return Err("tool-call delimiters must not be empty".to_string());
        }

        let mut calls = Vec::new();
        let mut search_from = 0;
        while let Some(relative_start) = text[search_from..].find(&self.start) {
            let start = search_from + relative_start;
            let body_start = start + self.start.len();
            let relative_end = text[body_start..]
                .find(&self.end)
                .ok_or_else(|| "unterminated structured tool call".to_string())?;
            let body_end = body_start + relative_end;
            calls.extend(parse_python_tool_calls(&text[body_start..body_end])?);
            search_from = body_end + self.end.len();
        }
        Ok(calls)
    }

    fn strip(&self, text: &str) -> String {
        if self.start.is_empty() || self.end.is_empty() {
            return text.trim().to_string();
        }

        let mut output = String::with_capacity(text.len());
        let mut search_from = 0;
        while let Some(relative_start) = text[search_from..].find(&self.start) {
            let start = search_from + relative_start;
            output.push_str(&text[search_from..start]);
            let body_start = start + self.start.len();
            let Some(relative_end) = text[body_start..].find(&self.end) else {
                break;
            };
            search_from = body_start + relative_end + self.end.len();
        }
        output.push_str(&text[search_from..]);
        output.trim().to_string()
    }

    fn opening_delimiter(&self) -> Option<&str> {
        (!self.start.is_empty()).then_some(self.start.as_str())
    }

    fn stream_update(&self, text: &str) -> Result<Option<ToolCallStreamUpdate>, String> {
        stream_python_tool_call_update(text, &self.start, &self.end)
    }
}

fn stream_python_tool_call_update(
    text: &str,
    start_delimiter: &str,
    end_delimiter: &str,
) -> Result<Option<ToolCallStreamUpdate>, String> {
    if start_delimiter.is_empty() || end_delimiter.is_empty() {
        return Ok(None);
    }

    let Some(start) = text.find(start_delimiter) else {
        return Ok(None);
    };
    let body_start = start + start_delimiter.len();
    let end = text[body_start..]
        .find(end_delimiter)
        .map(|relative_end| body_start + relative_end);
    let body_end = end.unwrap_or(text.len());
    let mut body = text[body_start..body_end].trim();
    if let Some(list_body) = body.strip_prefix('[') {
        body = list_body.trim_start();
    }

    let Some(open_paren) = body.find('(') else {
        return Ok(None);
    };
    let name = body[..open_paren].trim();
    if !is_tool_identifier(name) {
        return Ok(None);
    }

    let argument_source = &body[open_paren + 1..];
    let closing_paren = find_call_close(argument_source);
    let argument_end = closing_paren.unwrap_or(argument_source.len());
    let argument_source = &argument_source[..argument_end];
    let mut fields = Vec::new();
    let mut all_fields_complete = true;

    for segment in split_top_level_arguments(argument_source) {
        let segment = segment.trim();
        if segment.is_empty() {
            continue;
        }
        let Some(equal_index) = find_top_level_equals(segment) else {
            all_fields_complete = false;
            continue;
        };
        let key = segment[..equal_index].trim();
        let value_source = segment[equal_index + 1..].trim();
        if !is_tool_identifier(key) {
            all_fields_complete = false;
            continue;
        }
        match parse_complete_python_value(value_source) {
            Ok(value) => fields.push((key.to_string(), value)),
            Err(_) => all_fields_complete = false,
        }
    }

    let call_closed = closing_paren.is_some();
    let arguments = format_json_argument_prefix(&fields, call_closed && all_fields_complete)?;

    Ok(Some(ToolCallStreamUpdate {
        index: 0,
        name: name.to_string(),
        arguments,
        complete: end.is_some() && call_closed && all_fields_complete,
    }))
}

fn is_tool_identifier(value: &str) -> bool {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first.is_ascii_alphabetic() || first == '_')
        && chars.all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.')
        })
}

fn find_call_close(input: &str) -> Option<usize> {
    let mut depth = 1usize;
    let mut quote = None;
    let mut escaped = false;

    for (index, character) in input.char_indices() {
        if let Some(active_quote) = quote {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == active_quote {
                quote = None;
            }
            continue;
        }
        match character {
            '"' | '\'' => quote = Some(character),
            '(' => depth += 1,
            ')' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
    }
    None
}

fn split_top_level_arguments(input: &str) -> Vec<&str> {
    let mut segments = Vec::new();
    let mut segment_start = 0;
    let mut depth = 0usize;
    let mut quote = None;
    let mut escaped = false;

    for (index, character) in input.char_indices() {
        if let Some(active_quote) = quote {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == active_quote {
                quote = None;
            }
            continue;
        }
        match character {
            '"' | '\'' => quote = Some(character),
            '[' | '{' | '(' => depth += 1,
            ']' | '}' | ')' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                segments.push(&input[segment_start..index]);
                segment_start = index + character.len_utf8();
            }
            _ => {}
        }
    }
    segments.push(&input[segment_start..]);
    segments
}

fn find_top_level_equals(input: &str) -> Option<usize> {
    let mut depth = 0usize;
    let mut quote = None;
    let mut escaped = false;

    for (index, character) in input.char_indices() {
        if let Some(active_quote) = quote {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == active_quote {
                quote = None;
            }
            continue;
        }
        match character {
            '"' | '\'' => quote = Some(character),
            '[' | '{' | '(' => depth += 1,
            ']' | '}' | ')' => depth = depth.saturating_sub(1),
            '=' if depth == 0 => return Some(index),
            _ => {}
        }
    }
    None
}

fn format_json_argument_prefix(fields: &[(String, Value)], close: bool) -> Result<String, String> {
    if fields.is_empty() {
        return Ok(if close {
            "{}".to_string()
        } else {
            String::new()
        });
    }

    let mut output = String::from('{');
    for (index, (key, value)) in fields.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        output.push_str(&serde_json::to_string(key).map_err(|error| error.to_string())?);
        output.push(':');
        output.push_str(&serde_json::to_string(value).map_err(|error| error.to_string())?);
    }
    if close {
        output.push('}');
    }
    Ok(output)
}

fn parse_python_tool_calls(input: &str) -> Result<Vec<ToolCall>, String> {
    let mut parser = PythonValueParser::new(input);
    parser.skip_whitespace();

    let mut calls = Vec::new();
    if parser.consume('[') {
        parser.skip_whitespace();
        if !parser.consume(']') {
            loop {
                calls.push(parser.parse_call()?);
                parser.skip_whitespace();
                if parser.consume(']') {
                    break;
                }
                parser.expect(',')?;
                parser.skip_whitespace();
                if parser.consume(']') {
                    break;
                }
            }
        }
    } else {
        calls.push(parser.parse_call()?);
    }

    parser.skip_whitespace();
    if !parser.is_eof() {
        return Err(format!(
            "unexpected trailing characters in structured tool call at position {}",
            parser.position()
        ));
    }
    Ok(calls)
}

struct PythonValueParser<'a> {
    chars: Vec<char>,
    position: usize,
    _source: &'a str,
}

impl<'a> PythonValueParser<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            chars: source.chars().collect(),
            position: 0,
            _source: source,
        }
    }

    fn position(&self) -> usize {
        self.position
    }

    fn is_eof(&self) -> bool {
        self.position >= self.chars.len()
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.position).copied()
    }

    fn consume(&mut self, expected: char) -> bool {
        if self.peek() == Some(expected) {
            self.position += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, expected: char) -> Result<(), String> {
        if self.consume(expected) {
            Ok(())
        } else {
            Err(format!(
                "expected '{expected}' at position {}",
                self.position
            ))
        }
    }

    fn skip_whitespace(&mut self) {
        while self.peek().is_some_and(char::is_whitespace) {
            self.position += 1;
        }
    }

    fn parse_call(&mut self) -> Result<ToolCall, String> {
        let name = self.parse_identifier("tool name")?;
        self.skip_whitespace();
        self.expect('(')?;
        let arguments = self.parse_arguments()?;
        Ok(ToolCall::new(name, Value::Object(arguments)))
    }

    fn parse_arguments(&mut self) -> Result<Map<String, Value>, String> {
        let mut arguments = Map::new();
        self.skip_whitespace();
        if self.consume(')') {
            return Ok(arguments);
        }

        loop {
            let key = self.parse_identifier("keyword argument name")?;
            self.skip_whitespace();
            self.expect('=')?;
            self.skip_whitespace();
            let value = self.parse_value()?;
            arguments.insert(key, value);
            self.skip_whitespace();

            if self.consume(')') {
                return Ok(arguments);
            }
            self.expect(',')?;
            self.skip_whitespace();
            if self.consume(')') {
                return Ok(arguments);
            }
        }
    }

    fn parse_value(&mut self) -> Result<Value, String> {
        self.skip_whitespace();
        match self.peek() {
            Some('"') | Some('\'') => self.parse_string().map(Value::String),
            Some('[') => self.parse_array(),
            Some('{') => self.parse_object(),
            Some('-') | Some('0'..='9') => self.parse_number(),
            Some(_) => {
                let literal = self.parse_identifier("literal")?;
                match literal.as_str() {
                    "True" | "true" => Ok(Value::Bool(true)),
                    "False" | "false" => Ok(Value::Bool(false)),
                    "None" | "null" => Ok(Value::Null),
                    _ => Err(format!(
                        "unsupported literal '{literal}' at position {}",
                        self.position
                    )),
                }
            }
            None => Err("expected a tool argument value, found end of input".to_string()),
        }
    }

    fn parse_string(&mut self) -> Result<String, String> {
        let quote = self
            .peek()
            .filter(|character| *character == '"' || *character == '\'')
            .ok_or_else(|| format!("expected string at position {}", self.position))?;
        self.position += 1;
        let mut output = String::new();

        while let Some(character) = self.peek() {
            self.position += 1;
            if character == quote {
                return Ok(output);
            }
            if character != '\\' {
                output.push(character);
                continue;
            }

            let escaped = self
                .peek()
                .ok_or_else(|| "unterminated string escape".to_string())?;
            self.position += 1;
            match escaped {
                'n' => output.push('\n'),
                'r' => output.push('\r'),
                't' => output.push('\t'),
                'b' => output.push('\u{0008}'),
                'f' => output.push('\u{000c}'),
                '\\' => output.push('\\'),
                '/' => output.push('/'),
                '\'' => output.push('\''),
                '"' => output.push('"'),
                'u' => {
                    let code = self.parse_unicode_escape()?;
                    output.push(code);
                }
                other => {
                    return Err(format!(
                        "unsupported string escape '\\{other}' at position {}",
                        self.position.saturating_sub(1)
                    ));
                }
            }
        }

        Err("unterminated string in structured tool call".to_string())
    }

    fn parse_unicode_escape(&mut self) -> Result<char, String> {
        if self.position + 4 > self.chars.len() {
            return Err("incomplete unicode escape".to_string());
        }
        let digits: String = self.chars[self.position..self.position + 4]
            .iter()
            .collect();
        self.position += 4;
        let code = u32::from_str_radix(&digits, 16)
            .map_err(|_| format!("invalid unicode escape \\u{digits}"))?;
        char::from_u32(code).ok_or_else(|| format!("invalid unicode scalar \\u{digits}"))
    }

    fn parse_number(&mut self) -> Result<Value, String> {
        let start = self.position;
        while self.peek().is_some_and(|character| {
            character.is_ascii_digit() || matches!(character, '-' | '+' | '.' | 'e' | 'E')
        }) {
            self.position += 1;
        }
        let raw: String = self.chars[start..self.position].iter().collect();
        serde_json::from_str(&raw)
            .map_err(|error| format!("invalid numeric argument '{raw}': {error}"))
    }

    fn parse_array(&mut self) -> Result<Value, String> {
        self.expect('[')?;
        let mut values = Vec::new();
        self.skip_whitespace();
        if self.consume(']') {
            return Ok(Value::Array(values));
        }
        loop {
            values.push(self.parse_value()?);
            self.skip_whitespace();
            if self.consume(']') {
                return Ok(Value::Array(values));
            }
            self.expect(',')?;
            self.skip_whitespace();
            if self.consume(']') {
                return Ok(Value::Array(values));
            }
        }
    }

    fn parse_object(&mut self) -> Result<Value, String> {
        self.expect('{')?;
        let mut object = Map::new();
        self.skip_whitespace();
        if self.consume('}') {
            return Ok(Value::Object(object));
        }
        loop {
            let key = match self.peek() {
                Some('"') | Some('\'') => self.parse_string()?,
                Some(_) => self.parse_identifier("object key")?,
                None => return Err("expected object key, found end of input".to_string()),
            };
            self.skip_whitespace();
            self.expect(':')?;
            self.skip_whitespace();
            object.insert(key, self.parse_value()?);
            self.skip_whitespace();
            if self.consume('}') {
                return Ok(Value::Object(object));
            }
            self.expect(',')?;
            self.skip_whitespace();
            if self.consume('}') {
                return Ok(Value::Object(object));
            }
        }
    }

    fn parse_identifier(&mut self, label: &str) -> Result<String, String> {
        let start = self.position;
        while self.peek().is_some_and(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.')
        }) {
            self.position += 1;
        }
        if start == self.position {
            return Err(format!("expected {label} at position {}", self.position));
        }
        Ok(self.chars[start..self.position].iter().collect())
    }
}

fn parse_complete_python_value(source: &str) -> Result<Value, String> {
    let mut parser = PythonValueParser::new(source);
    parser.skip_whitespace();
    let value = parser.parse_value()?;
    parser.skip_whitespace();
    if parser.is_eof() {
        Ok(value)
    } else {
        Err(format!(
            "unexpected trailing characters in Python value at position {}",
            parser.position()
        ))
    }
}

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

    #[test]
    fn configured_delimited_codec_parses_pythonic_nested_arguments() {
        let codec = DelimitedPythonToolCallCodec::new("<call>", "</call>");
        let text = r#"<call>[read_file(path="README.md", lines=[1, 2], options={"encoding": "utf-8"})]</call>"#;

        let calls = codec.extract(text).expect("valid Python-style tool call");

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].arguments["path"], "README.md");
        assert_eq!(calls[0].arguments["lines"][1], 2);
        assert_eq!(calls[0].arguments["options"]["encoding"], "utf-8");
    }

    #[test]
    fn configured_delimited_codec_emits_monotonic_argument_deltas() {
        let codec = DelimitedPythonToolCallCodec::new("<call>", "</call>");

        let first = codec
            .stream_update(r#"<call>[read_file(path="README.md""#)
            .expect("partial tool call should be accepted")
            .expect("tool name should be available");
        assert_eq!(first.name, "read_file");
        assert_eq!(first.arguments, r#"{"path":"README.md""#);
        assert!(!first.complete);

        let complete = codec
            .stream_update(r#"<call>[read_file(path="README.md", lines=[1, 2])]</call>"#)
            .expect("complete tool call should be accepted")
            .expect("tool name should be available");
        assert_eq!(complete.arguments, r#"{"path":"README.md","lines":[1,2]}"#);
        assert!(complete.complete);
    }
}
