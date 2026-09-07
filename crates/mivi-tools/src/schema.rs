//! Tool schema representations matching JSON Schema & OpenAI tools spec.

use serde::{Deserialize, Serialize};
use serde_json::Value;

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

#[cfg(test)]
mod validation_tests {
    use super::validate_tool_arguments;
    use serde_json::json;

    #[test]
    fn accepts_arguments_matching_a_schema() {
        let schema = json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "line": {"type": "integer", "minimum": 1}
            },
            "required": ["path"]
        });
        let arguments = json!({"path": "README.md", "line": 3});

        assert!(validate_tool_arguments(&schema, &arguments).is_ok());
    }

    #[test]
    fn rejects_missing_required_arguments() {
        let schema = json!({
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
        });

        let error = validate_tool_arguments(&schema, &json!({})).unwrap_err();
        assert!(error.contains("path"));
    }

    #[test]
    fn rejects_wrong_argument_types() {
        let schema = json!({
            "type": "object",
            "properties": {"path": {"type": "string"}}
        });

        let error = validate_tool_arguments(&schema, &json!({"path": 42})).unwrap_err();
        assert!(error.contains("string"));
    }

    #[test]
    fn rejects_unknown_arguments_when_schema_disallows_them() {
        let schema = json!({
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "additionalProperties": false
        });

        let error =
            validate_tool_arguments(&schema, &json!({"path": "README.md", "unexpected": true}))
                .unwrap_err();
        assert!(error.contains("unexpected"));
    }

    #[test]
    fn supports_nested_arrays_and_enums() {
        let schema = json!({
            "type": "object",
            "properties": {
                "mode": {"enum": ["read", "write"]},
                "paths": {"type": "array", "items": {"type": "string"}}
            }
        });

        assert!(
            validate_tool_arguments(&schema, &json!({"mode": "read", "paths": ["a", "b"]})).is_ok()
        );
        assert!(
            validate_tool_arguments(&schema, &json!({"mode": "delete", "paths": ["a"]})).is_err()
        );
    }

    #[test]
    fn empty_schema_accepts_any_json_value() {
        assert!(validate_tool_arguments(&json!({}), &json!(null)).is_ok());
        assert!(validate_tool_arguments(&json!({}), &json!("anything")).is_ok());
    }
}

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

const MAX_SCHEMA_DEPTH: usize = 64;

/// Validate generated tool arguments against the JSON Schema supplied by the caller.
///
/// This intentionally implements the bounded subset used by common OpenAI and Anthropic
/// tool definitions: types, object properties/required/additionalProperties, array items,
/// enum/const, and basic string/array/number limits. Unknown schema keywords are ignored so
/// a provider-specific annotation cannot make an otherwise valid tool call fail.
pub fn validate_tool_arguments(schema: &Value, arguments: &Value) -> Result<(), String> {
    validate_schema_value(schema, arguments, "$", 0)
}

fn validate_schema_value(
    schema: &Value,
    value: &Value,
    path: &str,
    depth: usize,
) -> Result<(), String> {
    if depth > MAX_SCHEMA_DEPTH {
        return Err(format!("{path}: tool schema exceeds maximum nesting depth"));
    }
    let schema_object = schema
        .as_object()
        .ok_or_else(|| format!("{path}: tool schema must be a JSON object"))?;

    if let Some(expected) = schema_object.get("type") {
        let matches = match expected {
            Value::String(kind) => json_type_matches(kind, value),
            Value::Array(kinds) => kinds
                .iter()
                .filter_map(Value::as_str)
                .any(|kind| json_type_matches(kind, value)),
            _ => false,
        };
        if !matches {
            return Err(format!(
                "{path}: expected JSON type {}, got {}",
                format_schema_type(expected),
                json_value_type(value)
            ));
        }
    }

    if let Some(enum_values) = schema_object.get("enum") {
        let values = enum_values
            .as_array()
            .ok_or_else(|| format!("{path}: schema enum must be an array"))?;
        if !values.iter().any(|candidate| candidate == value) {
            return Err(format!(
                "{path}: value is not one of the permitted enum values"
            ));
        }
    }
    if let Some(constant) = schema_object.get("const") {
        if constant != value {
            return Err(format!("{path}: value does not match schema const"));
        }
    }

    match value {
        Value::Object(object) => validate_object(schema_object, object, path, depth)?,
        Value::Array(array) => validate_array(schema_object, array, path, depth)?,
        Value::String(string) => validate_string(schema_object, string, path)?,
        Value::Number(number) => validate_number(schema_object, number, path)?,
        Value::Null | Value::Bool(_) => {}
    }
    Ok(())
}

fn validate_object(
    schema: &serde_json::Map<String, Value>,
    object: &serde_json::Map<String, Value>,
    path: &str,
    depth: usize,
) -> Result<(), String> {
    if let Some(required) = schema.get("required") {
        let required = required
            .as_array()
            .ok_or_else(|| format!("{path}: schema required must be an array"))?;
        for key in required {
            let key = key
                .as_str()
                .ok_or_else(|| format!("{path}: schema required entries must be strings"))?;
            if !object.contains_key(key) {
                return Err(format!("{path}: missing required argument '{key}'"));
            }
        }
    }

    let properties = match schema.get("properties") {
        None => None,
        Some(properties) => Some(
            properties
                .as_object()
                .ok_or_else(|| format!("{path}: schema properties must be an object"))?,
        ),
    };

    for (key, value) in object {
        let child_path = format_path(path, key);
        match properties.and_then(|items| items.get(key)) {
            Some(property_schema) => {
                validate_schema_value(property_schema, value, &child_path, depth + 1)?;
            }
            None if schema.get("additionalProperties") == Some(&Value::Bool(false)) => {
                return Err(format!("{child_path}: unknown argument"));
            }
            None => {}
        }
    }

    if let Some(minimum) = schema.get("minProperties").and_then(Value::as_u64) {
        if object.len() < minimum as usize {
            return Err(format!("{path}: expected at least {minimum} properties"));
        }
    }
    if let Some(maximum) = schema.get("maxProperties").and_then(Value::as_u64) {
        if object.len() > maximum as usize {
            return Err(format!("{path}: expected at most {maximum} properties"));
        }
    }
    Ok(())
}

fn validate_array(
    schema: &serde_json::Map<String, Value>,
    array: &[Value],
    path: &str,
    depth: usize,
) -> Result<(), String> {
    if let Some(minimum) = schema.get("minItems").and_then(Value::as_u64) {
        if array.len() < minimum as usize {
            return Err(format!("{path}: expected at least {minimum} items"));
        }
    }
    if let Some(maximum) = schema.get("maxItems").and_then(Value::as_u64) {
        if array.len() > maximum as usize {
            return Err(format!("{path}: expected at most {maximum} items"));
        }
    }
    if let Some(item_schema) = schema.get("items") {
        if !item_schema.is_object() {
            return Err(format!("{path}: schema items must be an object"));
        }
        for (index, item) in array.iter().enumerate() {
            validate_schema_value(item_schema, item, &format!("{path}[{index}]"), depth + 1)?;
        }
    }
    Ok(())
}

fn validate_string(
    schema: &serde_json::Map<String, Value>,
    string: &str,
    path: &str,
) -> Result<(), String> {
    let length = string.chars().count();
    if let Some(minimum) = schema.get("minLength").and_then(Value::as_u64) {
        if length < minimum as usize {
            return Err(format!(
                "{path}: string is shorter than {minimum} characters"
            ));
        }
    }
    if let Some(maximum) = schema.get("maxLength").and_then(Value::as_u64) {
        if length > maximum as usize {
            return Err(format!(
                "{path}: string is longer than {maximum} characters"
            ));
        }
    }
    Ok(())
}

fn validate_number(
    schema: &serde_json::Map<String, Value>,
    number: &serde_json::Number,
    path: &str,
) -> Result<(), String> {
    let Some(value) = number.as_f64() else {
        return Ok(());
    };
    if let Some(minimum) = schema.get("minimum").and_then(Value::as_f64) {
        if value < minimum {
            return Err(format!("{path}: number is less than {minimum}"));
        }
    }
    if let Some(maximum) = schema.get("maximum").and_then(Value::as_f64) {
        if value > maximum {
            return Err(format!("{path}: number is greater than {maximum}"));
        }
    }
    if let Some(exclusive_minimum) = schema.get("exclusiveMinimum").and_then(Value::as_f64) {
        if value <= exclusive_minimum {
            return Err(format!(
                "{path}: number must be greater than {exclusive_minimum}"
            ));
        }
    }
    if let Some(exclusive_maximum) = schema.get("exclusiveMaximum").and_then(Value::as_f64) {
        if value >= exclusive_maximum {
            return Err(format!(
                "{path}: number must be less than {exclusive_maximum}"
            ));
        }
    }
    Ok(())
}

fn json_type_matches(kind: &str, value: &Value) -> bool {
    match kind {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => {
            value.as_i64().is_some()
                || value.as_u64().is_some()
                || value
                    .as_f64()
                    .is_some_and(|number| number.is_finite() && number.fract() == 0.0)
        }
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        _ => false,
    }
}

fn json_value_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn format_schema_type(value: &Value) -> String {
    match value {
        Value::String(kind) => format!("'{kind}'"),
        Value::Array(kinds) => kinds
            .iter()
            .filter_map(Value::as_str)
            .map(|kind| format!("'{kind}'"))
            .collect::<Vec<_>>()
            .join(" or "),
        _ => "a valid type declaration".to_string(),
    }
}

fn format_path(path: &str, key: &str) -> String {
    if key
        .chars()
        .all(|character| character.is_ascii_alphanumeric() || character == '_')
    {
        format!("{path}.{key}")
    } else {
        format!("{path}[{key:?}]")
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
