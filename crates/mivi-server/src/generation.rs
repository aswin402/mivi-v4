#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_openai_stop_string_and_array() {
        assert_eq!(
            parse_stop_sequences(Some(&json!("END"))).unwrap(),
            Some(vec!["END".to_string()])
        );
        assert_eq!(
            parse_stop_sequences(Some(&json!(["END", "DONE"]))).unwrap(),
            Some(vec!["END".to_string(), "DONE".to_string()])
        );
    }

    #[test]
    fn rejects_invalid_sampling_ranges() {
        assert!(validate_sampling_parameters(Some(-0.1), None, None, None).is_err());
        assert!(validate_sampling_parameters(None, Some(0.0), None, None).is_err());
        assert!(validate_sampling_parameters(None, Some(1.1), None, None).is_err());
        assert!(validate_sampling_parameters(None, None, Some(2.1), None).is_err());
        assert!(validate_additional_sampling_parameters(Some(0), None, None).is_err());
        assert!(validate_additional_sampling_parameters(None, Some(1.1), None).is_err());
        assert!(validate_additional_sampling_parameters(None, None, Some(0.0)).is_err());
    }

    #[test]
    fn parses_supported_response_formats() {
        assert_eq!(parse_response_mode(None).unwrap(), ResponseMode::Text);
        assert_eq!(
            parse_response_mode(Some(&json!({"type": "json_object"}))).unwrap(),
            ResponseMode::JsonObject
        );
        assert!(parse_response_mode(Some(&json!({"type": "json_schema"}))).is_err());
    }

    #[test]
    fn validates_json_object_output() {
        assert!(validate_json_output(r#"{"ok":true}"#).is_ok());
        assert!(validate_json_output("{incomplete").is_err());
    }

    #[test]
    fn validates_tool_choice_against_tools() {
        let tools = vec![json!({
            "type": "function",
            "function": {"name": "read_file"}
        })];

        assert_eq!(
            parse_tool_choice(Some(&tools), Some(&json!("none"))).unwrap(),
            ToolChoice::Disabled
        );
        assert_eq!(
            parse_tool_choice(
                Some(&tools),
                Some(&json!({
                    "type": "function",
                    "function": {"name": "read_file"}
                }))
            )
            .unwrap(),
            ToolChoice::Named("read_file".to_string())
        );
        assert!(parse_tool_choice(
            Some(&tools),
            Some(&json!({
                "type": "function",
                "function": {"name": "missing"}
            }))
        )
        .is_err());
    }

    #[test]
    fn filters_tools_for_a_named_choice_and_disables_none() {
        let tools = vec![
            json!({"type": "function", "function": {"name": "read_file"}}),
            json!({"type": "function", "function": {"name": "list_dir"}}),
        ];
        let named = parse_tool_choice(
            Some(&tools),
            Some(&json!({"type": "function", "function": {"name": "read_file"}})),
        )
        .unwrap();
        assert_eq!(
            filter_tools_for_choice(Some(tools.clone()), &named),
            Some(vec![tools[0].clone()])
        );
        assert_eq!(
            filter_tools_for_choice(Some(tools), &ToolChoice::Disabled),
            None
        );
    }
}
use serde_json::Value;

const MAX_STOP_SEQUENCES: usize = 16;
const MAX_STOP_SEQUENCE_BYTES: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseMode {
    Text,
    JsonObject,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    Disabled,
    Auto,
    Named(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenerationOptions {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub seed: Option<u64>,
    pub stop_tokens: Option<Vec<String>>,
    pub response_mode: ResponseMode,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            temperature: None,
            top_p: None,
            top_k: None,
            min_p: None,
            repetition_penalty: None,
            presence_penalty: None,
            frequency_penalty: None,
            seed: None,
            stop_tokens: None,
            response_mode: ResponseMode::Text,
        }
    }
}

impl ToolChoice {
    pub fn allows_tool_calls(&self, tools: Option<&[Value]>) -> bool {
        !matches!(self, Self::Disabled) && tools.is_some_and(|items| !items.is_empty())
    }
}

/// Parse the OpenAI `stop` field into additional stop sequences.
pub fn parse_stop_sequences(value: Option<&Value>) -> Result<Option<Vec<String>>, String> {
    let Some(value) = value else {
        return Ok(None);
    };

    let values: Vec<&Value> = match value {
        Value::String(_) => vec![value],
        Value::Array(items) => {
            if items.len() > MAX_STOP_SEQUENCES {
                return Err(format!(
                    "stop supports at most {} sequences",
                    MAX_STOP_SEQUENCES
                ));
            }
            items.iter().collect()
        }
        _ => return Err("stop must be a string or an array of strings".to_string()),
    };

    let stops = values
        .into_iter()
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| "stop array entries must be strings".to_string())
                .map(str::to_string)
        })
        .collect::<Result<Vec<_>, _>>()?;
    validate_stop_sequences(&stops)?;
    Ok(Some(stops))
}

/// Validate stop sequences supplied by adapters with a typed request field.
pub fn validate_stop_sequences(stops: &[String]) -> Result<(), String> {
    if stops.len() > MAX_STOP_SEQUENCES {
        return Err(format!(
            "stop supports at most {} sequences",
            MAX_STOP_SEQUENCES
        ));
    }
    for stop in stops {
        if stop.is_empty() {
            return Err("stop sequences cannot be empty".to_string());
        }
        if stop.len() > MAX_STOP_SEQUENCE_BYTES {
            return Err(format!(
                "stop sequences cannot exceed {} bytes",
                MAX_STOP_SEQUENCE_BYTES
            ));
        }
    }
    Ok(())
}

pub fn validate_sampling_parameters(
    temperature: Option<f32>,
    top_p: Option<f32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
) -> Result<(), String> {
    if let Some(value) = temperature {
        if !value.is_finite() || !(0.0..=2.0).contains(&value) {
            return Err("temperature must be finite and between 0 and 2".to_string());
        }
    }
    if let Some(value) = top_p {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) || value == 0.0 {
            return Err("top_p must be finite and greater than 0 and at most 1".to_string());
        }
    }
    for (name, value) in [
        ("presence_penalty", presence_penalty),
        ("frequency_penalty", frequency_penalty),
    ] {
        if let Some(value) = value {
            if !value.is_finite() || !(-2.0..=2.0).contains(&value) {
                return Err(format!("{name} must be finite and between -2 and 2"));
            }
        }
    }
    Ok(())
}

pub fn validate_additional_sampling_parameters(
    top_k: Option<usize>,
    min_p: Option<f32>,
    repetition_penalty: Option<f32>,
) -> Result<(), String> {
    if top_k == Some(0) {
        return Err("top_k must be greater than 0".to_string());
    }
    if let Some(value) = min_p {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err("min_p must be finite and between 0 and 1".to_string());
        }
    }
    if let Some(value) = repetition_penalty {
        if !value.is_finite() || !(0.0..=2.0).contains(&value) || value == 0.0 {
            return Err(
                "repetition_penalty must be finite, greater than 0, and at most 2".to_string(),
            );
        }
    }
    Ok(())
}

pub fn parse_response_mode(value: Option<&Value>) -> Result<ResponseMode, String> {
    let Some(value) = value else {
        return Ok(ResponseMode::Text);
    };
    let object = value
        .as_object()
        .ok_or_else(|| "response_format must be an object".to_string())?;
    match object.get("type").and_then(Value::as_str) {
        Some("text") => Ok(ResponseMode::Text),
        Some("json_object") => Ok(ResponseMode::JsonObject),
        Some("json_schema") => Err("json_schema response format is not supported yet".to_string()),
        Some(other) => Err(format!("unsupported response_format type '{other}'")),
        None => Err("response_format.type is required".to_string()),
    }
}

pub fn validate_json_output(output: &str) -> Result<(), String> {
    if output.trim().is_empty() {
        return Err("model returned empty output for json_object response format".to_string());
    }
    serde_json::from_str::<Value>(output)
        .map(|_| ())
        .map_err(|error| format!("model returned invalid JSON: {error}"))
}

pub fn parse_tool_choice(
    tools: Option<&[Value]>,
    choice: Option<&Value>,
) -> Result<ToolChoice, String> {
    let Some(choice) = choice else {
        return Ok(if tools.is_some_and(|items| !items.is_empty()) {
            ToolChoice::Auto
        } else {
            ToolChoice::Disabled
        });
    };

    match choice {
        Value::String(value) => match value.as_str() {
            "none" => Ok(ToolChoice::Disabled),
            "auto" => Ok(ToolChoice::Auto),
            "required" => Err("tool_choice 'required' is not supported yet".to_string()),
            other => Err(format!("unsupported tool_choice '{other}'")),
        },
        Value::Object(object) => {
            let name = object
                .get("function")
                .and_then(Value::as_object)
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
                .or_else(|| object.get("name").and_then(Value::as_str))
                .ok_or_else(|| "tool_choice function.name is required".to_string())?;
            let exists = tools.is_some_and(|items| {
                items.iter().any(|tool| {
                    tool.get("function")
                        .and_then(Value::as_object)
                        .and_then(|function| function.get("name"))
                        .and_then(Value::as_str)
                        == Some(name)
                })
            });
            if !exists {
                return Err(format!("tool_choice references unknown tool '{name}'"));
            }
            Ok(ToolChoice::Named(name.to_string()))
        }
        _ => Err("tool_choice must be 'none', 'auto', or a function object".to_string()),
    }
}

/// Validate the structural parts of OpenAI function-tool declarations that this server uses.
pub fn validate_openai_tool_definitions(tools: Option<&[Value]>) -> Result<(), String> {
    let Some(tools) = tools else {
        return Ok(());
    };

    for (index, tool) in tools.iter().enumerate() {
        let tool_object = tool
            .as_object()
            .ok_or_else(|| format!("tools[{index}] must be an object"))?;
        if tool_object.get("type").and_then(Value::as_str) != Some("function") {
            return Err(format!("tools[{index}].type must be 'function'"));
        }
        let function = tool_object
            .get("function")
            .and_then(Value::as_object)
            .ok_or_else(|| format!("tools[{index}].function must be an object"))?;
        function
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.trim().is_empty())
            .ok_or_else(|| format!("tools[{index}].function.name must be a non-empty string"))?;
        if let Some(parameters) = function.get("parameters") {
            if !parameters.is_object() {
                return Err(format!(
                    "tools[{index}].function.parameters must be a JSON object"
                ));
            }
        }
    }
    Ok(())
}

pub fn filter_tools_for_choice(
    tools: Option<Vec<Value>>,
    choice: &ToolChoice,
) -> Option<Vec<Value>> {
    match choice {
        ToolChoice::Disabled => None,
        ToolChoice::Auto => tools,
        ToolChoice::Named(name) => tools.map(|items| {
            items
                .into_iter()
                .filter(|tool| {
                    tool.get("function")
                        .and_then(Value::as_object)
                        .and_then(|function| function.get("name"))
                        .and_then(Value::as_str)
                        == Some(name.as_str())
                })
                .collect()
        }),
    }
}

/// Validate model-generated calls against the tool definitions sent by an OpenAI client.
///
/// Tool calls are model output, so a successful JSON parse is not enough: the tool must be
/// declared by the caller and its arguments must satisfy that declaration before the response
/// can expose the call to an executor.
pub fn validate_tool_calls_against_tools(
    calls: &[mivi_tools::ToolCall],
    tools: &[Value],
) -> Result<(), String> {
    for call in calls {
        if !call.arguments.is_object() {
            return Err(format!(
                "Arguments for tool '{}' must be a JSON object",
                call.name
            ));
        }
        let tool = tools.iter().find(|tool| {
            tool.get("function")
                .and_then(Value::as_object)
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
                == Some(call.name.as_str())
        });
        let Some(tool) = tool else {
            return Err(format!(
                "Model generated a call to undeclared tool '{}'",
                call.name
            ));
        };
        let empty_schema = Value::Object(serde_json::Map::new());
        let schema = tool
            .get("function")
            .and_then(Value::as_object)
            .and_then(|function| function.get("parameters"))
            .unwrap_or(&empty_schema);
        mivi_tools::validate_tool_arguments(schema, &call.arguments)
            .map_err(|error| format!("Invalid arguments for tool '{}': {error}", call.name))?;
    }
    Ok(())
}

#[cfg(test)]
mod tool_validation_tests {
    use super::{validate_openai_tool_definitions, validate_tool_calls_against_tools};
    use mivi_tools::ToolCall;
    use serde_json::json;

    #[test]
    fn rejects_calls_to_undeclared_tools() {
        let call = ToolCall::new("missing", json!({}));
        let error = validate_tool_calls_against_tools(
            &[call],
            &[json!({"type": "function", "function": {"name": "known"}})],
        )
        .unwrap_err();
        assert!(error.contains("undeclared"));
    }

    #[test]
    fn rejects_arguments_that_do_not_match_the_declared_schema() {
        let call = ToolCall::new("read_file", json!({"path": 42}));
        let error = validate_tool_calls_against_tools(
            &[call],
            &[json!({
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"]
                    }
                }
            })],
        )
        .unwrap_err();
        assert!(error.contains("read_file"));
        assert!(error.contains("string"));
    }

    #[test]
    fn rejects_non_object_tool_arguments() {
        let call = ToolCall::new("read_file", json!("README.md"));
        let error = validate_tool_calls_against_tools(
            &[call],
            &[json!({
                "type": "function",
                "function": {"name": "read_file", "parameters": {}}
            })],
        )
        .unwrap_err();
        assert!(error.contains("JSON object"));
    }

    #[test]
    fn rejects_malformed_openai_tool_definitions() {
        let error = validate_openai_tool_definitions(Some(&[json!({
            "type": "function",
            "function": {"name": "read_file", "parameters": []}
        })]))
        .unwrap_err();
        assert!(error.contains("parameters"));
    }

    #[test]
    fn accepts_arguments_matching_the_declared_schema() {
        let call = ToolCall::new("read_file", json!({"path": "README.md"}));
        assert!(validate_tool_calls_against_tools(
            &[call],
            &[json!({
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"]
                    }
                }
            })],
        )
        .is_ok());
    }
}
