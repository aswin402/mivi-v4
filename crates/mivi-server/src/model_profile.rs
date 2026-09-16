//! Model protocol profiles keep prompt and tool syntax outside HTTP routes.

use crate::engine_actor::EngineModelMetadata;
use mivi_protocol::{Message, ToolDefinition};
use mivi_tools::{DelimitedPythonToolCallCodec, LegacyJsonXmlToolCallCodec, ToolCallCodec};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt::Write;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelProfileKind {
    Legacy,
    DelimitedPython,
    TextOnly,
}

/// Declarative model protocol configuration loaded from model metadata or an external JSON file.
///
/// The values are model data, not server behavior. Adding a model with a different vocabulary or
/// delimiter set should only require another profile file (or a dedicated adapter kind), not a
/// change to the HTTP routes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ModelProfileConfig {
    Legacy,
    TextOnly {
        start_of_text: String,
        message_start: String,
        message_end: String,
    },
    DelimitedPython {
        start_of_text: String,
        message_start: String,
        message_end: String,
        tool_call_start: String,
        tool_call_end: String,
        #[serde(default)]
        thinking_instruction: Option<String>,
    },
}

#[derive(Clone)]
pub struct ModelProfile {
    config: ModelProfileConfig,
    tool_codec: Arc<dyn ToolCallCodec>,
}

impl ModelProfile {
    /// Resolve an explicit profile first, then derive one from embedded model metadata.
    pub fn resolve(
        metadata: Option<&EngineModelMetadata>,
        explicit: Option<&ModelProfileConfig>,
    ) -> Result<Self, String> {
        if let Some(config) = explicit {
            return Self::from_config(config);
        }

        if let Some(config) = config_from_metadata(metadata) {
            return Self::from_config(&config);
        }

        Self::from_config(&ModelProfileConfig::Legacy)
    }

    /// Select a profile from model metadata, never from the user-facing model name.
    pub fn from_metadata(metadata: Option<&EngineModelMetadata>) -> Self {
        Self::resolve(metadata, None).unwrap_or_else(|_| Self::legacy())
    }

    pub fn from_config(config: &ModelProfileConfig) -> Result<Self, String> {
        match config {
            ModelProfileConfig::Legacy => Ok(Self::legacy()),
            ModelProfileConfig::TextOnly {
                start_of_text,
                message_start,
                message_end,
            } => {
                validate_delimited_value("start_of_text", start_of_text)?;
                validate_delimited_value("message_start", message_start)?;
                validate_delimited_value("message_end", message_end)?;
                if message_start == message_end {
                    return Err("message_start and message_end must differ".to_string());
                }

                Ok(Self {
                    config: config.clone(),
                    tool_codec: Arc::new(TextOnlyToolCallCodec),
                })
            }
            ModelProfileConfig::DelimitedPython {
                start_of_text,
                message_start,
                message_end,
                tool_call_start,
                tool_call_end,
                ..
            } => {
                validate_delimited_value("start_of_text", start_of_text)?;
                validate_delimited_value("message_start", message_start)?;
                validate_delimited_value("message_end", message_end)?;
                validate_delimited_value("tool_call_start", tool_call_start)?;
                validate_delimited_value("tool_call_end", tool_call_end)?;
                if message_start == message_end {
                    return Err("message_start and message_end must differ".to_string());
                }
                if tool_call_start == tool_call_end {
                    return Err("tool_call_start and tool_call_end must differ".to_string());
                }

                Ok(Self {
                    config: config.clone(),
                    tool_codec: Arc::new(DelimitedPythonToolCallCodec::new(
                        tool_call_start,
                        tool_call_end,
                    )),
                })
            }
        }
    }

    fn legacy() -> Self {
        Self {
            config: ModelProfileConfig::Legacy,
            tool_codec: Arc::new(LegacyJsonXmlToolCallCodec),
        }
    }

    pub fn config(&self) -> &ModelProfileConfig {
        &self.config
    }

    pub fn kind(&self) -> ModelProfileKind {
        match self.config {
            ModelProfileConfig::Legacy => ModelProfileKind::Legacy,
            ModelProfileConfig::DelimitedPython { .. } => ModelProfileKind::DelimitedPython,
            ModelProfileConfig::TextOnly { .. } => ModelProfileKind::TextOnly,
        }
    }

    pub fn capability_name(&self) -> &'static str {
        match self.config {
            ModelProfileConfig::Legacy => "legacy",
            ModelProfileConfig::DelimitedPython { .. } => "delimited_python",
            ModelProfileConfig::TextOnly { .. } => "text_only",
        }
    }

    pub fn tool_codec_name(&self) -> &'static str {
        match self.config {
            ModelProfileConfig::Legacy => "legacy_json_xml",
            ModelProfileConfig::DelimitedPython { .. } => "delimited_python",
            ModelProfileConfig::TextOnly { .. } => "none",
        }
    }

    pub fn supports_tools(&self) -> bool {
        !matches!(self.config, ModelProfileConfig::TextOnly { .. })
    }

    pub fn tool_codec(&self) -> &dyn ToolCallCodec {
        self.tool_codec.as_ref()
    }

    pub fn tool_codec_handle(&self) -> Arc<dyn ToolCallCodec> {
        Arc::clone(&self.tool_codec)
    }

    pub fn render_prompt(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        enable_thinking: bool,
    ) -> Result<String, String> {
        match &self.config {
            ModelProfileConfig::DelimitedPython { .. } => {
                render_delimited_python_prompt(&self.config, messages, tools, enable_thinking)
            }
            ModelProfileConfig::TextOnly { .. } => {
                if enable_thinking {
                    return Err(
                        "the selected text-only model profile does not define thinking support"
                            .to_string(),
                    );
                }
                render_text_only_prompt(&self.config, messages, tools)
            }
            ModelProfileConfig::Legacy => render_legacy_prompt(messages, tools, enable_thinking),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct TextOnlyToolCallCodec;

impl ToolCallCodec for TextOnlyToolCallCodec {
    fn extract(&self, _text: &str) -> Result<Vec<mivi_tools::ToolCall>, String> {
        Ok(Vec::new())
    }

    fn strip(&self, text: &str) -> String {
        text.to_string()
    }
}

fn validate_delimited_value(name: &str, value: &str) -> Result<(), String> {
    if value.is_empty() {
        Err(format!("model profile field '{name}' cannot be empty"))
    } else {
        Ok(())
    }
}

fn config_from_metadata(metadata: Option<&EngineModelMetadata>) -> Option<ModelProfileConfig> {
    let template = metadata?.chat_template.as_deref()?;
    let start_of_text = template_token_containing(template, "startoftext").or_else(|| {
        if template.contains("bos_token") {
            metadata.and_then(|metadata| metadata.bos_token.clone())
        } else {
            None
        }
    })?;
    let message_start = template_token_containing(template, "im_start")?;
    let message_end = template_token_containing(template, "im_end")?;

    match (
        template_token_containing(template, "tool_call_start"),
        template_token_containing(template, "tool_call_end"),
    ) {
        (Some(tool_call_start), Some(tool_call_end)) => Some(ModelProfileConfig::DelimitedPython {
            start_of_text,
            message_start,
            message_end,
            tool_call_start,
            tool_call_end,
            thinking_instruction: None,
        }),
        _ => Some(ModelProfileConfig::TextOnly {
            start_of_text,
            message_start,
            message_end,
        }),
    }
}

/// Extract the model's literal special-token spelling from its embedded template. This keeps
/// metadata discovery independent of any particular model's token text; explicit profile files
/// remain the escape hatch for templates that use a different convention.
fn template_token_containing(template: &str, marker: &str) -> Option<String> {
    let mut search_from = 0;
    while let Some(relative_start) = template[search_from..].find("<|") {
        let start = search_from + relative_start;
        let relative_end = template[start + 2..].find("|>")?;
        let end = start + 2 + relative_end + 2;
        let token = &template[start..end];
        if token.contains(marker) {
            return Some(token.to_string());
        }
        search_from = end;
    }
    None
}

fn render_legacy_prompt(
    messages: &[Message],
    tools: &[ToolDefinition],
    enable_thinking: bool,
) -> Result<String, String> {
    let legacy_messages = messages
        .iter()
        .map(|message| {
            let mut content = message.content.clone().unwrap_or_default();
            for call in &message.tool_calls {
                if !content.is_empty() {
                    content.push('\n');
                }
                content.push_str(mivi_tokenizer::TOOL_CALL_START);
                content.push_str(
                    &serde_json::json!({
                        "name": call.name,
                        "arguments": call.arguments,
                    })
                    .to_string(),
                );
                content.push_str(mivi_tokenizer::TOOL_CALL_END);
            }
            mivi_tokenizer::ChatMessage {
                role: message.role.parse().unwrap_or(mivi_tokenizer::Role::User),
                content: if content.is_empty() {
                    None
                } else {
                    Some(content)
                },
                name: message.name.clone(),
            }
        })
        .collect::<Vec<_>>();
    let tools_json = if tools.is_empty() {
        None
    } else {
        let wrapped = tools
            .iter()
            .map(|tool| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    }
                })
            })
            .collect::<Vec<_>>();
        Some(serde_json::to_string(&wrapped).map_err(|error| error.to_string())?)
    };

    Ok(mivi_tokenizer::format_chatml(
        &legacy_messages,
        tools_json.as_deref(),
        enable_thinking,
    ))
}

fn render_delimited_python_prompt(
    config: &ModelProfileConfig,
    messages: &[Message],
    tools: &[ToolDefinition],
    enable_thinking: bool,
) -> Result<String, String> {
    let ModelProfileConfig::DelimitedPython {
        start_of_text,
        message_start,
        message_end,
        tool_call_start,
        tool_call_end,
        thinking_instruction,
    } = config
    else {
        return Err("delimited prompt renderer requires a delimited_python profile".to_string());
    };

    let has_system = messages.iter().any(|message| {
        message.role.eq_ignore_ascii_case("system")
            || message.role.eq_ignore_ascii_case("developer")
    });
    let mut output = String::from(start_of_text);
    let needs_injected_system = !has_system && (!tools.is_empty() || enable_thinking);
    let mut injected_tools = false;

    if needs_injected_system {
        write!(
            output,
            "{message_start}system\n{}",
            mivi_core::DEFAULT_SYSTEM_PROMPT
        )
        .unwrap();
        append_native_system_instructions(
            &mut output,
            tools,
            enable_thinking,
            thinking_instruction.as_deref(),
            &mut injected_tools,
        )?;
        writeln!(output, "{message_end}").unwrap();
    }

    for message in messages {
        let role = if message.role.eq_ignore_ascii_case("developer") {
            "system"
        } else {
            message.role.as_str()
        };
        if let Some(name) = &message.name {
            writeln!(output, "{message_start}{role}:{name}").unwrap();
        } else {
            writeln!(output, "{message_start}{role}").unwrap();
        }
        if let Some(content) = &message.content {
            output.push_str(content);
        }
        if role.eq_ignore_ascii_case("system") && !injected_tools {
            append_native_system_instructions(
                &mut output,
                tools,
                enable_thinking,
                thinking_instruction.as_deref(),
                &mut injected_tools,
            )?;
        }
        if role.eq_ignore_ascii_case("assistant") && !message.tool_calls.is_empty() {
            if message
                .content
                .as_deref()
                .is_some_and(|content| !content.is_empty())
            {
                output.push('\n');
            }
            output.push_str(tool_call_start);
            output.push_str(&format_python_tool_calls(&message.tool_calls)?);
            output.push_str(tool_call_end);
        }
        writeln!(output, "{message_end}").unwrap();
    }

    writeln!(output, "{message_start}assistant").unwrap();
    Ok(output)
}

fn render_text_only_prompt(
    config: &ModelProfileConfig,
    messages: &[Message],
    tools: &[ToolDefinition],
) -> Result<String, String> {
    let ModelProfileConfig::TextOnly {
        start_of_text,
        message_start,
        message_end,
    } = config
    else {
        return Err("text-only prompt renderer requires a text_only profile".to_string());
    };

    if !tools.is_empty() {
        return Err("text-only model profiles cannot render tools".to_string());
    }
    if messages
        .iter()
        .any(|message| !message.tool_calls.is_empty() || message.role.eq_ignore_ascii_case("tool"))
    {
        return Err("text-only model profiles cannot render tool history".to_string());
    }

    let mut output = String::from(start_of_text);
    for message in messages {
        let role = if message.role.eq_ignore_ascii_case("developer") {
            "system"
        } else {
            message.role.as_str()
        };
        if let Some(name) = &message.name {
            writeln!(output, "{message_start}{role}:{name}").unwrap();
        } else {
            writeln!(output, "{message_start}{role}").unwrap();
        }
        if let Some(content) = &message.content {
            output.push_str(content);
        }
        writeln!(output, "{message_end}").unwrap();
    }
    writeln!(output, "{message_start}assistant").unwrap();
    Ok(output)
}

fn append_native_system_instructions(
    output: &mut String,
    tools: &[ToolDefinition],
    enable_thinking: bool,
    thinking_instruction: Option<&str>,
    injected_tools: &mut bool,
) -> Result<(), String> {
    if !tools.is_empty() {
        let tools_json = serde_json::to_string(tools).map_err(|error| error.to_string())?;
        write!(output, "\nList of tools: {tools_json}").unwrap();
        *injected_tools = true;
    }
    if enable_thinking {
        let instruction = thinking_instruction.ok_or_else(|| {
            "the selected model profile does not define a thinking instruction".to_string()
        })?;
        output.push('\n');
        output.push_str(instruction);
    }
    Ok(())
}

fn format_python_tool_calls(calls: &[mivi_protocol::ToolCall]) -> Result<String, String> {
    let mut output = String::from('[');
    for (index, call) in calls.iter().enumerate() {
        if index > 0 {
            output.push_str(", ");
        }
        let arguments = call
            .arguments
            .as_object()
            .ok_or_else(|| format!("arguments for '{}' must be a JSON object", call.name))?;
        write!(output, "{}(", call.name).unwrap();
        for (argument_index, (name, value)) in arguments.iter().enumerate() {
            if argument_index > 0 {
                output.push_str(", ");
            }
            write!(output, "{name}={}", format_python_value(value)?).unwrap();
        }
        output.push(')');
    }
    output.push(']');
    Ok(output)
}

fn format_python_value(value: &Value) -> Result<String, String> {
    match value {
        Value::Null => Ok("None".to_string()),
        Value::Bool(value) => Ok(if *value { "True" } else { "False" }.to_string()),
        Value::Number(value) => Ok(value.to_string()),
        Value::String(value) => serde_json::to_string(value).map_err(|error| error.to_string()),
        Value::Array(values) => values
            .iter()
            .map(format_python_value)
            .collect::<Result<Vec<_>, _>>()
            .map(|values| format!("[{}]", values.join(", "))),
        Value::Object(values) => values
            .iter()
            .map(|(key, value)| {
                Ok(format!(
                    "{}: {}",
                    serde_json::to_string(key).map_err(|error| error.to_string())?,
                    format_python_value(value)?
                ))
            })
            .collect::<Result<Vec<_>, String>>()
            .map(|values| format!("{{{}}}", values.join(", "))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_tool_template_is_selected_from_model_metadata() {
        let metadata = crate::engine_actor::EngineModelMetadata {
            chat_template: Some(
                "<|startoftext|><|im_start|>{{ messages }}<|im_end|><|tool_call_start|>{{ tool_calls }}<|tool_call_end|>"
                    .to_string(),
            ),
            ..Default::default()
        };
        let profile = ModelProfile::from_metadata(Some(&metadata));

        assert_eq!(profile.kind(), ModelProfileKind::DelimitedPython);

        let messages = vec![mivi_protocol::Message {
            role: "user".to_string(),
            content: Some("Read README.md".to_string()),
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
            reasoning: None,
        }];
        let tools = vec![mivi_protocol::ToolDefinition {
            name: "read_file".to_string(),
            description: Some("Read a file".to_string()),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"path": {"type": "string"}}
            }),
        }];

        let prompt = profile
            .render_prompt(&messages, &tools, false)
            .expect("native prompt should render");

        assert!(prompt.starts_with("<|startoftext|><|im_start|>system"));
        assert!(prompt.contains("List of tools: [{"));
        assert!(prompt.contains("<|im_start|>user\nRead README.md<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn native_profile_injects_tools_for_case_insensitive_system_role() {
        let metadata = crate::engine_actor::EngineModelMetadata {
            chat_template: Some(
                "<|startoftext|><|im_start|>{{ messages }}<|im_end|><|tool_call_start|>{{ tool_calls }}<|tool_call_end|>"
                    .to_string(),
            ),
            ..Default::default()
        };
        let profile = ModelProfile::from_metadata(Some(&metadata));
        let messages = vec![mivi_protocol::Message {
            role: "System".to_string(),
            content: Some("Follow the available tools".to_string()),
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
            reasoning: None,
        }];
        let tools = vec![mivi_protocol::ToolDefinition {
            name: "read_file".to_string(),
            description: None,
            parameters: serde_json::json!({}),
        }];

        let prompt = profile
            .render_prompt(&messages, &tools, false)
            .expect("native prompt should render");

        assert!(prompt.contains("List of tools:"));
    }

    #[test]
    fn explicit_profile_controls_all_native_delimiters() {
        let config = ModelProfileConfig::DelimitedPython {
            start_of_text: "<BOS>".to_string(),
            message_start: "<MSG>".to_string(),
            message_end: "</MSG>".to_string(),
            tool_call_start: "<CALL>".to_string(),
            tool_call_end: "</CALL>".to_string(),
            thinking_instruction: Some("Use the configured reasoning format.".to_string()),
        };
        let profile = ModelProfile::from_config(&config).expect("profile should validate");
        let messages = vec![Message {
            role: "user".to_string(),
            content: Some("Inspect the project".to_string()),
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
            reasoning: None,
        }];

        let prompt = profile
            .render_prompt(&messages, &[], true)
            .expect("configured prompt should render");

        assert!(prompt.starts_with("<BOS><MSG>system"));
        assert!(prompt.contains("Use the configured reasoning format."));
        assert!(prompt.ends_with("<MSG>assistant\n"));
        assert!(!prompt.contains("<|im_start|>"));
    }

    #[test]
    fn native_tool_call_history_keeps_null_content_without_an_extra_blank_line() {
        let config = ModelProfileConfig::DelimitedPython {
            start_of_text: "<BOS>".to_string(),
            message_start: "<MSG>".to_string(),
            message_end: "</MSG>".to_string(),
            tool_call_start: "<CALL>".to_string(),
            tool_call_end: "</CALL>".to_string(),
            thinking_instruction: None,
        };
        let profile = ModelProfile::from_config(&config).expect("profile should validate");
        let messages = vec![Message {
            role: "assistant".to_string(),
            content: None,
            name: None,
            tool_call_id: None,
            tool_calls: vec![mivi_protocol::ToolCall {
                id: Some("call_1".to_string()),
                name: "read_file".to_string(),
                arguments: serde_json::json!({"path": "README.md"}),
            }],
            reasoning: None,
        }];

        let prompt = profile
            .render_prompt(&messages, &[], false)
            .expect("configured prompt should render");

        assert!(
            prompt.contains("<MSG>assistant\n<CALL>[read_file(path=\"README.md\")]</CALL></MSG>")
        );
        assert!(!prompt.contains("<MSG>assistant\n\n<CALL>"));
    }

    #[test]
    fn incomplete_metadata_does_not_guess_message_delimiters() {
        let metadata = crate::engine_actor::EngineModelMetadata {
            chat_template: Some("<|tool_call_start|>...<|tool_call_end|>".to_string()),
            ..Default::default()
        };

        let profile = ModelProfile::from_metadata(Some(&metadata));

        assert_eq!(profile.kind(), ModelProfileKind::Legacy);
    }

    #[test]
    fn chat_template_without_tool_protocol_is_text_only() {
        let metadata = crate::engine_actor::EngineModelMetadata {
            chat_template: Some("<|startoftext|><|im_start|>{{ messages }}<|im_end|>".to_string()),
            ..Default::default()
        };

        let profile = ModelProfile::from_metadata(Some(&metadata));

        assert_eq!(profile.kind(), ModelProfileKind::TextOnly);
        assert_eq!(profile.tool_codec_name(), "none");
        assert!(!profile.supports_tools());
    }

    #[test]
    fn chat_template_bos_variable_uses_metadata_token() {
        let metadata = crate::engine_actor::EngineModelMetadata {
            chat_template: Some(
                "{{- bos_token -}}<|im_start|>{{ messages }}<|im_end|>".to_string(),
            ),
            bos_token: Some("<|startoftext|>".to_string()),
            ..Default::default()
        };

        let profile = ModelProfile::from_metadata(Some(&metadata));

        assert_eq!(profile.kind(), ModelProfileKind::TextOnly);
        assert!(!profile.supports_tools());
    }

    #[test]
    fn invalid_explicit_profile_is_rejected() {
        let config = ModelProfileConfig::DelimitedPython {
            start_of_text: String::new(),
            message_start: "<MSG>".to_string(),
            message_end: "</MSG>".to_string(),
            tool_call_start: "<CALL>".to_string(),
            tool_call_end: "</CALL>".to_string(),
            thinking_instruction: None,
        };

        let error = match ModelProfile::from_config(&config) {
            Ok(_) => panic!("empty delimiter must fail"),
            Err(error) => error,
        };

        assert!(error.contains("start_of_text"));
    }

    #[test]
    fn text_only_profile_renders_configured_messages_without_tool_support() {
        let config = ModelProfileConfig::TextOnly {
            start_of_text: "<BOS>".to_string(),
            message_start: "<MSG>".to_string(),
            message_end: "</MSG>".to_string(),
        };
        let profile =
            ModelProfile::from_config(&config).expect("text-only profile should validate");
        let messages = vec![Message {
            role: "user".to_string(),
            content: Some("Hello".to_string()),
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
            reasoning: None,
        }];

        let prompt = profile
            .render_prompt(&messages, &[], false)
            .expect("text-only prompt should render");

        assert_eq!(profile.capability_name(), "text_only");
        assert_eq!(profile.tool_codec_name(), "none");
        assert!(!profile.supports_tools());
        assert_eq!(prompt, "<BOS><MSG>user\nHello</MSG>\n<MSG>assistant\n");
        assert!(!prompt.contains("tool_call"));
    }

    #[test]
    fn profile_conformance_suite_covers_every_profile_kind() {
        let profiles = [
            ModelProfile::from_config(&ModelProfileConfig::Legacy).expect("legacy profile"),
            ModelProfile::from_config(&ModelProfileConfig::DelimitedPython {
                start_of_text: "<BOS>".to_string(),
                message_start: "<MSG>".to_string(),
                message_end: "</MSG>".to_string(),
                tool_call_start: "<CALL>".to_string(),
                tool_call_end: "</CALL>".to_string(),
                thinking_instruction: None,
            })
            .expect("delimited profile"),
            ModelProfile::from_config(&ModelProfileConfig::TextOnly {
                start_of_text: "<TEXT_BOS>".to_string(),
                message_start: "<TEXT_MSG>".to_string(),
                message_end: "</TEXT_MSG>".to_string(),
            })
            .expect("text-only profile"),
        ];
        let messages = vec![Message {
            role: "user".to_string(),
            content: Some("hello".to_string()),
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
            reasoning: None,
        }];
        let tools = vec![ToolDefinition {
            name: "lookup".to_string(),
            description: Some("Look something up".to_string()),
            parameters: serde_json::json!({"type": "object"}),
        }];

        for profile in profiles {
            let prompt = profile
                .render_prompt(&messages, &[], false)
                .expect("every profile should render ordinary chat");
            assert!(prompt.contains("hello"));
            assert!(prompt.ends_with("assistant\n"));
            assert!(profile.tool_codec().extract("").unwrap().is_empty());

            let tool_result = profile.render_prompt(&messages, &tools, false);
            if profile.supports_tools() {
                let prompt = tool_result.expect("tool-capable profiles should render tools");
                assert!(prompt.contains("lookup"));
            } else {
                let error = tool_result.expect_err("text-only profile must reject tools");
                assert!(error.contains("cannot render tools"));
            }
        }
    }
}
