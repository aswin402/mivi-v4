//! Tool execution, registry, and markup parser.

pub mod broker;
pub mod builtins;
pub mod parser;
pub mod schema;

pub use broker::{
    CancellableToolHandler, ToolBroker, ToolCancellation, DEFAULT_MAX_CONCURRENT_TOOL_EXECUTIONS,
};
pub use builtins::{get_builtin_tool_definitions, register_builtin_tools};
pub use parser::{extract_thinking, extract_tool_calls, strip_thinking, strip_tool_calls};
pub use schema::{
    validate_tool_arguments, FunctionDefinition, ToolCall, ToolDefinition, ToolResult,
    PARSE_ERROR_TOOL_NAME,
};
