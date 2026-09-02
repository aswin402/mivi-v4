//! Tool execution, registry, and markup parser.

pub mod broker;
pub mod builtins;
pub mod parser;
pub mod schema;

pub use broker::ToolBroker;
pub use builtins::{get_builtin_tool_definitions, register_builtin_tools};
pub use parser::{extract_thinking, extract_tool_calls, strip_thinking, strip_tool_calls};
pub use schema::{FunctionDefinition, ToolCall, ToolDefinition, ToolResult};
