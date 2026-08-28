//! Special token constants for mivi-v4 agent workflows.

pub const BOS_TOKEN: &str = "<|im_start|>";
pub const EOS_TOKEN: &str = "<|im_end|>";
pub const THINK_START: &str = "<think>";
pub const THINK_END: &str = "</think>";
pub const TOOL_CALL_START: &str = "<tool_call>";
pub const TOOL_CALL_END: &str = "</tool_call>";
pub const TOOLS_DEF_START: &str = "<tools>";
pub const TOOLS_DEF_END: &str = "</tools>";
