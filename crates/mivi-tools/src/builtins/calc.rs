//! Calculator tool invocation handler.

use super::calc_parser::evaluate_expression;
use super::fs::get_str_arg;
use crate::broker::ToolCancellation;
use crate::schema::ToolResult;

pub fn handle_calculator(args: serde_json::Value) -> ToolResult {
    handle_calculator_with_cancellation(args, None)
}

pub fn handle_calculator_cancellable(
    args: serde_json::Value,
    cancellation: &ToolCancellation,
) -> ToolResult {
    handle_calculator_with_cancellation(args, Some(cancellation))
}

fn handle_calculator_with_cancellation(
    args: serde_json::Value,
    cancellation: Option<&ToolCancellation>,
) -> ToolResult {
    if cancellation.is_some_and(ToolCancellation::is_cancelled) {
        return ToolResult::err("calculator", "Tool execution cancelled");
    }
    let expr = match get_str_arg(&args, "expression") {
        Ok(e) => e,
        Err(e) => return ToolResult::err("calculator", e),
    };
    match evaluate_expression(expr) {
        Ok(val) if !cancellation.is_some_and(ToolCancellation::is_cancelled) => {
            ToolResult::ok("calculator", val.to_string())
        }
        Ok(_) => ToolResult::err("calculator", "Tool execution cancelled"),
        Err(err) => ToolResult::err("calculator", err),
    }
}
