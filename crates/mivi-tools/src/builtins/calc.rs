//! Calculator tool invocation handler.

use super::calc_parser::evaluate_expression;
use super::fs::get_str_arg;
use crate::schema::ToolResult;

pub fn handle_calculator(args: serde_json::Value) -> ToolResult {
    let expr = match get_str_arg(&args, "expression") {
        Ok(e) => e,
        Err(e) => return ToolResult::err("calculator", e),
    };
    match evaluate_expression(expr) {
        Ok(val) => ToolResult::ok("calculator", val.to_string()),
        Err(err) => ToolResult::err("calculator", err),
    }
}
