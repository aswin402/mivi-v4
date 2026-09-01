//! Filesystem workspace tools (read_file, write_file, list_dir).

use super::security::safe_join;
use crate::schema::ToolResult;
use std::path::Path;

#[inline]
pub(crate) fn get_str_arg<'a>(args: &'a serde_json::Value, key: &str) -> Result<&'a str, String> {
    args.get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("Missing required parameter '{}'", key))
}

fn with_safe_path<F>(
    tool_name: &'static str,
    args: &serde_json::Value,
    param: &str,
    ws: &Path,
    default_path: Option<&str>,
    op: F,
) -> ToolResult
where
    F: FnOnce(&Path, &str) -> std::result::Result<String, String>,
{
    let path_str = match default_path {
        Some(def) => args.get(param).and_then(|v| v.as_str()).unwrap_or(def),
        None => match get_str_arg(args, param) {
            Ok(p) => p,
            Err(e) => return ToolResult::err(tool_name, e),
        },
    };
    let target = match safe_join(ws, path_str) {
        Ok(t) => t,
        Err(e) => return ToolResult::err(tool_name, e),
    };
    match op(&target, path_str) {
        Ok(msg) => ToolResult::ok(tool_name, msg),
        Err(err) => ToolResult::err(tool_name, err),
    }
}

const MAX_FILE_READ_BYTES: u64 = 5 * 1024 * 1024; // 5 MB
const MAX_FILE_WRITE_BYTES: usize = 5 * 1024 * 1024; // 5 MB
const MAX_DIR_ENTRIES: usize = 500;

pub fn handle_read_file(args: serde_json::Value, ws: &Path) -> ToolResult {
    with_safe_path("read_file", &args, "path", ws, None, |target, path_str| {
        let meta = std::fs::metadata(target)
            .map_err(|e| format!("Failed to read metadata for '{}': {}", path_str, e))?;
        if !meta.is_file() {
            return Err(format!("'{}' is not a regular file", path_str));
        }
        if meta.len() > MAX_FILE_READ_BYTES {
            return Err(format!(
                "File '{}' size ({} bytes) exceeds maximum allowed read limit ({} bytes)",
                path_str,
                meta.len(),
                MAX_FILE_READ_BYTES
            ));
        }
        std::fs::read_to_string(target)
            .map_err(|e| format!("Failed to read file '{}': {}", path_str, e))
    })
}

pub fn handle_write_file(args: serde_json::Value, ws: &Path) -> ToolResult {
    let content = match get_str_arg(&args, "content") {
        Ok(c) => c,
        Err(e) => return ToolResult::err("write_file", e),
    };

    if content.len() > MAX_FILE_WRITE_BYTES {
        return ToolResult::err(
            "write_file",
            format!(
                "Content size ({} bytes) exceeds maximum write limit ({} bytes)",
                content.len(),
                MAX_FILE_WRITE_BYTES
            ),
        );
    }

    with_safe_path("write_file", &args, "path", ws, None, |target, path_str| {
        if let Some(parent) = target.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                return Err(format!("Failed to create parent directory: {}", e));
            }
        }
        std::fs::write(target, content)
            .map(|_| {
                format!(
                    "Successfully wrote {} bytes to '{}'",
                    content.len(),
                    path_str
                )
            })
            .map_err(|e| format!("Failed to write file '{}': {}", path_str, e))
    })
}

pub fn handle_list_dir(args: serde_json::Value, ws: &Path) -> ToolResult {
    with_safe_path(
        "list_dir",
        &args,
        "path",
        ws,
        Some("."),
        |target, path_str| {
            let entries = std::fs::read_dir(target)
                .map_err(|e| format!("Failed to list dir '{}': {}", path_str, e))?;
            let mut names = Vec::new();
            let mut truncated = false;
            for entry in entries.flatten() {
                if names.len() >= MAX_DIR_ENTRIES {
                    truncated = true;
                    break;
                }
                if let Ok(file_type) = entry.file_type() {
                    let kind = if file_type.is_dir() { "dir" } else { "file" };
                    let clean_name: String = entry
                        .file_name()
                        .to_string_lossy()
                        .chars()
                        .filter(|c| !c.is_control())
                        .collect();
                    names.push(format!("{}: {}", kind, clean_name));
                }
            }
            names.sort();
            if truncated {
                names.push(format!("... [truncated: showing first {} entries]", MAX_DIR_ENTRIES));
            }
            Ok(names.join("\n"))
        },
    )
}
