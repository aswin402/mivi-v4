//! Syntax-aware structural AST code minifier for Rust, Python, and TypeScript.
//!
//! Inspired by Headroom's syntax-aware signature extraction.
//! Replaces function and class bodies with compact stubs while preserving type definitions,
//! interfaces, imports, and method signatures to fit large codebases into small SLM context windows.

/// Supported programming languages for structural code minification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Rust,
    Python,
    TypeScript,
}

impl Language {
    /// Detects programming language from file extension.
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "rs" => Some(Self::Rust),
            "py" => Some(Self::Python),
            "ts" | "tsx" | "js" | "jsx" => Some(Self::TypeScript),
            _ => None,
        }
    }
}

/// Minifies Rust source code by extracting structural signatures and stripping function bodies.
pub fn minify_rust_code(source: &str) -> String {
    let mut out = String::with_capacity(source.len() / 2);
    let mut in_fn_body = false;
    let mut brace_depth = 0;
    let mut fn_brace_depth = 0;

    for line in source.lines() {
        let trimmed = line.trim();

        // Pass through comments, attributes, and imports directly
        if trimmed.starts_with("//")
            || trimmed.starts_with("#[")
            || trimmed.starts_with("use ")
            || trimmed.starts_with("pub use ")
            || trimmed.starts_with("mod ")
            || trimmed.starts_with("pub mod ")
            || trimmed.starts_with("type ")
            || trimmed.starts_with("pub type ")
            || trimmed.starts_with("const ")
            || trimmed.starts_with("pub const ")
        {
            if !in_fn_body {
                out.push_str(line);
                out.push('\n');
            }
            continue;
        }

        // Detect struct/enum/trait declarations (retain full structural shape)
        if !in_fn_body
            && (trimmed.starts_with("struct ")
                || trimmed.starts_with("pub struct ")
                || trimmed.starts_with("pub(crate) struct ")
                || trimmed.starts_with("enum ")
                || trimmed.starts_with("pub enum ")
                || trimmed.starts_with("trait ")
                || trimmed.starts_with("pub trait ")
                || trimmed.starts_with("impl ")
                || trimmed.starts_with("pub impl "))
        {
            out.push_str(line);
            out.push('\n');
            brace_depth += line.matches('{').count();
            brace_depth = brace_depth.saturating_sub(line.matches('}').count());
            continue;
        }

        // Detect function definitions
        if !in_fn_body
            && (trimmed.contains("fn ")
                || trimmed.starts_with("pub fn ")
                || trimmed.starts_with("async fn ")
                || trimmed.starts_with("pub async fn ")
                || trimmed.starts_with("unsafe fn ")
                || trimmed.starts_with("pub unsafe fn "))
        {
            if line.contains('{') {
                let sig_part = line.split('{').next().unwrap_or(line).trim_end();
                out.push_str(sig_part);
                out.push_str(" { /* ... */ }\n");
                let opens = line.matches('{').count();
                let closes = line.matches('}').count();
                if opens > closes {
                    in_fn_body = true;
                    fn_brace_depth = opens - closes;
                }
            } else {
                out.push_str(line);
                out.push('\n');
            }
            continue;
        }

        if in_fn_body {
            let opens = line.matches('{').count();
            let closes = line.matches('}').count();
            fn_brace_depth = fn_brace_depth.saturating_add(opens).saturating_sub(closes);
            if fn_brace_depth == 0 {
                in_fn_body = false;
            }
            continue;
        }

        // Normal structural lines
        out.push_str(line);
        out.push('\n');
        brace_depth += line.matches('{').count();
        brace_depth = brace_depth.saturating_sub(line.matches('}').count());
    }

    out
}

/// Minifies Python source code by retaining class/function signatures and docstrings.
pub fn minify_python_code(source: &str) -> String {
    let mut out = String::with_capacity(source.len() / 2);
    let mut in_def_body = false;
    let mut def_indent = 0;

    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let indent = line.chars().take_while(|c| c.is_whitespace()).count();

        if in_def_body {
            if indent <= def_indent && !trimmed.is_empty() {
                in_def_body = false;
            } else {
                continue;
            }
        }

        // Detect class or def
        if trimmed.starts_with("class ")
            || trimmed.starts_with("def ")
            || trimmed.starts_with("async def ")
            || trimmed.starts_with("@")
        {
            out.push_str(line);
            out.push('\n');
            if trimmed.starts_with("def ") || trimmed.starts_with("async def ") {
                if trimmed.ends_with(':') {
                    let indent_str = " ".repeat(indent + 4);
                    out.push_str(&format!("{}pass\n", indent_str));
                    in_def_body = true;
                    def_indent = indent;
                }
            }
            continue;
        }

        // Imports and constants
        if trimmed.starts_with("import ")
            || trimmed.starts_with("from ")
            || trimmed.starts_with("#")
            || trimmed.contains('=')
        {
            out.push_str(line);
            out.push('\n');
        }
    }

    out
}

/// Minifies TypeScript/JavaScript source code.
pub fn minify_typescript_code(source: &str) -> String {
    let mut out = String::with_capacity(source.len() / 2);
    let mut in_fn_body = false;
    let mut fn_brace_depth = 0;

    for line in source.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("//")
            || trimmed.starts_with("import ")
            || trimmed.starts_with("export type ")
            || trimmed.starts_with("export interface ")
            || trimmed.starts_with("type ")
            || trimmed.starts_with("interface ")
        {
            if !in_fn_body {
                out.push_str(line);
                out.push('\n');
            }
            continue;
        }

        if !in_fn_body
            && (trimmed.starts_with("function ")
                || trimmed.starts_with("export function ")
                || trimmed.starts_with("export async function ")
                || (trimmed.contains("(") && trimmed.contains(")") && trimmed.contains("=>")))
        {
            if line.contains('{') {
                let sig_part = line.split('{').next().unwrap_or(line).trim_end();
                out.push_str(sig_part);
                out.push_str(" { /* ... */ }\n");
                let opens = line.matches('{').count();
                let closes = line.matches('}').count();
                if opens > closes {
                    in_fn_body = true;
                    fn_brace_depth = opens - closes;
                }
            } else {
                out.push_str(line);
                out.push('\n');
            }
            continue;
        }

        if in_fn_body {
            let opens = line.matches('{').count();
            let closes = line.matches('}').count();
            fn_brace_depth = fn_brace_depth.saturating_add(opens).saturating_sub(closes);
            if fn_brace_depth == 0 {
                in_fn_body = false;
            }
            continue;
        }

        out.push_str(line);
        out.push('\n');
    }

    out
}

/// Minifies source code according to the specified language.
pub fn minify_source(source: &str, lang: Language) -> String {
    match lang {
        Language::Rust => minify_rust_code(source),
        Language::Python => minify_python_code(source),
        Language::TypeScript => minify_typescript_code(source),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minify_rust_code_extracts_signatures() {
        let code = r#"
use std::collections::HashMap;

pub struct Config {
    pub name: String,
    pub port: u16,
}

impl Config {
    pub fn new(name: String, port: u16) -> Self {
        let validated_port = port.max(1);
        Self { name, port: validated_port }
    }

    pub fn is_valid(&self) -> bool {
        self.port > 0
    }
}
"#;
        let minified = minify_rust_code(code);
        assert!(minified.contains("use std::collections::HashMap;"));
        assert!(minified.contains("pub struct Config"));
        assert!(minified.contains("pub fn new(name: String, port: u16) -> Self { /* ... */ }"));
        assert!(minified.contains("pub fn is_valid(&self) -> bool { /* ... */ }"));
        assert!(!minified.contains("let validated_port"));
    }

    #[test]
    fn test_minify_python_code_extracts_signatures() {
        let code = r#"
import os

class ModelLoader:
    def __init__(self, path: str):
        self.path = path
        self.load_weights()

    def load_weights(self) -> bool:
        weights = os.listdir(self.path)
        return len(weights) > 0
"#;
        let minified = minify_python_code(code);
        assert!(minified.contains("import os"));
        assert!(minified.contains("class ModelLoader:"));
        assert!(minified.contains("def __init__(self, path: str):"));
        assert!(minified.contains("pass"));
        assert!(!minified.contains("os.listdir"));
    }
}
