//! Syntax-aware code and command output minification engines.
//!
//! Inspired by Headroom and RTK for zero-allocation, deterministic context compaction.

pub mod code;
pub mod output;

pub use code::{
    minify_python_code, minify_rust_code, minify_source, minify_typescript_code, Language,
};
pub use output::{minify_git_diff, minify_json_payload, minify_test_output};
