//! Hono-style minimal, colored terminal logging middleware and formatters.

use axum::{body::Body, extract::Request, middleware::Next, response::Response};
use std::time::Instant;

/// ANSI Color Escape Sequences
pub mod ansi {
    pub const RESET: &str = "\x1b[0m";
    pub const BOLD: &str = "\x1b[1m";
    pub const DIM: &str = "\x1b[2m";
    pub const RED: &str = "\x1b[31m";
    pub const GREEN: &str = "\x1b[32m";
    pub const YELLOW: &str = "\x1b[33m";
    pub const BLUE: &str = "\x1b[34m";
    pub const MAGENTA: &str = "\x1b[35m";
    pub const CYAN: &str = "\x1b[36m";
    pub const WHITE: &str = "\x1b[37m";
    pub const BOLD_CYAN: &str = "\x1b[1;36m";
    pub const BOLD_GREEN: &str = "\x1b[1;32m";
    pub const BOLD_YELLOW: &str = "\x1b[1;33m";
    pub const BOLD_RED: &str = "\x1b[1;31m";
    pub const BOLD_MAGENTA: &str = "\x1b[1;35m";
}

/// Request/Response metadata attached by route handlers for rich terminal logging.
#[derive(Clone, Debug, Default)]
pub struct LogMetadata {
    pub prompt_summary: Option<String>,
    pub response_summary: Option<String>,
    pub thinking_summary: Option<String>,
    pub tokens_prompt: Option<usize>,
    pub tokens_completion: Option<usize>,
    pub tok_per_sec: Option<f64>,
    pub tool_calls: Option<Vec<String>>,
    pub is_agent: bool,
    pub is_streaming: bool,
    pub step_count: Option<usize>,
    pub finish_reason: Option<String>,
}

/// Print structured, beautiful multi-line box for user prompt, thinking tokens, and SLM output.
pub fn print_interaction_box(
    user_prompt: Option<&str>,
    thinking: Option<&str>,
    tools: Option<&[String]>,
    assistant_reply: Option<&str>,
    is_agent: bool,
) {
    if user_prompt.is_none() && assistant_reply.is_none() && thinking.is_none() {
        return;
    }
    let label = if is_agent { "user (agent)" } else { "user" };
    if let Some(prompt) = user_prompt {
        println!(
            "    {}┌─{} {}{}{} › \"{}\"",
            ansi::DIM,
            ansi::RESET,
            ansi::BOLD_CYAN,
            label,
            ansi::RESET,
            summarize_prompt(prompt, 140)
        );
    }
    if let Some(think) = thinking {
        if !think.trim().is_empty() {
            println!(
                "    {}│  💭 thinking{} › \"{}\"",
                ansi::DIM,
                ansi::RESET,
                summarize_prompt(think, 160)
            );
        }
    }
    if let Some(tool_list) = tools {
        for tool in tool_list {
            println!(
                "    {}│  🔧 tool call{} › {}{}{}",
                ansi::DIM,
                ansi::RESET,
                ansi::YELLOW,
                tool,
                ansi::RESET
            );
        }
    }
    if let Some(reply) = assistant_reply {
        if !reply.trim().is_empty() {
            let prefix = if user_prompt.is_some() || thinking.is_some() || tools.is_some() {
                "└─"
            } else {
                "┌─"
            };
            println!(
                "    {}{}{} {}{}{} › \"{}\"",
                ansi::DIM,
                prefix,
                ansi::RESET,
                ansi::BOLD_GREEN,
                "mivi",
                ansi::RESET,
                summarize_prompt(reply, 180)
            );
        } else if thinking.is_some() {
            println!(
                "    {}└─{} {}(thinking complete){}",
                ansi::DIM,
                ansi::RESET,
                ansi::DIM,
                ansi::RESET
            );
        }
    }
}

/// Axum middleware for minimal, beautiful Hono-style request/response logging.
pub async fn mivi_log_middleware(req: Request<Body>, next: Next) -> Response {
    let start = Instant::now();
    let method = req.method().clone();
    let uri = req.uri().clone();
    let path = uri.path().to_string();

    let response = next.run(req).await;
    let elapsed = start.elapsed();

    let status = response.status();
    let status_code = status.as_u16();

    // Color code status
    let status_str = if status.is_success() {
        format!("{}{}{}", ansi::GREEN, status_code, ansi::RESET)
    } else if status.is_client_error() {
        format!("{}{}{}", ansi::YELLOW, status_code, ansi::RESET)
    } else if status.is_server_error() {
        format!("{}{}{}", ansi::RED, status_code, ansi::RESET)
    } else {
        format!("{}{}{}", ansi::CYAN, status_code, ansi::RESET)
    };

    // Format duration
    let duration_str = if elapsed.as_secs() > 0 {
        format!("{:.2}s", elapsed.as_secs_f64())
    } else if elapsed.as_millis() > 0 {
        format!("{}ms", elapsed.as_millis())
    } else {
        format!("{}µs", elapsed.as_micros())
    };

    // Format method with color
    let method_str = match method.as_str() {
        "GET" => format!("{}{}{}", ansi::GREEN, method, ansi::RESET),
        "POST" => format!("{}{}{}", ansi::CYAN, method, ansi::RESET),
        "DELETE" => format!("{}{}{}", ansi::RED, method, ansi::RESET),
        "PUT" | "PATCH" => format!("{}{}{}", ansi::YELLOW, method, ansi::RESET),
        _ => format!("{}{}{}", ansi::WHITE, method, ansi::RESET),
    };

    // Extract metadata if inserted by route handler
    let mut extra_info = String::new();
    let meta = response.extensions().get::<LogMetadata>().cloned();

    if let Some(meta) = &meta {
        if let (Some(p), Some(c)) = (meta.tokens_prompt, meta.tokens_completion) {
            let tps_str = if let Some(tps) = meta.tok_per_sec {
                format!(" ({:.1} tok/s)", tps)
            } else if elapsed.as_secs_f64() > 0.0 && c > 0 {
                format!(" ({:.1} tok/s)", (c as f64) / elapsed.as_secs_f64())
            } else {
                String::new()
            };
            extra_info.push_str(&format!(
                "  {}tokens:{} {}→{}{}{}",
                ansi::DIM,
                ansi::RESET,
                p,
                c,
                tps_str,
                ansi::RESET
            ));
        }

        if let Some(tools) = &meta.tool_calls {
            if !tools.is_empty() {
                extra_info.push_str(&format!(
                    "  {}🔧 {}{}",
                    ansi::YELLOW,
                    tools.join(", "),
                    ansi::RESET
                ));
            }
        }

        if meta.is_agent {
            if let Some(steps) = meta.step_count {
                extra_info.push_str(&format!(
                    "  {}agent steps:{} {}{}{}",
                    ansi::MAGENTA,
                    ansi::RESET,
                    ansi::BOLD,
                    steps,
                    ansi::RESET
                ));
            }
        }

        if let Some(reason) = &meta.finish_reason {
            if reason != "stop" {
                extra_info.push_str(&format!("  {}reason:{} {}", ansi::DIM, ansi::RESET, reason));
            }
        }
    }

    let symbol = if status.is_success() {
        format!("{}←{}", ansi::DIM, ansi::RESET)
    } else if status.is_client_error() {
        format!("{}⚠{}", ansi::YELLOW, ansi::RESET)
    } else {
        format!("{}✗{}", ansi::RED, ansi::RESET)
    };

    println!(
        "  {} {} {:<24} {:<10} {:>6}{}",
        symbol,
        method_str,
        path,
        status_str,
        format!("{}{}{}", ansi::DIM, duration_str, ansi::RESET),
        extra_info
    );

    if let Some(meta) = &meta {
        if !meta.is_streaming {
            print_interaction_box(
                meta.prompt_summary.as_deref(),
                meta.thinking_summary.as_deref(),
                meta.tool_calls.as_deref(),
                meta.response_summary.as_deref(),
                meta.is_agent,
            );
        }
    }

    response
}

/// Helper to format a safe prompt summary string for terminal logs
pub fn summarize_prompt(text: &str, max_len: usize) -> String {
    let clean = text.trim().replace(['\r', '\n', '\t'], " ");
    if clean.chars().count() > max_len {
        let truncated: String = clean.chars().take(max_len.saturating_sub(3)).collect();
        format!("{truncated}...")
    } else {
        clean
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_summarize_prompt_short() {
        assert_eq!(summarize_prompt("hello world", 20), "hello world");
    }

    #[test]
    fn test_summarize_prompt_truncation() {
        let long_text = "What is the capital of France and what is its population?";
        let summary = summarize_prompt(long_text, 25);
        assert!(summary.ends_with("..."));
        assert!(summary.chars().count() <= 25);
    }

    #[test]
    fn test_summarize_prompt_replaces_newlines() {
        assert_eq!(
            summarize_prompt("line1\nline2\tline3\r", 30),
            "line1 line2 line3"
        );
    }
}
