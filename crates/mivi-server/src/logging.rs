//! Hono-style minimal, colored terminal logging middleware and formatters.

use axum::{body::Body, extract::Request, middleware::Next, response::Response};
use std::sync::Arc;
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
    pub stream_metrics: Option<Arc<crate::state::ServerMetrics>>,
    pub step_count: Option<usize>,
    pub finish_reason: Option<String>,
}

/// Print live incoming prompt notification immediately when HTTP body arrives.
pub fn print_incoming_prompt(prompt: &str, prompt_tokens: Option<usize>, is_agent: bool) {
    use std::io::Write;
    let label = if is_agent { "user (agent)" } else { "user" };
    println!(
        "    {}┌─{} {}{}{} › \"{}\"",
        ansi::DIM,
        ansi::RESET,
        ansi::BOLD_CYAN,
        label,
        ansi::RESET,
        summarize_prompt(prompt, 140)
    );
    let token_info = if let Some(n) = prompt_tokens {
        format!("{n} prompt tokens")
    } else {
        "prompt".to_string()
    };
    println!(
        "    {}│  ⏳ prefilling {} & generating on CPU...{}",
        ansi::DIM,
        token_info,
        ansi::RESET
    );
    let _ = std::io::stdout().flush();
}

/// Print completion response items (thinking, tools, final SLM answer) upon inference completion.
pub fn print_completion_response_box(
    thinking: Option<&str>,
    tools: Option<&[String]>,
    assistant_reply: Option<&str>,
) {
    use std::io::Write;
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
            println!(
                "    {}└─{} {}mivi{} › \"{}\"",
                ansi::DIM,
                ansi::RESET,
                ansi::BOLD_GREEN,
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
    let _ = std::io::stdout().flush();
}

/// Print structured, beautiful multi-line box for user prompt, thinking tokens, and SLM output.
pub fn print_interaction_box(
    user_prompt: Option<&str>,
    thinking: Option<&str>,
    tools: Option<&[String]>,
    assistant_reply: Option<&str>,
    is_agent: bool,
) {
    use std::io::Write;
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
                "    {}{}{} {}mivi{} › \"{}\"",
                ansi::DIM,
                prefix,
                ansi::RESET,
                ansi::BOLD_GREEN,
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
    let _ = std::io::stdout().flush();
}

/// Detect a streaming response even when a route did not attach logging metadata.
///
/// SSE is a transport detail, so the content type is the reliable fallback for
/// future streaming routes and model providers.
fn response_is_streaming(response: &Response) -> bool {
    response
        .extensions()
        .get::<LogMetadata>()
        .is_some_and(|meta| meta.is_streaming)
        || response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| {
                value.split(';').next().is_some_and(|media_type| {
                    media_type.trim().eq_ignore_ascii_case("text/event-stream")
                })
            })
}

/// Keep the request log honest: headers are available before an SSE body finishes.
fn wrap_streaming_body(
    response: Response,
    started: Instant,
    method: &str,
    path: &str,
    extra_info: String,
) -> Response {
    use futures::StreamExt;

    let status = response.status();
    let stream_metrics = response
        .extensions()
        .get::<LogMetadata>()
        .and_then(|meta| meta.stream_metrics.clone());
    let (parts, body) = response.into_parts();
    let lifecycle = StreamLifecycleLog {
        logged: false,
        method: method.to_owned(),
        path: path.to_owned(),
        status,
        started,
        extra_info,
        metrics: stream_metrics,
    };
    let body_stream = Box::pin(body.into_data_stream());

    let tracked_stream = futures::stream::unfold(
        (body_stream, lifecycle),
        move |(mut body_stream, mut lifecycle)| async move {
            match body_stream.next().await {
                Some(Ok(bytes)) => Some((Ok(bytes), (body_stream, lifecycle))),
                Some(Err(error)) => {
                    lifecycle.finish(StreamLifecycleOutcome::BodyError);
                    Some((Err(error), (body_stream, lifecycle)))
                }
                None => {
                    lifecycle.finish(StreamLifecycleOutcome::Complete);
                    None
                }
            }
        },
    );

    Response::from_parts(parts, Body::from_stream(tracked_stream))
}

enum StreamLifecycleOutcome {
    Complete,
    BodyError,
    ClientDisconnected,
}

impl StreamLifecycleOutcome {
    fn label(&self) -> &'static str {
        match self {
            Self::Complete => "complete",
            Self::BodyError => "body error",
            Self::ClientDisconnected => "client disconnected",
        }
    }
}

struct StreamLifecycleLog {
    logged: bool,
    method: String,
    path: String,
    status: axum::http::StatusCode,
    started: Instant,
    extra_info: String,
    metrics: Option<Arc<crate::state::ServerMetrics>>,
}

impl StreamLifecycleLog {
    fn finish(&mut self, outcome: StreamLifecycleOutcome) {
        if self.logged {
            return;
        }
        if let Some(metrics) = &self.metrics {
            match outcome {
                StreamLifecycleOutcome::Complete => metrics.record_stream_completion(),
                StreamLifecycleOutcome::BodyError => metrics.record_stream_body_error(),
                StreamLifecycleOutcome::ClientDisconnected => {
                    metrics.record_stream_client_disconnect();
                }
            }
        }
        print_stream_completion(
            &self.method,
            &self.path,
            self.status,
            self.started,
            outcome.label(),
            &self.extra_info,
        );
        self.logged = true;
    }
}

impl Drop for StreamLifecycleLog {
    fn drop(&mut self) {
        self.finish(StreamLifecycleOutcome::ClientDisconnected);
    }
}

fn print_stream_completion(
    method: &str,
    path: &str,
    status: axum::http::StatusCode,
    started: Instant,
    outcome: &str,
    extra_info: &str,
) {
    use std::io::Write;

    let status_code = status.as_u16();
    let status_str = if status.is_success() {
        format!("{}{}{}", ansi::GREEN, status_code, ansi::RESET)
    } else if status.is_client_error() {
        format!("{}{}{}", ansi::YELLOW, status_code, ansi::RESET)
    } else if status.is_server_error() {
        format!("{}{}{}", ansi::RED, status_code, ansi::RESET)
    } else {
        format!("{}{}{}", ansi::CYAN, status_code, ansi::RESET)
    };
    let symbol = if status.is_success() {
        format!("{}←{}", ansi::DIM, ansi::RESET)
    } else {
        format!("{}✗{}", ansi::RED, ansi::RESET)
    };
    let method_str = format!("{}{}{}", ansi::CYAN, method, ansi::RESET);
    let duration = started.elapsed();
    let duration_str = if duration.as_secs() > 0 {
        format!("{:.2}s", duration.as_secs_f64())
    } else if duration.as_millis() > 0 {
        format!("{}ms", duration.as_millis())
    } else {
        format!("{}µs", duration.as_micros())
    };

    println!(
        "  {} {} {:<24} {:<10} {:>6} {} {}{}{}",
        symbol,
        method_str,
        path,
        status_str,
        format!("{}{}{}", ansi::DIM, duration_str, ansi::RESET),
        extra_info,
        ansi::DIM,
        outcome,
        ansi::RESET
    );
    let _ = std::io::stdout().flush();
}

/// Axum middleware for minimal, beautiful Hono-style request/response logging.
pub async fn mivi_log_middleware(req: Request<Body>, next: Next) -> Response {
    use std::io::Write;
    let start = Instant::now();
    let method = req.method().clone();
    let uri = req.uri().clone();
    let path = uri.path().to_string();

    let is_inference_route = path.ends_with("/chat/completions")
        || path.ends_with("/messages")
        || path.ends_with("/agent");

    if is_inference_route {
        let method_str = match method.as_str() {
            "GET" => format!("{}{}{}", ansi::GREEN, method, ansi::RESET),
            "POST" => format!("{}{}{}", ansi::BOLD_CYAN, method, ansi::RESET),
            _ => format!("{}{}{}", ansi::WHITE, method, ansi::RESET),
        };
        println!(
            "  {}→{} {} {:<24} {}[inference started]{}",
            ansi::BOLD_CYAN,
            ansi::RESET,
            method_str,
            path,
            ansi::DIM,
            ansi::RESET
        );
        let _ = std::io::stdout().flush();
    }

    let response = next.run(req).await;
    let elapsed = start.elapsed();
    let is_streaming = response_is_streaming(&response);

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

    let duration_display = if is_streaming {
        format!("headers {duration_str}")
    } else {
        duration_str
    };
    println!(
        "  {} {} {:<24} {:<10} {:>6}{}",
        symbol,
        method_str,
        path,
        status_str,
        format!("{}{}{}", ansi::DIM, duration_display, ansi::RESET),
        extra_info
    );

    if let Some(meta) = &meta {
        if !meta.is_streaming && meta.prompt_summary.is_some() {
            print_interaction_box(
                meta.prompt_summary.as_deref(),
                meta.thinking_summary.as_deref(),
                meta.tool_calls.as_deref(),
                meta.response_summary.as_deref(),
                meta.is_agent,
            );
        }
    }

    if is_streaming {
        wrap_streaming_body(response, start, method.as_str(), &path, extra_info)
    } else {
        response
    }
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
    use crate::state::ServerMetrics;
    use axum::body::Bytes;
    use futures::StreamExt;
    use std::sync::Arc;

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

    #[test]
    fn event_stream_content_type_is_classified_as_streaming() {
        let response = Response::builder()
            .header("content-type", "text/event-stream")
            .body(Body::empty())
            .expect("valid response");

        assert!(response_is_streaming(&response));
    }

    #[tokio::test]
    async fn dropping_streaming_body_records_client_disconnect() {
        let metrics = Arc::new(ServerMetrics::default());
        let stream = futures::stream::iter([
            Ok::<_, std::io::Error>(Bytes::from_static(b"data: first\n\n")),
            Ok::<_, std::io::Error>(Bytes::from_static(b"data: second\n\n")),
        ]);
        let mut response = Response::builder()
            .header("content-type", "text/event-stream")
            .body(Body::from_stream(stream))
            .expect("valid response");
        response.extensions_mut().insert(LogMetadata {
            is_streaming: true,
            stream_metrics: Some(metrics.clone()),
            ..Default::default()
        });

        let wrapped = wrap_streaming_body(
            response,
            Instant::now(),
            "POST",
            "/v1/chat/completions",
            String::new(),
        );
        let mut body = wrapped.into_body().into_data_stream();

        let first_chunk = body
            .next()
            .await
            .expect("first chunk exists")
            .expect("first chunk is valid");
        assert_eq!(first_chunk, Bytes::from_static(b"data: first\n\n"));

        drop(body);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.stream_client_disconnects_total, 1);
        assert_eq!(snapshot.stream_completions_total, 0);
    }

    #[tokio::test]
    async fn consuming_streaming_body_to_eof_records_completion() {
        let metrics = Arc::new(ServerMetrics::default());
        let stream = futures::stream::iter([Ok::<_, std::io::Error>(Bytes::from_static(
            b"data: done\n\n",
        ))]);
        let mut response = Response::builder()
            .header("content-type", "text/event-stream")
            .body(Body::from_stream(stream))
            .expect("valid response");
        response.extensions_mut().insert(LogMetadata {
            is_streaming: true,
            stream_metrics: Some(metrics.clone()),
            ..Default::default()
        });

        let wrapped = wrap_streaming_body(
            response,
            Instant::now(),
            "POST",
            "/v1/messages",
            String::new(),
        );
        let mut body = wrapped.into_body().into_data_stream();

        assert!(body.next().await.expect("first chunk").is_ok());
        assert!(body.next().await.is_none());

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.stream_completions_total, 1);
        assert_eq!(snapshot.stream_body_errors_total, 0);
        assert_eq!(snapshot.stream_client_disconnects_total, 0);
    }

    #[tokio::test]
    async fn streaming_body_error_records_body_error_without_disconnect() {
        let metrics = Arc::new(ServerMetrics::default());
        let stream = futures::stream::iter([Err::<Bytes, _>(std::io::Error::new(
            std::io::ErrorKind::BrokenPipe,
            "stream failed",
        ))]);
        let mut response = Response::builder()
            .header("content-type", "text/event-stream")
            .body(Body::from_stream(stream))
            .expect("valid response");
        response.extensions_mut().insert(LogMetadata {
            is_streaming: true,
            stream_metrics: Some(metrics.clone()),
            ..Default::default()
        });

        let wrapped = wrap_streaming_body(
            response,
            Instant::now(),
            "POST",
            "/v1/messages",
            String::new(),
        );
        let mut body = wrapped.into_body().into_data_stream();

        assert!(body.next().await.expect("error item").is_err());
        drop(body);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.stream_completions_total, 0);
        assert_eq!(snapshot.stream_body_errors_total, 1);
        assert_eq!(snapshot.stream_client_disconnects_total, 0);
    }
}
