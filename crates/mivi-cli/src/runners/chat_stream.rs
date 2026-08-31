//! Real-time stream parser for CLI chat that handles thinking tags, tool calls, and telemetry.

use std::io::{self, Write};
use std::time::Instant;

/// Token stream state for interactive REPL rendering.
#[derive(Debug, PartialEq, Eq)]
enum StreamState {
    Normal,
    InThinking,
    InToolCall,
}

/// Statistics collected during model generation.
#[derive(Debug, Clone, Default)]
pub struct GenerationStats {
    pub total_tokens: usize,
    pub thinking_tokens: usize,
    pub thinking_duration_secs: f64,
    pub total_duration_secs: f64,
    pub tokens_per_sec: f64,
    pub memory_rss_mb: f32,
    pub tool_calls: Vec<String>,
}

/// StreamFilter processes token-by-token text output from `Model::generate_streaming`
/// and formats thinking blocks and tool calls with live ANSI terminal styling.
pub struct StreamFilter {
    state: StreamState,
    buffer: String,
    gen_start: Instant,
    think_start: Option<Instant>,
    total_tokens: usize,
    think_tokens: usize,
    think_duration: f64,
    tool_buffer: String,
    tool_calls: Vec<String>,
}

impl Default for StreamFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamFilter {
    /// Create a new stream filter starting at the current instant.
    pub fn new() -> Self {
        Self::with_thinking(false)
    }

    /// Create a new stream filter with initial thinking state.
    pub fn with_thinking(thinking: bool) -> Self {
        if thinking {
            print!("\n  \x1b[2;3m💭 Thinking...\x1b[0m\n  \x1b[2m│ \x1b[0;2;3m");
            let _ = io::stdout().flush();
            Self {
                state: StreamState::InThinking,
                buffer: String::new(),
                gen_start: Instant::now(),
                think_start: Some(Instant::now()),
                total_tokens: 0,
                think_tokens: 0,
                think_duration: 0.0,
                tool_buffer: String::new(),
                tool_calls: Vec::new(),
            }
        } else {
            Self {
                state: StreamState::Normal,
                buffer: String::new(),
                gen_start: Instant::now(),
                think_start: None,
                total_tokens: 0,
                think_tokens: 0,
                think_duration: 0.0,
                tool_buffer: String::new(),
                tool_calls: Vec::new(),
            }
        }
    }

    /// Process an incoming chunk of decoded token text.
    pub fn on_token(&mut self, text: &str) {
        self.total_tokens += 1;
        self.buffer.push_str(text);

        loop {
            match self.state {
                StreamState::Normal => {
                    // Check for <think> tag
                    if let Some(pos) = self.buffer.find("<think>") {
                        let before = &self.buffer[..pos];
                        if !before.is_empty() {
                            print!("{}", before);
                            let _ = io::stdout().flush();
                        }
                        self.buffer = self.buffer[pos + "<think>".len()..].to_string();
                        self.state = StreamState::InThinking;
                        self.think_start = Some(Instant::now());
                        print!("\n  \x1b[2;3m💭 Thinking...\x1b[0m\n  \x1b[2m│ \x1b[0;2;3m");
                        let _ = io::stdout().flush();
                        continue;
                    }

                    // Check for <tool_call> tag
                    if let Some(pos) = self.buffer.find("<tool_call>") {
                        let before = &self.buffer[..pos];
                        if !before.is_empty() {
                            print!("{}", before);
                            let _ = io::stdout().flush();
                        }
                        self.buffer = self.buffer[pos + "<tool_call>".len()..].to_string();
                        self.state = StreamState::InToolCall;
                        self.tool_buffer.clear();
                        continue;
                    }

                    // If buffer might be part of an opening tag, hold back
                    if self.buffer.ends_with('<')
                        || self.buffer.ends_with("<t")
                        || self.buffer.ends_with("<th")
                        || self.buffer.ends_with("<thi")
                        || self.buffer.ends_with("<thin")
                        || self.buffer.ends_with("<think")
                        || self.buffer.ends_with("<to")
                        || self.buffer.ends_with("<too")
                        || self.buffer.ends_with("<tool")
                        || self.buffer.ends_with("<tool_")
                        || self.buffer.ends_with("<tool_c")
                        || self.buffer.ends_with("<tool_ca")
                        || self.buffer.ends_with("<tool_cal")
                    {
                        break;
                    }

                    // Stream normal text
                    if !self.buffer.is_empty() {
                        print!("{}", self.buffer);
                        let _ = io::stdout().flush();
                        self.buffer.clear();
                    }
                    break;
                }
                StreamState::InThinking => {
                    self.think_tokens += 1;
                    if let Some(pos) = self.buffer.find("</think>") {
                        let think_chunk = &self.buffer[..pos];
                        if !think_chunk.is_empty() {
                            let formatted = think_chunk.replace('\n', "\n  \x1b[2m│ \x1b[0;2;3m");
                            print!("{}", formatted);
                        }
                        let dur = self
                            .think_start
                            .map(|s| s.elapsed().as_secs_f64())
                            .unwrap_or(0.0);
                        self.think_duration = dur;
                        print!(
                            "\x1b[0m\n  \x1b[2m└─ Thought for {:.1}s ({} tokens)\x1b[0m\n\n",
                            dur, self.think_tokens
                        );
                        let _ = io::stdout().flush();

                        self.buffer = self.buffer[pos + "</think>".len()..].to_string();
                        self.state = StreamState::Normal;
                        continue;
                    }

                    // Hold back if partial closing tag
                    if self.buffer.ends_with('<')
                        || self.buffer.ends_with("</")
                        || self.buffer.ends_with("</t")
                        || self.buffer.ends_with("</th")
                        || self.buffer.ends_with("</thi")
                        || self.buffer.ends_with("</thin")
                        || self.buffer.ends_with("</think")
                    {
                        break;
                    }

                    if !self.buffer.is_empty() {
                        let formatted = self.buffer.replace('\n', "\n  \x1b[2m│ \x1b[0;2;3m");
                        print!("{}", formatted);
                        let _ = io::stdout().flush();
                        self.buffer.clear();
                    }
                    break;
                }
                StreamState::InToolCall => {
                    if let Some(pos) = self.buffer.find("</tool_call>") {
                        self.tool_buffer.push_str(&self.buffer[..pos]);
                        let raw_tool = self.tool_buffer.trim().to_string();
                        self.render_tool_call(&raw_tool);
                        self.tool_calls.push(raw_tool);

                        self.buffer = self.buffer[pos + "</tool_call>".len()..].to_string();
                        self.state = StreamState::Normal;
                        continue;
                    }

                    // Hold back if partial closing tag
                    if self.buffer.ends_with('<')
                        || self.buffer.ends_with("</")
                        || self.buffer.ends_with("</t")
                        || self.buffer.ends_with("</to")
                        || self.buffer.ends_with("</too")
                        || self.buffer.ends_with("</tool")
                        || self.buffer.ends_with("</tool_")
                        || self.buffer.ends_with("</tool_c")
                        || self.buffer.ends_with("</tool_ca")
                        || self.buffer.ends_with("</tool_cal")
                        || self.buffer.ends_with("</tool_call")
                    {
                        break;
                    }

                    self.tool_buffer.push_str(&self.buffer);
                    self.buffer.clear();
                    break;
                }
            }
        }
    }

    /// Render a parsed tool call badge in terminal.
    fn render_tool_call(&self, raw_json: &str) {
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(raw_json) {
            let name = val["name"].as_str().unwrap_or("tool");
            let args = val
                .get("arguments")
                .map(|a| {
                    if a.is_string() {
                        a.as_str().unwrap_or("").to_string()
                    } else {
                        a.to_string()
                    }
                })
                .unwrap_or_default();

            println!(
                "\n  \x1b[1;33m🔧 [tool_call: \x1b[1m{}\x1b[0;33m({})\x1b[1;33m]\x1b[0m",
                name, args
            );
        } else {
            println!(
                "\n  \x1b[1;33m🔧 [tool_call: \x1b[0;33m{}\x1b[1;33m]\x1b[0m",
                raw_json
            );
        }
        let _ = io::stdout().flush();
    }

    /// Finalize stream, flush remaining buffer, and return computed stats.
    pub fn finish(mut self) -> GenerationStats {
        // Flush any remaining normal buffer
        if !self.buffer.is_empty() {
            if self.state == StreamState::InThinking {
                let formatted = self.buffer.replace('\n', "\n  \x1b[2m│ \x1b[0;2;3m");
                print!("{}", formatted);
                let dur = self
                    .think_start
                    .map(|s| s.elapsed().as_secs_f64())
                    .unwrap_or(0.0);
                self.think_duration = dur;
                print!(
                    "\x1b[0m\n  \x1b[2m└─ Thought for {:.1}s ({} tokens)\x1b[0m\n\n",
                    dur, self.think_tokens
                );
            } else if self.state == StreamState::InToolCall {
                self.tool_buffer.push_str(&self.buffer);
                let raw_tool = self.tool_buffer.trim().to_string();
                self.render_tool_call(&raw_tool);
                self.tool_calls.push(raw_tool);
            } else {
                print!("{}", self.buffer);
            }
            let _ = io::stdout().flush();
        } else if self.state == StreamState::InThinking {
            let dur = self
                .think_start
                .map(|s| s.elapsed().as_secs_f64())
                .unwrap_or(0.0);
            self.think_duration = dur;
            print!(
                "\x1b[0m\n  \x1b[2m└─ Thought for {:.1}s ({} tokens)\x1b[0m\n\n",
                dur, self.think_tokens
            );
            let _ = io::stdout().flush();
        }

        let elapsed = self.gen_start.elapsed().as_secs_f64();
        let tps = if elapsed > 0.0 {
            self.total_tokens as f64 / elapsed
        } else {
            0.0
        };

        GenerationStats {
            total_tokens: self.total_tokens,
            thinking_tokens: self.think_tokens,
            thinking_duration_secs: self.think_duration,
            total_duration_secs: elapsed,
            tokens_per_sec: tps,
            memory_rss_mb: mivi_core::estimate_process_memory_mb(),
            tool_calls: self.tool_calls,
        }
    }
}

/// Print real-time generation telemetry footer.
pub fn print_telemetry_footer(stats: &GenerationStats) {
    let dur_str = if stats.total_duration_secs >= 60.0 {
        format!(
            "{}m {:.1}s",
            (stats.total_duration_secs / 60.0) as u64,
            stats.total_duration_secs % 60.0
        )
    } else {
        format!("{:.2}s", stats.total_duration_secs)
    };

    println!(
        "  \x1b[2m⏱ {} • {} tokens • {:.1} tok/s • RAM {:.1} MB\x1b[0m\n",
        dur_str, stats.total_tokens, stats.tokens_per_sec, stats.memory_rss_mb
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_filter_normal_text() {
        let mut filter = StreamFilter::new();
        filter.on_token("Hello ");
        filter.on_token("world!");
        let stats = filter.finish();
        assert_eq!(stats.total_tokens, 2);
        assert_eq!(stats.thinking_tokens, 0);
    }

    #[test]
    fn test_stream_filter_with_thinking() {
        let mut filter = StreamFilter::new();
        filter.on_token("<think>");
        filter.on_token("Let's calculate ");
        filter.on_token("2+2=4");
        filter.on_token("</think>");
        filter.on_token("The answer is 4.");
        let stats = filter.finish();
        assert_eq!(stats.total_tokens, 5);
        assert!(stats.thinking_tokens > 0);
    }
}
