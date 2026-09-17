//! CLI commands definition.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

pub const DEFAULT_SERVE_PORT_STR: &str = "8080";
pub const DEFAULT_SERVE_HOST: &str = "127.0.0.1";

#[derive(Parser, Debug)]
#[command(
    name = "mivi",
    version,
    about = "Mivi-v4: CPU-first, low-memory, agent-native SLM engine"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Start the OpenAI-compatible HTTP API server
    Serve {
        #[arg(short, long, default_value = DEFAULT_SERVE_PORT_STR)]
        port: u16,
        #[arg(short = 'H', long, default_value = DEFAULT_SERVE_HOST)]
        host: String,
        /// Root directory exposed to built-in filesystem tools (default: current directory)
        #[arg(long, default_value = ".")]
        workspace: PathBuf,
        /// Browser origin allowed to access the HTTP API; may be repeated (default: disabled)
        #[arg(long = "cors-origin", value_name = "ORIGIN")]
        cors_origins: Vec<String>,
        #[arg(short, long)]
        model: Option<PathBuf>,
        /// JSON model protocol profile; overrides embedded model metadata
        #[arg(long, value_name = "PATH")]
        model_profile: Option<PathBuf>,
        /// Maximum RSS memory in MB before triggering safety shutdown (default: 3000 MB)
        #[arg(long, default_value = "3000")]
        max_memory: f32,
        /// Warning threshold RSS memory in MB (default: 2400 MB)
        #[arg(long, default_value = "2400")]
        warn_memory: f32,
        /// Maximum simultaneous inference requests (default: 1)
        #[arg(long, default_value = "1")]
        max_concurrent_requests: usize,
        /// Maximum simultaneous blocking tool executions (default: 4)
        #[arg(
            long,
            default_value_t = mivi_tools::DEFAULT_MAX_CONCURRENT_TOOL_EXECUTIONS
        )]
        max_concurrent_tool_executions: usize,
        /// Maximum wall-clock time per inference request, including streaming generation
        #[arg(
            long,
            default_value_t = mivi_server::DEFAULT_REQUEST_TIMEOUT_SECS
        )]
        request_timeout_secs: u64,
        /// Maximum wall-clock time before the first model output
        #[arg(
            long,
            default_value_t = mivi_server::DEFAULT_FIRST_TOKEN_TIMEOUT_SECS
        )]
        first_token_timeout_secs: u64,
        /// Disable the resource safety watchdog
        #[arg(long)]
        no_safelock: bool,
        /// KV Cache precision mode: f32, q8_0, tq4 (TurboQuant 4-bit), tq2 (TurboQuant 2-bit)
        #[arg(long)]
        kv_precision: Option<String>,
        /// Maximum context window in tokens (defaults to 16384, max 65536)
        #[arg(short = 'c', long)]
        ctx_size: Option<usize>,
        /// Prompt prefill execution strategy: token or chunked (default: chunked)
        #[arg(long, default_value = "chunked")]
        prefill_strategy: String,
        /// Number of prompt tokens per chunk when using chunked prefill (default: 64)
        #[arg(long, default_value = "64")]
        prefill_tile_tokens: usize,
    },
    /// Interactive terminal chat with local model
    Chat {
        #[arg(short, long)]
        model: PathBuf,
        #[arg(short = 't', long, default_value = "0.2")]
        temp: f32,
        #[arg(short = 'p', long, default_value = "0.9")]
        top_p: f32,
        #[arg(short = 'k', long, default_value = "40")]
        top_k: usize,
        #[arg(long, default_value = "0.05")]
        min_p: f32,
        #[arg(short = 'r', long, default_value = "1.1")]
        rep_penalty: f32,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(short = 'n', long, default_value = "512")]
        max_tokens: usize,
        #[arg(short = 'c', long)]
        ctx_size: Option<usize>,
        #[arg(short = 's', long)]
        system: Option<String>,
        #[arg(long)]
        thinking: bool,
        /// KV Cache precision mode: f32, q8_0, tq4 (TurboQuant 4-bit), tq2 (TurboQuant 2-bit)
        #[arg(long)]
        kv_precision: Option<String>,
    },
    /// Inspect model GGUF metadata and tensor shapes
    Info {
        #[arg(short, long)]
        model: PathBuf,
    },
    /// Run diagnostic health check and CPU capabilities report
    Doctor,
    /// Benchmark inference throughput and memory usage
    Bench {
        #[arg(short, long)]
        model: Option<PathBuf>,
        /// KV Cache precision mode: f32, q8_0, tq4 (TurboQuant 4-bit), tq2 (TurboQuant 2-bit)
        #[arg(long)]
        kv_precision: Option<String>,
        /// Prompt prefill execution strategy for the model benchmark: token or chunked
        #[arg(long, default_value = "chunked")]
        prefill_strategy: String,
        /// Number of prompt tokens per chunk when benchmarking chunked prefill
        #[arg(long, default_value = "64")]
        prefill_tile_tokens: usize,
        /// Approximate prompt size used for the focused model benchmark
        #[arg(long, default_value = "2048")]
        bench_prompt_tokens: usize,
    },
    /// Manage on-disk persistent KV cache files (.kvc)
    Cache {
        #[command(subcommand)]
        action: CacheCommands,
    },
}

#[derive(Subcommand, Debug)]
pub enum CacheCommands {
    /// List all persisted .kvc prefix cache files on disk
    List {
        #[arg(short, long)]
        dir: Option<PathBuf>,
    },
    /// Clear all persisted .kvc prefix cache files from disk
    Clear {
        #[arg(short, long)]
        dir: Option<PathBuf>,
    },
}

/// Parse string representation to KvPrecision enum.
pub fn parse_kv_precision(s: Option<&str>) -> Option<mivi_kv::KvPrecision> {
    match s?.to_ascii_lowercase().as_str() {
        "f32" | "fp32" => Some(mivi_kv::KvPrecision::F32),
        "q8_0" | "q8" => Some(mivi_kv::KvPrecision::Q8_0),
        "tq4" | "turboquant4" | "4bit" => Some(mivi_kv::KvPrecision::TurboQuant4),
        "tq2" | "turboquant2" | "2bit" => Some(mivi_kv::KvPrecision::TurboQuant2),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{Cli, Commands};
    use clap::Parser;

    #[test]
    fn serve_tool_concurrency_limit_is_configurable() {
        let cli = Cli::try_parse_from(["mivi", "serve", "--max-concurrent-tool-executions", "7"])
            .expect("serve arguments should parse");

        match cli.command {
            Commands::Serve {
                max_concurrent_tool_executions,
                ..
            } => assert_eq!(max_concurrent_tool_executions, 7),
            _ => panic!("expected serve command"),
        }
    }

    #[test]
    fn serve_tool_concurrency_limit_defaults_to_four() {
        let cli = Cli::try_parse_from(["mivi", "serve"]).expect("serve arguments should parse");

        match cli.command {
            Commands::Serve {
                max_concurrent_tool_executions,
                ..
            } => assert_eq!(max_concurrent_tool_executions, 4),
            _ => panic!("expected serve command"),
        }
    }

    #[test]
    fn serve_request_timeout_is_configurable() {
        let cli = Cli::try_parse_from(["mivi", "serve", "--request-timeout-secs", "17"])
            .expect("serve arguments should parse");

        match cli.command {
            Commands::Serve {
                request_timeout_secs,
                ..
            } => assert_eq!(request_timeout_secs, 17),
            _ => panic!("expected serve command"),
        }
    }

    #[test]
    fn serve_first_token_timeout_is_configurable() {
        let cli = Cli::try_parse_from(["mivi", "serve", "--first-token-timeout-secs", "9"])
            .expect("serve arguments should parse");

        match cli.command {
            Commands::Serve {
                first_token_timeout_secs,
                ..
            } => assert_eq!(first_token_timeout_secs, 9),
            _ => panic!("expected serve command"),
        }
    }

    #[test]
    fn serve_prefill_strategy_is_configurable() {
        let cli = Cli::try_parse_from([
            "mivi",
            "serve",
            "--prefill-strategy",
            "chunked",
            "--prefill-tile-tokens",
            "32",
        ])
        .expect("server should accept model-agnostic prefill configuration");

        match cli.command {
            Commands::Serve {
                prefill_strategy,
                prefill_tile_tokens,
                ..
            } => {
                assert_eq!(prefill_strategy, "chunked");
                assert_eq!(prefill_tile_tokens, 32);
            }
            _ => panic!("expected serve command"),
        }
    }

    #[test]
    fn serve_prefill_strategy_defaults_to_chunked() {
        let cli = Cli::try_parse_from(["mivi", "serve"]).expect("server defaults should parse");

        match cli.command {
            Commands::Serve {
                prefill_strategy,
                prefill_tile_tokens,
                ..
            } => {
                assert_eq!(prefill_strategy, "chunked");
                assert_eq!(prefill_tile_tokens, 64);
            }
            _ => panic!("expected serve command"),
        }
    }

    #[test]
    fn bench_prefill_options_default_to_agent_sized_chunked_prompt() {
        let cli = Cli::try_parse_from(["mivi", "bench"]).expect("bench defaults should parse");

        match cli.command {
            Commands::Bench {
                prefill_strategy,
                prefill_tile_tokens,
                bench_prompt_tokens,
                ..
            } => {
                assert_eq!(prefill_strategy, "chunked");
                assert_eq!(prefill_tile_tokens, 64);
                assert_eq!(bench_prompt_tokens, 2048);
            }
            _ => panic!("expected bench command"),
        }
    }

    #[test]
    fn bench_prefill_options_are_configurable() {
        let cli = Cli::try_parse_from([
            "mivi",
            "bench",
            "--prefill-strategy",
            "token",
            "--prefill-tile-tokens",
            "32",
            "--bench-prompt-tokens",
            "4096",
        ])
        .expect("bench prefill options should parse");

        match cli.command {
            Commands::Bench {
                prefill_strategy,
                prefill_tile_tokens,
                bench_prompt_tokens,
                ..
            } => {
                assert_eq!(prefill_strategy, "token");
                assert_eq!(prefill_tile_tokens, 32);
                assert_eq!(bench_prompt_tokens, 4096);
            }
            _ => panic!("expected bench command"),
        }
    }
}
