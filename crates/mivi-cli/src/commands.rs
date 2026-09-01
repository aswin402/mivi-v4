//! CLI commands definition.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

pub const DEFAULT_SERVE_PORT_STR: &str = "8080";
pub const DEFAULT_SERVE_HOST: &str = "0.0.0.0";

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
        #[arg(short, long)]
        model: Option<PathBuf>,
        /// Maximum RSS memory in MB before triggering safety shutdown (default: 900 MB)
        #[arg(long, default_value = "900")]
        max_memory: f32,
        /// Warning threshold RSS memory in MB (default: 700 MB)
        #[arg(long, default_value = "700")]
        warn_memory: f32,
        /// Disable the resource safety watchdog
        #[arg(long)]
        no_safelock: bool,
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
    },
}
