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
    },
    /// Interactive terminal chat with local model
    Chat {
        #[arg(short, long)]
        model: PathBuf,
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
