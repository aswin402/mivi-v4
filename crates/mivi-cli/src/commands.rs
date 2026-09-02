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
        /// Maximum RSS memory in MB before triggering safety shutdown (default: 450 MB)
        #[arg(long, default_value = "450")]
        max_memory: f32,
        /// Warning threshold RSS memory in MB (default: 350 MB)
        #[arg(long, default_value = "350")]
        warn_memory: f32,
        /// Disable the resource safety watchdog
        #[arg(long)]
        no_safelock: bool,
        /// KV Cache precision mode: f32, q8_0, tq4 (TurboQuant 4-bit), tq2 (TurboQuant 2-bit)
        #[arg(long)]
        kv_precision: Option<String>,
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
