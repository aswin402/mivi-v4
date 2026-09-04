use clap::Parser;
use mivi_cli::{Cli, Commands};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let subscriber = FmtSubscriber::builder()
        .with_env_filter(filter)
        .finish();
    let _ = tracing::subscriber::set_global_default(subscriber);

    // Initialize Rayon thread pool (respects MIVI_THREADS/RAYON_NUM_THREADS or defaults to at most 2).
    let num_threads = std::env::var("MIVI_THREADS")
        .or_else(|_| std::env::var("RAYON_NUM_THREADS"))
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get().clamp(1, 2))
                .unwrap_or(2)
        });
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global();

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve {
            port,
            host,
            workspace,
            cors_origins,
            model,
            max_memory,
            warn_memory,
            no_safelock,
            kv_precision,
            ctx_size,
        } => {
            mivi_cli::run_serve(mivi_cli::ServeArgs {
                port,
                host,
                workspace,
                cors_origins,
                model,
                max_memory,
                warn_memory,
                no_safelock,
                kv_precision,
                ctx_size,
            })
            .await?;
        }
        Commands::Chat {
            model,
            temp,
            top_p,
            top_k,
            min_p,
            rep_penalty,
            seed,
            max_tokens,
            ctx_size,
            system,
            thinking,
            kv_precision,
        } => {
            mivi_cli::run_chat(mivi_cli::ChatArgs {
                model,
                temp,
                top_p,
                top_k,
                min_p,
                rep_penalty,
                seed,
                max_tokens,
                ctx_size,
                system,
                thinking,
                kv_precision,
            })?;
        }
        Commands::Info { model } => {
            mivi_cli::run_info(model)?;
        }
        Commands::Doctor => {
            mivi_cli::run_doctor()?;
        }
        Commands::Bench {
            model,
            kv_precision,
        } => {
            mivi_cli::run_bench(model, kv_precision)?;
        }
        Commands::Cache { action } => match action {
            mivi_cli::commands::CacheCommands::List { dir } => {
                mivi_cli::run_cache_list(dir)?;
            }
            mivi_cli::commands::CacheCommands::Clear { dir } => {
                mivi_cli::run_cache_clear(dir)?;
            }
        },
    }

    Ok(())
}
