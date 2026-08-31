use clap::Parser;
use mivi_cli::{Cli, Commands};
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    // Initialize Rayon thread pool for low CPU resource usage (3 worker threads)
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build_global();

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve {
            port,
            host,
            model,
            max_memory,
            warn_memory,
            no_safelock,
        } => {
            mivi_cli::run_serve(mivi_cli::ServeArgs {
                port,
                host,
                model,
                max_memory,
                warn_memory,
                no_safelock,
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
            })?;
        }
        Commands::Info { model } => {
            mivi_cli::run_info(model)?;
        }
        Commands::Doctor => {
            mivi_cli::run_doctor()?;
        }
        Commands::Bench { model } => {
            mivi_cli::run_bench(model)?;
        }
    }

    Ok(())
}
