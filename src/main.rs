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
        Commands::Serve { port, host, model } => {
            mivi_cli::run_serve(port, host, model).await?;
        }
        Commands::Chat {
            model,
            temp,
            max_tokens,
        } => {
            mivi_cli::run_chat(model, temp, max_tokens)?;
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
