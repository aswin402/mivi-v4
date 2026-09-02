//! HTTP server runner command with Hono-style banner, logging, and resource safety watchdog.

use anyhow::Result;
use mivi_server::{create_router, AppState, ResourceWatchdog, WatchdogConfig};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::watch;

#[derive(Debug, Clone)]
pub struct ServeArgs {
    pub port: u16,
    pub host: String,
    pub model: Option<PathBuf>,
    pub max_memory: f32,
    pub warn_memory: f32,
    pub no_safelock: bool,
    pub kv_precision: Option<String>,
}

pub async fn run_serve(args: ServeArgs) -> Result<()> {
    let start_time = Instant::now();
    let model_name = mivi_core::DEFAULT_MODEL_ID.to_string();

    let broker = mivi_tools::ToolBroker::new();
    mivi_tools::register_builtin_tools(&broker, std::path::Path::new(".")).await;
    let tool_count = mivi_tools::get_builtin_tool_definitions().len();

    let loaded_model = if let Some(p) = args.model.as_ref() {
        if !p.exists() {
            anyhow::bail!("Model file does not exist: {:?}", p);
        }
        let precision = crate::commands::parse_kv_precision(args.kv_precision.as_deref());
        Some(mivi_model::Model::load_with_options(p, None, precision)?)
    } else {
        None
    };

    let engine = mivi_server::EngineActor::spawn(loaded_model);
    let api_key = std::env::var(mivi_core::ENV_API_KEY).ok();

    let state = Arc::new(AppState::new(model_name.clone(), broker, engine, api_key));

    let app = create_router(state.clone());
    let ip: std::net::IpAddr = args
        .host
        .parse()
        .map_err(|e| anyhow::anyhow!("Invalid host address '{}': {}", args.host, e))?;

    let (listener, actual_addr) =
        mivi_server::bind_with_fallback(ip, args.port, state.config.max_port_attempts).await?;

    let initial_rss = mivi_core::estimate_process_memory_mb();

    // Spawn resource safety watchdog
    let watchdog_config = WatchdogConfig {
        warn_mb: args.warn_memory,
        kill_mb: args.max_memory,
        enabled: !args.no_safelock,
        ..Default::default()
    };
    let (safelock_rx, _watchdog_handle) = ResourceWatchdog::spawn(watchdog_config);

    print_startup_banner(
        &model_name,
        &format!("http://{}", actual_addr),
        tool_count,
        initial_rss,
        args.max_memory,
        !args.no_safelock,
    );

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(safelock_rx, start_time))
        .await?;

    Ok(())
}

fn print_startup_banner(
    model: &str,
    addr: &str,
    tool_count: usize,
    rss_mb: f32,
    max_mb: f32,
    safelock_active: bool,
) {
    use mivi_server::logging::ansi::*;

    let safelock_str = if safelock_active {
        format!("{BOLD_GREEN}{:.0} MB limit{RESET}", max_mb)
    } else {
        format!("{YELLOW}disabled{RESET}")
    };

    println!(
        r#"
  {BOLD_CYAN}⚡ Mivi Agent Engine{RESET} {DIM}v{}{RESET}

  {DIM}•{RESET} {BOLD}Listening:{RESET} {GREEN}{}{RESET}
  {DIM}•{RESET} {BOLD}Model:{RESET}     {CYAN}{}{RESET}
  {DIM}•{RESET} {BOLD}Tools:{RESET}     {YELLOW}{} registered{RESET}
  {DIM}•{RESET} {BOLD}Memory:{RESET}    {:.1} MB {DIM}({}){RESET}

  {BOLD}Routes:{RESET}
    {GREEN}GET{RESET}   {BOLD_CYAN}/{RESET} {DIM}(Interactive Web UI Dashboard){RESET}
    {GREEN}GET{RESET}   {DIM}/health{RESET}
    {GREEN}GET{RESET}   {DIM}/v1/models{RESET}
    {GREEN}GET{RESET}   {DIM}/v1/mivi/status{RESET}
    {GREEN}GET{RESET}   {DIM}/v1/mivi/tools{RESET}
    {CYAN}POST{RESET}  {DIM}/v1/chat/completions{RESET}
    {CYAN}POST{RESET}  {DIM}/v1/messages{RESET}
    {CYAN}POST{RESET}  {DIM}/v1/mivi/agent{RESET}
"#,
        env!("CARGO_PKG_VERSION"),
        addr,
        model,
        tool_count,
        rss_mb,
        safelock_str,
    );
}

async fn shutdown_signal(mut safelock_rx: watch::Receiver<bool>, start_time: Instant) {
    let ctrl_c = async {
        if let Err(e) = tokio::signal::ctrl_c().await {
            tracing::warn!("Failed to install Ctrl+C handler: {}", e);
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(e) => {
                tracing::warn!("Failed to install SIGTERM signal handler: {}", e);
            }
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    let safelock_trigger = async {
        while safelock_rx.changed().await.is_ok() {
            if *safelock_rx.borrow() {
                break;
            }
        }
    };

    tokio::select! {
        _ = ctrl_c => {
            println!("\n  \x1b[2m⏹ [mivi]\x1b[0m Received Ctrl+C, shutting down gracefully...");
        },
        _ = terminate => {
            println!("\n  \x1b[2m⏹ [mivi]\x1b[0m Received SIGTERM, shutting down gracefully...");
        },
        _ = safelock_trigger => {
            println!("\n  \x1b[1;31m🛑 [mivi safelock]\x1b[0m Halting server to preserve host resources.");
        },
    }

    let uptime = start_time.elapsed();
    let uptime_str = if uptime.as_secs() > 60 {
        format!("{}m {}s", uptime.as_secs() / 60, uptime.as_secs() % 60)
    } else {
        format!("{:.1}s", uptime.as_secs_f32())
    };
    println!(
        "  \x1b[2m⏹ [mivi] Server stopped. Total uptime: {}\x1b[0m\n",
        uptime_str
    );
}
