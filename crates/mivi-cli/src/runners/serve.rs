//! HTTP server runner command with Hono-style banner, logging, and resource safety watchdog.

use anyhow::Result;
use mivi_server::{create_router, AppState, ResourceWatchdog, WatchdogConfig};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::watch;

#[derive(Debug, Clone)]
pub struct ServeArgs {
    pub port: u16,
    pub host: String,
    pub workspace: PathBuf,
    pub cors_origins: Vec<String>,
    pub model: Option<PathBuf>,
    pub model_profile: Option<PathBuf>,
    pub max_memory: f32,
    pub warn_memory: f32,
    pub max_concurrent_requests: usize,
    pub max_concurrent_tool_executions: usize,
    pub request_timeout_secs: u64,
    pub first_token_timeout_secs: u64,
    pub no_safelock: bool,
    pub kv_precision: Option<String>,
    pub ctx_size: Option<usize>,
    pub prefill_strategy: String,
    pub prefill_tile_tokens: usize,
}

fn validate_bind_security(ip: std::net::IpAddr, api_key: Option<&str>) -> Result<()> {
    if !ip.is_loopback() && api_key.is_none_or(|key| key.trim().is_empty()) {
        anyhow::bail!(
            "Refusing to bind to non-loopback address {} without MIVI_API_KEY; set the environment variable or use --host 127.0.0.1",
            ip
        );
    }
    Ok(())
}

pub async fn run_serve(args: ServeArgs) -> Result<()> {
    let start_time = Instant::now();
    let ip: std::net::IpAddr = args
        .host
        .parse()
        .map_err(|e| anyhow::anyhow!("Invalid host address '{}': {}", args.host, e))?;
    let api_key = std::env::var(mivi_core::ENV_API_KEY)
        .ok()
        .filter(|key| !key.trim().is_empty());
    validate_bind_security(ip, api_key.as_deref())?;

    let workspace = std::fs::canonicalize(&args.workspace).map_err(|e| {
        anyhow::anyhow!(
            "Workspace path does not exist or is not accessible '{}': {}",
            args.workspace.display(),
            e
        )
    })?;
    if !workspace.is_dir() {
        anyhow::bail!("Workspace path is not a directory: {:?}", workspace);
    }

    let model_profile = args
        .model_profile
        .as_deref()
        .map(load_model_profile)
        .transpose()?;

    let max_concurrent_tool_executions = args.max_concurrent_tool_executions.max(1);
    let prefill_strategy =
        mivi_model::PrefillStrategy::parse(&args.prefill_strategy, args.prefill_tile_tokens)
            .map_err(anyhow::Error::msg)?;
    let server_config = mivi_server::ServerConfig {
        model_profile,
        prefill_strategy,
        cors_allowed_origins: args.cors_origins,
        max_concurrent_requests: args.max_concurrent_requests.max(1),
        max_concurrent_tool_executions,
        request_timeout_secs: args.request_timeout_secs.max(1),
        first_token_timeout_secs: args.first_token_timeout_secs.max(1),
        ..mivi_server::ServerConfig::default()
    };

    let broker =
        mivi_tools::ToolBroker::with_max_concurrent_executions(max_concurrent_tool_executions);
    mivi_tools::register_builtin_tools(&broker, &workspace).await;
    let tool_count = mivi_tools::get_builtin_tool_definitions().len();

    let loaded_model = if let Some(p) = args.model.as_ref() {
        if !p.exists() {
            anyhow::bail!("Model file does not exist: {:?}", p);
        }
        let precision = crate::commands::parse_kv_precision(args.kv_precision.as_deref());
        Some(mivi_model::Model::load_with_options(
            p,
            args.ctx_size,
            precision,
        )?)
    } else {
        None
    };

    let model_name = loaded_model
        .as_ref()
        .map(|model| model.config.name.clone())
        .unwrap_or_else(|| mivi_core::DEFAULT_MODEL_ID.to_string());

    let engine = mivi_server::EngineActor::try_spawn_with_config(loaded_model, &server_config)?;
    let state = Arc::new(
        AppState::with_config(model_name.clone(), broker, engine, api_key, server_config)
            .with_workspace(workspace),
    );

    let app = create_router(state.clone());

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

fn load_model_profile(path: &Path) -> Result<mivi_server::ModelProfileConfig> {
    let contents = std::fs::read_to_string(path).map_err(|error| {
        anyhow::anyhow!("Unable to read model profile '{}': {error}", path.display())
    })?;
    let profile = serde_json::from_str::<mivi_server::ModelProfileConfig>(&contents)
        .map_err(|error| anyhow::anyhow!("Invalid model profile '{}': {error}", path.display()))?;
    mivi_server::ModelProfile::from_config(&profile)
        .map_err(|error| anyhow::anyhow!("Invalid model profile '{}': {error}", path.display()))?;
    Ok(profile)
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

async fn shutdown_signal(safelock_rx: watch::Receiver<bool>, start_time: Instant) {
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

    let safelock_trigger = wait_for_safelock(safelock_rx);

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

async fn wait_for_safelock(mut safelock_rx: watch::Receiver<bool>) {
    loop {
        match safelock_rx.changed().await {
            Ok(()) if *safelock_rx.borrow() => return,
            Ok(()) => {}
            // A disabled watchdog drops its sender. Channel closure is not a safelock event;
            // keep waiting for Ctrl+C/SIGTERM in shutdown_signal instead.
            Err(_) => std::future::pending::<()>().await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{load_model_profile, validate_bind_security, wait_for_safelock};
    use mivi_server::ModelProfileConfig;
    use std::io::Write;
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
    use std::time::Duration;
    use tempfile::NamedTempFile;

    #[test]
    fn loopback_bind_does_not_require_an_api_key() {
        assert!(validate_bind_security(IpAddr::V4(Ipv4Addr::LOCALHOST), None).is_ok());
        assert!(validate_bind_security(IpAddr::V6(Ipv6Addr::LOCALHOST), None).is_ok());
    }

    #[test]
    fn non_loopback_bind_requires_a_non_empty_api_key() {
        let public_ip = IpAddr::V4(Ipv4Addr::new(192, 0, 2, 1));
        assert!(validate_bind_security(public_ip, None).is_err());
        assert!(validate_bind_security(public_ip, Some("   ")).is_err());
        assert!(validate_bind_security(public_ip, Some("test-key")).is_ok());
    }

    #[test]
    fn external_model_profile_is_loaded_and_validated() {
        let mut file = NamedTempFile::new().expect("temporary profile");
        write!(
            file,
            "{{\"kind\":\"delimited_python\",\"start_of_text\":\"<BOS>\",\"message_start\":\"<MSG>\",\"message_end\":\"</MSG>\",\"tool_call_start\":\"<CALL>\",\"tool_call_end\":\"</CALL>\"}}"
        )
        .expect("write profile");

        let profile = load_model_profile(file.path()).expect("profile should load");

        assert!(matches!(
            profile,
            ModelProfileConfig::DelimitedPython { .. }
        ));
    }

    #[tokio::test]
    async fn closed_watchdog_channel_does_not_trigger_safelock() {
        let (sender, receiver) = tokio::sync::watch::channel(false);
        drop(sender);

        let result =
            tokio::time::timeout(Duration::from_millis(20), wait_for_safelock(receiver)).await;

        assert!(
            result.is_err(),
            "closed watchdog channel must not resolve shutdown"
        );
    }
}
