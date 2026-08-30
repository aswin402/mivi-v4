//! HTTP server runner command.

use anyhow::Result;
use mivi_server::{create_router, AppState};
use std::path::PathBuf;
use std::sync::Arc;

pub async fn run_serve(port: u16, host: String, model: Option<PathBuf>) -> Result<()> {
    let model_name = model
        .as_ref()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .unwrap_or(mivi_core::DEFAULT_MODEL_ID)
        .to_string();

    let broker = mivi_tools::ToolBroker::new();
    mivi_tools::register_builtin_tools(&broker, std::path::Path::new(".")).await;

    let loaded_model = if let Some(p) = model.as_ref() {
        if !p.exists() {
            anyhow::bail!("Model file does not exist: {:?}", p);
        }
        Some(mivi_model::Model::load(p)?)
    } else {
        None
    };

    let engine = mivi_server::EngineActor::spawn(loaded_model);
    let api_key = std::env::var(mivi_core::ENV_API_KEY).ok();

    let state = Arc::new(AppState::new(model_name.clone(), broker, engine, api_key));

    let app = create_router(state.clone());
    let ip: std::net::IpAddr = host
        .parse()
        .map_err(|e| anyhow::anyhow!("Invalid host address '{}': {}", host, e))?;

    let (listener, actual_addr) =
        mivi_server::bind_with_fallback(ip, port, state.config.max_port_attempts).await?;

    println!(
        r#"
  __  __ _____ _    _ _____          __   _  _   
 |  \/  |_   _| |  | |_   _|        / /  | || |  
 | \  / | | | | |  | | | |  ______ / /_  | || |_ 
 | |\/| | | | \ \  / / | | |______| '_ \ |__   _|
 | |  | |_| |_ \ \/ / _| |_        | (_) |  | |  
 |_|  |_|_____| \__/ |_____|        \___/   |_|  
                                                 
 Mivi-v4 Agent Engine
 Model: {}
 Listening on: http://{}
 OpenAI-compatible API ready.
"#,
        model_name, actual_addr
    );

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
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

    tokio::select! {
        _ = ctrl_c => {
            tracing::info!("\n[mivi] Received Ctrl+C, shutting down gracefully...");
        },
        _ = terminate => {
            tracing::info!("\n[mivi] Received SIGTERM, shutting down gracefully...");
        },
    }
}
