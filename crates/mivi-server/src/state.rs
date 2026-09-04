//! Application state container for mivi-server.

use crate::config::ServerConfig;
use crate::engine_actor::EngineHandle;
use mivi_tools::ToolBroker;
use std::path::PathBuf;
use std::time::Instant;

pub struct AppState {
    pub model_name: String,
    pub start_time: Instant,
    pub broker: ToolBroker,
    pub engine: EngineHandle,
    pub api_key: Option<String>,
    pub workspace: PathBuf,
    pub config: ServerConfig,
}

impl AppState {
    pub fn new(
        model_name: impl Into<String>,
        broker: ToolBroker,
        engine: EngineHandle,
        api_key: Option<String>,
    ) -> Self {
        Self::with_config(model_name, broker, engine, api_key, ServerConfig::default())
    }

    pub fn with_config(
        model_name: impl Into<String>,
        broker: ToolBroker,
        engine: EngineHandle,
        api_key: Option<String>,
        config: ServerConfig,
    ) -> Self {
        Self {
            model_name: model_name.into(),
            start_time: Instant::now(),
            broker,
            engine,
            api_key,
            workspace: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            config,
        }
    }

    pub fn with_workspace(mut self, workspace: impl Into<PathBuf>) -> Self {
        self.workspace = workspace.into();
        self
    }
}
