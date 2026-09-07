//! Application state container for mivi-server.

use crate::config::ServerConfig;
use crate::engine_actor::EngineHandle;
use mivi_tools::ToolBroker;
use serde::Serialize;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use std::time::{Duration, Instant};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InferenceSlotUnavailable;

#[derive(Debug, Default)]
pub struct ServerMetrics {
    inference_requests_total: AtomicU64,
    inference_requests_rejected_total: AtomicU64,
    inference_errors_total: AtomicU64,
    inference_slot_wait_microseconds_total: AtomicU64,
    generation_latency_microseconds_total: AtomicU64,
    generation_count: AtomicU64,
    prompt_tokens_total: AtomicU64,
    completion_tokens_total: AtomicU64,
    tool_timeouts_total: AtomicU64,
}

#[derive(Debug, Clone, Serialize)]
pub struct MetricsSnapshot {
    pub inference_requests_total: u64,
    pub inference_requests_rejected_total: u64,
    pub inference_errors_total: u64,
    pub inference_slot_wait_microseconds_total: u64,
    pub generation_latency_microseconds_total: u64,
    pub generation_count: u64,
    pub prompt_tokens_total: u64,
    pub completion_tokens_total: u64,
    pub tool_timeouts_total: u64,
}

impl ServerMetrics {
    pub fn record_inference_accepted(&self, slot_wait: Duration) {
        self.inference_requests_total
            .fetch_add(1, Ordering::Relaxed);
        self.inference_slot_wait_microseconds_total
            .fetch_add(duration_microseconds(slot_wait), Ordering::Relaxed);
    }

    pub fn record_inference_rejected(&self) {
        self.inference_requests_rejected_total
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_inference_error(&self) {
        self.inference_errors_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_generation(&self, elapsed: Duration) {
        self.generation_latency_microseconds_total
            .fetch_add(duration_microseconds(elapsed), Ordering::Relaxed);
        self.generation_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_tokens(&self, prompt_tokens: usize, completion_tokens: usize) {
        self.prompt_tokens_total.fetch_add(
            u64::try_from(prompt_tokens).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        self.completion_tokens_total.fetch_add(
            u64::try_from(completion_tokens).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
    }

    pub fn record_tool_timeouts(&self, count: usize) {
        self.tool_timeouts_total
            .fetch_add(u64::try_from(count).unwrap_or(u64::MAX), Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            inference_requests_total: self.inference_requests_total.load(Ordering::Relaxed),
            inference_requests_rejected_total: self
                .inference_requests_rejected_total
                .load(Ordering::Relaxed),
            inference_errors_total: self.inference_errors_total.load(Ordering::Relaxed),
            inference_slot_wait_microseconds_total: self
                .inference_slot_wait_microseconds_total
                .load(Ordering::Relaxed),
            generation_latency_microseconds_total: self
                .generation_latency_microseconds_total
                .load(Ordering::Relaxed),
            generation_count: self.generation_count.load(Ordering::Relaxed),
            prompt_tokens_total: self.prompt_tokens_total.load(Ordering::Relaxed),
            completion_tokens_total: self.completion_tokens_total.load(Ordering::Relaxed),
            tool_timeouts_total: self.tool_timeouts_total.load(Ordering::Relaxed),
        }
    }
}

fn duration_microseconds(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

pub struct AppState {
    pub model_name: String,
    pub start_time: Instant,
    pub broker: ToolBroker,
    pub engine: EngineHandle,
    pub api_key: Option<String>,
    pub workspace: PathBuf,
    pub config: ServerConfig,
    pub metrics: Arc<ServerMetrics>,
    inference_slots: Arc<Semaphore>,
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
        let config = config.normalized();
        let inference_slots = Arc::new(Semaphore::new(config.max_concurrent_requests.max(1)));
        Self {
            model_name: model_name.into(),
            start_time: Instant::now(),
            broker,
            engine,
            api_key,
            workspace: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            config,
            metrics: Arc::new(ServerMetrics::default()),
            inference_slots,
        }
    }

    pub fn with_workspace(mut self, workspace: impl Into<PathBuf>) -> Self {
        self.workspace = workspace.into();
        self
    }

    pub fn try_acquire_inference_slot(
        &self,
    ) -> Result<OwnedSemaphorePermit, InferenceSlotUnavailable> {
        self.inference_slots
            .clone()
            .try_acquire_owned()
            .map_err(|_| InferenceSlotUnavailable)
    }
}
