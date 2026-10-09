//! Tool broker and executor. Provides bounded blocking execution for tools.

use crate::schema::{ToolCall, ToolResult};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};

pub type ToolHandler = Arc<dyn Fn(serde_json::Value) -> ToolResult + Send + Sync>;
pub type CancellableToolHandler =
    Arc<dyn Fn(serde_json::Value, &ToolCancellation) -> ToolResult + Send + Sync>;

/// Cooperative cancellation state shared with a running tool handler.
#[derive(Clone, Debug)]
pub struct ToolCancellation {
    cancelled: Arc<AtomicBool>,
}

impl Default for ToolCancellation {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolCancellation {
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Request cancellation. Handlers should check `is_cancelled` at safe
    /// interruption points and return promptly when it becomes true.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    #[inline]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

#[derive(Clone)]
enum RegisteredToolHandler {
    Legacy(ToolHandler),
    Cancellable(CancellableToolHandler),
}

/// A dropped async waiter cannot abort spawn_blocking, but it can notify a
/// cooperative handler. Disarm on normal completion so shared tokens remain usable.
struct ToolExecutionCancellation {
    cancellation: ToolCancellation,
    armed: bool,
}

impl Drop for ToolExecutionCancellation {
    fn drop(&mut self) {
        if self.armed {
            self.cancellation.cancel();
        }
    }
}

/// Default maximum number of blocking tool handlers that may run at once.
pub const DEFAULT_MAX_CONCURRENT_TOOL_EXECUTIONS: usize = 4;

#[derive(Clone)]
pub struct ToolBroker {
    handlers: Arc<RwLock<HashMap<String, RegisteredToolHandler>>>,
    execution_slots: Arc<Semaphore>,
}

impl Default for ToolBroker {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolBroker {
    pub fn new() -> Self {
        Self::with_max_concurrent_executions(DEFAULT_MAX_CONCURRENT_TOOL_EXECUTIONS)
    }

    /// Create a broker with a bound on concurrently running blocking handlers.
    ///
    /// A value of zero is treated as one so a broker cannot deadlock all tool
    /// calls waiting for a permit that can never be issued.
    pub fn with_max_concurrent_executions(max: usize) -> Self {
        Self {
            handlers: Arc::new(RwLock::new(HashMap::new())),
            execution_slots: Arc::new(Semaphore::new(max.max(1))),
        }
    }

    pub async fn register(&self, name: &str, handler: ToolHandler) {
        let mut map = self.handlers.write().await;
        map.insert(name.to_string(), RegisteredToolHandler::Legacy(handler));
    }

    /// Register a handler that can cooperatively stop when its request times out.
    pub async fn register_cancellable(&self, name: &str, handler: CancellableToolHandler) {
        let mut map = self.handlers.write().await;
        map.insert(
            name.to_string(),
            RegisteredToolHandler::Cancellable(handler),
        );
    }

    pub async fn execute(&self, call: &ToolCall) -> ToolResult {
        self.execute_with_cancellation(call, ToolCancellation::new())
            .await
    }

    /// Execute a tool with a cancellation state that is visible to cancellable
    /// handlers. Legacy handlers continue to run without a cancellation hook.
    pub async fn execute_with_cancellation(
        &self,
        call: &ToolCall,
        cancellation: ToolCancellation,
    ) -> ToolResult {
        let mut guard = ToolExecutionCancellation {
            cancellation: cancellation.clone(),
            armed: true,
        };
        let result = self.execute_inner(call, cancellation).await;
        guard.armed = false;
        result
    }

    async fn execute_inner(&self, call: &ToolCall, cancellation: ToolCancellation) -> ToolResult {
        if call.name == crate::schema::PARSE_ERROR_TOOL_NAME {
            let error_msg = call
                .arguments
                .get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("Malformed JSON syntax in tool call");
            return ToolResult::err(
                call.name.clone(),
                format!("Tool call parse error: {}", error_msg),
            );
        }

        let handler = {
            let map = self.handlers.read().await;
            map.get(&call.name).cloned()
        };
        if let Some(handler) = handler {
            if cancellation.is_cancelled() {
                return ToolResult::err(
                    call.name.clone(),
                    "Tool execution cancelled before start".to_string(),
                );
            }
            let permit = match Arc::clone(&self.execution_slots).acquire_owned().await {
                Ok(permit) => permit,
                Err(_) => {
                    return ToolResult::err(
                        call.name.clone(),
                        "Tool execution capacity is unavailable".to_string(),
                    )
                }
            };
            if cancellation.is_cancelled() {
                return ToolResult::err(call.name.clone(), "Tool execution cancelled before start");
            }
            let args = call.arguments.clone();
            let call_name = call.name.clone();
            let cancelled_name = call_name.clone();
            let task = match handler {
                RegisteredToolHandler::Legacy(handler) => tokio::task::spawn_blocking(move || {
                    // Keep the permit until the handler returns. This matters
                    // when the caller's timeout drops the execute future after
                    // the blocking task has already started.
                    let _permit = permit;
                    if cancellation.is_cancelled() {
                        return ToolResult::err(
                            cancelled_name,
                            "Tool execution cancelled before start",
                        );
                    }
                    handler(args)
                }),
                RegisteredToolHandler::Cancellable(handler) => {
                    tokio::task::spawn_blocking(move || {
                        // Keep the permit until the handler returns. This
                        // bounds handlers that observe cancellation late.
                        let _permit = permit;
                        if cancellation.is_cancelled() {
                            return ToolResult::err(
                                cancelled_name,
                                "Tool execution cancelled before start",
                            );
                        }
                        handler(args, &cancellation)
                    })
                }
            };
            match task.await {
                Ok(result) => result,
                Err(e) => ToolResult::err(call_name, format!("Tool execution task failed: {}", e)),
            }
        } else {
            ToolResult::err(
                call.name.clone(),
                format!("Tool '{}' not registered in broker", call.name),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;

    #[test]
    fn queued_blocking_handler_does_not_start_after_cancellation() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .max_blocking_threads(1)
            .build()
            .unwrap();
        runtime.block_on(async {
            for cancellable in [false, true] {
                let broker = ToolBroker::new();
                let invoked = Arc::new(AtomicBool::new(false));
                let observed = invoked.clone();
                if cancellable {
                    broker
                        .register_cancellable(
                            "queued",
                            Arc::new(move |_, _| {
                                observed.store(true, Ordering::SeqCst);
                                ToolResult::ok("queued", "ran")
                            }),
                        )
                        .await;
                } else {
                    broker
                        .register(
                            "queued",
                            Arc::new(move |_| {
                                observed.store(true, Ordering::SeqCst);
                                ToolResult::ok("queued", "ran")
                            }),
                        )
                        .await;
                }
                let (release, wait) = std::sync::mpsc::channel::<()>();
                let (started, ready) = tokio::sync::oneshot::channel();
                let blocker = tokio::task::spawn_blocking(move || {
                    let _ = started.send(());
                    let _ = wait.recv_timeout(Duration::from_secs(2));
                });
                ready.await.unwrap();
                let cancellation = ToolCancellation::new();
                let call = ToolCall {
                    name: "queued".into(),
                    arguments: serde_json::json!({}),
                };
                let execution = broker.execute_with_cancellation(&call, cancellation.clone());
                tokio::pin!(execution);
                assert!(
                    tokio::time::timeout(Duration::from_millis(10), &mut execution)
                        .await
                        .is_err()
                );
                cancellation.cancel();
                release.send(()).unwrap();
                blocker.await.unwrap();
                let result = tokio::time::timeout(Duration::from_secs(1), execution)
                    .await
                    .unwrap();
                assert!(
                    !invoked.load(Ordering::SeqCst),
                    "cancelled queued handler must not run"
                );
                assert!(!result.success);
            }
        });
    }

    #[tokio::test]
    async fn normal_tool_completion_keeps_shared_cancellation_token_usable() {
        let broker = ToolBroker::new();
        broker
            .register_cancellable(
                "quick",
                Arc::new(|_, cancellation| {
                    assert!(!cancellation.is_cancelled());
                    ToolResult::ok("quick", "returned")
                }),
            )
            .await;
        let cancellation = ToolCancellation::new();
        let call = ToolCall {
            name: "quick".to_string(),
            arguments: serde_json::json!({}),
        };
        for _ in 0..2 {
            assert!(
                broker
                    .execute_with_cancellation(&call, cancellation.clone())
                    .await
                    .success
            );
            assert!(!cancellation.is_cancelled());
        }
    }

    #[tokio::test]
    async fn dropping_tool_execution_cancels_handler_and_releases_capacity() {
        let broker = ToolBroker::with_max_concurrent_executions(1);
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (finished_tx, finished_rx) = tokio::sync::oneshot::channel();
        let signals = std::sync::Mutex::new(Some((started_tx, finished_tx)));
        broker
            .register_cancellable(
                "wait",
                Arc::new(move |_, cancellation| {
                    if let Some((started, finished)) = signals.lock().unwrap().take() {
                        let _ = started.send(());
                        let watchdog = std::time::Instant::now();
                        while !cancellation.is_cancelled()
                            && watchdog.elapsed() < Duration::from_secs(1)
                        {
                            std::thread::sleep(Duration::from_millis(1));
                        }
                        let _ = finished.send(cancellation.is_cancelled());
                    }
                    ToolResult::ok("wait", "returned")
                }),
            )
            .await;
        let worker_broker = broker.clone();
        let task = tokio::spawn(async move {
            worker_broker
                .execute(&ToolCall {
                    name: "wait".to_string(),
                    arguments: serde_json::json!({}),
                })
                .await
        });
        tokio::time::timeout(Duration::from_secs(2), started_rx)
            .await
            .unwrap()
            .unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        let cancelled = tokio::time::timeout(Duration::from_secs(2), finished_rx)
            .await
            .unwrap()
            .unwrap();
        assert!(
            cancelled,
            "dropping a tool future must signal its blocking handler"
        );
        let result = tokio::time::timeout(
            Duration::from_secs(1),
            broker.execute(&ToolCall {
                name: "wait".to_string(),
                arguments: serde_json::json!({}),
            }),
        )
        .await
        .expect("handler must release execution capacity");
        assert!(result.success);
    }

    #[tokio::test]
    async fn blocking_tool_execution_can_be_bounded() {
        let broker = ToolBroker::with_max_concurrent_executions(1);
        let started = Arc::new(AtomicBool::new(false));
        let release = Arc::new(AtomicBool::new(false));
        let invocations = Arc::new(AtomicUsize::new(0));

        let started_by_tool = Arc::clone(&started);
        let release_by_tool = Arc::clone(&release);
        let invocations_by_tool = Arc::clone(&invocations);
        broker
            .register(
                "blocking_tool",
                Arc::new(move |_| {
                    let invocation = invocations_by_tool.fetch_add(1, Ordering::SeqCst);
                    if invocation == 0 {
                        started_by_tool.store(true, Ordering::SeqCst);
                        while !release_by_tool.load(Ordering::SeqCst) {
                            std::thread::yield_now();
                        }
                    }
                    ToolResult::ok("blocking_tool", "finished")
                }),
            )
            .await;

        let first_broker = broker.clone();
        let first = tokio::spawn(async move {
            first_broker
                .execute(&ToolCall {
                    name: "blocking_tool".to_string(),
                    arguments: serde_json::json!({}),
                })
                .await
        });

        tokio::time::timeout(Duration::from_secs(1), async {
            while !started.load(Ordering::SeqCst) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("first blocking tool must start");

        let second_broker = broker.clone();
        let second = tokio::time::timeout(
            Duration::from_millis(10),
            second_broker.execute(&ToolCall {
                name: "blocking_tool".to_string(),
                arguments: serde_json::json!({}),
            }),
        )
        .await;
        assert!(
            second.is_err(),
            "second tool must wait for the execution slot"
        );
        assert_eq!(invocations.load(Ordering::SeqCst), 1);

        release.store(true, Ordering::SeqCst);
        assert!(first.await.expect("first task must join").success);
    }
}
