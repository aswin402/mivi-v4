//! Exercise the actual agent endpoint, replacing only model output with an actor.
use super::*;
use crate::config::ServerConfig;
use crate::engine_actor::{EngineCommand, EngineHandle};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

fn request() -> AgentRunRequest {
    serde_json::from_value(serde_json::json!({"task": "calculate one", "max_steps": 3})).unwrap()
}

fn engine(delay: Duration) -> (EngineHandle, Arc<AtomicUsize>) {
    let (tx, mut rx) = mpsc::channel(4);
    let generations = Arc::new(AtomicUsize::new(0));
    let count = generations.clone();
    tokio::spawn(async move {
        while let Some(command) = rx.recv().await {
            match command {
                EngineCommand::GenerateStream { responder, .. } => {
                    let step = count.fetch_add(1, Ordering::SeqCst);
                    tokio::time::sleep(delay).await;
                    let output = if step == 0 {
                        r#"<tool_call>{"name":"calculator","arguments":{"expression":"1"}}</tool_call>"#
                    } else {
                        "final answer"
                    };
                    let _ = responder.send(Ok(output.to_string())).await;
                }
                EngineCommand::Encode { responder, .. } => {
                    let _ = responder.send(vec![1]);
                }
                EngineCommand::Generate { responder, .. } => {
                    let _ = responder.send(Err("unexpected blocking request".to_string()));
                }
            }
        }
    });
    (EngineHandle::new(tx, true), generations)
}

fn state(engine: EngineHandle, broker: mivi_tools::ToolBroker, capacity: usize) -> Arc<AppState> {
    Arc::new(AppState::with_config(
        "fixture",
        broker,
        engine,
        None,
        ServerConfig {
            request_timeout_secs: 1,
            first_token_timeout_secs: 2,
            channel_capacity: capacity,
            ..ServerConfig::default()
        },
    ))
}

#[tokio::test]
async fn task_deadline_does_not_restart_for_each_generation() {
    let broker = mivi_tools::ToolBroker::new();
    broker
        .register(
            "calculator",
            Arc::new(|_| mivi_tools::ToolResult::ok("calculator", "1")),
        )
        .await;
    let (engine, _) = engine(Duration::from_millis(600));
    let state = state(engine, broker, 16);
    let response = run_agent_task(State(state.clone()), Json(request())).await;
    let bytes = tokio::time::timeout(
        Duration::from_secs(3),
        axum::body::to_bytes(response.into_body(), 65536),
    )
    .await
    .unwrap()
    .unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(
        body.contains("Agent task timed out."),
        "full task must share one deadline: {body}"
    );
    assert!(!body.contains("final answer"));
    assert!(body.contains(r#""finish_reason":"error""#));
    assert!(body.contains("[DONE]"));
    assert!(
        state.try_acquire_inference_slot().is_ok(),
        "request permit must be released"
    );
}

#[tokio::test]
async fn normal_task_completes_and_releases_slot() {
    let broker = mivi_tools::ToolBroker::new();
    broker
        .register(
            "calculator",
            Arc::new(|_| mivi_tools::ToolResult::ok("calculator", "1")),
        )
        .await;
    let (engine, generations) = engine(Duration::ZERO);
    let state = state(engine, broker, 16);
    let response = run_agent_task(State(state.clone()), Json(request())).await;
    let bytes = tokio::time::timeout(
        Duration::from_secs(2),
        axum::body::to_bytes(response.into_body(), 65536),
    )
    .await
    .unwrap()
    .unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(body.contains("final answer"));
    assert!(body.contains(r#""finish_reason":"stop""#));
    assert!(body.contains("[DONE]"));
    assert!(!body.contains("timed out"));
    assert_eq!(generations.load(Ordering::SeqCst), 2);
    assert!(state.try_acquire_inference_slot().is_ok());
}

#[tokio::test]
async fn client_disconnect_during_tool_cancels_execution() {
    let broker = mivi_tools::ToolBroker::with_max_concurrent_executions(1);
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let (finished_tx, finished_rx) = tokio::sync::oneshot::channel();
    let signals = Mutex::new(Some((started_tx, finished_tx)));
    broker
        .register_cancellable(
            "calculator",
            Arc::new(move |_, cancellation| {
                if let Some((started, finished)) = signals.lock().unwrap().take() {
                    let _ = started.send(());
                    let watchdog = Instant::now();
                    while !cancellation.is_cancelled()
                        && watchdog.elapsed() < Duration::from_secs(1)
                    {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    let _ = finished.send(cancellation.is_cancelled());
                }
                mivi_tools::ToolResult::ok("calculator", "1")
            }),
        )
        .await;
    let (engine, generations) = engine(Duration::ZERO);
    let state = state(engine, broker, 16);
    let response = run_agent_task(State(state.clone()), Json(request())).await;
    tokio::time::timeout(Duration::from_secs(2), started_rx)
        .await
        .unwrap()
        .unwrap();
    drop(response);
    let cancelled = tokio::time::timeout(Duration::from_secs(2), finished_rx)
        .await
        .unwrap()
        .unwrap();
    assert!(cancelled, "disconnect must signal the active tool handler");
    assert_eq!(
        generations.load(Ordering::SeqCst),
        1,
        "no next generation after disconnect"
    );
    assert!(state.try_acquire_inference_slot().is_ok());
}

#[tokio::test]
async fn task_deadline_releases_slot_when_sse_buffer_is_full() {
    let (engine, generations) = engine(Duration::ZERO);
    let state = state(engine, mivi_tools::ToolBroker::new(), 1);
    let response = run_agent_task(State(state.clone()), Json(request())).await;
    // Keep the body open without draining it: the initial event fills the buffer.
    let released = tokio::time::timeout(Duration::from_millis(1500), async {
        loop {
            if let Ok(permit) = state.try_acquire_inference_slot() {
                return permit;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await;
    drop(response);
    assert!(
        released.is_ok(),
        "SSE backpressure must not outlive the task deadline"
    );
    assert_eq!(generations.load(Ordering::SeqCst), 0);
}
