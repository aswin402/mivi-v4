//! Comprehensive end-to-end integration tests for Mivi-v4.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use mivi_agent::{AgentLoop, AgentPhase, AgentState};
use mivi_context::{ContextOp, ContextStore, ContextVm};
use mivi_router::{IntentClassifier, TaskFamily};
use mivi_server::{create_router, AppState, ChatCompletionRequest, MessageDto};
use mivi_tools::{get_builtin_tool_definitions, register_builtin_tools, ToolBroker, ToolCall};
use std::sync::Arc;
use tokio::sync::Mutex;
use tower::ServiceExt;

#[tokio::test]
async fn test_builtins_and_tool_broker() {
    let broker = ToolBroker::new();
    let temp_dir = std::env::temp_dir().join("mivi_test_tools");
    let _ = std::fs::create_dir_all(&temp_dir);

    register_builtin_tools(&broker, &temp_dir).await;

    // 1. Test calculator tool
    let calc_call = ToolCall {
        name: "calculator".to_string(),
        arguments: serde_json::json!({ "expression": "15 * 4 + 10" }),
    };
    let calc_res = broker.execute(&calc_call).await;
    assert!(calc_res.success);
    assert_eq!(calc_res.output, "70");

    // 2. Test write_file tool
    let write_call = ToolCall {
        name: "write_file".to_string(),
        arguments: serde_json::json!({
            "path": "hello.txt",
            "content": "Mivi v4 engine is fast and low memory"
        }),
    };
    let write_res = broker.execute(&write_call).await;
    assert!(write_res.success);

    // 3. Test read_file tool
    let read_call = ToolCall {
        name: "read_file".to_string(),
        arguments: serde_json::json!({ "path": "hello.txt" }),
    };
    let read_res = broker.execute(&read_call).await;
    assert!(read_res.success);
    assert_eq!(read_res.output, "Mivi v4 engine is fast and low memory");

    // 4. Test list_dir tool
    let list_call = ToolCall {
        name: "list_dir".to_string(),
        arguments: serde_json::json!({ "path": "." }),
    };
    let list_res = broker.execute(&list_call).await;
    assert!(list_res.success);
    assert!(list_res.output.contains("hello.txt"));

    let defs = get_builtin_tool_definitions();
    assert_eq!(defs.len(), 4);
}

#[tokio::test]
async fn test_context_store_and_vm() {
    let mut store = ContextStore::new();
    store.add_block("blk_1", "spec.md", "Mivi uses hybrid SSM and GQA", true);
    store.add_block("blk_2", "readme.md", "Run mivi chat to interact", false);

    assert_eq!(store.blocks.len(), 2);
    let mut vm = ContextVm::new(&mut store);

    let search_res = vm.execute(ContextOp::Search {
        query: "hybrid".to_string(),
    });
    assert!(search_res.contains("Found 1 relevant"));

    let slice_res = vm.execute(ContextOp::Slice {
        source: "spec.md".to_string(),
        start: 0,
        end: 50,
    });
    assert!(slice_res.contains("Slice"));
}

#[tokio::test]
async fn test_intent_classifier_routing() {
    let debug_route = IntentClassifier::classify("Please fix this bug in my python code");
    assert_eq!(debug_route.primary, TaskFamily::Debug);

    let research_route = IntentClassifier::classify("Search the web for latest Rust 2026 news");
    assert_eq!(research_route.primary, TaskFamily::Research);

    let code_route = IntentClassifier::classify("Write a function in rust to compute dot product");
    assert_eq!(code_route.primary, TaskFamily::Code);

    let chat_route = IntentClassifier::classify("Hi there");
    assert_eq!(chat_route.primary, TaskFamily::Chat);
}

#[tokio::test]
async fn test_agent_loop_execution() {
    let broker = ToolBroker::new();
    let temp_dir = std::env::temp_dir();
    register_builtin_tools(&broker, &temp_dir).await;

    let state = AgentState::new("Compute 25 * 4 and tell me", 10);
    let mut agent = AgentLoop::new(state, &broker);

    let simulated_model_output = r#"<think>I need to compute 25 * 4 using the calculator tool.</think>
<tool_call>
{"name": "calculator", "arguments": {"expression": "25 * 4"}}
</tool_call>"#;

    let observation = agent.step(simulated_model_output).await;
    assert_eq!(agent.state.phase, AgentPhase::Verifying);
    assert_eq!(agent.state.step_count, 1);
    assert!(observation.contains("<tool_result name=\"calculator\">100</tool_result>"));

    // Final answer step
    let final_model_output = "The result of 25 * 4 is 100.";
    let final_res = agent.step(final_model_output).await;
    assert_eq!(agent.state.phase, AgentPhase::Completed);
    assert_eq!(final_res, "The result of 25 * 4 is 100.");
}

#[tokio::test]
async fn test_http_server_endpoints() {
    let broker = ToolBroker::new();
    let state = Arc::new(Mutex::new(AppState {
        model_name: "mivi-v4-test".to_string(),
        start_time: std::time::Instant::now(),
        broker,
        model: None,
    }));
    let app = create_router(state);

    // 1. GET /health
    let req = Request::builder()
        .uri("/health")
        .method("GET")
        .body(Body::empty())
        .unwrap();

    let res = app.clone().oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], "ok");

    // 2. GET /v1/models
    let req = Request::builder()
        .uri("/v1/models")
        .method("GET")
        .body(Body::empty())
        .unwrap();

    let res = app.clone().oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    // 3. POST /v1/chat/completions (JSON mode)
    let chat_req = ChatCompletionRequest {
        model: "mivi-v4-test".to_string(),
        messages: vec![MessageDto {
            role: "user".to_string(),
            content: Some("Hello".to_string()),
            name: None,
            thinking: None,
            tool_calls: None,
        }],
        temperature: Some(0.7),
        top_p: Some(0.9),
        max_tokens: Some(64),
        stream: Some(false),
        tools: None,
        tool_choice: None,
    };

    let req = Request::builder()
        .uri("/v1/chat/completions")
        .method("POST")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_vec(&chat_req).unwrap()))
        .unwrap();

    let res = app.clone().oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["model"], "mivi-v4-test");
    assert_eq!(json["choices"][0]["message"]["role"], "assistant");

    // 4. POST /v1/chat/completions (SSE Stream mode)
    let mut stream_req = chat_req.clone();
    stream_req.stream = Some(true);

    let req = Request::builder()
        .uri("/v1/chat/completions")
        .method("POST")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_vec(&stream_req).unwrap()))
        .unwrap();

    let res = app.clone().oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let content_type = res.headers().get("content-type").unwrap().to_str().unwrap();
    assert!(content_type.contains("text/event-stream"));

    // 5. GET /v1/mivi/status
    let req = Request::builder()
        .uri("/v1/mivi/status")
        .method("GET")
        .body(Body::empty())
        .unwrap();

    let res = app.clone().oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["engine"], "mivi-v4");
    assert_eq!(json["status"], "healthy");

    // 6. GET /v1/mivi/tools
    let req = Request::builder()
        .uri("/v1/mivi/tools")
        .method("GET")
        .body(Body::empty())
        .unwrap();

    let res = app.clone().oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json["tools"].as_array().unwrap().len() >= 4);

    // 7. POST /v1/mivi/agent (Autonomous Agent Loop)
    let agent_req = serde_json::json!({
        "task": "Find error in codebase and patch it",
        "max_steps": 5
    });

    let req = Request::builder()
        .uri("/v1/mivi/agent")
        .method("POST")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_vec(&agent_req).unwrap()))
        .unwrap();

    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let content_type = res.headers().get("content-type").unwrap().to_str().unwrap();
    assert!(content_type.contains("text/event-stream"));
}
