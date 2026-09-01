//! Comprehensive end-to-end integration tests for Mivi-v4.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use mivi_agent::{AgentLoop, AgentPhase, AgentState};
use mivi_context::{ContextOp, ContextStore, ContextVm};
use mivi_router::{IntentClassifier, TaskFamily};
use mivi_server::{create_router, AppState, ChatCompletionRequest, MessageDto};
use mivi_tools::{get_builtin_tool_definitions, register_builtin_tools, ToolBroker, ToolCall};
use std::path::PathBuf;
use std::sync::Arc;
use tower::ServiceExt;

#[tokio::test]
async fn test_builtins_and_tool_broker() {
    let broker = ToolBroker::new();
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");

    register_builtin_tools(&broker, temp_dir.path()).await;

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
        start: 5,
        end: 25,
    });
    assert!(slice_res.contains("uses hybrid SSM"));
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
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    register_builtin_tools(&broker, temp_dir.path()).await;

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
async fn test_agent_max_steps_exhaustion() {
    let broker = ToolBroker::new();
    let state = AgentState::new("Infinite loop task", 2);
    let mut agent = AgentLoop::new(state, &broker);

    let call_step =
        r#"<tool_call>{"name": "calculator", "arguments": {"expression": "1 + 1"}}</tool_call>"#;
    let _ = agent.step(call_step).await;
    let _ = agent.step(call_step).await;
    let res = agent.step(call_step).await;

    assert_eq!(agent.state.phase, AgentPhase::Failed);
    assert!(res.contains("exceeded maximum step limit"));
}

#[tokio::test]
async fn test_agent_stagnation_guard() {
    let broker = ToolBroker::new();
    let state = AgentState::new("Stagnant task", 10);
    let mut agent = AgentLoop::new(state, &broker);

    let same_call =
        r#"<tool_call>{"name": "calculator", "arguments": {"expression": "2 + 2"}}</tool_call>"#;
    let _ = agent.step(same_call).await;
    let _ = agent.step(same_call).await;
    let res = agent.step(same_call).await;

    assert_eq!(agent.state.phase, AgentPhase::Failed);
    assert!(res.contains("Agent stagnation detected"));
}

#[tokio::test]
async fn test_agent_tool_failure_propagation() {
    let broker = ToolBroker::new();
    let state = AgentState::new("Unknown tool call", 5);
    let mut agent = AgentLoop::new(state, &broker);

    let unknown_call = r#"<tool_call>{"name": "non_existent_tool", "arguments": {}}</tool_call>"#;
    let res = agent.step(unknown_call).await;

    assert!(res.contains("not registered in broker"));
}

#[tokio::test]
async fn test_http_server_endpoints() {
    let broker = ToolBroker::new();
    let engine = mivi_server::EngineActor::spawn(None);
    let state = Arc::new(AppState::new("mivi-v4-test", broker, engine, None));
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
    let body_bytes = res.into_body().collect().await.unwrap().to_bytes();
    let body_str = String::from_utf8_lossy(&body_bytes);
    assert!(body_str.contains("data: "));
    assert!(body_str.contains("[DONE]"));

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
    let body_bytes = res.into_body().collect().await.unwrap().to_bytes();
    let body_str = String::from_utf8_lossy(&body_bytes);
    assert!(body_str.contains("data: "));
    assert!(body_str.contains("[DONE]"));
}

#[tokio::test]
async fn test_http_server_error_responses() {
    let broker = ToolBroker::new();
    let engine = mivi_server::EngineActor::spawn(None);
    let state = Arc::new(AppState::new(
        "mivi-v4-test",
        broker,
        engine,
        Some("my-secret-key".to_string()),
    ));
    let app = create_router(state);

    // 1. Public health check succeeds without API key
    let health_req = Request::builder()
        .uri("/health")
        .method("GET")
        .body(Body::empty())
        .unwrap();
    let health_res = app.clone().oneshot(health_req).await.unwrap();
    assert_eq!(health_res.status(), StatusCode::OK);

    // 2. Unauthorized when missing API key on protected endpoint
    let unauth_req = Request::builder()
        .uri("/v1/chat/completions")
        .method("POST")
        .header("Content-Type", "application/json")
        .body(Body::from(r#"{"messages":[]}"#))
        .unwrap();

    let unauth_res = app.clone().oneshot(unauth_req).await.unwrap();
    assert_eq!(unauth_res.status(), StatusCode::UNAUTHORIZED);

    // 3. Bad request with empty messages array (with auth header)
    let chat_req = serde_json::json!({
        "model": "mivi-v4-test",
        "messages": []
    });

    let req = Request::builder()
        .uri("/v1/chat/completions")
        .method("POST")
        .header("Authorization", "Bearer my-secret-key")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_vec(&chat_req).unwrap()))
        .unwrap();

    let res = app.clone().oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_http_server_with_real_model() {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let model_path = manifest_dir.join("models/mivi-tiny-test.gguf");
    if !model_path.exists() {
        eprintln!(
            "Skipping test_http_server_with_real_model: models/mivi-tiny-test.gguf not found"
        );
        return;
    }

    let model = mivi_model::Model::load(&model_path).expect("Failed to load test model");
    let broker = ToolBroker::new();
    let engine = mivi_server::EngineActor::spawn(Some(model));
    let state = Arc::new(AppState::new("mivi-tiny-test", broker, engine, None));
    let app = create_router(state);

    let chat_req = ChatCompletionRequest {
        model: "mivi-tiny-test".to_string(),
        messages: vec![MessageDto {
            role: "user".to_string(),
            content: Some("hello".to_string()),
            name: None,
            thinking: None,
            tool_calls: None,
        }],
        temperature: Some(0.0),
        top_p: Some(1.0),
        max_tokens: Some(4),
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

    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["model"], "mivi-tiny-test");
    assert!(json["choices"][0]["message"]["content"].is_string());
}

#[tokio::test]
async fn test_server_port_fallback_integration() {
    use std::net::{IpAddr, Ipv4Addr};
    let ip = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1));

    // Bind port 0 to get an assigned occupied port
    let initial_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let occupied_port = initial_listener.local_addr().unwrap().port();

    // Call bind_with_fallback on the occupied port
    let (fallback_listener, fallback_addr) = mivi_server::bind_with_fallback(ip, occupied_port, 10)
        .await
        .unwrap();

    assert_ne!(fallback_addr.port(), occupied_port);
    assert!(fallback_addr.port() > occupied_port);

    drop(initial_listener);
    drop(fallback_listener);
}

#[test]
fn test_lfm2_350m_model_load_and_forward() {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let model_path = manifest_dir.join("models/mivi-v4-q4_k_m.gguf");
    if !model_path.exists() {
        eprintln!(
            "Skipping test_lfm2_350m_model_load_and_forward: models/mivi-v4-q4_k_m.gguf not found"
        );
        return;
    }

    let mut model = mivi_model::Model::load(&model_path).expect("Failed to load LFM2.5-350M model");
    assert_eq!(model.config.dim, 1024);
    assert_eq!(model.config.n_layers, 16);

    // Test 1: Raw completion prompt
    let raw_prompt = "The capital of France is";
    let mut tokens = model.tokenizer.encode(raw_prompt);
    if model
        .gguf
        .metadata
        .contains_key("tokenizer.ggml.add_bos_token")
    {
        tokens.insert(0, 1); // Prepend BOS
    }
    println!("Raw prompt encoded: {:?}", tokens);
    for &t in &tokens {
        println!("  tok {}: {:?}", t, model.tokenizer.decode_token(t));
    }
    model.state.reset();
    model.kv_cache.reset();
    for (i, &t) in tokens.iter().enumerate() {
        let logits = model.forward(t, i).expect("forward failed");
        if i == tokens.len() - 1 {
            let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            println!("Top 10 predicted tokens after 'The capital of France is':");
            for &(idx, val) in &indexed[..10] {
                println!(
                    "  id {}: logit {:.4} ({:?})",
                    idx,
                    val,
                    model.tokenizer.decode_token(idx as u32)
                );
            }
        }
    }

    // Test 2: Generate 10 tokens from raw prompt
    model.sampler.config.temperature = 0.0; // Greedy sampling
    let gen_output = model.generate(raw_prompt, 10).expect("generate failed");
    println!(
        "Greedy generation for 'The capital of France is': {:?}",
        gen_output
    );

    // Test 3: ChatML structured conversation
    let mut history = Vec::new();
    let questions = ["hii", "who are you?", "what is 8 + 8?"];
    for q in questions {
        history.push(mivi_tokenizer::ChatMessage {
            role: mivi_tokenizer::Role::User,
            content: Some(q.to_string()),
            name: None,
        });
        let prompt = mivi_tokenizer::format_chatml(&history, None, false);
        model.reset_context();
        let resp = model.generate(&prompt, 32).expect("chat generate failed");
        println!("\nUser: {}\nAssistant: {}", q, resp);
        history.push(mivi_tokenizer::ChatMessage {
            role: mivi_tokenizer::Role::Assistant,
            content: Some(resp),
            name: None,
        });
    }

    assert_eq!(history.len(), 6);
}

#[tokio::test]
async fn test_chat_completion_with_custom_sampling_params() {
    let broker = ToolBroker::new();
    let engine = mivi_server::EngineActor::spawn(None);
    let state = Arc::new(AppState::new("mivi-v4-test", broker, engine, None));
    let app = create_router(state);

    let chat_req = serde_json::json!({
        "model": "mivi-v4-test",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 16
    });

    let req = Request::builder()
        .uri("/v1/chat/completions")
        .method("POST")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_vec(&chat_req).unwrap()))
        .unwrap();

    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}

#[test]
fn test_special_token_decoding_comprehensive() {
    let vocab = mivi_tokenizer::Vocab::new(vec![
        "<|im_start|>".to_string(),
        "<|im_end|>".to_string(),
        "<think>".to_string(),
        "</think>".to_string(),
        "<tool_call>".to_string(),
        "</tool_call>".to_string(),
        "hello".to_string(),
    ]);
    let tokenizer = mivi_tokenizer::Tokenizer::new(vocab, std::collections::HashMap::new());

    let decoded = tokenizer.decode(&[0, 6, 1, 2, 6, 3]);
    assert_eq!(decoded, "<|im_start|>hello<|im_end|><think>hello</think>");
}

#[tokio::test]
async fn test_server_blocking_chat_with_tool_calls_extraction() {
    let broker = ToolBroker::new();
    let engine = mivi_server::EngineActor::spawn(None);
    let state = Arc::new(AppState::new("mivi-v4-test", broker, engine, None));
    let app = create_router(state);

    let chat_req = serde_json::json!({
        "model": "mivi-v4-test",
        "messages": [
            {"role": "user", "content": "What is 2+2?"}
        ],
        "max_tokens": 32
    });

    let req = Request::builder()
        .uri("/v1/chat/completions")
        .method("POST")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_vec(&chat_req).unwrap()))
        .unwrap();

    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body_bytes = axum::body::to_bytes(res.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let json_res: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(json_res["choices"][0]["finish_reason"], "stop");
}

#[test]
fn test_prefix_caching_hybrid_state_acceleration() {
    let model_path = PathBuf::from("models/mivi-v4-q4_k_m.gguf");
    if !model_path.exists() {
        println!("Skipping real model prefix cache test: model file not found");
        return;
    }

    let mut model = mivi_model::Model::load(&model_path).expect("Failed to load model");
    model.sampler.config.temperature = 0.0;

    // Create a long prompt with >= 64 tokens (1 full chunk)
    let long_system_prefix = "You are Mivi, an intelligent, concise, and helpful AI assistant designed for high-performance software engineering and systems programming in pure Rust. Always write clean code, follow standard formatting practices, verify all edge cases, and explain your technical reasoning clearly and concisely to the developer.";
    let prompt_a = format!("<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n", long_system_prefix);

    // Turn 1: Fresh prompt, fills prefix cache
    model.reset_context();
    let resp_1 = model.generate(&prompt_a, 16).expect("Generation 1 failed");
    assert!(!resp_1.is_empty());
    assert!(model.prefix_cache.len() >= 1, "PrefixCache must have cached at least 1 chunk");

    // Turn 2: Same long system prefix, new user question
    let prompt_b = format!("<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\nWhat is 3+3?<|im_end|>\n<|im_start|>assistant\n", long_system_prefix);
    model.reset_context();
    let resp_2 = model.generate(&prompt_b, 16).expect("Generation 2 with prefix cache failed");
    assert!(!resp_2.is_empty());
}

#[test]
fn test_disk_kvc_persistence_integration() {
    let tmp_dir = std::env::temp_dir().join(format!("mivi_kvc_integ_{}", std::process::id()));
    let file_path = tmp_dir.join("system_prompt.kvc");

    let tokens: Vec<u32> = (1..=128).collect();
    let snapshot = mivi_kv::HybridStateSnapshot::new(
        128,
        vec![1.0; 128 * 4],
        vec![2.0; 128 * 4],
        vec![0.5; 32],
        vec![],
    );
    let model_hash = 0xDEAD_BEEF_CAFE_BABE;

    // Save
    assert!(mivi_kv::save_to_disk(&file_path, &tokens, &snapshot, model_hash).is_ok());

    // List
    let files = mivi_kv::list_cached_files(Some(&tmp_dir)).unwrap();
    assert_eq!(files.len(), 1);
    assert_eq!(files[0].token_count, 128);
    assert_eq!(files[0].model_hash, model_hash);

    // Load
    let (loaded_tokens, loaded_snapshot) = mivi_kv::load_from_disk(&file_path, Some(model_hash)).unwrap();
    assert_eq!(loaded_tokens, tokens);
    assert_eq!(loaded_snapshot, snapshot);

    // Clear
    let removed = mivi_kv::clear_cache_dir(Some(&tmp_dir)).unwrap();
    assert_eq!(removed, 1);
    assert_eq!(mivi_kv::list_cached_files(Some(&tmp_dir)).unwrap().len(), 0);

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[test]
fn test_semantic_anchor_checkpointing_and_rollback() {
    let mut cache = mivi_kv::SemanticAnchorCache::new(16);
    let mock_snapshot = mivi_kv::HybridStateSnapshot::new(32, vec![0.1; 64], vec![0.2; 64], vec![0.3; 16], vec![]);

    let prompt_tokens: Vec<u32> = (1..=32).collect();
    cache.insert_anchor(
        mivi_kv::SemanticAnchorType::TurnAssistant,
        32,
        &prompt_tokens,
        mock_snapshot.clone(),
    );

    assert_eq!(cache.len(), 1);

    // Query with longer conversational token stream
    let longer_stream: Vec<u32> = (1..=64).collect();
    let (matched_pos, anchor) = cache.find_deepest_anchor(&longer_stream).expect("Should match semantic anchor");
    assert_eq!(matched_pos, 32);
    assert_eq!(anchor.anchor_type, mivi_kv::SemanticAnchorType::TurnAssistant);
    assert_eq!(anchor.state, mock_snapshot);
}

#[tokio::test]
async fn test_anthropic_messages_endpoint() {
    let broker = ToolBroker::new();
    let engine = mivi_server::EngineActor::spawn(None);
    let state = Arc::new(AppState::new("mivi-v4-test", broker, engine, None));
    let app = create_router(state);

    let anthropic_req = serde_json::json!({
        "model": "mivi-v4-test",
        "messages": [
            {"role": "user", "content": "Tell me about Rust."}
        ],
        "max_tokens": 16,
        "stream": false
    });

    let req = Request::builder()
        .uri("/v1/messages")
        .method("POST")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_vec(&anthropic_req).unwrap()))
        .unwrap();

    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let body = axum::body::to_bytes(res.into_body(), 1024 * 1024).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["type"], "message");
    assert_eq!(json["role"], "assistant");
    assert!(json["content"].is_array());
}

#[test]
fn test_elastic_memory_pruning_under_pressure() {
    let mut cache = mivi_kv::PrefixCache::new(10, 64);
    for i in 0..5 {
        let chunk_tokens: Vec<u32> = (i * 64..(i + 1) * 64).collect();
        let snapshot = mivi_kv::HybridStateSnapshot::new(
            64,
            vec![1.0; 1000],
            vec![2.0; 1000],
            vec![0.5; 100],
            vec![],
        );
        cache.insert_chunk(i as u64, &chunk_tokens, i as usize, snapshot);
    }

    assert_eq!(cache.len(), 5);
    let initial_memory = cache.memory_usage_bytes();
    assert!(initial_memory > 0);

    // Prune to half memory budget
    let target = initial_memory / 2;
    let evicted = cache.prune_to_bytes(target);
    assert!(evicted > 0);
    assert!(cache.memory_usage_bytes() <= target);
}
