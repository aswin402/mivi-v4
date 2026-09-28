//! Agent task execution endpoint supporting tool loops and SSE streaming.

use crate::engine_actor::GenerationCancellation;
use crate::generation::{
    forced_tool_call_prefix, validate_additional_sampling_parameters, validate_sampling_parameters,
    GenerationOptions,
};
use crate::model_profile::ModelProfile;
use crate::routes::chat::sse_response;
use crate::state::AppState;
use crate::streaming::*;
use crate::types::*;
use axum::{
    extract::{Json, State},
    response::{sse::Event, IntoResponse, Response},
};
use mivi_protocol::Message;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

#[cfg(test)]
mod context_tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn context_documents_reject_paths_outside_workspace() {
        let result =
            load_context_documents(Path::new("/workspace"), &["../../secret.txt".to_string()]);
        assert!(result.is_err());
    }

    #[cfg(unix)]
    #[test]
    fn context_documents_reject_symlinked_files_and_directories() {
        use std::fs;
        use std::os::unix::fs::symlink;
        use std::time::{SystemTime, UNIX_EPOCH};

        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must be after the Unix epoch")
            .as_nanos();
        let workspace = std::env::temp_dir().join(format!(
            "mivi-server-context-{}-{}",
            std::process::id(),
            suffix
        ));
        fs::create_dir(&workspace).expect("create isolated workspace");
        fs::write(workspace.join("real.txt"), "workspace context").expect("create target file");
        fs::create_dir(workspace.join("real-dir")).expect("create target directory");
        symlink("real.txt", workspace.join("linked.txt")).expect("create file symlink");
        symlink("real-dir", workspace.join("linked-dir")).expect("create directory symlink");

        let linked_file = load_context_documents(&workspace, &["linked.txt".to_string()]);
        let linked_directory =
            load_context_documents(&workspace, &["linked-dir/context.txt".to_string()]);

        assert!(linked_file.is_err());
        assert!(linked_directory.is_err());
        fs::remove_file(workspace.join("linked.txt")).unwrap();
        fs::remove_file(workspace.join("linked-dir")).unwrap();
        fs::remove_dir(workspace.join("real-dir")).unwrap();
        fs::remove_file(workspace.join("real.txt")).unwrap();
        fs::remove_dir(workspace).unwrap();
    }
}

#[cfg(test)]
mod prompt_tests {
    use super::native_agent_user_message;

    #[test]
    fn native_agent_prompt_preserves_task_without_optional_planning_instruction() {
        let message = native_agent_user_message(
            "Use the calculator tool to compute 45 * 12, then report only the final number.",
            "",
        );

        assert_eq!(
            message.content.as_deref(),
            Some("Use the calculator tool to compute 45 * 12, then report only the final number.")
        );
    }
}

const MAX_CONTEXT_DOCS: usize = 16;
const MAX_CONTEXT_DOC_BYTES: u64 = 512 * 1024;
const MAX_CONTEXT_TOTAL_BYTES: usize = 2 * 1024 * 1024;
const MAX_AGENT_TOOL_CALL_RETRIES: usize = 3;

fn canonical_builtin_tool_definitions(
    allowed_tools: Option<&[String]>,
) -> Vec<mivi_protocol::ToolDefinition> {
    mivi_tools::get_builtin_tool_definitions()
        .into_iter()
        .filter(|tool| {
            allowed_tools
                .is_none_or(|allowed| allowed.iter().any(|name| name == &tool.function.name))
        })
        .map(|tool| mivi_protocol::ToolDefinition {
            name: tool.function.name,
            description: tool.function.description,
            parameters: tool.function.parameters,
        })
        .collect()
}

fn native_agent_user_message(task: &str, context_prompt: &str) -> Message {
    let mut content = task.to_string();
    if !context_prompt.is_empty() {
        content.push('\n');
        content.push_str(context_prompt);
    }
    Message {
        role: "user".to_string(),
        content: Some(content),
        name: None,
        tool_call_id: None,
        tool_calls: Vec::new(),
        reasoning: None,
    }
}

fn native_agent_tool_retry_message() -> Message {
    Message {
        role: "user".to_string(),
        content: Some(
            "The previous response did not contain a tool call. Call one of the available tools before producing a final answer."
                .to_string(),
        ),
        name: None,
        tool_call_id: None,
        tool_calls: Vec::new(),
        reasoning: None,
    }
}

fn append_native_agent_turn(
    messages: &mut Vec<Message>,
    profile: &ModelProfile,
    model_output: &str,
    tool_results: &[mivi_tools::ToolResult],
) -> Result<(), String> {
    let tool_calls = profile.tool_codec().extract(model_output)?;
    let clean_text = profile
        .tool_codec()
        .strip(&mivi_tools::strip_thinking(model_output));
    messages.push(Message {
        role: "assistant".to_string(),
        content: (!clean_text.is_empty()).then_some(clean_text),
        name: None,
        tool_call_id: None,
        tool_calls: tool_calls
            .into_iter()
            .map(|call| mivi_protocol::ToolCall {
                id: None,
                name: call.name,
                arguments: call.arguments,
            })
            .collect(),
        reasoning: None,
    });

    for result in tool_results {
        let content = if result.success {
            result.output.clone()
        } else {
            result.error.clone().unwrap_or_default()
        };
        messages.push(Message {
            role: "tool".to_string(),
            content: Some(content),
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
            reasoning: None,
        });
    }
    Ok(())
}

struct AgentGenerationCancellation(GenerationCancellation);

impl AgentGenerationCancellation {
    fn cancel(&self) {
        self.0.cancel();
    }
}

impl Drop for AgentGenerationCancellation {
    fn drop(&mut self) {
        self.cancel();
    }
}

#[allow(clippy::too_many_arguments)]
async fn generate_agent_step(
    engine: &crate::engine_actor::EngineHandle,
    prompt: &str,
    max_tokens: usize,
    admitted_prompt_tokens: Option<usize>,
    options: GenerationOptions,
    request_timeout: Duration,
    first_token_timeout: Duration,
    generation_started: Instant,
    output_events: &mpsc::Sender<std::result::Result<Event, std::convert::Infallible>>,
) -> Result<(String, usize, usize, Option<Duration>), String> {
    let (mut receiver, cancellation) = engine
        .generate_stream_with_options_cancelable(prompt, max_tokens, options)
        .await?;
    let cancellation = AgentGenerationCancellation(cancellation);
    let mut deadline = Box::pin(tokio::time::sleep(request_timeout));
    let mut first_token_deadline = Box::pin(tokio::time::sleep(first_token_timeout));
    let mut first_token_received = false;
    let mut first_token_latency = None;
    let mut output = String::new();

    loop {
        tokio::select! {
            _ = &mut first_token_deadline, if !first_token_received => {
                cancellation.cancel();
                return Err("Model did not produce a first token before the configured deadline.".to_string());
            }
            _ = &mut deadline => {
                cancellation.cancel();
                return Err("Inference request timed out.".to_string());
            }
            _ = output_events.closed() => {
                cancellation.cancel();
                return Err("Agent client disconnected.".to_string());
            }
            chunk = receiver.recv() => {
                match chunk {
                    Some(Ok(chunk)) => {
                        if !chunk.is_empty() && !first_token_received {
                            first_token_latency = Some(generation_started.elapsed());
                            first_token_received = true;
                        }
                        output.push_str(&chunk);
                    }
                    Some(Err(error)) => return Err(format!("Inference error: {error}")),
                    None => break,
                }
            }
        }
    }

    let prompt_tokens = match admitted_prompt_tokens {
        Some(prompt_tokens) => prompt_tokens,
        None => engine.encode(prompt).await.len(),
    };
    let completion_tokens = engine.encode(&output).await.len();
    Ok((
        output,
        prompt_tokens,
        completion_tokens,
        first_token_latency,
    ))
}

fn load_context_documents(workspace: &Path, paths: &[String]) -> Result<String, String> {
    if paths.len() > MAX_CONTEXT_DOCS {
        return Err(format!(
            "context_docs supports at most {} documents",
            MAX_CONTEXT_DOCS
        ));
    }

    let mut total_bytes = 0usize;
    let mut rendered = String::new();
    for path in paths {
        let content =
            mivi_tools::builtins::read_workspace_file(workspace, path, MAX_CONTEXT_DOC_BYTES)
                .map_err(|e| format!("Failed to read context document '{}': {}", path, e))?;
        total_bytes = total_bytes
            .checked_add(content.len())
            .ok_or_else(|| "Total context document size overflowed".to_string())?;
        if total_bytes > MAX_CONTEXT_TOTAL_BYTES {
            return Err(format!(
                "context_docs exceeds the {} byte total limit",
                MAX_CONTEXT_TOTAL_BYTES
            ));
        }
        rendered.push_str("\n<context_document path=\"");
        rendered.push_str(&mivi_agent::escape_xml_attr(path));
        rendered.push_str("\">\n");
        rendered.push_str(&mivi_agent::escape_xml_content(&content));
        rendered.push_str("\n</context_document>\n");
    }
    Ok(rendered)
}

pub async fn run_agent_task(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AgentRunRequest>,
) -> Response {
    if !state.engine.has_model() {
        return crate::types::AppError::ServiceUnavailable("No model is loaded".to_string())
            .into_response();
    }
    let profile = match ModelProfile::resolve(
        state.engine.model_metadata(),
        state.config.model_profile.as_ref(),
    ) {
        Ok(profile) => profile,
        Err(message) => return crate::types::AppError::InvalidRequest(message).into_response(),
    };
    if !profile.supports_tools() {
        return crate::types::AppError::InvalidRequest(
            "selected text-only model profile cannot run the tool-based agent endpoint".to_string(),
        )
        .into_response();
    }
    if let Err(message) = validate_sampling_parameters(
        req.temperature,
        req.top_p,
        req.presence_penalty,
        req.frequency_penalty,
    ) {
        return crate::types::AppError::InvalidRequest(message).into_response();
    }
    if let Err(message) =
        validate_additional_sampling_parameters(req.top_k, req.min_p, req.repetition_penalty)
    {
        return crate::types::AppError::InvalidRequest(message).into_response();
    }
    let profile_codec = profile.tool_codec_handle();
    let allowed_tools = req.allowed_tools.clone();
    let canonical_tools = canonical_builtin_tool_definitions(allowed_tools.as_deref());
    if req.tool_choice == AgentToolChoice::Required && canonical_tools.is_empty() {
        return crate::types::AppError::InvalidRequest(
            "tool_choice 'required' requires at least one available tool".to_string(),
        )
        .into_response();
    }
    let require_tool_call = req.tool_choice == AgentToolChoice::Required;
    let forced_output_prefix = if require_tool_call {
        forced_tool_call_prefix(
            &crate::generation::ToolChoice::Required,
            profile.tool_codec(),
        )
    } else {
        None
    };
    let agent_generation_options = GenerationOptions {
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        min_p: req.min_p,
        repetition_penalty: req.repetition_penalty,
        presence_penalty: req.presence_penalty,
        frequency_penalty: req.frequency_penalty,
        seed: req.seed,
        forced_output_prefix,
        ..GenerationOptions::default()
    };
    let model_name = state.model_name.clone();
    let cid = format!("{}{}", mivi_core::AGENT_RUN_ID_PREFIX, uuid::Uuid::new_v4());
    let broker = state.broker.clone();
    let engine = state.engine.clone();
    let context_prompt = match load_context_documents(
        &state.workspace,
        req.context_docs.as_deref().unwrap_or(&[]),
    ) {
        Ok(context) => context,
        Err(message) => return crate::types::AppError::InvalidRequest(message).into_response(),
    };
    let slot_wait_started = Instant::now();
    let inference_permit = match state.try_acquire_inference_slot() {
        Ok(permit) => permit,
        Err(_) => {
            state.metrics.record_inference_rejected();
            return crate::types::AppError::TooManyRequests(
                "The maximum number of concurrent inference requests is active".to_string(),
            )
            .into_response();
        }
    };
    state
        .metrics
        .record_inference_accepted(slot_wait_started.elapsed());
    let channel_capacity = state.config.channel_capacity;
    let default_max_steps = state.config.default_max_agent_steps;
    let agent_gen_tokens = state.config.agent_gen_tokens;
    let request_timeout = Duration::from_secs(state.config.request_timeout_secs);
    let first_token_timeout = Duration::from_secs(state.config.first_token_timeout_secs);
    let metrics = state.metrics.clone();

    let (tx, rx) =
        mpsc::channel::<std::result::Result<Event, std::convert::Infallible>>(channel_capacity);
    let cid_clone = cid.clone();
    let mname = model_name.clone();
    let task_summary = crate::logging::summarize_prompt(&req.task, 40);

    let task_for_log = req.task.clone();
    tokio::spawn(async move {
        let _inference_permit = inference_permit;
        const MAX_AGENT_STEPS_LIMIT: usize = 50;
        let max_steps = if req.max_steps == 0 {
            default_max_steps
        } else {
            req.max_steps.min(MAX_AGENT_STEPS_LIMIT)
        };
        let agent_state = mivi_agent::AgentState::new(&req.task, max_steps);
        let mut agent = mivi_agent::AgentLoop::new(agent_state, &broker)
            .with_allowed_tools(allowed_tools)
            .with_tool_codec(profile_codec)
            .with_required_tool_call(require_tool_call);
        let thinking_msg = format!("Initializing agent for task: '{}'", req.task);

        send_sse_sequence_with_finish(&tx, &cid_clone, &mname, Some(&thinking_msg), || async {
            let mut finish_reason = "stop";
            let mut native_messages = vec![native_agent_user_message(&req.task, &context_prompt)];
            let mut tool_call_retries_remaining =
                req.tool_call_retries.min(MAX_AGENT_TOOL_CALL_RETRIES);
            let mut current_prompt =
                match profile.render_prompt(&native_messages, &canonical_tools, false) {
                    Ok(prompt) => prompt,
                    Err(error) => {
                        metrics.record_inference_error();
                        let _ = tx
                            .send(Ok(create_error_chunk_event(&cid_clone, &mname, &error)))
                            .await;
                        return "error";
                    }
                };

            let mut agent_steps = 0;
            while agent_steps < max_steps {
                let admitted_prompt_tokens = match crate::routes::admit_model_context(
                    &engine,
                    &current_prompt,
                    agent_gen_tokens,
                )
                .await
                {
                    Ok(prompt_tokens) => prompt_tokens,
                    Err(error) => {
                        metrics.record_inference_error();
                        tracing::warn!(%error, "Agent prompt exceeds model context");
                        let _ = tx
                            .send(Ok(create_error_chunk_event(
                                &cid_clone,
                                &mname,
                                &error.to_string(),
                            )))
                            .await;
                        finish_reason = "error";
                        break;
                    }
                };

                let generation_started = Instant::now();
                let generation_options = if agent.last_tool_results().is_empty() {
                    agent_generation_options.clone()
                } else {
                    let mut options = agent_generation_options.clone();
                    options.forced_output_prefix = None;
                    options
                };
                match generate_agent_step(
                    &engine,
                    &current_prompt,
                    agent_gen_tokens,
                    admitted_prompt_tokens,
                    generation_options,
                    request_timeout,
                    first_token_timeout,
                    generation_started,
                    &tx,
                )
                .await
                {
                    Ok((model_out, prompt_tokens, completion_tokens, first_token_latency)) => {
                        metrics.record_generation(generation_started.elapsed());
                        if let Some(first_token_latency) = first_token_latency {
                            metrics.record_time_to_first_token(first_token_latency);
                        }
                        metrics.record_tokens(prompt_tokens, completion_tokens);
                        let result = agent.step(&model_out).await;

                        if agent.required_tool_call_missing() && tool_call_retries_remaining > 0 {
                            tool_call_retries_remaining -= 1;
                            let _ = agent.retry_required_tool_call();
                            native_messages.push(native_agent_tool_retry_message());
                            match profile.render_prompt(&native_messages, &canonical_tools, false) {
                                Ok(prompt) => {
                                    current_prompt = prompt;
                                    continue;
                                }
                                Err(error) => {
                                    metrics.record_inference_error();
                                    let _ = tx
                                        .send(Ok(create_error_chunk_event(
                                            &cid_clone, &mname, &error,
                                        )))
                                        .await;
                                    finish_reason = "error";
                                    break;
                                }
                            }
                        }

                        agent_steps += 1;
                        if tx
                            .send(Ok(create_content_chunk_event(&cid_clone, &mname, &result)))
                            .await
                            .is_err()
                        {
                            if agent.state.phase == mivi_agent::AgentPhase::Failed {
                                finish_reason = "error";
                            }
                            break;
                        }

                        if agent.state.phase == mivi_agent::AgentPhase::Failed {
                            finish_reason = "error";
                            break;
                        }
                        if agent.state.phase == mivi_agent::AgentPhase::Completed {
                            break;
                        }

                        if let Err(error) = append_native_agent_turn(
                            &mut native_messages,
                            &profile,
                            &model_out,
                            agent.last_tool_results(),
                        ) {
                            metrics.record_inference_error();
                            let _ = tx
                                .send(Ok(create_error_chunk_event(&cid_clone, &mname, &error)))
                                .await;
                            finish_reason = "error";
                            break;
                        } else {
                            match profile.render_prompt(&native_messages, &canonical_tools, false) {
                                Ok(prompt) => current_prompt = prompt,
                                Err(error) => {
                                    metrics.record_inference_error();
                                    let _ = tx
                                        .send(Ok(create_error_chunk_event(
                                            &cid_clone, &mname, &error,
                                        )))
                                        .await;
                                    finish_reason = "error";
                                    break;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        metrics.record_generation(generation_started.elapsed());
                        metrics.record_inference_error();
                        tracing::error!("Agent inference failed: {}", e);
                        let _ = tx
                            .send(Ok(create_error_chunk_event(&cid_clone, &mname, &e)))
                            .await;
                        finish_reason = "error";
                        break;
                    }
                }
            }

            let last_reply = agent
                .state
                .memory
                .back()
                .cloned()
                .unwrap_or_else(|| "Completed agent task execution.".to_string());
            metrics.record_tool_timeouts(agent.timed_out_tools);
            crate::logging::print_interaction_box(
                Some(&task_for_log),
                None,
                None,
                Some(&last_reply),
                true,
            );
            finish_reason
        })
        .await;
    });

    let mut resp = sse_response(rx);
    let log_meta = crate::logging::LogMetadata {
        prompt_summary: Some(task_summary),
        is_agent: true,
        is_streaming: true,
        stream_metrics: Some(state.metrics.clone()),
        ..Default::default()
    };
    resp.extensions_mut().insert(log_meta);
    resp
}
