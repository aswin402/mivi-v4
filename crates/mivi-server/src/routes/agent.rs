//! Agent task execution endpoint supporting tool loops and SSE streaming.

use crate::routes::chat::sse_response;
use crate::state::AppState;
use crate::streaming::*;
use crate::types::*;
use axum::{
    extract::{Json, State},
    response::{sse::Event, IntoResponse, Response},
};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;

pub const AGENT_PROMPT_TEMPLATE: &str =
    "<user_request>\n{}\n</user_request>\nFormulate a plan and call appropriate tools if necessary.";

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

const MAX_CONTEXT_DOCS: usize = 16;
const MAX_CONTEXT_DOC_BYTES: u64 = 512 * 1024;
const MAX_CONTEXT_TOTAL_BYTES: usize = 2 * 1024 * 1024;

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
    let model_name = state.model_name.clone();
    let cid = format!("{}{}", mivi_core::AGENT_RUN_ID_PREFIX, uuid::Uuid::new_v4());
    let broker = state.broker.clone();
    let engine = state.engine.clone();
    let allowed_tools = req.allowed_tools.clone();
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
        let mut agent =
            mivi_agent::AgentLoop::new(agent_state, &broker).with_allowed_tools(allowed_tools);
        let thinking_msg = format!("Initializing agent for task: '{}'", req.task);

        send_sse_sequence(&tx, &cid_clone, &mname, Some(&thinking_msg), || async {
            let sanitized_task = mivi_agent::escape_xml_content(&req.task);
            let base_prompt = format!(
                "{}\n{}",
                AGENT_PROMPT_TEMPLATE.replace("{}", &sanitized_task),
                context_prompt
            );
            let mut conversation_history = String::new();
            let mut current_prompt = base_prompt.clone();

            for _ in 0..max_steps {
                let generation_started = Instant::now();
                match engine.generate(&current_prompt, agent_gen_tokens).await {
                    Ok((model_out, prompt_tokens, completion_tokens)) => {
                        metrics.record_generation(generation_started.elapsed());
                        metrics.record_tokens(prompt_tokens, completion_tokens);
                        let result = agent.step(&model_out).await;
                        if tx
                            .send(Ok(create_content_chunk_event(&cid_clone, &mname, &result)))
                            .await
                            .is_err()
                        {
                            break;
                        }

                        if agent.state.phase == mivi_agent::AgentPhase::Completed
                            || agent.state.phase == mivi_agent::AgentPhase::Failed
                        {
                            break;
                        }

                        conversation_history.push_str(&format!(
                            "\n<assistant>\n{}\n</assistant>\n{}\n",
                            model_out, result
                        ));
                        current_prompt = format!(
                            "{}\n{}\nContinue with the task. If complete, indicate the final answer.",
                            base_prompt, conversation_history
                        );
                    }
                    Err(e) => {
                        metrics.record_generation(generation_started.elapsed());
                        metrics.record_inference_error();
                        tracing::error!("Agent inference failed: {}", e);
                        let _ = tx
                            .send(Ok(create_error_chunk_event(
                                &cid_clone,
                                &mname,
                                "Agent inference failed. Check server logs.",
                            )))
                            .await;
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
        })
        .await;
    });

    let mut resp = sse_response(rx);
    let log_meta = crate::logging::LogMetadata {
        prompt_summary: Some(task_summary),
        is_agent: true,
        ..Default::default()
    };
    resp.extensions_mut().insert(log_meta);
    resp
}
