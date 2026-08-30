//! Agent task execution endpoint supporting tool loops and SSE streaming.

use crate::routes::chat::sse_response;
use crate::state::AppState;
use crate::streaming::*;
use crate::types::*;
use axum::{
    extract::{Json, State},
    response::{sse::Event, Response},
};
use std::sync::Arc;
use tokio::sync::mpsc;

pub const AGENT_PROMPT_TEMPLATE: &str =
    "<user_request>\n{}\n</user_request>\nFormulate a plan and call appropriate tools if necessary.";

pub async fn run_agent_task(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AgentRunRequest>,
) -> Response {
    let model_name = state.model_name.clone();
    let cid = format!("{}{}", mivi_core::AGENT_RUN_ID_PREFIX, uuid::Uuid::new_v4());
    let broker = state.broker.clone();
    let engine = state.engine.clone();
    let channel_capacity = state.config.channel_capacity;
    let default_max_steps = state.config.default_max_agent_steps;
    let agent_gen_tokens = state.config.agent_gen_tokens;

    let (tx, rx) =
        mpsc::channel::<std::result::Result<Event, std::convert::Infallible>>(channel_capacity);
    let cid_clone = cid.clone();
    let mname = model_name.clone();

    tokio::spawn(async move {
        let max_steps = if req.max_steps == 0 {
            default_max_steps
        } else {
            req.max_steps
        };
        let agent_state = mivi_agent::AgentState::new(&req.task, max_steps);
        let mut agent = mivi_agent::AgentLoop::new(agent_state, &broker);
        let thinking_msg = format!("Initializing agent for task: '{}'", req.task);

        send_sse_sequence(&tx, &cid_clone, &mname, Some(&thinking_msg), || async {
            let sanitized_task = mivi_agent::escape_xml_content(&req.task);
            let mut current_prompt = AGENT_PROMPT_TEMPLATE.replace("{}", &sanitized_task);

            for _ in 0..max_steps {
                match engine.generate(&current_prompt, agent_gen_tokens).await {
                    Ok((model_out, _, _)) => {
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

                        current_prompt = result;
                    }
                    Err(e) => {
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
        })
        .await;
    });

    sse_response(rx)
}
