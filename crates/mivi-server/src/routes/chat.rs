//! Chat completions HTTP endpoint with blocking and SSE streaming modes.

use crate::engine_actor::EngineHandle;
use crate::generation::{
    filter_tools_for_choice, forced_tool_call_prefix, parse_response_mode, parse_stop_sequences,
    parse_tool_choice, validate_additional_sampling_parameters, validate_openai_tool_definitions,
    validate_sampling_parameters, validate_tool_calls_against_tools, validate_tool_choice_result,
    GenerationOptions, ResponseMode, ToolChoice,
};
use crate::model_profile::ModelProfile;
use crate::state::AppState;
use crate::streaming::*;
use crate::types::*;
use axum::{
    extract::{Json, State},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, OwnedSemaphorePermit};
use tokio_stream::wrappers::ReceiverStream;

#[inline]
pub(crate) fn sse_response(
    rx: mpsc::Receiver<std::result::Result<Event, std::convert::Infallible>>,
) -> Response {
    let stream = ReceiverStream::new(rx);
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    if req.messages.is_empty() {
        return AppError::InvalidRequest("messages array cannot be empty".to_string())
            .into_response();
    }
    if req.messages.len() > state.config.max_messages {
        return AppError::InvalidRequest(format!(
            "messages array exceeds limit of {} items",
            state.config.max_messages
        ))
        .into_response();
    }
    if !state.engine.has_model() {
        return AppError::ServiceUnavailable("No model is loaded".to_string()).into_response();
    }
    if !crate::model_matches(req.model.as_deref(), &state.model_name) {
        return AppError::InvalidRequest(format!(
            "Unknown model '{}'; loaded model is '{}'",
            req.model.as_deref().unwrap_or_default(),
            state.model_name
        ))
        .into_response();
    }

    if let Err(message) = validate_sampling_parameters(
        req.temperature,
        req.top_p,
        req.presence_penalty,
        req.frequency_penalty,
    ) {
        return AppError::InvalidRequest(message).into_response();
    }
    if let Err(message) =
        validate_additional_sampling_parameters(req.top_k, req.min_p, req.repetition_penalty)
    {
        return AppError::InvalidRequest(message).into_response();
    }
    let response_mode = match parse_response_mode(req.response_format.as_ref()) {
        Ok(mode) => mode,
        Err(message) => return AppError::InvalidRequest(message).into_response(),
    };
    let is_streaming = req.stream.unwrap_or(false);
    if is_streaming && response_mode == ResponseMode::JsonObject {
        return AppError::InvalidRequest(
            "json_object response format is not supported for streaming".to_string(),
        )
        .into_response();
    }
    if let Err(message) = validate_openai_tool_definitions(req.tools.as_deref()) {
        return AppError::InvalidRequest(message).into_response();
    }
    let tool_choice = match parse_tool_choice(req.tools.as_deref(), req.tool_choice.as_ref()) {
        Ok(choice) => choice,
        Err(message) => return AppError::InvalidRequest(message).into_response(),
    };
    let tools = filter_tools_for_choice(req.tools.clone(), &tool_choice);
    let tool_calls_enabled = tool_choice.allows_tool_calls(tools.as_deref());
    let stop_tokens = match parse_stop_sequences(req.stop.as_ref()) {
        Ok(stops) => stops,
        Err(message) => return AppError::InvalidRequest(message).into_response(),
    };
    let mut options = GenerationOptions {
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        min_p: req.min_p,
        repetition_penalty: req.repetition_penalty,
        presence_penalty: req.presence_penalty,
        frequency_penalty: req.frequency_penalty,
        seed: req.seed,
        stop_tokens,
        forced_output_prefix: None,
        response_mode,
    };

    let max_tokens = req
        .max_tokens
        .unwrap_or(state.config.default_max_tokens)
        .min(state.config.max_allowed_tokens);
    let slot_wait_started = Instant::now();
    let inference_permit = match state.try_acquire_inference_slot() {
        Ok(permit) => permit,
        Err(_) => {
            state.metrics.record_inference_rejected();
            return AppError::TooManyRequests(
                "The maximum number of concurrent inference requests is active".to_string(),
            )
            .into_response();
        }
    };
    state
        .metrics
        .record_inference_accepted(slot_wait_started.elapsed());
    let completion_id = format!("{}{}", mivi_core::CHATCMPL_ID_PREFIX, uuid::Uuid::new_v4());
    let last_user_prompt = req
        .messages
        .iter()
        .rev()
        .find(|m| m.role.eq_ignore_ascii_case("user") || m.role.eq_ignore_ascii_case("developer"))
        .and_then(|m| m.content.as_deref())
        .map(|s| crate::logging::summarize_prompt(s, 140));

    let model_name = state.model_name.clone();

    let chat_messages = match req
        .messages
        .iter()
        .map(MessageDto::to_canonical_message)
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(messages) => messages,
        Err(message) => return AppError::InvalidRequest(message).into_response(),
    };
    let canonical_tools = match canonical_tools_from_values(tools.as_deref()) {
        Ok(tools) => tools,
        Err(message) => return AppError::InvalidRequest(message).into_response(),
    };
    let profile = match ModelProfile::resolve(
        state.engine.model_metadata(),
        state.config.model_profile.as_ref(),
    ) {
        Ok(profile) => profile,
        Err(message) => return AppError::InvalidRequest(message).into_response(),
    };
    if !profile.supports_tools()
        && (req.tools.as_ref().is_some_and(|items| !items.is_empty())
            || chat_messages.iter().any(|message| {
                !message.tool_calls.is_empty() || message.role.eq_ignore_ascii_case("tool")
            }))
    {
        return AppError::InvalidRequest(
            "selected model profile does not support tool calls".to_string(),
        )
        .into_response();
    }
    options.forced_output_prefix = if tool_calls_enabled {
        forced_tool_call_prefix(&tool_choice, profile.tool_codec())
    } else {
        None
    };

    let enable_thinking = req
        .reasoning_effort
        .as_deref()
        .map(|r| r != "none")
        .unwrap_or(false);
    let prompt = match profile.render_prompt(&chat_messages, &canonical_tools, enable_thinking) {
        Ok(prompt) => prompt,
        Err(message) => return AppError::InvalidRequest(message).into_response(),
    };

    let admitted_prompt_tokens =
        match crate::routes::admit_model_context(&state.engine, &prompt, max_tokens).await {
            Ok(prompt_tokens) => prompt_tokens,
            Err(error) => return error.into_openai_error().into_response(),
        };

    if let Some(prompt_text) = &last_user_prompt {
        crate::logging::print_incoming_prompt(prompt_text, admitted_prompt_tokens, false);
    }

    if is_streaming {
        let ctx = ChatStreamContext {
            prompt,
            max_tokens,
            options,
            tool_choice,
            tool_calls_enabled,
            tool_definitions: tools.clone(),
            profile: profile.clone(),
            inference_permit,
            completion_id,
            model_name,
            engine: state.engine.clone(),
            channel_capacity: state.config.channel_capacity,
            request_timeout: Duration::from_secs(state.config.request_timeout_secs),
            first_token_timeout: Duration::from_secs(state.config.first_token_timeout_secs),
            metrics: state.metrics.clone(),
        };
        let mut resp = handle_chat_streaming(ctx);
        let log_meta = crate::logging::LogMetadata {
            prompt_summary: last_user_prompt,
            is_streaming: true,
            ..Default::default()
        };
        resp.extensions_mut().insert(log_meta);
        resp
    } else {
        let ctx = ChatBlockingContext {
            prompt: &prompt,
            max_tokens,
            options,
            tool_choice,
            tool_calls_enabled,
            tool_definitions: tools.clone(),
            profile,
            inference_permit,
            completion_id,
            model_name,
            engine: &state.engine,
            metrics: state.metrics.clone(),
        };
        handle_chat_blocking(ctx).await
    }
}

struct ChatStreamContext {
    prompt: String,
    max_tokens: usize,
    options: GenerationOptions,
    tool_choice: ToolChoice,
    tool_calls_enabled: bool,
    tool_definitions: Option<Vec<serde_json::Value>>,
    profile: ModelProfile,
    inference_permit: OwnedSemaphorePermit,
    completion_id: String,
    model_name: String,
    engine: EngineHandle,
    channel_capacity: usize,
    request_timeout: Duration,
    first_token_timeout: Duration,
    metrics: Arc<crate::state::ServerMetrics>,
}

struct ChatBlockingContext<'a> {
    prompt: &'a str,
    max_tokens: usize,
    options: GenerationOptions,
    tool_choice: ToolChoice,
    tool_calls_enabled: bool,
    tool_definitions: Option<Vec<serde_json::Value>>,
    profile: ModelProfile,
    inference_permit: OwnedSemaphorePermit,
    completion_id: String,
    model_name: String,
    engine: &'a EngineHandle,
    metrics: Arc<crate::state::ServerMetrics>,
}

pub const THINKING_INIT_MSG: &str = "Generating completion with Mivi engine...";

fn handle_chat_streaming(ctx: ChatStreamContext) -> Response {
    let (tx, rx) =
        mpsc::channel::<std::result::Result<Event, std::convert::Infallible>>(ctx.channel_capacity);
    let cid = ctx.completion_id;
    let mname = ctx.model_name;
    let engine = ctx.engine;
    let metrics = ctx.metrics;
    let prompt = ctx.prompt;
    let max_tokens = ctx.max_tokens;
    let options = ctx.options;
    let tool_choice = ctx.tool_choice;
    let tool_calls_enabled = ctx.tool_calls_enabled;
    let tool_definitions = ctx.tool_definitions;
    let profile = ctx.profile;
    let inference_permit = ctx.inference_permit;
    let request_timeout = ctx.request_timeout;
    let first_token_timeout = ctx.first_token_timeout;

    tokio::spawn(async move {
        let _inference_permit = inference_permit;
        send_sse_sequence_with_finish(&tx, &cid, &mname, None, || async {
            let generation_started = Instant::now();
            match engine
                .generate_stream_with_options_cancelable(&prompt, max_tokens, options)
                .await
            {
                Ok((mut stream_rx, cancellation)) => {
                    let tool_codec = profile.tool_codec();
                    let mut incremental_text = IncrementalToolText::new(tool_codec);
                    let mut streamed_tool_call = None;
                    let mut stream_failed = false;
                    let mut client_disconnected = false;
                    let mut keepalive = tokio::time::interval(std::time::Duration::from_secs(2));
                    let deadline = tokio::time::sleep(request_timeout);
                    let first_token_deadline = tokio::time::sleep(first_token_timeout);
                    tokio::pin!(deadline);
                    tokio::pin!(first_token_deadline);
                    let mut first_token_received = false;
                    keepalive.tick().await;

                    loop {
                        tokio::select! {
                            _ = &mut first_token_deadline, if !first_token_received => {
                                cancellation.cancel();
                                stream_failed = true;
                                if tx
                                    .send(Ok(create_error_chunk_event(
                                        &cid,
                                        &mname,
                                        "Model did not produce a first token before the configured deadline.",
                                    )))
                                    .await
                                    .is_err()
                                {
                                    client_disconnected = true;
                                }
                                break;
                            }
                            _ = &mut deadline => {
                                cancellation.cancel();
                                stream_failed = true;
                                if tx
                                    .send(Ok(create_error_chunk_event(
                                        &cid,
                                        &mname,
                                        "Inference request timed out.",
                                    )))
                                    .await
                                    .is_err()
                                {
                                    client_disconnected = true;
                                }
                                break;
                            }
                            _ = tx.closed() => {
                                cancellation.cancel();
                                client_disconnected = true;
                                break;
                            }
                            chunk_res = stream_rx.recv() => {
                                match chunk_res {
                                    Some(Ok(word)) => {
                                        if !word.is_empty() {
                                            if !first_token_received {
                                                metrics.record_time_to_first_token(
                                                    generation_started.elapsed(),
                                                );
                                            }
                                            first_token_received = true;
                                        }
                                        let incremental_content = incremental_text.push(&word);
                                        let content = if tool_calls_enabled {
                                            incremental_content
                                        } else {
                                            Some(word)
                                        };
                                        if let Some(content) = content {
                                            if tx
                                                .send(Ok(create_content_chunk_event(
                                                    &cid, &mname, &content,
                                                )))
                                                .await
                                                .is_err()
                                            {
                                                client_disconnected = true;
                                                cancellation.cancel();
                                                break;
                                            }
                                        }
                                        if tool_calls_enabled {
                                            let update = match tool_codec
                                                .stream_update(incremental_text.assembled())
                                            {
                                                Ok(update) => update,
                                                Err(error) => {
                                                    stream_failed = true;
                                                    let _ = tx
                                                        .send(Ok(create_error_chunk_event(
                                                            &cid, &mname, &error,
                                                        )))
                                                        .await;
                                                    break;
                                                }
                                            };
                                            if let Some(update) = update {
                                                let mut deltas = Vec::new();
                                                if tool_choice_allows_name(&tool_choice, &update.name)
                                                {
                                                    if streamed_tool_call.is_none() {
                                                        let id = format!(
                                                            "call_{}",
                                                            uuid::Uuid::new_v4().simple()
                                                        );
                                                        deltas.push(serde_json::json!({
                                                            "index": update.index,
                                                            "id": id,
                                                            "type": "function",
                                                            "function": {
                                                                "name": update.name,
                                                                "arguments": ""
                                                            }
                                                        }));
                                                        streamed_tool_call = Some(
                                                            StreamedToolCall {
                                                                index: update.index,
                                                                name: update.name.clone(),
                                                                arguments_emitted: 0,
                                                            },
                                                        );
                                                    }
                                                    if let Some(streamed) =
                                                        streamed_tool_call.as_mut()
                                                    {
                                                        if streamed.name == update.name
                                                            && update.arguments.len()
                                                                > streamed.arguments_emitted
                                                        {
                                                            let arguments = &update.arguments
                                                                [streamed.arguments_emitted..];
                                                            deltas.push(serde_json::json!({
                                                                "index": streamed.index,
                                                                "function": {
                                                                    "arguments": arguments
                                                                }
                                                            }));
                                                            streamed.arguments_emitted =
                                                                update.arguments.len();
                                                        }
                                                    }
                                                }
                                                if !deltas.is_empty()
                                                    && tx
                                                        .send(Ok(create_tool_calls_chunk_event(
                                                            &cid, &mname, &deltas,
                                                        )))
                                                        .await
                                                        .is_err()
                                                {
                                                    client_disconnected = true;
                                                    cancellation.cancel();
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                    Some(Err(err_msg)) => {
                                        stream_failed = true;
                                        tracing::error!("Inference stream error: {}", err_msg);
                                        if tx
                                            .send(Ok(create_error_chunk_event(
                                                &cid,
                                                &mname,
                                                &format!("Inference error: {err_msg}"),
                                            )))
                                            .await
                                            .is_err()
                                        {
                                            client_disconnected = true;
                                            cancellation.cancel();
                                        }
                                        break;
                                    }
                                    None => break,
                                }
                            }
                            _ = keepalive.tick() => {
                                // Emit SSE comment heartbeat to keep client socket active during CPU prefill
                                if tx.send(Ok(create_keepalive_event())).await.is_err() {
                                    client_disconnected = true;
                                    cancellation.cancel();
                                    break;
                                }
                            }
                        }
                    }
                    if client_disconnected {
                        drop(stream_rx);
                        metrics.record_generation(generation_started.elapsed());
                        return "error";
                    }
                    metrics.record_generation(generation_started.elapsed());
                    if stream_failed {
                        metrics.record_inference_error();
                        return "error";
                    }
                    let assembled = incremental_text.assembled().to_string();
                    if !assembled.is_empty() {
                        if tool_calls_enabled && incremental_text.has_tool_call_start() {
                            // The structured call is emitted below after the complete delimiter
                            // body has been validated by the model profile codec.
                        } else if tool_calls_enabled {
                            let pending = incremental_text.finish();
                            if let Some(pending) = pending {
                                let pending = profile.tool_codec().strip(&pending);
                                if !pending.is_empty()
                                    && tx
                                        .send(Ok(create_content_chunk_event(
                                            &cid, &mname, &pending,
                                        )))
                                        .await
                                        .is_err()
                                {
                                    client_disconnected = true;
                                    cancellation.cancel();
                                }
                            }
                        }
                        if client_disconnected {
                            metrics.record_generation(generation_started.elapsed());
                            return "error";
                        }
                        let thinking = mivi_tools::extract_thinking(&assembled);
                        let clean_reply = mivi_tools::strip_thinking(&assembled);
                        let tool_calls = match extract_tool_calls_for_choice(
                            &assembled,
                            &tool_choice,
                            tool_codec,
                        ) {
                            Ok(tool_calls) => tool_calls,
                            Err(error) => {
                                metrics.record_inference_error();
                                let _ = tx
                                    .send(Ok(create_error_chunk_event(&cid, &mname, &error)))
                                    .await;
                                return "error";
                            }
                        };
                        if let Err(error) = validate_tool_choice_result(&tool_choice, &tool_calls) {
                            metrics.record_inference_error();
                            let _ = tx
                                .send(Ok(create_error_chunk_event(&cid, &mname, &error)))
                                .await;
                            return "error";
                        }
                        if let Err(error) = validate_tool_calls_against_tools(
                            &tool_calls,
                            tool_definitions.as_deref().unwrap_or_default(),
                        ) {
                            tracing::error!(%error, "Model generated invalid tool call");
                            metrics.record_inference_error();
                            let _ = tx
                                .send(Ok(create_error_chunk_event(&cid, &mname, &error)))
                                .await;
                            return "error";
                        }
                        let tc_names: Vec<String> = tool_calls
                            .into_iter()
                            .map(|tc| format!("{}(...)", tc.name))
                            .collect();
                        if tool_calls_enabled {
                            let values = tool_call_values(
                                &assembled,
                                &tool_choice,
                                tool_codec,
                                streamed_tool_call.as_ref(),
                            );
                            if !values.is_empty() {
                                if tx
                                    .send(Ok(create_tool_calls_chunk_event(&cid, &mname, &values)))
                                    .await
                                    .is_err()
                                {
                                    cancellation.cancel();
                                    return "error";
                                }
                            }
                        }
                        crate::logging::print_completion_response_box(
                            thinking.as_deref(),
                            if tc_names.is_empty() {
                                None
                            } else {
                                Some(&tc_names)
                            },
                            Some(&clean_reply),
                        );
                        return if tool_calls_enabled && !tc_names.is_empty() {
                            "tool_calls"
                        } else {
                            "stop"
                        };
                    }
                    if let Err(error) = validate_tool_choice_result(&tool_choice, &[]) {
                        metrics.record_inference_error();
                        let _ = tx
                            .send(Ok(create_error_chunk_event(&cid, &mname, &error)))
                            .await;
                        return "error";
                    }
                }
                Err(e) => {
                    metrics.record_generation(generation_started.elapsed());
                    metrics.record_inference_error();
                    tracing::error!("Failed to start stream: {}", e);
                    let _ = tx
                        .send(Ok(create_error_chunk_event(
                            &cid,
                            &mname,
                            "Failed to initialize streaming.",
                        )))
                        .await;
                    return "error";
                }
            }
            "stop"
        })
        .await;
    });

    sse_response(rx)
}

async fn handle_chat_blocking(ctx: ChatBlockingContext<'_>) -> Response {
    let _inference_permit = ctx.inference_permit;
    let generation_started = Instant::now();
    let generation_result = ctx
        .engine
        .generate_with_options(ctx.prompt, ctx.max_tokens, ctx.options.clone())
        .await;
    ctx.metrics.record_generation(generation_started.elapsed());
    match generation_result {
        Ok((output, p_tokens, c_tokens)) => {
            ctx.metrics.record_tokens(p_tokens, c_tokens);
            let thinking = mivi_tools::extract_thinking(&output);
            let tool_calls_extracted = match extract_tool_calls_for_choice(
                &output,
                &ctx.tool_choice,
                ctx.profile.tool_codec(),
            ) {
                Ok(tool_calls) => tool_calls,
                Err(error) => {
                    ctx.metrics.record_inference_error();
                    return AppError::InferenceError(error).into_response();
                }
            };
            if let Err(error) = validate_tool_choice_result(&ctx.tool_choice, &tool_calls_extracted)
            {
                ctx.metrics.record_inference_error();
                return AppError::InferenceError(error).into_response();
            }
            if let Err(error) = validate_tool_calls_against_tools(
                &tool_calls_extracted,
                ctx.tool_definitions.as_deref().unwrap_or_default(),
            ) {
                ctx.metrics.record_inference_error();
                return AppError::InferenceError(error).into_response();
            }

            let (tool_calls, finish_reason) = if !tool_calls_extracted.is_empty() {
                let tc_vals: Vec<serde_json::Value> = tool_calls_extracted
                    .iter()
                    .enumerate()
                    .map(|(i, tc)| {
                        serde_json::json!({
                            "id": format!("call_{}_{}", i, uuid::Uuid::new_v4().simple()),
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments.to_string(),
                            }
                        })
                    })
                    .collect();
                (Some(tc_vals), "tool_calls")
            } else if c_tokens >= ctx.max_tokens {
                (None, "length")
            } else {
                (None, "stop")
            };

            let content = if tool_calls.is_some() || !ctx.tool_calls_enabled {
                let cleaned = ctx.profile.tool_codec().strip(&output);
                if cleaned.is_empty() {
                    None
                } else {
                    Some(cleaned)
                }
            } else {
                Some(output.clone())
            };

            let response = ChatCompletionResponse {
                id: ctx.completion_id,
                object: OPENAI_COMPLETION_OBJECT.to_string(),
                created: chrono::Utc::now().timestamp() as u64,
                model: ctx.model_name,
                choices: vec![ChoiceDto {
                    index: 0,
                    message: MessageDto {
                        role: ROLE_ASSISTANT.to_string(),
                        content: content.clone(),
                        name: None,
                        thinking: thinking.clone(),
                        tool_calls,
                        tool_call_id: None,
                    },
                    finish_reason: Some(finish_reason.to_string()),
                }],
                usage: UsageDto {
                    prompt_tokens: p_tokens,
                    completion_tokens: c_tokens,
                    total_tokens: p_tokens + c_tokens,
                },
            };

            let mut resp = Json(response).into_response();
            let reply_preview = mivi_tools::strip_thinking(&output);
            let tc_names: Vec<String> = tool_calls_extracted
                .into_iter()
                .map(|tc| format!("{}(...)", tc.name))
                .collect();
            crate::logging::print_completion_response_box(
                thinking.as_deref(),
                if tc_names.is_empty() {
                    None
                } else {
                    Some(&tc_names)
                },
                Some(&reply_preview),
            );
            let log_meta = crate::logging::LogMetadata {
                prompt_summary: None,
                response_summary: Some(reply_preview),
                thinking_summary: thinking,
                tokens_prompt: Some(p_tokens),
                tokens_completion: Some(c_tokens),
                finish_reason: Some(finish_reason.to_string()),
                tool_calls: if tc_names.is_empty() {
                    None
                } else {
                    Some(tc_names)
                },
                ..Default::default()
            };
            resp.extensions_mut().insert(log_meta);
            resp
        }
        Err(e) => {
            ctx.metrics.record_inference_error();
            AppError::InferenceError(e).into_response()
        }
    }
}

fn extract_tool_calls_for_choice(
    output: &str,
    choice: &ToolChoice,
    codec: &dyn mivi_tools::ToolCallCodec,
) -> Result<Vec<mivi_tools::ToolCall>, String> {
    if matches!(choice, ToolChoice::Disabled) {
        return Ok(Vec::new());
    }
    let calls = codec.extract(output)?;
    if let ToolChoice::Named(expected) = choice {
        if let Some(unexpected) = calls.iter().find(|call| {
            call.name != expected.as_str() && call.name != mivi_tools::PARSE_ERROR_TOOL_NAME
        }) {
            return Err(format!(
                "Model emitted tool '{}' but tool_choice requires '{}'",
                unexpected.name, expected
            ));
        }
    }
    Ok(calls
        .into_iter()
        .filter(|call| match choice {
            ToolChoice::Named(name) => {
                call.name == name.as_str() || call.name == mivi_tools::PARSE_ERROR_TOOL_NAME
            }
            ToolChoice::Auto | ToolChoice::Required => true,
            ToolChoice::Disabled => false,
        })
        .collect())
}

fn tool_call_values(
    output: &str,
    choice: &ToolChoice,
    codec: &dyn mivi_tools::ToolCallCodec,
    streamed: Option<&StreamedToolCall>,
) -> Vec<serde_json::Value> {
    extract_tool_calls_for_choice(output, choice, codec)
        .unwrap_or_default()
        .into_iter()
        .enumerate()
        .filter(|(index, _)| streamed.is_none_or(|call| call.index != *index))
        .map(|(index, call)| {
            serde_json::json!({
                "index": index,
                "id": format!("call_{}", uuid::Uuid::new_v4().simple()),
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.arguments.to_string(),
                }
            })
        })
        .collect()
}

struct StreamedToolCall {
    index: usize,
    name: String,
    arguments_emitted: usize,
}

fn tool_choice_allows_name(choice: &ToolChoice, name: &str) -> bool {
    match choice {
        ToolChoice::Auto | ToolChoice::Required => true,
        ToolChoice::Named(expected) => expected == name,
        ToolChoice::Disabled => false,
    }
}

/// Classifies generated text without exposing a partial tool-call marker to the client.
///
/// Text before a tool-call marker is safe to send immediately. The marker itself and everything
/// after it stay buffered until the codec can parse a complete structured call.
struct IncrementalToolText<'a> {
    codec: &'a dyn mivi_tools::ToolCallCodec,
    assembled: String,
    emitted_bytes: usize,
    tool_call_started: bool,
}

impl<'a> IncrementalToolText<'a> {
    fn new(codec: &'a dyn mivi_tools::ToolCallCodec) -> Self {
        Self {
            codec,
            assembled: String::new(),
            emitted_bytes: 0,
            tool_call_started: false,
        }
    }

    fn push(&mut self, chunk: &str) -> Option<String> {
        self.assembled.push_str(chunk);
        self.emit_safe_text(false)
    }

    fn finish(&mut self) -> Option<String> {
        self.emit_safe_text(true)
    }

    fn assembled(&self) -> &str {
        &self.assembled
    }

    fn has_tool_call_start(&self) -> bool {
        self.tool_call_started
    }

    fn emit_safe_text(&mut self, at_end: bool) -> Option<String> {
        if self.tool_call_started {
            return None;
        }

        let Some(delimiter) = self.codec.opening_delimiter() else {
            return self.emit_all_at_end(at_end);
        };
        if delimiter.is_empty() {
            return self.emit_all_at_end(at_end);
        }

        let safe_end = if let Some(start) = self.assembled.find(delimiter) {
            self.tool_call_started = true;
            start
        } else if at_end {
            self.assembled.len()
        } else {
            self.assembled
                .len()
                .saturating_sub(partial_delimiter_suffix_len(&self.assembled, delimiter))
        };

        if safe_end <= self.emitted_bytes {
            return None;
        }

        let text = self.assembled[self.emitted_bytes..safe_end].to_string();
        self.emitted_bytes = safe_end;
        Some(text)
    }

    fn emit_all_at_end(&mut self, at_end: bool) -> Option<String> {
        if !at_end || self.emitted_bytes >= self.assembled.len() {
            return None;
        }
        let text = self.assembled[self.emitted_bytes..].to_string();
        self.emitted_bytes = self.assembled.len();
        Some(text)
    }
}

fn partial_delimiter_suffix_len(text: &str, delimiter: &str) -> usize {
    delimiter
        .char_indices()
        .skip(1)
        .filter_map(|(index, _)| {
            let prefix = &delimiter[..index];
            text.ends_with(prefix).then_some(prefix.len())
        })
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod streaming_tests {
    use super::IncrementalToolText;
    use crate::routes::{validate_context_budget, ContextBudgetError};
    use mivi_tools::DelimitedPythonToolCallCodec;

    #[test]
    fn context_budget_accepts_exact_capacity() {
        assert!(validate_context_budget(6, 2, 8).is_ok());
    }

    #[test]
    fn context_budget_rejects_reserved_output_overflow() {
        let error = validate_context_budget(7, 2, 8).expect_err("budget should be rejected");
        assert!(matches!(
            error,
            ContextBudgetError {
                prompt_tokens: 7,
                requested_output_tokens: 2,
                context_length: 8,
            }
        ));
    }

    #[test]
    fn emits_safe_text_but_holds_split_tool_delimiters_and_body() {
        let codec = DelimitedPythonToolCallCodec::new("<call>", "</call>");
        let mut stream = IncrementalToolText::new(&codec);

        assert_eq!(
            stream.push("I will inspect "),
            Some("I will inspect ".to_string())
        );
        assert_eq!(stream.push("<ca"), None);
        assert_eq!(stream.push("ll>[read_file(path=\"README.md\")"), None);
        assert_eq!(stream.push("]</call>"), None);
        assert!(stream.has_tool_call_start());
        assert_eq!(stream.finish(), None);
    }

    #[test]
    fn flushes_plain_text_at_generation_end() {
        let codec = DelimitedPythonToolCallCodec::new("<call>", "</call>");
        let mut stream = IncrementalToolText::new(&codec);

        assert_eq!(stream.push("plain"), Some("plain".to_string()));
        assert_eq!(stream.finish(), None);
    }
}
