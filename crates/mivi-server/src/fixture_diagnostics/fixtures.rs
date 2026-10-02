use axum::body::Body;
use axum::http::Request;
use futures::StreamExt;
use mivi_model::fixture_diagnostics::{CaptureLimits, CapturedText};
use serde_json::{json, Value};
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tower::ServiceExt;

const SOURCE: &str = "pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n";
const MODEL_TEXT_CAP: usize = 65_536;
const MODEL_TOKEN_CAP: usize = 256;
const STREAM_DEADLINE: Duration = Duration::from_secs(120);
const GENERATION_WATCHDOG: Duration = Duration::from_secs(120);
const MAX_SSE_EVENTS: usize = 512;
const MAX_TOOL_CALLS: usize = 16;
const MAX_TOOL_CALL_FRAGMENTS: usize = 256;

struct ObservedStream {
    first_visible_delta: Option<Duration>,
    elapsed: Duration,
    finish: Option<String>,
    saw_done: bool,
    saw_error: bool,
    calls: Vec<ObservedCall>,
    content: String,
}

struct ObservedCall {
    index: usize,
    id: String,
    name: String,
    arguments: String,
}

async fn drain_fixture_sse(
    body: Body,
    cap: usize,
    started: Instant,
    transcript: &mut CapturedText,
) -> Result<ObservedStream, &'static str> {
    let mut stream = Box::pin(body.into_data_stream());
    let mut pending = Vec::<u8>::new();
    let mut decoder = mivi_tokenizer::Utf8StreamDecoder::new();
    let mut result = ObservedStream {
        first_visible_delta: None,
        elapsed: Duration::ZERO,
        finish: None,
        saw_done: false,
        saw_error: false,
        calls: Vec::new(),
        content: String::new(),
    };
    let mut event_count = 0usize;
    let mut call_fragment_count = 0usize;

    while let Some(frame) = stream.next().await {
        let frame = frame.map_err(|_| "fixture body error")?;
        if frame.len() > cap {
            transcript.push(&decoder.feed(&frame[..cap]));
            transcript.truncated = true;
            return Err("fixture frame exceeds bound");
        }
        transcript.push(&decoder.feed(&frame));
        if transcript.truncated || transcript.counter_overflow {
            return Err("fixture SSE retention exceeds bound");
        }
        let next = pending
            .len()
            .checked_add(frame.len())
            .ok_or("fixture event overflow")?;
        if next > cap {
            return Err("fixture event exceeds bound");
        }
        pending.extend_from_slice(&frame);

        loop {
            let lf = pending
                .windows(2)
                .position(|window| window == b"\n\n")
                .map(|at| (at, 2));
            let crlf = pending
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|at| (at, 4));
            let boundary = match (lf, crlf) {
                (Some(left), Some(right)) => Some(if left.0 < right.0 { left } else { right }),
                (left, right) => left.or(right),
            };
            let Some((end, separator)) = boundary else {
                break;
            };
            event_count = event_count
                .checked_add(1)
                .ok_or("fixture event count overflow")?;
            if event_count > MAX_SSE_EVENTS {
                return Err("fixture event count exceeds bound");
            }

            let event = std::str::from_utf8(&pending[..end]).map_err(|_| "fixture event UTF-8")?;
            let data = event
                .lines()
                .filter_map(|line| line.strip_prefix("data:"))
                .map(|line| line.strip_prefix(' ').unwrap_or(line))
                .collect::<Vec<_>>()
                .join("\n");
            if !data.is_empty() {
                if result.saw_done {
                    return Err("fixture data after DONE");
                }
                if data == "[DONE]" {
                    result.saw_done = true;
                } else {
                    observe_event(&data, started, cap, &mut call_fragment_count, &mut result)?;
                }
            }
            pending.drain(..end + separator);
        }
    }

    transcript.push(&decoder.flush());
    if transcript.truncated || transcript.counter_overflow {
        return Err("fixture SSE retention exceeds bound");
    }
    if !result.saw_done {
        return Err("fixture stream ended before DONE");
    }
    if !pending.is_empty() {
        return Err("fixture incomplete trailing event");
    }
    result.elapsed = started.elapsed();
    result.calls.sort_by_key(|call| call.index);
    Ok(result)
}

fn observe_event(
    data: &str,
    started: Instant,
    cap: usize,
    call_fragment_count: &mut usize,
    result: &mut ObservedStream,
) -> Result<(), &'static str> {
    let value: serde_json::Value = serde_json::from_str(data).map_err(|_| "fixture event JSON")?;
    if value.get("error").is_some() {
        result.saw_error = true;
        return Ok(());
    }
    let choices = value
        .get("choices")
        .and_then(serde_json::Value::as_array)
        .ok_or("fixture choices missing")?;
    if choices.len() != 1 {
        return Err("fixture expects one choice");
    }
    let choice = &choices[0];
    let delta = &choice["delta"];
    let content = delta
        .get("content")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let tool_fragments = delta
        .get("tool_calls")
        .and_then(serde_json::Value::as_array);
    let tool_output = tool_fragments.is_some_and(|calls| {
        calls.iter().any(|call| {
            call.get("function").is_some_and(|function| {
                function
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|name| !name.is_empty())
                    || function
                        .get("arguments")
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|arguments| !arguments.is_empty())
            })
        })
    });
    if (!content.is_empty() || tool_output) && result.first_visible_delta.is_none() {
        result.first_visible_delta = Some(started.elapsed());
    }
    append_field(&mut result.content, content, cap)?;

    if let Some(fragments) = tool_fragments {
        if fragments.len() > MAX_TOOL_CALLS {
            return Err("fixture tool-call count exceeds bound");
        }
        for fragment in fragments {
            *call_fragment_count = call_fragment_count
                .checked_add(1)
                .ok_or("fixture call fragment count overflow")?;
            if *call_fragment_count > MAX_TOOL_CALL_FRAGMENTS {
                return Err("fixture call fragment count exceeds bound");
            }
            let index = fragment
                .get("index")
                .and_then(serde_json::Value::as_u64)
                .and_then(|index| usize::try_from(index).ok())
                .ok_or("fixture tool index")?;
            if index >= MAX_TOOL_CALLS {
                return Err("fixture tool index exceeds bound");
            }
            let position = match result.calls.iter().position(|call| call.index == index) {
                Some(position) => position,
                None => {
                    result.calls.push(ObservedCall {
                        index,
                        id: String::new(),
                        name: String::new(),
                        arguments: String::new(),
                    });
                    result.calls.len() - 1
                }
            };
            let call = &mut result.calls[position];
            if let Some(id) = fragment.get("id").and_then(serde_json::Value::as_str) {
                if !call.id.is_empty() && call.id != id {
                    return Err("fixture conflicting call ID");
                }
                if call.id.is_empty() {
                    append_field(&mut call.id, id, 256)?;
                }
            }
            let function = &fragment["function"];
            append_field(
                &mut call.name,
                function
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or(""),
                256,
            )?;
            append_field(
                &mut call.arguments,
                function
                    .get("arguments")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or(""),
                cap,
            )?;
        }
    }
    if let Some(finish) = choice
        .get("finish_reason")
        .and_then(serde_json::Value::as_str)
    {
        if finish.len() > 64 {
            return Err("fixture finish reason exceeds bound");
        }
        if result
            .finish
            .as_deref()
            .is_some_and(|prior| prior != finish)
        {
            return Err("fixture conflicting finish reason");
        }
        result.finish = Some(finish.to_owned());
    }
    Ok(())
}

fn append_field(field: &mut String, chunk: &str, cap: usize) -> Result<(), &'static str> {
    let next = field
        .len()
        .checked_add(chunk.len())
        .ok_or("fixture field overflow")?;
    if next > cap {
        return Err("fixture field exceeds bound");
    }
    field.push_str(chunk);
    Ok(())
}

struct FixtureWorkspace {
    path: PathBuf,
}

impl FixtureWorkspace {
    fn create() -> io::Result<Self> {
        for _ in 0..8 {
            let path = std::env::temp_dir()
                .join(format!("mivi-fixture-workspace-{}", uuid::Uuid::new_v4()));
            match fs::create_dir(&path) {
                Ok(()) => {
                    let workspace = Self { path };
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::PermissionsExt;
                        fs::set_permissions(&workspace.path, fs::Permissions::from_mode(0o700))?;
                    }
                    let example = workspace.path.join("example.rs");
                    let mut options = OpenOptions::new();
                    options.write(true).create_new(true);
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::OpenOptionsExt;
                        options.mode(0o600);
                    }
                    let mut file = options.open(example)?;
                    file.write_all(SOURCE.as_bytes())?;
                    file.flush()?;
                    return Ok(workspace);
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "could not exclusively create fixture workspace",
        ))
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for FixtureWorkspace {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

struct RequestResult {
    record: Option<super::FixtureRecord>,
    stream: Option<ObservedStream>,
    http_success: bool,
    capture_complete: bool,
}

struct FixtureRunSummary {
    artifact_directory: PathBuf,
    capture_complete: bool,
    continuation_exercised: bool,
    answer_quality: &'static str,
    cleanup_complete: bool,
    elapsed: Duration,
}

struct ContinuationDecision {
    capture_complete: bool,
    continuation_exercised: bool,
    answer_quality: &'static str,
}

fn continuation_decision(capture_complete: bool, continuation_ready: bool) -> ContinuationDecision {
    ContinuationDecision {
        capture_complete,
        continuation_exercised: capture_complete && continuation_ready,
        answer_quality: if capture_complete && continuation_ready {
            "not_assessed"
        } else {
            "continuation_unexercised"
        },
    }
}

fn fixture_payload() -> Value {
    json!({
        "model": "mivi",
        "stream": true,
        "temperature": 0,
        "seed": 7,
        "max_tokens": 48,
        "messages": [
            {"role":"system","content":"Use read_file to inspect the requested file before answering. Do not guess its contents."},
            {"role":"user","content":"Read example.rs using read_file. Then tell me the Rust function name and what it returns."}
        ],
        "tools": [{"type":"function","function":{
            "name":"read_file",
            "description":"Read a UTF-8 file and return its contents.",
            "parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}
        }}],
        "tool_choice":"required"
    })
}

fn has_clipped_metadata(value: &Value) -> bool {
    match value {
        Value::Object(object) => {
            object.get("clipped").and_then(Value::as_bool) == Some(true)
                || object.get("stop_tokens_clipped").and_then(Value::as_bool) == Some(true)
                || object.values().any(has_clipped_metadata)
        }
        Value::Array(values) => values.iter().any(has_clipped_metadata),
        _ => false,
    }
}

fn text_is_complete(text: &CapturedText) -> bool {
    !text.truncated && !text.counter_overflow
}

fn record_is_complete(record: &super::FixtureRecord) -> bool {
    let model_is_complete = record.model.as_ref().is_some_and(|model| {
        !model.raw_decoded.truncated
            && !model.raw_decoded.counter_overflow
            && !model.delivered.truncated
            && !model.delivered.counter_overflow
            && !model.generated_ids.truncated
            && !model.generated_ids.counter_overflow
            && !model.progress_counter_overflow
            && model.tokenization.is_some()
            && model.prefill.is_some()
            && model.prefill_outcome
                == Some(mivi_model::fixture_diagnostics::StageOutcome::Complete)
            && model.decode.is_some()
            && model.first_raw.is_some()
            && model.first_delivered.is_some()
            && model.outcome == Some(mivi_model::fixture_diagnostics::ModelOutcome::Complete)
    });
    model_is_complete
        && record.engine_terminal == super::EngineTerminal::Returned
        && !record.capture_incomplete
        && record.router_parse_error.is_none()
        && record.saw_done
        && text_is_complete(&record.rendered_prompt)
        && text_is_complete(&record.forced_prefix)
        && text_is_complete(&record.conditioned_prompt)
        && text_is_complete(&record.router_stream)
        && !has_clipped_metadata(&record.descriptor)
        && !has_clipped_metadata(&record.effective_settings)
}

async fn run_fixture_request(
    app: &axum::Router,
    session: &super::FixtureSession,
    metrics: &crate::state::ServerMetrics,
    fixture_id: &str,
    payload: Value,
) -> RequestResult {
    let sequence = match session.arm(fixture_id) {
        Ok(sequence) => sequence,
        Err(_) => {
            return RequestResult {
                record: None,
                stream: None,
                http_success: false,
                capture_complete: false,
            }
        }
    };
    let metrics_before = metrics.snapshot();
    let dispatch_started = Instant::now();
    let transcript_cap = session.capture_limits().text_bytes;
    let request_deadline = session.stream_deadline();
    let mut transcript = CapturedText::new(transcript_cap);
    let request_body = match serde_json::to_vec(&payload) {
        Ok(bytes) => Body::from(bytes),
        Err(_) => {
            session.mark_incomplete();
            return RequestResult {
                record: session.partial(sequence),
                stream: None,
                http_success: false,
                capture_complete: false,
            };
        }
    };
    let request = match Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(request_body)
    {
        Ok(request) => request,
        Err(_) => {
            session.mark_incomplete();
            return RequestResult {
                record: session.partial(sequence),
                stream: None,
                http_success: false,
                capture_complete: false,
            };
        }
    };

    let mut http_success = false;
    let observed_result =
        match tokio::time::timeout(request_deadline, app.clone().oneshot(request)).await {
            Ok(Ok(response)) => {
                http_success = response.status().is_success();
                match tokio::time::timeout(
                    request_deadline,
                    drain_fixture_sse(
                        response.into_body(),
                        transcript_cap,
                        dispatch_started,
                        &mut transcript,
                    ),
                )
                .await
                {
                    Ok(result) => result,
                    Err(_) => Err("fixture stream deadline"),
                }
            }
            Ok(Err(never)) => match never {},
            Err(_) => Err("fixture router dispatch deadline"),
        };

    let metrics_after = metrics.snapshot();
    let engine_result = session.wait_finished(sequence).await;
    let engine_observed = engine_result.is_ok();
    let mut record = match engine_result {
        Ok(record) => Some(record),
        Err(_) => session.partial(sequence),
    };
    let mut stream = None;
    let mut capture_complete = false;
    if let Some(record) = record.as_mut() {
        record.metrics_before = metrics_before;
        record.metrics_after = metrics_after;
        record.router_stream = transcript;
        match observed_result {
            Ok(observed) => {
                record.first_visible_delta = observed.first_visible_delta;
                record.stream_elapsed = Some(observed.elapsed);
                if record.capture_router_timing().is_err() {
                    record.capture_incomplete = true;
                    session.mark_incomplete();
                }
                record.router_finish = observed.finish.clone();
                record.saw_done = observed.saw_done;
                if observed.saw_error {
                    record.router_parse_error = Some("fixture SSE reported an error event");
                }
                stream = Some(observed);
            }
            Err(error) => {
                record.router_parse_error = Some(error);
                record.capture_incomplete = true;
            }
        }
        if !engine_observed {
            record.capture_incomplete = true;
            record.engine_terminal = super::EngineTerminal::Unobserved;
            if record.router_parse_error.is_none() {
                record.router_parse_error = Some("fixture engine snapshot unavailable");
            }
        }
        capture_complete = http_success
            && record_is_complete(record)
            && record.metrics_after.stream_completions_total
                > record.metrics_before.stream_completions_total;
    }
    RequestResult {
        record,
        stream,
        http_success,
        capture_complete,
    }
}

fn valid_fixture_tool_call(
    stream: &ObservedStream,
    workspace: &FixtureWorkspace,
) -> Result<(String, String), &'static str> {
    if stream.saw_error || !stream.saw_done || stream.finish.as_deref() != Some("tool_calls") {
        return Err("fixture tool request did not finish successfully");
    }
    let [call] = stream.calls.as_slice() else {
        return Err("fixture expected one tool call");
    };
    if call.id.is_empty() || call.name != "read_file" {
        return Err("fixture tool call identity was invalid");
    }
    let arguments: Value =
        serde_json::from_str(&call.arguments).map_err(|_| "fixture tool arguments were invalid")?;
    if arguments.get("path").and_then(Value::as_str) != Some("example.rs") {
        return Err("fixture tool path was invalid");
    }
    let actual = fs::read(workspace.path().join("example.rs"))
        .map_err(|_| "fixture source file could not be read")?;
    if actual != SOURCE.as_bytes() {
        return Err("fixture source bytes changed");
    }
    Ok((call.id.clone(), call.arguments.clone()))
}

fn fixture_followup(
    payload: &Value,
    call_id: &str,
    arguments: &str,
) -> Result<Value, &'static str> {
    let mut followup = payload.clone();
    followup["tool_choice"] = json!("none");
    let messages = followup["messages"]
        .as_array_mut()
        .ok_or("fixture message array was invalid")?;
    messages.push(json!({
        "role": "assistant",
        "content": null,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": "read_file", "arguments": arguments}
        }]
    }));
    messages.push(json!({
        "role": "tool",
        "tool_call_id": call_id,
        "content": SOURCE
    }));
    Ok(followup)
}

fn final_answer_is_qualified(stream: &ObservedStream) -> bool {
    !stream.saw_error
        && stream.saw_done
        && stream.finish.as_deref() == Some("stop")
        && stream.content.contains("add")
        && stream.content.contains("a + b")
}

async fn run_fixture_capture(profile_model: bool) -> Result<FixtureRunSummary, &'static str> {
    let start = Instant::now();
    let capture_limits = CaptureLimits {
        text_bytes: MODEL_TEXT_CAP,
        token_ids: MODEL_TOKEN_CAP,
    }
    .validate()?;
    let limits = super::FixtureLimits {
        model: capture_limits,
        profile_model,
        requests: 2,
        records: 2,
        stream_deadline: STREAM_DEADLINE,
        generation_watchdog: GENERATION_WATCHDOG,
    };
    let session = super::FixtureSession::new(limits)?;
    let workspace = FixtureWorkspace::create().map_err(|_| "fixture workspace setup failed")?;
    let artifact = super::ArtifactDirectory::create()
        .map_err(|_| "fixture artifact directory setup failed")?;

    let model_path = std::env::var_os("MIVI_TEST_MODEL")
        .ok_or("configuration required: set MIVI_TEST_MODEL to a supported model file")?;
    let model_path = PathBuf::from(model_path);
    if !model_path.is_absolute() || !model_path.is_file() {
        return Err("configuration required: MIVI_TEST_MODEL must be an absolute model file path");
    }
    let mut config = crate::config::ServerConfig::default();
    config.prefill_strategy = mivi_model::PrefillStrategy::Chunked { tile_tokens: 64 };
    config.request_timeout_secs = 120;
    config.first_token_timeout_secs = 90;
    config.max_concurrent_requests = 1;
    let model = mivi_model::Model::load_with_ctx(&model_path, Some(4096))
        .map_err(|_| "fixture model load failed for configured MIVI_TEST_MODEL")?;
    let mut engine =
        crate::engine_actor::EngineActor::try_spawn_fixture(model, &config, session.clone())
            .map_err(|_| "fixture engine actor startup failed")?;
    let Some(engine_handle) = engine.handle().cloned() else {
        drop(engine.take_handle());
        if engine.wait_and_join(GENERATION_WATCHDOG).await.is_err() {
            return Err("fixture actor cleanup was not observed within its watchdog");
        }
        return Err("fixture engine shut down");
    };
    let state = crate::state::AppState::with_config(
        "fixture-model",
        mivi_tools::ToolBroker::new(),
        engine_handle,
        None,
        config,
    )
    .with_workspace(workspace.path().to_path_buf());
    let metrics = Arc::clone(&state.metrics);
    let app = crate::create_router(Arc::new(state));
    let initial_payload = fixture_payload();
    let mut first = run_fixture_request(
        &app,
        &session,
        &metrics,
        "fixture_initial",
        initial_payload.clone(),
    )
    .await;
    let followup_payload = first
        .stream
        .as_ref()
        .and_then(|stream| valid_fixture_tool_call(stream, &workspace).ok())
        .and_then(|(id, args)| fixture_followup(&initial_payload, &id, &args).ok());
    let decision = continuation_decision(first.capture_complete, followup_payload.is_some());
    let continuation_exercised = decision.continuation_exercised;
    let mut answer_quality = decision.answer_quality;
    let mut capture_complete = decision.capture_complete;
    let mut persistence_complete = true;
    if !continuation_exercised {
        if let Some(record) = first.record.as_mut() {
            record.answer_quality = super::QualityOutcome::ContinuationUnexercised;
        }
    }
    if let Some(record) = first.record.as_ref() {
        persistence_complete &= artifact.write(record.sequence, record).is_ok();
    }

    if let Some(followup_payload) = followup_payload.filter(|_| continuation_exercised) {
        let second = run_fixture_request(
            &app,
            &session,
            &metrics,
            "fixture_followup",
            followup_payload,
        )
        .await;
        capture_complete &= second.capture_complete;
        let assessed_stream = second.stream.as_ref().filter(|stream| {
            !stream.saw_error && stream.saw_done && stream.finish.as_deref() == Some("stop")
        });
        let qualified = assessed_stream.is_some_and(final_answer_is_qualified);
        let quality_outcome = match assessed_stream {
            Some(_) if qualified => super::QualityOutcome::Passed,
            Some(_) => super::QualityOutcome::Failed,
            None => super::QualityOutcome::NotAssessed,
        };
        answer_quality = match quality_outcome {
            super::QualityOutcome::Passed => "passed",
            super::QualityOutcome::Failed => "failed",
            super::QualityOutcome::NotAssessed => "not_assessed",
            super::QualityOutcome::ContinuationUnexercised => "continuation_unexercised",
        };
        if let Some(mut record) = second.record {
            record.answer_quality = quality_outcome;
            persistence_complete &= artifact.write(record.sequence, &record).is_ok();
        }
    }

    let artifact_directory = artifact.path().to_path_buf();
    drop(app);
    drop(metrics);
    drop(engine.take_handle());
    let cleanup_complete = engine
        .wait_and_join(engine.session.generation_watchdog())
        .await
        .is_ok();
    if !cleanup_complete {
        return Err("fixture actor cleanup was not observed within its watchdog");
    }
    if !persistence_complete {
        return Err("fixture artifact persistence failed");
    }

    // Retain the original response status in the summary without inspecting captured content.
    if !first.http_success {
        capture_complete = false;
    }
    Ok(FixtureRunSummary {
        artifact_directory,
        capture_complete,
        continuation_exercised,
        answer_quality,
        cleanup_complete,
        elapsed: start.elapsed(),
    })
}

#[derive(Debug, PartialEq, Eq)]
struct FixtureProfileOutput {
    raw_decoded: String,
    delivered: String,
    generated_ids: Vec<u32>,
    router_content: String,
}

struct FixtureProfileControlSummary {
    output: FixtureProfileOutput,
    profile_present: bool,
    terminal: super::EngineTerminal,
    worker_return_us: Option<u64>,
    request_done: bool,
    cleanup_complete: bool,
    artifact_directory: PathBuf,
}

async fn run_fixture_profile_control(
    profile_model: bool,
) -> Result<FixtureProfileControlSummary, &'static str> {
    const TEXT_LIMIT: usize = 4 * 1024;
    const TOKEN_LIMIT: usize = 32;
    const OUTPUT_LIMIT: usize = 16;
    const CONTEXT_LIMIT: usize = 512;

    let mut limits = super::FixtureLimits::default();
    limits.model = CaptureLimits {
        text_bytes: TEXT_LIMIT,
        token_ids: TOKEN_LIMIT,
    }
    .validate()?;
    limits.profile_model = profile_model;
    limits.requests = 1;
    limits.records = 1;
    let session = super::FixtureSession::new(limits)?;
    let workspace = FixtureWorkspace::create().map_err(|_| "fixture workspace setup failed")?;
    let artifact = super::ArtifactDirectory::create()
        .map_err(|_| "fixture artifact directory setup failed")?;
    let model_path = std::env::var_os("MIVI_TEST_MODEL")
        .ok_or("configuration required: set MIVI_TEST_MODEL to a supported model file")?;
    let model_path = PathBuf::from(model_path);
    if !model_path.is_absolute() || !model_path.is_file() {
        return Err("configuration required: MIVI_TEST_MODEL must be an absolute model file path");
    }
    let mut config = crate::config::ServerConfig::default();
    config.request_timeout_secs = 120;
    config.first_token_timeout_secs = 90;
    config.max_concurrent_requests = 1;
    let model = mivi_model::Model::load_with_ctx(&model_path, Some(CONTEXT_LIMIT))
        .map_err(|_| "fixture model load failed for configured MIVI_TEST_MODEL")?;
    let mut engine =
        crate::engine_actor::EngineActor::try_spawn_fixture(model, &config, session.clone())
            .map_err(|_| "fixture engine actor startup failed")?;
    let Some(engine_handle) = engine.handle().cloned() else {
        drop(engine.take_handle());
        if engine.wait_and_join(GENERATION_WATCHDOG).await.is_err() {
            return Err("fixture actor cleanup was not observed within its watchdog");
        }
        return Err("fixture engine shut down");
    };
    let state = crate::state::AppState::with_config(
        "fixture-model",
        mivi_tools::ToolBroker::new(),
        engine_handle,
        None,
        config,
    )
    .with_workspace(workspace.path().to_path_buf());
    let metrics = Arc::clone(&state.metrics);
    let app = crate::create_router(Arc::new(state));
    let payload = json!({
        "model": "mivi",
        "stream": true,
        "temperature": 0,
        "seed": 7,
        "max_tokens": OUTPUT_LIMIT,
        "messages": [{"role":"user","content":"Return the word hello."}]
    });
    let request_result =
        run_fixture_request(&app, &session, &metrics, "fixture_profile", payload).await;
    let result = (|| {
        let record = request_result
            .record
            .as_ref()
            .ok_or("fixture profile record was unavailable")?;
        let model = record
            .model
            .as_ref()
            .ok_or("fixture profile model capture was unavailable")?;
        let stream = request_result
            .stream
            .as_ref()
            .ok_or("fixture profile router stream was unavailable")?;
        if !request_result.http_success {
            return Err("fixture profile request failed");
        }
        if !request_result.capture_complete || !record_is_complete(record) {
            return Err("fixture profile request capture was incomplete or truncated");
        }
        if model.reused_tokens != Some(0)
            || model.prefill_outcome
                != Some(mivi_model::fixture_diagnostics::StageOutcome::Complete)
            || model.outcome != Some(mivi_model::fixture_diagnostics::ModelOutcome::Complete)
        {
            return Err("fixture profile request did not complete a cold model prefill");
        }
        if profile_model {
            let snapshot = model
                .prefill_profile
                .as_ref()
                .ok_or("fixture profile snapshot was unavailable")?;
            if snapshot.tokens == 0 || snapshot.total_stage_time().is_zero() {
                return Err("fixture profile snapshot contained no measured prefill work");
            }
        } else if model.prefill_profile.is_some() {
            return Err("unprofiled fixture unexpectedly retained a model profile");
        }
        let output = FixtureProfileOutput {
            raw_decoded: model.raw_decoded.text.clone(),
            delivered: model.delivered.text.clone(),
            generated_ids: model.generated_ids.ids.clone(),
            router_content: stream.content.clone(),
        };
        let summary = FixtureProfileControlSummary {
            output,
            profile_present: record.model_prefill_profile_us.is_some(),
            terminal: record.engine_terminal,
            worker_return_us: record.actor_dequeue_to_worker_return_us,
            request_done: stream.saw_done,
            cleanup_complete: false,
            artifact_directory: artifact.path().to_path_buf(),
        };
        artifact
            .write(record.sequence, record)
            .map_err(|_| "fixture profile artifact persistence failed")?;
        Ok(summary)
    })();

    drop(app);
    drop(metrics);
    drop(engine.take_handle());
    let cleanup_complete = engine
        .wait_and_join(engine.session.generation_watchdog())
        .await
        .is_ok();
    if !cleanup_complete {
        return Err("fixture actor cleanup was not observed within its watchdog");
    }
    let mut summary = result?;
    summary.cleanup_complete = true;
    Ok(summary)
}

#[cfg(test)]
#[tokio::test]
#[ignore = "requires explicit MIVI_TEST_MODEL; exercises the actual router and actor"]
async fn fixture_generation_capture() -> Result<(), Box<dyn std::error::Error>> {
    if cfg!(debug_assertions) {
        return Err(io::Error::other("run fixture diagnostics in release profile").into());
    }
    let summary = run_fixture_capture(false).await.map_err(io::Error::other)?;
    println!(
        "fixture_generation_capture: profile=release capture_status={} continuation={} answer_quality={} cleanup={} context=4096 tile=64 request_deadline_s=120 first_output_deadline_s=90 stream_cap_bytes={} token_cap={} threads=2 concurrency=1 elapsed_ms={} artifact_dir={}",
        if summary.capture_complete { "complete" } else { "incomplete" },
        summary.continuation_exercised,
        summary.answer_quality,
        summary.cleanup_complete,
        MODEL_TEXT_CAP,
        MODEL_TOKEN_CAP,
        summary.elapsed.as_millis(),
        summary.artifact_directory.display(),
    );
    assert!(
        summary.capture_complete,
        "fixture capture infrastructure incomplete"
    );
    assert!(
        summary.continuation_exercised,
        "fixture tool continuation unexercised"
    );
    Ok(())
}

#[cfg(test)]
#[tokio::test]
#[ignore = "requires explicit MIVI_TEST_MODEL; compares profiled and unprofiled router runs"]
async fn fixture_profile_control_parity() -> Result<(), Box<dyn std::error::Error>> {
    if cfg!(debug_assertions) {
        return Err(io::Error::other("run fixture diagnostics in release profile").into());
    }
    let baseline = run_fixture_profile_control(false)
        .await
        .map_err(io::Error::other)?;
    let profiled = run_fixture_profile_control(true)
        .await
        .map_err(io::Error::other)?;

    assert!(
        baseline.output == profiled.output,
        "profiling changed output"
    );
    assert!(
        !baseline.profile_present,
        "baseline unexpectedly has model profile"
    );
    assert!(profiled.profile_present, "enabled model profile is missing");
    for run in [&baseline, &profiled] {
        assert!(run.request_done, "router stream did not reach DONE");
        assert_eq!(run.terminal, super::EngineTerminal::Returned);
        assert!(
            run.worker_return_us.is_some(),
            "request worker return was not observed"
        );
        assert!(
            run.cleanup_complete,
            "owned actor teardown was not observed"
        );
    }
    println!(
        "fixture_profile_control_parity: status=passed context=512 max_tokens=16 capture_cap_bytes=4096 token_cap=32 sequential=true baseline_artifact_dir={} profiled_artifact_dir={}",
        baseline.artifact_directory.display(),
        profiled.artifact_directory.display(),
    );
    Ok(())
}

#[cfg(test)]
#[test]
#[ignore = "requires explicit MIVI_TEST_MODEL; compares observer off/on"]
fn fixture_observer_parity() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::var_os("MIVI_TEST_MODEL")
        .ok_or_else(|| io::Error::other("configuration required: set MIVI_TEST_MODEL"))?;
    let path = PathBuf::from(path);
    if !path.is_absolute() || !path.is_file() {
        return Err(io::Error::other(
            "configuration required: MIVI_TEST_MODEL must be an absolute model file path",
        )
        .into());
    }
    let prompt = "Return the word hello.";
    let mut baseline = mivi_model::Model::load_with_ctx(&path, Some(512))
        .map_err(|_| io::Error::other("fixture parity baseline model load failed"))?;
    baseline.sampler.config.temperature = 0.0;
    baseline.sampler.set_seed(7);
    let mut baseline_chunks = CapturedText::new(4096);
    let baseline_output = baseline
        .generate_streaming(prompt, 16, |_, text| {
            baseline_chunks.push(text);
            true
        })
        .map_err(|_| io::Error::other("fixture parity baseline generation failed"))?;
    let baseline_rng = baseline.sampler.rng_state();
    assert!(!baseline_chunks.truncated && !baseline_chunks.counter_overflow);
    assert!(baseline.take_fixture_capture().is_none());
    drop(baseline);

    let mut observed = mivi_model::Model::load_with_ctx(&path, Some(512))
        .map_err(|_| io::Error::other("fixture parity observed model load failed"))?;
    observed.sampler.config.temperature = 0.0;
    observed.sampler.set_seed(7);
    observed
        .start_fixture_capture(CaptureLimits {
            text_bytes: 4096,
            token_ids: 32,
        })
        .map_err(|_| io::Error::other("fixture parity capture setup failed"))?;
    let mut observed_chunks = CapturedText::new(4096);
    let observed_output = observed
        .generate_streaming(prompt, 16, |_, text| {
            observed_chunks.push(text);
            true
        })
        .map_err(|_| io::Error::other("fixture parity observed generation failed"))?;
    let snapshot = observed
        .take_fixture_capture()
        .ok_or_else(|| io::Error::other("fixture parity capture missing"))?;
    assert!(!observed_chunks.truncated && !observed_chunks.counter_overflow);
    assert!(!snapshot.delivered.truncated && !snapshot.delivered.counter_overflow);
    assert!(
        baseline_output == observed_output,
        "observer changed generation"
    );
    assert!(
        baseline_chunks.text == observed_chunks.text,
        "observer changed delivery"
    );
    assert!(
        snapshot.delivered.text == observed_chunks.text,
        "capture differs from delivery"
    );
    assert_eq!(baseline_rng, observed.sampler.rng_state());
    println!("fixture_observer_parity: status=passed context=512 max_tokens=16 capture_cap_bytes=4096 token_cap=32 loads=2 sequential=true");
    Ok(())
}

#[cfg(test)]
mod tests {
    #[cfg(debug_assertions)]
    #[test]
    fn debug_profile_returns_configuration_error_before_model_loading() {
        assert!(super::fixture_generation_capture().is_err());
    }

    use super::*;
    use axum::body::Bytes;
    use std::convert::Infallible;

    fn body_from_chunks(chunks: Vec<Vec<u8>>) -> Body {
        Body::from_stream(futures::stream::iter(
            chunks
                .into_iter()
                .map(|chunk| Ok::<Bytes, Infallible>(Bytes::from(chunk))),
        ))
    }

    fn chunk(value: &str) -> Vec<u8> {
        value.as_bytes().to_vec()
    }

    fn assert_stream_error(observed: Result<ObservedStream, &'static str>, expected: &'static str) {
        assert_eq!(observed.err(), Some(expected));
    }

    #[tokio::test]
    async fn observes_done_after_draining_the_body() {
        let body = Body::from("data: [DONE]\n\n");
        let mut transcript = CapturedText::new(128);

        let observed = drain_fixture_sse(body, 128, Instant::now(), &mut transcript).await;

        assert!(observed.unwrap().saw_done);
    }

    #[tokio::test]
    async fn fixture_profile_records_visible_content_and_preserves_output() {
        let body = body_from_chunks(vec![
            chunk(": keep-alive\n\n"),
            chunk(
                "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n",
            ),
            chunk(
                "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_fixture\",\"function\":{\"name\":\"\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n",
            ),
            chunk(
                "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"example.rs\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
            ),
            chunk(
                "data: {\"choices\":[{\"delta\":{\"content\":\"unchanged\"},\"finish_reason\":null}]}\n\n",
            ),
            chunk("data: [DONE]\n\n"),
        ]);
        let started = Instant::now();
        let mut transcript = CapturedText::new(4096);

        let observed = drain_fixture_sse(body, 4096, started, &mut transcript)
            .await
            .unwrap();

        assert_eq!(observed.content, "unchanged");
        assert_eq!(observed.calls.len(), 1);
        assert_eq!(observed.calls[0].name, "read_file");
        assert_eq!(observed.calls[0].arguments, r#"{"path":"example.rs"}"#);
        assert!(observed.first_visible_delta.is_some());
        assert!(observed.first_visible_delta.unwrap() <= observed.elapsed);
        assert!(observed.saw_done);
        assert!(transcript.text.contains("unchanged"));
    }

    #[test]
    fn fixture_profile_ignores_heartbeats_and_empty_envelopes_until_output() {
        let started = Instant::now();
        let mut result = ObservedStream {
            first_visible_delta: None,
            elapsed: Duration::ZERO,
            finish: None,
            saw_done: false,
            saw_error: false,
            calls: Vec::new(),
            content: String::new(),
        };
        let mut call_fragments = 0;

        observe_event(
            r#"{"choices":[{"delta":{"role":"assistant","content":""}}]}"#,
            started,
            64,
            &mut call_fragments,
            &mut result,
        )
        .unwrap();
        assert!(result.first_visible_delta.is_none());
        observe_event(
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"","arguments":""}}]}}]}"#,
            started,
            64,
            &mut call_fragments,
            &mut result,
        )
        .unwrap();
        assert!(result.first_visible_delta.is_none());
        observe_event(
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"read_file","arguments":""}}]}}]}"#,
            started,
            64,
            &mut call_fragments,
            &mut result,
        )
        .unwrap();

        assert!(result.first_visible_delta.is_some());
        assert_eq!(result.calls[0].name, "read_file");
    }

    #[test]
    fn unusable_tool_output_does_not_invalidate_a_complete_capture() {
        let decision = continuation_decision(true, false);

        assert!(
            decision.capture_complete,
            "complete capture was downgraded by tool quality"
        );
        assert!(!decision.continuation_exercised);
        assert!(decision.answer_quality == "continuation_unexercised");
    }

    #[test]
    fn clipped_stop_tokens_make_fixture_record_incomplete() {
        let session =
            super::super::FixtureSession::new(super::super::FixtureLimits::default()).unwrap();
        let sequence = session.arm("clipped_stop_tokens").unwrap();
        let mut record = session
            .begin(
                sequence,
                "prompt",
                "",
                serde_json::json!({"stop_tokens": ["<stop>"]}),
                serde_json::json!({"model": "fixture"}),
            )
            .unwrap();
        record.model = Some(mivi_model::fixture_diagnostics::ModelCapture {
            raw_decoded: CapturedText::new(8),
            delivered: CapturedText::new(8),
            generated_ids: mivi_model::fixture_diagnostics::CapturedIds::new(4),
            terminal_token_id: None,
            stopping_reason: None,
            prefill_profile: None,
            tokenization: Some(Duration::ZERO),
            prefill: Some(Duration::ZERO),
            prefill_outcome: Some(mivi_model::fixture_diagnostics::StageOutcome::Complete),
            decode: Some(Duration::ZERO),
            first_raw: Some(Duration::ZERO),
            first_delivered: Some(Duration::ZERO),
            prompt_tokens: Some(1),
            reused_tokens: Some(0),
            processed_tokens: Some(1),
            outcome: Some(mivi_model::fixture_diagnostics::ModelOutcome::Complete),
            progress_counter_overflow: false,
        });
        record.engine_terminal = super::super::EngineTerminal::Returned;
        record.saw_done = true;
        assert!(record_is_complete(&record));

        record.effective_settings["stop_tokens_clipped"] = serde_json::json!(true);

        assert!(!record_is_complete(&record));
    }

    #[tokio::test]
    async fn reconstructs_split_utf8_events_and_fragmented_tool_calls() {
        let initial = chunk(
            ": keep-alive\r\n\r\ndata: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\r\n\r\n",
        );
        let content = chunk(
            "data: {\"choices\":[{\"delta\":{\"content\":\"café\"},\"finish_reason\":null}]}\n\n",
        );
        let split_at = content
            .windows(2)
            .position(|window| window == "é".as_bytes())
            .unwrap()
            + 1;
        let first_call = chunk(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"call_fixture\",\"type\":\"function\",\"function\":{\"name\":\"read_\",\"arguments\":\"{\\\"pa\"}}]},\"finish_reason\":null}]}\n\n",
        );
        let second_call = chunk(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"name\":\"file\",\"arguments\":\"th\\\":\\\"example.rs\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
        );
        let body = body_from_chunks(vec![
            initial,
            content[..split_at].to_vec(),
            content[split_at..].to_vec(),
            first_call,
            second_call,
            chunk("data: [DONE]\n\n: trailing keep-alive\n\n"),
        ]);
        let mut transcript = CapturedText::new(4096);

        let observed = drain_fixture_sse(body, 4096, Instant::now(), &mut transcript)
            .await
            .unwrap();

        assert!(
            observed.content == "café",
            "fixture content reconstruction mismatch"
        );
        assert_eq!(observed.calls.len(), 1);
        assert_eq!(observed.calls[0].index, 1);
        assert!(
            observed.calls[0].id == "call_fixture",
            "fixture call ID mismatch"
        );
        assert!(
            observed.calls[0].name == "read_file",
            "fixture call name mismatch"
        );
        assert!(
            observed.calls[0].arguments == r#"{"path":"example.rs"}"#,
            "fixture call arguments mismatch"
        );
        assert_eq!(observed.finish.as_deref(), Some("tool_calls"));
        assert!(observed.first_visible_delta.is_some());
        assert!(observed.saw_done);
        assert!(!observed.saw_error);
        assert!(!transcript.truncated);
        assert!(!transcript.counter_overflow);
        assert!(transcript.text.contains("café"));
    }

    #[tokio::test]
    async fn retains_partial_transcript_for_malformed_event() {
        let body = Body::from("data: {malformed}\n\n");
        let mut transcript = CapturedText::new(128);

        let error = drain_fixture_sse(body, 128, Instant::now(), &mut transcript)
            .await
            .err()
            .expect("malformed fixture event is rejected");

        assert_eq!(error, "fixture event JSON");
        assert!(transcript.text.contains("malformed"));
    }

    #[tokio::test]
    async fn reports_error_event_without_calling_it_a_parse_failure() {
        let body = Body::from("data: {\"error\":{\"message\":\"private\"}}\n\ndata: [DONE]\n\n");
        let mut transcript = CapturedText::new(128);

        let observed = drain_fixture_sse(body, 128, Instant::now(), &mut transcript)
            .await
            .unwrap();

        assert!(observed.saw_error);
        assert!(observed.saw_done);
        assert!(observed.calls.is_empty());

        let mut finish_transcript = CapturedText::new(256);
        let error_finish = drain_fixture_sse(
            Body::from(
                "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"error\"}]}\n\ndata: [DONE]\n\n",
            ),
            256,
            Instant::now(),
            &mut finish_transcript,
        )
        .await
        .unwrap();
        assert!(error_finish.finish.as_deref() == Some("error"));
        assert!(!error_finish.saw_error);
    }

    #[tokio::test]
    async fn rejects_truncation_overflow_and_incomplete_streams() {
        let mut truncated_transcript = CapturedText::new(8);
        let truncated = drain_fixture_sse(
            Body::from("data: [DONE]\n\n"),
            8,
            Instant::now(),
            &mut truncated_transcript,
        )
        .await;
        assert_stream_error(truncated, "fixture frame exceeds bound");
        assert!(truncated_transcript.truncated);

        let mut incomplete_transcript = CapturedText::new(128);
        let incomplete = drain_fixture_sse(
            Body::from("data: {\"choices\":[]}"),
            128,
            Instant::now(),
            &mut incomplete_transcript,
        )
        .await;
        assert_stream_error(incomplete, "fixture stream ended before DONE");
        assert!(!incomplete_transcript.truncated);

        let mut retained_transcript = CapturedText::new(20);
        let retained_overflow = drain_fixture_sse(
            body_from_chunks(vec![chunk(": 123456789\n\n"), chunk(": 123456789\n\n")]),
            20,
            Instant::now(),
            &mut retained_transcript,
        )
        .await;
        assert_stream_error(retained_overflow, "fixture SSE retention exceeds bound");
        assert!(retained_transcript.truncated);
    }

    #[tokio::test]
    async fn rejects_data_after_done_but_allows_trailing_comments() {
        let mut comment_transcript = CapturedText::new(128);
        let comments = drain_fixture_sse(
            Body::from("data: [DONE]\n\n: final heartbeat\n\n"),
            128,
            Instant::now(),
            &mut comment_transcript,
        )
        .await;
        assert!(comments.unwrap().saw_done);

        let mut data_transcript = CapturedText::new(256);
        let data = drain_fixture_sse(
            Body::from("data: [DONE]\n\ndata: [DONE]\n\n"),
            256,
            Instant::now(),
            &mut data_transcript,
        )
        .await;
        assert_stream_error(data, "fixture data after DONE");
    }

    #[tokio::test]
    async fn enforces_event_call_and_field_limits() {
        let mut event_transcript = CapturedText::new(20_000);
        let many_comments = ": ping\n\n".repeat(513);
        let event_overflow = drain_fixture_sse(
            Body::from(many_comments),
            20_000,
            Instant::now(),
            &mut event_transcript,
        )
        .await;
        assert_stream_error(event_overflow, "fixture event count exceeds bound");

        let mut index_transcript = CapturedText::new(1024);
        let index_overflow = drain_fixture_sse(
            Body::from("data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":16}]}}]}\n\n"),
            1024,
            Instant::now(),
            &mut index_transcript,
        )
        .await;
        assert_stream_error(index_overflow, "fixture tool index exceeds bound");

        let many_calls = serde_json::json!({
            "choices": [{
                "delta": {
                    "tool_calls": (0..17)
                        .map(|index| serde_json::json!({"index": index}))
                        .collect::<Vec<_>>()
                }
            }]
        });
        let mut count_transcript = CapturedText::new(4096);
        let call_count_overflow = drain_fixture_sse(
            Body::from(format!("data: {many_calls}\n\n")),
            4096,
            Instant::now(),
            &mut count_transcript,
        )
        .await;
        assert_stream_error(call_count_overflow, "fixture tool-call count exceeds bound");

        let long_name = serde_json::json!({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"name": "x".repeat(257)}
                    }]
                }
            }]
        });
        let mut field_transcript = CapturedText::new(4096);
        let field_overflow = drain_fixture_sse(
            Body::from(format!("data: {long_name}\n\n")),
            4096,
            Instant::now(),
            &mut field_transcript,
        )
        .await;
        assert_stream_error(field_overflow, "fixture field exceeds bound");

        let fragment = chunk(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{}}]}}]}\n\n",
        );
        let mut fragments = vec![fragment; 257];
        fragments.push(chunk("data: [DONE]\n\n"));
        let mut fragment_transcript = CapturedText::new(32_768);
        let fragment_overflow = drain_fixture_sse(
            body_from_chunks(fragments),
            32_768,
            Instant::now(),
            &mut fragment_transcript,
        )
        .await;
        assert_stream_error(
            fragment_overflow,
            "fixture call fragment count exceeds bound",
        );
    }

    #[tokio::test]
    async fn rejects_conflicting_ids_and_finish_reasons() {
        let conflicting_ids = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"first\"}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"second\"}]}}]}\n\n",
        );
        let mut transcript = CapturedText::new(1024);
        let ids = drain_fixture_sse(
            Body::from(conflicting_ids),
            1024,
            Instant::now(),
            &mut transcript,
        )
        .await;
        assert_stream_error(ids, "fixture conflicting call ID");

        let conflicting_finish = concat!(
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n",
        );
        let mut transcript = CapturedText::new(1024);
        let finish = drain_fixture_sse(
            Body::from(conflicting_finish),
            1024,
            Instant::now(),
            &mut transcript,
        )
        .await;
        assert_stream_error(finish, "fixture conflicting finish reason");
    }
}
