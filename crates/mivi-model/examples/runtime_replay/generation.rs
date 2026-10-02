use mivi_model::fixture_diagnostics::replay::{ProfileMicros, ReplayInput};
use mivi_model::fixture_diagnostics::{CaptureLimits, CapturedText, ModelCapture, ModelOutcome};
use mivi_model::{Model, PrefillStrategy};
use serde_json::{json, Value};
use std::cell::Cell;
use std::error::Error;
use std::path::Path;
use std::time::{Duration, Instant};

const GENERATION_DEADLINE: Duration = Duration::from_secs(180);
const GENERATED_ID_LIMIT: usize = 64;
const TEXT_CAPTURE_LIMIT: usize = 65_536;
const ENTRY_BOUNDARY: &str = "process_cli_entry_including_argument_parsing_input_read_output_preflight_model_load_and_inference";

pub fn run(
    model_path: &Path,
    input: ReplayInput,
    started: Instant,
) -> Result<Value, Box<dyn Error>> {
    input.validate().map_err(invalid_input)?;
    let load_started = Instant::now();
    let mut model = match Model::load_with_options(
        model_path,
        Some(input.context),
        Some(mivi_kv::KvPrecision::F32),
    ) {
        Ok(model) => model,
        Err(error) => {
            return Ok(json!({
                "schema": 1,
                "runtime_version": env!("CARGO_PKG_VERSION"),
                "status": "model_error",
                "error": error.to_string(),
                "model": { "path": model_path.to_string_lossy() },
                "timing_us": { "load": micros(load_started.elapsed())? },
                "process_exit": "not_observed"
            }));
        }
    };
    let load_us = micros(load_started.elapsed())?;
    validate_model_input(&model, &input)?;

    let (add_bos, bos_id) = bos_settings(&model)?;
    let eos_id = eos_settings(&model)?;
    let (normalized_prompt, bos_inserted) =
        normalize_prompt_ids(&input.prompt_ids, add_bos, bos_id);
    validate_normalized_context(&model, &input, normalized_prompt.len())?;
    model.reset_context();
    model.prefix_cache.clear();
    model.sampler.config.temperature = 0.0;
    model.sampler.config.seed = Some(7);
    model.sampler.config.repetition_penalty = 1.0;
    model.sampler.config.presence_penalty = 0.0;
    model.sampler.config.frequency_penalty = 0.0;
    model.sampler.set_seed(7);
    model.set_prefill_strategy(PrefillStrategy::Chunked {
        tile_tokens: input.tile,
    })?;
    if input.profile {
        model.enable_forward_profile();
        model.reset_forward_profile();
    } else {
        model.disable_forward_profile();
    }

    let (text, content_ids, mut capture, phase_prefill_us, mut run_error) = if input.split_prefill {
        let first = observed_call(&mut model, &normalized_prompt, 0, 0, started)?;
        let prefill_us = first.capture.prefill.map(micros).transpose()?;
        if first.error.is_some() || first.capture.outcome == Some(ModelOutcome::Cancelled) {
            (
                first.text,
                first.ids,
                first.capture,
                Some(json!({
                    "initial_prefill": prefill_us,
                    "continuation_prefill": null
                })),
                first.error,
            )
        } else {
            let continuation_pos = model.current_pos();
            let second =
                observed_call(&mut model, &[], continuation_pos, input.max_tokens, started)?;
            let phase_two_prefill_us = second.capture.prefill.map(micros).transpose()?;
            let error = second.error;
            let combined = combine_split_capture(first.capture, second.capture);
            (
                format!("{}{}", first.text, second.text),
                [first.ids, second.ids].concat(),
                combined,
                Some(json!({
                    "initial_prefill": prefill_us,
                    "continuation_prefill": phase_two_prefill_us
                })),
                error,
            )
        }
    } else {
        let call = observed_call(&mut model, &normalized_prompt, 0, input.max_tokens, started)?;
        (call.text, call.ids, call.capture, None, call.error)
    };

    let final_position = model.current_pos();
    let mut status = status_for_capture(
        &capture,
        content_ids.len(),
        input.max_tokens,
        final_position,
        input.context,
    );
    if run_error.is_some() {
        status = "model_error";
    }
    if capture.stopping_reason.is_none() {
        capture.stopping_reason = Some(if content_ids.len() >= input.max_tokens {
            "output_limit".to_owned()
        } else if final_position >= input.context {
            "context_limit".to_owned()
        } else {
            "generation_returned_early".to_owned()
        });
    }

    let teacher_started = Instant::now();
    let has_teacher_probes = !input.teacher_forced_ids.is_empty();
    let (teacher_forced, teacher_cancelled, teacher_error) = if input.teacher_forced_ids.is_empty()
    {
        (Vec::new(), false, None)
    } else {
        match run_teacher_forcing(
            &mut model,
            &normalized_prompt,
            &input.teacher_forced_ids,
            &input.logit_ids,
            started,
        ) {
            Ok((records, cancelled)) => (records, cancelled, None),
            Err(error) => (Vec::new(), false, Some(error.to_string())),
        }
    };
    if teacher_cancelled {
        status = "cancelled";
    } else if teacher_error.is_some() {
        status = "model_error";
    }
    if run_error.is_none() {
        run_error = teacher_error;
    }
    let teacher_forced_us = micros(teacher_started.elapsed())?;
    let mut returned_capture = CapturedText::new(TEXT_CAPTURE_LIMIT);
    returned_capture.push(&text);
    let prefill_profile = capture
        .prefill_profile
        .map(ProfileMicros::from_snapshot)
        .transpose()
        .map_err(invalid_input)?;
    let architecture = model
        .gguf
        .metadata
        .get("general.architecture")
        .and_then(|value| value.as_str())
        .unwrap_or(&model.config.name);
    let model_size_bytes = std::fs::metadata(model_path)?.len();

    Ok(json!({
        "schema": 1,
        "runtime_version": env!("CARGO_PKG_VERSION"),
        "status": status,
        "error": run_error,
        "model": {
            "path": model_path.to_string_lossy(),
            "size_bytes": model_size_bytes,
            "architecture": architecture,
            "config": model.config,
            "hash_status": "not_computed_by_generation_runner"
        },
        "effective": {
            "context": input.context,
            "tile": input.tile,
            "kv_precision": "F32",
            "temperature": model.sampler.config.temperature,
            "seed": 7,
            "repetition_penalty": model.sampler.config.repetition_penalty,
            "presence_penalty": model.sampler.config.presence_penalty,
            "frequency_penalty": model.sampler.config.frequency_penalty,
            "stop_tokens": model.sampler.config.stop_tokens,
            "worker_threads": rayon::current_num_threads(),
            "terminal_policy": {
                "eos_id": eos_id,
                "suppress_first_step": true
            },
        },
        "input": {
            "prompt_ids": input.prompt_ids,
            "normalized_prompt_ids": normalized_prompt,
            "bos_policy": {
                "metadata_add_bos": add_bos,
                "metadata_bos_id": bos_id,
                "inserted": bos_inserted
            }
        },
        "generation": {
            "content_ids": content_ids,
            "captured_content_ids": capture.generated_ids,
            "prefill_progress": {
                "prompt_tokens": capture.prompt_tokens,
                "reused_tokens": capture.reused_tokens,
                "processed_tokens": capture.processed_tokens,
                "outcome": capture.prefill_outcome,
            },
            "terminal_ids": capture.terminal_token_id.into_iter().collect::<Vec<_>>(),
            "raw_capture": capture.raw_decoded,
            "delivered_capture": capture.delivered,
            "stopping_reason": capture.stopping_reason,
            "returned_text": returned_capture.text,
            "returned_text_capture": returned_capture,
            "final_position": final_position,
            "outcome": capture.outcome,
        },
        "timing_us": {
            "load": load_us,
            "prefill": capture.prefill.map(micros).transpose()?,
            "prefill_phases": phase_prefill_us,
            "decode": capture.decode.map(micros).transpose()?,
            "first_raw": capture.first_raw.map(micros).transpose()?,
            "first_delivered": capture.first_delivered.map(micros).transpose()?,
            "entry_boundary": ENTRY_BOUNDARY,
            "prefill_boundary": "model_prefill_only",
            "first_raw_boundary": "first_nonempty_raw_decoded_callback",
            "first_delivered_boundary": "first_nonempty_filtered_delivery_callback"
        },
        "profile": {
            "enabled": input.profile,
            "prefill": prefill_profile,
        },
        "teacher_forced": teacher_forced,
        "teacher_forced_us": if has_teacher_probes { Some(teacher_forced_us) } else { None::<u64> },
        "teacher_forced_timing_boundary": "token_major_forward_and_scoring_only",
        "process_exit": "not_observed"
    }))
}

struct ObservedCall {
    text: String,
    ids: Vec<u32>,
    capture: ModelCapture,
    error: Option<String>,
}

fn observed_call(
    model: &mut Model,
    prompt: &[u32],
    start_pos: usize,
    max_tokens: usize,
    started: Instant,
) -> Result<ObservedCall, Box<dyn Error>> {
    model.start_fixture_capture(CaptureLimits {
        text_bytes: TEXT_CAPTURE_LIMIT,
        token_ids: GENERATED_ID_LIMIT,
    })?;
    model.begin_fixture_capture_observation(started)?;
    let cancelled = Cell::new(false);
    let result = model.generate_tokens_incremental_with_cancel(
        prompt,
        start_pos,
        max_tokens,
        |_, _| true,
        || {
            let expired = started.elapsed() >= GENERATION_DEADLINE;
            if expired {
                cancelled.set(true);
            }
            expired
        },
    );
    let outcome = match &result {
        Err(_) => ModelOutcome::ModelError,
        Ok(_) if cancelled.get() => ModelOutcome::Cancelled,
        Ok(_) => ModelOutcome::Complete,
    };
    model.finish_fixture_capture_observation(outcome, Instant::now());
    let capture = model
        .take_fixture_capture()
        .ok_or("fixture capture disappeared before take")?;
    match result {
        Ok((text, ids)) => Ok(ObservedCall {
            text,
            ids,
            capture,
            error: None,
        }),
        Err(error) => Ok(ObservedCall {
            text: String::new(),
            ids: Vec::new(),
            capture,
            error: Some(error.to_string()),
        }),
    }
}

fn combine_split_capture(mut first: ModelCapture, second: ModelCapture) -> ModelCapture {
    first.raw_decoded = second.raw_decoded;
    first.delivered = second.delivered;
    first.generated_ids = second.generated_ids;
    first.terminal_token_id = second.terminal_token_id;
    first.stopping_reason = second.stopping_reason;
    first.decode = second.decode;
    first.first_raw = second.first_raw;
    first.first_delivered = second.first_delivered;
    first.outcome = second.outcome;
    first
}

fn status_for_capture(
    capture: &ModelCapture,
    content_count: usize,
    max_tokens: usize,
    final_position: usize,
    context: usize,
) -> &'static str {
    match capture.outcome {
        Some(ModelOutcome::Cancelled) => "cancelled",
        Some(ModelOutcome::ModelError) => "model_error",
        _ if capture.terminal_token_id.is_some()
            || capture.stopping_reason.as_deref() == Some("stop_sequence")
            || content_count >= max_tokens =>
        {
            "complete"
        }
        _ if final_position >= context => "partial",
        _ => "partial",
    }
}

fn run_teacher_forcing(
    model: &mut Model,
    prefix: &[u32],
    teacher_ids: &[u32],
    selected_ids: &[u32],
    started: Instant,
) -> Result<(Vec<Value>, bool), Box<dyn Error>> {
    model.reset_context();
    model.prefix_cache.clear();
    let mut position = 0usize;
    let mut logits = Vec::new();
    for &token_id in prefix {
        if deadline_expired(started) {
            return Ok((Vec::new(), true));
        }
        logits.clear();
        logits.extend_from_slice(model.forward(token_id, position)?);
        position += 1;
    }

    let mut records = Vec::with_capacity(teacher_ids.len());
    for &token_id in teacher_ids {
        if deadline_expired(started) {
            return Ok((records, true));
        }
        let mut top1 = None;
        let mut top2 = None;
        for (id, &logit) in logits.iter().enumerate() {
            if !logit.is_finite() {
                return Err("teacher-forced logits contain a non-finite value".into());
            }
            let pair = (id as u32, logit);
            if top1.is_none_or(|best: (u32, f32)| pair.1 > best.1) {
                top2 = top1;
                top1 = Some(pair);
            } else if top2.is_none_or(|best: (u32, f32)| pair.1 > best.1) {
                top2 = Some(pair);
            }
        }
        let (top1_id, top1_logit) = top1.ok_or("teacher-forced model produced no logits")?;
        let (top2_id, top2_logit) = top2.unwrap_or((top1_id, top1_logit));
        let selected = selected_ids
            .iter()
            .map(|&id| json!({ "id": id, "logit": logits[id as usize] }))
            .collect::<Vec<_>>();
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum = logits
            .iter()
            .map(|value| f64::from((*value - max_logit).exp()))
            .sum::<f64>();
        let log_probability = f64::from(logits[token_id as usize] - max_logit) - exp_sum.ln();
        if !log_probability.is_finite() || !(top1_logit - top2_logit).is_finite() {
            return Err("teacher-forced score is non-finite".into());
        }
        records.push(json!({
            "position": position,
            "token_id": token_id,
            "log_probability": log_probability,
            "selected_logits": selected,
            "top1": { "id": top1_id, "logit": top1_logit },
            "top2": { "id": top2_id, "logit": top2_logit },
            "top_margin": top1_logit - top2_logit
        }));
        if deadline_expired(started) {
            return Ok((records, true));
        }
        logits.clear();
        logits.extend_from_slice(model.forward(token_id, position)?);
        position += 1;
    }
    Ok((records, deadline_expired(started)))
}

fn validate_model_input(model: &Model, input: &ReplayInput) -> Result<(), Box<dyn Error>> {
    let vocab_size = model.config.vocab_size;
    validate_vocabulary_ids(
        &input.prompt_ids,
        &input.teacher_forced_ids,
        &input.logit_ids,
        vocab_size,
    )
    .map_err(invalid_input)?;
    let (add_bos, bos_id) = bos_settings(model)?;
    let (normalized, _) = normalize_prompt_ids(&input.prompt_ids, add_bos, bos_id);
    if normalized.iter().any(|id| (*id as usize) >= vocab_size) {
        return Err("normalized prompt contains a token outside the loaded vocabulary".into());
    }
    let _ = eos_settings(model)?;
    Ok(())
}

fn validate_vocabulary_ids(
    prompt_ids: &[u32],
    teacher_ids: &[u32],
    logit_ids: &[u32],
    vocab_size: usize,
) -> Result<(), String> {
    for id in prompt_ids.iter().chain(teacher_ids).chain(logit_ids) {
        if (*id as usize) >= vocab_size {
            return Err(format!("token ID {id} is outside the loaded vocabulary"));
        }
    }
    Ok(())
}

fn validate_normalized_context(
    model: &Model,
    input: &ReplayInput,
    normalized_len: usize,
) -> Result<(), Box<dyn Error>> {
    let needed = normalized_len
        .checked_add(input.max_tokens)
        .and_then(|n| n.checked_add(input.teacher_forced_ids.len()))
        .ok_or("normalized context length overflow")?;
    if needed > input.context || input.context > model.config.max_seq_len {
        return Err("normalized replay exceeds loaded model context".into());
    }
    Ok(())
}

fn bos_settings(model: &Model) -> Result<(bool, u32), Box<dyn Error>> {
    let add_bos = parse_add_bos_setting(model.gguf.metadata.get("tokenizer.ggml.add_bos_token"))
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    let bos_id = metadata_token_id(model, "tokenizer.ggml.bos_token_id", 1)?;
    Ok((add_bos, bos_id))
}

fn parse_add_bos_setting(value: Option<&mivi_model::GgufValue>) -> Result<bool, &'static str> {
    match value {
        None => Ok(false),
        Some(mivi_model::GgufValue::Bool(value)) => Ok(*value),
        Some(_) => Err("invalid tokenizer.ggml.add_bos_token metadata"),
    }
}

fn eos_settings(model: &Model) -> Result<u32, Box<dyn Error>> {
    metadata_token_id(
        model,
        "tokenizer.ggml.eos_token_id",
        mivi_tokenizer::EOS_TOKEN_ID,
    )
}

fn metadata_token_id(model: &Model, key: &str, fallback: u32) -> Result<u32, Box<dyn Error>> {
    let id = match model.gguf.metadata.get(key) {
        Some(value) => {
            let id = value
                .as_usize()
                .ok_or_else(|| format!("invalid token ID metadata: {key}"))?;
            u32::try_from(id).map_err(|_| format!("token ID metadata exceeds u32: {key}"))?
        }
        None => fallback,
    };
    if (id as usize) >= model.config.vocab_size {
        return Err(format!("token ID metadata is outside the loaded vocabulary: {key}").into());
    }
    Ok(id)
}

fn deadline_expired(started: Instant) -> bool {
    started.elapsed() >= GENERATION_DEADLINE
}

fn normalize_prompt_ids(prompt: &[u32], add_bos: bool, bos_id: u32) -> (Vec<u32>, bool) {
    let insert = add_bos && prompt.first() != Some(&bos_id);
    if insert {
        let mut normalized = Vec::with_capacity(prompt.len() + 1);
        normalized.push(bos_id);
        normalized.extend_from_slice(prompt);
        (normalized, true)
    } else {
        (prompt.to_vec(), false)
    }
}

fn micros(duration: Duration) -> Result<u64, Box<dyn Error>> {
    Ok(u64::try_from(duration.as_micros())?)
}

fn invalid_input(error: impl ToString) -> Box<dyn Error> {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, error.to_string()).into()
}

#[cfg(test)]
mod tests {
    use super::{
        normalize_prompt_ids, parse_add_bos_setting, run, validate_vocabulary_ids, Model,
        ReplayInput, ENTRY_BOUNDARY,
    };
    use std::time::Instant;

    #[test]
    fn prompt_normalization_matches_metadata_bos_policy() {
        assert_eq!(normalize_prompt_ids(&[7, 8], true, 7), (vec![7, 8], false));
        assert_eq!(normalize_prompt_ids(&[8], true, 7), (vec![7, 8], true));
        assert_eq!(normalize_prompt_ids(&[8], false, 7), (vec![8], false));
    }

    #[test]
    fn teacher_forcing_accepts_repeated_prefix_and_terminal_ids() {
        assert!(validate_vocabulary_ids(
            &[1, 7, 7],
            &[7, mivi_tokenizer::EOS_TOKEN_ID],
            &[1, mivi_tokenizer::EOS_TOKEN_ID],
            8,
        )
        .is_ok());
        assert!(validate_vocabulary_ids(&[], &[8], &[], 8).is_err());
    }

    #[test]
    fn present_non_boolean_add_bos_metadata_is_rejected() {
        assert_eq!(parse_add_bos_setting(None), Ok(false));
        assert_eq!(
            parse_add_bos_setting(Some(&mivi_model::GgufValue::Bool(false))),
            Ok(false)
        );
        assert_eq!(
            parse_add_bos_setting(Some(&mivi_model::GgufValue::Bool(true))),
            Ok(true)
        );
        assert!(parse_add_bos_setting(Some(&mivi_model::GgufValue::U32(1))).is_err());
        assert!(
            parse_add_bos_setting(Some(&mivi_model::GgufValue::String("true".into()))).is_err()
        );
    }

    #[test]
    fn first_event_boundary_covers_cli_and_file_preflight() {
        assert_eq!(
            ENTRY_BOUNDARY,
            "process_cli_entry_including_argument_parsing_input_read_output_preflight_model_load_and_inference"
        );
    }

    #[test]
    #[ignore = "requires an explicitly configured real model and bounded runtime"]
    fn replay_split_prefill_matches_single_call() -> Result<(), Box<dyn std::error::Error>> {
        let model_path = std::env::var_os("MIVI_TEST_MODEL")
            .map(std::path::PathBuf::from)
            .ok_or("MIVI_TEST_MODEL is not set")?;
        let model = Model::load(&model_path)?;
        let max_context = model.config.max_seq_len;
        drop(model);
        if max_context.min(4096) < 2636 + 8 {
            return Err(format!(
                "configured model context {max_context} cannot validate the required 2636-token parity case"
            )
            .into());
        }
        for prefix_len in [110usize, 2636] {
            let prompt_ids = vec![1; prefix_len];
            let base = ReplayInput {
                prompt_ids,
                context: max_context.min(4096),
                tile: 64,
                max_tokens: 8,
                profile: true,
                split_prefill: false,
                teacher_forced_ids: Vec::new(),
                logit_ids: Vec::new(),
            };
            let split = ReplayInput {
                prompt_ids: base.prompt_ids.clone(),
                context: base.context,
                tile: base.tile,
                max_tokens: base.max_tokens,
                profile: base.profile,
                split_prefill: true,
                teacher_forced_ids: Vec::new(),
                logit_ids: Vec::new(),
            };
            let single_value = run(&model_path, base, Instant::now())?;
            let split_value = run(&model_path, split, Instant::now())?;
            let expected_normalized_len =
                prefix_len + usize::from(single_value["input"]["bos_policy"]["inserted"] == true);
            let single = &single_value["generation"];
            let split_result = &split_value["generation"];
            let readiness_error = parity_readiness(&single_value, expected_normalized_len)
                .and_then(|()| parity_readiness(&split_value, expected_normalized_len));
            let mismatch = single["content_ids"] != split_result["content_ids"]
                || single["delivered_capture"]["text"] != split_result["delivered_capture"]["text"]
                || single["terminal_ids"] != split_result["terminal_ids"]
                || single["final_position"] != split_result["final_position"];
            if readiness_error.is_err() || mismatch {
                let artifact_dir = tempfile::Builder::new()
                    .prefix("mivi-replay-parity-")
                    .tempdir()?;
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    std::fs::set_permissions(
                        artifact_dir.path(),
                        std::fs::Permissions::from_mode(0o700),
                    )?;
                }
                let artifact_path = artifact_dir.path().join("failure.json");
                use std::io::Write;
                #[cfg(unix)]
                let mut artifact = {
                    use std::os::unix::fs::OpenOptionsExt;
                    std::fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .mode(0o600)
                        .open(&artifact_path)?
                };
                #[cfg(not(unix))]
                let mut artifact = std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&artifact_path)?;
                artifact.write_all(&serde_json::to_vec(&serde_json::json!({
                    "single": single_value,
                    "split": split_value
                }))?)?;
                artifact.sync_all()?;
                let artifact_dir = artifact_dir.keep();
                let detail = readiness_error
                    .err()
                    .unwrap_or_else(|| "generation outputs differ".to_owned());
                panic!(
                    "split parity check failed ({detail}); private artifact: {}",
                    artifact_dir.join("failure.json").display()
                );
            }
        }
        Ok(())
    }

    fn parity_readiness(
        record: &serde_json::Value,
        expected_prompt_tokens: usize,
    ) -> Result<(), String> {
        if record["status"] != "complete" {
            return Err("run status was not complete".to_owned());
        }
        let generation = &record["generation"];
        if generation["raw_capture"]["truncated"] == true
            || generation["delivered_capture"]["truncated"] == true
            || generation["returned_text_capture"]["truncated"] == true
            || generation["captured_content_ids"]["truncated"] == true
        {
            return Err("a bounded output capture was truncated".to_owned());
        }
        let normalized_len = record["input"]["normalized_prompt_ids"]
            .as_array()
            .map(Vec::len)
            .ok_or_else(|| "normalized prompt IDs were unavailable".to_owned())?;
        let progress = &generation["prefill_progress"];
        let prompt_tokens = progress["prompt_tokens"]
            .as_u64()
            .ok_or_else(|| "observed prompt token count was unavailable".to_owned())?
            as usize;
        let reused = progress["reused_tokens"]
            .as_u64()
            .ok_or_else(|| "reused prefix token count was unavailable".to_owned())?
            as usize;
        let processed = progress["processed_tokens"]
            .as_u64()
            .ok_or_else(|| "processed prefix token count was unavailable".to_owned())?
            as usize;
        if normalized_len != expected_prompt_tokens
            || prompt_tokens != expected_prompt_tokens
            || reused.checked_add(processed) != Some(expected_prompt_tokens)
        {
            return Err("prefill did not account for the normalized prompt".to_owned());
        }
        Ok(())
    }
}
