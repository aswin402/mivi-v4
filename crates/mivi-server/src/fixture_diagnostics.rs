//! Private, feature-gated server-side fixture capture.
#![allow(dead_code)] // The fixture runner consuming these APIs is Task 4.

mod artifacts;

#[allow(unused_imports)]
pub(crate) use artifacts::ArtifactDirectory;
use mivi_model::fixture_diagnostics::{CaptureLimits, CapturedText, ModelCapture};
use serde::Serialize;
use serde_json::Value;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, TryLockError};
use std::time::Duration;
use tokio::sync::Notify;

const MAX_REQUESTS_OR_RECORDS: usize = 128;
const MAX_TOTAL_BUDGET: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy)]
pub(crate) struct FixtureLimits {
    pub model: CaptureLimits,
    pub requests: usize,
    pub records: usize,
    pub stream_deadline: Duration,
    pub generation_watchdog: Duration,
}

impl Default for FixtureLimits {
    fn default() -> Self {
        Self {
            model: CaptureLimits {
                text_bytes: 65_536,
                token_ids: 256,
            },
            requests: 4,
            records: 4,
            stream_deadline: Duration::from_secs(120),
            generation_watchdog: Duration::from_secs(120),
        }
    }
}

impl FixtureLimits {
    fn validate(self) -> Result<Self, &'static str> {
        self.model.validate()?;
        if self.requests == 0
            || self.records == 0
            || self.requests > MAX_REQUESTS_OR_RECORDS
            || self.records > MAX_REQUESTS_OR_RECORDS
            || self.stream_deadline.is_zero()
            || self.generation_watchdog.is_zero()
            || self.stream_deadline > Duration::from_secs(600)
            || self.generation_watchdog > Duration::from_secs(600)
        {
            return Err("invalid fixture session limits");
        }
        let per_record = self
            .model
            .text_bytes
            .checked_mul(16)
            .and_then(|text| {
                self.model
                    .token_ids
                    .checked_mul(4)
                    .and_then(|ids| text.checked_add(ids))
            })
            .and_then(|n| n.checked_add(4096))
            .ok_or("fixture budget overflow")?;
        let total = per_record
            .checked_mul(self.records)
            .ok_or("fixture budget overflow")?;
        if total > MAX_TOTAL_BUDGET {
            return Err("fixture budget exceeds limit");
        }
        Ok(self)
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct FixtureRecord {
    pub sequence: usize,
    pub fixture_id: String,
    pub rendered_prompt: CapturedText,
    pub forced_prefix: CapturedText,
    pub conditioned_prompt: CapturedText,
    pub model: Option<ModelCapture>,
    pub descriptor: Value,
    pub effective_settings: Value,
    pub router_stream: CapturedText,
    pub first_visible_delta: Option<Duration>,
    pub stream_elapsed: Option<Duration>,
    pub metrics_before: crate::state::MetricsSnapshot,
    pub metrics_after: crate::state::MetricsSnapshot,
    pub router_finish: Option<String>,
    pub saw_done: bool,
    pub router_parse_error: Option<&'static str>,
    pub answer_quality: QualityOutcome,
    pub engine_terminal: EngineTerminal,
    pub capture_incomplete: bool,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
pub(crate) enum EngineTerminal {
    NotStarted,
    Unobserved,
    Returned,
    Cancelled,
    ReceiverClosed,
    ModelError,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
pub(crate) enum QualityOutcome {
    NotAssessed,
    Passed,
    Failed,
    ContinuationUnexercised,
}

struct SessionState {
    next_sequence: usize,
    requests_started: usize,
    armed: Option<(usize, String)>,
    partial: Option<FixtureRecord>,
    finished: VecDeque<FixtureRecord>,
}

struct SessionInner {
    limits: FixtureLimits,
    state: Mutex<SessionState>,
    incomplete: AtomicBool,
    notify: Notify,
}

#[derive(Clone)]
pub(crate) struct FixtureSession(Arc<SessionInner>);

impl FixtureSession {
    pub(crate) fn new(limits: FixtureLimits) -> Result<Self, &'static str> {
        let limits = limits.validate()?;
        Ok(Self(Arc::new(SessionInner {
            limits,
            state: Mutex::new(SessionState {
                next_sequence: 0,
                requests_started: 0,
                armed: None,
                partial: None,
                finished: VecDeque::with_capacity(limits.records),
            }),
            incomplete: AtomicBool::new(false),
            notify: Notify::new(),
        })))
    }

    pub(crate) fn arm(&self, fixture_id: &str) -> Result<usize, &'static str> {
        if fixture_id.is_empty()
            || fixture_id.len() > 64
            || !fixture_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
        {
            return Err("invalid fixture id");
        }
        let mut state = self.lock_boundary()?;
        if state.armed.is_some() || state.partial.is_some() {
            return Err("fixture request already active");
        }
        if state.requests_started >= self.0.limits.requests {
            return Err("fixture request limit reached");
        }
        if state.finished.len() >= self.0.limits.records {
            drop(state);
            self.mark_incomplete();
            return Err("fixture record capacity reached");
        }
        let sequence = state
            .next_sequence
            .checked_add(1)
            .ok_or("fixture sequence overflow")?;
        state.next_sequence = sequence;
        state.requests_started += 1;
        state.armed = Some((sequence, fixture_id.to_owned()));
        Ok(sequence)
    }

    pub(crate) fn is_active(&self) -> bool {
        match self.0.state.try_lock() {
            Ok(state) => {
                state.armed.is_some() || state.partial.is_some() || !state.finished.is_empty()
            }
            Err(_) => {
                self.mark_incomplete();
                true
            }
        }
    }

    pub(crate) fn armed_sequence(&self) -> Option<usize> {
        match self.0.state.try_lock() {
            Ok(state) => state.armed.as_ref().map(|(sequence, _)| *sequence),
            Err(_) => {
                self.mark_incomplete();
                None
            }
        }
    }

    pub(crate) fn capture_limits(&self) -> CaptureLimits {
        self.0.limits.model
    }

    pub(crate) fn text_limit(&self) -> usize {
        self.0.limits.model.text_bytes
    }

    #[allow(dead_code)]
    pub(crate) fn stream_deadline(&self) -> Duration {
        self.0.limits.stream_deadline
    }

    #[allow(dead_code)]
    pub(crate) fn generation_watchdog(&self) -> Duration {
        self.0.limits.generation_watchdog
    }

    pub(crate) fn begin(
        &self,
        sequence: usize,
        prompt: &str,
        prefix: &str,
        settings: Value,
        descriptor: Value,
    ) -> Option<FixtureRecord> {
        let mut state = match self.0.state.try_lock() {
            Ok(state) => state,
            Err(_) => {
                self.mark_incomplete();
                return None;
            }
        };
        let Some((armed_sequence, fixture_id)) = state.armed.take() else {
            drop(state);
            self.mark_incomplete();
            return None;
        };
        if armed_sequence != sequence
            || !bounded_object(&settings, self.0.limits.model.text_bytes)
            || !bounded_object(&descriptor, self.0.limits.model.text_bytes)
        {
            state.armed = Some((armed_sequence, fixture_id));
            drop(state);
            self.mark_incomplete();
            return None;
        }
        let text_cap = self.0.limits.model.text_bytes;
        let mut rendered_prompt = CapturedText::new(text_cap);
        rendered_prompt.push(prompt);
        let mut forced_prefix = CapturedText::new(text_cap);
        forced_prefix.push(prefix);
        let mut conditioned_prompt = CapturedText::new(text_cap);
        conditioned_prompt.push(prompt);
        conditioned_prompt.push(prefix);
        let record = FixtureRecord {
            sequence,
            fixture_id,
            rendered_prompt,
            forced_prefix,
            conditioned_prompt,
            model: None,
            descriptor,
            effective_settings: settings,
            router_stream: CapturedText::new(text_cap),
            first_visible_delta: None,
            stream_elapsed: None,
            metrics_before: empty_metrics(),
            metrics_after: empty_metrics(),
            router_finish: None,
            saw_done: false,
            router_parse_error: None,
            answer_quality: QualityOutcome::NotAssessed,
            engine_terminal: EngineTerminal::Unobserved,
            capture_incomplete: self.0.incomplete.load(Ordering::Acquire),
        };
        state.partial = Some(record.clone());
        Some(record)
    }

    pub(crate) fn finish_engine(&self, mut record: FixtureRecord) {
        let mut state = match self.0.state.try_lock() {
            Ok(state) => state,
            Err(_) => {
                self.mark_incomplete();
                return;
            }
        };
        let partial_matches = match state.partial.as_ref() {
            Some(partial) => partial.sequence == record.sequence,
            None => false,
        };
        if !partial_matches || state.finished.len() >= self.0.limits.records {
            drop(state);
            self.mark_incomplete();
            return;
        }
        record.capture_incomplete |= self.0.incomplete.load(Ordering::Acquire);
        state.partial = None;
        state.armed = None;
        state.finished.push_back(record);
        drop(state);
        self.0.notify.notify_one();
    }

    pub(crate) fn take_finished(&self, sequence: usize) -> Option<FixtureRecord> {
        let mut state = match self.0.state.try_lock() {
            Ok(state) => state,
            Err(_) => {
                self.mark_incomplete();
                return None;
            }
        };
        let index = state
            .finished
            .iter()
            .position(|record| record.sequence == sequence)?;
        let mut record = state.finished.remove(index)?;
        record.capture_incomplete |= self.0.incomplete.load(Ordering::Acquire);
        Some(record)
    }

    pub(crate) fn partial(&self, sequence: usize) -> Option<FixtureRecord> {
        let state = match self.0.state.try_lock() {
            Ok(state) => state,
            Err(_) => {
                self.mark_incomplete();
                return None;
            }
        };
        let mut record = state
            .partial
            .as_ref()
            .filter(|record| record.sequence == sequence)
            .cloned()?;
        record.capture_incomplete |= self.0.incomplete.load(Ordering::Acquire);
        Some(record)
    }

    pub(crate) async fn wait_finished(
        &self,
        sequence: usize,
    ) -> Result<FixtureRecord, &'static str> {
        let wait = async {
            loop {
                let notified = self.0.notify.notified();
                tokio::pin!(notified);
                notified.as_mut().enable();
                if let Some(record) = self.take_finished(sequence) {
                    return Ok(record);
                }
                if self.0.incomplete.load(Ordering::Acquire) {
                    return Err("fixture capture became incomplete");
                }
                notified.await;
            }
        };
        match tokio::time::timeout(self.0.limits.generation_watchdog, wait).await {
            Ok(result) => result,
            Err(_) => {
                self.mark_unobserved(sequence);
                Err("fixture generation watchdog expired")
            }
        }
    }

    pub(crate) fn mark_unobserved(&self, sequence: usize) {
        if let Ok(mut state) = self.0.state.try_lock() {
            if let Some(record) = state
                .partial
                .as_mut()
                .filter(|record| record.sequence == sequence)
            {
                record.engine_terminal = EngineTerminal::Unobserved;
                record.capture_incomplete = true;
            }
        }
        self.mark_incomplete();
    }

    pub(crate) fn finish_unobserved(&self, sequence: usize) {
        let mut state = match self.0.state.try_lock() {
            Ok(state) => state,
            Err(_) => {
                self.mark_incomplete();
                return;
            }
        };
        let partial_matches = state
            .partial
            .as_ref()
            .is_some_and(|record| record.sequence == sequence);
        if partial_matches {
            let Some(mut record) = state.partial.take() else {
                drop(state);
                self.mark_incomplete();
                return;
            };
            record.engine_terminal = EngineTerminal::Unobserved;
            record.capture_incomplete = true;
            state.armed = None;
            if state.finished.len() < self.0.limits.records {
                state.finished.push_back(record);
            }
        }
        drop(state);
        self.mark_incomplete();
    }

    pub(crate) fn incomplete(&self) -> bool {
        self.0.incomplete.load(Ordering::Acquire)
    }

    fn lock_boundary(&self) -> Result<std::sync::MutexGuard<'_, SessionState>, &'static str> {
        match self.0.state.try_lock() {
            Ok(state) => Ok(state),
            Err(TryLockError::WouldBlock) | Err(TryLockError::Poisoned(_)) => {
                self.mark_incomplete();
                Err("fixture session unavailable")
            }
        }
    }

    pub(crate) fn mark_incomplete(&self) {
        self.0.incomplete.store(true, Ordering::Release);
        self.0.notify.notify_one();
    }
}

#[allow(dead_code)]
pub(crate) struct FixtureEngine {
    pub(crate) handle: Option<crate::engine_actor::EngineHandle>,
    pub(crate) join_handle: Option<std::thread::JoinHandle<()>>,
    pub(crate) completion: tokio::sync::oneshot::Receiver<()>,
    pub(crate) session: FixtureSession,
}

#[allow(dead_code)]
impl FixtureEngine {
    pub(crate) fn handle(&self) -> Option<&crate::engine_actor::EngineHandle> {
        self.handle.as_ref()
    }

    pub(crate) fn take_handle(&mut self) -> Option<crate::engine_actor::EngineHandle> {
        self.handle.take()
    }

    pub(crate) async fn wait_and_join(&mut self, watchdog: Duration) -> Result<(), &'static str> {
        tokio::time::timeout(watchdog, &mut self.completion)
            .await
            .map_err(|_| "fixture actor completion watchdog expired")?
            .map_err(|_| "fixture actor completion notification was lost")?;
        let join = self
            .join_handle
            .as_ref()
            .ok_or("fixture actor join handle unavailable")?;
        tokio::time::timeout(watchdog, async {
            while !join.is_finished() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .map_err(|_| "fixture actor return was not observed")?;
        self.join_handle
            .take()
            .expect("checked fixture actor join handle")
            .join()
            .map_err(|_| "fixture actor thread panicked")
    }
}

fn bounded_object(value: &Value, byte_limit: usize) -> bool {
    value.is_object()
        && value.as_object().is_some_and(|object| object.len() <= 32)
        && serde_json::to_vec(value).is_ok_and(|serialized| serialized.len() <= byte_limit)
}

fn empty_metrics() -> crate::state::MetricsSnapshot {
    crate::state::MetricsSnapshot {
        inference_requests_total: 0,
        inference_requests_rejected_total: 0,
        inference_errors_total: 0,
        inference_slot_wait_microseconds_total: 0,
        generation_latency_microseconds_total: 0,
        generation_count: 0,
        time_to_first_token_microseconds_total: 0,
        first_token_count: 0,
        prompt_tokens_total: 0,
        completion_tokens_total: 0,
        tool_timeouts_total: 0,
        stream_completions_total: 0,
        stream_body_errors_total: 0,
        stream_client_disconnects_total: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_session() -> FixtureSession {
        FixtureSession::new(FixtureLimits::default()).unwrap()
    }

    fn begin(session: &FixtureSession, id: &str, prompt: &str, prefix: &str) -> FixtureRecord {
        let sequence = session.arm(id).unwrap();
        session
            .begin(
                sequence,
                prompt,
                prefix,
                serde_json::json!({"sampler": {"temperature": 0.4}}),
                serde_json::json!({"model_config_name": {"value": "test", "clipped": false}}),
            )
            .unwrap()
    }

    #[test]
    fn retains_byte_exact_prompt_prefix_and_router_stream() {
        let session = fixture_session();
        let mut record = begin(&session, "fixture_one", "prompt\n", "prefix:");
        record.router_stream.push("body");
        let mut raw = CapturedText::new(8);
        raw.push("body");
        let mut delivered = CapturedText::new(8);
        delivered.push("body");
        record.model = Some(ModelCapture {
            raw_decoded: raw,
            delivered,
            generated_ids: mivi_model::fixture_diagnostics::CapturedIds::new(4),
            tokenization: None,
            prefill: None,
            prefill_outcome: None,
            decode: None,
            first_raw: None,
            first_delivered: None,
            prompt_tokens: None,
            reused_tokens: None,
            processed_tokens: None,
            outcome: None,
            progress_counter_overflow: false,
        });
        record.router_parse_error = Some("synthetic parser result");
        let sequence = record.sequence;
        session.finish_engine(record);

        let retained = session.take_finished(sequence).unwrap();
        assert_eq!(retained.rendered_prompt.text.as_bytes(), b"prompt\n");
        assert_eq!(retained.forced_prefix.text.as_bytes(), b"prefix:");
        assert_eq!(
            retained.conditioned_prompt.text.as_bytes(),
            b"prompt\nprefix:"
        );
        assert_eq!(retained.router_stream.text.as_bytes(), b"body");
        let model = retained.model.unwrap();
        assert_eq!(model.raw_decoded.text.as_bytes(), b"body");
        assert_eq!(model.delivered.text.as_bytes(), b"body");
        assert_eq!(retained.router_parse_error, Some("synthetic parser result"));
    }

    #[test]
    fn sequences_are_monotonic_and_never_overwrite_finished_records() {
        let session = fixture_session();
        let first = begin(&session, "first", "one", "");
        let first_sequence = first.sequence;
        session.finish_engine(first);
        let second = begin(&session, "second", "two", "");
        let second_sequence = second.sequence;
        assert!(second_sequence > first_sequence);
        session.finish_engine(second);
        assert_eq!(
            session.take_finished(first_sequence).unwrap().fixture_id,
            "first"
        );
        assert_eq!(
            session.take_finished(second_sequence).unwrap().fixture_id,
            "second"
        );
    }

    #[test]
    fn rejects_invalid_limits_and_fixture_ids() {
        let mut limits = FixtureLimits::default();
        limits.records = 129;
        assert_eq!(
            FixtureSession::new(limits).err(),
            Some("invalid fixture session limits")
        );
        assert_eq!(
            fixture_session().arm("../unsafe").err(),
            Some("invalid fixture id")
        );

        let mut limits = FixtureLimits::default();
        limits.requests = 0;
        assert!(FixtureSession::new(limits).is_err());
        let mut limits = FixtureLimits::default();
        limits.generation_watchdog = Duration::from_secs(601);
        assert!(FixtureSession::new(limits).is_err());
        let mut limits = FixtureLimits::default();
        limits.model.text_bytes = mivi_model::fixture_diagnostics::MAX_TEXT_BYTES;
        assert_eq!(
            FixtureSession::new(limits).err(),
            Some("fixture budget exceeds limit")
        );
    }

    #[test]
    fn request_and_record_caps_are_enforced() {
        let session = fixture_session();
        for sequence in 1..=4 {
            let record = begin(&session, "valid", "x", "");
            assert_eq!(record.sequence, sequence);
            session.finish_engine(record);
        }
        assert_eq!(
            session.arm("fifth").err(),
            Some("fixture request limit reached")
        );
        for sequence in 1..=4 {
            session.take_finished(sequence).unwrap();
        }
    }

    #[test]
    fn full_retention_capacity_marks_capture_incomplete() {
        let mut limits = FixtureLimits::default();
        limits.records = 1;
        let session = FixtureSession::new(limits).unwrap();
        let record = begin(&session, "first", "one", "");
        session.finish_engine(record);
        assert_eq!(
            session.arm("second").err(),
            Some("fixture record capacity reached")
        );
        assert!(session.incomplete());
    }

    #[test]
    fn contention_marks_capture_incomplete_without_waiting() {
        let session = fixture_session();
        let _guard = session.0.state.lock().unwrap();
        assert_eq!(
            session.arm("busy").err(),
            Some("fixture session unavailable")
        );
        assert!(session.is_active());
        assert!(session.incomplete());
    }

    #[test]
    fn partial_record_is_available_until_engine_finalization() {
        let session = fixture_session();
        let record = begin(&session, "partial", "prompt", "prefix");
        let sequence = record.sequence;
        let partial = session.partial(sequence).unwrap();
        assert_eq!(partial.engine_terminal, EngineTerminal::Unobserved);
        session.finish_engine(record);
        assert!(session.partial(sequence).is_none());
        assert!(session.take_finished(sequence).is_some());
    }

    #[test]
    fn capture_remains_active_until_the_router_consumes_the_finished_record() {
        let session = fixture_session();
        let record = begin(&session, "route", "prompt", "");
        let sequence = record.sequence;
        assert!(session.is_active());
        session.finish_engine(record);
        assert!(session.is_active());
        session.take_finished(sequence).unwrap();
        assert!(!session.is_active());
    }

    #[test]
    fn not_started_and_model_error_states_are_retained_without_fabricated_model_output() {
        let session = fixture_session();
        let mut not_started = begin(&session, "not_started", "prompt", "prefix");
        let sequence = not_started.sequence;
        not_started.engine_terminal = EngineTerminal::NotStarted;
        session.finish_engine(not_started);
        let retained = session.take_finished(sequence).unwrap();
        assert_eq!(retained.engine_terminal, EngineTerminal::NotStarted);
        assert!(retained.model.is_none());

        let mut failed = begin(&session, "model_error", "prompt", "");
        let sequence = failed.sequence;
        failed.engine_terminal = EngineTerminal::ModelError;
        failed.capture_incomplete = true;
        let mut model_capture = ModelCapture {
            raw_decoded: CapturedText::new(8),
            delivered: CapturedText::new(8),
            generated_ids: mivi_model::fixture_diagnostics::CapturedIds::new(4),
            tokenization: None,
            prefill: None,
            prefill_outcome: None,
            decode: None,
            first_raw: None,
            first_delivered: None,
            prompt_tokens: None,
            reused_tokens: None,
            processed_tokens: None,
            outcome: Some(mivi_model::fixture_diagnostics::ModelOutcome::ModelError),
            progress_counter_overflow: false,
        };
        model_capture.raw_decoded.push("partial");
        failed.model = Some(model_capture);
        session.finish_engine(failed);
        let retained = session.take_finished(sequence).unwrap();
        assert_eq!(retained.engine_terminal, EngineTerminal::ModelError);
        assert_eq!(
            retained.model.unwrap().outcome,
            Some(mivi_model::fixture_diagnostics::ModelOutcome::ModelError)
        );
    }

    #[tokio::test]
    async fn wait_finished_observes_completion_and_watchdog_marks_partial_unobserved() {
        let session = fixture_session();
        let record = begin(&session, "wait", "prompt", "");
        let sequence = record.sequence;
        session.finish_engine(record);
        assert_eq!(
            session.wait_finished(sequence).await.unwrap().sequence,
            sequence
        );

        let mut limits = FixtureLimits::default();
        limits.generation_watchdog = Duration::from_millis(1);
        let stalled = FixtureSession::new(limits).unwrap();
        let record = begin(&stalled, "stalled", "prompt", "");
        let sequence = record.sequence;
        assert_eq!(
            stalled.wait_finished(sequence).await.err(),
            Some("fixture generation watchdog expired")
        );
        assert!(stalled.partial(sequence).unwrap().capture_incomplete);
        assert_eq!(
            stalled.partial(sequence).unwrap().engine_terminal,
            EngineTerminal::Unobserved
        );
    }
}
