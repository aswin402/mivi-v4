use serde::Serialize;
use std::time::{Duration, Instant};

pub const MAX_TEXT_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_CAPTURE_IDS: usize = 65_536;

#[derive(Clone, Copy, Debug)]
pub struct CaptureLimits {
    pub text_bytes: usize,
    pub token_ids: usize,
}

impl CaptureLimits {
    pub fn validate(self) -> Result<Self, &'static str> {
        if self.text_bytes == 0
            || self.text_bytes > MAX_TEXT_BYTES
            || self.token_ids == 0
            || self.token_ids > MAX_CAPTURE_IDS
        {
            return Err("invalid fixture capture bounds");
        }
        self.text_bytes
            .checked_mul(4)
            .and_then(|n| {
                self.token_ids
                    .checked_mul(4)
                    .and_then(|ids| n.checked_add(ids))
            })
            .ok_or("fixture capture budget overflow")?;
        Ok(self)
    }
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct CapturedText {
    pub text: String,
    pub observed_bytes: usize,
    pub truncated: bool,
    pub counter_overflow: bool,
    #[serde(skip)]
    cap: usize,
}

impl CapturedText {
    pub fn new(cap: usize) -> Self {
        Self {
            text: String::with_capacity(cap),
            observed_bytes: 0,
            truncated: false,
            counter_overflow: false,
            cap,
        }
    }

    pub fn push(&mut self, chunk: &str) {
        match self.observed_bytes.checked_add(chunk.len()) {
            Some(n) => self.observed_bytes = n,
            None => {
                self.observed_bytes = usize::MAX;
                self.counter_overflow = true;
            }
        }
        if self.truncated {
            return;
        }
        let available = self.cap.saturating_sub(self.text.len());
        let mut keep = available.min(chunk.len());
        while !chunk.is_char_boundary(keep) {
            keep -= 1;
        }
        self.text.push_str(&chunk[..keep]);
        self.truncated = keep < chunk.len();
    }
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct CapturedIds {
    pub ids: Vec<u32>,
    pub observed_tokens: usize,
    pub truncated: bool,
    pub counter_overflow: bool,
    #[serde(skip)]
    cap: usize,
}

impl CapturedIds {
    pub fn new(cap: usize) -> Self {
        Self {
            ids: Vec::with_capacity(cap),
            observed_tokens: 0,
            truncated: false,
            counter_overflow: false,
            cap,
        }
    }

    pub fn push(&mut self, id: u32) {
        match self.observed_tokens.checked_add(1) {
            Some(n) => self.observed_tokens = n,
            None => {
                self.observed_tokens = usize::MAX;
                self.counter_overflow = true;
            }
        }
        if self.ids.len() < self.cap {
            self.ids.push(id);
        } else {
            self.truncated = true;
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
pub enum StageOutcome {
    Complete,
    Cancelled,
    ModelError,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
pub enum ModelOutcome {
    Complete,
    Cancelled,
    DeliveryStopped,
    ModelError,
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelCapture {
    pub raw_decoded: CapturedText,
    pub delivered: CapturedText,
    pub generated_ids: CapturedIds,
    pub tokenization: Option<Duration>,
    pub prefill: Option<Duration>,
    pub prefill_outcome: Option<StageOutcome>,
    pub decode: Option<Duration>,
    pub first_raw: Option<Duration>,
    pub first_delivered: Option<Duration>,
    pub prompt_tokens: Option<usize>,
    pub reused_tokens: Option<usize>,
    pub processed_tokens: Option<usize>,
    pub outcome: Option<ModelOutcome>,
    pub progress_counter_overflow: bool,
}

pub struct ModelRecorder {
    pub snapshot: ModelCapture,
    entry: Option<Instant>,
    tokenization_start: Option<Instant>,
    prefill_start: Option<Instant>,
    decode_start: Option<Instant>,
}

impl ModelRecorder {
    pub fn new(limits: CaptureLimits) -> Result<Self, &'static str> {
        let limits = limits.validate()?;
        Ok(Self {
            snapshot: ModelCapture {
                raw_decoded: CapturedText::new(limits.text_bytes),
                delivered: CapturedText::new(limits.text_bytes),
                generated_ids: CapturedIds::new(limits.token_ids),
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
            },
            entry: None,
            tokenization_start: None,
            prefill_start: None,
            decode_start: None,
        })
    }

    pub fn begin(&mut self, now: Instant) {
        self.entry = Some(now);
        self.tokenization_start = Some(now);
    }

    pub fn tokenization_done(&mut self, now: Instant) {
        self.snapshot.tokenization = self
            .tokenization_start
            .map(|start| now.saturating_duration_since(start));
    }

    pub fn prefill_begin(&mut self, now: Instant, tokens: usize) {
        self.prefill_start = Some(now);
        self.snapshot.prompt_tokens = Some(tokens);
        self.snapshot.processed_tokens = Some(0);
    }

    pub fn processed(&mut self, count: usize) {
        let previous = self.snapshot.processed_tokens.unwrap_or(0);
        self.snapshot.processed_tokens = Some(match previous.checked_add(count) {
            Some(total) => total,
            None => {
                self.snapshot.progress_counter_overflow = true;
                usize::MAX
            }
        });
    }

    pub fn prefill_end(
        &mut self,
        now: Instant,
        reused: usize,
        processed: usize,
        outcome: StageOutcome,
    ) {
        self.snapshot.prefill = self
            .prefill_start
            .map(|start| now.saturating_duration_since(start));
        self.snapshot.reused_tokens = Some(reused);
        self.snapshot.processed_tokens = Some(processed);
        self.snapshot.prefill_outcome = Some(outcome);
        if outcome == StageOutcome::Complete {
            self.decode_start = Some(now);
        }
    }

    pub fn raw(&mut self, now: Instant, text: &str) {
        if !text.is_empty() && self.snapshot.first_raw.is_none() {
            self.snapshot.first_raw = self.entry.map(|start| now.saturating_duration_since(start));
        }
        self.snapshot.raw_decoded.push(text);
    }

    pub fn delivered(&mut self, now: Instant, text: &str) {
        if !text.is_empty() && self.snapshot.first_delivered.is_none() {
            self.snapshot.first_delivered =
                self.entry.map(|start| now.saturating_duration_since(start));
        }
        self.snapshot.delivered.push(text);
    }

    pub fn finish(&mut self, now: Instant, outcome: ModelOutcome) {
        self.snapshot.decode = self
            .decode_start
            .map(|start| now.saturating_duration_since(start));
        self.snapshot.outcome = Some(outcome);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_capture_preserves_newline_and_utf8_prefix() {
        let mut text = CapturedText::new(5);
        text.push("a\n");
        text.push("éé");
        text.push("tail");
        assert_eq!(text.text, "a\né");
        assert_eq!(text.observed_bytes, 10);
        assert!(text.truncated);
        assert!(!text.counter_overflow);
    }

    #[test]
    fn captured_text_preserves_whitespace_and_newlines_exactly() {
        let mut text = CapturedText::new(32);
        text.push("  first\n\tsecond  \n");

        assert_eq!(text.text, "  first\n\tsecond  \n");
        assert_eq!(text.observed_bytes, 18);
        assert!(!text.truncated);
    }

    #[test]
    fn captured_text_truncates_only_at_utf8_boundaries() {
        for (cap, expected) in [(1, ""), (2, "é"), (3, "é"), (4, "éé")] {
            let mut text = CapturedText::new(cap);
            text.push("éé");

            assert_eq!(text.text, expected, "cap {cap}");
            assert_eq!(text.observed_bytes, 4);
            assert_eq!(text.truncated, expected.len() < 4);
        }
    }

    #[test]
    fn captured_text_does_not_append_after_multibyte_truncation() {
        let mut text = CapturedText::new(1);
        text.push("é");
        text.push("x");

        assert_eq!(text.text, "");
        assert_eq!(text.observed_bytes, 3);
        assert!(text.truncated);
    }

    #[test]
    fn captured_text_marks_observed_byte_counter_overflow() {
        let mut text = CapturedText::new(8);
        text.observed_bytes = usize::MAX;
        text.push("x");

        assert_eq!(text.observed_bytes, usize::MAX);
        assert!(text.counter_overflow);
        assert_eq!(text.text, "x");
    }

    #[test]
    fn captured_ids_preserve_order_and_stop_at_cap() {
        let mut captured = CapturedIds::new(2);
        captured.push(7);
        captured.push(11);
        captured.push(13);

        assert_eq!(captured.ids, [7, 11]);
        assert_eq!(captured.observed_tokens, 3);
        assert!(captured.truncated);
        assert!(!captured.counter_overflow);
    }

    #[test]
    fn captured_ids_mark_observed_token_counter_overflow() {
        let mut captured = CapturedIds::new(2);
        captured.observed_tokens = usize::MAX;
        captured.push(7);

        assert_eq!(captured.observed_tokens, usize::MAX);
        assert!(captured.counter_overflow);
        assert_eq!(captured.ids, [7]);
    }

    #[test]
    fn capture_limits_reject_zero_and_over_limit_values() {
        for limits in [
            CaptureLimits {
                text_bytes: 0,
                token_ids: 1,
            },
            CaptureLimits {
                text_bytes: MAX_TEXT_BYTES + 1,
                token_ids: 1,
            },
            CaptureLimits {
                text_bytes: 1,
                token_ids: 0,
            },
            CaptureLimits {
                text_bytes: 1,
                token_ids: MAX_CAPTURE_IDS + 1,
            },
        ] {
            assert!(matches!(
                limits.validate(),
                Err("invalid fixture capture bounds")
            ));
            assert!(ModelRecorder::new(limits).is_err());
        }
    }

    #[test]
    fn processed_counter_saturates_and_records_overflow() {
        let mut recorder = ModelRecorder::new(CaptureLimits {
            text_bytes: 1,
            token_ids: 1,
        })
        .unwrap();
        recorder.snapshot.processed_tokens = Some(usize::MAX);
        recorder.processed(1);

        assert_eq!(recorder.snapshot.processed_tokens, Some(usize::MAX));
        assert!(recorder.snapshot.progress_counter_overflow);
    }

    #[test]
    fn empty_text_events_leave_first_token_times_absent() {
        let start = Instant::now();
        let mut recorder = ModelRecorder::new(CaptureLimits {
            text_bytes: 8,
            token_ids: 1,
        })
        .unwrap();
        recorder.begin(start);
        recorder.raw(start, "");
        recorder.delivered(start, "");

        assert_eq!(recorder.snapshot.first_raw, None);
        assert_eq!(recorder.snapshot.first_delivered, None);
    }

    #[test]
    fn explicit_instants_record_stage_and_first_text_timings() {
        let start = Instant::now();
        let at = |ms| start.checked_add(Duration::from_millis(ms)).unwrap();
        let mut recorder = ModelRecorder::new(CaptureLimits {
            text_bytes: 64,
            token_ids: 8,
        })
        .unwrap();
        recorder.begin(at(0));
        recorder.tokenization_done(at(2));
        recorder.prefill_begin(at(2), 7);
        recorder.prefill_end(at(5), 2, 5, StageOutcome::Complete);
        recorder.raw(at(6), "x");
        recorder.delivered(at(8), "x");
        recorder.finish(at(10), ModelOutcome::Complete);

        assert_eq!(
            recorder.snapshot.tokenization,
            Some(Duration::from_millis(2))
        );
        assert_eq!(recorder.snapshot.prefill, Some(Duration::from_millis(3)));
        assert_eq!(recorder.snapshot.first_raw, Some(Duration::from_millis(6)));
        assert_eq!(
            recorder.snapshot.first_delivered,
            Some(Duration::from_millis(8))
        );
        assert_eq!(recorder.snapshot.decode, Some(Duration::from_millis(5)));
        assert_eq!(recorder.snapshot.prompt_tokens, Some(7));
        assert_eq!(recorder.snapshot.reused_tokens, Some(2));
        assert_eq!(recorder.snapshot.processed_tokens, Some(5));
        assert_eq!(recorder.snapshot.outcome, Some(ModelOutcome::Complete));
    }

    #[test]
    fn incomplete_prefill_outcomes_do_not_start_decode_timing() {
        let start = Instant::now();
        for outcome in [StageOutcome::Cancelled, StageOutcome::ModelError] {
            let mut recorder = ModelRecorder::new(CaptureLimits {
                text_bytes: 1,
                token_ids: 1,
            })
            .unwrap();
            recorder.prefill_begin(start, 0);
            recorder.prefill_end(start, 0, 0, outcome);
            recorder.finish(start, ModelOutcome::Cancelled);

            assert_eq!(recorder.snapshot.prefill_outcome, Some(outcome));
            assert_eq!(recorder.snapshot.decode, None);
        }
    }

    #[test]
    fn fixture_lifecycle_tokenization_only_has_no_prefill_or_decode() {
        let entry = Instant::now();
        for outcome in [
            ModelOutcome::Complete,
            ModelOutcome::Cancelled,
            ModelOutcome::ModelError,
        ] {
            let mut recorder = ModelRecorder::new(CaptureLimits {
                text_bytes: 8,
                token_ids: 2,
            })
            .unwrap();
            recorder.begin(entry);
            recorder.tokenization_done(entry + Duration::from_millis(2));
            recorder.finish(entry + Duration::from_millis(5), outcome);

            assert_eq!(
                recorder.snapshot.tokenization,
                Some(Duration::from_millis(2))
            );
            assert_eq!(recorder.snapshot.prefill, None);
            assert_eq!(recorder.snapshot.decode, None);
            assert_eq!(recorder.snapshot.processed_tokens, None);
            assert_eq!(recorder.snapshot.first_raw, None);
            assert_eq!(recorder.snapshot.first_delivered, None);
            assert_eq!(recorder.snapshot.outcome, Some(outcome));
        }
    }

    #[test]
    fn fixture_lifecycle_partial_prefill_retains_completed_work() {
        let entry = Instant::now();
        for (stage, outcome) in [
            (StageOutcome::Cancelled, ModelOutcome::Cancelled),
            (StageOutcome::ModelError, ModelOutcome::ModelError),
        ] {
            let mut recorder = ModelRecorder::new(CaptureLimits {
                text_bytes: 8,
                token_ids: 2,
            })
            .unwrap();
            recorder.begin(entry);
            recorder.prefill_begin(entry + Duration::from_millis(2), 9);
            recorder.processed(2);
            let completed = recorder.snapshot.processed_tokens.unwrap();
            recorder.prefill_end(entry + Duration::from_millis(8), 4, completed, stage);
            assert_eq!(recorder.snapshot.outcome, None);
            recorder.finish(entry + Duration::from_millis(12), outcome);

            assert_eq!(recorder.snapshot.prefill, Some(Duration::from_millis(6)));
            assert_eq!(recorder.snapshot.reused_tokens, Some(4));
            assert_eq!(recorder.snapshot.processed_tokens, Some(2));
            assert_eq!(recorder.snapshot.prefill_outcome, Some(stage));
            assert_eq!(recorder.snapshot.decode, None);
            assert_eq!(recorder.snapshot.outcome, Some(outcome));
        }
    }

    #[test]
    fn fixture_lifecycle_finish_measures_through_physical_return() {
        let entry = Instant::now();
        let mut recorder = ModelRecorder::new(CaptureLimits {
            text_bytes: 8,
            token_ids: 2,
        })
        .unwrap();
        recorder.begin(entry);
        recorder.prefill_begin(entry + Duration::from_millis(2), 1);
        recorder.prefill_end(
            entry + Duration::from_millis(3),
            0,
            1,
            StageOutcome::Complete,
        );
        recorder.raw(entry + Duration::from_millis(4), "x");
        recorder.delivered(entry + Duration::from_millis(7), "x");
        assert_eq!(recorder.snapshot.outcome, None);
        assert_eq!(recorder.snapshot.decode, None);
        recorder.finish(
            entry + Duration::from_millis(20),
            ModelOutcome::DeliveryStopped,
        );

        assert_eq!(recorder.snapshot.first_raw, Some(Duration::from_millis(4)));
        assert_eq!(
            recorder.snapshot.first_delivered,
            Some(Duration::from_millis(7))
        );
        assert_eq!(recorder.snapshot.decode, Some(Duration::from_millis(17)));
        assert_eq!(
            recorder.snapshot.outcome,
            Some(ModelOutcome::DeliveryStopped)
        );
    }

    #[test]
    fn fixture_lifecycle_raw_stop_and_utf8_flush_remain_separate_from_delivery() {
        let entry = Instant::now();
        let mut recorder = ModelRecorder::new(CaptureLimits {
            text_bytes: 16,
            token_ids: 2,
        })
        .unwrap();
        let mut decoder = mivi_tokenizer::Utf8StreamDecoder::new();
        recorder.begin(entry);
        let raw = decoder.feed(b"xSTOP");
        recorder.raw(entry + Duration::from_millis(4), &raw);
        recorder.snapshot.generated_ids.push(9);
        recorder.delivered(entry + Duration::from_millis(5), &raw[..1]);
        let incomplete = decoder.feed(&[0xc3]);
        assert!(incomplete.is_empty());
        recorder.raw(entry + Duration::from_millis(6), &incomplete);
        recorder.snapshot.generated_ids.push(10);
        let flushed = decoder.flush();
        recorder.raw(entry + Duration::from_millis(7), &flushed);
        recorder.delivered(entry + Duration::from_millis(8), &flushed);

        assert_eq!(recorder.snapshot.raw_decoded.observed_bytes, 8);
        assert_eq!(recorder.snapshot.delivered.observed_bytes, 4);
        assert!(recorder.snapshot.raw_decoded.text.ends_with(&flushed));
        assert!(recorder.snapshot.delivered.text.ends_with(&flushed));
        assert_eq!(recorder.snapshot.generated_ids.ids, [9, 10]);
        assert_eq!(recorder.snapshot.first_raw, Some(Duration::from_millis(4)));
        assert_eq!(
            recorder.snapshot.first_delivered,
            Some(Duration::from_millis(5))
        );
    }
}
