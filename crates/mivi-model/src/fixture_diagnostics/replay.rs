//! Bounded, opt-in replay inputs and numeric forward-profile exports.
//!
//! Input validation is model-independent. A replay runner must also check token
//! IDs against the loaded vocabulary and the length after BOS normalization.
//! Callers must bound input bytes before deserialization: `validate()` checks
//! vectors only after serde has allocated them and is not an input-reader guard.

use crate::ForwardProfileSnapshot;
use serde::{Deserialize, Serialize};

pub const MAX_REPLAY_CONTEXT: usize = 4096;
pub const MAX_REPLAY_TILE: usize = 128;
pub const MAX_REPLAY_OUTPUT: usize = 64;
pub const MAX_TEACHER_POSITIONS: usize = 16;
pub const MAX_LOGIT_IDS: usize = 64;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayInput {
    pub prompt_ids: Vec<u32>,
    pub context: usize,
    pub tile: usize,
    pub max_tokens: usize,
    pub profile: bool,
    pub split_prefill: bool,
    pub teacher_forced_ids: Vec<u32>,
    pub logit_ids: Vec<u32>,
}

/// Aggregate and substage times overlap: do not sum them as independent work.
#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct ProfileMicros {
    pub tokens: usize,
    pub embedding: u64,
    pub attention: u64,
    pub ssm: u64,
    pub logits: u64,
    /// Order: norm, QKV projection, causal attention, output projection, FFN.
    pub attention_substages: [u64; 5],
    /// Order: norm, input projection, convolution, output projection, FFN.
    pub ssm_substages: [u64; 5],
}

impl ReplayInput {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.prompt_ids.is_empty() || self.prompt_ids.len() > MAX_REPLAY_CONTEXT {
            return Err("invalid replay prompt length");
        }
        if self.context == 0 || self.context > MAX_REPLAY_CONTEXT {
            return Err("invalid replay context");
        }
        if self.tile == 0 || self.tile > MAX_REPLAY_TILE {
            return Err("invalid replay tile");
        }
        if self.max_tokens > MAX_REPLAY_OUTPUT {
            return Err("invalid replay output limit");
        }
        if self.teacher_forced_ids.len() > MAX_TEACHER_POSITIONS {
            return Err("too many teacher-forced positions");
        }
        if !self.teacher_forced_ids.is_empty() && self.max_tokens != 0 {
            return Err("teacher forcing cannot be combined with sampling");
        }
        if self.logit_ids.len() > MAX_LOGIT_IDS {
            return Err("too many selected logit IDs");
        }
        for (index, id) in self.logit_ids.iter().enumerate() {
            if self.logit_ids[..index].contains(id) {
                return Err("duplicate selected logit ID");
            }
        }
        let required = self
            .prompt_ids
            .len()
            .checked_add(self.max_tokens)
            .and_then(|n| n.checked_add(self.teacher_forced_ids.len()))
            .ok_or("replay context length overflow")?;
        if required > self.context {
            return Err("replay exceeds configured context");
        }
        Ok(())
    }
}

impl ProfileMicros {
    pub fn from_snapshot(s: ForwardProfileSnapshot) -> Result<Self, &'static str> {
        fn micros(duration: std::time::Duration) -> Result<u64, &'static str> {
            u64::try_from(duration.as_micros()).map_err(|_| "profile duration overflow")
        }
        Ok(Self {
            tokens: s.tokens,
            embedding: micros(s.embedding)?,
            attention: micros(s.attention)?,
            ssm: micros(s.ssm)?,
            logits: micros(s.logits)?,
            attention_substages: [
                micros(s.attention_stages.norm)?,
                micros(s.attention_stages.qkv_projection)?,
                micros(s.attention_stages.causal_attention)?,
                micros(s.attention_stages.output_projection)?,
                micros(s.attention_stages.ffn)?,
            ],
            ssm_substages: [
                micros(s.ssm_stages.norm)?,
                micros(s.ssm_stages.input_projection)?,
                micros(s.ssm_stages.convolution)?,
                micros(s.ssm_stages.output_projection)?,
                micros(s.ssm_stages.ffn)?,
            ],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn input() -> ReplayInput {
        ReplayInput {
            prompt_ids: vec![1, 2],
            context: 128,
            tile: 64,
            max_tokens: 48,
            profile: false,
            split_prefill: false,
            teacher_forced_ids: vec![],
            logit_ids: vec![],
        }
    }

    #[test]
    fn replay_accepts_valid_generation_and_teacher_inputs() {
        let mut r = input();
        assert_eq!(r.validate(), Ok(()));
        r.max_tokens = 0;
        r.teacher_forced_ids = vec![3, 4];
        r.logit_ids = vec![0, 7];
        assert_eq!(r.validate(), Ok(()));
    }

    #[test]
    fn replay_accepts_exact_limits() {
        let mut r = input();
        r.context = MAX_REPLAY_CONTEXT;
        r.tile = MAX_REPLAY_TILE;
        r.max_tokens = MAX_REPLAY_OUTPUT;
        r.prompt_ids = vec![0; MAX_REPLAY_CONTEXT - MAX_REPLAY_OUTPUT];
        assert_eq!(r.validate(), Ok(()));
        r.max_tokens = 0;
        r.teacher_forced_ids = vec![1; MAX_TEACHER_POSITIONS];
        r.prompt_ids = vec![0; MAX_REPLAY_CONTEXT - MAX_TEACHER_POSITIONS];
        r.logit_ids = (0..MAX_LOGIT_IDS as u32).collect();
        assert_eq!(r.validate(), Ok(()));
        r.teacher_forced_ids.clear();
        r.prompt_ids = vec![0; MAX_REPLAY_CONTEXT];
        assert_eq!(r.validate(), Ok(()));
        r.context = 1;
        r.tile = 1;
        r.prompt_ids = vec![0];
        assert_eq!(r.validate(), Ok(()));
    }

    #[test]
    fn replay_rejects_unbounded_input() {
        for context in [0, MAX_REPLAY_CONTEXT + 1, usize::MAX] {
            let mut r = input();
            r.context = context;
            assert!(r.validate().is_err());
        }
        for tile in [0, MAX_REPLAY_TILE + 1, usize::MAX] {
            let mut r = input();
            r.tile = tile;
            assert!(r.validate().is_err());
        }
        for output in [MAX_REPLAY_OUTPUT + 1, usize::MAX] {
            let mut r = input();
            r.max_tokens = output;
            assert!(r.validate().is_err());
        }
        let mut r = input();
        r.prompt_ids = vec![0; MAX_REPLAY_CONTEXT + 1];
        assert!(r.validate().is_err());
    }

    #[test]
    fn replay_rejects_empty_prompt_and_context_overrun() {
        let mut r = input();
        r.prompt_ids.clear();
        assert!(r.validate().is_err());
        r.prompt_ids = vec![1, 2];
        r.context = r.prompt_ids.len() + r.max_tokens - 1;
        assert!(r.validate().is_err());
        r.max_tokens = 0;
        r.context = 2;
        r.teacher_forced_ids = vec![3];
        assert!(r.validate().is_err());
    }

    #[test]
    fn replay_rejects_probe_limits_and_duplicate_logits() {
        let mut r = input();
        r.max_tokens = 0;
        r.teacher_forced_ids = vec![1; MAX_TEACHER_POSITIONS + 1];
        assert!(r.validate().is_err());
        r.teacher_forced_ids.clear();
        r.logit_ids = (0..=MAX_LOGIT_IDS as u32).collect();
        assert!(r.validate().is_err());
        r.logit_ids = vec![7, 1, 7];
        assert!(r.validate().is_err());
    }

    #[test]
    fn replay_rejects_teacher_forcing_with_sampling() {
        let mut r = input();
        r.teacher_forced_ids = vec![3];
        assert!(r.validate().is_err());
    }

    #[test]
    fn replay_json_requires_known_fields_and_integer_values() {
        let valid = r#"{"prompt_ids":[1],"context":128,"tile":64,"max_tokens":0,"profile":false,"split_prefill":false,"teacher_forced_ids":[],"logit_ids":[]}"#;
        let parsed: ReplayInput = serde_json::from_str(valid).unwrap();
        assert_eq!(parsed.validate(), Ok(()));
        let unknown = valid.replace("\"context\":128", "\"context\":128,\"extra\":1");
        assert!(serde_json::from_str::<ReplayInput>(&unknown).is_err());
        for value in ["-1", "1.5", "null", "\"128\""] {
            let bad = valid.replace("\"context\":128", &format!("\"context\":{value}"));
            assert!(serde_json::from_str::<ReplayInput>(&bad).is_err());
        }
        let missing = valid.replace("\"profile\":false,", "");
        assert!(serde_json::from_str::<ReplayInput>(&missing).is_err());
    }

    #[test]
    fn profile_export_preserves_substages() {
        let mut s = ForwardProfileSnapshot::default();
        s.tokens = 7;
        s.embedding = Duration::from_micros(11);
        s.attention = Duration::from_micros(101);
        s.ssm = Duration::from_micros(103);
        s.logits = Duration::from_micros(13);
        s.attention_stages.norm = Duration::from_micros(2);
        s.attention_stages.qkv_projection = Duration::from_micros(17);
        s.attention_stages.causal_attention = Duration::from_micros(3);
        s.attention_stages.output_projection = Duration::from_micros(5);
        s.attention_stages.ffn = Duration::from_micros(7);
        s.ssm_stages.norm = Duration::from_micros(11);
        s.ssm_stages.input_projection = Duration::from_micros(13);
        s.ssm_stages.convolution = Duration::from_micros(23);
        s.ssm_stages.output_projection = Duration::from_micros(17);
        s.ssm_stages.ffn = Duration::from_micros(19);
        let p = ProfileMicros::from_snapshot(s).unwrap();
        assert_eq!(p.attention_substages, [2, 17, 3, 5, 7]);
        assert_eq!(p.ssm_substages, [11, 13, 23, 17, 19]);
        assert_eq!(
            (p.tokens, p.embedding, p.attention, p.ssm, p.logits),
            (7, 11, 101, 103, 13)
        );
        let json = serde_json::to_value(p).unwrap();
        assert_eq!(json["attention_substages"][1], 17);
        assert_eq!(json["ssm_substages"][2], 23);
    }

    #[test]
    fn profile_export_handles_zero_and_submicrosecond_values() {
        let mut s = ForwardProfileSnapshot::default();
        s.embedding = Duration::from_nanos(999);
        let p = ProfileMicros::from_snapshot(s).unwrap();
        assert_eq!(p.embedding, 0);
        assert_eq!(p.attention_substages, [0; 5]);
        assert_eq!(p.ssm_substages, [0; 5]);
    }

    #[test]
    fn profile_export_accepts_maximum_microseconds() {
        let mut s = ForwardProfileSnapshot::default();
        s.embedding = Duration::from_micros(u64::MAX);
        assert_eq!(ProfileMicros::from_snapshot(s).unwrap().embedding, u64::MAX);
    }

    #[test]
    fn profile_export_rejects_overflow_in_every_field() {
        let too_large = Duration::from_secs(u64::MAX);
        for index in 0..14 {
            let mut s = ForwardProfileSnapshot::default();
            let fields = [
                &mut s.embedding,
                &mut s.attention,
                &mut s.ssm,
                &mut s.logits,
                &mut s.attention_stages.norm,
                &mut s.attention_stages.qkv_projection,
                &mut s.attention_stages.causal_attention,
                &mut s.attention_stages.output_projection,
                &mut s.attention_stages.ffn,
                &mut s.ssm_stages.norm,
                &mut s.ssm_stages.input_projection,
                &mut s.ssm_stages.convolution,
                &mut s.ssm_stages.output_projection,
                &mut s.ssm_stages.ffn,
            ];
            *fields[index] = too_large;
            assert!(ProfileMicros::from_snapshot(s).is_err(), "field {index}");
        }
    }
}
