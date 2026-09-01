use crate::config::GenerationConfig;
use mivi_core::math::softmax;

pub type SamplerConfig = GenerationConfig;

/// Default RNG seed when system clock returns zero.
const FALLBACK_RNG_SEED: u64 = 0x853c49e6748fea9b;
/// Initial scratch buffer capacity for sampler sorting.
const SAMPLER_SCRATCH_CAPACITY: usize = 65536;
/// Xorshift64* multiplier (Vigna, 2014).
const XORSHIFT_MULTIPLIER: u64 = 0x2545F4914F6CDD1D;
/// Divisor for converting 24-bit integer to [0.0, 1.0) float (2^24).
const F32_FROM_U24: f32 = 16_777_216.0;

#[derive(Debug, Clone)]
pub struct Sampler {
    pub config: GenerationConfig,
    rng_state: u64,
    scratch: Vec<(usize, f32)>,
    seen_tokens: Vec<u32>,
}

impl Sampler {
    pub fn new(config: GenerationConfig) -> Self {
        let seed = config.seed;
        Self::with_seed(config, seed)
    }

    pub fn with_seed(config: GenerationConfig, seed: Option<u64>) -> Self {
        let mut rng_state = seed.unwrap_or_else(|| {
            use std::time::{SystemTime, UNIX_EPOCH};
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;
            if nanos == 0 {
                FALLBACK_RNG_SEED
            } else {
                nanos
            }
        });
        if rng_state == 0 {
            rng_state = FALLBACK_RNG_SEED;
        }
        Self {
            config,
            rng_state,
            scratch: Vec::with_capacity(SAMPLER_SCRATCH_CAPACITY),
            seen_tokens: Vec::with_capacity(256),
        }
    }

    /// Set RNG seed explicitly.
    pub fn set_seed(&mut self, seed: u64) {
        self.rng_state = if seed == 0 { FALLBACK_RNG_SEED } else { seed };
    }

    /// Fast stateful xorshift64* pseudo-random number generator producing a float in [0.0, 1.0).
    #[inline]
    pub fn random_f32(&mut self) -> f32 {
        self.rng_state ^= self.rng_state >> 12;
        self.rng_state ^= self.rng_state << 25;
        self.rng_state ^= self.rng_state >> 27;
        let r = self.rng_state.wrapping_mul(XORSHIFT_MULTIPLIER) >> 32;
        (r >> 8) as f32 / F32_FROM_U24
    }

    /// Sample token index applying an optional grammar bitmask prior to sampling.
    pub fn sample_with_mask(
        &mut self,
        logits: &mut [f32],
        recent_tokens: &[u32],
        mask: Option<&crate::grammar::TokenBitMask>,
    ) -> u32 {
        if let Some(mask) = mask {
            mask.apply_to_logits(logits);
        }
        self.sample(logits, recent_tokens)
    }

    /// Sample token index from unnormalized logits without per-token heap allocations.
    pub fn sample(&mut self, logits: &mut [f32], recent_tokens: &[u32]) -> u32 {
        if self.config.temperature <= 0.0 {
            return argmax(logits);
        }

        // 1. Penalties (repetition, presence, frequency)
        if !recent_tokens.is_empty()
            && (self.config.repetition_penalty > 1.0
                || self.config.presence_penalty != 0.0
                || self.config.frequency_penalty != 0.0)
        {
            apply_penalties(
                logits,
                recent_tokens,
                self.config.repetition_penalty,
                self.config.presence_penalty,
                self.config.frequency_penalty,
                &mut self.seen_tokens,
            );
        }

        // 2. Temperature scaling
        apply_temperature(logits, self.config.temperature);

        // 3. Min-P dynamic thresholding
        if self.config.min_p > 0.0 && self.config.min_p < 1.0 {
            apply_min_p(logits, self.config.min_p);
        }

        // 4. Top-K filtering
        if self.config.top_k > 0 && self.config.top_k < logits.len() {
            apply_top_k(logits, &mut self.scratch, self.config.top_k);
        }

        // 5. Softmax normalization
        softmax(logits);

        // 6. Top-P / Categorical sampling
        if self.config.top_p < 1.0 {
            self.sample_top_p(logits, self.config.top_p)
        } else {
            self.sample_categorical(logits)
        }
    }

    fn sample_categorical(&mut self, probs: &[f32]) -> u32 {
        let r = self.random_f32();
        let mut cumsum = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r <= cumsum {
                return i as u32;
            }
        }
        probs.len().saturating_sub(1) as u32
    }

    fn sample_top_p(&mut self, probs: &[f32], top_p: f32) -> u32 {
        self.scratch.clear();
        self.scratch.extend(
            probs
                .iter()
                .copied()
                .enumerate()
                .filter(|&(_, p)| p > 0.0 && p.is_finite()),
        );

        if self.scratch.is_empty() {
            return 0;
        }

        // Quickselect top candidates (up to 1024) to avoid sorting entire vocabulary
        let candidate_limit = 1024.min(self.scratch.len());
        if candidate_limit < self.scratch.len() {
            self.scratch
                .select_nth_unstable_by(candidate_limit - 1, |a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
            self.scratch.truncate(candidate_limit);
        }

        self.scratch
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Find nucleus: smallest set of top tokens whose cumulative probability >= top_p
        let mut nucleus_mass = 0.0f32;
        let mut cutoff_idx = self.scratch.len();
        for (i, &(_, p)) in self.scratch.iter().enumerate() {
            nucleus_mass += p;
            if nucleus_mass >= top_p {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Sample uniformly within the nucleus mass
        let r = self.random_f32() * nucleus_mass;
        let mut cumsum = 0.0f32;
        for &(idx, p) in &self.scratch[..cutoff_idx] {
            cumsum += p;
            if cumsum >= r {
                return idx as u32;
            }
        }
        self.scratch.first().map(|s| s.0 as u32).unwrap_or(0)
    }
}

#[inline]
fn apply_penalties(
    logits: &mut [f32],
    recent_tokens: &[u32],
    rep_penalty: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    seen_tokens: &mut Vec<u32>,
) {
    seen_tokens.clear();
    for &tok in recent_tokens {
        let idx = tok as usize;
        if idx >= logits.len() {
            continue;
        }
        if !seen_tokens.contains(&tok) {
            seen_tokens.push(tok);
            let count = recent_tokens.iter().filter(|&&t| t == tok).count();

            // 1. Multiplicative Repetition Penalty
            if rep_penalty > 1.0 {
                if logits[idx] > 0.0 {
                    logits[idx] /= rep_penalty;
                } else {
                    logits[idx] *= rep_penalty;
                }
            }

            // 2. Additive Presence Penalty
            if presence_penalty != 0.0 {
                logits[idx] -= presence_penalty;
            }

            // 3. Additive Frequency Penalty
            if frequency_penalty != 0.0 {
                logits[idx] -= frequency_penalty * (count as f32);
            }
        }
    }
}

#[inline]
fn apply_min_p(logits: &mut [f32], min_p: f32) {
    let mut max_logit = f32::NEG_INFINITY;
    for &l in logits.iter() {
        if l > max_logit && l.is_finite() {
            max_logit = l;
        }
    }
    if !max_logit.is_finite() {
        return;
    }
    let threshold = max_logit + min_p.ln();
    for l in logits.iter_mut() {
        if *l < threshold {
            *l = f32::NEG_INFINITY;
        }
    }
}

#[inline]
fn apply_temperature(logits: &mut [f32], temperature: f32) {
    let inv_temp = 1.0 / temperature;
    for l in logits.iter_mut() {
        *l *= inv_temp;
    }
}

#[inline]
fn apply_top_k(logits: &mut [f32], scratch: &mut Vec<(usize, f32)>, top_k: usize) {
    scratch.clear();
    scratch.extend(logits.iter().copied().enumerate());
    scratch.select_nth_unstable_by(top_k - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    let cutoff = scratch[top_k - 1].1;
    for l in logits.iter_mut() {
        if *l < cutoff {
            *l = f32::NEG_INFINITY;
        }
    }
}

#[inline]
pub fn argmax(logits: &[f32]) -> u32 {
    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    max_idx as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_seed_not_absorbing() {
        let mut sampler = Sampler::with_seed(GenerationConfig::default(), Some(0));
        let r1 = sampler.random_f32();
        let r2 = sampler.random_f32();
        assert_ne!(r1, 0.0);
        assert_ne!(r1, r2);
    }

    #[test]
    fn test_argmax() {
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        assert_eq!(argmax(&logits), 1);
    }

    #[test]
    fn test_min_p_sampling() {
        let mut logits = vec![10.0, 8.0, 5.0, 1.0];
        // max_logit is 10.0. For min_p = 0.1, threshold = 10.0 + ln(0.1) ≈ 10.0 - 2.3026 = 7.6974
        // 10.0 and 8.0 survive (> 7.6974), 5.0 and 1.0 become -inf (< 7.6974).
        apply_min_p(&mut logits, 0.1);
        assert_eq!(logits[0], 10.0);
        assert_eq!(logits[1], 8.0);
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[3], f32::NEG_INFINITY);
    }

    #[test]
    fn test_penalties_repetition_presence_frequency() {
        let mut logits = vec![5.0, 5.0, 5.0, 5.0];
        let recent = vec![0, 1, 1, 1]; // token 0 appears 1x, token 1 appears 3x
        let mut seen = Vec::new();
        apply_penalties(&mut logits, &recent, 1.0, 1.0, 0.5, &mut seen);

        // token 0: presence(1.0) + freq(0.5 * 1) = 1.5 subtracted -> 3.5
        assert!((logits[0] - 3.5).abs() < 1e-5);
        // token 1: presence(1.0) + freq(0.5 * 3) = 2.5 subtracted -> 2.5
        assert!((logits[1] - 2.5).abs() < 1e-5);
        // tokens 2 and 3 unpenalized -> 5.0
        assert_eq!(logits[2], 5.0);
        assert_eq!(logits[3], 5.0);
    }

    #[test]
    fn test_deterministic_seed() {
        let cfg = GenerationConfig {
            seed: Some(12345),
            ..Default::default()
        };
        let mut s1 = Sampler::new(cfg.clone());
        let mut s2 = Sampler::new(cfg);
        assert_eq!(s1.random_f32(), s2.random_f32());
        assert_eq!(s1.random_f32(), s2.random_f32());
    }
}
