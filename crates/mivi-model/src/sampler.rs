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
}

impl Sampler {
    pub fn new(config: GenerationConfig) -> Self {
        Self::with_seed(config, None)
    }

    pub fn with_seed(config: GenerationConfig, seed: Option<u64>) -> Self {
        let rng_state = seed.unwrap_or_else(|| {
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
        Self {
            config,
            rng_state,
            scratch: Vec::with_capacity(SAMPLER_SCRATCH_CAPACITY),
        }
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

    /// Sample token index from raw unnormalized logits without per-token heap allocations.
    pub fn sample(&mut self, logits: &mut [f32], recent_tokens: &[u32]) -> u32 {
        if self.config.temperature <= 0.0 {
            return argmax(logits);
        }

        // 1. Repetition penalty
        if self.config.repetition_penalty != 1.0 {
            apply_repetition_penalty(logits, recent_tokens, self.config.repetition_penalty);
        }

        // 2. Temperature scaling
        apply_temperature(logits, self.config.temperature);

        // 3. Top-K filtering
        if self.config.top_k > 0 && self.config.top_k < logits.len() {
            apply_top_k(logits, &mut self.scratch, self.config.top_k);
        }

        // 4. Softmax
        softmax(logits);

        // 5. Top-P (nucleus) sampling
        self.sample_top_p(logits)
    }

    fn sample_top_p(&mut self, logits: &[f32]) -> u32 {
        self.scratch.clear();
        self.scratch
            .extend(logits.iter().copied().enumerate().filter(|(_, p)| *p > 0.0));
        self.scratch
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumsum = 0.0;
        let mut cutoff_idx = self.scratch.len();
        for (i, &(_, prob)) in self.scratch.iter().enumerate() {
            cumsum += prob;
            if cumsum >= self.config.top_p {
                cutoff_idx = i + 1;
                break;
            }
        }

        let rand_val = self.random_f32();
        let eligible = &self.scratch[..cutoff_idx];
        let sum: f32 = eligible.iter().map(|(_, p)| *p).sum();
        let r = rand_val * sum;
        let mut acc = 0.0;
        for &(idx, prob) in eligible {
            acc += prob;
            if acc >= r {
                return idx as u32;
            }
        }

        argmax(logits)
    }
}

#[inline]
fn apply_repetition_penalty(logits: &mut [f32], recent_tokens: &[u32], penalty: f32) {
    for &token in recent_tokens {
        let idx = token as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
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
