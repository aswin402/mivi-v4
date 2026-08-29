//! Token sampler with Temperature, Top-K, Top-P, and Repetition Penalty.

use mivi_core::math::softmax;

#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Sampler {
    config: SamplerConfig,
    rng_state: u64,
}

impl Sampler {
    pub fn new(config: SamplerConfig) -> Self {
        Self::with_seed(config, None)
    }

    pub fn with_seed(config: SamplerConfig, seed: Option<u64>) -> Self {
        let rng_state = seed.unwrap_or_else(|| {
            use std::time::{SystemTime, UNIX_EPOCH};
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;
            if nanos == 0 { 0x853c49e6748fea9b } else { nanos }
        });
        Self { config, rng_state }
    }

    /// Fast stateful xorshift64* pseudo-random number generator producing a float in [0.0, 1.0).
    #[inline]
    pub fn random_f32(&mut self) -> f32 {
        self.rng_state ^= self.rng_state >> 12;
        self.rng_state ^= self.rng_state << 25;
        self.rng_state ^= self.rng_state >> 27;
        let r = self.rng_state.wrapping_mul(0x2545F4914F6CDD1D) >> 32;
        (r >> 8) as f32 / 16777216.0
    }

    /// Sample token index from raw unnormalized logits.
    pub fn sample(&mut self, logits: &mut [f32], recent_tokens: &[u32]) -> u32 {
        if self.config.temperature <= 0.0 {
            // Greedy argmax
            return argmax(logits);
        }

        // 1. Repetition penalty
        if self.config.repetition_penalty != 1.0 {
            for &token in recent_tokens {
                let idx = token as usize;
                if idx < logits.len() {
                    if logits[idx] > 0.0 {
                        logits[idx] /= self.config.repetition_penalty;
                    } else {
                        logits[idx] *= self.config.repetition_penalty;
                    }
                }
            }
        }

        // 2. Temperature scaling
        let inv_temp = 1.0 / self.config.temperature;
        for l in logits.iter_mut() {
            *l *= inv_temp;
        }

        // 3. Top-K filter
        if self.config.top_k > 0 && self.config.top_k < logits.len() {
            let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let cutoff = indexed[self.config.top_k - 1].1;
            for l in logits.iter_mut() {
                if *l < cutoff {
                    *l = f32::NEG_INFINITY;
                }
            }
        }

        // 4. Softmax
        softmax(logits);

        // 5. Top-P (nucleus) sampling
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumsum = 0.0;
        let mut filtered = Vec::new();
        for (idx, prob) in indexed {
            cumsum += prob;
            filtered.push((idx, prob));
            if cumsum >= self.config.top_p {
                break;
            }
        }

        // Renormalize and sample with random uniform
        let sum: f32 = filtered.iter().map(|(_, p)| *p).sum();
        let r = self.random_f32() * sum;
        let mut acc = 0.0;
        for (idx, prob) in filtered {
            acc += prob;
            if acc >= r {
                return idx as u32;
            }
        }

        argmax(logits)
    }
}

#[inline]
fn argmax(logits: &[f32]) -> u32 {
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
