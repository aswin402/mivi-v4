//! Low-Rank Adaptation (LoRA) dynamic adapter loading and forward compute.

use std::collections::HashMap;

/// Individual LoRA projection pair (A: down projection, B: up projection).
#[derive(Debug, Clone)]
pub struct LoraWeightPair {
    pub rank: usize,
    pub alpha: f32,
    // Matrix A: [rank, in_dim]
    pub a: Vec<f32>,
    // Matrix B: [out_dim, rank]
    pub b: Vec<f32>,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl LoraWeightPair {
    pub fn new(rank: usize, alpha: f32, in_dim: usize, out_dim: usize) -> Self {
        assert!(rank > 0, "LoRA rank must be greater than 0");
        Self {
            rank,
            alpha,
            a: vec![0.0f32; rank * in_dim],
            b: vec![0.0f32; out_dim * rank],
            in_dim,
            out_dim,
        }
    }

    /// Compute LoRA delta: delta = (alpha / rank) * B * (A * x)
    /// down_buf: temporary scratch buffer of size `rank`
    /// out_buf: output accumulation buffer of size `out_dim`
    #[inline]
    pub fn apply(&self, x: &[f32], down_buf: &mut [f32], out_buf: &mut [f32], scale_factor: f32) {
        if self.rank == 0 {
            return;
        }
        debug_assert_eq!(x.len(), self.in_dim);
        debug_assert!(down_buf.len() >= self.rank);
        debug_assert_eq!(out_buf.len(), self.out_dim);

        // 1. A * x -> down_buf [rank]
        for r in 0..self.rank {
            let row = &self.a[r * self.in_dim..(r + 1) * self.in_dim];
            let mut sum = 0.0f32;
            for j in 0..self.in_dim {
                sum += row[j] * x[j];
            }
            down_buf[r] = sum;
        }

        // 2. B * down_buf -> out_buf [out_dim] with scaling (alpha / rank) * scale_factor
        let eff_scale = (self.alpha / (self.rank as f32)) * scale_factor;
        for i in 0..self.out_dim {
            let row = &self.b[i * self.rank..(i + 1) * self.rank];
            let mut sum = 0.0f32;
            for r in 0..self.rank {
                sum += row[r] * down_buf[r];
            }
            out_buf[i] += sum * eff_scale;
        }
    }
}

/// Expert LoRA Adapter (e.g. AGENT, CODE, DEBUG, RESEARCH, CHAT, GENERAL).
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    pub name: String,
    pub weights: HashMap<String, LoraWeightPair>,
}

impl LoraAdapter {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            weights: HashMap::new(),
        }
    }

    pub fn add_weight_pair(&mut self, module_path: &str, pair: LoraWeightPair) {
        self.weights.insert(module_path.to_string(), pair);
    }

    pub fn apply_to_module(
        &self,
        module_path: &str,
        x: &[f32],
        down_buf: &mut [f32],
        out_buf: &mut [f32],
        blend_weight: f32,
    ) {
        if let Some(pair) = self.weights.get(module_path) {
            pair.apply(x, down_buf, out_buf, blend_weight);
        }
    }
}

/// Composed Multi-Adapter Engine (combines multiple active LoRA experts dynamically).
#[derive(Debug, Default, Clone)]
pub struct ActiveAdapters {
    // List of (adapter, weight)
    pub active: Vec<(LoraAdapter, f32)>,
}

impl ActiveAdapters {
    pub fn new() -> Self {
        Self { active: Vec::new() }
    }

    pub fn add(&mut self, adapter: LoraAdapter, weight: f32) {
        self.active.push((adapter, weight));
    }

    pub fn clear(&mut self) {
        self.active.clear();
    }

    /// Apply all active adapters to target module output in-place
    pub fn apply_module(
        &self,
        module_path: &str,
        x: &[f32],
        down_buf: &mut [f32],
        out_buf: &mut [f32],
    ) {
        for (adapter, weight) in &self.active {
            adapter.apply_to_module(module_path, x, down_buf, out_buf, *weight);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_apply() {
        let in_dim = 4;
        let out_dim = 4;
        let rank = 2;
        let alpha = 2.0;

        let mut pair = LoraWeightPair::new(rank, alpha, in_dim, out_dim);
        // A matrix = [[1, 0, 0, 0], [0, 1, 0, 0]]
        pair.a[0] = 1.0;
        pair.a[1 * in_dim + 1] = 1.0;
        // B matrix = [[1, 0], [0, 1], [0, 0], [0, 0]]
        pair.b[0] = 1.0;
        pair.b[1 * rank + 1] = 1.0;

        let x = vec![2.0, 3.0, 0.0, 0.0];
        let mut down = vec![0.0; rank];
        let mut out = vec![0.0; out_dim];

        pair.apply(&x, &mut down, &mut out, 1.0);

        // scale = alpha / rank = 2.0 / 2.0 = 1.0
        // down = [2.0, 3.0]
        // out[0] = 2.0 * 1.0 = 2.0
        // out[1] = 3.0 * 1.0 = 3.0
        assert_eq!(out[0], 2.0);
        assert_eq!(out[1], 3.0);
    }
}
