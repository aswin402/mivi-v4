//! Workload-Adaptive Expert Learning Cache (`.mivi/expert_heat.json`).
//!
//! Inspired by AirLLM and Colibrì research:
//! 1. Tracks Mixture-of-Experts (MoE) routing frequencies across user sessions.
//! 2. Maintains Exponential Moving Average (EMA) heat scores to capture workload shifts.
//! 3. Dynamically pins the hottest specialist experts in high-speed RAM while streaming cold experts.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// Exponential decay factor applied to historical activations per session / request batch.
pub const DEFAULT_HEAT_DECAY_FACTOR: f32 = 0.95;
/// Default file path for persisted expert heat tracking.
pub const DEFAULT_EXPERT_HEAT_FILE: &str = ".mivi/expert_heat.json";

/// Key identifying a specific expert in an MoE model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExpertKey {
    pub layer: usize,
    pub expert_id: usize,
}

/// Statistics and dynamic heat metrics for an individual expert.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertHeatStat {
    pub total_activations: u64,
    pub ema_heat: f32,
    pub last_accessed_timestamp: u64,
}

impl Default for ExpertHeatStat {
    fn default() -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self {
            total_activations: 0,
            ema_heat: 0.0,
            last_accessed_timestamp: now,
        }
    }
}

/// Workload-Adaptive Expert Heat Tracker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertHeatTracker {
    pub model_id: String,
    pub total_inference_tokens: u64,
    pub experts: HashMap<String, ExpertHeatStat>,
}

impl ExpertHeatTracker {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            total_inference_tokens: 0,
            experts: HashMap::new(),
        }
    }

    #[inline]
    fn make_key(layer: usize, expert_id: usize) -> String {
        format!("L{}_E{}", layer, expert_id)
    }

    #[inline]
    fn parse_key(key: &str) -> Option<ExpertKey> {
        let parts: Vec<&str> = key.split('_').collect();
        if parts.len() != 2 || !parts[0].starts_with('L') || !parts[1].starts_with('E') {
            return None;
        }
        let layer = parts[0][1..].parse::<usize>().ok()?;
        let expert_id = parts[1][1..].parse::<usize>().ok()?;
        Some(ExpertKey { layer, expert_id })
    }

    /// Record an activation of an expert in a specific layer.
    pub fn record_activation(&mut self, layer: usize, expert_id: usize) {
        let key = Self::make_key(layer, expert_id);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let entry = self.experts.entry(key).or_insert_with(ExpertHeatStat::default);
        entry.total_activations += 1;
        entry.ema_heat += 1.0;
        entry.last_accessed_timestamp = now;
        self.total_inference_tokens += 1;
    }

    /// Apply exponential decay to all expert heat values (called periodically or per request).
    pub fn decay_heat(&mut self, decay_factor: f32) {
        for stat in self.experts.values_mut() {
            stat.ema_heat *= decay_factor;
        }
    }

    /// Return the top-K hottest experts ranked by EMA heat score.
    pub fn get_hottest_experts(&self, top_k: usize) -> Vec<(ExpertKey, f32)> {
        let mut ranked: Vec<(ExpertKey, f32)> = self
            .experts
            .iter()
            .filter_map(|(k, stat)| {
                let key = Self::parse_key(k)?;
                Some((key, stat.ema_heat))
            })
            .collect();

        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(top_k);
        ranked
    }

    /// Check if a specific expert is considered "hot" (within top percentile threshold).
    pub fn is_expert_hot(&self, layer: usize, expert_id: usize, top_k: usize) -> bool {
        let hottest = self.get_hottest_experts(top_k);
        let target = ExpertKey { layer, expert_id };
        hottest.iter().any(|(k, _)| *k == target)
    }

    /// Save expert heat tracking profile to a JSON file.
    pub fn save_to_file(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let serialized = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, serialized)?;
        Ok(())
    }

    /// Load expert heat tracking profile from a JSON file.
    pub fn load_from_file(path: &Path) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let tracker: Self = serde_json::from_slice(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        Ok(tracker)
    }
}

/// Policy for pinning hot experts in high-speed RAM.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExpertPinningStrategy {
    /// Pin top N hottest experts across the entire model.
    TopGlobal(usize),
    /// Pin top N hottest experts per layer.
    TopPerLayer(usize),
    /// Disable pinning (stream all on demand).
    Disabled,
}

/// Expert Pinning Manager for dynamic RAM residency management.
#[derive(Debug, Clone)]
pub struct ExpertPinningManager {
    pub strategy: ExpertPinningStrategy,
    pub heat_tracker: ExpertHeatTracker,
}

impl ExpertPinningManager {
    pub fn new(model_id: impl Into<String>, strategy: ExpertPinningStrategy) -> Self {
        Self {
            strategy,
            heat_tracker: ExpertHeatTracker::new(model_id),
        }
    }

    /// Check if an expert should be pinned in high-speed RAM.
    pub fn should_pin_expert(&self, layer: usize, expert_id: usize) -> bool {
        match self.strategy {
            ExpertPinningStrategy::Disabled => false,
            ExpertPinningStrategy::TopGlobal(k) => {
                self.heat_tracker.is_expert_hot(layer, expert_id, k)
            }
            ExpertPinningStrategy::TopPerLayer(k) => {
                let hottest = self.heat_tracker.get_hottest_experts(1024);
                let layer_experts: Vec<_> = hottest
                    .into_iter()
                    .filter(|(key, _)| key.layer == layer)
                    .take(k)
                    .collect();
                let target = ExpertKey { layer, expert_id };
                layer_experts.iter().any(|(key, _)| *key == target)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expert_heat_tracker_record_and_ranking() {
        let mut tracker = ExpertHeatTracker::new("mivi");

        // Simulate activations
        for _ in 0..10 {
            tracker.record_activation(0, 3); // L0 E3: 10
        }
        for _ in 0..5 {
            tracker.record_activation(0, 1); // L0 E1: 5
        }
        for _ in 0..20 {
            tracker.record_activation(1, 2); // L1 E2: 20
        }

        let hottest = tracker.get_hottest_experts(2);
        assert_eq!(hottest.len(), 2);
        assert_eq!(hottest[0].0, ExpertKey { layer: 1, expert_id: 2 });
        assert_eq!(hottest[1].0, ExpertKey { layer: 0, expert_id: 3 });

        assert!(tracker.is_expert_hot(1, 2, 2));
        assert!(tracker.is_expert_hot(0, 3, 2));
        assert!(!tracker.is_expert_hot(0, 1, 2));
    }

    #[test]
    fn test_expert_heat_decay_and_persistence() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_expert_heat.json");

        let mut tracker = ExpertHeatTracker::new("mivi");
        tracker.record_activation(0, 0);
        tracker.record_activation(0, 0);

        tracker.decay_heat(0.5);
        let hottest = tracker.get_hottest_experts(1);
        assert!((hottest[0].1 - 1.0).abs() < 1e-4);

        tracker.save_to_file(&path).unwrap();
        let loaded = ExpertHeatTracker::load_from_file(&path).unwrap();
        assert_eq!(loaded.model_id, "mivi");
        assert_eq!(loaded.total_inference_tokens, 2);

        let _ = std::fs::remove_file(path);
    }
}
