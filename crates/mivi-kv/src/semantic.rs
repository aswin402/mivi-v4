//! Semantic Anchor Checkpointing for agentic workflows (FreeToken-inspired).
//!
//! Snapshots hybrid Attention KV and SSM states at natural structural boundaries
//! (<think>, </think>, <tool_call>, </tool_call>, <tool_response>), allowing non-contiguous
//! context edits (like trimming reasoning traces or updating tool outputs) without full re-prefill.

use crate::prefix::HybridStateSnapshot;

/// Types of semantic structural boundaries in agent conversations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SemanticAnchorType {
    TurnSystem,
    TurnUser,
    TurnAssistant,
    ThinkingStart,
    ThinkingEnd,
    ToolCallStart,
    ToolCallEnd,
    ToolResponseStart,
    ToolResponseEnd,
}

/// A state checkpoint associated with a semantic anchor boundary.
#[derive(Debug, Clone)]
pub struct SemanticAnchor {
    pub anchor_type: SemanticAnchorType,
    pub token_pos: usize,
    pub token_prefix: Vec<u32>,
    pub state: HybridStateSnapshot,
}

/// In-memory cache of semantic anchor checkpoints.
#[derive(Debug, Clone)]
pub struct SemanticAnchorCache {
    pub anchors: Vec<SemanticAnchor>,
    pub max_anchors: usize,
}

pub const DEFAULT_MAX_SEMANTIC_ANCHORS: usize = 32;

impl Default for SemanticAnchorCache {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_SEMANTIC_ANCHORS)
    }
}

impl SemanticAnchorCache {
    /// Create a new SemanticAnchorCache with a maximum anchor budget.
    pub fn new(max_anchors: usize) -> Self {
        Self {
            anchors: Vec::with_capacity(max_anchors),
            max_anchors: max_anchors.max(4),
        }
    }

    /// Insert or update a semantic anchor checkpoint.
    pub fn insert_anchor(
        &mut self,
        anchor_type: SemanticAnchorType,
        token_pos: usize,
        token_prefix: &[u32],
        state: HybridStateSnapshot,
    ) {
        // If an anchor for the exact same token position exists, update it
        if let Some(existing) = self.anchors.iter_mut().find(|a| a.token_pos == token_pos) {
            existing.anchor_type = anchor_type;
            existing.token_prefix = token_prefix.to_vec();
            existing.state = state;
            return;
        }

        // If capacity exceeded, remove oldest anchor
        if self.anchors.len() >= self.max_anchors {
            self.anchors.remove(0);
        }

        self.anchors.push(SemanticAnchor {
            anchor_type,
            token_pos,
            token_prefix: token_prefix.to_vec(),
            state,
        });
    }

    /// Find the deepest matching semantic anchor that matches the beginning of `tokens`.
    pub fn find_deepest_anchor(&self, tokens: &[u32]) -> Option<(usize, &SemanticAnchor)> {
        let mut best: Option<(usize, &SemanticAnchor)> = None;

        for anchor in &self.anchors {
            if anchor.token_pos <= tokens.len()
                && tokens[..anchor.token_pos] == anchor.token_prefix[..]
            {
                if let Some((best_pos, _)) = best {
                    if anchor.token_pos > best_pos {
                        best = Some((anchor.token_pos, anchor));
                    }
                } else {
                    best = Some((anchor.token_pos, anchor));
                }
            }
        }

        best
    }

    /// Clear all cached semantic anchors.
    pub fn clear(&mut self) {
        self.anchors.clear();
    }

    /// Number of cached anchors.
    pub fn len(&self) -> usize {
        self.anchors.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.anchors.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_anchor_cache_insert_and_find() {
        let mut cache = SemanticAnchorCache::new(8);
        let snapshot1 = HybridStateSnapshot::new(10, vec![], vec![], vec![], vec![]);
        let snapshot2 = HybridStateSnapshot::new(25, vec![], vec![], vec![], vec![]);

        cache.insert_anchor(
            SemanticAnchorType::ThinkingStart,
            10,
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            snapshot1,
        );

        let prefix2: Vec<u32> = (1..=25).collect();
        cache.insert_anchor(
            SemanticAnchorType::ToolCallStart,
            25,
            &prefix2,
            snapshot2,
        );

        // Query with a 30-token sequence starting with prefix2
        let query_tokens: Vec<u32> = (1..=30).collect();
        let (matched_pos, matched_anchor) = cache.find_deepest_anchor(&query_tokens).unwrap();

        assert_eq!(matched_pos, 25);
        assert_eq!(matched_anchor.anchor_type, SemanticAnchorType::ToolCallStart);
    }
}
