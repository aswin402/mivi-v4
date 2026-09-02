//! Context store supporting paging, slicing, pinned blocks, and importance ranking.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextBlock {
    pub id: String,
    pub source: String,
    pub content: String,
    pub importance: f32,
    pub pinned: bool,
}

pub const DEFAULT_MAX_CONTEXT_BLOCKS: usize = 256;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextStore {
    pub blocks: Vec<ContextBlock>,
    pub max_blocks: usize,
}

impl Default for ContextStore {
    fn default() -> Self {
        Self::new()
    }
}

impl ContextStore {
    pub fn new() -> Self {
        Self {
            blocks: Vec::new(),
            max_blocks: DEFAULT_MAX_CONTEXT_BLOCKS,
        }
    }

    pub fn with_capacity(max_blocks: usize) -> Self {
        Self {
            blocks: Vec::new(),
            max_blocks,
        }
    }

    pub fn add_block(&mut self, id: &str, source: &str, content: &str, pinned: bool) {
        if self.blocks.len() >= self.max_blocks {
            if let Some(pos) = self.blocks.iter().position(|b| !b.pinned) {
                self.blocks.remove(pos);
            } else if !pinned {
                // If all blocks are pinned, reject unpinned insertions rather than evicting pinned blocks
                return;
            } else if !self.blocks.is_empty() {
                self.blocks.remove(0);
            }
        }
        self.blocks.push(ContextBlock {
            id: id.to_string(),
            source: source.to_string(),
            content: content.to_string(),
            importance: 1.0,
            pinned,
        });
    }

    pub fn search(&self, query: &str) -> Vec<&ContextBlock> {
        let q = query.to_lowercase();
        self.blocks
            .iter()
            .filter(|b| b.content.to_lowercase().contains(&q))
            .collect()
    }

    pub fn find_block(&self, source_or_id: &str) -> Option<&ContextBlock> {
        self.blocks
            .iter()
            .find(|b| b.source == source_or_id || b.id == source_or_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_store_eviction() {
        let mut store = ContextStore::with_capacity(2);
        store.add_block("1", "src1", "content 1", true); // pinned
        store.add_block("2", "src2", "content 2", false); // unpinned
        store.add_block("3", "src3", "content 3", false); // unpinned -> should evict 2

        assert_eq!(store.blocks.len(), 2);
        assert!(store.find_block("1").is_some());
        assert!(store.find_block("2").is_none());
        assert!(store.find_block("3").is_some());
    }
}
