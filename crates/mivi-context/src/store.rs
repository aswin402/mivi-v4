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

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ContextStore {
    pub blocks: Vec<ContextBlock>,
}

impl ContextStore {
    pub fn new() -> Self {
        Self { blocks: Vec::new() }
    }

    pub fn add_block(&mut self, id: &str, source: &str, content: &str, pinned: bool) {
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
}
