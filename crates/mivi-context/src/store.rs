use mivi_core::TurboQuant4Bit;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextBlock {
    pub id: String,
    pub source: String,
    pub content: String,
    pub importance: f32,
    pub pinned: bool,
    #[serde(default)]
    pub embedding_norm: f32,
    #[serde(default)]
    pub embedding_4bit: Vec<u8>,
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
        self.add_block_with_embedding(id, source, content, pinned, None, None);
    }

    pub fn add_block_with_embedding(
        &mut self,
        id: &str,
        source: &str,
        content: &str,
        pinned: bool,
        embedding: Option<&[f32]>,
        quantizer: Option<&TurboQuant4Bit>,
    ) {
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

        let (embedding_norm, embedding_4bit) = match (embedding, quantizer) {
            (Some(emb), Some(q)) if emb.len() == q.dim() => q.quantize(emb),
            _ => (0.0, Vec::new()),
        };

        self.blocks.push(ContextBlock {
            id: id.to_string(),
            source: source.to_string(),
            content: content.to_string(),
            importance: 1.0,
            pinned,
            embedding_norm,
            embedding_4bit,
        });
    }

    pub fn search(&self, query: &str) -> Vec<&ContextBlock> {
        let q = query.to_lowercase();
        self.blocks
            .iter()
            .filter(|b| b.content.to_lowercase().contains(&q))
            .collect()
    }

    /// Perform rapid 4-bit TurboQuant semantic similarity search across indexed context blocks.
    pub fn search_semantic(
        &self,
        query_embedding: &[f32],
        quantizer: &TurboQuant4Bit,
        top_k: usize,
    ) -> Vec<(&ContextBlock, f32)> {
        if self.blocks.is_empty() || top_k == 0 || query_embedding.len() != quantizer.dim() {
            return Vec::new();
        }

        let query_lut = quantizer.build_query_lut(query_embedding);
        let query_norm_sq: f32 = query_embedding.iter().map(|x| x * x).sum();
        let query_norm = query_norm_sq.sqrt().max(1e-8);

        let mut scored: Vec<(&ContextBlock, f32)> = self
            .blocks
            .iter()
            .filter(|b| !b.embedding_4bit.is_empty() && b.embedding_norm > 0.0)
            .map(|block| {
                let raw_dot = quantizer.score_query_lut(
                    &query_lut,
                    block.embedding_norm,
                    &block.embedding_4bit,
                );
                let cosine = raw_dot / (query_norm * block.embedding_norm);
                (block, cosine)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
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

    #[test]
    fn test_context_store_semantic_search() {
        let dim = 32;
        let quantizer = TurboQuant4Bit::new(dim);
        let mut store = ContextStore::new();

        let mut emb1 = vec![0.0f32; dim];
        let mut emb2 = vec![0.0f32; dim];
        for i in 0..dim {
            emb1[i] = (i as f32 + 1.0) * 0.1;
            emb2[i] = -(i as f32 + 1.0) * 0.1;
        }

        store.add_block_with_embedding("doc1", "docs/db.rs", "Database connection pool", false, Some(&emb1), Some(&quantizer));
        store.add_block_with_embedding("doc2", "docs/ui.rs", "Frontend UI rendering", false, Some(&emb2), Some(&quantizer));

        let results = store.search_semantic(&emb1, &quantizer, 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.id, "doc1");
        assert!(results[0].1 > 0.8, "Cosine similarity must be high for matching embedding");
    }
}
