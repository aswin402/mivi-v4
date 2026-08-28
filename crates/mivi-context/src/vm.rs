//! RLM Context VM supporting typed functional operations over external context.

use crate::store::ContextStore;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContextOp {
    Search { query: String },
    Slice { source: String, start: usize, end: usize },
    Summarize { source: String },
    Recurse { task: String, context_ids: Vec<String> },
}

pub struct ContextVm<'a> {
    pub store: &'a mut ContextStore,
}

impl<'a> ContextVm<'a> {
    pub fn new(store: &'a mut ContextStore) -> Self {
        Self { store }
    }

    pub fn execute(&mut self, op: ContextOp) -> String {
        match op {
            ContextOp::Search { query } => {
                let results = self.store.search(&query);
                if results.is_empty() {
                    format!("No matching context blocks found for query '{}'", query)
                } else {
                    let mut out = format!("Found {} relevant context blocks:\n", results.len());
                    for b in results.iter().take(5) {
                        let snippet: String = b.content.chars().take(120).collect();
                        out.push_str(&format!("- [{}] (source: {}): {}...\n", b.id, b.source, snippet));
                    }
                    out
                }
            }
            ContextOp::Slice { source, start, end } => {
                let matching = self
                    .store
                    .blocks
                    .iter()
                    .find(|b| b.source == source || b.id == source);

                if let Some(b) = matching {
                    let mut indices = b.content.char_indices().map(|(idx, _)| idx);
                    let byte_start = if start == 0 {
                        0
                    } else {
                        indices.nth(start - 1).unwrap_or(b.content.len())
                    };
                    let byte_end = if end <= start {
                        byte_start
                    } else {
                        b.content
                            .char_indices()
                            .nth(end)
                            .map(|(idx, _)| idx)
                            .unwrap_or(b.content.len())
                    };
                    let slice = &b.content[byte_start..byte_end];
                    format!("Slice [{}..{}] from '{}':\n{}", start, end, source, slice)
                } else {
                    format!("Context source '{}' not found in store.", source)
                }
            }
            ContextOp::Summarize { source } => {
                let matching = self
                    .store
                    .blocks
                    .iter()
                    .find(|b| b.source == source || b.id == source);

                if let Some(b) = matching {
                    let preview: String = b.content.lines().take(3).collect::<Vec<_>>().join(" ");
                    format!("Summary of '{}' ({} chars): {}", source, b.content.len(), preview)
                } else {
                    format!("Context source '{}' not found in store.", source)
                }
            }
            ContextOp::Recurse { task, context_ids } => {
                let mut combined_len = 0;
                for cid in &context_ids {
                    if let Some(b) = self.store.blocks.iter().find(|b| &b.id == cid) {
                        combined_len += b.content.len();
                    }
                }
                format!(
                    "Recursive subtask '{}' scheduled on {} blocks (total {} bytes).",
                    task,
                    context_ids.len(),
                    combined_len
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_vm_operations() {
        let mut store = ContextStore::new();
        store.add_block(
            "doc1",
            "rust_intro.md",
            "Rust is a systems programming language focused on safety and speed.",
            false,
        );

        let mut vm = ContextVm::new(&mut store);
        let search_res = vm.execute(ContextOp::Search { query: "safety".to_string() });
        assert!(search_res.contains("Found 1"));

        let slice_res = vm.execute(ContextOp::Slice {
            source: "rust_intro.md".to_string(),
            start: 0,
            end: 4,
        });
        assert!(slice_res.contains("Rust"));
    }
}
