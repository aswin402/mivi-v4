//! RLM Context VM supporting typed functional operations over external context.

use crate::store::ContextStore;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContextOp {
    Search {
        query: String,
    },
    Slice {
        source: String,
        start: usize,
        end: usize,
    },
    Summarize {
        source: String,
    },
    Recurse {
        task: String,
        context_ids: Vec<String>,
    },
}

pub const MAX_SEARCH_RESULTS_PREVIEW: usize = 5;
pub const SEARCH_SNIPPET_CHAR_LIMIT: usize = 120;
pub const SUMMARY_PREVIEW_LINES: usize = 3;

pub struct ContextVm<'a> {
    pub store: &'a mut ContextStore,
}

impl<'a> ContextVm<'a> {
    pub fn new(store: &'a mut ContextStore) -> Self {
        Self { store }
    }

    pub fn execute(&mut self, op: ContextOp) -> String {
        match op {
            ContextOp::Search { query } => self.op_search(&query),
            ContextOp::Slice { source, start, end } => self.op_slice(&source, start, end),
            ContextOp::Summarize { source } => self.op_summarize(&source),
            ContextOp::Recurse { task, context_ids } => self.op_recurse(&task, &context_ids),
        }
    }

    fn op_search(&self, query: &str) -> String {
        let results = self.store.search(query);
        if results.is_empty() {
            format!("No matching context blocks found for query '{}'", query)
        } else {
            let mut out = format!("Found {} relevant context blocks:\n", results.len());
            for b in results.iter().take(MAX_SEARCH_RESULTS_PREVIEW) {
                let snippet: String = b.content.chars().take(SEARCH_SNIPPET_CHAR_LIMIT).collect();
                out.push_str(&format!(
                    "- [{}] (source: {}): {}...\n",
                    b.id, b.source, snippet
                ));
            }
            out
        }
    }

    fn op_slice(&self, source: &str, start: usize, end: usize) -> String {
        let matching = self.store.find_block(source);

        if let Some(b) = matching {
            let total_chars = b.content.chars().count();
            if start >= total_chars {
                return format!(
                    "Slice error: start index {} out of bounds (total chars: {}) for '{}'",
                    start, total_chars, source
                );
            }
            let byte_start = b
                .content
                .char_indices()
                .nth(start)
                .map(|(idx, _)| idx)
                .unwrap_or(b.content.len());
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

    fn op_summarize(&self, source: &str) -> String {
        let matching = self.store.find_block(source);

        if let Some(b) = matching {
            let total_chars = b.content.chars().count();
            let preview_lines: Vec<&str> = b.content.lines().take(SUMMARY_PREVIEW_LINES).collect();
            let snippet = preview_lines.join("\n");
            format!(
                "Summary of '{}' ({} chars):\n{}...",
                source, total_chars, snippet
            )
        } else {
            format!("Context source '{}' not found in store.", source)
        }
    }

    fn op_recurse(&self, task: &str, context_ids: &[String]) -> String {
        let mut combined_len = 0;
        for cid in context_ids {
            if let Some(b) = self.store.find_block(cid) {
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
        let search_res = vm.execute(ContextOp::Search {
            query: "safety".to_string(),
        });
        assert!(search_res.contains("Found 1"));

        let slice_res = vm.execute(ContextOp::Slice {
            source: "rust_intro.md".to_string(),
            start: 0,
            end: 4,
        });
        assert!(slice_res.contains("Rust"));

        let slice_offset = vm.execute(ContextOp::Slice {
            source: "rust_intro.md".to_string(),
            start: 5,
            end: 7,
        });
        assert!(slice_offset.contains("is"));

        let oob_slice = vm.execute(ContextOp::Slice {
            source: "rust_intro.md".to_string(),
            start: 9999,
            end: 10000,
        });
        assert!(oob_slice.contains("out of bounds"));

        let summarize_res = vm.execute(ContextOp::Summarize {
            source: "rust_intro.md".to_string(),
        });
        assert!(summarize_res.contains("Summary of 'rust_intro.md'"));
    }
}
