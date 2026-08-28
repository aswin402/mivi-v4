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
                format!("Found {} relevant blocks for '{}'", results.len(), query)
            }
            ContextOp::Slice { source, start, end } => {
                format!("Sliced region [{}..{}] from source {}", start, end, source)
            }
            ContextOp::Summarize { source } => {
                format!("Summary of {}", source)
            }
            ContextOp::Recurse { task, context_ids } => {
                format!("Recursive dispatch of subtask '{}' on {} blocks", task, context_ids.len())
            }
        }
    }
}
