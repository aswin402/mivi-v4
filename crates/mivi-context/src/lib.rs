//! Context management and RLM VM operations.

pub mod store;
pub mod vm;

pub use store::{ContextBlock, ContextStore};
pub use vm::{ContextOp, ContextVm};
