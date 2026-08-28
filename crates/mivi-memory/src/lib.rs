//! Memory subsystem for mivi-v4.

pub mod record;
pub mod store;

pub use record::MemoryRecord;
pub use store::{MemoryError, MemoryStore, Result};
