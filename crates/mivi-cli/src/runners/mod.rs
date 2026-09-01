//! Subcommand runners for CLI operations.

pub mod bench;
pub mod cache;
pub mod chat;
pub mod chat_stream;
pub mod doctor;
pub mod info;
pub mod serve;

pub use bench::run_bench;
pub use cache::{run_cache_clear, run_cache_list};
pub use chat::{run_chat, ChatArgs};
pub use doctor::run_doctor;
pub use info::run_info;
pub use serve::{run_serve, ServeArgs};
