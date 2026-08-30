//! Subcommand runners for CLI operations.

pub mod bench;
pub mod chat;
pub mod doctor;
pub mod info;
pub mod serve;

pub use bench::run_bench;
pub use chat::run_chat;
pub use doctor::run_doctor;
pub use info::run_info;
pub use serve::run_serve;
