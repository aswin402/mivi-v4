//! Cache management commands for disk-persisted hybrid KV & SSM states.

use anyhow::Result;
use mivi_kv::{clear_cache_dir, list_cached_files};
use std::path::PathBuf;

pub fn run_cache_list(dir: Option<PathBuf>) -> Result<()> {
    let files = list_cached_files(dir.as_deref())?;
    println!("  ⚡ Mivi Persistent Prefix Cache (.kvc)");
    println!("  ─────────────────────────────────────────────────────────────────");
    if files.is_empty() {
        println!("  No cached prefix files found.");
    } else {
        println!("  {:<32} {:<12} {:<12} {:<16}", "FILE", "TOKENS", "SIZE", "MODEL HASH");
        println!("  ─────────────────────────────────────────────────────────────────");
        let mut total_size = 0u64;
        let mut total_tokens = 0usize;
        for f in &files {
            let size_kb = (f.size_bytes as f64) / 1024.0;
            println!(
                "  {:<32} {:<12} {:<10.1} KB 0x{:016x}",
                f.filename, f.token_count, size_kb, f.model_hash
            );
            total_size += f.size_bytes;
            total_tokens += f.token_count;
        }
        println!("  ─────────────────────────────────────────────────────────────────");
        println!(
            "  Total: {} files, {} tokens, {:.1} KB",
            files.len(),
            total_tokens,
            (total_size as f64) / 1024.0
        );
    }
    Ok(())
}

pub fn run_cache_clear(dir: Option<PathBuf>) -> Result<()> {
    let removed = clear_cache_dir(dir.as_deref())?;
    println!("  🗑️  Cleared {} persistent cache file(s).", removed);
    Ok(())
}
