//! Key-Value Cache, Hierarchical Prefix Caching, and Persistent Disk Storage for mivi-v4.

pub mod cache;
pub mod disk;
pub mod prefix;

pub use cache::{KvCache, KvError, Result};
pub use disk::{
    clear_cache_dir, default_cache_dir, ensure_cache_dir, list_cached_files, load_from_disk,
    save_to_disk, CacheFileInfo, DEFAULT_CACHE_DIR, KVC_MAGIC,
};
pub use prefix::{
    compute_chunk_hash, HybridStateSnapshot, PrefixCache, PrefixChunk, DEFAULT_MAX_CACHED_CHUNKS,
    PREFIX_CHUNK_SIZE,
};
