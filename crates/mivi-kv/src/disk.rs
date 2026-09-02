//! Persistent on-disk storage and retrieval for prefilled hybrid KV & SSM states.
//!
//! Allows long system prompts, documentation, codebases, and tool schemas to be
//! persisted to `.mivi/cache/<hash>.kvc` files and reloaded instantly across CLI/Server restarts
//! without recomputing expensive prompt forward steps.

use crate::prefix::HybridStateSnapshot;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

/// Magic header identifier for Mivi KV Cache files: "MIVIKVC1"
pub const KVC_MAGIC: &[u8; 8] = b"MIVIKVC1";

/// Default directory for persisting KV cache files.
pub const DEFAULT_CACHE_DIR: &str = ".mivi/cache";

/// Information about a cached `.kvc` file on disk.
#[derive(Debug, Clone)]
pub struct CacheFileInfo {
    pub path: PathBuf,
    pub filename: String,
    pub size_bytes: u64,
    pub token_count: usize,
    pub model_hash: u64,
}

/// Returns the default cache directory path (`.mivi/cache`).
pub fn default_cache_dir() -> PathBuf {
    PathBuf::from(DEFAULT_CACHE_DIR)
}

/// Ensures the cache directory exists, creating parent folders if needed.
pub fn ensure_cache_dir(dir: Option<&Path>) -> io::Result<PathBuf> {
    let path = dir.map(PathBuf::from).unwrap_or_else(default_cache_dir);
    if !path.exists() {
        fs::create_dir_all(&path)?;
    }
    Ok(path)
}

/// Saves a hybrid state snapshot and its corresponding token sequence to a `.kvc` file.
pub fn save_to_disk(
    file_path: &Path,
    tokens: &[u32],
    state: &HybridStateSnapshot,
    model_hash: u64,
) -> io::Result<()> {
    if let Some(parent) = file_path.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)?;
        }
    }

    let file = File::create(file_path)?;
    let mut writer = BufWriter::new(file);

    // 1. Magic header
    writer.write_all(KVC_MAGIC)?;

    // 2. Metadata header
    writer.write_u64::<LittleEndian>(model_hash)?;
    writer.write_u32::<LittleEndian>(tokens.len() as u32)?;
    writer.write_u32::<LittleEndian>(state.pos as u32)?;
    writer.write_u32::<LittleEndian>(state.k_cache.len() as u32)?;
    writer.write_u32::<LittleEndian>(state.v_cache.len() as u32)?;
    writer.write_u32::<LittleEndian>(state.ssm_conv_states.len() as u32)?;
    writer.write_u32::<LittleEndian>(state.ssm_hidden_states.len() as u32)?;

    // 3. Tokens payload
    for &tok in tokens {
        writer.write_u32::<LittleEndian>(tok)?;
    }

    // 4. K-Cache floats
    for &val in &state.k_cache {
        writer.write_f32::<LittleEndian>(val)?;
    }

    // 5. V-Cache floats
    for &val in &state.v_cache {
        writer.write_f32::<LittleEndian>(val)?;
    }

    // 6. SSM Conv States floats
    for &val in &state.ssm_conv_states {
        writer.write_f32::<LittleEndian>(val)?;
    }

    // 7. SSM Hidden States floats
    for &val in &state.ssm_hidden_states {
        writer.write_f32::<LittleEndian>(val)?;
    }

    writer.flush()?;
    Ok(())
}

/// Loads a hybrid state snapshot and token sequence from a `.kvc` file on disk.
pub fn load_from_disk(
    file_path: &Path,
    expected_model_hash: Option<u64>,
) -> io::Result<(Vec<u32>, HybridStateSnapshot)> {
    let file = File::open(file_path)?;
    let mut reader = BufReader::new(file);

    // 1. Validate magic header
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != KVC_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid KV cache file: magic mismatch",
        ));
    }

    // 2. Read metadata
    let model_hash = reader.read_u64::<LittleEndian>()?;
    if let Some(expected) = expected_model_hash {
        if model_hash != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Model hash mismatch: expected {expected}, found {model_hash}"),
            ));
        }
    }

    let n_tokens = reader.read_u32::<LittleEndian>()? as usize;
    let pos = reader.read_u32::<LittleEndian>()? as usize;
    let k_len = reader.read_u32::<LittleEndian>()? as usize;
    let v_len = reader.read_u32::<LittleEndian>()? as usize;
    let conv_len = reader.read_u32::<LittleEndian>()? as usize;
    let ssm_len = reader.read_u32::<LittleEndian>()? as usize;

    let file_meta = file_path.metadata()?;
    let file_len = file_meta.len() as usize;
    const HEADER_BYTES: usize = 40;
    let expected_payload_bytes = (n_tokens * 4) + (k_len + v_len + conv_len + ssm_len) * 4;
    if HEADER_BYTES + expected_payload_bytes > file_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Cache header payload ({} bytes total) exceeds file size ({} bytes)", HEADER_BYTES + expected_payload_bytes, file_len),
        ));
    }

    // 3. Read tokens
    let mut tokens = Vec::with_capacity(n_tokens);
    for _ in 0..n_tokens {
        tokens.push(reader.read_u32::<LittleEndian>()?);
    }

    // 4. Read K-Cache
    let mut k_cache = Vec::with_capacity(k_len);
    for _ in 0..k_len {
        k_cache.push(reader.read_f32::<LittleEndian>()?);
    }

    // 5. Read V-Cache
    let mut v_cache = Vec::with_capacity(v_len);
    for _ in 0..v_len {
        v_cache.push(reader.read_f32::<LittleEndian>()?);
    }

    // 6. Read SSM Conv States
    let mut ssm_conv_states = Vec::with_capacity(conv_len);
    for _ in 0..conv_len {
        ssm_conv_states.push(reader.read_f32::<LittleEndian>()?);
    }

    // 7. Read SSM Hidden States
    let mut ssm_hidden_states = Vec::with_capacity(ssm_len);
    for _ in 0..ssm_len {
        ssm_hidden_states.push(reader.read_f32::<LittleEndian>()?);
    }

    let snapshot = HybridStateSnapshot::new(pos, k_cache, v_cache, ssm_conv_states, ssm_hidden_states);
    Ok((tokens, snapshot))
}

/// Lists all `.kvc` cache files in the specified directory.
pub fn list_cached_files(dir: Option<&Path>) -> io::Result<Vec<CacheFileInfo>> {
    let cache_dir = dir.map(PathBuf::from).unwrap_or_else(default_cache_dir);
    if !cache_dir.exists() {
        return Ok(Vec::new());
    }

    let mut result = Vec::new();
    for entry in fs::read_dir(cache_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|e| e.to_str()) == Some("kvc") {
            let metadata = entry.metadata()?;
            let size_bytes = metadata.len();
            let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();

            // Try reading header for token count and model hash
            if let Ok(file) = File::open(&path) {
                let mut reader = BufReader::new(file);
                let mut magic = [0u8; 8];
                if reader.read_exact(&mut magic).is_ok() && &magic == KVC_MAGIC {
                    let model_hash = reader.read_u64::<LittleEndian>().unwrap_or(0);
                    let token_count = reader.read_u32::<LittleEndian>().unwrap_or(0) as usize;
                    result.push(CacheFileInfo {
                        path,
                        filename,
                        size_bytes,
                        token_count,
                        model_hash,
                    });
                }
            }
        }
    }

    Ok(result)
}

/// Clears all `.kvc` cache files in the cache directory, returning the number of removed files.
pub fn clear_cache_dir(dir: Option<&Path>) -> io::Result<usize> {
    let cache_dir = dir.map(PathBuf::from).unwrap_or_else(default_cache_dir);
    if !cache_dir.exists() {
        return Ok(0);
    }

    let mut removed = 0;
    for entry in fs::read_dir(cache_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|e| e.to_str()) == Some("kvc") {
            fs::remove_file(path)?;
            removed += 1;
        }
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disk_cache_save_and_load_roundtrip() {
        let tmp_dir = std::env::temp_dir().join(format!("mivi_kvc_test_{}", std::process::id()));
        let file_path = tmp_dir.join("test_prefix.kvc");

        let tokens = vec![1, 100, 200, 300, 400];
        let original_snapshot = HybridStateSnapshot::new(
            5,
            vec![1.1, 2.2, 3.3],
            vec![4.4, 5.5, 6.6],
            vec![0.1, 0.2],
            vec![0.9],
        );
        let model_hash = 0x1234_5678_9ABC_DEF0;

        assert!(save_to_disk(&file_path, &tokens, &original_snapshot, model_hash).is_ok());

        let (loaded_tokens, loaded_snapshot) = load_from_disk(&file_path, Some(model_hash)).unwrap();
        assert_eq!(loaded_tokens, tokens);
        assert_eq!(loaded_snapshot, original_snapshot);

        // Test hash mismatch rejection
        let mismatch_res = load_from_disk(&file_path, Some(0x9999));
        assert!(mismatch_res.is_err());

        // Cleanup
        let _ = fs::remove_dir_all(&tmp_dir);
    }
}
