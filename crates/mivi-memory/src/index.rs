//! Ultra-compact TurboQuant 4-bit semantic memory vector index.
//!
//! Stores 4-bit quantized embeddings with fast asymmetric SIMD cosine similarity search
//! and zero codebook training overhead.

use mivi_core::TurboQuant4Bit;
use serde::{Deserialize, Serialize};
use std::path::Path;
use uuid::Uuid;
use crate::store::{MemoryError, Result};

/// A single 4-bit quantized vector memory entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboMemoryEntry {
    pub id: Uuid,
    pub norm: f32,
    pub packed_4bit: Vec<u8>,
}

/// Inverted semantic memory index using 4-bit TurboQuant vector compression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboMemoryIndex {
    dim: usize,
    entries: Vec<TurboMemoryEntry>,
    #[serde(skip)]
    quantizer: Option<TurboQuant4Bit>,
}

impl TurboMemoryIndex {
    /// Create a new TurboMemoryIndex for embeddings of dimension `dim`.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            entries: Vec::new(),
            quantizer: Some(TurboQuant4Bit::new(dim)),
        }
    }

    #[inline]
    fn get_or_init_quantizer(&mut self) -> &TurboQuant4Bit {
        if self.quantizer.is_none() {
            self.quantizer = Some(TurboQuant4Bit::new(self.dim));
        }
        self.quantizer.as_ref().unwrap()
    }

    /// Number of indexed vector entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the index is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Dimension of the indexed vectors.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Total memory footprint in bytes occupied by packed 4-bit vector entries.
    #[inline]
    pub fn memory_bytes(&self) -> usize {
        let entry_size = std::mem::size_of::<TurboMemoryEntry>();
        let vector_bytes: usize = self.entries.iter().map(|e| e.packed_4bit.len()).sum();
        self.entries.len() * entry_size + vector_bytes
    }

    /// Add or update a vector embedding for a memory record ID.
    pub fn add_record(&mut self, id: Uuid, embedding: &[f32]) -> Result<()> {
        if embedding.len() != self.dim {
            return Err(MemoryError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Embedding dimension mismatch: expected {}, got {}", self.dim, embedding.len()),
            )));
        }

        let quantizer = self.get_or_init_quantizer().clone();
        let (norm, packed_4bit) = quantizer.quantize(embedding);

        // Remove existing entry if ID matches
        self.entries.retain(|e| e.id != id);

        self.entries.push(TurboMemoryEntry {
            id,
            norm,
            packed_4bit,
        });

        Ok(())
    }

    /// Remove a memory record from the index by UUID.
    pub fn remove_record(&mut self, id: &Uuid) -> bool {
        let before = self.entries.len();
        self.entries.retain(|e| &e.id != id);
        self.entries.len() < before
    }

    /// Perform rapid asymmetric SIMD cosine similarity search.
    ///
    /// Returns the top `k` matching record UUIDs with their approximate cosine similarity scores.
    pub fn search(&mut self, query_embedding: &[f32], top_k: usize) -> Result<Vec<(Uuid, f32)>> {
        if query_embedding.len() != self.dim {
            return Err(MemoryError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Query embedding dimension mismatch: expected {}, got {}", self.dim, query_embedding.len()),
            )));
        }

        if self.entries.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }

        let quantizer = self.get_or_init_quantizer().clone();
        let query_lut = quantizer.build_query_lut(query_embedding);

        // Calculate query norm for cosine normalization
        let query_norm_sq: f32 = query_embedding.iter().map(|x| x * x).sum();
        let query_norm = query_norm_sq.sqrt().max(1e-8);

        let mut scored: Vec<(Uuid, f32)> = self
            .entries
            .iter()
            .map(|entry| {
                let raw_dot = quantizer.score_query_lut(&query_lut, entry.norm, &entry.packed_4bit);
                let cosine_sim = if entry.norm > 0.0 {
                    raw_dot / (query_norm * entry.norm)
                } else {
                    0.0
                };
                (entry.id, cosine_sim)
            })
            .collect();

        // Sort descending by similarity score
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        Ok(scored)
    }

    /// Save the 4-bit quantized index to a binary/JSON file.
    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let serialized = serde_json::to_vec(self)
            .map_err(|e| MemoryError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
        std::fs::write(path, serialized)?;
        Ok(())
    }

    /// Load a 4-bit quantized index from a file.
    pub fn load_from_file(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let mut index: Self = serde_json::from_slice(&bytes)
            .map_err(|e| MemoryError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
        index.quantizer = Some(TurboQuant4Bit::new(index.dim));
        Ok(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_turbo_memory_index_add_search_and_persistence() {
        let dim = 32;
        let mut index = TurboMemoryIndex::new(dim);

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        let mut emb1 = vec![0.0f32; dim];
        let mut emb2 = vec![0.0f32; dim];

        // emb1 points mostly in positive direction, emb2 in negative
        for i in 0..dim {
            emb1[i] = (i as f32 + 1.0) * 0.1;
            emb2[i] = -(i as f32 + 1.0) * 0.1;
        }

        index.add_record(id1, &emb1).unwrap();
        index.add_record(id2, &emb2).unwrap();

        assert_eq!(index.len(), 2);
        assert!(index.memory_bytes() > 0);

        // Search with query matching emb1
        let results = index.search(&emb1, 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, id1);
        assert!(results[0].1 > 0.8, "Cosine similarity for emb1 must be high");
        assert!(results[1].1 < 0.0, "Cosine similarity for emb2 must be negative");

        // Test persistence
        let temp_file = std::env::temp_dir().join(format!("turbo_mem_{}.json", Uuid::new_v4()));
        index.save_to_file(&temp_file).unwrap();

        let mut loaded = TurboMemoryIndex::load_from_file(&temp_file).unwrap();
        assert_eq!(loaded.len(), 2);
        let loaded_results = loaded.search(&emb1, 1).unwrap();
        assert_eq!(loaded_results[0].0, id1);

        let _ = std::fs::remove_file(&temp_file);
    }
}
