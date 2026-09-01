//! Chunk-based prefix caching for zero-latency prompt reuse and TTFT optimization.
//!
//! Inspired by LMCache, this module breaks token streams into fixed-size chunks (e.g. 64 tokens)
//! and stores snapshots of the hybrid recurrent inference state (Attention KV slices + SSM Conv states).
//! When a new request shares a prefix (such as a system prompt, tool definitions, or multi-turn history),
//! the engine restores the snapshot in O(1) time and skips prefill computation for all matched tokens.

use std::collections::{HashMap, VecDeque};

/// Default token chunk size for hierarchical prefix caching (64 tokens).
pub const PREFIX_CHUNK_SIZE: usize = 64;

/// Default maximum number of cached chunks in memory (~64 MB budget).
pub const DEFAULT_MAX_CACHED_CHUNKS: usize = 256;

/// Snapshot of the complete hybrid inference state at a given sequence position.
#[derive(Debug, Clone, PartialEq)]
pub struct HybridStateSnapshot {
    /// Token position reached at this snapshot boundary.
    pub pos: usize,
    /// Key cache elements up to `pos` across all allocated attention layers.
    pub k_cache: Vec<f32>,
    /// Value cache elements up to `pos` across all allocated attention layers.
    pub v_cache: Vec<f32>,
    /// SSM 1D ShortConv rolling buffer state across all layers.
    pub ssm_conv_states: Vec<f32>,
    /// SSM recurrent hidden state across all layers (if applicable).
    pub ssm_hidden_states: Vec<f32>,
}

impl HybridStateSnapshot {
    /// Create a new hybrid state snapshot.
    pub fn new(
        pos: usize,
        k_cache: Vec<f32>,
        v_cache: Vec<f32>,
        ssm_conv_states: Vec<f32>,
        ssm_hidden_states: Vec<f32>,
    ) -> Self {
        Self {
            pos,
            k_cache,
            v_cache,
            ssm_conv_states,
            ssm_hidden_states,
        }
    }

    /// Estimate memory consumption of this snapshot in bytes.
    pub fn memory_bytes(&self) -> usize {
        (self.k_cache.len()
            + self.v_cache.len()
            + self.ssm_conv_states.len()
            + self.ssm_hidden_states.len())
            * std::mem::size_of::<f32>()
            + std::mem::size_of::<Self>()
    }
}

/// Computes a fast 64-bit rolling hash for a token chunk chained with the previous chunk's hash.
#[inline]
pub fn compute_chunk_hash(prev_hash: u64, tokens: &[u32]) -> u64 {
    // 64-bit FNV-1a hash with domain separator
    const FNV_PRIME: u64 = 0x00000100_000001B3;
    let mut hash = if prev_hash == 0 {
        0xCBF29CE4_84222325
    } else {
        prev_hash ^ 0x9E3779B9_7F4A7C15
    };

    for &tok in tokens {
        for b in tok.to_le_bytes() {
            hash ^= b as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }

    hash
}

/// Cached prefix chunk entry.
#[derive(Debug, Clone)]
pub struct PrefixChunk {
    /// Chunk index in the hierarchical token sequence (0, 1, 2, ...).
    pub chunk_index: usize,
    /// Chained hash identifier for this prefix path.
    pub hash: u64,
    /// Tokens contained in this chunk.
    pub tokens: Vec<u32>,
    /// Serialized hybrid state at the end of this chunk.
    pub state: HybridStateSnapshot,
}

/// In-memory LRU prefix cache manager for hybrid SLM states.
#[derive(Debug)]
pub struct PrefixCache {
    chunk_size: usize,
    max_chunks: usize,
    chunks: HashMap<u64, PrefixChunk>,
    lru_order: VecDeque<u64>,
}

impl Default for PrefixCache {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_CACHED_CHUNKS, PREFIX_CHUNK_SIZE)
    }
}

impl PrefixCache {
    /// Create a new prefix cache with specified capacity and chunk size.
    pub fn new(max_chunks: usize, chunk_size: usize) -> Self {
        Self {
            chunk_size: if chunk_size > 0 { chunk_size } else { PREFIX_CHUNK_SIZE },
            max_chunks: if max_chunks > 0 { max_chunks } else { DEFAULT_MAX_CACHED_CHUNKS },
            chunks: HashMap::with_capacity(max_chunks),
            lru_order: VecDeque::with_capacity(max_chunks),
        }
    }

    /// Returns the chunk size used by this prefix cache.
    #[inline]
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Returns the number of currently cached chunks.
    #[inline]
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Returns whether the prefix cache is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Clear all cached chunks.
    pub fn clear(&mut self) {
        self.chunks.clear();
        self.lru_order.clear();
    }

    /// Find the longest matching cached prefix for a given sequence of tokens.
    ///
    /// Returns `Some((matched_tokens_count, &PrefixChunk))` representing the deepest
    /// matched chunk boundary, or `None` if no prefix chunk was found.
    pub fn find_longest_prefix<'a>(&'a mut self, tokens: &[u32]) -> Option<(usize, &'a PrefixChunk)> {
        if tokens.len() < self.chunk_size {
            return None;
        }

        let mut current_hash = 0u64;
        let mut deepest_match: Option<(usize, u64)> = None;

        let num_full_chunks = tokens.len() / self.chunk_size;
        for i in 0..num_full_chunks {
            let start = i * self.chunk_size;
            let end = start + self.chunk_size;
            let chunk_tokens = &tokens[start..end];

            current_hash = compute_chunk_hash(current_hash, chunk_tokens);

            if self.chunks.contains_key(&current_hash) {
                deepest_match = Some(((i + 1) * self.chunk_size, current_hash));
            } else {
                // Break on first miss in hierarchical prefix tree
                break;
            }
        }

        if let Some((matched_len, match_hash)) = deepest_match {
            // Touch LRU order
            self.touch(&match_hash);
            self.chunks.get(&match_hash).map(|chunk| (matched_len, chunk))
        } else {
            None
        }
    }

    /// Insert a newly computed chunk and its corresponding hybrid state into the cache.
    pub fn insert_chunk(
        &mut self,
        prev_hash: u64,
        tokens: &[u32],
        chunk_index: usize,
        state: HybridStateSnapshot,
    ) -> u64 {
        let hash = compute_chunk_hash(prev_hash, tokens);

        if self.chunks.contains_key(&hash) {
            self.touch(&hash);
            return hash;
        }

        // Evict LRU chunk if at max capacity
        if self.chunks.len() >= self.max_chunks {
            if let Some(oldest_hash) = self.lru_order.pop_front() {
                self.chunks.remove(&oldest_hash);
            }
        }

        let chunk = PrefixChunk {
            chunk_index,
            hash,
            tokens: tokens.to_vec(),
            state,
        };

        self.chunks.insert(hash, chunk);
        self.lru_order.push_back(hash);
        hash
    }

    /// Calculate total estimated memory consumption of all cached hybrid state snapshots in bytes.
    pub fn memory_usage_bytes(&self) -> usize {
        self.chunks.values().map(|c| c.state.memory_bytes()).sum()
    }

    /// Dynamically prune oldest LRU chunks until total memory is below `target_bytes`.
    /// Returns the number of chunks evicted.
    pub fn prune_to_bytes(&mut self, target_bytes: usize) -> usize {
        let mut evicted = 0;
        while self.memory_usage_bytes() > target_bytes && !self.lru_order.is_empty() {
            if let Some(oldest_hash) = self.lru_order.pop_front() {
                if self.chunks.remove(&oldest_hash).is_some() {
                    evicted += 1;
                }
            }
        }
        evicted
    }

    /// Move a chunk hash to the back of the LRU queue.
    fn touch(&mut self, hash: &u64) {
        if let Some(pos) = self.lru_order.iter().position(|h| h == hash) {
            self.lru_order.remove(pos);
            self.lru_order.push_back(*hash);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_hash_determinism_and_chaining() {
        let chunk_a: Vec<u32> = (0..64).collect();
        let chunk_b: Vec<u32> = (64..128).collect();

        let h0_a = compute_chunk_hash(0, &chunk_a);
        let h0_a_repeat = compute_chunk_hash(0, &chunk_a);
        assert_eq!(h0_a, h0_a_repeat);

        let h1_b = compute_chunk_hash(h0_a, &chunk_b);
        assert_ne!(h0_a, h1_b);

        // Different predecessor yields different chained hash
        let h1_b_different_prev = compute_chunk_hash(42, &chunk_b);
        assert_ne!(h1_b, h1_b_different_prev);
    }

    #[test]
    fn test_prefix_cache_insert_and_lookup() {
        let mut cache = PrefixCache::new(10, 64);
        let chunk_0: Vec<u32> = (0..64).collect();
        let chunk_1: Vec<u32> = (64..128).collect();

        let state_0 = HybridStateSnapshot::new(
            64,
            vec![1.0; 64 * 8],
            vec![2.0; 64 * 8],
            vec![0.5; 16],
            vec![],
        );

        let state_1 = HybridStateSnapshot::new(
            128,
            vec![1.5; 128 * 8],
            vec![2.5; 128 * 8],
            vec![0.7; 16],
            vec![],
        );

        let h0 = cache.insert_chunk(0, &chunk_0, 0, state_0);
        let _h1 = cache.insert_chunk(h0, &chunk_1, 1, state_1);

        assert_eq!(cache.len(), 2);

        // Lookup exact match of 2 full chunks + 10 extra tokens
        let mut full_prompt = Vec::new();
        full_prompt.extend_from_slice(&chunk_0);
        full_prompt.extend_from_slice(&chunk_1);
        full_prompt.extend_from_slice(&[999, 1000, 1001]);

        let lookup = cache.find_longest_prefix(&full_prompt);
        assert!(lookup.is_some());
        let (matched_len, chunk) = lookup.unwrap();
        assert_eq!(matched_len, 128);
        assert_eq!(chunk.chunk_index, 1);
        assert_eq!(chunk.state.pos, 128);

        // Lookup with partial match (only chunk 0 matches)
        let mut partial_prompt = Vec::new();
        partial_prompt.extend_from_slice(&chunk_0);
        partial_prompt.extend_from_slice(&[500; 64]); // completely different chunk 1

        let partial_lookup = cache.find_longest_prefix(&partial_prompt);
        assert!(partial_lookup.is_some());
        let (matched_len_partial, chunk_partial) = partial_lookup.unwrap();
        assert_eq!(matched_len_partial, 64);
        assert_eq!(chunk_partial.chunk_index, 0);
    }

    #[test]
    fn test_prefix_cache_lru_eviction() {
        let mut cache = PrefixCache::new(2, 64);
        let c0: Vec<u32> = vec![1; 64];
        let c1: Vec<u32> = vec![2; 64];
        let c2: Vec<u32> = vec![3; 64];

        let state = HybridStateSnapshot::new(64, vec![], vec![], vec![], vec![]);

        let h0 = cache.insert_chunk(0, &c0, 0, state.clone());
        let _h1 = cache.insert_chunk(0, &c1, 0, state.clone());
        assert_eq!(cache.len(), 2);

        // Inserting third chunk must evict h0 (oldest)
        let _h2 = cache.insert_chunk(0, &c2, 0, state);
        assert_eq!(cache.len(), 2);
        assert!(!cache.chunks.contains_key(&h0));
    }
}
