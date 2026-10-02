//! Chunk-based prefix caching for prompt reuse and TTFT optimization.
//!
//! Inspired by LMCache, this module breaks token streams into fixed-size chunks (e.g. 64 tokens)
//! and stores shared Attention KV intervals plus per-boundary recurrent SSM states.
//! When a new request shares a prefix (such as a system prompt, tool definitions, or multi-turn history),
//! the engine copies the matched KV intervals back and skips their prefill computation.
//! Restore work is linear in matched KV bytes, not constant time.

use std::collections::{HashMap, VecDeque};

/// Default token chunk size for hierarchical prefix caching (64 tokens).
pub const PREFIX_CHUNK_SIZE: usize = 64;

/// Default maximum number of cached chunks in memory (1024 chunks = 65,536 tokens / 64k).
pub const DEFAULT_MAX_CACHED_CHUNKS: usize = 1024;
/// Default maximum RAM budget for PrefixCache (512 MB).
pub const DEFAULT_MAX_PREFIX_CACHE_BYTES: usize = 512 * 1024 * 1024;

/// Snapshot of the complete hybrid inference state at a given sequence position.
#[derive(Debug, Clone, PartialEq)]
pub struct HybridStateSnapshot {
    /// Token position reached at this snapshot boundary.
    pub pos: usize,
    /// Standalone hash of the chunk tokens (prev_hash=0).
    /// Used by suffix matching to locate this chunk without chaining context.
    pub standalone_hash: u64,
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
        standalone_hash: u64,
        k_cache: Vec<f32>,
        v_cache: Vec<f32>,
        ssm_conv_states: Vec<f32>,
        ssm_hidden_states: Vec<f32>,
    ) -> Self {
        Self {
            pos,
            standalone_hash,
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
    /// Recurrent state at the boundary and owned KV interval. For delta entries,
    /// KV data covers `[kv_start, state.pos)`, not the whole prefix. Use
    /// `restore_longest_prefix` to restore an entry and its shared ancestors.
    pub state: HybridStateSnapshot,
    /// Beginning of the KV interval owned by this node (zero for full snapshots).
    kv_start: usize,
    /// Shared ancestor holding the preceding interval, for delta nodes.
    parent_hash: Option<u64>,
}

/// Compact checkpoint returned after restoring shared KV prefix blocks.
pub struct RestoredPrefix {
    pub matched_tokens: usize,
    pub hash: u64,
    pub ssm_conv_states: Vec<f32>,
    pub ssm_hidden_states: Vec<f32>,
}

impl PrefixChunk {
    fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.tokens.capacity() * std::mem::size_of::<u32>()
            + self.state.memory_bytes()
    }
}

/// In-memory LRU prefix cache manager for hybrid SLM states.
#[derive(Debug)]
pub struct PrefixCache {
    chunk_size: usize,
    max_chunks: usize,
    chunks: HashMap<u64, PrefixChunk>,
    lru_order: VecDeque<u64>,
    total_memory_bytes: usize,
    max_memory_bytes: usize,
}

impl Default for PrefixCache {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_CACHED_CHUNKS, PREFIX_CHUNK_SIZE)
    }
}

impl PrefixCache {
    /// Create a new prefix cache with specified capacity and chunk size.
    pub fn new(max_chunks: usize, chunk_size: usize) -> Self {
        Self::with_budget(max_chunks, chunk_size, DEFAULT_MAX_PREFIX_CACHE_BYTES)
    }

    /// Create a prefix cache with an explicit maximum memory byte budget.
    pub fn with_budget(max_chunks: usize, chunk_size: usize, max_memory_bytes: usize) -> Self {
        Self {
            chunk_size: if chunk_size > 0 {
                chunk_size
            } else {
                PREFIX_CHUNK_SIZE
            },
            max_chunks: if max_chunks > 0 {
                max_chunks
            } else {
                DEFAULT_MAX_CACHED_CHUNKS
            },
            chunks: HashMap::with_capacity(max_chunks.min(32)),
            lru_order: VecDeque::with_capacity(max_chunks.min(32)),
            total_memory_bytes: 0,
            max_memory_bytes: if max_memory_bytes > 0 {
                max_memory_bytes
            } else {
                DEFAULT_MAX_PREFIX_CACHE_BYTES
            },
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
        self.total_memory_bytes = 0;
    }

    /// Find the longest matching cached prefix for a given sequence of tokens.
    ///
    /// Returns `Some((matched_tokens_count, &PrefixChunk))` representing the deepest
    /// matched chunk boundary, or `None` if no prefix chunk was found.
    pub fn find_longest_prefix<'a>(
        &'a mut self,
        tokens: &[u32],
    ) -> Option<(usize, &'a PrefixChunk)> {
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
            self.chunks
                .get(&match_hash)
                .map(|chunk| (matched_len, chunk))
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
        self.insert_node(prev_hash, tokens, chunk_index, state, 0, None)
    }

    /// Store one KV interval and its recurrent checkpoint. Ancestors are shared
    /// by chained hash rather than copied into every descendant snapshot.
    pub fn insert_delta_chunk(
        &mut self,
        prev_hash: u64,
        tokens: &[u32],
        chunk_index: usize,
        state: HybridStateSnapshot,
    ) -> crate::Result<u64> {
        let start =
            chunk_index
                .checked_mul(self.chunk_size)
                .ok_or(crate::KvError::ContextOverflow {
                    pos: usize::MAX,
                    max: state.pos,
                })?;
        let end = start
            .checked_add(self.chunk_size)
            .ok_or(crate::KvError::ContextOverflow {
                pos: usize::MAX,
                max: state.pos,
            })?;
        if tokens.len() != self.chunk_size || state.pos != end {
            return Err(crate::KvError::DimMismatch {
                expected: end,
                got: state.pos,
            });
        }
        let parent = if start == 0 {
            if prev_hash != 0 {
                return Err(crate::KvError::DimMismatch {
                    expected: 0,
                    got: start,
                });
            }
            None
        } else {
            let ancestor = self
                .chunks
                .get(&prev_hash)
                .ok_or(crate::KvError::DimMismatch {
                    expected: start,
                    got: 0,
                })?;
            if ancestor.state.pos != start || ancestor.chunk_index != chunk_index - 1 {
                return Err(crate::KvError::DimMismatch {
                    expected: start,
                    got: ancestor.state.pos,
                });
            }
            Some(prev_hash)
        };
        Ok(self.insert_node(prev_hash, tokens, chunk_index, state, start, parent))
    }

    /// Restore KV blocks in causal order and return only the final SSM checkpoint.
    pub fn restore_longest_prefix(
        &mut self,
        tokens: &[u32],
        kv: &mut crate::KvCache,
    ) -> crate::Result<Option<RestoredPrefix>> {
        let Some((matched_tokens, chunk)) = self.find_longest_prefix(tokens) else {
            return Ok(None);
        };
        let hash = chunk.hash;
        if chunk.state.pos != matched_tokens {
            return Err(crate::KvError::DimMismatch {
                expected: matched_tokens,
                got: chunk.state.pos,
            });
        }
        let mut path = Vec::new();
        let mut current = Some(hash);
        while let Some(key) = current {
            let node = self.chunks.get(&key).ok_or(crate::KvError::DimMismatch {
                expected: matched_tokens,
                got: 0,
            })?;
            path.push(key);
            current = node.parent_hash;
        }
        for key in path.iter().rev() {
            let node = &self.chunks[key];
            kv.import_state_range(
                node.kv_start,
                node.state.pos - node.kv_start,
                &node.state.k_cache,
                &node.state.v_cache,
            )?;
        }
        let node = &self.chunks[&hash];
        Ok(Some(RestoredPrefix {
            matched_tokens,
            hash,
            ssm_conv_states: node.state.ssm_conv_states.clone(),
            ssm_hidden_states: node.state.ssm_hidden_states.clone(),
        }))
    }

    fn insert_node(
        &mut self,
        prev_hash: u64,
        tokens: &[u32],
        chunk_index: usize,
        state: HybridStateSnapshot,
        kv_start: usize,
        parent_hash: Option<u64>,
    ) -> u64 {
        let hash = compute_chunk_hash(prev_hash, tokens);

        if self.chunks.contains_key(&hash) {
            self.touch(&hash);
            return hash;
        }

        let chunk = PrefixChunk {
            chunk_index,
            hash,
            tokens: tokens.to_vec(),
            state,
            kv_start,
            parent_hash,
        };
        let chunk_mem = chunk.memory_bytes();

        self.chunks.insert(hash, chunk);
        self.lru_order.push_back(hash);
        self.total_memory_bytes = self.total_memory_bytes.saturating_add(chunk_mem);
        while self.chunks.len() > self.max_chunks {
            if let Some(eviction_hash) = self.select_eviction_hash() {
                self.remove_from_lru(&eviction_hash);
                if let Some(evicted_chunk) = self.chunks.remove(&eviction_hash) {
                    self.total_memory_bytes = self
                        .total_memory_bytes
                        .saturating_sub(evicted_chunk.memory_bytes());
                }
            } else {
                break;
            }
        }
        self.prune_to_bytes(self.max_memory_bytes);
        hash
    }

    /// Calculate total estimated memory consumption of all cached hybrid state snapshots in bytes (O(1)).
    #[inline]
    pub fn memory_usage_bytes(&self) -> usize {
        self.total_memory_bytes
    }

    /// Dynamically prune oldest LRU chunks until total memory is below `target_bytes` (O(K)).
    /// Returns the number of chunks evicted.
    pub fn prune_to_bytes(&mut self, target_bytes: usize) -> usize {
        let mut evicted = 0;
        while self.total_memory_bytes > target_bytes && !self.lru_order.is_empty() {
            if let Some(eviction_hash) = self.select_eviction_hash() {
                self.remove_from_lru(&eviction_hash);
                if let Some(removed) = self.chunks.remove(&eviction_hash) {
                    self.total_memory_bytes = self
                        .total_memory_bytes
                        .saturating_sub(removed.memory_bytes());
                    evicted += 1;
                }
            }
        }
        evicted
    }

    fn select_eviction_hash(&self) -> Option<u64> {
        let mut selected = None;
        let mut selected_index = 0usize;
        for hash in &self.lru_order {
            if let Some(chunk) = self.chunks.get(hash) {
                if selected.is_none() || chunk.chunk_index > selected_index {
                    selected = Some(*hash);
                    selected_index = chunk.chunk_index;
                }
            }
        }
        selected.or_else(|| self.lru_order.front().copied())
    }

    fn remove_from_lru(&mut self, hash: &u64) {
        if let Some(pos) = self.lru_order.iter().position(|entry| entry == hash) {
            self.lru_order.remove(pos);
        }
    }

    /// Move a chunk hash to the back of the LRU queue.
    fn touch(&mut self, hash: &u64) {
        if let Some(pos) = self.lru_order.iter().position(|entry| entry == hash) {
            self.lru_order.remove(pos);
            self.lru_order.push_back(*hash);
        }
    }
    /// Search cached chunks for a suffix match at `start_pos`.
    ///
    /// Returns `None` if no cached state matches the suffix.
    /// This identifies matching token bytes only; callers must separately verify
    /// that restoring the snapshot is valid for the complete preceding context.
    /// Returns `Some((suffix_len, match_pos, chunk))` where:
    ///   - `suffix_len`: number of tokens to skip (matched suffix length)
    ///   - `match_pos`: absolute position where the cached KV was originally computed
    ///   - `chunk`: the matched PrefixChunk with KV data
    pub fn find_longest_suffix_match(
        &mut self,
        start_pos: usize,
        tokens: &[u32],
    ) -> Option<(usize, usize, PrefixChunk)> {
        if tokens.len() < self.chunk_size || self.chunks.is_empty() {
            return None;
        }

        let n = tokens.len();
        let abs_offset = start_pos;
        // Use saturating_sub to prevent underflow when start_pos > tokens.len()
        // (can happen in continuation paths where start_pos is the absolute model position)
        let max_k = n.saturating_sub(abs_offset) / self.chunk_size;
        if max_k == 0 {
            return None;
        }

        for k in (1..=max_k).rev() {
            let suffix_len = k * self.chunk_size;
            let suffix_start = n.saturating_sub(suffix_len);
            if suffix_start < abs_offset {
                continue;
            }

            let suffix_abs_pos = abs_offset + suffix_start;
            if !suffix_abs_pos.is_multiple_of(self.chunk_size) {
                continue;
            }

            let suffix_tokens = &tokens[suffix_start..n];
            let suffix_hash = compute_chunk_hash(0, suffix_tokens);

            let match_hash = self
                .chunks
                .iter()
                .find(|(_, chunk)| {
                    if chunk.tokens.len() != suffix_len {
                        return false;
                    }
                    let standalone_hash = chunk.state.standalone_hash;
                    standalone_hash == suffix_hash
                        || (standalone_hash == 0
                            && compute_chunk_hash(0, &chunk.tokens) == suffix_hash)
                })
                .map(|(hash, _)| *hash);

            if let Some(match_hash) = match_hash {
                if let Some(chunk) = self.chunks.get(&match_hash) {
                    let state_pos = chunk.state.pos;
                    let matched = chunk.clone();
                    self.touch(&match_hash);
                    return Some((suffix_len, state_pos, matched));
                }
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_prefix_retains_nine_thousand_token_context_with_linear_storage() {
        let mut source = crate::KvCache::new(2, 9000, 16);
        for pos in 0..9000 {
            for layer in 0..2 {
                source
                    .store(layer, pos, &[pos as f32; 16], &[layer as f32; 16])
                    .unwrap();
            }
        }
        let budget = 4 * 1024 * 1024;
        let mut cache = PrefixCache::with_budget(256, 64, budget);
        let tokens: Vec<u32> = (0..9000).collect();
        let mut hash = 0;
        for index in 0..9000 / 64 {
            let (k, v) = source.export_state_range(index * 64, 64).unwrap();
            let state =
                HybridStateSnapshot::new((index + 1) * 64, 0, k, v, vec![index as f32; 12], vec![]);
            hash = cache
                .insert_delta_chunk(hash, &tokens[index * 64..(index + 1) * 64], index, state)
                .unwrap();
        }
        assert_eq!(cache.len(), 140);
        assert!(cache.memory_usage_bytes() < budget);
        let mut restored = crate::KvCache::new(2, 9000, 16);
        let matched = cache
            .restore_longest_prefix(&tokens, &mut restored)
            .unwrap()
            .unwrap();
        assert_eq!(matched.matched_tokens, 8960);
        assert_eq!(
            restored.export_state(8960).unwrap(),
            source.export_state(8960).unwrap()
        );
    }

    #[test]
    fn delta_branches_share_root_under_byte_budget() {
        let mut cache = PrefixCache::new(10, 64);
        let state = |pos, value| {
            HybridStateSnapshot::new(
                pos,
                0,
                vec![value; 64],
                vec![value + 1.0; 64],
                vec![value],
                vec![],
            )
        };
        let root = cache
            .insert_delta_chunk(0, &[1; 64], 0, state(64, 1.0))
            .unwrap();
        let first = cache
            .insert_delta_chunk(root, &[2; 64], 1, state(128, 2.0))
            .unwrap();
        let second = cache
            .insert_delta_chunk(root, &[3; 64], 1, state(128, 3.0))
            .unwrap();
        assert_eq!(cache.len(), 3);
        let budget = cache.chunks[&root].memory_bytes() + cache.chunks[&second].memory_bytes();
        assert_eq!(cache.prune_to_bytes(budget), 1);
        assert!(cache.memory_usage_bytes() <= budget);
        assert!(!cache.chunks.contains_key(&first));
        let mut kv = crate::KvCache::new(1, 128, 1);
        let tokens: Vec<u32> = [vec![1; 64], vec![3; 64]].concat();
        let matched = cache
            .restore_longest_prefix(&tokens, &mut kv)
            .unwrap()
            .unwrap();
        assert_eq!(matched.matched_tokens, 128);
        assert_eq!(matched.ssm_conv_states, vec![3.0]);
        assert_eq!(
            kv.export_state(128).unwrap().0,
            [vec![1.0; 64], vec![3.0; 64]].concat()
        );
    }

    #[test]
    fn delta_prefix_blocks_share_ancestors_and_restore_after_eviction() {
        let mut source = crate::KvCache::new(2, 256, 4);
        for pos in 0..256 {
            for layer in 0..2 {
                source
                    .store(layer, pos, &[pos as f32; 4], &[layer as f32 + 1.0; 4])
                    .unwrap();
            }
        }
        let tokens: Vec<u32> = (0..256).collect();
        let mut cache = PrefixCache::new(3, 64);
        let mut hash = 0;
        for index in 0..4 {
            let (k, v) = source.export_state_range(index * 64, 64).unwrap();
            let state =
                HybridStateSnapshot::new((index + 1) * 64, 0, k, v, vec![index as f32], vec![]);
            hash = cache
                .insert_delta_chunk(hash, &tokens[index * 64..(index + 1) * 64], index, state)
                .unwrap();
        }
        assert_eq!(cache.len(), 3);
        // Each of the three retained nodes owns exactly one KV block.
        assert_eq!(
            cache
                .chunks
                .values()
                .map(|node| node.state.k_cache.len())
                .sum::<usize>(),
            3 * 2 * 64 * 4
        );
        let mut restored = crate::KvCache::new(2, 256, 4);
        let matched = cache
            .restore_longest_prefix(&tokens, &mut restored)
            .unwrap()
            .unwrap();
        assert_eq!(matched.matched_tokens, 192);
        assert_eq!(matched.ssm_conv_states, vec![2.0]);
        assert_eq!(restored.current_pos(), 192);
        assert_eq!(
            restored.export_state(192).unwrap(),
            source.export_state(192).unwrap()
        );

        let mut changed = tokens.clone();
        changed[0] = 999;
        assert!(cache
            .restore_longest_prefix(&changed, &mut restored)
            .unwrap()
            .is_none());
    }

    #[test]
    fn delta_prefix_rejects_missing_ancestor() {
        let mut cache = PrefixCache::new(3, 64);
        let state = HybridStateSnapshot::new(128, 0, vec![1.0; 64], vec![2.0; 64], vec![], vec![]);
        assert!(cache.insert_delta_chunk(123, &[1; 64], 1, state).is_err());
        assert!(cache.is_empty());
    }

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
            0,
            vec![1.0; 64 * 8],
            vec![2.0; 64 * 8],
            vec![0.5; 16],
            vec![],
        );

        let state_1 = HybridStateSnapshot::new(
            128,
            0,
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

        let state = HybridStateSnapshot::new(64, 0, vec![], vec![], vec![], vec![]);

        let h0 = cache.insert_chunk(0, &c0, 0, state.clone());
        let _h1 = cache.insert_chunk(0, &c1, 0, state.clone());
        assert_eq!(cache.len(), 2);

        // Inserting third chunk must evict h0 (oldest)
        let _h2 = cache.insert_chunk(0, &c2, 0, state);
        assert_eq!(cache.len(), 2);
        assert!(!cache.chunks.contains_key(&h0));
    }

    #[test]
    fn tight_prefix_cache_preserves_reusable_root_chain() {
        let mut cache = PrefixCache::new(2, 64);
        let c0: Vec<u32> = (0..64).collect();
        let c1: Vec<u32> = (64..128).collect();
        let c2: Vec<u32> = (128..192).collect();

        let state0 = HybridStateSnapshot::new(64, 0, vec![], vec![], vec![], vec![]);
        let state1 = HybridStateSnapshot::new(128, 0, vec![], vec![], vec![], vec![]);
        let state2 = HybridStateSnapshot::new(192, 0, vec![], vec![], vec![], vec![]);

        let h0 = cache.insert_chunk(0, &c0, 0, state0);
        let h1 = cache.insert_chunk(h0, &c1, 1, state1);
        let _h2 = cache.insert_chunk(h1, &c2, 2, state2);

        let mut prompt = Vec::new();
        prompt.extend_from_slice(&c0);
        prompt.extend_from_slice(&c1);
        prompt.extend_from_slice(&[999, 1000, 1001]);

        let (matched_len, chunk) = cache
            .find_longest_prefix(&prompt)
            .expect("root prefix chain should remain reusable");
        assert_eq!(matched_len, 128);
        assert_eq!(chunk.hash, h1);
    }
    #[test]
    fn test_suffix_match_finds_cached_chunk() {
        let mut cache = PrefixCache::new(2, 64);
        let chunk_tokens: Vec<u32> = (0..64).collect();
        let state = HybridStateSnapshot::new(64, 0, vec![], vec![], vec![], vec![]);
        let hash = cache.insert_chunk(0, &chunk_tokens, 0, state.clone());
        assert_eq!(cache.len(), 1);

        // Step 2 prompt: history (64 tokens) + same chunk suffix (64 tokens)
        let history: Vec<u32> = (100..164).collect();
        let step2: Vec<u32> = history.iter().chain(chunk_tokens.iter()).cloned().collect();

        let result = cache.find_longest_suffix_match(64, &step2);
        assert!(result.is_some(), "Should find suffix match");
        let (skip, pos, matched) = result.unwrap();
        assert_eq!(skip, 64);
        assert_eq!(pos, 64);
        assert_eq!(matched.hash, hash);
    }

    #[test]
    fn test_suffix_match_no_match_on_different_content() {
        let mut cache = PrefixCache::new(2, 64);
        let chunk_tokens: Vec<u32> = (0..64).collect();
        let state = HybridStateSnapshot::new(64, 0, vec![], vec![], vec![], vec![]);
        cache.insert_chunk(0, &chunk_tokens, 0, state);

        let different: Vec<u32> = (200..264).collect();
        let history: Vec<u32> = (100..164).collect();
        let step2: Vec<u32> = history.iter().chain(different.iter()).cloned().collect();

        let result = cache.find_longest_suffix_match(64, &step2);
        assert!(result.is_none(), "No match when content differs");
    }

    #[test]
    fn test_suffix_match_returns_correct_chunk() {
        let mut cache = PrefixCache::new(5, 64);
        let chunk0: Vec<u32> = (0..64).collect();
        let chunk1: Vec<u32> = (1000..1064).collect();
        let state0 = HybridStateSnapshot::new(64, 0, vec![], vec![], vec![], vec![]);
        let state1 = HybridStateSnapshot::new(128, 0, vec![], vec![], vec![], vec![]);
        let _h0 = cache.insert_chunk(0, &chunk0, 0, state0);
        let hash2 = cache.insert_chunk(0, &chunk1, 0, state1);

        let history: Vec<u32> = (500..564).collect();
        let step2: Vec<u32> = history.iter().chain(chunk1.iter()).cloned().collect();

        let result = cache.find_longest_suffix_match(64, &step2);
        assert!(result.is_some());
        let (_, _, matched) = result.unwrap();
        assert_eq!(matched.hash, hash2);
    }

    #[test]
    fn test_suffix_match_finds_chunk_stored_under_chained_hash() {
        let mut cache = PrefixCache::new(5, 64);
        let chunk0: Vec<u32> = (0..64).collect();
        let chunk1: Vec<u32> = (1000..1064).collect();
        let state0 = HybridStateSnapshot::new(
            64,
            compute_chunk_hash(0, &chunk0),
            vec![],
            vec![],
            vec![],
            vec![],
        );
        let state1 = HybridStateSnapshot::new(
            128,
            compute_chunk_hash(0, &chunk1),
            vec![],
            vec![],
            vec![],
            vec![],
        );
        let h0 = cache.insert_chunk(0, &chunk0, 0, state0);
        let h1 = cache.insert_chunk(h0, &chunk1, 1, state1);
        assert_ne!(h1, compute_chunk_hash(0, &chunk1));

        let history: Vec<u32> = (500..564).collect();
        let step2: Vec<u32> = history.iter().chain(chunk1.iter()).cloned().collect();

        let result = cache.find_longest_suffix_match(64, &step2);
        assert!(result.is_some());
        let (_, _, matched) = result.unwrap();
        assert_eq!(matched.hash, h1);
    }

    #[test]
    fn test_memory_usage_includes_cached_tokens() {
        let mut cache = PrefixCache::new(2, 64);
        let tokens: Vec<u32> = (0..64).collect();
        let state = HybridStateSnapshot::new(64, 0, vec![], vec![], vec![], vec![]);
        let state_bytes = state.memory_bytes();
        let token_bytes = tokens.capacity() * std::mem::size_of::<u32>();

        cache.insert_chunk(0, &tokens, 0, state);

        assert!(cache.memory_usage_bytes() >= state_bytes + token_bytes);
    }
}
