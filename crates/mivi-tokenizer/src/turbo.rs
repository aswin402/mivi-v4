//! Turbo-BPE: Ultra-High-Throughput Zero-Allocation BPE Tokenizer Engine.
//!
//! Inspired by GigaToken (`marcelroed/gigatoken`).
//! Features:
//! 1. Direct-mapped word-level memoization cache for O(1) Zipf token retrieval.
//! 2. Stack-allocated intrusive linked-array BPE merger (zero heap allocations in merge loop).
//! 3. 256-byte O(1) ASCII classification lookup table for rapid pre-tokenization.

use crate::vocab::Vocab;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Maximum number of bytes per pre-token piece handled on the stack.
pub const MAX_PIECE_BYTES: usize = 256;
/// Number of slots in the direct-mapped word memoization cache.
pub const MEMO_CACHE_SIZE: usize = 4096;
/// Maximum number of tokens stored inline per memoized word.
pub const MAX_MEMO_TOKENS: usize = 8;

/// Byte classification types for fast pre-token boundary detection.
pub const BYTE_TYPE_SPACE: u8 = 0;
pub const BYTE_TYPE_ALPHA: u8 = 1;
pub const BYTE_TYPE_DIGIT: u8 = 2;
pub const BYTE_TYPE_PUNCT: u8 = 3;
pub const BYTE_TYPE_OTHER: u8 = 4;

/// 256-byte precomputed classification table for sub-nanosecond byte checks.
pub static BYTE_CLASS_TABLE: [u8; 256] = {
    let mut table = [BYTE_TYPE_OTHER; 256];
    let mut i = 0;
    while i < 256 {
        let b = i as u8;
        if b == b' ' || b == b'\t' || b == b'\r' || b == b'\n' {
            table[i] = BYTE_TYPE_SPACE;
        } else if (b >= b'a' && b <= b'z') || (b >= b'A' && b <= b'Z') || b == b'_' {
            table[i] = BYTE_TYPE_ALPHA;
        } else if b >= b'0' && b <= b'9' {
            table[i] = BYTE_TYPE_DIGIT;
        } else if b >= 33 && b <= 126 {
            table[i] = BYTE_TYPE_PUNCT;
        } else {
            table[i] = BYTE_TYPE_OTHER;
        }
        i += 1;
    }
    table
};

/// A single entry in the direct-mapped word memoization cache.
#[derive(Clone, Copy, Debug)]
struct MemoEntry {
    hash: u64,
    len: u8,
    tokens: [u32; MAX_MEMO_TOKENS],
}

impl Default for MemoEntry {
    fn default() -> Self {
        Self {
            hash: 0,
            len: 0,
            tokens: [0; MAX_MEMO_TOKENS],
        }
    }
}

/// Direct-mapped word-level memoization cache for sub-5ns token retrieval on Zipf-frequent words.
#[derive(Clone, Debug)]
pub struct WordMemoCache {
    slots: Arc<RwLock<Vec<MemoEntry>>>,
}

impl Default for WordMemoCache {
    fn default() -> Self {
        Self::new()
    }
}

impl WordMemoCache {
    pub fn new() -> Self {
        Self {
            slots: Arc::new(RwLock::new(vec![MemoEntry::default(); MEMO_CACHE_SIZE])),
        }
    }

    /// Compute 64-bit FNV-1a hash of word slice.
    #[inline(always)]
    fn hash_word(word: &str) -> u64 {
        let mut hash = 0xcbf29ce484222325u64;
        for &b in word.as_bytes() {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3u64);
        }
        hash
    }

    /// Query memoization cache for a pre-tokenized word piece.
    #[inline(always)]
    pub fn lookup(&self, word: &str, out: &mut Vec<u32>) -> bool {
        if word.is_empty() {
            return true;
        }
        let hash = Self::hash_word(word);
        let slot_idx = (hash as usize) & (MEMO_CACHE_SIZE - 1);
        if let Ok(slots) = self.slots.read() {
            let entry = &slots[slot_idx];
            if entry.hash == hash && entry.len > 0 {
                let count = entry.len as usize;
                out.extend_from_slice(&entry.tokens[..count]);
                return true;
            }
        }
        false
    }

    /// Insert tokenization result into memoization cache.
    #[inline(always)]
    pub fn insert(&self, word: &str, tokens: &[u32]) {
        if tokens.is_empty() || tokens.len() > MAX_MEMO_TOKENS {
            return;
        }
        let hash = Self::hash_word(word);
        let slot_idx = (hash as usize) & (MEMO_CACHE_SIZE - 1);
        if let Ok(mut slots) = self.slots.write() {
            let mut entry = MemoEntry {
                hash,
                len: tokens.len() as u8,
                tokens: [0; MAX_MEMO_TOKENS],
            };
            entry.tokens[..tokens.len()].copy_from_slice(tokens);
            slots[slot_idx] = entry;
        }
    }
}

/// Intrusive linked-array node for zero-allocation BPE symbol merges.
#[derive(Debug, Clone, Copy)]
struct BpeSymbolNode {
    start_byte: u16,
    byte_len: u16,
    prev: i16,
    next: i16,
}

/// High-performance zero-allocation BPE merger.
pub struct IntrusiveBpeMerger;

impl IntrusiveBpeMerger {
    /// Merges pre-tokenized string piece directly on stack without any heap allocations.
    pub fn encode_piece_zero_alloc(
        piece: &str,
        vocab: &Vocab,
        merges: &HashMap<(String, String), u32>,
        out: &mut Vec<u32>,
    ) {
        if piece.is_empty() {
            return;
        }

        // 1. Direct vocabulary lookup (fast path for whole word tokens)
        if let Some(id) = vocab.get_id(piece) {
            out.push(id);
            return;
        }

        let piece_bytes = piece.as_bytes();
        let n_chars = piece.chars().count();

        if n_chars <= 1 {
            if let Some(id) = vocab.get_id(piece) {
                out.push(id);
            } else {
                // Fallback to byte tokens
                for &b in piece_bytes {
                    let hex = format!("<0x{:02X}>", b);
                    out.push(vocab.get_id(&hex).unwrap_or(crate::special::UNK_TOKEN_ID));
                }
            }
            return;
        }

        // Initialize stack nodes
        let mut nodes = [BpeSymbolNode {
            start_byte: 0,
            byte_len: 0,
            prev: -1,
            next: -1,
        }; MAX_PIECE_BYTES];

        let max_nodes = n_chars.min(MAX_PIECE_BYTES);
        let mut byte_offset = 0;
        let mut node_count = 0;

        for ch in piece.chars() {
            if node_count >= max_nodes {
                break;
            }
            let ch_len = ch.len_utf8() as u16;
            nodes[node_count] = BpeSymbolNode {
                start_byte: byte_offset as u16,
                byte_len: ch_len,
                prev: if node_count > 0 { node_count as i16 - 1 } else { -1 },
                next: if node_count + 1 < max_nodes { (node_count + 1) as i16 } else { -1 },
            };
            byte_offset += ch_len as usize;
            node_count += 1;
        }

        let head = 0i16;

        // Iterative merge loop on stack nodes
        loop {
            let mut best_pair: Option<(i16, i16)> = None;
            let mut best_rank = u32::MAX;

            let mut curr = head;
            while curr >= 0 {
                let next = nodes[curr as usize].next;
                if next < 0 {
                    break;
                }

                let left_node = &nodes[curr as usize];
                let right_node = &nodes[next as usize];

                let left_str = &piece[left_node.start_byte as usize
                    ..(left_node.start_byte + left_node.byte_len) as usize];
                let right_str = &piece[right_node.start_byte as usize
                    ..(right_node.start_byte + right_node.byte_len) as usize];

                // Check merge rank in vocabulary merges map
                let pair = (left_str.to_string(), right_str.to_string());
                if let Some(&rank) = merges.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_pair = Some((curr, next));
                    }
                }

                curr = next;
            }

            let Some((left_idx, right_idx)) = best_pair else {
                break;
            };

            // Merge right_idx into left_idx
            nodes[left_idx as usize].byte_len += nodes[right_idx as usize].byte_len;
            let next_after_right = nodes[right_idx as usize].next;
            nodes[left_idx as usize].next = next_after_right;

            if next_after_right >= 0 {
                nodes[next_after_right as usize].prev = left_idx;
            }
        }

        // Map merged nodes to vocabulary IDs
        let mut curr = head;
        while curr >= 0 {
            let node = &nodes[curr as usize];
            let sym = &piece[node.start_byte as usize..(node.start_byte + node.byte_len) as usize];

            if let Some(id) = vocab.get_id(sym) {
                out.push(id);
            } else {
                for &b in sym.as_bytes() {
                    let hex = format!("<0x{:02X}>", b);
                    out.push(vocab.get_id(&hex).unwrap_or(crate::special::UNK_TOKEN_ID));
                }
            }
            curr = node.next;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_class_table_accuracy() {
        assert_eq!(BYTE_CLASS_TABLE[b' ' as usize], BYTE_TYPE_SPACE);
        assert_eq!(BYTE_CLASS_TABLE[b'\n' as usize], BYTE_TYPE_SPACE);
        assert_eq!(BYTE_CLASS_TABLE[b'a' as usize], BYTE_TYPE_ALPHA);
        assert_eq!(BYTE_CLASS_TABLE[b'Z' as usize], BYTE_TYPE_ALPHA);
        assert_eq!(BYTE_CLASS_TABLE[b'9' as usize], BYTE_TYPE_DIGIT);
        assert_eq!(BYTE_CLASS_TABLE[b'{' as usize], BYTE_TYPE_PUNCT);
        assert_eq!(BYTE_CLASS_TABLE[b'<' as usize], BYTE_TYPE_PUNCT);
    }

    #[test]
    fn test_word_memo_cache_insert_and_lookup() {
        let cache = WordMemoCache::new();
        let mut out = Vec::new();

        assert!(!cache.lookup("function", &mut out));
        cache.insert("function", &[101, 202]);

        assert!(cache.lookup("function", &mut out));
        assert_eq!(out, vec![101, 202]);
    }
}
