//! Byte-Pair Encoding (BPE) implementation with standard merge loop.

use crate::vocab::Vocab;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("Vocabulary is missing token for byte: {0}")]
    MissingByteToken(u8),
    #[error("GGUF metadata missing tokenizer information")]
    MissingTokenizerMetadata,
    #[error("Failed to parse regex pattern: {0}")]
    RegexError(#[from] regex::Error),
}

pub type Result<T> = std::result::Result<T, TokenizerError>;

static PRE_TOKENIZE_REGEX: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    // Standard Rust-compatible GPT-2 / GPT-4 / LLaMA pre-tokenization regex pattern
    regex::Regex::new(
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+",
    )
    .expect("Invalid pre-tokenizer regex pattern")
});

#[derive(Debug, Clone)]
pub struct Tokenizer {
    vocab: Vocab,
    merges: HashMap<(String, String), u32>,
    byte_ranks: HashMap<Vec<u8>, usize>,
    merge_ranks: HashMap<Vec<u8>, usize>,
}

impl Tokenizer {
    pub fn new(vocab: Vocab, merges: HashMap<(String, String), u32>) -> Self {
        let mut byte_ranks = HashMap::new();
        for (i, token) in vocab.id_to_token.iter().enumerate() {
            byte_ranks.insert(token.as_bytes().to_vec(), i);
        }

        let mut merge_ranks: HashMap<Vec<u8>, usize> = HashMap::new();
        if merges.is_empty() {
            for (i, token) in vocab.id_to_token.iter().enumerate() {
                merge_ranks.insert(token.as_bytes().to_vec(), i);
            }
        } else {
            for ((left, right), rank) in &merges {
                let mut key = left.as_bytes().to_vec();
                key.extend_from_slice(right.as_bytes());
                merge_ranks.insert(key, *rank as usize);
            }
        }

        Self {
            vocab,
            merges,
            byte_ranks,
            merge_ranks,
        }
    }

    pub fn vocab(&self) -> &Vocab {
        &self.vocab
    }

    pub fn merges(&self) -> &HashMap<(String, String), u32> {
        &self.merges
    }

    /// Encode input text into a sequence of BPE token IDs using standard regex pre-tokenization.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        let mut tokens = Vec::new();

        // Split text with pre-tokenization regex
        for m in PRE_TOKENIZE_REGEX.find_iter(text) {
            let piece = m.as_str().as_bytes();
            let piece_tokens = self.bpe_encode_piece(piece);
            tokens.extend(piece_tokens);
        }

        tokens
    }

    /// Tiktoken / minbpe-style BPE merge loop for a single contiguous byte slice.
    fn bpe_encode_piece(&self, piece: &[u8]) -> Vec<u32> {
        if piece.is_empty() {
            return Vec::new();
        }

        if piece.len() == 1 {
            if let Some(&rank) = self.byte_ranks.get(piece) {
                return vec![rank as u32];
            }
            if let Some(id) = self.vocab.get_id(&format!("<0x{:02X}>", piece[0])) {
                return vec![id];
            }
            return vec![0];
        }

        // parts stores: (byte_start_index, rank_of_pair_starting_here)
        let mut parts: Vec<(usize, usize)> = Vec::with_capacity(piece.len() + 1);
        let mut min_rank = (usize::MAX, usize::MAX);

        for i in 0..piece.len() - 1 {
            let rank = self.merge_ranks.get(&piece[i..i + 2]).copied().unwrap_or(usize::MAX);
            if rank < min_rank.0 {
                min_rank = (rank, i);
            }
            parts.push((i, rank));
        }
        parts.push((piece.len() - 1, usize::MAX));
        parts.push((piece.len(), usize::MAX)); // Sentinel

        let get_rank = |parts: &[(usize, usize)], idx: usize| -> usize {
            if idx + 2 < parts.len() {
                let start = parts[idx].0;
                let end = parts[idx + 2].0;
                self.merge_ranks.get(&piece[start..end]).copied().unwrap_or(usize::MAX)
            } else {
                usize::MAX
            }
        };

        // Greedy iterative merge loop: merge lowest rank pair
        while min_rank.0 != usize::MAX {
            let i = min_rank.1;

            // Remove the swallowed right part
            parts.remove(i + 1);

            // Update rank with next element
            parts[i].1 = get_rank(&parts, i);
            // Update rank with previous element if exists
            if i > 0 {
                parts[i - 1].1 = get_rank(&parts, i - 1);
            }

            min_rank = (usize::MAX, usize::MAX);
            for (idx, &(_, rank)) in parts[..parts.len() - 1].iter().enumerate() {
                if rank < min_rank.0 {
                    min_rank = (rank, idx);
                }
            }
        }

        // Convert merged byte chunks to token IDs
        let mut out = Vec::with_capacity(parts.len() - 1);
        for i in 0..parts.len() - 1 {
            let start = parts[i].0;
            let end = parts[i + 1].0;
            let chunk_slice = &piece[start..end];

            if let Some(&id) = self.byte_ranks.get(chunk_slice) {
                out.push(id as u32);
            } else if let Ok(s) = std::str::from_utf8(chunk_slice) {
                if let Some(id) = self.vocab.get_id(s) {
                    out.push(id);
                } else {
                    for &b in chunk_slice {
                        let hex = format!("<0x{:02X}>", b);
                        out.push(self.vocab.get_id(&hex).unwrap_or(0));
                    }
                }
            } else {
                for &b in chunk_slice {
                    let hex = format!("<0x{:02X}>", b);
                    out.push(self.vocab.get_id(&hex).unwrap_or(0));
                }
            }
        }

        out
    }

    /// Decode sequence of token IDs to text with lossless byte fallback accumulation.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut raw_bytes = Vec::new();
        for &id in ids {
            if let Some(token_str) = self.vocab.get_token(id) {
                // Check if byte fallback token like <0x0A> or <0xE2>
                if token_str.starts_with("<0x") && token_str.ends_with('>') && token_str.len() == 6 {
                    if let Ok(byte_val) = u8::from_str_radix(&token_str[3..5], 16) {
                        raw_bytes.push(byte_val);
                        continue;
                    }
                }
                raw_bytes.extend_from_slice(token_str.as_bytes());
            }
        }
        String::from_utf8_lossy(&raw_bytes).into_owned()
    }

    /// Decode single token ID
    pub fn decode_token(&self, id: u32) -> Option<&str> {
        self.vocab.get_token(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpe_encoder() {
        let tokens = vec![
            "h".to_string(),
            "e".to_string(),
            "l".to_string(),
            "o".to_string(),
            "he".to_string(),
            "ll".to_string(),
            "hell".to_string(),
            "hello".to_string(),
        ];
        let vocab = Vocab::new(tokens);
        let tokenizer = Tokenizer::new(vocab, HashMap::new());

        let encoded = tokenizer.encode("hello");
        eprintln!("ENCODED: {:?}", encoded);
        assert_eq!(encoded, vec![7]);

        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_byte_fallback_utf8_decode() {
        // Multi-byte UTF-8 symbol: € is [0xE2, 0x82, 0xAC]
        let tokens = vec![
            "<unk>".to_string(),
            "<0xE2>".to_string(),
            "<0x82>".to_string(),
            "<0xAC>".to_string(),
        ];
        let vocab = Vocab::new(tokens);
        let tokenizer = Tokenizer::new(vocab, HashMap::new());

        let decoded = tokenizer.decode(&[1, 2, 3]);
        assert_eq!(decoded, "€");
    }
}
