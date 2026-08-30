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

pub const BYTE_FALLBACK_PREFIX: &str = "<0x";
pub const BYTE_FALLBACK_TOKEN_LEN: usize = 6;

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
            return vec![crate::special::UNK_TOKEN_ID];
        }

        #[derive(Clone, Copy)]
        struct Part {
            start: usize,
            len: usize,
            next: isize,
        }

        let n = piece.len();
        let mut parts: Vec<Part> = (0..n)
            .map(|i| Part {
                start: i,
                len: 1,
                next: if i + 1 < n { i as isize + 1 } else { -1 },
            })
            .collect();

        let get_pair_rank = |p_idx: usize, parts: &[Part]| -> usize {
            let next_idx = parts[p_idx].next;
            if next_idx >= 0 {
                let start = parts[p_idx].start;
                let next_u = next_idx as usize;
                let end = parts[next_u].start + parts[next_u].len;
                self.merge_ranks
                    .get(&piece[start..end])
                    .copied()
                    .unwrap_or(usize::MAX)
            } else {
                usize::MAX
            }
        };

        // Greedy iterative merge loop using index linked list to avoid O(N) array shifts
        loop {
            let mut min_rank = usize::MAX;
            let mut best_i = None;
            let mut curr = 0isize;

            while curr >= 0 {
                let i = curr as usize;
                let rank = get_pair_rank(i, &parts);
                if rank < min_rank {
                    min_rank = rank;
                    best_i = Some(i);
                }
                curr = parts[i].next;
            }

            if min_rank == usize::MAX {
                break;
            }

            if let Some(i) = best_i {
                let j = parts[i].next as usize;
                // Merge j into i
                parts[i].len += parts[j].len;
                parts[i].next = parts[j].next;
            }
        }

        // Convert merged byte chunks to token IDs
        let mut out = Vec::new();
        let mut curr = 0isize;
        while curr >= 0 {
            let i = curr as usize;
            let start = parts[i].start;
            let end = start + parts[i].len;
            let chunk_slice = &piece[start..end];

            if let Some(&id) = self.byte_ranks.get(chunk_slice) {
                out.push(id as u32);
            } else if let Ok(s) = std::str::from_utf8(chunk_slice) {
                if let Some(id) = self.vocab.get_id(s) {
                    out.push(id);
                } else {
                    encode_byte_fallback(chunk_slice, &self.vocab, &mut out);
                }
            } else {
                encode_byte_fallback(chunk_slice, &self.vocab, &mut out);
            }

            curr = parts[i].next;
        }

        out
    }

    /// Decode sequence of token IDs to text with lossless byte fallback accumulation.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut raw_bytes = Vec::new();
        for &id in ids {
            if let Some(token_str) = self.vocab.get_token(id) {
                // Check if byte fallback token like <0x0A> or <0xE2>
                if token_str.starts_with(BYTE_FALLBACK_PREFIX)
                    && token_str.ends_with('>')
                    && token_str.len() == BYTE_FALLBACK_TOKEN_LEN
                {
                    let hex_start = BYTE_FALLBACK_PREFIX.len();
                    let hex_end = hex_start + 2;
                    if let Ok(byte_val) = u8::from_str_radix(&token_str[hex_start..hex_end], 16) {
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

#[inline]
fn encode_byte_fallback(chunk_slice: &[u8], vocab: &Vocab, out: &mut Vec<u32>) {
    for &b in chunk_slice {
        let hex = format!("<0x{:02X}>", b);
        out.push(vocab.get_id(&hex).unwrap_or(crate::special::UNK_TOKEN_ID));
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

    #[test]
    fn test_bpe_empty_string() {
        let vocab = Vocab::new(vec!["<unk>".to_string()]);
        let tokenizer = Tokenizer::new(vocab, HashMap::new());
        assert!(tokenizer.encode("").is_empty());
        assert_eq!(tokenizer.decode(&[]), "");
    }
}
