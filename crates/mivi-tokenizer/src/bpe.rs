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

/// Standard GPT-2 / Hugging Face bytes-to-unicode bijection table.
pub fn bytes_to_unicode() -> HashMap<u8, char> {
    let mut bs: Vec<u8> = (b'!'..=b'~')
        .chain(b'\xa1'..=b'\xac')
        .chain(b'\xae'..=b'\xff')
        .collect();
    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
    let mut n = 0u32;
    for b in 0..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    bs.into_iter()
        .zip(cs.into_iter().filter_map(char::from_u32))
        .collect()
}

pub fn unicode_to_bytes() -> HashMap<char, u8> {
    bytes_to_unicode()
        .into_iter()
        .map(|(b, c)| (c, b))
        .collect()
}

static BYTE_TO_UNICODE: std::sync::LazyLock<HashMap<u8, char>> =
    std::sync::LazyLock::new(bytes_to_unicode);
static UNICODE_TO_BYTE: std::sync::LazyLock<HashMap<char, u8>> =
    std::sync::LazyLock::new(unicode_to_bytes);

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
}

impl Tokenizer {
    pub fn new(vocab: Vocab, merges: HashMap<(String, String), u32>) -> Self {
        Self { vocab, merges }
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
            let piece = m.as_str();
            let piece_tokens = self.bpe_encode_piece(piece);
            tokens.extend(piece_tokens);
        }

        tokens
    }

    /// BPE merge loop for a single pre-tokenized piece.
    fn bpe_encode_piece(&self, piece: &str) -> Vec<u32> {
        if piece.is_empty() {
            return Vec::new();
        }

        // 1. Direct vocabulary lookup for special tokens or exact tokens
        if let Some(id) = self.vocab.get_id(piece) {
            return vec![id];
        }

        // 2. Convert UTF-8 bytes to GPT-2 unicode symbols
        let mut symbols: Vec<String> = piece
            .as_bytes()
            .iter()
            .map(|&b| {
                if let Some(&c) = BYTE_TO_UNICODE.get(&b) {
                    c.to_string()
                } else {
                    (b as char).to_string()
                }
            })
            .collect();

        if symbols.len() <= 1 {
            if let Some(s) = symbols.first() {
                if let Some(id) = self.vocab.get_id(s) {
                    return vec![id];
                }
            }
            return vec![crate::special::UNK_TOKEN_ID];
        }

        // 3. Iterative BPE pair merging using lowest rank in self.merges
        loop {
            if symbols.len() < 2 {
                break;
            }

            let mut best_pair = None;
            let mut min_rank = u32::MAX;

            for i in 0..symbols.len() - 1 {
                let pair = (symbols[i].clone(), symbols[i + 1].clone());
                if let Some(&rank) = self.merges.get(&pair) {
                    if rank < min_rank {
                        min_rank = rank;
                        best_pair = Some((i, pair));
                    }
                }
            }

            let Some((idx, _)) = best_pair else {
                break;
            };

            // Merge symbols[idx] and symbols[idx+1]
            let merged = format!("{}{}", symbols[idx], symbols[idx + 1]);
            symbols[idx] = merged;
            symbols.remove(idx + 1);
        }

        // 4. Map merged symbols to vocabulary IDs
        let mut out = Vec::with_capacity(symbols.len());
        for sym in &symbols {
            if let Some(id) = self.vocab.get_id(sym) {
                out.push(id);
            } else {
                // Fallback to byte tokens
                for &b in sym.as_bytes() {
                    let hex = format!("<0x{:02X}>", b);
                    out.push(
                        self.vocab
                            .get_id(&hex)
                            .unwrap_or(crate::special::UNK_TOKEN_ID),
                    );
                }
            }
        }

        out
    }

    /// Decode sequence of token IDs to clean UTF-8 text.
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

                // If special token (e.g. <|im_start|>, <|im_end|>), write literal bytes
                if token_str.starts_with("<|") && token_str.ends_with("|>") {
                    raw_bytes.extend_from_slice(token_str.as_bytes());
                    continue;
                }

                // Map GPT-2 unicode characters back to raw bytes
                for c in token_str.chars() {
                    if let Some(&b) = UNICODE_TO_BYTE.get(&c) {
                        raw_bytes.push(b);
                    } else {
                        let mut buf = [0u8; 4];
                        let encoded = c.encode_utf8(&mut buf);
                        raw_bytes.extend_from_slice(encoded.as_bytes());
                    }
                }
            }
        }
        String::from_utf8_lossy(&raw_bytes).into_owned()
    }

    /// Decode single token ID to clean UTF-8 text string.
    pub fn decode_token(&self, id: u32) -> String {
        self.decode(&[id])
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
