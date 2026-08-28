//! Byte-Pair Encoding (BPE) implementation.

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

    /// Basic byte-level fallback encoder
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        // Check for special tokens first or fallback to piece matching
        let bytes = text.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            // Greedy match against vocabulary
            let mut matched = false;
            for len in (1..=std::cmp::min(32, bytes.len() - i)).rev() {
                if let Ok(substr) = std::str::from_utf8(&bytes[i..i + len]) {
                    if let Some(id) = self.vocab.get_id(substr) {
                        tokens.push(id);
                        i += len;
                        matched = true;
                        break;
                    }
                }
            }
            if !matched {
                // Byte fallback
                let b = bytes[i];
                let byte_str = format!("<0x{:02X}>", b);
                if let Some(id) = self.vocab.get_id(&byte_str) {
                    tokens.push(id);
                } else if let Some(id) = self.vocab.get_id(&format!("{}", b as char)) {
                    tokens.push(id);
                }
                i += 1;
            }
        }
        tokens
    }

    /// Decode sequence of token IDs to text
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        for &id in ids {
            if let Some(token_str) = self.vocab.get_token(id) {
                // Check if byte fallback token like <0x0A>
                if token_str.starts_with("<0x") && token_str.ends_with('>') && token_str.len() == 6
                {
                    if let Ok(byte_val) = u8::from_str_radix(&token_str[3..5], 16) {
                        out.push(byte_val as char);
                        continue;
                    }
                }
                out.push_str(token_str);
            }
        }
        out
    }

    /// Decode single token ID
    pub fn decode_token(&self, id: u32) -> Option<&str> {
        self.vocab.get_token(id)
    }
}
