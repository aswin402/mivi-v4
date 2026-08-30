//! Vocabulary data structures and token mappings.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Vocab {
    pub id_to_token: Vec<String>,
    pub token_to_id: HashMap<String, u32>,
    pub special_tokens: HashMap<String, u32>,
}

impl Vocab {
    pub fn new(tokens: Vec<String>) -> Self {
        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (i, token) in tokens.iter().enumerate() {
            token_to_id.insert(token.clone(), i as u32);
        }

        Self {
            id_to_token: tokens,
            token_to_id,
            special_tokens: HashMap::new(),
        }
    }

    pub fn add_special_token(&mut self, token: String, id: u32) {
        self.special_tokens.insert(token.clone(), id);
        self.token_to_id.insert(token.clone(), id);
        let id_idx = id as usize;
        if id_idx >= self.id_to_token.len() {
            self.id_to_token.resize(id_idx + 1, String::new());
        }
        self.id_to_token[id_idx] = token;
    }

    #[inline]
    pub fn is_special(&self, token: &str) -> bool {
        self.special_tokens.contains_key(token)
    }

    #[inline]
    pub fn is_special_id(&self, id: u32) -> bool {
        self.special_tokens.values().any(|&v| v == id)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.id_to_token.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.id_to_token.is_empty()
    }

    #[inline]
    pub fn get_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(|s| s.as_str())
    }

    #[inline]
    pub fn get_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }
}
