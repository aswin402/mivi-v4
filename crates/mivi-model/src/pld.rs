//! Prompt Lookup Speculative Decoding (PLD) for CPU inference acceleration.
//!
//! Inspired by Prompt Lookup Decoding (Saxena, Google Research, 2023), this module
//! finds recurring n-grams in the prompt and conversation context to propose speculative
//! draft tokens. These tokens are verified in forward passes, achieving 1.5x-2.5x speedups
//! on code generation, JSON extraction, and RAG without requiring an auxiliary draft model.

/// Default n-gram context matching size (3-gram).
pub const DEFAULT_PLD_NGRAM_SIZE: usize = 3;

/// Default speculative draft window size (3 tokens).
pub const DEFAULT_PLD_DRAFT_SIZE: usize = 3;

/// Proposer that identifies continuation n-grams within the prompt and context history.
#[derive(Debug, Clone)]
pub struct PromptLookupProposer {
    pub ngram_size: usize,
    pub draft_size: usize,
}

impl Default for PromptLookupProposer {
    fn default() -> Self {
        Self::new(DEFAULT_PLD_NGRAM_SIZE, DEFAULT_PLD_DRAFT_SIZE)
    }
}

impl PromptLookupProposer {
    /// Create a new PromptLookupProposer with custom ngram and draft window sizes.
    pub fn new(ngram_size: usize, draft_size: usize) -> Self {
        Self {
            ngram_size: ngram_size.max(1),
            draft_size: draft_size.max(1),
        }
    }

    /// Search for matching n-gram continuation in the token history.
    ///
    /// Given `all_tokens` (prompt + generated tokens so far), inspects the last `ngram_size`
    /// tokens and searches for their previous occurrence earlier in the sequence.
    pub fn propose(&self, all_tokens: &[u32]) -> Option<Vec<u32>> {
        if all_tokens.len() <= self.ngram_size {
            return None;
        }

        let query_start = all_tokens.len() - self.ngram_size;
        let query_ngram = &all_tokens[query_start..];

        // Search in the historical window before the query ngram
        let search_limit = query_start;
        if search_limit < self.ngram_size {
            return None;
        }

        // Scan backwards to favor recent occurrences
        for i in (0..=(search_limit - self.ngram_size)).rev() {
            if &all_tokens[i..i + self.ngram_size] == query_ngram {
                let draft_start = i + self.ngram_size;
                if draft_start < query_start {
                    let draft_end = (draft_start + self.draft_size).min(query_start);
                    let draft = all_tokens[draft_start..draft_end].to_vec();
                    if !draft.is_empty() {
                        return Some(draft);
                    }
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
    fn test_pld_proposer_finds_exact_match() {
        let proposer = PromptLookupProposer::new(3, 3);
        // Prompt has: [10, 20, 30, 40, 50, 60, 70]
        // Generation just produced: [10, 20, 30]
        let tokens = vec![10, 20, 30, 40, 50, 60, 70, 999, 10, 20, 30];

        let proposed = proposer.propose(&tokens);
        assert!(proposed.is_some());
        let draft = proposed.unwrap();
        assert_eq!(draft, vec![40, 50, 60]);
    }

    #[test]
    fn test_pld_proposer_no_match_returns_none() {
        let proposer = PromptLookupProposer::new(3, 3);
        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        assert_eq!(proposer.propose(&tokens), None);
    }
}
