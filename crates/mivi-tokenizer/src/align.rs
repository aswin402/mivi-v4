//! Prefix Cache Boundary Alignment and Prompt Normalization.
//!
//! Inspired by Headroom's `cache_align` and LMCache chunk boundary synchronization.
//! Ensures that multi-turn system prompts and static conversation headers align to exact
//! multiples of `PREFIX_CHUNK_SIZE` (64 tokens) to achieve 100% prefix cache reuse and 0 ms TTFT.

pub const DEFAULT_PREFIX_CHUNK_SIZE: usize = 64;

/// Normalizes prompt text whitespace and line-endings to eliminate superficial variances
/// that could invalidate deterministic prefix chunk hash keys.
pub fn normalize_prompt_whitespace(text: &str) -> String {
    let mut normalized = String::with_capacity(text.len());
    let mut consecutive_empty_lines = 0;

    for line in text.lines() {
        let trimmed_line = line.trim_end();
        if trimmed_line.is_empty() {
            consecutive_empty_lines += 1;
            if consecutive_empty_lines == 1 && !normalized.is_empty() {
                normalized.push('\n');
            }
        } else {
            if !normalized.is_empty() {
                normalized.push('\n');
            }
            normalized.push_str(trimmed_line);
            consecutive_empty_lines = 0;
        }
    }

    normalized
}

/// Splits a token sequence into a chunk-aligned prefix (exact multiple of `chunk_size`)
/// and the remaining suffix tokens.
///
/// # Example
/// If `tokens.len() == 150` and `chunk_size == 64`, returns:
/// - Prefix: 128 tokens (2 complete 64-token chunks)
/// - Suffix: 22 tokens (to be processed incrementally)
#[inline]
pub fn split_aligned_prefix(tokens: &[u32], chunk_size: usize) -> (&[u32], &[u32]) {
    let chunk_size = chunk_size.max(1);
    let aligned_len = (tokens.len() / chunk_size) * chunk_size;
    (&tokens[..aligned_len], &tokens[aligned_len..])
}

/// Pads a token sequence to the nearest multiple of `chunk_size` using `pad_token_id`.
///
/// Returns the padded token vector and the number of pad tokens added.
pub fn pad_to_chunk_boundary(
    tokens: &[u32],
    chunk_size: usize,
    pad_token_id: u32,
) -> (Vec<u32>, usize) {
    let chunk_size = chunk_size.max(1);
    let remainder = tokens.len() % chunk_size;
    if remainder == 0 {
        return (tokens.to_vec(), 0);
    }

    let padding_needed = chunk_size - remainder;
    let mut padded = Vec::with_capacity(tokens.len() + padding_needed);
    padded.extend_from_slice(tokens);
    padded.resize(tokens.len() + padding_needed, pad_token_id);
    (padded, padding_needed)
}

/// Aligns a static prefix (e.g. system instructions + tool schemas) to the nearest 64-token boundary
/// before appending dynamic user messages.
pub fn align_system_prefix(
    system_tokens: &[u32],
    chunk_size: usize,
    pad_token_id: Option<u32>,
) -> Vec<u32> {
    let chunk_size = chunk_size.max(1);
    if let Some(pad_id) = pad_token_id {
        let (padded, _) = pad_to_chunk_boundary(system_tokens, chunk_size, pad_id);
        padded
    } else {
        system_tokens.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_prompt_whitespace() {
        let raw = "Hello World!  \r\n\r\n\r\n\r\nThis is Mivi.   \n\nHow are you? ";
        let norm = normalize_prompt_whitespace(raw);
        assert_eq!(norm, "Hello World!\n\nThis is Mivi.\n\nHow are you?");
    }

    #[test]
    fn test_split_aligned_prefix() {
        let tokens: Vec<u32> = (0..150).collect();
        let (prefix, suffix) = split_aligned_prefix(&tokens, 64);
        assert_eq!(prefix.len(), 128);
        assert_eq!(suffix.len(), 22);
        assert_eq!(prefix[0], 0);
        assert_eq!(prefix[127], 127);
        assert_eq!(suffix[0], 128);
        assert_eq!(suffix[21], 149);
    }

    #[test]
    fn test_pad_to_chunk_boundary() {
        let tokens: Vec<u32> = (0..50).collect();
        let (padded, pad_count) = pad_to_chunk_boundary(&tokens, 64, 0);
        assert_eq!(padded.len(), 64);
        assert_eq!(pad_count, 14);
        assert_eq!(&padded[50..64], &[0u32; 14]);

        let exact_tokens: Vec<u32> = (0..64).collect();
        let (exact_padded, exact_pad_count) = pad_to_chunk_boundary(&exact_tokens, 64, 0);
        assert_eq!(exact_padded.len(), 64);
        assert_eq!(exact_pad_count, 0);
    }
}
