//! Integration tests for 64K/128K Long-Context Scaling, YaRN RoPE, and Selective KV Cache.

use mivi_core::rope::{RopeCache, RopeScaling};
use mivi_kv::prefix::{HybridStateSnapshot, PREFIX_CHUNK_SIZE};
use mivi_kv::{KvCache, PrefixCache};

#[test]
fn test_long_context_64k_kv_cache_selective_integrity() {
    // 16 total layers, 6 attention layers (indices: 2, 5, 8, 11, 13, 15), 10 SSM layers
    let attention_layers = vec![2, 5, 8, 11, 13, 15];
    let max_seq_len = 65536; // 64K
    let kv_dim = 512;

    let mut kv = KvCache::try_new_selective(16, max_seq_len, kv_dim, &attention_layers)
        .expect("Should allocate 64k selective KV cache");

    assert_eq!(kv.capacity_tokens(), 65536);
    assert_eq!(kv.n_allocated_layers(), 6);

    let k_sample = vec![0.5f32; kv_dim];
    let v_sample = vec![0.75f32; kv_dim];

    // Store at various long-context checkpoints
    assert!(kv.store(2, 0, &k_sample, &v_sample).is_ok());
    assert!(kv.store(5, 4096, &k_sample, &v_sample).is_ok());
    assert!(kv.store(8, 16384, &k_sample, &v_sample).is_ok());
    assert!(kv.store(11, 32768, &k_sample, &v_sample).is_ok());
    assert!(kv.store(15, 65535, &k_sample, &v_sample).is_ok());

    // Verify stored values at maximum sequence length
    let k_read = kv.get_k(15, 65535).expect("Read at pos 65535");
    assert_eq!(k_read[0], 0.5);
    assert_eq!(k_read[kv_dim - 1], 0.5);

    let v_read = kv.get_v(15, 65535).expect("Read at pos 65535");
    assert_eq!(v_read[0], 0.75);
    assert_eq!(v_read[kv_dim - 1], 0.75);

    // Verify SSM layers correctly reject KV storage (0 memory allocated)
    assert!(kv.store(0, 0, &k_sample, &v_sample).is_err());
    assert!(kv.store(1, 100, &k_sample, &v_sample).is_err());
    assert!(kv.store(3, 1000, &k_sample, &v_sample).is_err());
}

#[test]
fn test_long_context_yarn_rope_extrapolation_stability() {
    let head_dim = 64;
    let max_seq_len = 65536;
    let rope_base = 1000000.0;

    let yarn = RopeCache::new_with_scaling(
        head_dim,
        max_seq_len,
        rope_base,
        RopeScaling::YaRN {
            scale: 16.0,
            orig_max_seq_len: 4096,
            extrapolation_factor: 1.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
        },
    );

    let test_positions = [0, 512, 4096, 8192, 16384, 32768, 65535];

    for &pos in &test_positions {
        let mut q = vec![1.0f32; head_dim * 16]; // 16 heads
        let mut k = vec![1.0f32; head_dim * 8]; // 8 KV heads

        assert!(
            yarn.try_apply(&mut q, &mut k, pos, 16, 8).is_ok(),
            "YaRN rotation should succeed at pos {pos}"
        );

        // Ensure all values are finite and valid numbers (no NaN / Inf)
        assert!(q.iter().all(|&v| v.is_finite()));
        assert!(k.iter().all(|&v| v.is_finite()));

        // Ensure vector magnitude per head is preserved under orthogonal rotation
        for h in 0..16 {
            let head = &q[h * head_dim..(h + 1) * head_dim];
            let norm: f32 = head.iter().map(|x| x * x).sum::<f32>().sqrt();
            let orig_norm: f32 = (head_dim as f32).sqrt();
            assert!(
                (norm - orig_norm).abs() < 1e-4,
                "Norm should be preserved at pos {pos}, got {norm} vs {orig_norm}"
            );
        }
    }
}

#[test]
fn test_long_context_prefix_caching_64k_chaining() {
    let mut cache = PrefixCache::new(500, PREFIX_CHUNK_SIZE);
    let mut prev_hash = 0u64;

    let mut full_prompt = Vec::with_capacity(6400);

    // Create 100 consecutive 64-token chunks (6,400 tokens)
    for chunk_idx in 0..100 {
        let chunk_tokens: Vec<u32> = (0..64).map(|i| (chunk_idx * 64 + i) as u32).collect();
        full_prompt.extend_from_slice(&chunk_tokens);

        let state = HybridStateSnapshot::new(
            (chunk_idx + 1) * 64,
            vec![1.0f32; 100],
            vec![2.0f32; 100],
            vec![0.5f32; 50],
            vec![0.25f32; 50],
        );

        prev_hash = cache.insert_chunk(prev_hash, &chunk_tokens, chunk_idx, state);
    }

    assert_eq!(cache.len(), 100);
    let matched = cache.find_longest_prefix(&full_prompt);
    assert!(matched.is_some());
    assert_eq!(matched.unwrap().0, 6400);
}
