use mivi_kv::{HybridStateSnapshot, KvCache, KvPrecision, PrefixCache, PREFIX_CHUNK_SIZE};
use mivi_quant::q8_0::{dot_q8_0_f32, quantize_f32_to_q8_0_block, Q8_0_BYTES};

#[test]
fn test_quantized_kv_cache_q8_0_memory_reduction() {
    let n_layers = 16;
    let attention_layers = vec![2, 5, 8, 11, 13, 15]; // 6 attention layers
    let max_seq_len = 65536; // 64K context
    let kv_dim = 512;

    let kv_f32 = KvCache::try_new_selective_with_precision(
        n_layers,
        max_seq_len,
        kv_dim,
        &attention_layers,
        KvPrecision::F32,
    )
    .expect("Should allocate F32 KV cache");

    let kv_q8 = KvCache::try_new_selective_with_precision(
        n_layers,
        max_seq_len,
        kv_dim,
        &attention_layers,
        KvPrecision::Q8_0,
    )
    .expect("Should allocate Q8_0 KV cache");

    let f32_bytes = kv_f32.memory_bytes();
    let q8_bytes = kv_q8.memory_bytes();

    assert_eq!(f32_bytes, 1_610_612_736); // ~1.61 GB
    assert_eq!(q8_bytes, 427_819_008);    // ~427 MB

    let reduction_ratio = 1.0 - (q8_bytes as f64 / f32_bytes as f64);
    assert!(reduction_ratio > 0.73, "Q8_0 must provide at least 73% memory reduction");

    let kv_tq4 = KvCache::try_new_selective_with_precision(
        n_layers,
        max_seq_len,
        kv_dim,
        &attention_layers,
        KvPrecision::TurboQuant4,
    )
    .expect("Should allocate TurboQuant4 KV cache");

    let tq4_bytes = kv_tq4.memory_bytes();
    assert_eq!(tq4_bytes, 204_472_320); // ~204.4 MB
    let tq4_reduction = 1.0 - (tq4_bytes as f64 / f32_bytes as f64);
    assert!(tq4_reduction > 0.87, "TurboQuant4 must provide >87% memory reduction (got: {tq4_reduction})");

    let kv_tq2 = KvCache::try_new_selective_with_precision(
        n_layers,
        max_seq_len,
        kv_dim,
        &attention_layers,
        KvPrecision::TurboQuant2,
    )
    .expect("Should allocate TurboQuant2 KV cache");

    let tq2_bytes = kv_tq2.memory_bytes();
    assert_eq!(tq2_bytes, 103_809_024); // ~103.8 MB
    let tq2_reduction = 1.0 - (tq2_bytes as f64 / f32_bytes as f64);
    assert!(tq2_reduction > 0.93, "TurboQuant2 must provide >93% memory reduction (got: {tq2_reduction})");
}

#[test]
fn test_fused_q8_0_dot_product_accuracy() {
    let dim = 32;
    let mut q = vec![0.0f32; dim];
    let mut k = vec![0.0f32; dim];

    for i in 0..dim {
        q[i] = ((i as f32 * 0.17).sin() * 2.5).clamp(-3.0, 3.0);
        k[i] = ((i as f32 * 0.23).cos() * 1.8).clamp(-3.0, 3.0);
    }

    let mut k_block = [0u8; Q8_0_BYTES];
    quantize_f32_to_q8_0_block(&k, &mut k_block);

    let approx_dot = dot_q8_0_f32(&q, &k_block);
    let true_dot: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();

    let diff = (approx_dot - true_dot).abs();
    let relative_err = diff / true_dot.abs().max(1e-4);

    assert!(
        relative_err < 0.05,
        "Q8_0 fused dot product error must be < 5% (got diff: {diff}, true: {true_dot}, approx: {approx_dot})"
    );
}

#[test]
fn test_chunked_prefix_caching_multi_chunk_chain() {
    let mut prefix_cache = PrefixCache::new(4, PREFIX_CHUNK_SIZE);

    let chunk_1_tokens: Vec<u32> = (0..PREFIX_CHUNK_SIZE as u32).collect();
    let chunk_2_tokens: Vec<u32> = (PREFIX_CHUNK_SIZE as u32..2 * PREFIX_CHUNK_SIZE as u32).collect();

    let snap_1 = HybridStateSnapshot::new(PREFIX_CHUNK_SIZE, vec![1.0; 64], vec![2.0; 64], vec![0.1; 16], vec![0.2; 16]);
    let snap_2 = HybridStateSnapshot::new(2 * PREFIX_CHUNK_SIZE, vec![1.1; 64], vec![2.1; 64], vec![0.11; 16], vec![0.21; 16]);

    let hash_1 = prefix_cache.insert_chunk(0, &chunk_1_tokens, 0, snap_1);
    assert_ne!(hash_1, 0);

    let hash_2 = prefix_cache.insert_chunk(hash_1, &chunk_2_tokens, 1, snap_2);
    assert_ne!(hash_2, 0);
    assert_ne!(hash_2, hash_1);

    let full_prompt: Vec<u32> = (0..2 * PREFIX_CHUNK_SIZE as u32).collect();
    let match_res = prefix_cache.find_longest_prefix(&full_prompt);
    assert!(match_res.is_some());

    let (matched_len, chunk) = match_res.unwrap();
    assert_eq!(matched_len, 2 * PREFIX_CHUNK_SIZE);
    assert_eq!(chunk.hash, hash_2);
}
