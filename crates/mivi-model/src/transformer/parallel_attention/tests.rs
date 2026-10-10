use super::*;
use mivi_core::arena::ArenaConfig;

fn fixture(
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    tokens: usize,
) -> (ModelConfig, RunState, KvCache) {
    let cfg = ModelConfig {
        dim: heads * head_dim,
        hidden_dim: heads * head_dim * 2,
        n_layers: 1,
        n_heads: heads,
        n_kv_heads: kv_heads,
        head_dim,
        kv_dim: kv_heads * head_dim,
        vocab_size: 8,
        max_seq_len: tokens,
        block_types: vec![crate::BlockType::Attention],
        ..ModelConfig::default()
    };
    let arena = ArenaConfig {
        dim: cfg.dim,
        hidden_dim: cfg.hidden_dim,
        n_layers: 1,
        n_heads: heads,
        n_kv_heads: kv_heads,
        head_dim,
        kv_dim: cfg.kv_dim,
        vocab_size: 8,
        max_seq_len: tokens,
        ssm_state_dim: 1,
        ssm_conv_kernel: 3,
        max_lora_rank: 1,
        n_experts: 1,
    };
    let mut state = RunState::new(&arena);
    for (i, q) in state.q.iter_mut().enumerate() {
        *q = ((i * 13 % 37) as f32 - 18.0) * 0.07;
    }
    state.attn_out.fill(123.0);
    let mut kv = KvCache::try_new(1, tokens, cfg.kv_dim).unwrap();
    for pos in 0..tokens {
        let k: Vec<_> = (0..cfg.kv_dim)
            .map(|i| ((i * 7 + pos * 3) % 31) as f32 * 0.09 - 1.0)
            .collect();
        let v: Vec<_> = (0..cfg.kv_dim)
            .map(|i| ((i * 5 + pos * 11) % 29) as f32 * 0.11 - 1.5)
            .collect();
        kv.store(0, pos, &k, &v).unwrap();
    }
    (cfg, state, kv)
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn parallel_attention_matches_independent_serial_head_bits() {
    for workers in [1, 2] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        pool.install(|| {
            for heads in [1, 3, 5, 8] {
                for kv_heads in [1, heads] {
                    for head_dim in [4, 7, 16, 64] {
                        let (cfg, mut state, kv) = fixture(heads, kv_heads, head_dim, 17);
                        let original_kv = kv.export_state(17).unwrap();
                        let original_q = bits(&state.q);
                        let original_hb = bits(&state.hb);
                        for pos in [0, 1, 16] {
                            super::super::compute_gqa_attention(&mut state, &kv, 0, pos, &cfg).unwrap();
                            let expected = bits(&state.attn_out);
                            state.attn_out.fill(123.0);
                            compute(&mut state, &kv, 0, pos, &cfg).unwrap();
                            assert_eq!(bits(&state.attn_out), expected, "heads={heads} kv_heads={kv_heads} head_dim={head_dim} pos={pos} workers={workers}");
                            assert!(state.attn_out.iter().all(|v| v.is_finite()));
                            assert_eq!(bits(&state.q), original_q);
                            assert_eq!(bits(&state.hb), original_hb);
                            assert_eq!(kv.export_state(17).unwrap(), original_kv);
                        }
                    }
                }
            }
        });
    }
}

#[test]
fn parallel_attention_rejects_invalid_requests_before_writes() {
    let (cfg, mut state, kv) = fixture(3, 1, 7, 3);
    for case in 0..10 {
        let mut invalid = cfg.clone();
        let (mut layer, mut pos) = (0, 2);
        match case {
            0 => invalid.n_heads = 0,
            1 => invalid.n_kv_heads = 0,
            2 => invalid.head_dim = 0,
            3 => invalid.n_kv_heads = 2,
            4 => invalid.dim += 1,
            5 => invalid.kv_dim += 1,
            6 => invalid.n_heads = usize::MAX,
            7 => layer = 1,
            8 => pos = 3,
            _ => pos = usize::MAX,
        }
        let sentinel = bits(&state.attn_out);
        assert!(
            compute(&mut state, &kv, layer, pos, &invalid).is_err(),
            "case={case}"
        );
        assert_eq!(bits(&state.attn_out), sentinel);
    }
    state.q = vec![0.0; cfg.dim - 1].into_boxed_slice();
    let sentinel = bits(&state.attn_out);
    assert!(compute(&mut state, &kv, 0, 2, &cfg).is_err());
    assert_eq!(bits(&state.attn_out), sentinel);
    let (cfg, mut state, kv) = fixture(3, 1, 7, 3);
    state.attn_out = vec![123.0; cfg.dim - 1].into_boxed_slice();
    let sentinel = bits(&state.attn_out);
    assert!(compute(&mut state, &kv, 0, 2, &cfg).is_err());
    assert_eq!(bits(&state.attn_out), sentinel);
}

#[test]
fn parallel_attention_dispatch_is_explicit_and_precisions_fall_back() {
    for workers in [1, 2] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        pool.install(|| {
            let (cfg, mut state, kv) = fixture(1, 1, 64, 3);
            state.parallel_attention_enabled = true;
            super::super::compute_gqa_attention(&mut state, &kv, 0, 2, &cfg).unwrap();
            assert_eq!(state.parallel_attention_calls, 0, "one head stays serial");
            for precision in [
                mivi_kv::KvPrecision::F32,
                mivi_kv::KvPrecision::Q8_0,
                mivi_kv::KvPrecision::TurboQuant4,
                mivi_kv::KvPrecision::TurboQuant2,
            ] {
                let (cfg, mut state, source) = fixture(4, 2, 64, 3);
                let mut kv =
                    KvCache::try_new_selective_with_precision(1, 3, cfg.kv_dim, &[0], precision)
                        .unwrap();
                for pos in 0..3 {
                    kv.store(
                        0,
                        pos,
                        source.get_k(0, pos).unwrap(),
                        source.get_v(0, pos).unwrap(),
                    )
                    .unwrap();
                }
                assert!(!state.parallel_attention_enabled);
                super::super::compute_gqa_attention(&mut state, &kv, 0, 2, &cfg).unwrap();
                let expected = bits(&state.attn_out);
                assert_eq!(state.parallel_attention_calls, 0);
                state.parallel_attention_enabled = true;
                super::super::compute_gqa_attention(&mut state, &kv, 0, 2, &cfg).unwrap();
                assert_eq!(bits(&state.attn_out), expected);
                assert_eq!(
                    state.parallel_attention_calls,
                    usize::from(workers == 2 && precision == mivi_kv::KvPrecision::F32)
                );
                state.reset();
                assert!(state.parallel_attention_enabled);
                assert_eq!(state.parallel_attention_calls, 0);
            }
        });
    }
}

#[test]
fn parallel_attention_retains_output_tail_and_rejects_unstored_or_selective_layers() {
    let (cfg, mut state, kv) = fixture(3, 1, 7, 3);
    state.attn_out = vec![123.0; cfg.dim + 5].into_boxed_slice();
    compute(&mut state, &kv, 0, 2, &cfg).unwrap();
    assert_eq!(&state.attn_out[cfg.dim..], &[123.0; 5]);
    for cache in [
        KvCache::try_new(1, 3, cfg.kv_dim).unwrap(),
        KvCache::try_new_selective(2, 3, cfg.kv_dim, &[1]).unwrap(),
        KvCache::try_new(1, 3, cfg.kv_dim + 1).unwrap(),
    ] {
        let mut cache = cache;
        let width = if cache.get_k(0, 0).is_ok() {
            cache.get_k(0, 0).unwrap().len()
        } else {
            cfg.kv_dim
        };
        if cache.get_k(0, 0).is_ok() && width != cfg.kv_dim {
            cache
                .store(0, 0, &vec![0.0; width], &vec![0.0; width])
                .unwrap();
        } else if cache.get_k(0, 0).is_err() {
            cache
                .store(1, 0, &vec![0.0; width], &vec![0.0; width])
                .unwrap();
        }
        let sentinel = bits(&state.attn_out);
        assert!(compute(&mut state, &cache, 0, 0, &cfg).is_err());
        assert_eq!(bits(&state.attn_out), sentinel);
    }
}

#[test]
#[ignore = "explicit release operator pilot only"]
fn parallel_attention_operator_measurement() {
    assert!(!cfg!(debug_assertions));
    assert_eq!(rayon::current_num_threads(), 2);
    for tokens in [32, 512, 2048] {
        let (cfg, mut state, kv) = fixture(32, 8, 64, tokens);
        for pair in 0..7 {
            let mut expected = None;
            let mut walls = [0u128; 2];
            for enabled in [pair % 2 != 0, pair % 2 == 0] {
                state.parallel_attention_enabled = enabled;
                let started = Instant::now();
                for _ in 0..16 {
                    super::super::compute_gqa_attention(&mut state, &kv, 0, tokens - 1, &cfg)
                        .unwrap();
                }
                walls[usize::from(enabled)] = started.elapsed().as_nanos();
                let actual = bits(&state.attn_out);
                if let Some(expected) = &expected {
                    assert_eq!(&actual, expected);
                } else {
                    expected = Some(actual);
                }
            }
            println!("parallel_attention_operator tokens={tokens} pair={pair} warmup={} serial_ns={} candidate_ns={} ratio={:.6}", pair == 0, walls[0], walls[1], walls[1] as f64 / walls[0] as f64);
        }
    }
}
