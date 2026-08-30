//! Grouped Query Attention (GQA) Transformer layer implementation with RoPE.

use crate::config::{ModelConfig, DEFAULT_RMS_NORM_EPS};
use crate::ffn::{ffn_swiglu_forward, linear_forward, FfnSwigluParams, LinearParams};
use crate::lora::ActiveAdapters;
use crate::model::Result;
use crate::weights::AttentionLayerWeights;
use mivi_core::arena::RunState;
use mivi_core::math::{dot_product, softmax, vec_add};
use mivi_core::simd::rms_norm_simd;
use mivi_kv::KvCache;

/// Parameter descriptor for GQA Attention forward pass.
pub struct AttentionParams<'a> {
    pub layer: usize,
    pub pos: usize,
    pub weights: &'a AttentionLayerWeights,
    pub mmap: &'a [u8],
    pub config: &'a ModelConfig,
    pub adapters: &'a ActiveAdapters,
    pub rope: &'a mivi_core::RopeCache,
}

/// Compute Q, K, V projections, apply RoPE, and store in KV cache.
#[inline]
fn compute_qkv(state: &mut RunState, kv: &mut KvCache, params: &AttentionParams) -> Result<()> {
    let cfg = params.config;
    let w = params.weights;
    let dim = cfg.dim;
    let kv_dim = cfg.kv_dim;
    let mmap = params.mmap;
    let adapters = params.adapters;

    let mut project =
        |out: &mut [f32], weight: &crate::weights::QuantizedTensor, rows: usize, name: &str| {
            let p = LinearParams {
                weight,
                input: &state.xb,
                rows,
                cols: dim,
                mmap,
                adapters,
                module_name: name,
            };
            linear_forward(out, &p, &mut state.lora_down)
        };

    project(&mut state.q, &w.wq, dim, &w.q_name)?;
    project(&mut state.k, &w.wk, kv_dim, &w.k_name)?;
    project(&mut state.v, &w.wv, kv_dim, &w.v_name)?;

    // Apply QK-Norm per head if weights are present
    let head_dim = cfg.head_dim;
    if let Some(ref q_norm_w) = w.q_norm {
        let mut temp = [0.0f32; 128];
        for h in 0..cfg.n_heads {
            let offset = h * head_dim;
            let head_slice = &mut state.q[offset..offset + head_dim];
            rms_norm_simd(
                &mut temp[..head_dim],
                head_slice,
                q_norm_w,
                DEFAULT_RMS_NORM_EPS,
            );
            head_slice.copy_from_slice(&temp[..head_dim]);
        }
    }
    if let Some(ref k_norm_w) = w.k_norm {
        let mut temp = [0.0f32; 128];
        for kv_h in 0..cfg.n_kv_heads {
            let offset = kv_h * head_dim;
            let head_slice = &mut state.k[offset..offset + head_dim];
            rms_norm_simd(
                &mut temp[..head_dim],
                head_slice,
                k_norm_w,
                DEFAULT_RMS_NORM_EPS,
            );
            head_slice.copy_from_slice(&temp[..head_dim]);
        }
    }

    // Apply RoPE using zero-allocation precomputed lookup table
    params.rope.apply(
        &mut state.q,
        &mut state.k,
        params.pos,
        cfg.n_heads,
        cfg.n_kv_heads,
    );

    // Store K, V in KV cache
    kv.store(params.layer, params.pos, &state.k, &state.v)?;
    Ok(())
}

/// Compute multi-head Grouped Query Attention (GQA) over cached keys and values.
#[inline]
fn compute_gqa_attention(
    state: &mut RunState,
    kv: &KvCache,
    layer: usize,
    pos: usize,
    cfg: &ModelConfig,
) -> Result<()> {
    // Validate layer and pos bounds once upfront
    let _ = kv.get_k(layer, pos)?;
    let _ = kv.get_v(layer, pos)?;

    let head_dim = cfg.head_dim;
    let n_heads = cfg.n_heads;
    let heads_per_kv = n_heads / cfg.n_kv_heads.max(1);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let seq_len = pos + 1;

    for h in 0..n_heads {
        let kv_head = h / heads_per_kv;
        let q_head = &state.q[h * head_dim..(h + 1) * head_dim];
        let att_scores = &mut state.att[h * cfg.max_seq_len..h * cfg.max_seq_len + seq_len];

        // Compute dot products with past keys in cache (unchecked inner loop)
        for (t, att_val) in att_scores.iter_mut().enumerate().take(seq_len) {
            // SAFETY: layer bounds checked upfront; t <= pos < max_seq_len.
            let k_cached = unsafe { kv.get_k_unchecked(layer, t) };
            let k_head = &k_cached[kv_head * head_dim..(kv_head + 1) * head_dim];
            *att_val = dot_product(q_head, k_head) * scale;
        }

        // Softmax over attention scores
        softmax(att_scores);

        // Weighted sum of cached values (unchecked inner loop)
        let out_head = &mut state.attn_out[h * head_dim..(h + 1) * head_dim];
        out_head.fill(0.0);

        for (t, &weight) in att_scores.iter().enumerate().take(seq_len) {
            // SAFETY: layer bounds checked upfront; t <= pos < max_seq_len.
            let v_cached = unsafe { kv.get_v_unchecked(layer, t) };
            let v_head = &v_cached[kv_head * head_dim..(kv_head + 1) * head_dim];
            for i in 0..head_dim {
                out_head[i] += weight * v_head[i];
            }
        }
    }
    Ok(())
}

/// Forward pass through a single GQA Transformer layer for a single token.
pub fn attention_forward(
    state: &mut RunState,
    kv: &mut KvCache,
    params: &AttentionParams,
) -> Result<()> {
    let cfg = params.config;
    let w = params.weights;
    let dim = cfg.dim;
    let hidden_dim = cfg.hidden_dim;
    let mmap = params.mmap;
    let adapters = params.adapters;

    // 1. Attention Pre-Norm (SIMD accelerated)
    rms_norm_simd(&mut state.xb, &state.x, &w.attn_norm, DEFAULT_RMS_NORM_EPS);

    // 2-4. Q, K, V projections + RoPE + Cache store
    compute_qkv(state, kv, params)?;

    // 5. Multi-head Attention with GQA
    compute_gqa_attention(state, kv, params.layer, params.pos, cfg)?;

    // 6. Output projection: xb = W_o * attn_out + LoRA
    let out_params = LinearParams {
        weight: &w.wo,
        input: &state.attn_out,
        rows: dim,
        cols: dim,
        mmap,
        adapters,
        module_name: &w.o_name,
    };
    linear_forward(&mut state.xb, &out_params, &mut state.lora_down)?;

    // 7. Residual connection: x = x + xb
    vec_add(&mut state.x, &state.xb);

    // 8-11. Shared FFN SwiGLU forward
    let ffn_params = FfnSwigluParams {
        weights: &w.ffn,
        dim,
        hidden_dim,
        mmap,
        adapters,
        eps: DEFAULT_RMS_NORM_EPS,
    };
    ffn_swiglu_forward(state, &ffn_params)
}
