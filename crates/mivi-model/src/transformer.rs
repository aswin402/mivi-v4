//! Grouped-Query Attention (GQA) transformer block forward pass.

use crate::config::ModelConfig;
use mivi_core::arena::RunState;
use mivi_core::math::{dot_product, rms_norm, softmax, swiglu, vec_add};
use mivi_kv::KvCache;
use mivi_quant::{quantized_matvec, GgmlType};

pub struct AttentionWeights<'a> {
    pub attn_norm: &'a [f32],
    pub q_weight: (GgmlType, &'a [u8]),
    pub k_weight: (GgmlType, &'a [u8]),
    pub v_weight: (GgmlType, &'a [u8]),
    pub o_weight: (GgmlType, &'a [u8]),

    pub ffn_norm: &'a [f32],
    pub ffn_gate: (GgmlType, &'a [u8]),
    pub ffn_up: (GgmlType, &'a [u8]),
    pub ffn_down: (GgmlType, &'a [u8]),
}

/// Forward pass through a single GQA Transformer layer for a single token.
pub fn attention_forward(
    layer: usize,
    pos: usize,
    state: &mut RunState,
    kv: &mut KvCache,
    w: &AttentionWeights,
    cfg: &ModelConfig,
    adapters: &crate::lora::ActiveAdapters,
    rope: &mivi_core::RopeCache,
) -> crate::model::Result<()> {
    let dim = cfg.dim;
    let kv_dim = cfg.kv_dim;
    let head_dim = cfg.head_dim;
    let n_heads = cfg.n_heads;
    let n_kv_heads = cfg.n_kv_heads;
    let hidden_dim = cfg.hidden_dim;
    let heads_per_kv = n_heads / n_kv_heads;

    // 1. Attention Pre-Norm: xb = rms_norm(x, attn_norm)
    rms_norm(&mut state.xb, &state.x, w.attn_norm, 1e-5);

    // 2. Q, K, V projections
    quantized_matvec(&mut state.q, w.q_weight.0, w.q_weight.1, &state.xb, dim, dim)?;
    adapters.apply_module(
        &format!("blk.{}.attn_q", layer),
        &state.xb,
        &mut state.lora_down,
        &mut state.q,
    );

    quantized_matvec(&mut state.k, w.k_weight.0, w.k_weight.1, &state.xb, kv_dim, dim)?;
    adapters.apply_module(
        &format!("blk.{}.attn_k", layer),
        &state.xb,
        &mut state.lora_down,
        &mut state.k,
    );

    quantized_matvec(&mut state.v, w.v_weight.0, w.v_weight.1, &state.xb, kv_dim, dim)?;
    adapters.apply_module(
        &format!("blk.{}.attn_v", layer),
        &state.xb,
        &mut state.lora_down,
        &mut state.v,
    );

    // 3. Apply RoPE using zero-allocation precomputed lookup table
    rope.apply(&mut state.q, &mut state.k, pos, n_heads, n_kv_heads);

    // 4. Store K, V in KV cache
    kv.store(layer, pos, &state.k, &state.v)?;

    // 5. Multi-head Attention with GQA
    let scale = 1.0 / (head_dim as f32).sqrt();
    let seq_len = pos + 1;

    for h in 0..n_heads {
        let kv_head = h / heads_per_kv;
        let q_head = &state.q[h * head_dim..(h + 1) * head_dim];
        let att_scores = &mut state.att[h * cfg.max_seq_len..h * cfg.max_seq_len + seq_len];

        // Compute dot products with past keys in cache
        for t in 0..seq_len {
            let k_cached = kv.get_k(layer, t)?;
            let k_head = &k_cached[kv_head * head_dim..(kv_head + 1) * head_dim];
            att_scores[t] = dot_product(q_head, k_head) * scale;
        }

        // Softmax over attention scores
        softmax(att_scores);

        // Weighted sum of cached values
        let out_head = &mut state.attn_out[h * head_dim..(h + 1) * head_dim];
        out_head.fill(0.0);

        for t in 0..seq_len {
            let weight = att_scores[t];
            let v_cached = kv.get_v(layer, t)?;
            let v_head = &v_cached[kv_head * head_dim..(kv_head + 1) * head_dim];
            for i in 0..head_dim {
                out_head[i] += weight * v_head[i];
            }
        }
    }

    // 6. Output projection: xb = W_o * attn_out
    quantized_matvec(
        &mut state.xb,
        w.o_weight.0,
        w.o_weight.1,
        &state.attn_out,
        dim,
        dim,
    )?;
    adapters.apply_module(
        &format!("blk.{}.attn_output", layer),
        &state.attn_out,
        &mut state.lora_down,
        &mut state.xb,
    );

    // 7. Residual connection: x = x + xb
    vec_add(&mut state.x, &state.xb);

    // 8. FFN Pre-Norm: xb = rms_norm(x, ffn_norm)
    rms_norm(&mut state.xb, &state.x, w.ffn_norm, 1e-5);

    // 9. SwiGLU FFN: hb = gate(xb), hb2 = up(xb)
    quantized_matvec(
        &mut state.hb,
        w.ffn_gate.0,
        w.ffn_gate.1,
        &state.xb,
        hidden_dim,
        dim,
    )?;
    adapters.apply_module(
        &format!("blk.{}.ffn_gate", layer),
        &state.xb,
        &mut state.lora_down,
        &mut state.hb,
    );

    quantized_matvec(
        &mut state.hb2,
        w.ffn_up.0,
        w.ffn_up.1,
        &state.xb,
        hidden_dim,
        dim,
    )?;
    adapters.apply_module(
        &format!("blk.{}.ffn_up", layer),
        &state.xb,
        &mut state.lora_down,
        &mut state.hb2,
    );

    swiglu(&mut state.hb, &state.hb2);

    // 10. Down projection: xb = W_down * hb
    quantized_matvec(
        &mut state.xb,
        w.ffn_down.0,
        w.ffn_down.1,
        &state.hb,
        dim,
        hidden_dim,
    )?;
    adapters.apply_module(
        &format!("blk.{}.ffn_down", layer),
        &state.hb,
        &mut state.lora_down,
        &mut state.xb,
    );

    // 11. Residual connection: x = x + xb
    vec_add(&mut state.x, &state.xb);
    Ok(())
}
