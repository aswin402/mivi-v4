//! Grouped Query Attention (GQA) Transformer layer implementation with RoPE.

use crate::config::ModelConfig;
use crate::ffn::{ffn_swiglu_forward, linear_forward, FfnSwigluParams, LinearParams};
use crate::lora::ActiveAdapters;
use crate::model::Result;
use crate::weights::AttentionLayerWeights;
use mivi_core::arena::RunState;
use mivi_core::math::{dot_product, vec_add};
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
        for h in 0..cfg.n_heads {
            let offset = h * head_dim;
            let head_slice = &mut state.q[offset..offset + head_dim];
            mivi_core::rms_norm_in_place_simd(head_slice, q_norm_w, cfg.rms_norm_eps);
        }
    }
    if let Some(ref k_norm_w) = w.k_norm {
        for kv_h in 0..cfg.n_kv_heads {
            let offset = kv_h * head_dim;
            let head_slice = &mut state.k[offset..offset + head_dim];
            mivi_core::rms_norm_in_place_simd(head_slice, k_norm_w, cfg.rms_norm_eps);
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

/// Compute multi-head Grouped Query Attention (GQA) over cached keys and values using FlashDecoding (online softmax).
#[inline]
fn compute_gqa_attention(
    state: &mut RunState,
    kv: &KvCache,
    layer: usize,
    pos: usize,
    cfg: &ModelConfig,
) -> Result<()> {
    let head_dim = cfg.head_dim;
    let n_heads = cfg.n_heads;
    let heads_per_kv = (n_heads / cfg.n_kv_heads.max(1)).max(1);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let seq_len = pos + 1;
    let precision = kv.precision();

    for h in 0..n_heads {
        let kv_head = h / heads_per_kv;
        let q_head = &state.q[h * head_dim..(h + 1) * head_dim];
        let out_head = &mut state.attn_out[h * head_dim..(h + 1) * head_dim];
        out_head.fill(0.0);

        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0f32;

        match precision {
            mivi_kv::KvPrecision::F32 => {
                // FlashDecoding online softmax single-pass accumulation for FP32
                for t in 0..seq_len {
                    // SAFETY: layer bounds checked upfront; t <= pos < max_seq_len.
                    let k_cached = unsafe { kv.get_k_unchecked(layer, t) };
                    let k_head = &k_cached[kv_head * head_dim..(kv_head + 1) * head_dim];
                    let score = dot_product(q_head, k_head) * scale;

                    let v_cached = unsafe { kv.get_v_unchecked(layer, t) };
                    let v_head = &v_cached[kv_head * head_dim..(kv_head + 1) * head_dim];

                    if score > f32::NEG_INFINITY {
                        if score > running_max || running_max == f32::NEG_INFINITY {
                            let alpha = if running_max == f32::NEG_INFINITY {
                                0.0
                            } else {
                                (running_max - score).exp()
                            };
                            running_sum = running_sum * alpha + 1.0;
                            for i in 0..head_dim {
                                out_head[i] = out_head[i] * alpha + v_head[i];
                            }
                            running_max = score;
                        } else {
                            let beta = (score - running_max).exp();
                            running_sum += beta;
                            mivi_core::vec_fmadd(out_head, beta, v_head);
                        }
                    }
                }
            }
            mivi_kv::KvPrecision::Q8_0 => {
                let blocks_per_head = (head_dim + 31) / 32;
                let kv_head_block_start = kv_head * blocks_per_head;
                let mut v_head_buf = [0.0f32; 256];

                for t in 0..seq_len {
                    // Compute fused dot product directly over Q8_0 blocks
                    let mut score = 0.0f32;
                    for b in 0..blocks_per_head {
                        let q_slice = &q_head[b * 32..(b * 32 + 32).min(head_dim)];
                        let k_block = unsafe {
                            kv.get_k_q8_block_unchecked(layer, t, kv_head_block_start + b)
                        };
                        score += mivi_quant::q8_0::dot_q8_0_f32(q_slice, k_block);
                    }
                    score *= scale;

                    // Dequantize value blocks into stack buffer
                    for b in 0..blocks_per_head {
                        let v_block = unsafe {
                            kv.get_v_q8_block_unchecked(layer, t, kv_head_block_start + b)
                        };
                        let start = b * 32;
                        let end = (start + 32).min(head_dim);
                        let mut block_buf = [0.0f32; 32];
                        mivi_quant::q8_0::dequantize_q8_0(v_block, &mut block_buf);
                        v_head_buf[start..end].copy_from_slice(&block_buf[..end - start]);
                    }
                    let v_head = &v_head_buf[..head_dim];

                    if score > f32::NEG_INFINITY {
                        if score > running_max || running_max == f32::NEG_INFINITY {
                            let alpha = if running_max == f32::NEG_INFINITY {
                                0.0
                            } else {
                                (running_max - score).exp()
                            };
                            running_sum = running_sum * alpha + 1.0;
                            for i in 0..head_dim {
                                out_head[i] = out_head[i] * alpha + v_head[i];
                            }
                            running_max = score;
                        } else {
                            let beta = (score - running_max).exp();
                            running_sum += beta;
                            mivi_core::vec_fmadd(out_head, beta, v_head);
                        }
                    }
                }
            }
            mivi_kv::KvPrecision::TurboQuant4 => {
                let head_tq = mivi_core::TurboQuant4Bit::new(head_dim);
                let q_lut = head_tq.build_query_lut(q_head);
                let head_bytes = head_dim / 2;
                let head_offset = kv_head * head_bytes;

                for t in 0..seq_len {
                    let (norm_k, packed_k) =
                        unsafe { kv.get_k_tq4_packed_unchecked(layer, t) };
                    let k_head_packed = &packed_k[head_offset..head_offset + head_bytes];
                    let score = head_tq.score_query_lut(&q_lut, norm_k, k_head_packed) * scale;

                    unsafe {
                        kv.get_v_tq4_dequantized_unchecked(
                            layer,
                            t,
                            &mut state.hb[..cfg.kv_dim],
                        )
                    };
                    let v_head = &state.hb[kv_head * head_dim..(kv_head + 1) * head_dim];

                    if score > f32::NEG_INFINITY {
                        if score > running_max || running_max == f32::NEG_INFINITY {
                            let alpha = if running_max == f32::NEG_INFINITY {
                                0.0
                            } else {
                                (running_max - score).exp()
                            };
                            running_sum = running_sum * alpha + 1.0;
                            for i in 0..head_dim {
                                out_head[i] = out_head[i] * alpha + v_head[i];
                            }
                            running_max = score;
                        } else {
                            let beta = (score - running_max).exp();
                            running_sum += beta;
                            mivi_core::vec_fmadd(out_head, beta, v_head);
                        }
                    }
                }
            }
            mivi_kv::KvPrecision::TurboQuant2 => {
                let head_tq = mivi_core::TurboQuant2Bit::new(head_dim);
                let q_lut = head_tq.build_query_lut(q_head);
                let head_bytes = head_dim / 4;
                let head_offset = kv_head * head_bytes;

                for t in 0..seq_len {
                    let (norm_k, packed_k) =
                        unsafe { kv.get_k_tq2_packed_unchecked(layer, t) };
                    let k_head_packed = &packed_k[head_offset..head_offset + head_bytes];
                    let score = head_tq.score_query_lut(&q_lut, norm_k, k_head_packed) * scale;

                    unsafe {
                        kv.get_v_tq2_dequantized_unchecked(
                            layer,
                            t,
                            &mut state.hb[..cfg.kv_dim],
                        )
                    };
                    let v_head = &state.hb[kv_head * head_dim..(kv_head + 1) * head_dim];

                    if score > f32::NEG_INFINITY {
                        if score > running_max || running_max == f32::NEG_INFINITY {
                            let alpha = if running_max == f32::NEG_INFINITY {
                                0.0
                            } else {
                                (running_max - score).exp()
                            };
                            running_sum = running_sum * alpha + 1.0;
                            for i in 0..head_dim {
                                out_head[i] = out_head[i] * alpha + v_head[i];
                            }
                            running_max = score;
                        } else {
                            let beta = (score - running_max).exp();
                            running_sum += beta;
                            mivi_core::vec_fmadd(out_head, beta, v_head);
                        }
                    }
                }
            }
        }

        if running_sum > 0.0 {
            let inv_sum = 1.0 / running_sum;
            for v in out_head.iter_mut() {
                *v *= inv_sum;
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
    rms_norm_simd(&mut state.xb, &state.x, &w.attn_norm, cfg.rms_norm_eps);

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
        eps: cfg.rms_norm_eps,
    };
    ffn_swiglu_forward(state, &ffn_params)
}
