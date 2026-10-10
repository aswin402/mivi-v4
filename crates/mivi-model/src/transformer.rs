//! Grouped Query Attention (GQA) Transformer layer implementation with RoPE.

use crate::config::ModelConfig;
use crate::ffn::{ffn_swiglu_forward, linear_forward, FfnSwigluParams, LinearParams};
use crate::lora::ActiveAdapters;
use crate::model::{ModelError, Result};
use crate::prefill::{add_rows_in_place, rms_norm_rows, swiglu_rows, TileActivations};
use crate::weights::AttentionLayerWeights;
use mivi_core::arena::RunState;
use mivi_core::math::{dot_product, vec_add};
use mivi_core::simd::rms_norm_simd;
use mivi_kv::KvCache;
use mivi_quant::quantized_matmul_rows;
use std::time::{Duration, Instant};

#[cfg(feature = "parallel-attention-experiment")]
mod parallel_attention;

/// Optional timings for the stages within an attention block.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AttentionStageProfile {
    pub norm: Duration,
    pub qkv_projection: Duration,
    /// Includes per-head norm, RoPE, KV insertion, and the causal attention scan.
    pub causal_attention: Duration,
    /// Includes output projection and its residual connection.
    pub output_projection: Duration,
    pub ffn: Duration,
}

impl AttentionStageProfile {
    pub fn total(self) -> Duration {
        self.norm + self.qkv_projection + self.causal_attention + self.output_projection + self.ffn
    }

    pub fn add_assign(&mut self, other: Self) {
        self.norm += other.norm;
        self.qkv_projection += other.qkv_projection;
        self.causal_attention += other.causal_attention;
        self.output_projection += other.output_projection;
        self.ffn += other.ffn;
    }
}

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
fn compute_qkv(
    state: &mut RunState,
    kv: &mut KvCache,
    params: &AttentionParams,
    mut profile: Option<&mut AttentionStageProfile>,
) -> Result<()> {
    let cfg = params.config;
    let w = params.weights;
    let dim = cfg.dim;
    let kv_dim = cfg.kv_dim;
    let mmap = params.mmap;
    let adapters = params.adapters;

    let stage_start = profile.is_some().then(Instant::now);
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
    if let (Some(profile), Some(start)) = (profile.as_deref_mut(), stage_start) {
        profile.qkv_projection += start.elapsed();
    }

    let stage_start = profile.is_some().then(Instant::now);
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
    if let (Some(profile), Some(start)) = (profile, stage_start) {
        profile.causal_attention += start.elapsed();
    }
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
    #[cfg(feature = "parallel-attention-experiment")]
    if state.parallel_attention_enabled
        && kv.precision() == mivi_kv::KvPrecision::F32
        && cfg.n_heads > 1
        && rayon::current_num_threads() > 1
    {
        parallel_attention::compute(state, kv, layer, pos, cfg)?;
        state.parallel_attention_calls = state.parallel_attention_calls.saturating_add(1);
        return Ok(());
    }
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
                let blocks_per_head = head_dim.div_ceil(32);
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
                    let (norm_k, packed_k) = unsafe { kv.get_k_tq4_packed_unchecked(layer, t) };
                    let k_head_packed = &packed_k[head_offset..head_offset + head_bytes];
                    let score = head_tq.score_query_lut(&q_lut, norm_k, k_head_packed) * scale;

                    unsafe {
                        kv.get_v_tq4_dequantized_unchecked(layer, t, &mut state.hb[..cfg.kv_dim])
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
                    let (norm_k, packed_k) = unsafe { kv.get_k_tq2_packed_unchecked(layer, t) };
                    let k_head_packed = &packed_k[head_offset..head_offset + head_bytes];
                    let score = head_tq.score_query_lut(&q_lut, norm_k, k_head_packed) * scale;

                    unsafe {
                        kv.get_v_tq2_dequantized_unchecked(layer, t, &mut state.hb[..cfg.kv_dim])
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
    attention_forward_profiled(state, kv, params, None)
}

/// Run the same single-token arithmetic with optional stage timing.
pub(crate) fn attention_forward_profiled(
    state: &mut RunState,
    kv: &mut KvCache,
    params: &AttentionParams,
    mut profile: Option<&mut AttentionStageProfile>,
) -> Result<()> {
    let cfg = params.config;
    let w = params.weights;
    let dim = cfg.dim;
    let hidden_dim = cfg.hidden_dim;
    let mmap = params.mmap;
    let adapters = params.adapters;

    // 1. Attention Pre-Norm (SIMD accelerated)
    let stage_start = profile.is_some().then(Instant::now);
    rms_norm_simd(&mut state.xb, &state.x, &w.attn_norm, cfg.rms_norm_eps);
    if let (Some(profile), Some(start)) = (profile.as_deref_mut(), stage_start) {
        profile.norm += start.elapsed();
    }

    // 2-4. Q, K, V projections + RoPE + Cache store
    compute_qkv(state, kv, params, profile.as_deref_mut())?;

    // 5. Multi-head Attention with GQA
    let stage_start = profile.is_some().then(Instant::now);
    compute_gqa_attention(state, kv, params.layer, params.pos, cfg)?;
    if let (Some(profile), Some(start)) = (profile.as_deref_mut(), stage_start) {
        profile.causal_attention += start.elapsed();
    }

    // 6. Output projection: xb = W_o * attn_out + LoRA
    let stage_start = profile.is_some().then(Instant::now);
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
    if let (Some(profile), Some(start)) = (profile.as_deref_mut(), stage_start) {
        profile.output_projection += start.elapsed();
    }

    // 8-11. Shared FFN SwiGLU forward
    let stage_start = profile.is_some().then(Instant::now);
    let ffn_params = FfnSwigluParams {
        weights: &w.ffn,
        dim,
        hidden_dim,
        mmap,
        adapters,
        eps: cfg.rms_norm_eps,
    };
    ffn_swiglu_forward(state, &ffn_params)?;
    if let (Some(profile), Some(start)) = (profile, stage_start) {
        profile.ffn += start.elapsed();
    }
    Ok(())
}

/// Forward an attention block over a prompt tile.
///
/// Q/K/V projections are computed for all rows together. KV insertion and
/// causal attention remain ordered so each query sees only its valid prefix.
pub fn attention_forward_tile(
    tile: &mut TileActivations,
    state: &mut RunState,
    kv: &mut KvCache,
    params: &AttentionParams,
    start_pos: usize,
    rows: usize,
) -> Result<()> {
    attention_forward_tile_profiled(tile, state, kv, params, start_pos, rows, None)
}

/// Same tile computation with optional stage timing; no timestamps are created
/// when profiling is disabled.
pub(crate) fn attention_forward_tile_profiled(
    tile: &mut TileActivations,
    state: &mut RunState,
    kv: &mut KvCache,
    params: &AttentionParams,
    start_pos: usize,
    rows: usize,
    mut profile: Option<&mut AttentionStageProfile>,
) -> Result<()> {
    if rows == 0 || rows > tile.tile_tokens() {
        return Err(ModelError::DimMismatch(format!(
            "invalid attention tile row count: {rows}"
        )));
    }
    if !params.adapters.active.is_empty() {
        return Err(ModelError::ExecutionFailed(
            "chunked attention prefill does not support active LoRA adapters yet".to_string(),
        ));
    }

    let cfg = params.config;
    let w = params.weights;
    let dim = cfg.dim;
    let kv_dim = cfg.kv_dim;
    let hidden_dim = cfg.hidden_dim;
    let dim_rows = rows.checked_mul(dim).ok_or(ModelError::ExecutionFailed(
        "attention tile size overflow".to_string(),
    ))?;
    let kv_rows = rows.checked_mul(kv_dim).ok_or(ModelError::ExecutionFailed(
        "attention KV tile size overflow".to_string(),
    ))?;
    let hidden_rows = rows
        .checked_mul(hidden_dim)
        .ok_or(ModelError::ExecutionFailed(
            "attention FFN tile size overflow".to_string(),
        ))?;

    let stage_start = profile.is_some().then(Instant::now);
    rms_norm_rows(
        &mut tile.norm[..dim_rows],
        &tile.current[..dim_rows],
        &w.attn_norm,
        rows,
        dim,
        cfg.rms_norm_eps,
    )
    .map_err(|error| ModelError::ExecutionFailed(error.to_string()))?;
    if let (Some(profile), Some(start)) = (profile.as_deref_mut(), stage_start) {
        profile.norm += start.elapsed();
    }
    let stage_start = profile.is_some().then(Instant::now);
    quantized_matmul_rows(
        &mut tile.q[..dim_rows],
        w.wq.quant_type,
        w.wq.as_slice(params.mmap),
        &tile.norm[..dim_rows],
        rows,
        dim,
        dim,
    )?;
    quantized_matmul_rows(
        &mut tile.k[..kv_rows],
        w.wk.quant_type,
        w.wk.as_slice(params.mmap),
        &tile.norm[..dim_rows],
        rows,
        kv_dim,
        dim,
    )?;
    quantized_matmul_rows(
        &mut tile.v[..kv_rows],
        w.wv.quant_type,
        w.wv.as_slice(params.mmap),
        &tile.norm[..dim_rows],
        rows,
        kv_dim,
        dim,
    )?;
    if let (Some(profile), Some(start)) = (profile.as_deref_mut(), stage_start) {
        profile.qkv_projection += start.elapsed();
    }
    let stage_start = profile.is_some().then(Instant::now);
    for row in 0..rows {
        let pos = start_pos + row;
        let q_start = row * dim;
        let k_start = row * kv_dim;
        let q_row = &mut tile.q[q_start..q_start + dim];
        let k_row = &mut tile.k[k_start..k_start + kv_dim];
        let v_row = &tile.v[k_start..k_start + kv_dim];

        if let Some(q_norm) = &w.q_norm {
            for head in q_row.chunks_exact_mut(cfg.head_dim) {
                mivi_core::rms_norm_in_place_simd(head, q_norm, cfg.rms_norm_eps);
            }
        }
        if let Some(k_norm) = &w.k_norm {
            for head in k_row.chunks_exact_mut(cfg.head_dim) {
                mivi_core::rms_norm_in_place_simd(head, k_norm, cfg.rms_norm_eps);
            }
        }
        params
            .rope
            .apply(q_row, k_row, pos, cfg.n_heads, cfg.n_kv_heads);
        kv.store(params.layer, pos, k_row, v_row)?;

        state.q.copy_from_slice(q_row);
        state.k.copy_from_slice(k_row);
        state.v.copy_from_slice(v_row);
        compute_gqa_attention(state, kv, params.layer, pos, cfg)?;
        tile.norm[q_start..q_start + dim].copy_from_slice(&state.attn_out);
    }
    if let (Some(profile), Some(start)) = (profile.as_deref_mut(), stage_start) {
        profile.causal_attention += start.elapsed();
    }
    let stage_start = profile.is_some().then(Instant::now);
    quantized_matmul_rows(
        &mut tile.next[..dim_rows],
        w.wo.quant_type,
        w.wo.as_slice(params.mmap),
        &tile.norm[..dim_rows],
        rows,
        dim,
        dim,
    )?;
    add_rows_in_place(
        &mut tile.current[..dim_rows],
        &tile.next[..dim_rows],
        rows,
        dim,
    )
    .map_err(|error| ModelError::ExecutionFailed(error.to_string()))?;
    if let (Some(profile), Some(start)) = (profile.as_deref_mut(), stage_start) {
        profile.output_projection += start.elapsed();
    }
    let stage_start = profile.is_some().then(Instant::now);
    rms_norm_rows(
        &mut tile.norm[..dim_rows],
        &tile.current[..dim_rows],
        &w.ffn.ffn_norm,
        rows,
        dim,
        cfg.rms_norm_eps,
    )
    .map_err(|error| ModelError::ExecutionFailed(error.to_string()))?;
    quantized_matmul_rows(
        &mut tile.gate[..hidden_rows],
        w.ffn.w_gate.quant_type,
        w.ffn.w_gate.as_slice(params.mmap),
        &tile.norm[..dim_rows],
        rows,
        hidden_dim,
        dim,
    )?;
    quantized_matmul_rows(
        &mut tile.up[..hidden_rows],
        w.ffn.w_up.quant_type,
        w.ffn.w_up.as_slice(params.mmap),
        &tile.norm[..dim_rows],
        rows,
        hidden_dim,
        dim,
    )?;
    swiglu_rows(
        &mut tile.gate[..hidden_rows],
        &tile.up[..hidden_rows],
        rows,
        hidden_dim,
    )
    .map_err(|error| ModelError::ExecutionFailed(error.to_string()))?;
    quantized_matmul_rows(
        &mut tile.next[..dim_rows],
        w.ffn.w_down.quant_type,
        w.ffn.w_down.as_slice(params.mmap),
        &tile.gate[..hidden_rows],
        rows,
        dim,
        hidden_dim,
    )?;
    add_rows_in_place(
        &mut tile.current[..dim_rows],
        &tile.next[..dim_rows],
        rows,
        dim,
    )
    .map_err(|error| ModelError::ExecutionFailed(error.to_string()))?;

    if let (Some(profile), Some(start)) = (profile, stage_start) {
        profile.ffn += start.elapsed();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BlockType;
    use crate::prefill::TileActivations;
    use crate::weights::{FfnLayerWeights, QuantizedTensor};
    use mivi_core::arena::{ArenaConfig, RunState};
    use mivi_quant::GgmlType;

    fn f32_tensor(mmap: &mut Vec<u8>, rows: usize, cols: usize, values: &[f32]) -> QuantizedTensor {
        assert_eq!(values.len(), rows * cols);
        let offset = mmap.len();
        for value in values {
            mmap.extend_from_slice(&value.to_le_bytes());
        }
        QuantizedTensor {
            quant_type: GgmlType::F32,
            offset,
            len: values.len() * 4,
            rows,
            cols,
        }
    }

    fn identity_matrix(dim: usize) -> Vec<f32> {
        let mut matrix = vec![0.0; dim * dim];
        for idx in 0..dim {
            matrix[idx * dim + idx] = 1.0;
        }
        matrix
    }

    #[test]
    fn attention_stage_profile_sums_and_accumulates() {
        use std::time::Duration;
        let stages = AttentionStageProfile {
            norm: Duration::from_millis(1),
            qkv_projection: Duration::from_millis(2),
            causal_attention: Duration::from_millis(3),
            output_projection: Duration::from_millis(4),
            ffn: Duration::from_millis(5),
        };
        assert_eq!(stages.total(), Duration::from_millis(15));
        let mut aggregate = AttentionStageProfile::default();
        aggregate.add_assign(stages);
        aggregate.add_assign(stages);
        assert_eq!(aggregate.total(), Duration::from_millis(30));
        assert_eq!(aggregate.causal_attention, Duration::from_millis(6));
    }

    #[test]
    fn attention_tile_matches_token_path_and_kv_state() {
        let dim = 4;
        let hidden_dim = 8;
        let cfg = ModelConfig {
            name: "attention-tile-test".to_string(),
            dim,
            hidden_dim,
            n_layers: 1,
            n_heads: 1,
            n_kv_heads: 1,
            head_dim: dim,
            kv_dim: dim,
            vocab_size: 32,
            max_seq_len: 16,
            rope_base: 10_000.0,
            rms_norm_eps: 1e-5,
            ssm_state_dim: 1,
            ssm_conv_kernel: 3,
            block_types: vec![BlockType::Attention],
        };
        let mut mmap = Vec::new();
        let identity = identity_matrix(dim);
        let q = f32_tensor(&mut mmap, dim, dim, &identity);
        let k = f32_tensor(&mut mmap, dim, dim, &identity);
        let v = f32_tensor(&mut mmap, dim, dim, &identity);
        let wo = f32_tensor(&mut mmap, dim, dim, &identity);
        let ffn = FfnLayerWeights {
            ffn_norm: vec![1.0; dim].into_boxed_slice(),
            w_gate: f32_tensor(&mut mmap, hidden_dim, dim, &vec![0.0; hidden_dim * dim]),
            w_up: f32_tensor(&mut mmap, hidden_dim, dim, &vec![0.0; hidden_dim * dim]),
            w_down: f32_tensor(&mut mmap, dim, hidden_dim, &vec![0.0; dim * hidden_dim]),
            ffn_gate_name: "ffn_gate".to_string(),
            ffn_up_name: "ffn_up".to_string(),
            ffn_down_name: "ffn_down".to_string(),
        };
        let weights = AttentionLayerWeights {
            attn_norm: vec![1.0; dim].into_boxed_slice(),
            q_norm: None,
            k_norm: None,
            wq: q,
            wk: k,
            wv: v,
            wo,
            ffn,
            q_name: "q".to_string(),
            k_name: "k".to_string(),
            v_name: "v".to_string(),
            o_name: "o".to_string(),
        };
        let adapters = ActiveAdapters::new();
        let rope = mivi_core::RopeCache::new(cfg.head_dim, cfg.max_seq_len, cfg.rope_base);
        let arena = ArenaConfig {
            dim,
            hidden_dim,
            n_layers: 1,
            n_heads: 1,
            n_kv_heads: 1,
            head_dim: dim,
            kv_dim: dim,
            vocab_size: cfg.vocab_size,
            max_seq_len: cfg.max_seq_len,
            ssm_state_dim: cfg.ssm_state_dim,
            ssm_conv_kernel: cfg.ssm_conv_kernel,
            max_lora_rank: 4,
            n_experts: 0,
        };
        let params = AttentionParams {
            layer: 0,
            pos: 0,
            weights: &weights,
            mmap: &mmap,
            config: &cfg,
            adapters: &adapters,
            rope: &rope,
        };
        let inputs = [
            [1.0, 0.5, -1.0, 2.0],
            [0.25, -2.0, 1.0, 0.5],
            [2.0, 1.0, 0.0, -0.5],
        ];

        let mut token_state = RunState::new(&arena);
        let mut token_kv = KvCache::try_new_selective(1, cfg.max_seq_len, dim, &[0]).unwrap();
        let mut expected = Vec::new();
        for (pos, input) in inputs.iter().enumerate() {
            token_state.x.copy_from_slice(input);
            let token_params = AttentionParams { pos, ..params };
            attention_forward(&mut token_state, &mut token_kv, &token_params).unwrap();
            expected.push(token_state.x.to_vec());
        }

        let mut tile = TileActivations::with_kv_dim(inputs.len(), dim, hidden_dim, dim).unwrap();
        for (row, input) in inputs.iter().enumerate() {
            tile.current_row_mut(row).unwrap().copy_from_slice(input);
        }
        let mut tile_state = RunState::new(&arena);
        let mut tile_kv = KvCache::try_new_selective(1, cfg.max_seq_len, dim, &[0]).unwrap();
        attention_forward_tile(
            &mut tile,
            &mut tile_state,
            &mut tile_kv,
            &params,
            0,
            inputs.len(),
        )
        .unwrap();

        for (row, expected_row) in expected.iter().enumerate() {
            let actual = tile.current_row(row).unwrap();
            for (actual, expected) in actual.iter().zip(expected_row) {
                assert!((actual - expected).abs() < 1e-5);
            }
        }
        let (token_k, token_v) = token_kv.export_state(inputs.len()).unwrap();
        let (tile_k, tile_v) = tile_kv.export_state(inputs.len()).unwrap();
        assert_eq!(token_k, tile_k);
        assert_eq!(token_v, tile_v);

        let mut profiled_token_state = RunState::new(&arena);
        let mut profiled_token_kv =
            KvCache::try_new_selective(1, cfg.max_seq_len, dim, &[0]).unwrap();
        let mut token_profile = AttentionStageProfile::default();
        for (pos, input) in inputs.iter().enumerate() {
            profiled_token_state.x.copy_from_slice(input);
            let token_params = AttentionParams { pos, ..params };
            attention_forward_profiled(
                &mut profiled_token_state,
                &mut profiled_token_kv,
                &token_params,
                Some(&mut token_profile),
            )
            .unwrap();
            assert_eq!(profiled_token_state.x.as_ref(), expected[pos].as_slice());
        }
        assert_eq!(
            profiled_token_kv.export_state(inputs.len()).unwrap(),
            (token_k, token_v)
        );
        assert!(!token_profile.qkv_projection.is_zero());
        assert!(!token_profile.causal_attention.is_zero());
        assert!(!token_profile.ffn.is_zero());

        let expected_tile = tile.current.clone();
        for (row, input) in inputs.iter().enumerate() {
            tile.current_row_mut(row).unwrap().copy_from_slice(input);
        }
        let mut profiled_state = RunState::new(&arena);
        let mut profiled_kv = KvCache::try_new_selective(1, cfg.max_seq_len, dim, &[0]).unwrap();
        let mut profile = AttentionStageProfile::default();
        attention_forward_tile_profiled(
            &mut tile,
            &mut profiled_state,
            &mut profiled_kv,
            &params,
            0,
            inputs.len(),
            Some(&mut profile),
        )
        .unwrap();
        assert_eq!(tile.current, expected_tile);
        assert_eq!(
            profiled_kv.export_state(inputs.len()).unwrap(),
            (tile_k, tile_v)
        );
        assert!(!profile.total().is_zero());
    }
}
