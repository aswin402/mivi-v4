//! Default-off, checked F32 query-head scheduling experiment.

use super::*;
use rayon::prelude::*;

pub(super) fn compute(
    state: &mut RunState,
    kv: &KvCache,
    layer: usize,
    pos: usize,
    cfg: &ModelConfig,
) -> Result<()> {
    let invalid =
        || ModelError::DimMismatch("invalid parallel F32 attention shape or position".into());
    let query_len = cfg.n_heads.checked_mul(cfg.head_dim).ok_or_else(invalid)?;
    let kv_len = cfg
        .n_kv_heads
        .checked_mul(cfg.head_dim)
        .ok_or_else(invalid)?;
    if cfg.n_heads == 0
        || cfg.n_kv_heads == 0
        || cfg.head_dim == 0
        || !cfg.n_heads.is_multiple_of(cfg.n_kv_heads)
        || cfg.dim != query_len
        || cfg.kv_dim != kv_len
        || state.q.len() < query_len
        || state.attn_out.len() < query_len
        || kv.precision() != mivi_kv::KvPrecision::F32
        || pos >= cfg.max_seq_len
        || pos >= kv.current_pos()
    {
        return Err(invalid());
    }
    // Checked access validates the selective layer mapping, capacity and actual KV
    // width before any output is touched. Intermediate positions share that layout.
    for t in [0, pos] {
        if kv.get_k(layer, t)?.len() != kv_len || kv.get_v(layer, t)?.len() != kv_len {
            return Err(invalid());
        }
    }
    let workers = rayon::current_num_threads().min(cfg.n_heads).max(1);
    let heads_per_task = cfg.n_heads.div_ceil(workers);
    let task_width = heads_per_task
        .checked_mul(cfg.head_dim)
        .ok_or_else(invalid)?;
    let queries = &state.q[..query_len];
    state.attn_out[..query_len]
        .par_chunks_mut(task_width)
        .enumerate()
        .try_for_each(|(task, outputs)| {
            let first_head = task * heads_per_task;
            for (offset, output) in outputs.chunks_exact_mut(cfg.head_dim).enumerate() {
                let h = first_head + offset;
                let query = &queries[h * cfg.head_dim..(h + 1) * cfg.head_dim];
                compute_head(output, query, kv, layer, pos, h, cfg)?;
            }
            Ok(())
        })
}

// Keep the production serial loop independent as the bit-pattern oracle. This
// copy preserves each head's ascending scan and arithmetic; only scheduling changes.
fn compute_head(
    out_head: &mut [f32],
    q_head: &[f32],
    kv: &KvCache,
    layer: usize,
    pos: usize,
    h: usize,
    cfg: &ModelConfig,
) -> Result<()> {
    let head_dim = cfg.head_dim;
    let kv_head = h / (cfg.n_heads / cfg.n_kv_heads);
    let scale = 1.0 / (head_dim as f32).sqrt();
    out_head.fill(0.0);
    let mut running_max = f32::NEG_INFINITY;
    let mut running_sum = 0.0f32;
    for t in 0..=pos {
        let k_cached = kv.get_k(layer, t)?;
        let k_head = &k_cached[kv_head * head_dim..(kv_head + 1) * head_dim];
        let score = dot_product(q_head, k_head) * scale;
        let v_cached = kv.get_v(layer, t)?;
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
    if running_sum > 0.0 {
        let inv_sum = 1.0 / running_sum;
        for v in out_head.iter_mut() {
            *v *= inv_sum;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
