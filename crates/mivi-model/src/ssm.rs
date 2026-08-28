//! State Space Model (SSM) and double-gated short convolution block forward pass.

use crate::config::ModelConfig;
use mivi_core::arena::RunState;
use mivi_core::math::{rms_norm, silu, swiglu, vec_add};
use mivi_quant::{quantized_matvec, GgmlType};

pub struct SsmWeights<'a> {
    pub ssm_norm: &'a [f32],
    pub in_proj: (GgmlType, &'a [u8]),
    pub conv_weight: &'a [f32],
    pub ssm_a: &'a [f32],
    pub ssm_b: (GgmlType, &'a [u8]),
    pub ssm_c: (GgmlType, &'a [u8]),
    pub out_proj: (GgmlType, &'a [u8]),

    pub ffn_norm: &'a [f32],
    pub ffn_gate: (GgmlType, &'a [u8]),
    pub ffn_up: (GgmlType, &'a [u8]),
    pub ffn_down: (GgmlType, &'a [u8]),
}

/// Forward pass through an SSM / Gated Conv block.
pub fn ssm_forward(
    layer: usize,
    state: &mut RunState,
    w: &SsmWeights,
    cfg: &ModelConfig,
) {
    let dim = cfg.dim;
    let hidden_dim = cfg.hidden_dim;
    let state_dim = cfg.ssm_state_dim;

    // 1. Pre-Norm: xb = rms_norm(x, ssm_norm)
    rms_norm(&mut state.xb, &state.x, w.ssm_norm, 1e-5);

    // 2. In-projection
    let _ = quantized_matvec(&mut state.xb2, w.in_proj.0, w.in_proj.1, &state.xb, dim, dim);

    // 3. Short convolution & Recurrence update
    let state_offset = layer * state_dim;
    let ssm_state = &mut state.ssm_states[state_offset..state_offset + state_dim];

    // Recurrent update: h_t = A * h_{t-1} + B * x_t, y_t = C * h_t
    for i in 0..std::cmp::min(state_dim, dim) {
        let a = if i < w.ssm_a.len() { w.ssm_a[i] } else { 0.9 };
        ssm_state[i] = a * ssm_state[i] + state.xb2[i];
    }

    // 4. Output projection
    let _ = quantized_matvec(&mut state.xb, w.out_proj.0, w.out_proj.1, &state.xb2, dim, dim);
    silu(&mut state.xb);

    // 5. Residual connection
    vec_add(&mut state.x, &state.xb);

    // 6. FFN Pre-Norm + SwiGLU
    rms_norm(&mut state.xb, &state.x, w.ffn_norm, 1e-5);
    let _ = quantized_matvec(
        &mut state.hb,
        w.ffn_gate.0,
        w.ffn_gate.1,
        &state.xb,
        hidden_dim,
        dim,
    );
    let _ = quantized_matvec(
        &mut state.hb2,
        w.ffn_up.0,
        w.ffn_up.1,
        &state.xb,
        hidden_dim,
        dim,
    );

    swiglu(&mut state.hb, &state.hb2);

    let _ = quantized_matvec(
        &mut state.xb,
        w.ffn_down.0,
        w.ffn_down.1,
        &state.hb,
        dim,
        hidden_dim,
    );

    vec_add(&mut state.x, &state.xb);
}
