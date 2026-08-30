//! State Space Model (SSM) layer with diagonal recurrence and depthwise 1D conv.

use crate::config::{ModelConfig, DEFAULT_RMS_NORM_EPS, DEFAULT_SSM_A_VAL};
use crate::ffn::{ffn_swiglu_forward, linear_forward, FfnSwigluParams, LinearParams};
use crate::lora::ActiveAdapters;
use crate::model::Result;
use crate::weights::SsmLayerWeights;
use mivi_core::arena::RunState;
use mivi_core::math::{silu, vec_add};
use mivi_core::simd::rms_norm_simd;

/// Fallback depthwise convolution weight when conv weights are absent.
const DEFAULT_CONV_WEIGHT: f32 = 0.25;

/// Parameter descriptor for SSM forward pass.
pub struct SsmParams<'a> {
    pub layer: usize,
    pub weights: &'a SsmLayerWeights,
    pub mmap: &'a [u8],
    pub config: &'a ModelConfig,
    pub adapters: &'a ActiveAdapters,
}

/// Apply short depthwise 1D convolution over sequence history.
#[inline]
fn apply_depthwise_conv(
    layer: usize,
    state: &mut RunState,
    conv_weights: &[f32],
    dim: usize,
    kernel_size: usize,
) {
    if conv_weights.is_empty() || kernel_size == 0 {
        return;
    }
    let conv_layer_offset = layer * dim * kernel_size;
    for d in 0..dim {
        let conv_offset = conv_layer_offset + d * kernel_size;
        // Shift history left
        for k in 0..(kernel_size - 1) {
            state.conv_states[conv_offset + k] = state.conv_states[conv_offset + k + 1];
        }
        state.conv_states[conv_offset + (kernel_size - 1)] = state.xb2[d];

        // 1D depthwise convolution dot product
        let mut conv_out = 0.0f32;
        for k in 0..kernel_size {
            let w_k = if conv_weights.len() == dim * kernel_size {
                conv_weights[d * kernel_size + k]
            } else if k < conv_weights.len() {
                conv_weights[k]
            } else {
                DEFAULT_CONV_WEIGHT
            };
            conv_out += state.conv_states[conv_offset + k] * w_k;
        }
        // SiLU activation after short convolution
        state.xb2[d] = mivi_core::math::silu_scalar(conv_out);
    }
}

/// Apply diagonal SSM recurrence: h_t = A * h_{t-1} + in_t
#[inline]
fn apply_ssm_recurrence(
    layer: usize,
    state: &mut RunState,
    ssm_a: &[f32],
    dim: usize,
    state_dim: usize,
) {
    let state_offset = layer * state_dim;
    let ssm_state = &mut state.ssm_states[state_offset..state_offset + state_dim];
    let effective_dim = std::cmp::min(state_dim, dim);
    for (i, state_val) in ssm_state.iter_mut().enumerate().take(effective_dim) {
        let a = if i < ssm_a.len() {
            ssm_a[i]
        } else {
            DEFAULT_SSM_A_VAL
        };
        *state_val = a * *state_val + state.xb2[i];
    }
    state.xb2[..effective_dim].copy_from_slice(&ssm_state[..effective_dim]);
}

/// Forward pass through an SSM / Gated Conv block.
pub fn ssm_forward(state: &mut RunState, params: &SsmParams) -> Result<()> {
    let cfg = params.config;
    let w = params.weights;
    let layer = params.layer;
    let mmap = params.mmap;
    let adapters = params.adapters;
    let dim = cfg.dim;
    let hidden_dim = cfg.hidden_dim;
    let state_dim = cfg.ssm_state_dim;
    let kernel_size = cfg.ssm_conv_kernel;

    // 1. Pre-Norm: xb = rms_norm(x, ssm_norm) (SIMD accelerated)
    rms_norm_simd(&mut state.xb, &state.x, &w.ssm_norm, DEFAULT_RMS_NORM_EPS);

    // 2. In-projection: xb2 = W_in * xb + LoRA
    let in_params = LinearParams {
        weight: &w.in_proj,
        input: &state.xb,
        rows: dim,
        cols: dim,
        mmap,
        adapters,
        module_name: &w.in_name,
    };
    linear_forward(&mut state.xb2, &in_params, &mut state.lora_down)?;

    // 3. Short depthwise 1D convolution
    apply_depthwise_conv(layer, state, &w.ssm_conv, dim, kernel_size);

    // 4. Recurrence update: h_t = A * h_{t-1} + in_t
    apply_ssm_recurrence(layer, state, &w.ssm_a, dim, state_dim);

    // 5. Output projection: xb = W_out * xb2 + LoRA
    let out_params = LinearParams {
        weight: &w.out_proj,
        input: &state.xb2,
        rows: dim,
        cols: dim,
        mmap,
        adapters,
        module_name: &w.out_name,
    };
    linear_forward(&mut state.xb, &out_params, &mut state.lora_down)?;
    silu(&mut state.xb);

    // 6. Residual connection: x = x + xb
    vec_add(&mut state.x, &state.xb);

    // 7-10. Shared FFN SwiGLU forward
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
