//! State Space Model (SSM) / Gated Short Convolution layer.

use crate::config::{ModelConfig, DEFAULT_RMS_NORM_EPS};
use crate::ffn::{ffn_swiglu_forward, linear_forward, FfnSwigluParams, LinearParams};
use crate::lora::ActiveAdapters;
use crate::model::Result;
use crate::weights::SsmLayerWeights;
use mivi_core::arena::RunState;
use mivi_core::math::vec_add;
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

/// Forward pass through an LFM2 Gated ShortConv block.
pub fn ssm_forward(state: &mut RunState, params: &SsmParams) -> Result<()> {
    let cfg = params.config;
    let w = params.weights;
    let layer = params.layer;
    let mmap = params.mmap;
    let adapters = params.adapters;
    let dim = cfg.dim;
    let hidden_dim = cfg.hidden_dim;
    let kernel_size = if !w.ssm_conv.is_empty() && dim > 0 {
        w.ssm_conv.len() / dim
    } else {
        3
    };

    // 1. Pre-Norm: xb = rms_norm(x, ssm_norm) (SIMD accelerated)
    rms_norm_simd(&mut state.xb, &state.x, &w.ssm_norm, DEFAULT_RMS_NORM_EPS);

    // 2. In-projection: shortconv_in (3 * dim) = W_in (3*dim x dim) * xb + LoRA
    let in_rows = 3 * dim;
    let in_params = LinearParams {
        weight: &w.in_proj,
        input: &state.xb,
        rows: in_rows,
        cols: dim,
        mmap,
        adapters,
        module_name: &w.in_name,
    };
    linear_forward(&mut state.shortconv_in, &in_params, &mut state.lora_down)?;

    // 3. Chunk into B (dim), C (dim), X (dim)
    // bx[d] = B[d] * X[d]
    for d in 0..dim {
        let b_val = state.shortconv_in[d];
        let x_val = state.shortconv_in[2 * dim + d];
        state.xb2[d] = b_val * x_val;
    }

    // 4. Depthwise 1D causal convolution over sequence history
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
            let w_k = if w.ssm_conv.len() == dim * kernel_size {
                w.ssm_conv[d * kernel_size + k]
            } else if k < w.ssm_conv.len() {
                w.ssm_conv[k]
            } else {
                DEFAULT_CONV_WEIGHT
            };
            conv_out += state.conv_states[conv_offset + k] * w_k;
        }

        // Modulate with C channel: y[d] = conv_out * C[d]
        let c_val = state.shortconv_in[dim + d];
        state.xb2[d] = conv_out * c_val;
    }

    // 5. Output projection: xb = W_out (dim x dim) * xb2 + LoRA
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
