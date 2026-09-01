//! State Space Model (SSM) / Gated Short Convolution layer.

use crate::config::ModelConfig;
use crate::ffn::{ffn_swiglu_forward, linear_forward, FfnSwigluParams, LinearParams};
use crate::lora::ActiveAdapters;
use crate::model::Result;
use crate::weights::SsmLayerWeights;
use mivi_core::arena::RunState;
use mivi_core::math::vec_add;
use mivi_core::simd::rms_norm_simd;

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
    let kernel_size = cfg.ssm_conv_kernel;
    if kernel_size == 0 {
        return Ok(());
    }

    // 1. Pre-Norm: xb = rms_norm(x, ssm_norm) (SIMD accelerated)
    rms_norm_simd(&mut state.xb, &state.x, &w.ssm_norm, cfg.rms_norm_eps);

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

    // 3. Chunk into B (dim), C (dim), X (dim) and compute bx[d] = B[d] * X[d]
    let (b_slice, rest) = state.shortconv_in.split_at(dim);
    let (c_slice, x_slice) = rest.split_at(dim);
    for d in 0..dim {
        state.xb2[d] = b_slice[d] * x_slice[d];
    }

    // Depthwise 1D causal convolution on bx using persistent conv_states buffer.
    let conv_layer_offset = layer * dim * kernel_size;
    let has_full_conv = !w.ssm_conv.is_empty() && w.ssm_conv.len() >= dim * kernel_size;
    let has_shared_conv = !w.ssm_conv.is_empty() && w.ssm_conv.len() >= kernel_size;

    if kernel_size == 3 && has_full_conv {
        for (d, &c_val) in c_slice.iter().enumerate().take(dim) {
            let conv_offset = conv_layer_offset + d * 3;
            let conv_w_offset = d * 3;

            let s0 = state.conv_states[conv_offset + 1];
            let s1 = state.conv_states[conv_offset + 2];
            let s2 = state.xb2[d];

            state.conv_states[conv_offset] = s0;
            state.conv_states[conv_offset + 1] = s1;
            state.conv_states[conv_offset + 2] = s2;

            let w0 = w.ssm_conv[conv_w_offset];
            let w1 = w.ssm_conv[conv_w_offset + 1];
            let w2 = w.ssm_conv[conv_w_offset + 2];

            let conv_out = w0 * s0 + w1 * s1 + w2 * s2;
            state.xb2[d] = c_val * conv_out;
        }
    } else {
        for (d, &c_val) in c_slice.iter().enumerate().take(dim) {
            let conv_offset = conv_layer_offset + d * kernel_size;
            let mut conv_out = 0.0f32;
            let conv_w_offset = d * kernel_size;

            for k in 0..(kernel_size - 1) {
                state.conv_states[conv_offset + k] = state.conv_states[conv_offset + k + 1];
            }
            state.conv_states[conv_offset + (kernel_size - 1)] = state.xb2[d];

            if has_full_conv {
                for k in 0..kernel_size {
                    conv_out += w.ssm_conv[conv_w_offset + k] * state.conv_states[conv_offset + k];
                }
            } else if has_shared_conv {
                for k in 0..kernel_size {
                    conv_out += w.ssm_conv[k] * state.conv_states[conv_offset + k];
                }
            } else {
                let default_w = 1.0 / kernel_size as f32;
                for k in 0..kernel_size {
                    conv_out += default_w * state.conv_states[conv_offset + k];
                }
            }

            state.xb2[d] = c_val * conv_out;
        }
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
        eps: cfg.rms_norm_eps,
    };
    ffn_swiglu_forward(state, &ffn_params)
}
