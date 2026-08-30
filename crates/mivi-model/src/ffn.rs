//! Shared Feed-Forward Network (FFN) SwiGLU forward pass for Attention and SSM blocks.

use crate::lora::ActiveAdapters;
use crate::model::Result;
use crate::weights::{FfnLayerWeights, QuantizedTensor};
use mivi_core::arena::RunState;
use mivi_core::math::{swiglu, vec_add};
use mivi_core::simd::rms_norm_simd;
use mivi_quant::quantized_matvec;

/// Parameter descriptor for linear matrix-vector multiplication with dynamic LoRA.
pub struct LinearParams<'a> {
    pub weight: &'a QuantizedTensor,
    pub input: &'a [f32],
    pub rows: usize,
    pub cols: usize,
    pub mmap: &'a [u8],
    pub adapters: &'a ActiveAdapters,
    pub module_name: &'a str,
}

/// Helper function to perform quantized matrix-vector multiplication with dynamic LoRA adaptation.
#[inline]
pub fn linear_forward(out: &mut [f32], params: &LinearParams, lora_down: &mut [f32]) -> Result<()> {
    quantized_matvec(
        out,
        params.weight.quant_type,
        params.weight.as_slice(params.mmap),
        params.input,
        params.rows,
        params.cols,
    )?;
    params
        .adapters
        .apply_module(params.module_name, params.input, lora_down, out);
    Ok(())
}

/// Parameter descriptor for FFN SwiGLU forward pass.
pub struct FfnSwigluParams<'a> {
    pub weights: &'a FfnLayerWeights,
    pub dim: usize,
    pub hidden_dim: usize,
    pub mmap: &'a [u8],
    pub adapters: &'a ActiveAdapters,
    pub eps: f32,
}

/// Execute FFN SwiGLU forward pass:
/// 1. Pre-norm: xb = rms_norm(x, ffn_norm)
/// 2. Gate projection: hb = W_gate * xb + LoRA
/// 3. Up projection: hb2 = W_up * xb + LoRA
/// 4. Non-linearity: hb = swiglu(hb, hb2)
/// 5. Down projection: xb = W_down * hb + LoRA
/// 6. Residual connection: x = x + xb
pub fn ffn_swiglu_forward(state: &mut RunState, params: &FfnSwigluParams) -> Result<()> {
    let w = params.weights;
    let dim = params.dim;
    let hidden_dim = params.hidden_dim;
    let mmap = params.mmap;
    let adapters = params.adapters;

    // 1. FFN Pre-Norm (SIMD accelerated)
    rms_norm_simd(&mut state.xb, &state.x, &w.ffn_norm, params.eps);

    // 2. Gate projection
    let gate_params = LinearParams {
        weight: &w.w_gate,
        input: &state.xb,
        rows: hidden_dim,
        cols: dim,
        mmap,
        adapters,
        module_name: &w.ffn_gate_name,
    };
    linear_forward(&mut state.hb, &gate_params, &mut state.lora_down)?;

    // 3. Up projection
    let up_params = LinearParams {
        weight: &w.w_up,
        input: &state.xb,
        rows: hidden_dim,
        cols: dim,
        mmap,
        adapters,
        module_name: &w.ffn_up_name,
    };
    linear_forward(&mut state.hb2, &up_params, &mut state.lora_down)?;

    // 4. SwiGLU activation
    swiglu(&mut state.hb, &state.hb2);

    // 5. Down projection
    let down_params = LinearParams {
        weight: &w.w_down,
        input: &state.hb,
        rows: dim,
        cols: hidden_dim,
        mmap,
        adapters,
        module_name: &w.ffn_down_name,
    };
    linear_forward(&mut state.xb, &down_params, &mut state.lora_down)?;

    // 6. Residual connection
    vec_add(&mut state.x, &state.xb);
    Ok(())
}
