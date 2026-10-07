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

#[cfg(all(test, feature = "q4-cached-sums-experiment"))]
mod cached_sums_tests {
    use super::*;
    use mivi_core::arena::ArenaConfig;
    use mivi_quant::GgmlType;

    #[test]
    fn cached_sums_ffn_dispatch_and_exact_parity() {
        let dim = 256;
        let hidden_dim = 512;
        for (up_type, with_lora) in [
            (GgmlType::Q4_K, false),
            (GgmlType::F32, false),
            (GgmlType::Q4_K, true),
            (GgmlType::F32, true),
        ] {
            let mut mmap = Vec::new();
            let mut tensor = |rows: usize, cols: usize, quant_type: GgmlType| {
                let offset = mmap.len();
                if quant_type == GgmlType::Q4_K {
                    for seed in 0..rows * cols / 256 {
                        let mut block = vec![0u8; 144];
                        block[..2].copy_from_slice(&[0x00, 0x14]); // finite tiny f16 scale
                        block[2..4].copy_from_slice(&[0x00, 0x10]);
                        for (i, b) in block[4..].iter_mut().enumerate() {
                            *b = (i * 53 + seed * 17) as u8;
                        }
                        mmap.extend(block);
                    }
                } else {
                    for _ in 0..rows * cols {
                        mmap.extend(0.001f32.to_le_bytes());
                    }
                }
                QuantizedTensor {
                    quant_type,
                    offset,
                    len: mmap.len() - offset,
                    rows,
                    cols,
                }
            };
            let weights = FfnLayerWeights {
                ffn_norm: vec![1.0; dim].into_boxed_slice(),
                w_gate: tensor(hidden_dim, dim, GgmlType::Q4_K),
                w_up: tensor(hidden_dim, dim, up_type),
                w_down: tensor(dim, hidden_dim, GgmlType::Q4_K),
                ffn_gate_name: "gate".into(),
                ffn_up_name: "up".into(),
                ffn_down_name: "down".into(),
            };
            let cfg = ArenaConfig {
                dim,
                hidden_dim,
                n_layers: 1,
                n_heads: 1,
                n_kv_heads: 1,
                head_dim: dim,
                kv_dim: dim,
                vocab_size: 8,
                max_seq_len: 4,
                ssm_state_dim: 0,
                ssm_conv_kernel: 0,
                max_lora_rank: 2,
                n_experts: 0,
            };
            let mut baseline = RunState::new(&cfg);
            let mut candidate = RunState::new(&cfg);
            candidate.q4_cached_sums_enabled = true;
            let mut adapters = ActiveAdapters::new();
            if with_lora {
                let mut adapter = crate::lora::LoraAdapter::new("synthetic");
                for (name, input, output) in [
                    ("gate", dim, hidden_dim),
                    ("up", dim, hidden_dim),
                    ("down", hidden_dim, dim),
                ] {
                    let mut pair = crate::lora::LoraWeightPair::new(2, 2.0, input, output);
                    pair.a.fill(0.002);
                    pair.b.fill(0.003);
                    adapter.add_weight_pair(name, pair);
                }
                adapters.add(adapter, 0.75);
            }
            let params = FfnSwigluParams {
                weights: &weights,
                dim,
                hidden_dim,
                mmap: &mmap,
                adapters: &adapters,
                eps: 1e-5,
            };
            for seed in [3, 19] {
                for (i, x) in baseline.x.iter_mut().enumerate() {
                    *x = ((i * 37 + seed) % 113) as f32 / 17.0 - 3.0;
                }
                candidate.x.copy_from_slice(&baseline.x);
                candidate.q4_cached_sums_calls = 0;
                ffn_swiglu_forward(&mut baseline, &params).unwrap();
                ffn_swiglu_forward(&mut candidate, &params).unwrap();
                assert_eq!(
                    candidate.q4_cached_sums_calls,
                    if up_type == GgmlType::Q4_K { 3 } else { 2 }
                );
                assert_eq!(baseline.q4_cached_sums_calls, 0);
                assert!(candidate.x.iter().all(|x| x.is_finite()));
                assert_eq!(
                    candidate.x.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    baseline.x.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
                assert_eq!(candidate.lora_down, baseline.lora_down);
                if with_lora {
                    assert!(candidate.lora_down.iter().any(|&x| x != 0.0));
                }
            }
        }
    }
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

/// Experimental FFN-only route. The compiled feature alone does not select it.
#[cfg(feature = "q4-cached-sums-experiment")]
#[inline]
fn linear_forward_experimental(
    out: &mut [f32],
    params: &LinearParams,
    lora_down: &mut [f32],
    sums: &mut [f32],
    enabled: bool,
) -> Result<usize> {
    if enabled && params.weight.quant_type == mivi_quant::GgmlType::Q4_K {
        mivi_quant::q4_k_m::try_matvec_q4_k_m_cached(
            out,
            params.weight.as_slice(params.mmap),
            params.input,
            params.rows,
            params.cols,
            sums,
        )?;
        params
            .adapters
            .apply_module(params.module_name, params.input, lora_down, out);
        Ok(1)
    } else {
        linear_forward(out, params, lora_down)?;
        Ok(0)
    }
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
    #[cfg(not(feature = "q4-cached-sums-experiment"))]
    linear_forward(&mut state.hb, &gate_params, &mut state.lora_down)?;
    #[cfg(feature = "q4-cached-sums-experiment")]
    {
        state.q4_cached_sums_calls += linear_forward_experimental(
            &mut state.hb,
            &gate_params,
            &mut state.lora_down,
            &mut state.q4_activation_sums,
            state.q4_cached_sums_enabled,
        )?;
    }

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
    #[cfg(not(feature = "q4-cached-sums-experiment"))]
    linear_forward(&mut state.hb2, &up_params, &mut state.lora_down)?;
    #[cfg(feature = "q4-cached-sums-experiment")]
    {
        state.q4_cached_sums_calls += linear_forward_experimental(
            &mut state.hb2,
            &up_params,
            &mut state.lora_down,
            &mut state.q4_activation_sums,
            state.q4_cached_sums_enabled,
        )?;
    }

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
    #[cfg(not(feature = "q4-cached-sums-experiment"))]
    linear_forward(&mut state.xb, &down_params, &mut state.lora_down)?;
    #[cfg(feature = "q4-cached-sums-experiment")]
    {
        state.q4_cached_sums_calls += linear_forward_experimental(
            &mut state.xb,
            &down_params,
            &mut state.lora_down,
            &mut state.q4_activation_sums,
            state.q4_cached_sums_enabled,
        )?;
    }

    // 6. Residual connection
    vec_add(&mut state.x, &state.xb);
    Ok(())
}
