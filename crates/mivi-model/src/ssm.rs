//! State Space Model (SSM) / Gated Short Convolution layer.

use crate::config::ModelConfig;
use crate::ffn::{ffn_swiglu_forward, linear_forward, FfnSwigluParams, LinearParams};
use crate::lora::ActiveAdapters;
use crate::model::{ModelError, Result};
use crate::prefill::{add_rows_in_place, rms_norm_rows, swiglu_rows, TileActivations};
use crate::weights::SsmLayerWeights;
use mivi_core::arena::RunState;
use mivi_core::math::vec_add;
use mivi_core::simd::rms_norm_simd;
use mivi_quant::quantized_matmul_rows;

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

/// Forward an SSM block over a prompt tile.
///
/// Linear projections are evaluated for all tile rows together. The causal
/// convolution is still advanced in token order because each row depends on
/// the recurrent state produced by the previous row.
pub fn ssm_forward_tile(
    tile: &mut TileActivations,
    state: &mut RunState,
    params: &SsmParams,
    rows: usize,
) -> Result<()> {
    if rows == 0 || rows > tile.tile_tokens() {
        return Err(ModelError::DimMismatch(format!(
            "invalid SSM tile row count: {rows}"
        )));
    }
    if !params.adapters.active.is_empty() {
        return Err(ModelError::ExecutionFailed(
            "chunked SSM prefill does not support active LoRA adapters yet".to_string(),
        ));
    }

    let cfg = params.config;
    let w = params.weights;
    let dim = cfg.dim;
    let hidden_dim = cfg.hidden_dim;
    let kernel_size = cfg.ssm_conv_kernel;
    if kernel_size == 0 {
        return Ok(());
    }

    let dim_rows = rows.checked_mul(dim).ok_or(ModelError::ExecutionFailed(
        "SSM tile size overflow".to_string(),
    ))?;
    let in_rows = 3 * dim;
    let in_output_len = rows
        .checked_mul(in_rows)
        .ok_or(ModelError::ExecutionFailed(
            "SSM projection size overflow".to_string(),
        ))?;
    let hidden_output_len = rows
        .checked_mul(hidden_dim)
        .ok_or(ModelError::ExecutionFailed(
            "SSM FFN size overflow".to_string(),
        ))?;

    rms_norm_rows(
        &mut tile.norm[..dim_rows],
        &tile.current[..dim_rows],
        &w.ssm_norm,
        rows,
        dim,
        cfg.rms_norm_eps,
    )
    .map_err(|error| ModelError::ExecutionFailed(error.to_string()))?;

    quantized_matmul_rows(
        &mut tile.projection[..in_output_len],
        w.in_proj.quant_type,
        w.in_proj.as_slice(params.mmap),
        &tile.norm[..dim_rows],
        rows,
        in_rows,
        dim,
    )?;

    let layer_offset = params.layer * dim * kernel_size;
    let has_full_conv = !w.ssm_conv.is_empty() && w.ssm_conv.len() >= dim * kernel_size;
    let has_shared_conv = !w.ssm_conv.is_empty() && w.ssm_conv.len() >= kernel_size;

    for row_idx in 0..rows {
        let projection_start = row_idx * in_rows;
        let projection = &tile.projection[projection_start..projection_start + in_rows];
        let (b_slice, rest) = projection.split_at(dim);
        let (c_slice, x_slice) = rest.split_at(dim);
        let output_start = row_idx * dim;

        for d in 0..dim {
            let bx = b_slice[d] * x_slice[d];
            let conv_offset = layer_offset + d * kernel_size;
            let conv_out = if kernel_size == 3 && has_full_conv {
                let s0 = state.conv_states[conv_offset + 1];
                let s1 = state.conv_states[conv_offset + 2];
                state.conv_states[conv_offset] = s0;
                state.conv_states[conv_offset + 1] = s1;
                state.conv_states[conv_offset + 2] = bx;

                let weights_start = d * 3;
                w.ssm_conv[weights_start] * s0
                    + w.ssm_conv[weights_start + 1] * s1
                    + w.ssm_conv[weights_start + 2] * bx
            } else {
                for k in 0..(kernel_size - 1) {
                    state.conv_states[conv_offset + k] = state.conv_states[conv_offset + k + 1];
                }
                state.conv_states[conv_offset + kernel_size - 1] = bx;

                let mut value = 0.0f32;
                for k in 0..kernel_size {
                    let weight = if has_full_conv {
                        w.ssm_conv[d * kernel_size + k]
                    } else if has_shared_conv {
                        w.ssm_conv[k]
                    } else {
                        1.0 / kernel_size as f32
                    };
                    value += weight * state.conv_states[conv_offset + k];
                }
                value
            };
            tile.norm[output_start + d] = c_slice[d] * conv_out;
        }
    }

    quantized_matmul_rows(
        &mut tile.next[..dim_rows],
        w.out_proj.quant_type,
        w.out_proj.as_slice(params.mmap),
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
        &mut tile.gate[..hidden_output_len],
        w.ffn.w_gate.quant_type,
        w.ffn.w_gate.as_slice(params.mmap),
        &tile.norm[..dim_rows],
        rows,
        hidden_dim,
        dim,
    )?;
    quantized_matmul_rows(
        &mut tile.up[..hidden_output_len],
        w.ffn.w_up.quant_type,
        w.ffn.w_up.as_slice(params.mmap),
        &tile.norm[..dim_rows],
        rows,
        hidden_dim,
        dim,
    )?;
    swiglu_rows(
        &mut tile.gate[..hidden_output_len],
        &tile.up[..hidden_output_len],
        rows,
        hidden_dim,
    )
    .map_err(|error| ModelError::ExecutionFailed(error.to_string()))?;
    quantized_matmul_rows(
        &mut tile.next[..dim_rows],
        w.ffn.w_down.quant_type,
        w.ffn.w_down.as_slice(params.mmap),
        &tile.gate[..hidden_output_len],
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

    fn test_config(dim: usize, hidden_dim: usize) -> ModelConfig {
        ModelConfig {
            name: "ssm-tile-test".to_string(),
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
            block_types: vec![BlockType::SSM],
        }
    }

    #[test]
    fn ssm_tile_matches_token_path_and_preserves_conv_state() {
        let dim = 4;
        let hidden_dim = 8;
        let cfg = test_config(dim, hidden_dim);
        let mut mmap = Vec::new();

        let mut in_proj_values = vec![0.0; 3 * dim * dim];
        for idx in 0..dim {
            in_proj_values[idx * dim + idx] = 0.5;
            in_proj_values[(dim + idx) * dim + idx] = 1.0;
            in_proj_values[(2 * dim + idx) * dim + idx] = 1.0;
        }
        let in_proj = f32_tensor(&mut mmap, 3 * dim, dim, &in_proj_values);
        let out_proj = f32_tensor(&mut mmap, dim, dim, &identity_matrix(dim));
        let zero_ffn = vec![0.0; hidden_dim * dim];
        let zero_down = vec![0.0; dim * hidden_dim];
        let ffn = FfnLayerWeights {
            ffn_norm: vec![1.0; dim].into_boxed_slice(),
            w_gate: f32_tensor(&mut mmap, hidden_dim, dim, &zero_ffn),
            w_up: f32_tensor(&mut mmap, hidden_dim, dim, &zero_ffn),
            w_down: f32_tensor(&mut mmap, dim, hidden_dim, &zero_down),
            ffn_gate_name: "ffn_gate".to_string(),
            ffn_up_name: "ffn_up".to_string(),
            ffn_down_name: "ffn_down".to_string(),
        };
        let weights = SsmLayerWeights {
            ssm_norm: vec![1.0; dim].into_boxed_slice(),
            in_proj,
            ssm_a: vec![].into_boxed_slice(),
            ssm_conv: vec![0.0, 0.0, 1.0].into_boxed_slice(),
            out_proj,
            ffn,
            in_name: "ssm_in".to_string(),
            out_name: "ssm_out".to_string(),
        };
        let adapters = ActiveAdapters::new();
        let params = SsmParams {
            layer: 0,
            weights: &weights,
            mmap: &mmap,
            config: &cfg,
            adapters: &adapters,
        };

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
        let inputs = [
            [1.0, 2.0, 3.0, 4.0],
            [0.5, -1.0, 2.0, 1.5],
            [-2.0, 1.0, 0.25, 3.0],
        ];
        let mut token_state = RunState::new(&arena);
        let mut expected = Vec::new();
        for input in &inputs {
            token_state.x.copy_from_slice(input);
            ssm_forward(&mut token_state, &params).unwrap();
            expected.push(token_state.x.to_vec());
        }

        let mut tile = TileActivations::new(inputs.len(), dim, hidden_dim).unwrap();
        for (row, input) in inputs.iter().enumerate() {
            tile.current_row_mut(row).unwrap().copy_from_slice(input);
        }
        let mut tile_state = RunState::new(&arena);
        ssm_forward_tile(&mut tile, &mut tile_state, &params, inputs.len()).unwrap();

        for (row, expected_row) in expected.iter().enumerate() {
            let actual = tile.current_row(row).unwrap();
            for (actual, expected) in actual.iter().zip(expected_row) {
                assert!((actual - expected).abs() < 1e-5);
            }
        }
        assert_eq!(tile_state.conv_states, token_state.conv_states);
    }
}
