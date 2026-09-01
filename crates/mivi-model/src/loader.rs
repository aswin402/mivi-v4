//! GGUF model loader and tensor weight resolution for mivi-v4.

use crate::config::{BlockType, ModelConfig, DEFAULT_HEAD_DIM, DEFAULT_SSM_A_VAL};
use crate::gguf::{GgufFile, GgufValue};
use crate::model::{ModelError, Result};
use crate::weights::{
    AttentionLayerWeights, FfnLayerWeights, LayerWeights, ModelWeights, QuantizedTensor,
    SsmLayerWeights,
};

/// Convert raw aligned byte buffer into f32 slice.
#[inline]
pub fn safe_f32_slice(raw: &[u8]) -> Result<&[f32]> {
    let ptr = raw.as_ptr();
    if !(ptr as usize).is_multiple_of(std::mem::align_of::<f32>()) {
        return Err(ModelError::MalformedTensorAlignment(
            "GGUF tensor buffer is not 4-byte aligned".to_string(),
        ));
    }
    // SAFETY: Pointer is validated to be 4-byte aligned, raw slice length is checked,
    // and the resulting &[f32] lifetime is strictly bound to the input &[u8] slice.
    Ok(unsafe { std::slice::from_raw_parts(ptr as *const f32, raw.len() / mivi_quant::F32_BYTES) })
}

macro_rules! read_gguf_meta {
    ($gguf:expr, $key:expr, $accessor:ident, $target:expr) => {
        if let Some(v) = $gguf.metadata.get($key).and_then(|v| v.$accessor()) {
            $target = v;
        }
    };
}

pub const GGUF_KEY_GENERAL_NAME: &str = "general.name";
pub const GGUF_KEY_CONTEXT_LENGTH: &str = "lfm.context_length";
pub const GGUF_KEY_EMBEDDING_LENGTH: &str = "lfm.embedding_length";
pub const GGUF_KEY_FEED_FORWARD_LENGTH: &str = "lfm.feed_forward_length";
pub const GGUF_KEY_BLOCK_COUNT: &str = "lfm.block_count";
pub const GGUF_KEY_ATTENTION_HEAD_COUNT: &str = "lfm.attention.head_count";
pub const GGUF_KEY_ATTENTION_HEAD_COUNT_KV: &str = "lfm.attention.head_count_kv";
pub const GGUF_KEY_ROPE_FREQ_BASE: &str = "lfm.rope.freq_base";
pub const GGUF_KEY_TOKENIZER_TOKENS: &str = "tokenizer.ggml.tokens";

/// Extract model hyperparameters from GGUF metadata.
pub fn extract_model_config(gguf: &GgufFile) -> Result<ModelConfig> {
    let mut config = ModelConfig::default();
    if let Some(s) = gguf
        .metadata
        .get(GGUF_KEY_GENERAL_NAME)
        .and_then(|v| v.as_str())
    {
        config.name = s.to_string();
    }
    read_gguf_meta!(gguf, "lfm2.context_length", as_usize, config.max_seq_len);
    read_gguf_meta!(gguf, "lfm.context_length", as_usize, config.max_seq_len);

    read_gguf_meta!(gguf, "lfm2.embedding_length", as_usize, config.dim);
    read_gguf_meta!(gguf, "lfm.embedding_length", as_usize, config.dim);

    read_gguf_meta!(
        gguf,
        "lfm2.feed_forward_length",
        as_usize,
        config.hidden_dim
    );
    read_gguf_meta!(gguf, "lfm.feed_forward_length", as_usize, config.hidden_dim);

    read_gguf_meta!(gguf, "lfm2.block_count", as_usize, config.n_layers);
    read_gguf_meta!(gguf, "lfm.block_count", as_usize, config.n_layers);

    read_gguf_meta!(gguf, "lfm2.attention.head_count", as_usize, config.n_heads);
    read_gguf_meta!(gguf, "lfm.attention.head_count", as_usize, config.n_heads);

    // head_count_kv can be a scalar or a per-layer array (LFM2 uses per-layer).
    // SSM layers have kv=0, attention layers have the actual count.
    // We take the max non-zero value as the global n_kv_heads.
    for key in &[
        "lfm2.attention.head_count_kv",
        "lfm.attention.head_count_kv",
    ] {
        match gguf.metadata.get(*key) {
            Some(GgufValue::Array(arr)) => {
                let max_kv = arr
                    .iter()
                    .filter_map(|v| v.as_usize())
                    .filter(|&v| v > 0)
                    .max()
                    .unwrap_or(config.n_kv_heads);
                if max_kv > 0 {
                    config.n_kv_heads = max_kv;
                }
            }
            Some(v) => {
                if let Some(kv) = v.as_usize() {
                    config.n_kv_heads = kv;
                }
            }
            None => {}
        }
    }

    read_gguf_meta!(gguf, "lfm2.rope.freq_base", as_f32, config.rope_base);
    read_gguf_meta!(gguf, "lfm.rope.freq_base", as_f32, config.rope_base);

    read_gguf_meta!(
        gguf,
        "lfm2.attention.layer_norm_rms_epsilon",
        as_f32,
        config.rms_norm_eps
    );
    read_gguf_meta!(
        gguf,
        "lfm.attention.layer_norm_rms_epsilon",
        as_f32,
        config.rms_norm_eps
    );

    read_gguf_meta!(
        gguf,
        "lfm2.ssm.conv_kernel",
        as_usize,
        config.ssm_conv_kernel
    );
    read_gguf_meta!(
        gguf,
        "lfm.ssm.conv_kernel",
        as_usize,
        config.ssm_conv_kernel
    );
    for i in 0..config.n_layers {
        let conv_name = format!("blk.{}.ssm_conv.weight", i);
        let conv_name_short = format!("blk.{}.shortconv.conv.weight", i);
        if let Some(t) = gguf
            .tensors
            .get(&conv_name)
            .or_else(|| gguf.tensors.get(&conv_name_short))
        {
            let total_elems: usize = t.dims.iter().product();
            if config.dim > 0 && total_elems >= config.dim && total_elems.is_multiple_of(config.dim)
            {
                config.ssm_conv_kernel = total_elems / config.dim;
                break;
            }
        }
    }

    if let Some(GgufValue::Array(tokens)) = gguf.metadata.get("tokenizer.ggml.tokens") {
        config.vocab_size = tokens.len();
    }

    config.head_dim = if config.n_heads > 0 {
        config.dim / config.n_heads
    } else {
        DEFAULT_HEAD_DIM
    };
    config.kv_dim = config.n_kv_heads * config.head_dim;

    let mut block_types = Vec::with_capacity(config.n_layers);
    for i in 0..config.n_layers {
        let is_ssm = gguf
            .tensors
            .contains_key(&format!("blk.{}.ssm_in.weight", i))
            || gguf
                .tensors
                .contains_key(&format!("blk.{}.shortconv.in_proj.weight", i))
            || gguf
                .tensors
                .contains_key(&format!("blk.{}.shortconv.conv.weight", i));
        if is_ssm {
            block_types.push(BlockType::SSM);
        } else {
            block_types.push(BlockType::Attention);
        }
    }
    config.block_types = block_types;
    config.validate().map_err(ModelError::InvalidConfig)?;
    Ok(config)
}

/// Extract vocabulary string list from GGUF metadata tokens.
pub fn extract_vocab(gguf: &GgufFile, default_size: usize) -> Vec<String> {
    let mut tokens = Vec::new();
    if let Some(GgufValue::Array(arr)) = gguf.metadata.get(GGUF_KEY_TOKENIZER_TOKENS) {
        for v in arr {
            if let Some(s) = v.as_str() {
                tokens.push(s.to_string());
            }
        }
    }
    if tokens.is_empty() {
        for i in 0..default_size {
            tokens.push(format!("<tok_{}>", i));
        }
    }
    tokens
}

/// Extract BPE merge table from GGUF metadata.
pub fn extract_merges(gguf: &GgufFile) -> std::collections::HashMap<(String, String), u32> {
    let mut merges = std::collections::HashMap::new();
    if let Some(GgufValue::Array(arr)) = gguf.metadata.get("tokenizer.ggml.merges") {
        merges.reserve(arr.len());
        for (rank, val) in arr.iter().enumerate() {
            if let GgufValue::String(s) = val {
                if let Some((left, right)) = s.split_once(' ') {
                    merges.insert((left.to_string(), right.to_string()), rank as u32);
                }
            }
        }
    }
    merges
}

#[inline]
fn get_tensor_entry<'a>(
    gguf: &'a GgufFile,
    name: &str,
) -> Result<(&'a crate::gguf::TensorInfo, &'a [u8])> {
    gguf.get_tensor_data(name)
        .map_err(|_| ModelError::MissingWeight(name.to_string()))
}

/// Resolve a quantized tensor by name from GGUF file.
pub fn resolve_tensor(gguf: &GgufFile, name: &str) -> Result<QuantizedTensor> {
    let (info, data) = get_tensor_entry(gguf, name)?;
    if info.dims.is_empty() {
        return Err(ModelError::DimMismatch(format!(
            "Tensor '{}' has empty dimensions",
            name
        )));
    }
    Ok(QuantizedTensor {
        quant_type: info.ggml_type,
        offset: gguf.data_offset + info.offset as usize,
        len: data.len(),
        rows: if info.dims.len() > 1 {
            info.dims[1]
        } else {
            info.dims[0]
        },
        cols: info.dims[0],
    })
}

/// Resolve a tensor vector by name and dequantize to f32 (supporting F32, F16, and BF16 formats).
pub fn resolve_f32_vec(gguf: &GgufFile, name: &str) -> Result<Box<[f32]>> {
    let (info, raw) = get_tensor_entry(gguf, name)?;
    let num_elements = info.dims.iter().product::<usize>();
    let mut out = vec![0.0f32; num_elements];
    mivi_quant::dequantize_slice(info.ggml_type, raw, &mut out)?;
    Ok(out.into_boxed_slice())
}

#[inline]
fn layer_tensor_name(layer_idx: usize, tensor: &str) -> String {
    format!("blk.{}.{}.weight", layer_idx, tensor)
}

#[inline]
fn layer_module_name(layer_idx: usize, module: &str) -> String {
    format!("blk.{}.{}", layer_idx, module)
}

/// Resolve feed-forward network weights for a given layer index.
pub fn resolve_ffn_weights(gguf: &GgufFile, layer_idx: usize) -> Result<FfnLayerWeights> {
    let ffn_norm = resolve_f32_vec(gguf, &layer_tensor_name(layer_idx, "ffn_norm"))?;
    let w_gate = resolve_tensor(gguf, &layer_tensor_name(layer_idx, "ffn_gate"))?;
    let w_up = resolve_tensor(gguf, &layer_tensor_name(layer_idx, "ffn_up"))?;
    let w_down = resolve_tensor(gguf, &layer_tensor_name(layer_idx, "ffn_down"))?;

    Ok(FfnLayerWeights {
        ffn_norm,
        w_gate,
        w_up,
        w_down,
        ffn_gate_name: layer_module_name(layer_idx, "ffn_gate"),
        ffn_up_name: layer_module_name(layer_idx, "ffn_up"),
        ffn_down_name: layer_module_name(layer_idx, "ffn_down"),
    })
}

/// Resolve grouped-query attention weights for a given layer index.
pub fn resolve_attention_layer(gguf: &GgufFile, layer_idx: usize) -> Result<AttentionLayerWeights> {
    let attn_norm = resolve_f32_vec(gguf, &layer_tensor_name(layer_idx, "attn_norm"))?;
    let q_norm = resolve_f32_vec(gguf, &layer_tensor_name(layer_idx, "attn_q_norm")).ok();
    let k_norm = resolve_f32_vec(gguf, &layer_tensor_name(layer_idx, "attn_k_norm")).ok();
    let wq = resolve_tensor(gguf, &layer_tensor_name(layer_idx, "attn_q"))?;
    let wk = resolve_tensor(gguf, &layer_tensor_name(layer_idx, "attn_k"))?;
    let wv = resolve_tensor(gguf, &layer_tensor_name(layer_idx, "attn_v"))?;
    let wo = resolve_tensor(gguf, &layer_tensor_name(layer_idx, "attn_output"))?;
    let ffn = resolve_ffn_weights(gguf, layer_idx)?;

    Ok(AttentionLayerWeights {
        attn_norm,
        q_norm,
        k_norm,
        wq,
        wk,
        wv,
        wo,
        ffn,
        q_name: layer_module_name(layer_idx, "attn_q"),
        k_name: layer_module_name(layer_idx, "attn_k"),
        v_name: layer_module_name(layer_idx, "attn_v"),
        o_name: layer_module_name(layer_idx, "attn_output"),
    })
}

/// Resolve state space model (SSM) layer weights for a given layer index.
pub fn resolve_ssm_layer(
    gguf: &GgufFile,
    layer_idx: usize,
    config: &ModelConfig,
) -> Result<SsmLayerWeights> {
    let ssm_norm = resolve_f32_vec(gguf, &layer_tensor_name(layer_idx, "ssm_norm"))
        .or_else(|_| resolve_f32_vec(gguf, &layer_tensor_name(layer_idx, "attn_norm")))?;
    let in_proj = resolve_tensor(gguf, &layer_tensor_name(layer_idx, "ssm_in"))
        .or_else(|_| resolve_tensor(gguf, &layer_tensor_name(layer_idx, "shortconv.in_proj")))?;
    let out_proj = resolve_tensor(gguf, &layer_tensor_name(layer_idx, "ssm_out"))
        .or_else(|_| resolve_tensor(gguf, &layer_tensor_name(layer_idx, "shortconv.out_proj")))?;

    let a_name = layer_tensor_name(layer_idx, "ssm_a");
    let ssm_a = match gguf.get_tensor_data(&a_name) {
        Ok((_, raw)) => safe_f32_slice(raw)?.to_vec().into_boxed_slice(),
        Err(_) => {
            tracing::debug!("Using default SSM A weights for layer {}", layer_idx);
            vec![DEFAULT_SSM_A_VAL; config.ssm_state_dim].into_boxed_slice()
        }
    };

    let conv_name = layer_tensor_name(layer_idx, "ssm_conv");
    let conv_name_short = layer_tensor_name(layer_idx, "shortconv.conv");
    let ssm_conv = if let Ok((_, raw)) = gguf.get_tensor_data(&conv_name) {
        safe_f32_slice(raw)?.to_vec().into_boxed_slice()
    } else if let Ok((_, raw)) = gguf.get_tensor_data(&conv_name_short) {
        safe_f32_slice(raw)?.to_vec().into_boxed_slice()
    } else {
        tracing::debug!(
            "SSM conv weights missing for layer {}, using empty buffer",
            layer_idx
        );
        Box::new([])
    };

    let ffn = resolve_ffn_weights(gguf, layer_idx)?;

    Ok(SsmLayerWeights {
        ssm_norm,
        in_proj,
        ssm_a,
        ssm_conv,
        out_proj,
        ffn,
        in_name: layer_module_name(layer_idx, "ssm_in"),
        out_name: layer_module_name(layer_idx, "ssm_out"),
    })
}

/// Resolve all model weights (embeddings, layers, norms, head) from GGUF.
pub fn resolve_model_weights(gguf: &GgufFile, config: &ModelConfig) -> Result<ModelWeights> {
    let (emb_info, emb_data) = gguf
        .get_tensor_data("token_embd.weight")
        .or_else(|_| gguf.get_tensor_data("model.embed_tokens.weight"))
        .map_err(|_| ModelError::MissingWeight("token_embd.weight".into()))?;

    let token_embd = QuantizedTensor {
        quant_type: emb_info.ggml_type,
        offset: gguf.data_offset + emb_info.offset as usize,
        len: emb_data.len(),
        rows: config.vocab_size,
        cols: config.dim,
    };

    let mut layers = Vec::with_capacity(config.n_layers);
    for layer_idx in 0..config.n_layers {
        match config.block_types[layer_idx] {
            BlockType::Attention => {
                layers.push(LayerWeights::Attention(resolve_attention_layer(
                    gguf, layer_idx,
                )?));
            }
            BlockType::SSM => {
                layers.push(LayerWeights::Ssm(resolve_ssm_layer(
                    gguf, layer_idx, config,
                )?));
            }
        }
    }

    let output_norm = if let Ok((_, raw)) = gguf
        .get_tensor_data("output_norm.weight")
        .or_else(|_| gguf.get_tensor_data("token_embd_norm.weight"))
        .or_else(|_| gguf.get_tensor_data("token_embd_norm"))
    {
        Some(safe_f32_slice(raw)?.to_vec().into_boxed_slice())
    } else {
        None
    };

    let output_proj = if let Ok((info, data)) = gguf
        .get_tensor_data("output.weight")
        .or_else(|_| gguf.get_tensor_data("lm_head.weight"))
    {
        Some(QuantizedTensor {
            quant_type: info.ggml_type,
            offset: gguf.data_offset + info.offset as usize,
            len: data.len(),
            rows: config.vocab_size,
            cols: config.dim,
        })
    } else {
        None
    };

    Ok(ModelWeights {
        token_embd,
        layers,
        output_norm,
        output_proj,
    })
}
