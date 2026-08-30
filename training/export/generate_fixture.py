#!/usr/bin/env python3
"""
Generate deterministic tiny GGUF test fixture and ground-truth oracle outputs
to verify the Rust inference engine against the Python reference engine.
"""

import os
import sys
import json
import math
import struct

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from reference.reference_engine import LFMConfig, ReferenceEngine
from training.export.convert_to_gguf import GgufWriter, quantize_f32_to_q8_0, GGML_TYPE_Q8_0, GGML_TYPE_F32


def generate_deterministic_matrix(rows: int, cols: int, seed: float = 0.1) -> list:
    matrix = []
    val = seed
    for r in range(rows):
        row = []
        for c in range(cols):
            val = (val * 1.337 + 0.123) % 1.0 - 0.5
            row.append(round(val * 0.2, 4))
        matrix.append(row)
    return matrix


def generate_deterministic_vector(size: int, scale: float = 1.0) -> list:
    val = 0.5
    vec = []
    for _ in range(size):
        val = (val * 1.414 + 0.271) % 1.0 - 0.5
        vec.append(round(val * 0.1 + scale, 4))
    return vec


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("tests/fixtures", exist_ok=True)

    # 1. Tiny model config for test
    dim = 64
    hidden_dim = 128
    n_layers = 2
    n_heads = 2
    n_kv_heads = 1
    head_dim = 32
    kv_dim = 32
    vocab_size = 64
    max_seq_len = 512

    config = LFMConfig(
        name="mivi-tiny-oracle",
        dim=dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        kv_dim=kv_dim,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        block_types=["ssm", "attention"],
    )

    # 2. Build synthetic weights
    weights = {}
    # Embeddings [vocab, dim]
    weights["token_embd.weight"] = generate_deterministic_matrix(vocab_size, dim, 0.15)
    # LM Head [vocab, dim]
    weights["output.weight"] = generate_deterministic_matrix(vocab_size, dim, 0.25)
    weights["output_norm.weight"] = generate_deterministic_vector(dim, 1.0)

    # Layer 0: ShortConv SSM Block
    weights["blk.0.ssm_norm.weight"] = generate_deterministic_vector(dim, 1.0)
    weights["blk.0.shortconv.in_proj.weight"] = generate_deterministic_matrix(3 * dim, dim, 0.3)
    weights["blk.0.shortconv.conv.weight"] = generate_deterministic_vector(3 * dim, 0.25)
    weights["blk.0.shortconv.out_proj.weight"] = generate_deterministic_matrix(dim, dim, 0.4)
    weights["blk.0.ffn_norm.weight"] = generate_deterministic_vector(dim, 1.0)
    weights["blk.0.ffn_gate.weight"] = generate_deterministic_matrix(hidden_dim, dim, 0.5)
    weights["blk.0.ffn_up.weight"] = generate_deterministic_matrix(hidden_dim, dim, 0.6)
    weights["blk.0.ffn_down.weight"] = generate_deterministic_matrix(dim, hidden_dim, 0.7)

    # Layer 1: Attention Block
    weights["blk.1.attn_norm.weight"] = generate_deterministic_vector(dim, 1.0)
    weights["blk.1.attn_q.weight"] = generate_deterministic_matrix(dim, dim, 0.8)
    weights["blk.1.attn_k.weight"] = generate_deterministic_matrix(kv_dim, dim, 0.9)
    weights["blk.1.attn_v.weight"] = generate_deterministic_matrix(kv_dim, dim, 1.1)
    weights["blk.1.attn_output.weight"] = generate_deterministic_matrix(dim, dim, 1.2)
    weights["blk.1.ffn_norm.weight"] = generate_deterministic_vector(dim, 1.0)
    weights["blk.1.ffn_gate.weight"] = generate_deterministic_matrix(hidden_dim, dim, 1.3)
    weights["blk.1.ffn_up.weight"] = generate_deterministic_matrix(hidden_dim, dim, 1.4)
    weights["blk.1.ffn_down.weight"] = generate_deterministic_matrix(dim, hidden_dim, 1.5)

    # 3. Run Reference Oracle Forward Pass
    oracle = ReferenceEngine(config, weights)
    tokens_to_feed = [1, 5, 12, 3]
    oracle_traces = []

    for pos, tok in enumerate(tokens_to_feed):
        res = oracle.forward_token(tok, pos)
        oracle_traces.append({
            "pos": pos,
            "token": tok,
            "top_token": res["top_token"],
            "logits_sample": res["logits_sample"],
        })

    with open("tests/fixtures/oracle_output.json", "w") as f:
        json.dump(oracle_traces, f, indent=2)
    print("✅ Generated oracle ground-truth traces in tests/fixtures/oracle_output.json")

    # 4. Write GGUF model file
    gguf_path = "models/mivi-tiny-test.gguf"
    writer = GgufWriter(gguf_path)

    # Metadata
    writer.add_string("general.architecture", "lfm")
    writer.add_string("general.name", "mivi-tiny-test")
    writer.add_uint32("lfm.context_length", max_seq_len)
    writer.add_uint32("lfm.embedding_length", dim)
    writer.add_uint32("lfm.block_count", n_layers)
    writer.add_uint32("lfm.feed_forward_length", hidden_dim)
    writer.add_uint32("lfm.attention.head_count", n_heads)
    writer.add_uint32("lfm.attention.head_count_kv", n_kv_heads)
    writer.add_float32("lfm.rope.freq_base", config.rope_base)

    dummy_tokens = [f"<tok_{i}>" for i in range(vocab_size)]
    dummy_tokens[0] = "<|im_start|>"
    dummy_tokens[1] = "<|im_end|>"
    writer.add_string_array("tokenizer.ggml.tokens", dummy_tokens)

    # Convert tensors to GGUF
    for name, tensor_data in weights.items():
        if isinstance(tensor_data[0], list):
            # 2D matrix
            rows = len(tensor_data)
            cols = len(tensor_data[0])
            flat = [v for row in tensor_data for v in row]
            q8_bytes = quantize_f32_to_q8_0(flat)
            writer.add_tensor(name, [cols, rows], GGML_TYPE_Q8_0, q8_bytes)
        else:
            # 1D vector (norms) -> F32
            raw_f32 = bytearray()
            for val in tensor_data:
                raw_f32.extend(struct.pack("<f", val))
            writer.add_tensor(name, [len(tensor_data)], GGML_TYPE_F32, bytes(raw_f32))

    writer.write()
    print(f"✅ Generated synthetic test model at {gguf_path}")


if __name__ == "__main__":
    main()
