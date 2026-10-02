#!/usr/bin/env python3
"""Generate a private synthetic hybrid GGUF and byte-derived oracle traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from reference.reference_engine import LFMConfig, ReferenceEngine  # noqa: E402
from training.export.convert_to_gguf import GgufWriter, GGML_TYPE_F32  # noqa: E402


DIM = 64
HIDDEN = 64
VOCAB = 64
LAYERS = 4
HEADS = 4
KV_HEADS = 2
HEAD_DIM = DIM // HEADS
CONTEXT = 128
ROPE_BASE = 10000.0
CONV_KERNEL = 3
BLOCKS = ("ssm", "attention", "ssm", "attention")
TEACHER_SEQUENCES = ((3, 7, 5, 11, 2), (9, 1, 13, 6, 4))
CASES = (
    ("teacher_a", TEACHER_SEQUENCES[0]),
    ("teacher_b_after_reset", TEACHER_SEQUENCES[1]),
    ("changed_prefix", (*TEACHER_SEQUENCES[0][:2], 13, 6, 4)),
)


def _matrix(rows: int, cols: int, seed: int, scale: float = 0.02) -> list[list[float]]:
    return [
        [(((seed * 31 + row * 17 + col * 43 + row * col * 7) % 101) - 50) * scale / 50.0
         for col in range(cols)]
        for row in range(rows)
    ]


def _norm(size: int, seed: int) -> list[float]:
    return [1.0 + (((seed * 11 + i * 7) % 19) - 9) / 128.0 for i in range(size)]


def synthetic_weights() -> dict[str, Any]:
    weights: dict[str, Any] = {
        "token_embd.weight": _matrix(VOCAB, DIM, 2, 0.15),
        "output.weight": _matrix(VOCAB, DIM, 89, 0.025),
        "output_norm.weight": _norm(DIM, 91),
    }
    for layer, block in enumerate(BLOCKS):
        prefix = f"blk.{layer}."
        weights[prefix + ("ssm_norm.weight" if block == "ssm" else "attn_norm.weight")] = _norm(DIM, 3 + layer)
        if block == "ssm":
            weights[prefix + "shortconv.in_proj.weight"] = _matrix(3 * DIM, DIM, 10 + layer, 0.018)
            weights[prefix + "shortconv.conv.weight"] = [
                (((layer * 13 + i * 5) % 23) - 11) / 32.0 for i in range(DIM * CONV_KERNEL)
            ]
            weights[prefix + "shortconv.out_proj.weight"] = _matrix(DIM, DIM, 20 + layer, 0.012)
        else:
            weights[prefix + "attn_q.weight"] = _matrix(DIM, DIM, 30 + layer, 0.018)
            weights[prefix + "attn_k.weight"] = _matrix(KV_HEADS * HEAD_DIM, DIM, 40 + layer, 0.018)
            weights[prefix + "attn_v.weight"] = _matrix(KV_HEADS * HEAD_DIM, DIM, 50 + layer, 0.018)
            weights[prefix + "attn_output.weight"] = _matrix(DIM, DIM, 60 + layer, 0.012)
        weights[prefix + "ffn_norm.weight"] = _norm(DIM, 70 + layer)
        weights[prefix + "ffn_gate.weight"] = _matrix(HIDDEN, DIM, 80 + layer, 0.018)
        weights[prefix + "ffn_up.weight"] = _matrix(HIDDEN, DIM, 90 + layer, 0.018)
        weights[prefix + "ffn_down.weight"] = _matrix(DIM, HIDDEN, 100 + layer, 0.012)
    return weights


def _add_metadata(writer: GgufWriter) -> None:
    writer.add_string("general.architecture", "lfm")
    writer.add_string("general.name", "mivi-synthetic-adversarial-hybrid")
    writer.add_uint32("lfm.context_length", CONTEXT)
    writer.add_uint32("lfm.embedding_length", DIM)
    writer.add_uint32("lfm.block_count", LAYERS)
    writer.add_uint32("lfm.feed_forward_length", HIDDEN)
    writer.add_uint32("lfm.attention.head_count", HEADS)
    writer.add_uint32("lfm.attention.head_count_kv", KV_HEADS)
    writer.add_float32("lfm.rope.freq_base", ROPE_BASE)
    writer.add_uint32("lfm.ssm.conv_kernel", CONV_KERNEL)
    writer.add_string_array("tokenizer.ggml.tokens", [f"<synthetic_{i}>" for i in range(VOCAB)])


def write_gguf(path: Path, weights: dict[str, Any]) -> None:
    writer = GgufWriter(str(path))
    _add_metadata(writer)
    for name, values in weights.items():
        if values and isinstance(values[0], list):
            rows, cols = len(values), len(values[0])
            raw = struct.pack(f"<{rows * cols}f", *(v for row in values for v in row))
            writer.add_tensor(name, [cols, rows], GGML_TYPE_F32, raw)
        else:
            writer.add_tensor(name, [len(values)], GGML_TYPE_F32, struct.pack(f"<{len(values)}f", *values))
    writer.write()


def _read_exact(file, size: int) -> bytes:
    data = file.read(size)
    if len(data) != size:
        raise ValueError("truncated GGUF while reading serialized fixture")
    return data


def _read_string(file) -> str:
    length = struct.unpack("<Q", _read_exact(file, 8))[0]
    return _read_exact(file, length).decode("utf-8")


def _skip_value(file, value_type: int) -> None:
    fixed_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if value_type in fixed_sizes:
        _read_exact(file, fixed_sizes[value_type])
    elif value_type == 8:
        _read_string(file)
    elif value_type == 9:
        element_type = struct.unpack("<I", _read_exact(file, 4))[0]
        count = struct.unpack("<Q", _read_exact(file, 8))[0]
        for _ in range(count):
            _skip_value(file, element_type)
    else:
        raise ValueError(f"unsupported GGUF metadata type in synthetic fixture: {value_type}")


def read_serialized_f32_tensors(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Independently parse F32 tensor payloads from completed GGUF bytes."""
    infos = []
    with path.open("rb") as file:
        magic, version = struct.unpack("<II", _read_exact(file, 8))
        tensor_count, metadata_count = struct.unpack("<QQ", _read_exact(file, 16))
        if magic != 0x46554747 or version != 3:
            raise ValueError("generated fixture is not GGUF v3")
        for _ in range(metadata_count):
            _read_string(file)
            value_type = struct.unpack("<I", _read_exact(file, 4))[0]
            _skip_value(file, value_type)
        for _ in range(tensor_count):
            name = _read_string(file)
            dimensions = struct.unpack("<I", _read_exact(file, 4))[0]
            dims = struct.unpack(f"<{dimensions}Q", _read_exact(file, dimensions * 8))
            tensor_type, offset = struct.unpack("<IQ", _read_exact(file, 12))
            infos.append((name, dims, tensor_type, offset))
        data_start = (file.tell() + 31) & ~31
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        tensors: dict[str, Any] = {}
        hashes: dict[str, str] = {}
        for name, dims, tensor_type, offset in infos:
            if tensor_type != GGML_TYPE_F32:
                raise ValueError(f"fixture tensor {name} is not serialized F32")
            count = math.prod(dims)
            byte_count = count * 4
            start = data_start + offset
            if start + byte_count > file_size:
                raise ValueError(f"serialized payload for {name} extends beyond GGUF")
            file.seek(start)
            raw = _read_exact(file, byte_count)
            flat = list(struct.unpack(f"<{count}f", raw))
            hashes[name] = hashlib.sha256(raw).hexdigest()
            if len(dims) == 1:
                tensors[name] = flat
            elif len(dims) == 2:
                cols, rows = dims
                tensors[name] = [flat[row * cols:(row + 1) * cols] for row in range(rows)]
            else:
                raise ValueError(f"unsupported synthetic tensor rank for {name}: {len(dims)}")
    return tensors, hashes


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_trace(tensors: dict[str, Any], tensor_hashes: dict[str, str], gguf_path: Path) -> dict[str, Any]:
    config = LFMConfig(
        name="mivi-synthetic-adversarial-hybrid",
        dim=DIM,
        hidden_dim=HIDDEN,
        n_layers=LAYERS,
        n_heads=HEADS,
        n_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        kv_dim=KV_HEADS * HEAD_DIM,
        vocab_size=VOCAB,
        max_seq_len=CONTEXT,
        rope_base=ROPE_BASE,
        ssm_state_dim=DIM,
        ssm_conv_kernel=CONV_KERNEL,
        block_types=list(BLOCKS),
    )
    oracle = ReferenceEngine(config, tensors)
    cases = []
    for name, sequence in CASES:
        oracle.reset()
        steps = []
        for pos, token_id in enumerate(sequence):
            result = oracle.forward_token(token_id, pos)
            conv_carry_nonzero = sum(
                value != 0.0
                for key, channels in oracle.ssm_states.items()
                if isinstance(key, str) and key.startswith("conv_")
                for history in channels
                for value in history
            )
            if conv_carry_nonzero == 0:
                raise ValueError("synthetic sequence failed to produce nonzero convolution carry")
            ordered = sorted(result["logits"], reverse=True)
            steps.append({
                "pos": pos,
                "token_id": token_id,
                "logits": result["logits"],
                "top_token": result["top_token"],
                "top_margin": ordered[0] - ordered[1],
                "conv_carry_nonzero_values": conv_carry_nonzero,
            })
        cases.append({"name": name, "token_ids": list(sequence), "steps": steps})
    return {
        "schema": 1,
        "provenance": {
            "exporter_revision": "sha256:" + _source_hash(Path(__file__).resolve()),
            "reference_revision": "sha256:" + _source_hash(ROOT / "reference/reference_engine.py"),
            "gguf_sha256": hashlib.sha256(gguf_path.read_bytes()).hexdigest(),
            "tensor_sha256": tensor_hashes,
            "sources": [
                "https://github.com/ggml-org/ggml/blob/master/docs/gguf.md",
                "https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/main/tests/fixtures/README.md",
            ],
        },
        "config": {
            "dim": DIM,
            "hidden_dim": HIDDEN,
            "vocab_size": VOCAB,
            "context": CONTEXT,
            "n_layers": LAYERS,
            "block_types": list(BLOCKS),
            "n_heads": HEADS,
            "n_kv_heads": KV_HEADS,
            "gqa_ratio": HEADS // KV_HEADS,
            "head_dim": HEAD_DIM,
            "rope_base": ROPE_BASE,
            "ssm_conv_kernel": CONV_KERNEL,
            "dtype": "F32",
        },
        "numerical_policy": {
            "f32_atol": 1.0e-4,
            "f32_rtol": 1.0e-4,
            "quant_dot_abs_sum_factor": 2.0e-6,
            "quant_dot_absolute_floor": 1.0e-5,
            "logits_per_step": VOCAB,
        },
        "cases": cases,
    }


def generate(output_dir: Path) -> tuple[Path, Path]:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing existing output directory: {output_dir}")
    output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    if os.name == "posix":
        os.chmod(output_dir, 0o700)
        if output_dir.stat().st_mode & 0o777 != 0o700:
            raise PermissionError("fixture output directory is not private mode 0700")
    gguf_path = output_dir / "hybrid_adversarial.gguf"
    trace_path = output_dir / "hybrid_adversarial.json"
    old_umask = os.umask(0o077)
    try:
        write_gguf(gguf_path, synthetic_weights())
        tensors, tensor_hashes = read_serialized_f32_tensors(gguf_path)
        trace = build_trace(tensors, tensor_hashes, gguf_path)
        with trace_path.open("x", encoding="utf-8") as file:
            json.dump(trace, file, indent=2, allow_nan=False)
            file.write("\n")
    finally:
        os.umask(old_umask)
    if os.name == "posix":
        os.chmod(gguf_path, 0o600)
        os.chmod(trace_path, 0o600)
        for path in (gguf_path, trace_path):
            if path.stat().st_mode & 0o777 != 0o600:
                raise PermissionError(f"fixture artifact is not private mode 0600: {path.name}")
    return gguf_path, trace_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path, help="new private directory; existing paths are refused")
    args = parser.parse_args()
    try:
        gguf_path, trace_path = generate(args.output_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"synthetic fixture generated: {gguf_path}")
    print(f"synthetic oracle generated: {trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
