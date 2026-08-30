#!/usr/bin/env python3
"""
PyTorch Reference Engine (Oracle) for Mivi-v4 / LFM2.5 hybrid architecture.
Serves as ground-truth oracle for validating the Rust inference engine.
"""

import math
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any


@dataclass
class LFMConfig:
    name: str = "mivi-v4-reference"
    dim: int = 1024
    hidden_dim: int = 2816
    n_layers: int = 16
    n_heads: int = 16
    n_kv_heads: int = 4
    head_dim: int = 64
    kv_dim: int = 256
    vocab_size: int = 65536
    max_seq_len: int = 32768
    rope_base: float = 1_000_000.0
    ssm_state_dim: int = 512
    ssm_conv_kernel: int = 4
    block_types: Optional[List[str]] = None

    def __post_init__(self):
        if self.block_types is None:
            # 10 SSM + 6 GQA Attention
            self.block_types = []
            for i in range(self.n_layers):
                if i % 3 == 2:
                    self.block_types.append("attention")
                else:
                    self.block_types.append("ssm")
            while len(self.block_types) < self.n_layers:
                self.block_types.append("attention")


def rms_norm(x: List[float], weight: List[float], eps: float = 1e-5) -> List[float]:
    mean_sq = sum(v * v for v in x) / len(x)
    scale = 1.0 / math.sqrt(mean_sq + eps)
    return [xi * scale * wi for xi, wi in zip(x, weight)]


def silu(x: float) -> float:
    return x / (1.0 + math.exp(-x)) if -x < 700 else 0.0


def swiglu(gate: List[float], up: List[float]) -> List[float]:
    return [silu(g) * u for g, u in zip(gate, up)]


def softmax(x: List[float]) -> List[float]:
    if not x:
        return []
    max_v = max(x)
    exps = [math.exp(v - max_v) for v in x]
    s = sum(exps)
    return [e / s for e in exps] if s > 0 else [0.0] * len(x)


def apply_rope(q: List[float], k: List[float], head_dim: int, pos: int, rope_base: float = 1_000_000.0):
    for i in range(0, head_dim, 2):
        freq = 1.0 / (rope_base ** (i / head_dim))
        angle = pos * freq
        cos_val = math.cos(angle)
        sin_val = math.sin(angle)

        if i + 1 < len(q):
            q0, q1 = q[i], q[i + 1]
            q[i] = q0 * cos_val - q1 * sin_val
            q[i + 1] = q0 * sin_val + q1 * cos_val

        if i + 1 < len(k):
            k0, k1 = k[i], k[i + 1]
            k[i] = k0 * cos_val - k1 * sin_val
            k[i + 1] = k0 * sin_val + k1 * cos_val


def matvec(w: List[List[float]], x: List[float]) -> List[float]:
    """w: [n, d], x: [d] -> [n]"""
    return [sum(row[j] * x[j] for j in range(len(x))) for row in w]


class ReferenceEngine:
    """Pure reference forward pass engine."""

    def __init__(self, config: LFMConfig, weights: Dict[str, Any]):
        self.config = config
        self.weights = weights
        self.kv_k: Dict[Tuple[int, int], List[float]] = {}
        self.kv_v: Dict[Tuple[int, int], List[float]] = {}
        self.ssm_states: Dict[int, List[float]] = {
            i: [0.0] * min(config.ssm_state_dim, config.dim) for i in range(config.n_layers)
        }

    def reset(self):
        self.kv_k.clear()
        self.kv_v.clear()
        for i in range(self.config.n_layers):
            self.ssm_states[i] = [0.0] * min(self.config.ssm_state_dim, self.config.dim)

    def forward_token(self, token_id: int, pos: int) -> Dict[str, Any]:
        dim = self.config.dim
        hidden_dim = self.config.hidden_dim
        head_dim = self.config.head_dim
        n_heads = self.config.n_heads
        n_kv_heads = self.config.n_kv_heads
        heads_per_kv = n_heads // n_kv_heads

        # 1. Token Embedding
        embed_w = self.weights["token_embd.weight"]  # [vocab, dim]
        x = list(embed_w[token_id])

        traces: Dict[str, Any] = {"token_id": token_id, "pos": pos, "emb": x[:4]}

        # 2. Layer Loop
        for l in range(self.config.n_layers):
            btype = self.config.block_types[l]

            if btype == "attention":
                attn_norm_w = self.weights[f"blk.{l}.attn_norm.weight"]
                xb = rms_norm(x, attn_norm_w)

                # Q, K, V
                q = matvec(self.weights[f"blk.{l}.attn_q.weight"], xb)
                k = matvec(self.weights[f"blk.{l}.attn_k.weight"], xb)
                v = matvec(self.weights[f"blk.{l}.attn_v.weight"], xb)

                # RoPE
                apply_rope(q, k, head_dim, pos, self.config.rope_base)

                # Store KV
                self.kv_k[(l, pos)] = list(k)
                self.kv_v[(l, pos)] = list(v)

                # GQA Multi-head Attention
                attn_out = [0.0] * dim
                scale = 1.0 / math.sqrt(head_dim)
                seq_len = pos + 1

                for h in range(n_heads):
                    kv_h = h // heads_per_kv
                    q_h = q[h * head_dim : (h + 1) * head_dim]

                    # Compute attention scores
                    scores = []
                    for t in range(seq_len):
                        k_cached = self.kv_k[(l, t)]
                        k_h = k_cached[kv_h * head_dim : (kv_h + 1) * head_dim]
                        dot = sum(q_h[i] * k_h[i] for i in range(head_dim))
                        scores.append(dot * scale)

                    probs = softmax(scores)

                    # Compute head output
                    for t in range(seq_len):
                        weight = probs[t]
                        v_cached = self.kv_v[(l, t)]
                        v_h = v_cached[kv_h * head_dim : (kv_h + 1) * head_dim]
                        for i in range(head_dim):
                            attn_out[h * head_dim + i] += weight * v_h[i]

                # Output proj + residual
                proj_out = matvec(self.weights[f"blk.{l}.attn_output.weight"], attn_out)
                x = [xi + pi for xi, pi in zip(x, proj_out)]

                # FFN Pre-Norm + SwiGLU
                ffn_norm_w = self.weights[f"blk.{l}.ffn_norm.weight"]
                xb = rms_norm(x, ffn_norm_w)
                gate = matvec(self.weights[f"blk.{l}.ffn_gate.weight"], xb)
                up = matvec(self.weights[f"blk.{l}.ffn_up.weight"], xb)
                hb = swiglu(gate, up)
                down = matvec(self.weights[f"blk.{l}.ffn_down.weight"], hb)
                x = [xi + di for xi, di in zip(x, down)]

            elif btype == "ssm":
                ssm_norm_w = self.weights.get(f"blk.{l}.ssm_norm.weight") or self.weights[f"blk.{l}.attn_norm.weight"]
                xb = rms_norm(x, ssm_norm_w)
                in_w = self.weights.get(f"blk.{l}.shortconv.in_proj.weight") or self.weights[f"blk.{l}.ssm_in.weight"]
                in_proj = matvec(in_w, xb)

                # ShortConv: B, C, X
                b_part = in_proj[:dim]
                c_part = in_proj[dim:2*dim]
                x_part = in_proj[2*dim:3*dim]
                bx = [b_i * x_i for b_i, x_i in zip(b_part, x_part)]

                # Conv state & depthwise 1D
                conv_w = self.weights.get(f"blk.{l}.shortconv.conv.weight") or self.weights.get(f"blk.{l}.ssm_conv.weight") or [0.25] * (dim * 3)
                k_size = 3
                conv_key = f"conv_{l}"
                if conv_key not in self.ssm_states:
                    self.ssm_states[conv_key] = [[0.0] * k_size for _ in range(dim)]
                c_state = self.ssm_states[conv_key]
                conv_out = [0.0] * dim
                for d in range(dim):
                    c_state[d].pop(0)
                    c_state[d].append(bx[d])
                    w_k = [conv_w[d * k_size + k] if len(conv_w) == dim * k_size else conv_w[k] if k < len(conv_w) else 0.25 for k in range(k_size)]
                    conv_out[d] = sum(c_state[d][k] * w_k[k] for k in range(k_size))

                # Gating with C
                gated = [conv_out[d] * c_part[d] for d in range(dim)]

                out_w = self.weights.get(f"blk.{l}.shortconv.out_proj.weight") or self.weights[f"blk.{l}.ssm_out.weight"]
                out_proj = matvec(out_w, gated)
                x = [xi + oi for xi, oi in zip(x, out_proj)]

                # FFN
                ffn_norm_w = self.weights[f"blk.{l}.ffn_norm.weight"]
                xb = rms_norm(x, ffn_norm_w)
                gate = matvec(self.weights[f"blk.{l}.ffn_gate.weight"], xb)
                up = matvec(self.weights[f"blk.{l}.ffn_up.weight"], xb)
                hb = swiglu(gate, up)
                down = matvec(self.weights[f"blk.{l}.ffn_down.weight"], hb)
                x = [xi + di for xi, di in zip(x, down)]

        # 3. Final Norm
        out_norm_w = self.weights.get("output_norm.weight", [1.0] * dim)
        x_norm = rms_norm(x, out_norm_w)

        # 4. LM Head
        head_w = self.weights["output.weight"]
        logits = matvec(head_w, x_norm)

        traces["logits_sample"] = logits[:8]
        traces["top_token"] = max(range(len(logits)), key=lambda i: logits[i])
        traces["logits"] = logits
        return traces


if __name__ == "__main__":
    cfg = LFMConfig(dim=128, hidden_dim=256, n_layers=4, n_heads=4, n_kv_heads=2, head_dim=32, kv_dim=64, vocab_size=512)
    print(f"Reference Engine initialized for {cfg.name}")
