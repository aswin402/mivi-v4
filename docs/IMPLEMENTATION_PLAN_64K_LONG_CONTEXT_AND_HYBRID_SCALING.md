# 🚀 Implementation Plan: 64K/128K Long-Context Scaling & Hybrid Architecture for Mivi

**Document Version:** `1.0.0`  
**Target Engine:** `Mivi-v4` (Pure-Rust Hybrid SSM + Attention Inference Engine & Agent Platform)  
**Goal:** Scale Mivi's context window from 4,096 tokens to **65,536 tokens (64K)** and **131,072 tokens (128K)** while maintaining $< 1\text{ GB}$ total RAM footprint and $< 500\text{ ms}$ TTFT via LMCache on consumer CPUs.

---

## 1. Executive Summary & Research Foundation

Scaling transformer context windows traditionally triggers a **quadratic compute wall** ($O(N^2)$) and a **memory wall** (multi-gigabyte KV caches). Mivi breaks through both walls by uniting the latest advances from leading open-source research:

```
┌────────────────────────────────────────────────────────────────────────┐
│               Mivi 64K Long-Context Multi-Source Blueprint              │
└────────────────────────────────────────────────────────────────────────┘

  1. Pokee-Isaac 28B & Liquid AI LFM2.5 (August 2026)
     └── Non-decoder hybrid architecture: 10 SSM layers (0 bytes KV cache)
         + 6 Attention layers (selective GQA).

  2. KIVI & llama.cpp KV Quantization (Q8_0 / Q4_0)
     └── Compresses Key and Value tensors into 8-bit / 4-bit blocks.
     └── 64K KV Cache shrinks from 1.61 GB (F32) down to 402 MB (Q8_0).

  3. YaRN (Yet another RoPE extensioN) & LongRoPE
     └── Dynamic base frequency scaling: θ_base = 500,000 to 1,000,000.
     └── Preserves positional resolution up to 64K–128K tokens without retuning.

  4. Chunked Tiled Prefill (vLLM / SGLang / FlashInfer CPU)
     └── Processes long prompts in 512/1024-token tiles to maximize L2/L3 cache residency.

  5. Andrej Karpathy (llama2.c) & Fareed Khan (kimi-k3-in-c)
     └── Zero-heap forward pass scratchpads + Top-P cutoff filtering.
     └── Tiered storage-backed memory budget dials (--memory-budget-mb).
```

---

## 2. Memory & Hardware Requirements @ 64K Context

### Model Configuration (350M–1B Hybrid Model):
- Hidden dimension: `dim = 1024`, `heads = 16`, `kv_heads = 8` (`kv_dim = 512`)
- Layers: 16 layers (10 SSM + 6 Attention)

| Configuration | 4K Context RAM | 16K Context RAM | 64K Context RAM | 128K Context RAM |
|---|:---:|:---:|:---:|:---:|
| **Standard Transformer (F32)** | 270 MB | 1.07 GB | 4.29 GB | 8.58 GB |
| **Mivi Hybrid SSM + Attn (F32)** | 100 MB | 402 MB | 1.61 GB | 3.22 GB |
| **Mivi Hybrid SSM + Attn (F16)** | 50 MB | 201 MB | 805 MB | 1.61 GB |
| **Mivi Hybrid SSM + Attn (Q8_0)**| **25 MB** | **100 MB** | **402 MB** | **805 MB** |
| **Mivi Hybrid SSM + Attn (Q4_0)**| **12.5 MB**| **50 MB** | **201 MB** | **402 MB** |

> [!TIP]
> At **Q8_0 quantization**, Mivi's 64K KV cache takes only **402 MB** of RAM! Adding base model weights (~400MB), the entire 64K engine runs comfortably inside **~800 MB RAM** on any standard laptop or cloud instance.

---

## 3. Core Architectural Modules

```
                          ┌─────────────────────────────┐
                          │   64K Prompt Stream (CLI)   │
                          └──────────────┬──────────────┘
                                         │
                                         ▼
                          ┌─────────────────────────────┐
                          │    Chunked Tiled Prefill    │
                          │   (512-Token L2/L3 Tiles)   │
                          └──────────────┬──────────────┘
                                         │
                     ┌───────────────────┴───────────────────┐
                     ▼                                       ▼
       ┌───────────────────────────┐           ┌───────────────────────────┐
       │   10 Linear SSM Layers    │           │    6 Attention Layers     │
       │  • Recurrent state: 500 KB│           │  • Dynamic Q8_0 KV Cache  │
       │  • O(1) Memory Footprint  │           │  • YaRN RoPE Base Scaler  │
       └─────────────┬─────────────┘           └─────────────┬─────────────┘
                     │                                       │
                     └───────────────────┬───────────────────┘
                                         │
                                         ▼
                          ┌─────────────────────────────┐
                          │  LMCache 64K Prefix Index   │
                          │ (7.0x Faster TTFT on Turns) │
                          └──────────────┬──────────────┘
                                         │
                                         ▼
                          ┌─────────────────────────────┐
                          │ 262k Pushdown Grammar Mask  │
                          │  & llama2.c Cutoff Sampler  │
                          └─────────────────────────────┘
```

---

## 4. Phase-by-Phase Implementation Plan

### 📋 Phase 1: RoPE Frequency Scaling & High-Position Extrapolation
- [ ] **Dynamic RoPE Base Calculation** (`crates/mivi-core/src/rope.rs`):
  - Scale base frequency $\theta_{\text{base}}$ automatically when `ctx_size > orig_ctx_size`:
    $$\theta_{\text{scaled}} = \theta_{\text{base}} \times \left( \frac{\text{ctx\_size}}{\text{orig\_ctx\_size}} \right)^{\frac{d}{d-2}}$$
  - Support YaRN / NTK-aware frequency scaling for seamless 64K extrapolation without perplexity explosion.
- [ ] **Precomputed Trigonometric Tables** (`crates/mivi-core/src/rope.rs`):
  - Precompute `cos` and `sin` tables up to 65,536 positions at model startup for zero trigonometric overhead in the forward pass.

### 📋 Phase 2: Q8_0 & Q4_0 Quantized KV Cache
- [ ] **Quantized KV Tensor Blocks** (`crates/mivi-kv/src/cache.rs`):
  - Add `KvCachePrecision` enum: `F32`, `F16`, `Q8_0`, `Q4_0`.
  - Store Key and Value tokens in 32-element quantized blocks with FP16/FP32 scale factors.
- [ ] **Fused Vectorized Attention Kernel** (`crates/mivi-model/src/transformer.rs`):
  - Implement AVX2 / NEON fused dot-products for $Q \cdot K_{\text{quant}}^T$ and $A \cdot V_{\text{quant}}$.
  - Verify that selective layer mapping (`layer_map`) continues to allocate memory exclusively for the 6 attention layers.

### 📋 Phase 3: Chunked Tiled Prefill & Cache-Blocking
- [ ] **Cache-Friendly Chunking Loop** (`crates/mivi-model/src/model.rs`):
  - Break cold prompt sequences into chunks of `CHUNK_SIZE = 512` or `1024` tokens.
  - Compute SSM states and Attention KV projections chunk-by-chunk, keeping intermediate activations resident in CPU L2/L3 cache (16MB–32MB).
- [ ] **LMCache 64K Prefix Hash Acceleration** (`crates/mivi-kv/src/prefix.rs`):
  - Hash 64K prompt prefixes in 64-token chunks.
  - Enable instant sub-second resumption for long code repositories and large documents.

### 📋 Phase 4: Long-Context Retrieval & Needle-In-A-Haystack (NIAH) Test Suite
- [ ] **Automated NIAH / RULER Harness** (`tests/long_context_retrieval.rs`):
  - Insert secret passkeys/facts at varying depths (0%, 25%, 50%, 75%, 100%) across 4K, 16K, 32K, and 64K tokens.
  - Verify 100% retrieval accuracy across all depths.
- [ ] **Multi-Turn Stress Testing** (`tests/multi_turn_64k_stress.rs`):
  - Test 50+ turn conversations with continuous tool executions to verify zero memory leaks and stable token throughput.

### 📋 Phase 5: CLI, Server & Memory Budget Ergonomics
- [ ] **CLI Flags & Configuration** (`crates/mivi-cli/src/commands.rs`):
  - Add `--ctx-size <TOKENS>` default support up to `65536`.
  - Add `--quant-kv <f32|f16|q8_0|q4_0>` flag.
  - Expose `--memory-budget-mb <MB>` to automatically pick the optimal KV precision based on available system RAM.
- [ ] **Server Telemetry** (`crates/mivi-server/src/routes/models.rs`):
  - Expose `max_context_length = 65536` in `/v1/models` and `/v1/mivi/status`.

---

## 5. Master TODO Task List

- [ ] **Task 1: RoPE YaRN / NTK-Aware Scaling**
  - [ ] Implement `RopeConfig` with `base_frequency`, `scale_factor`, and `yarn_alpha/beta`.
  - [ ] Extend precomputed RoPE tables from 4,096 to 65,536 positions.
  - [ ] Add unit tests for positional rotation accuracy up to 65,536.
- [ ] **Task 2: Quantized KV Cache (Q8_0 / Q4_0)**
  - [ ] Implement `QuantizedKvCache` in `mivi-kv`.
  - [ ] Add block-level dequantization and fused SIMD dot-product for attention scores.
  - [ ] Verify 402 MB RAM footprint at 64K tokens.
- [ ] **Task 3: Chunked Tiled Prefill Engine**
  - [ ] Implement `forward_prefill_chunked` in `mivi-model`.
  - [ ] Add L2 cache blocking for 512-token tiles.
- [ ] **Task 4: Long-Context RULER Benchmark Suite**
  - [ ] Create `tests/long_context_retrieval.rs` with synthetic needle injection.
  - [ ] Benchmark retrieval across 4k, 16k, 32k, and 64k token depths.
- [ ] **Task 5: Server & CLI Integration**
  - [ ] Add `--ctx-size` and `--quant-kv` flags to `mivi chat`, `mivi serve`, and `mivi bench`.
  - [ ] Update `CHANGELOG.md` with source citations and bump version.

---

## 6. Key Citations & Source References

1. **Pokee AI & explainx.ai**: *Pokee-Isaac 28B: A Real 10M-Token Context Model on One GPU*, [explainx.ai](https://explainx.ai/blog/pokee-isaac-28b-10m-context-agentic-model-august-2026) (2026).
2. **Bowen Peng et al.**: *YaRN: Efficient Context Window Extension of Large Language Models*, [arXiv:2309.00071](https://arxiv.org/abs/2309.00071) (2023).
3. **Yuhong Li et al.**: *LongRoPE: Extending LLM Context Window to 2 Million Tokens*, [arXiv:2402.13753](https://arxiv.org/abs/2402.13753) (2024).
4. **Zirui Liu et al.**: *KIVI: A Tuning-Free 2-bit/4-bit Quantization Method for KV Cache*, [arXiv:2402.02750](https://arxiv.org/abs/2402.02750) (2024).
5. **Andrej Karpathy**: *llama2.c - Minimalist Pure C LLM Inference*, [GitHub](https://github.com/karpathy/llama2.c) (2023).
6. **Fareed Khan**: *kimi-k3-in-c - Trillion-Scale MoE on Consumer Hardware*, [GitHub](https://github.com/FareedKhan-dev/kimi-k3-in-c) (2024).
