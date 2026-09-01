# 🔬 Architectural Blueprint: Sparse Mixture-of-Experts (MoE), Hybrid SSM+Attention & Tiered Storage Streaming

**Document Version:** `1.0.0`  
**Target Engine:** `Mivi-v4` (Pure-Rust Hybrid SSM + Attention Inference Engine & Agent System)  
**Foundational Inspirations:**  
- **Andrej Karpathy (`llama2.c`)**: Zero-heap forward pass (`RunState`), sub-microsecond Top-P cutoff filtering, cache-aligned row-major linear algebra.
- **Fareed Khan (`kimi-k3-in-c`)**: Sparse Mixture-of-Experts, dense trunk vs sparse expert partitioning, dynamic LRU expert caching, NVMe storage-backed inference.
- **DeepSeek-V3 / Moonshot Kimi K3 / Jamba**: Shared + Routed Expert architectures, Multi-Head Latent Attention (MLA), auxiliary-loss-free routing.

---

## 1. Executive Summary & Vision

Modern small and mid-sized language models (SLMs/LLMs) face two conflicting challenges on consumer hardware:
1. **Compute & Context Scaling**: Dense models must evaluate 100% of parameters for every token, resulting in high latency and quadratic attention overhead on long sequences.
2. **RAM & Hardware Limits**: Edge devices, laptops, and budget cloud VPS instances often have strict memory budgets (e.g. 2GB–8GB RAM), preventing the deployment of larger, more capable models.

### The Mivi Solution: The Triad Architecture
By combining three complementary paradigms, Mivi achieves state-of-the-art capability within consumer hardware limits:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Mivi Triad Architecture                         │
└────────────────────────────────────────────────────────────────────────┘
  1. Hybrid Linear SSM + Attention (LFM2 / Mamba)
     └── Constant-time O(1) state recurrence + Sparse global attention
  
  2. Sparse Mixture-of-Experts (MoE) Routing
     └── Activating only K out of E experts per token (e.g. 2 of 16)
  
  3. Tiered Storage-Backed Memory & LRU Expert Caching
     └── Dense trunk pinned in RAM; Sparse experts streamed from NVMe SSD
```

---

## 2. Core Architectural Components

### 2.1 The Hybrid SSM + Attention + MoE Block
In standard transformers, every layer contains a dense Feed-Forward Network (FFN) that accounts for ~65% of total parameters. In Mivi's hybrid MoE design, layers are heterogeneous:

1. **SSM Layers (Recurrent Trunk)**: Linear state-space recurrence (Mamba/LFM2) for fast, infinite-context prefill with constant memory footprint.
2. **Attention Layers (Global Recall)**: Grouped Query Attention (GQA) with rotary embeddings (RoPE) at strategic intervals (e.g. every 3rd or 4th layer) for high-precision recall.
3. **Sparse MoE FFN Layers**: Replacing dense SwiGLU with a Gating Router and an ensemble of $E$ specialized expert networks, activating only the Top-$K$ ($K \ll E$) per token.

```
                  Input Activation: x [dim]
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
      SSM / Attention                     Gating Router
      Residual Layer                   W_gate [num_exp, dim]
            │                                 │
            │                           Top-K Selection
            │                           & Softmax Weights
            │                                 │
            ▼                                 ▼
       LayerNorm                         Active Experts
            │                        FFN_i(x) for i ∈ Top-K
            │                                 │
            └───────────────┬─────────────────┘
                            ▼
                    Weighted Sum:
            y = x + LayerNorm(Σ w_i · FFN_i(x))
```

---

## 3. Mathematical Formulation

### 3.1 Top-K Softmax Gating Router
Given input token activation $x \in \mathbb{R}^d$, the router computes raw routing logits for $E$ experts:
$$H(x) = W_{\text{gate}} \cdot x + b_{\text{gate}} \quad \text{where } W_{\text{gate}} \in \mathbb{R}^{E \times d}$$

The Top-$K$ indices $\mathcal{T} \subset \{1, \dots, E\}$ are selected:
$$\mathcal{T} = \text{TopK}(H(x), K)$$

Routing weights $w_i$ for $i \in \mathcal{T}$ are normalized via Softmax:
$$w_i = \frac{\exp(H(x)_i)}{\sum_{j \in \mathcal{T}} \exp(H(x)_j)}$$

### 3.2 Shared Expert + Routed Experts Composition (DeepSeek-V3 / Kimi Style)
To ensure stable baseline representation, the feedforward output combines a permanently active **Shared Expert** with the **Routed Experts**:
$$y_{\text{ffn}} = \text{FFN}_{\text{shared}}(x) + \sum_{i \in \mathcal{T}} w_i \cdot \text{FFN}_{\text{expert}_i}(x)$$

Each expert $\text{FFN}_i$ implements SwiGLU:
$$\text{FFN}_i(x) = W_{\text{down}}^{(i)} \cdot \left( \text{SiLU}(W_{\text{gate}}^{(i)} \cdot x) \odot (W_{\text{up}}^{(i)} \cdot x) \right)$$

---

## 4. Tiered Storage-Backed Memory & LRU Expert Caching

### 4.1 Memory Partitioning Strategy (Inspired by `kimi-k3-in-c`)
Rather than requiring the entire model to fit into RAM, memory is partitioned into three tiers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Tiered Memory Hierarchy                         │
└────────────────────────────────────────────────────────────────────────┘

  TIER 1: Always-On Dense State (Pinned in RAM • ~300MB - 1GB)
  ├── Token Embeddings & Output Projections (Q4_K / Q8_0)
  ├── RMSNorm Weights & Biases (F32 / F16)
  ├── SSM Recurrent States (Conv states + SSM hidden states)
  ├── Attention Q, K, V Projections & Gating Router Weights
  └── Shared Expert FFNs

  TIER 2: Dynamic LRU Expert Cache (In-Memory Hot Pool • ~500MB - 2GB)
  ├── Retains the C most recently activated experts (e.g. 16 of 64 experts)
  └── O(1) cache hits on repetitive domains, code blocks, and conversation turns

  TIER 3: NVMe SSD Storage Pool (Storage-Backed Cold Pool • 5GB - 50GB+)
  ├── Memory-mapped GGUF file containing all E sparse expert weights
  └── Asynchronous prefetching via posix_madvise(MADV_WILLNEED)
```

### 4.2 Asynchronous Page Prefetching
When layer $L$ is executing its attention forward pass, the router for layer $L+1$ can predict candidate experts and issue asynchronous OS page prefetch hints:
```rust
#[inline]
pub fn prefetch_expert_weights(file_offset: usize, byte_len: usize, mmap_ptr: *const u8) {
    #[cfg(unix)]
    unsafe {
        libc::madvise(
            mmap_ptr.add(file_offset) as *mut libc::c_void,
            byte_len,
            libc::MADV_WILLNEED,
        );
    }
}
```

---

## 5. Rust Data Structures & API Design

### 5.1 MoE Configuration (`crates/mivi-model/src/config.rs`)
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeConfig {
    pub num_experts: usize,          // Total experts E (e.g. 8, 16, 64)
    pub num_active_experts: usize,   // Active experts K per token (e.g. 2, 4)
    pub shared_experts: usize,       // Number of shared always-on experts (e.g. 1)
    pub expert_hidden_dim: usize,    // Intermediate FFN hidden dimension
    pub routing_temperature: f32,    // Softmax temperature for gating
}
```

### 5.2 Gating Router (`crates/mivi-model/src/moe/router.rs`)
```rust
pub struct TopKRouter {
    pub gate_weights: QuantizedTensor, // [num_experts, dim]
    pub num_active: usize,
    pub temperature: f32,
}

impl TopKRouter {
    pub fn route(&self, x: &[f32], scratch: &mut [f32]) -> Vec<(usize, f32)> {
        // 1. Compute gating logits: logits[e] = dot_product(gate_weights[e], x)
        // 2. Select Top-K indices using partial quickselect
        // 3. Normalize Top-K logits with Softmax
        // 4. Return Vec of (expert_id, weight) pairs
    }
}
```

### 5.3 Dynamic Expert LRU Cache (`crates/mivi-model/src/moe/cache.rs`)
```rust
pub struct ExpertLruCache {
    capacity_bytes: usize,
    current_bytes: usize,
    cache: std::collections::HashMap<(usize, usize), ExpertWeights>, // (layer, expert_id)
    lru_order: std::collections::VecDeque<(usize, usize)>,
}

impl ExpertLruCache {
    pub fn get_or_load(
        &mut self,
        layer_idx: usize,
        expert_id: usize,
        loader: &GgufFile,
    ) -> Result<&ExpertWeights> {
        let key = (layer_idx, expert_id);
        if self.cache.contains_key(&key) {
            self.touch(&key);
            return Ok(self.cache.get(&key).unwrap());
        }
        self.evict_if_needed(loader.expert_size_bytes(layer_idx, expert_id));
        let weights = loader.load_expert(layer_idx, expert_id)?;
        self.insert(key, weights);
        Ok(self.cache.get(&key).unwrap())
    }
}
```

---

## 6. Implementation Roadmap

| Phase | Milestone | Deliverables | Verification Metric |
|---|---|---|---|
| **Phase 1** | **MoE Config & GGUF Metadata** | Add `BlockType::Moe`, `MoeConfig`, parse GGUF MoE tensors (`ffn_gate_exps`, `ffn_down_exps`, `ffn_up_exps`, `ffn_gate_inp`). | 100% GGUF tensor validation tests pass. |
| **Phase 2** | **In-Memory MoE Forward Pass** | Implement `TopKRouter`, SwiGLU expert accumulation, and SIMD integer dot-products for expert matrices. | Bit-identical forward pass vs PyTorch reference. |
| **Phase 3** | **Storage-Backed `ExpertLruCache`** | Implement tiered memory manager with LRU eviction and memory budget dial (`--memory-budget-mb`). | Zero OOMs under constrained RAM workloads (e.g. 2GB budget). |
| **Phase 4** | **Async Prefetching & Server Integration** | Overlap `madvise` disk prefetching with attention passes; expose MoE telemetry in `/v1/mivi/status`. | Sustained interactive tokens/sec with minimal disk stall latency. |

---

## 7. Key Citations & Source References

1. **Andrej Karpathy**: *llama2.c - Inference Llama 2 in one file of pure C*, [GitHub Repository](https://github.com/karpathy/llama2.c) (2023).
2. **Fareed Khan**: *kimi-k3-in-c - Running 2.78T parameter MoE models on consumer CPUs with 8GB RAM*, [GitHub Repository](https://github.com/FareedKhan-dev/kimi-k3-in-c) (2024).
3. **DeepSeek AI**: *DeepSeek-V3 Technical Report - Multi-Head Latent Attention & DeepSeekMoE Architecture*, [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) (2024).
4. **Moonshot AI**: *Kimi K3 Technical Architecture & Large-Scale MoE Sparsity*, Moonshot AI Research (2024).
5. **AI21 Labs**: *Jamba: A Hybrid Transformer-Mamba Language Model*, [arXiv:2403.19887](https://arxiv.org/abs/2403.19887) (2024).
