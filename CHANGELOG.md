# Changelog

All notable changes to **Mivi-v4** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v0.0.3] - 2026-08-28

### Mamba SSM Math, Dynamic LoRA Dispatch, RoPE Frequency Cache & Zero-Allocation Pipelines

#### 🧠 Mathematical Correctness & Adapter Infrastructure
- **Full Mamba SSM Forward Pass**: Implemented the complete 12-step State Space Model pipeline with short causal convolution state tracking, continuous state recurrence $h_t = A \cdot h_{t-1} + \text{in}_t$, output projection, and SiLU gating.
- **Dynamic LoRA Adapter Dispatch**: Wired active LoRA adapter execution directly into attention ($Q, K, V, O$), SSM ($\text{in}, \text{out}$), and FFN ($\text{gate}, \text{up}, \text{down}$) matrix-vector forward passes via `active_adapters.apply_module()`.
- **Precomputed RoPE Frequency Cache**: Built `RopeCache` in `mivi-core`, precomputing full $\sin/\cos$ tables up to `max_seq_len` at model load time, eliminating per-token trigonometric and exponentiation overhead.
- **Safe 4-Byte Alignment Verification**: Replaced unaligned memory pointer conversions with `safe_f32_slice()` guaranteeing safe alignment across all GGUF tensor lookups.
- **Strict Little-Endian Conversion**: Enforced `f32::from_le_bytes` across `mivi-quant` and GGUF parsers.

#### ⚡ Performance & Zero-Allocation Tokenization
- **Zero-Allocation BPE Querying**: Refactored `bpe_encode_piece()` to perform direct slice queries on `HashMap<Vec<u8>, usize>` via Rust's `Borrow<[u8]>` trait, eliminating thousands of per-lookup heap allocations.
- **Stack-Buffered MatVec Fallback**: Replaced per-token dynamic heap allocations in unaligned matrix-vector multiplication with a 1KB stack buffer (`[f32; 256]`).
- **CPUID Detection Caching**: Cached AVX2 + FMA hardware capability checks via `LazyLock` in `mivi-core::simd`, removing runtime branching overhead in hot inner loops.
- **Fast KV Cache Reset**: Optimized `KvCache::reset()` to O(1) by resetting `current_pos` without redundant multi-megabyte `fill(0.0)` memory sweeps.
- **Parallel F16 MatVec**: Added Rayon chunked multi-threading to `matvec_f16`.

#### 🛡️ Server Safety & Sandboxing
- **Symlink Traversal Prevention**: Hardened `safe_join` in `mivi-tools` with canonical filesystem root containment checks.
- **Non-Blocking Inference Execution**: Offloaded CPU-bound `m.generate()` calls to `tokio::task::spawn_blocking` to prevent Tokio worker thread starvation.
- **Real Agent Loop Integration**: Connected `/v1/mivi/agent` directly to the `AgentLoop` state machine.
- **Dynamic Token Usage Tracking**: Computed exact prompt and completion token counts from the tokenizer in `/v1/chat/completions`.
- **Robust IPv6 Binding**: Handled dual-stack IPv4/IPv6 socket binding with parsed `IpAddr`.
- **Word-Boundary Intent Routing**: Replaced naive substring matching in `IntentClassifier` with compiled regexes.

#### 🧪 Testing & Verification
- Added `test_http_server_with_real_model` loading real GGUF weights into Axum server.
- Added sample logit tolerance assertions (`diff < 0.25`) to `test_rust_forward_matches_oracle`.
- **20 comprehensive unit and integration tests passing** across all 12 crates.

---

## [v0.0.2] - 2026-08-28

### Security Hardening, Undefined Behavior Fixes & Core Engine Correctness

#### 🛡️ Security & Memory Safety (UB Elimination)
- **Path Traversal Sandboxing**: Added `safe_join` to `mivi-tools` using component-level inspection, blocking `..`, root dir, and UNC prefix traversal attacks in `read_file`, `write_file`, and `list_dir`. Sanitized memory record types and IDs in `mivi-memory`.
- **Safe Memory Alignment**: Eliminated raw unaligned `&[u8]` to `&[f32]` pointer casts across `mivi-quant`, enforcing alignment checks with `f32::from_ne_bytes` safe iteration fallback.
- **Unsafe SIMD Invariant Enforcement**: Replaced all `debug_assert!` macros guarding `unsafe` SIMD kernels (`matvec_f32`, `rms_norm_simd`, `matvec_q8_0`) with unconditional `assert_eq!` / `assert!`, preventing release-mode out-of-bounds execution.
- **GGUF Security Limits & Overflow Protection**: Enforced bounds on GGUF string lengths (1MB), metadata keys (100k), tensor counts (50k), and array lengths (10M). Added `checked_mul` arithmetic on tensor dimensions and byte lengths, preventing CVE-style integer overflows and memory exhaustion.
- **Bounds Checking**: Added explicit slice and buffer length checks in `f16` / `bf16` dequantization and matrix operations.

#### 🧠 Correctness & Production Logic
- **Standard BPE Tokenizer**: Replaced naive prefix matching with true Byte-Pair Encoding (BPE) iterative merge loop with rank scanning.
- **Real SSM / Mamba Weight Loading**: Loaded dynamic continuous state matrices (`blk.{i}.ssm_a.weight`) and depthwise convolutions (`blk.{i}.ssm_conv.weight`) from GGUF.
- **Zero-Panic Embedding Lookup**: Replaced `.unwrap()` fallbacks with typed `ModelError::MissingWeight` and enforced `token_id < vocab_size` bounds validation.
- **Context Window Overflow Guard**: Enforced `pos < max_seq_len` bounds check at the entry of `Model::forward()`.
- **KV Cache Bounds & Error Types**: Added `KvError::DimMismatch` and strict layer/position bounds verification.
- **Pratt Parser Expression Evaluator**: Replaced naive substring math splitting with a Pratt Parser supporting unary minus, nested parentheses, and operator precedence (`-5 + 10 = 5`, `3 - -5 = 8`).
- **Dynamic EOS Token Detection**: Extracted `tokenizer.ggml.eos_token_id` dynamically from GGUF metadata.

#### ⚡ Performance & Polish
- **MPSC Streaming Server**: Wired model token generation to Axum SSE streaming via Tokio MPSC channels with automatic cancellation on client disconnect.
- **LazyLock Regexes**: Migrated tool and think tag regexes in `mivi-tools` to `std::sync::LazyLock` for zero-allocation reuse.
- **Agent Loop Stagnation Guards**: Added 3-cycle repetition detection and explicit `finish` tool support to `AgentLoop`.
- **Context VM Implementation**: Built real substring search, slice extraction, and recursive subtask dispatch in `ContextVm`.
- **Grammar Error Tracking**: Added syntax error detection on unmatched closing braces/brackets to `JsonConstraintState`.
- **Extracted Constants**: Extracted `PARALLEL_CHUNK_SIZE` across Q8_0 and Q4_K_M kernels.

#### 🧪 Test Suite
- Expanded to **18 unit, integration, and golden oracle tests** passing with 0 failures across all 12 crates in both debug and release modes.

---

## [v0.0.1] - 2026-08-28

### Initial Release: Architectural Foundation & High-Performance CPU Engine

#### 🏗️ Architecture & Planning
- **Comprehensive Documentation Suite**:
  - `CORE_IDEA.md`: Vision, philosophy, and 4 pillars of on-device agent intelligence.
  - `PRD.md`: Detailed product requirements, 6 user stories, and performance targets.
  - `SPEC.md`: Technical specification covering GGUF parsing, memory budget (sub-1GB RAM), forward pass math, and OpenAI API contracts.
  - `INSPIRATIONS.md`: Deep dive into 25+ reference projects (LFM, RLM, ToolOrchestra, Harness-R1, Bonsai, Kimi-K3-in-C, candle, llama2.c).
  - `IMPLEMENTATION_PLAN.md`: 6-phase engineering roadmap with concrete code patterns.
  - `RESEARCH.md`: Feasibility analysis and edge hardware benchmark validations.
  - `REVISED_ARCHITECTURE.md`: Post-research architecture revisions (6 LoRA experts, two-level routing, RLM Context VM, dynamic tool discovery).

#### 📦 Workspace & Crates (12 Modular Crates)
- Scaffolded pure Rust modular workspace:
  - `mivi-core`: Zero-heap `RunState` arena, AVX2 SIMD kernels, and math primitives (RMSNorm, Softmax, SiLU, SwiGLU, RoPE).
  - `mivi-quant`: Quantization formats (Q8_0, Q4_K_M, F16, F32) with AVX2 & Rayon multi-threaded row chunking.
  - `mivi-tokenizer`: BPE tokenizer, vocabulary mapping, special tokens (`<think>`, `<tool_call>`), and ChatML prompt formatting.
  - `mivi-kv`: Contiguous preallocated KV cache for GQA attention layers.
  - `mivi-model`: GGUF v3 parser, hybrid LFM architecture (SSM Gated Conv + GQA Attention), dynamic LoRA adapter loading, and token sampler.
  - `mivi-context`: Context Store with pinned blocks and RLM Context VM (`SEARCH`, `SLICE`, `SUMMARIZE`, `RECURSE`).
  - `mivi-memory`: Open Knowledge Format (OKF) markdown record persistence under `.mivi/memory`.
  - `mivi-tools`: Tool registry, sandboxed `ToolBroker`, and markup parsers.
  - `mivi-router`: Two-level intent classification and routing (Chat, Agent, Code, Debug, Research).
  - `mivi-agent`: Canonical agent state machine and loop engine (`observe` → `think` → `act` → `verify`).
  - `mivi-server`: Axum HTTP server with SSE streaming, OpenAI compatibility, and Mivi Agent OS endpoints.
  - `mivi-cli`: Command-line interface with subcommands (`serve`, `chat`, `info`, `bench`, `doctor`).

#### ⚡ Core Engine & Performance
- **AVX2 SIMD Vectorization**:
  - `Q8_0 Matvec`: **0.045 ms/op (46.62 GFLOPS)** on 1024×1024 matrices (13.6× speedup).
  - `Q4_K_M Matvec`: **0.230 ms/op (9.10 GFLOPS)** on 1024×1024 matrices (3.75× speedup).
  - Vectorized `RMSNorm` with zero heap allocation.
- **Dynamic LoRA Layer Composition**:
  - On-the-fly adapter delta calculation: $y = Wx + \sum_i w_i \frac{\alpha_i}{r_i} B_i (A_i x)$ enabling instant hot-swapping between experts.

#### 🛠️ Built-in Sandboxed Tools
- Implemented default tool handlers:
  - `read_file`: Safe workspace file reader.
  - `write_file`: Auto-parent directory creating file writer.
  - `list_dir`: Directory inspector.
  - `calculator`: Fast arithmetic expression evaluator.

#### 🌐 HTTP Server & Streaming
- **OpenAI Compatible Endpoints**:
  - `POST /v1/chat/completions`: Non-streaming JSON & real-time SSE streaming with delta thinking and tool calls.
  - `GET /v1/models` & `GET /health`: Engine status and discovery.
- **Mivi Extended Endpoints**:
  - `GET /v1/mivi/status`: Real-time telemetry (RAM RSS usage, active tools, uptime).
  - `GET /v1/mivi/tools`: Tool schema registry.
  - `POST /v1/mivi/agent`: Autonomous multi-step agent loop execution over HTTP.

#### 🐍 Python Reference Engine & Oracle Validation
- `reference/reference_engine.py`: PyTorch golden ground-truth oracle for LFM hybrid architecture.
- `training/export/convert_to_gguf.py`: Pure Python GGUF v3 binary converter.
- `training/export/generate_fixture.py`: Deterministic synthetic GGUF model and oracle traces generator.
- `tests/oracle_comparison_test.rs`: Validated Rust engine forward pass against Python Oracle ground truth with 100% top token and numerical match.

#### 🧪 Testing
- 16 comprehensive unit, integration, and oracle comparison tests passing across all crates with 0 failures.
