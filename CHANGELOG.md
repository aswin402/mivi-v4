# Changelog

All notable changes to **Mivi-v4** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
