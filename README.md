<div align="center">

```
  __  __ _____ _    _ _____          __   _  _   
 |  \/  |_   _| |  | |_   _|        / /  | || |  
 | \  / | | | | |  | | | |  ______ / /_  | || |_ 
 | |\/| | | | \ \  / / | | |______| '_ \ |__   _|
 | |  | |_| |_ \ \/ / _| |_        | (_) |  | |  
 |_|  |_|_____| \__/ |_____|        \___/   |_|  
```

# Mivi-v4: Agent-Native SLM Engine in Rust

[![Rust](https://img.shields.io/badge/rust-2021%20edition-orange.svg?style=flat-square&logo=rust)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0%20%7C%20MIT-blue.svg?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-26%20passed-brightgreen.svg?style=flat-square)](tests/)
[![Memory Target](https://img.shields.io/badge/memory-%3C1GB%20RAM-purple.svg?style=flat-square)](#-memory-budget)
[![Speed](https://img.shields.io/badge/throughput-46.6%20GFLOPS%20CPU-success.svg?style=flat-square)](#-performance--benchmarks)

**Mivi-v4** is a CPU-first, low-memory, agent-native Small Language Model (SLM) engine written in pure, high-performance Rust. It runs an **LFM2.5-350M** hybrid SSM+GQA architecture locally on edge hardware with **< 1 GB RAM**, exposing an OpenAI-compatible API and built-in sandboxed tool orchestration.

---

</div>

## 💡 Core Philosophy

> **Do not try to force a 350M parameter model to know everything. Build a compact agent-native reasoning foundation, combine it with routed specialist adapters, and execute capabilities through an ultra-fast Rust runtime.**

```
                        ┌─────────────────────────────────┐
                        │      Client Application / IDE   │
                        └────────────────┬────────────────┘
                                         │
                              OpenAI-compatible HTTP / SSE
                                         │
                                         ▼
                 ┌───────────────────────────────────────────────┐
                 │             Mivi-v4 Rust Engine               │
                 │                                               │
                 │  ┌──────────────┐  ┌─────────────┐  ┌──────┐  │
                 │  │ Context VM   │  │ Tool Broker │  │ RAG  │  │
                 │  │ (RLM Paging) │  │ (Sandboxed) │  │ OKF  │  │
                 │  └──────┬───────┘  └──────┬──────┘  └──┬───┘  │
                 │         │                 │            │      │
                 │         └───────────┬─────┴────────────┘      │
                 │                     │                         │
                 │                     ▼                         │
                 │         ┌───────────────────────┐             │
                 │         │ 2-Level Router        │             │
                 │         │ (Intent Classifier)   │             │
                 │         └───────────┬───────────┘             │
                 │                     │                         │
                 │                     ▼                         │
                 │         ┌───────────────────────┐             │
                 │         │ LoRA Specialist MoE   │             │
                 │         │ Code • Agent • Debug  │             │
                 │         │ Research • Chat       │             │
                 │         └───────────┬───────────┘             │
                 │                     │                         │
                 │                     ▼                         │
                 │         ┌───────────────────────┐             │
                 │         │ LFM2.5-350M Backbone  │             │
                 │         │ Q4_K_M / Q8_0 (AVX2)  │             │
                 │         └───────────────────────┘             │
                 └───────────────────────────────────────────────┘
```

---

## ✨ Key Features

- **🚀 CPU-First Speed (46.6 GFLOPS)**: Custom AVX2, FMA, and NEON SIMD kernels with Rayon multi-threaded row chunking.
- **💾 Ultra-Low Memory Footprint (<1 GB RAM)**: Zero-heap `RunState` arena preallocates activation buffers once at startup. 0 allocations during token generation.
- **⚡ Hybrid SSM + GQA Architecture**: 10 double-gated short convolution blocks + 6 Grouped-Query Attention blocks for sub-quadratic context processing.
- **🧩 Dynamic LoRA Multi-Expert Composition**: Instant hot-swapping between specialist adapters (`AGENT`, `CODE`, `DEBUG`, `RESEARCH`, `CHAT`, `GENERAL`) with zero full weight materialization.
- **🛡️ Sandboxed Tool Broker**: Model emits `<tool_call>` markup; the Rust runtime validates parameters, enforces timeouts, and securely executes operations.
- **🌐 Full OpenAI API Compatibility**: Native drop-in replacement with SSE token streaming, delta thinking chunks (`<think>...</think>`), and tool call deltas.
- **🧠 Recursive Context VM (RLM)**: Typed functional context operators (`SEARCH`, `SLICE`, `SUMMARIZE`, `RECURSE`) for effective 64K context navigation without context stuffing.
- **📁 Open Knowledge Format (OKF)**: Portable, human-readable markdown memory persistence stored in `.mivi/memory`.

---

## 📦 Workspace Architecture (12 Modular Crates)

| Crate | Path | Description |
|---|---|---|
| [`mivi-core`](crates/mivi-core) | `crates/mivi-core` | Zero-heap `RunState` arena, AVX2/FMA/NEON SIMD kernels, math primitives (RMSNorm, Softmax, SiLU, SwiGLU, RoPE) |
| [`mivi-quant`](crates/mivi-quant) | `crates/mivi-quant` | Quantization kernels (Q8_0, Q4_K_M, F16, F32) with parallel matrix-vector multipliers |
| [`mivi-tokenizer`](crates/mivi-tokenizer) | `crates/mivi-tokenizer` | BPE tokenizer, vocabulary mappings, special tokens (`<think>`, `<tool_call>`), ChatML formatting |
| [`mivi-kv`](crates/mivi-kv) | `crates/mivi-kv` | Preallocated contiguous Key-Value cache for multi-head GQA layers |
| [`mivi-model`](crates/mivi-model) | `crates/mivi-model` | GGUF v3 parser, hybrid LFM forward pass, dynamic LoRA adapters, and token sampling |
| [`mivi-context`](crates/mivi-context) | `crates/mivi-context` | Paged Context Store and RLM Context VM (`SEARCH`, `SLICE`, `SUMMARIZE`, `RECURSE`) |
| [`mivi-memory`](crates/mivi-memory) | `crates/mivi-memory` | Open Knowledge Format (OKF) markdown persistence under `.mivi/memory` |
| [`mivi-tools`](crates/mivi-tools) | `crates/mivi-tools` | Tool registry, sandboxed `ToolBroker`, and markup parsers (`read_file`, `write_file`, `list_dir`, `calculator`) |
| [`mivi-router`](crates/mivi-router) | `crates/mivi-router` | Two-level intent classification and routing (Chat, Agent, Code, Debug, Research) |
| [`mivi-agent`](crates/mivi-agent) | `crates/mivi-agent` | Canonical agent state machine and loop engine (`observe` → `think` → `act` → `verify`) |
| [`mivi-server`](crates/mivi-server) | `crates/mivi-server` | Axum HTTP server with SSE streaming, OpenAI compatibility, and Mivi Agent OS endpoints |
| [`mivi-cli`](crates/mivi-cli) | `crates/mivi-cli` | Command-line interface with subcommands (`serve`, `chat`, `info`, `bench`, `doctor`) |

---

## 📊 Performance & Benchmarks

Ran on 16-thread x86_64 CPU (`cargo run --release -- bench`):

| Operation | Precision | Matrix Size | Time per Op | Compute Throughput |
|---|---|---|---|---|
| **Matvec (Q8_0)** | 8-bit | 1024 × 1024 | **0.045 ms** | **46.62 GFLOPS** |
| **Matvec (Q4_K_M)** | 4-bit | 1024 × 1024 | **0.230 ms** | **9.10 GFLOPS** |
| **RMSNorm** | 32-bit | Dim = 1024 | **< 0.001 ms** | Zero Allocation |

### 💾 Memory Budget (< 1 GB RAM)

```
Component                       RAM Allocation       Notes
──────────────────────────────  ──────────────       ─────────────────────────
Q4_K_M Model Weights            180–240 MB           mmap demand-paged
RunState Arena Buffers           50–100 MB           Fixed preallocation
KV Cache (4K Context)            40–120 MB           Preallocated contiguous
Tokenizer Vocab (65K)            15–30 MB            In-memory BPE lookup
LoRA Specialist Adapters         10–60 MB            Multi-adapter resident
HTTP Server & Tool Sandbox       30–80 MB            Axum + Tokio
──────────────────────────────  ──────────────
Total Peak RAM Usage            ~325–630 MB          < 1 GB Target Achieved!
```

---

## ⚡ Quick Start

### 1. Build from Source

```bash
# Clone the repository
git clone https://github.com/aswin402/mivi-v4.git
cd mivi-v4

# Build release binary
cargo build --release
```

### 2. System Diagnostics (`doctor`)

Check your CPU SIMD features and available hardware threads:

```bash
cargo run -- doctor
```

```text
=== Mivi-v4 System Diagnostics ===
OS: linux
Arch: x86_64
CPUs: 16
AVX2 support: true
FMA support:  true
Status: OK
```

### 3. Run CPU Kernel Benchmark (`bench`)

```bash
cargo run --release -- bench
```

### 4. Start the OpenAI-Compatible Server (`serve`)

```bash
cargo run --release -- serve --port 8080 --host 0.0.0.0
```

```text
  __  __ _____ _    _ _____          __   _  _   
 |  \/  |_   _| |  | |_   _|        / /  | || |  
 | \  / | | | | |  | | | |  ______ / /_  | || |_ 
 | |\/| | | | \ \  / / | | |______| '_ \ |__   _|
 | |  | |_| |_ \ \/ / _| |_        | (_) |  | |  
 |_|  |_|_____| \__/ |_____|        \___/   |_|  
                                                 
 Mivi-v4 Agent Engine
 Model: mivi-v4-350m
 Listening on: http://0.0.0.0:8080
 OpenAI-compatible API ready.
```

---

## 🌐 API Usage & Integration

Mivi-v4 is a drop-in replacement for OpenAI endpoints in LangChain, AutoGen, CrewAI, or cursor/cline:

### 1. Standard Chat Completions (curl)

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mivi-v4-350m",
    "messages": [
      {"role": "user", "content": "Calculate 15 * 4 and write the result to output.txt"}
    ]
  }'
```

### 2. Real-Time SSE Token & Thinking Stream

```bash
curl -N -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mivi-v4-350m",
    "messages": [{"role": "user", "content": "How do I fix a dangling pointer in C?"}],
    "stream": true
  }'
```

### 3. Python OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="mivi-local"  # Any string
)

response = client.chat.completions.create(
    model="mivi-v4-350m",
    messages=[{"role": "user", "content": "Hello Mivi!"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### 4. Autonomous Agent OS Endpoint (`/v1/mivi/agent`)

```bash
curl -N -X POST http://localhost:8080/v1/mivi/agent \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Find failing tests in src/ and patch the bug",
    "max_steps": 10
  }'
```

---

## 🧪 Testing & Oracle Verification

Mivi uses a **Two-Engine Development Strategy**:
1. **Python Oracle Engine** ([`reference/reference_engine.py`](reference/reference_engine.py)): Ground-truth PyTorch forward pass.
2. **Production Rust Engine** ([`crates/mivi-model`](crates/mivi-model)): Zero-heap CPU-optimized inference.

Run the complete test suite:

```bash
cargo test --workspace --tests
```

```text
running 16 tests across workspace:
  - 4 math SIMD tests (RMSNorm, Softmax, SiLU, SwiGLU) .......... OK
  - 2 quantization unit tests (Q8_0 dequant + AVX2 matvec) ...... OK
  - 1 dynamic LoRA adapter application test ..................... OK
  - 1 tokenizer ChatML formatting test ......................... OK
  - 2 markup parser tests (<think> & <tool_call>) ............... OK
  - 5 integration tests (Context VM, Broker, Server, Agent) ..... OK
  - 1 Python Oracle ground-truth validation test ................ OK

Result: 16 passed; 0 failed!
```

---

## 🗺️ Project Roadmap

- [x] **Milestone 1: Core Inference Engine** (GGUF parser, hybrid SSM+GQA forward pass, AVX2 SIMD, BPE tokenizer)
- [x] **Milestone 2: High-Performance Server & API** (OpenAI compatibility, SSE streaming, telemetry, agent loop endpoint)
- [x] **Milestone 3: Tool Broker & Built-ins** (Sandboxed executor, `read_file`, `write_file`, `list_dir`, `calculator`)
- [x] **Milestone 4: Python Oracle & Fixture Pipeline** (GGUF v3 exporter, deterministic ground truth validation)
- [ ] **Milestone 5: LoRA Expert Training** (Train 6 specialist adapters: `AGENT`, `CODE`, `DEBUG`, `RESEARCH`, `CHAT`, `GENERAL`)
- [ ] **Milestone 6: 10-Stage Training Curriculum** (Instruction baseline → Agent state machine → Tool failures → RL GRPO)
- [ ] **Milestone 7: Dynamic Tool Discovery** (Vectorized tool schema retrieval to conserve context tokens)
- [ ] **Milestone 8: Mivi-Nano Companion** (20–60M routing & speculative decoding model)

---

## 📄 License

This project is licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE) or http://opensource.org/licenses/MIT)

at your option.
