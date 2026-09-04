<div align="center">

```
  __  __ _____ _    _ _____          __   _  _   
 |  \/  |_   _| |  | |_   _|        / /  | || |  
 | \  / | | | | |  | | | |  ______ / /_  | || |_ 
 | |\/| | | | \ \  / / | | |______| '_ \ |__   _|
 | |  | |_| |_ \ \/ / _| |_        | (_) |  | |  
 |_|  |_|_____| \__/ |_____|        \___/   |_|  
```

# ⚡ Mivi-v4: Agent-Native SLM Engine & Server in Pure Rust

[![Rust 2021](https://img.shields.io/badge/rust-2021%20edition-orange.svg?style=flat-square&logo=rust)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0%20%7C%20MIT-blue.svg?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-89%20passed%20(100%25)-brightgreen.svg?style=flat-square)](tests/)
[![Architecture](https://img.shields.io/badge/arch-Hybrid%20SSM%20%2B%20GQA%20Attention-blueviolet.svg?style=flat-square)](#-hybrid-ssm--attention-architecture)
[![Memory](https://img.shields.io/badge/memory-%7E42--260%20MB%20RAM-purple.svg?style=flat-square)](#-memory-footprint--efficiency)
[![Speed](https://img.shields.io/badge/throughput-46.6%20GFLOPS%20CPU-success.svg?style=flat-square)](#-performance--benchmarks)

**Mivi-v4** is a high-performance, ultra-low-memory, agent-native Small Language Model (SLM) inference engine and server written from scratch in **100% pure Rust** with **zero C/C++ dependencies**. 

It runs **Hybrid SSM + GQA Attention** architectures (such as Liquid AI's **LFM2.5-350M**) directly on CPU with native SIMD acceleration (AVX2/FMA/NEON), preallocated zero-heap execution, built-in sandboxed tool orchestration, and a drop-in OpenAI-compatible streaming HTTP server.

---

</div>

## 💡 Why Mivi? (Comparison with llama.cpp and Ollama)

While engines like `llama.cpp` focus primarily on C++ execution for standard Transformers and `Ollama` acts as a Go wrapper around `llama.cpp`, **Mivi is designed from the ground up in memory-safe Rust for Hybrid SLMs and autonomous Agent workflows**.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               MIVI ENGINE STACK                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  CLI & Interactive REPL  │  OpenAI HTTP & SSE Server │  ReAct Agent & Sandbox   │
│  (Slash Commands, Stats) │  (Port Hunting, Watchdog) │  (Pratt Calc, FS Guards) │
├─────────────────────────────────────────────────────────────────────────────────┤
│                    Intent Classifier Router & Context VM                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│            Hybrid LFM2.5 Inference Core (10 SSM ShortConv + 6 GQA Attn)         │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Selective KV Cache (62.5% Savings)  │  Pure Rust SIMD (AVX2/NEON) Quant Kernels│
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 🥊 Technical Comparison

| Dimension | ⚡ **Mivi-v4** | 🦙 **llama.cpp** | 🦙 **Ollama** |
|---|---|---|---|
| **Implementation Language** | **100% Pure Rust** (Zero C/C++ dependencies) | C / C++ | Go (wrapper daemon around llama.cpp) |
| **Architecture Focus** | **Hybrid SLMs** (Gated ShortConv SSM + GQA Attention) | Pure Transformers (Llama, Mistral, Gemma) | Same as llama.cpp |
| **Agent & Tools Support** | **Native Built-in** (ReAct agent loop, Pratt parser calc, sandboxed FS) | ❌ None (Text completion only) | ❌ Needs external framework (LangChain, AutoGen) |
| **KV Cache Footprint** | **Selective Allocation** (Only 6 of 16 layers allocate KV memory; **62.5% savings**) | Allocates full KV cache for all layers | Allocates full KV cache for all layers |
| **RAM Footprint (350M)** | **~42 MB – 260 MB RSS** | ~500 MB – 1.5 GB | ~1 GB – 3 GB+ (Go runtime + subprocesses) |
| **Memory Safety** | **100% Rust Safe Memory** (No segfaults, zero UB) | Manual C/C++ pointer management | Go GC + C++ backend |
| **Single Binary** | **Yes** (Single standalone executable `mivi`) | Multiple CLI binaries & shared libraries | Daemon binary + bundled llama.cpp dynamic libraries |
| **HTTP Server & SSE** | **Built-in Axum server** with dynamic port hunting & watchdog | `llama-server` | Built-in Go API daemon |

---

## ✨ Key Architectural Features

### 1. ⚡ Hybrid SSM + GQA Attention Architecture
Standard LLMs use pure self-attention with quadratic $O(N^2)$ memory and compute costs. Mivi is optimized for hybrid architectures:
- **10 Gated ShortConv SSM Layers**: 1D causal depthwise convolution with linear $O(N)$ time complexity and constant $O(1)$ state memory.
- **6 Grouped-Query Attention (GQA) Layers**: High-precision associative recall with FlashDecoding online softmax.
- **Selective KV Cache**: KV cache is dynamically mapped *only* to attention layers. Non-attention SSM layers consume zero KV memory, reducing RAM requirements by **62.5%**.

### 2. 🛡️ Native Sandboxed Agent & Tool Engine
Mivi eliminates the need for heavyweight Python agent runtimes:
- **Autonomous ReAct Agent Loop**: State machine with observation, reasoning, action, and stagnation guards.
- **Pratt Parser Calculator**: Full recursive descent math engine evaluating mathematical expressions safely with zero `eval()` vulnerabilities.
- **Sandboxed Filesystem (`read_file`, `write_file`, `list_dir`)**: Enforces path canonicalization and directory prefix checks to prevent `../` directory traversal attacks.

### 3. 🌐 OpenAI-Compatible Server with Enterprise Reliability
- **Drop-in Replacement**: Supports `/v1/models`, `/v1/chat/completions` (JSON & SSE streaming), and `/v1/mivi/agent`.
- **Dynamic Port Hunting**: Automatically hunts for available adjacent ports if the requested port is in use.
- **Resource Safety Watchdog**: Monitors process RSS memory every 500ms; issues warnings and performs graceful shutdown if limits are exceeded.
- **Hono-Style Minimalist Logging**: Clean terminal logs reporting method, path, status, latency, and tokens/sec.

### 4. 💬 Modern Interactive Terminal Chat REPL
- **Live Stream Rendering**: Streaming token output with live ANSI styling.
- **Thinking Mode**: Real-time `<think>` trace formatting and duration tracking.
- **Rich Telemetry**: Displays token count, duration, generation speed (tok/s), and real-time process RAM RSS.
- **Slash Commands**: `/help`, `/clear`, `/history`, `/temp`, `/top_p`, `/rep`, `/thinking`, `/exit`.

### 5. ⚡ LMCache-Inspired Prefix Caching & Disk Persistence
- **$O(1)$ Instant Time-To-First-Token (TTFT)**: Input prompts are chunked into 64-token blocks and hashed with 64-bit rolling FNV-1a. Shared system prompts, tool schemas, and multi-turn prefixes hit the in-memory cache and skip forward passes.
- **Hybrid State Snapshotting**: Automatically snapshots both the 6 Attention KV layers and 10 Gated ShortConv SSM convolution states (`conv_states`).
- **On-Disk Persistence (`.mivi/cache/*.kvc`)**: Saves prefilled prompt states to disk, allowing instant zero-prefill startup across process restarts.

---

## 📦 Workspace Architecture (12 Modular Crates)

The codebase is organized into 12 cleanly isolated workspace crates:

| Crate | Directory | Description |
|---|---|---|
| [`mivi-core`](crates/mivi-core) | `crates/mivi-core` | Zero-heap `RunState` arena, AVX2/NEON SIMD dispatch, RMSNorm, Softmax, RoPE cache, and brand constants. |
| [`mivi-quant`](crates/mivi-quant) | `crates/mivi-quant` | Quantization kernels for **Q4_K_M**, **Q6_K**, **Q8_0**, and **F16** with parallel matrix-vector multipliers. |
| [`mivi-kv`](crates/mivi-kv) | `crates/mivi-kv` | Selective-layer KV cache, 64-token chunk prefix caching (`PrefixCache`), and `.kvc` on-disk state persistence. |
| [`mivi-model`](crates/mivi-model) | `crates/mivi-model` | GGUF v3 file parser, LFM2.5 forward pass, FlashDecoding attention, Gated ShortConv SSM, and Min-P/Top-P sampler. |
| [`mivi-tokenizer`](crates/mivi-tokenizer) | `crates/mivi-tokenizer` | GPT-2 byte bijection BPE tokenizer, vocabulary lookup, UTF-8 streaming decoder, and ChatML prompt templating. |
| [`mivi-context`](crates/mivi-context) | `crates/mivi-context` | Persistent conversation store with LRU eviction and micro-VM for context operators. |
| [`mivi-memory`](crates/mivi-memory) | `crates/mivi-memory` | Open Knowledge Format (OKF) markdown-based episodic and semantic persistence. |
| [`mivi-router`](crates/mivi-router) | `crates/mivi-router` | Zero-shot intent classification and query routing across Agent, Code, Debug, Research, and Chat personas. |
| [`mivi-tools`](crates/mivi-tools) | `crates/mivi-tools` | Tool registry, XML `<tool_call>` extraction, Pratt parser calculator, and sandboxed filesystem tools. |
| [`mivi-agent`](crates/mivi-agent) | `crates/mivi-agent` | ReAct agent execution loop with step bounding, stagnation detection, and tool error propagation. |
| [`mivi-server`](crates/mivi-server) | `crates/mivi-server` | Axum HTTP server with SSE streaming, OpenAI compatibility, port fallback hunting, and memory watchdog. |
| [`mivi-cli`](crates/mivi-cli) | `crates/mivi-cli` | CLI entry point with subcommands (`serve`, `chat`, `info`, `bench`, `doctor`). |

---

## 📊 Performance & Benchmarks

Benchmarked on x86_64 CPU (16 threads, AVX2 + FMA):

| Kernel Operation | Quantization | Dimensions | Time per Op | Compute Throughput |
|---|---|---|---|---|
| **Matvec (Q8_0)** | 8-bit | 1024 × 1024 | **0.045 ms** | **46.62 GFLOPS** |
| **Matvec (Q4_K_M)** | 4-bit | 1024 × 1024 | **0.230 ms** | **9.10 GFLOPS** |
| **RMSNorm** | 32-bit | Dim = 1024 | **< 0.001 ms** | Zero Allocation |
| **Token Generation** | Q4_K_M | LFM2.5-350M | **~43 ms / token** | **~23.0 tok/s** (CPU) |

### 💾 Memory Footprint (LFM2.5-350M Q4_K_M)

```
Component                       RAM Allocation       Notes
──────────────────────────────  ──────────────       ─────────────────────────
Q4_K_M Model Weights            ~190–210 MB          Memory-mapped (demand paged)
RunState Activation Buffers     ~30–45 MB            Fixed preallocation (0 heap churn)
Selective KV Cache (4K Context) ~8–15 MB             Allocated only for 6 attention layers
Tokenizer Vocab & BPE Merges    ~15–20 MB            65K token lookup table
Axum HTTP Server & Tool Sandbox ~10–25 MB            Tokio async runtime
──────────────────────────────  ──────────────
Total Peak RAM RSS              ~260 MB              < 300 MB (Full Engine + Model + Server!)
```

---

## ⚡ Quick Start

### 1. Prerequisites

- **Rust**: 1.75+ (2021 edition)
- **Just**: (Optional task runner) `cargo install just`

### 2. Build

```bash
# Clone the repository
git clone https://github.com/aswin402/mivi-v4.git
cd mivi-v4

# Build release binary (uses low-memory 2 concurrent jobs)
just build-release
# Or: cargo build --release --jobs 2
```

### 3. System Diagnostics (`doctor`)

Verify CPU SIMD features and available execution threads:

```bash
just doctor
# Or: cargo run --release -- doctor
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

### 4. Interactive Terminal Chat (`chat`)

Start an interactive chat REPL session:

```bash
just chat
# Or: cargo run --release -- chat --model models/mivi-v4-q4_k_m.gguf
```

```text
  ⚡ Mivi Chat v0.1.2 (LFM2.5-350M • 4K ctx • 42.7 MB RAM)
  Type your prompt, or /help for interactive commands, Ctrl+C to cancel.
  ─────────────────────────────────────────────────────────────────
  user › Write a python function to check if a number is prime
  mivi › ```python
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
```
  ⏱ 1.48s • 47 tokens • 31.7 tok/s • RAM 260.1 MB
```

### 5. Launch the OpenAI-Compatible HTTP Server (`serve`)

```bash
just serve
# Or: cargo run --release -- serve --model models/mivi-v4-q4_k_m.gguf --port 8080 --workspace .
# Public binds (for example --host 0.0.0.0) require MIVI_API_KEY.
# Optional browser access: repeat --cors-origin for each exact allowed origin.
# Example: --cors-origin http://localhost:3000
```

```text
  ╭──────────────────────────────────────────────────────────╮
  │                                                          │
  │   ⚡ Mivi Agent Engine                                   │
  │   Lightweight, Fast & Sandboxed Local Agent Server       │
  │                                                          │
  │   • Model:      mivi                                     │
  │   • Context:    128K tokens                              │
  │   • Local API:  http://127.0.0.1:8080/v1                 │
  │                                                          │
  │   OpenAI-compatible endpoints:                           │
  │   POST /v1/chat/completions (SSE streaming)              │
  │   POST /v1/mivi/agent       (Autonomous loop)            │
  │                                                          │
  ╰──────────────────────────────────────────────────────────╯
```

Inference endpoints require a loaded GGUF model; otherwise they return `503 Service Unavailable`.
The default server does not enable cross-origin browser requests. Configure an explicit deployment
proxy/allowlist if a browser client is required.

OpenAI-compatible requests support validated sampling parameters (`temperature`, `top_p`, `top_k`, `min_p`,
`repetition_penalty`, presence/frequency penalties, and `seed`), custom stop sequences, `none`/`auto`/
named tool choice, and non-streaming `response_format: {"type":"json_object"}`. JSON Schema responses,
JSON streaming, and forced `tool_choice: "required"` are currently rejected explicitly. Anthropic
`/v1/messages` supports validated sampling, `stop_sequences`, and structured streaming tool-use blocks.

Agent context documents must be relative to the configured `--workspace` and are size-bounded before being
added to the prompt.

---

## 🌐 API Usage & Integrations

Mivi-v4 is a drop-in replacement for OpenAI endpoints across tools like **OpenAI Python/TS SDKs**, **LangChain**, **Cursor**, **Continue.dev**, or **cURL**:

### 1. cURL (Standard Chat Completion)

```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mivi",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.2
  }'
```

### 2. cURL (SSE Real-Time Token Streaming)

```bash
curl -N -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mivi",
    "messages": [
      {"role": "user", "content": "Explain binary search in 2 sentences."}
    ],
    "stream": true
  }'
```

### 3. Python OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="mivi-local"  # Not required unless MIVI_API_KEY is set
)

response = client.chat.completions.create(
    model="mivi",
    messages=[
        {"role": "user", "content": "Write a Rust hello world function."}
    ],
    stream=True
)

for chunk in response:
    content = chunk.choices[0].delta.content or ""
    print(content, end="", flush=True)
print()
```

### 4. Autonomous Agent Loop Endpoint (`/v1/mivi/agent`)

Execute multi-step tasks where the engine autonomously plans, runs tools (calculator, filesystem), and returns the final synthesized result:

```bash
curl -X POST http://127.0.0.1:8080/v1/mivi/agent \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Calculate (45 * 12) + 180 and write the result to math_output.txt",
    "max_steps": 5
  }'
```

---

## 🧪 Two-Engine Verification Strategy

Mivi employs a strict **Two-Engine Verification Strategy**:
1. **PyTorch Oracle Engine** ([`reference/reference_engine.py`](reference/reference_engine.py)): Ground-truth reference implementation.
2. **Rust Production Engine** ([`crates/mivi-model`](crates/mivi-model)): High-performance native SIMD implementation.

Every layer forward pass (RMSNorm, RoPE, Attention, ShortConv, SwiGLU) is cross-checked against PyTorch golden outputs.

Run the test suite:

```bash
just test
# Or: cargo test --workspace --jobs 2
```

```text
running 84 tests across workspace:
  - 17 integration tests (Server, Agent, VM, Tokenizer, Prefix Cache, Disk KVC) ... OK
  - 1 PyTorch Oracle Golden Ground-Truth validation test .......................... OK
  - 66 unit tests (SIMD, Math, Quant, KV, Router, Tools, Grammar, PLD) ............ OK

test result: ok. 84 passed; 0 failed; finished in 100% success!
```

---

## 🛠️ Justfile Command Reference

| Command | Description |
|---|---|
| `just build` | Compile workspace in debug mode (max 2 jobs) |
| `just build-release` | Compile optimized release binary |
| `just test` | Run complete 84-test suite |
| `just clippy` | Run Clippy linter with `-D warnings` |
| `just fmt-check` | Verify code formatting with `rustfmt` |
| `just verify` | Run full quality gate (`fmt-check` + `clippy` + `test`) |
| `just chat` | Launch interactive terminal chat REPL |
| `just serve` | Start the OpenAI-compatible HTTP API server |
| `just cache-list` | List all persistent on-disk `.kvc` prefix cache files |
| `just cache-clear` | Clear all persistent on-disk `.kvc` prefix cache files |
| `just doctor` | Check CPU SIMD features and execution environment |
| `just bench` | Benchmark SIMD matrix-vector compute kernels |
| `just info` | Inspect GGUF model metadata, hyperparameters, and tensors |

---

## 📄 License

This project is dual-licensed under:
- **Apache License, Version 2.0** ([LICENSE-APACHE](LICENSE) or [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0))
- **MIT License** ([LICENSE-MIT](LICENSE) or [http://opensource.org/licenses/MIT](http://opensource.org/licenses/MIT))

at your option.
