<div align="center">

```
  __  __ _____ _    _ _____          __   _  _   
 |  \/  |_   _| |  | |_   _|        / /  | || |  
 | \  / | | | | |  | | | |  ______ / /_  | || |_ 
 | |\/| | | | \ \  / / | | |______| '_ \ |__   _|
 | |  | |_| |_ \ \/ / _| |_        | (_) |  | |  
 |_|  |_|_____| \__/ |_____|        \___/   |_|  
```

# ⚡ Mivi-v4: Agent-Native SLM Engine & Server in Rust

[![Rust 2021](https://img.shields.io/badge/rust-2021%20edition-orange.svg?style=flat-square&logo=rust)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0%20%7C%20MIT-blue.svg?style=flat-square)](LICENSE)
[![Architecture](https://img.shields.io/badge/arch-Hybrid%20SSM%20%2B%20GQA%20Attention-blueviolet.svg?style=flat-square)](#-hybrid-ssm--attention-architecture)

**Mivi-v4** is a CPU-first, low-memory, agent-native Small Language Model (SLM) inference engine and server implemented primarily in Rust. Unix filesystem hardening uses the platform libc interface; no C++ runtime is required.

It runs **Hybrid SSM + GQA Attention** architectures (such as Liquid AI's **LFM2.5** family) directly on CPU with native SIMD acceleration (AVX2/FMA/NEON), preallocated inference buffers, built-in sandboxed tool orchestration, and a drop-in OpenAI-compatible streaming HTTP server.

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
│                     Hybrid SSM + GQA Inference Core                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Selective KV Cache for Attention    │  Pure Rust SIMD (AVX2/NEON) Quant Kernels│
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 🥊 Technical Comparison

| Dimension | ⚡ **Mivi-v4** | 🦙 **llama.cpp** | 🦙 **Ollama** |
|---|---|---|---|
| **Implementation Language** | **Rust** (with small Unix libc FFI for filesystem hardening) | C / C++ | Go (wrapper daemon around llama.cpp) |
| **Architecture Focus** | **Hybrid SLMs** (Gated ShortConv SSM + GQA Attention) | Pure Transformers (Llama, Mistral, Gemma) | Same as llama.cpp |
| **Agent & Tools Support** | **Native Built-in** (ReAct agent loop, Pratt parser calc, sandboxed FS) | ❌ None (Text completion only) | ❌ Needs external framework (LangChain, AutoGen) |
| **KV Cache Footprint** | **Selective Allocation** for attention layers | Implementation-dependent | Implementation-dependent |
| **RAM Footprint** | Model, context, KV precision, and OS paging dependent | Model/configuration dependent | Model/configuration dependent |
| **Memory Safety** | Rust ownership with isolated low-level SIMD/mmap code | Manual C/C++ pointer management | Go runtime plus native backend |
| **Single Binary** | **Yes** (Single standalone executable `mivi`) | Multiple CLI binaries & shared libraries | Daemon binary + bundled llama.cpp dynamic libraries |
| **HTTP Server & SSE** | **Built-in Axum server** with dynamic port hunting & watchdog | `llama-server` | Built-in Go API daemon |

---

## ✨ Key Architectural Features

### 1. ⚡ Hybrid SSM + GQA Attention Architecture
Standard LLMs use pure self-attention with quadratic $O(N^2)$ memory and compute costs. Mivi is optimized for hybrid architectures:
- **Gated ShortConv SSM Layers**: 1D causal depthwise convolution with linear $O(N)$ time complexity and bounded recurrent state.
- **Grouped-Query Attention (GQA) Layers**: High-precision associative recall with FlashDecoding online softmax.
- **Selective KV Cache**: KV cache is dynamically mapped *only* to attention layers; non-attention SSM layers do not allocate KV entries.

### 2. 🛡️ Native Sandboxed Agent & Tool Engine
Mivi eliminates the need for heavyweight Python agent runtimes:
- **Autonomous ReAct Agent Loop**: State machine with observation, reasoning, action, and stagnation guards.
- **Pratt Parser Calculator**: Full recursive descent math engine evaluating mathematical expressions safely with zero `eval()` vulnerabilities.
- **Sandboxed Filesystem (`read_file`, `write_file`, `list_dir`)**: Enforces relative-path checks and Unix descriptor-relative no-follow access to prevent traversal and symlink redirection.

### 3. 🌐 OpenAI-Compatible Local Server
- **Drop-in Replacement**: Supports `/v1/models`, `/v1/chat/completions` (JSON & SSE streaming), and `/v1/mivi/agent`.
- **Dynamic Port Hunting**: Automatically hunts for available adjacent ports if the requested port is in use.
- **Resource Safety Watchdog**: Monitors process RSS memory and can perform a graceful shutdown when configured limits are exceeded.
- **Hono-Style Minimalist Logging**: Clean terminal logs reporting method, path, status, latency, and tokens/sec.

### 4. 💬 Modern Interactive Terminal Chat REPL
- **Live Stream Rendering**: Streaming token output with live ANSI styling.
- **Thinking Mode**: Real-time `<think>` trace formatting and duration tracking.
- **Rich Telemetry**: Displays token count, duration, generation speed (tok/s), and real-time process RAM RSS.
- **Slash Commands**: `/help`, `/clear`, `/history`, `/temp`, `/top_p`, `/rep`, `/thinking`, `/exit`.

### 5. ⚡ LMCache-Inspired Prefix Caching & Disk Persistence
- **Prefix reuse on cache hits**: Input prompts are chunked into 64-token blocks and hashed with 64-bit rolling FNV-1a. Shared system prompts, tool schemas, and multi-turn prefixes can skip repeated forward passes when the cached state is compatible.
- **Hybrid State Snapshotting**: Automatically snapshots both the 6 Attention KV layers and 10 Gated ShortConv SSM convolution states (`conv_states`).
- **On-Disk Persistence (`.mivi/cache/*.kvc`)**: Saves prefilled prompt states to disk for possible reuse across compatible process restarts.

---

## 📦 Workspace Architecture (12 Modular Crates)

The codebase is organized into 12 cleanly isolated workspace crates:

| Crate | Directory | Description |
|---|---|---|
| [`mivi-core`](crates/mivi-core) | `crates/mivi-core` | Preallocated `RunState` arena, AVX2/NEON SIMD dispatch, RMSNorm, Softmax, RoPE cache, and brand constants. |
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

Run `just bench` to measure the local quantized matvec kernels. Supplying a model path also
enables the runner's model-generation and prefix-cache measurements. Throughput and latency
depend on the CPU, SIMD features, compiler profile, model, context length, and KV precision;
the project does not treat a single machine's numbers as a universal performance guarantee.

### 💾 Memory Footprint

Runtime RSS depends on the loaded GGUF, context size, KV precision, adapters, and OS page
residency. Use the server's runtime telemetry and `--max-memory`/`--warn-memory` limits when
sizing a deployment.

---

## ⚡ Quick Start

### 1. Prerequisites

- **Rust**: current stable toolchain (2021 edition)
- **Just**: (Optional task runner) `cargo install just`

### 2. Build

```bash
# Clone the repository
git clone https://github.com/aswin402/mivi-v4.git
cd mivi-v4

# Build release binary (uses low-memory one concurrent job)
just build-release
# Or: cargo build --release --jobs 1
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
# Or: cargo run --release -- chat --model models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf
```

```text
  ⚡ Mivi Chat v<version> (<loaded model> • <configured context> ctx)
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
  ⏱ <duration> • <tokens> tokens • <tok/s> • RAM <RSS> MB
```

The chat output above is illustrative; model output and measurements vary by local hardware and configuration.

### 5. Launch the OpenAI-Compatible HTTP Server (`serve`)

```bash
just serve
# Or: cargo run --release -- serve --model models/mivi-v4-q4_k_m.gguf --port 8080 --workspace .
# Public binds (for example --host 0.0.0.0) require MIVI_API_KEY; with a key configured,
# all API/model/control routes require authentication. Health and the embedded UI remain public.
# Optional browser access: repeat --cors-origin for each exact allowed origin.
# Example: --cors-origin http://localhost:3000
# Tool handlers are bounded to 4 concurrent blocking executions by default;
# tune with --max-concurrent-tool-executions when serving multiple agent requests.
# Streaming and blocking inference requests default to a 300-second deadline;
# tune with --request-timeout-secs for slower models or long agent runs.
# First model output defaults to a 120-second deadline;
# tune with --first-token-timeout-secs for long CPU prefill workloads.
# For a model with a non-standard prompt/tool protocol, supply its declarative profile:
# cargo run --release -- serve --model models/model.gguf --model-profile docs/lfm2.5-profile.json
# For a model that supports chat but has no tool-call protocol, use a text-only profile:
# cargo run --release -- serve --model models/model.gguf --model-profile docs/text-only-profile.json
# Text-only profiles reject tool-enabled OpenAI/Anthropic requests and the internal tool agent.
```

Model profiles are resolved from the explicit `--model-profile` file first, then from the loaded
model's embedded chat-template metadata. A profile owns prompt delimiters, tool-call decoding, and
capability reporting; HTTP routes do not identify models by name. To add a model family, provide a
validated profile (or an isolated codec/profile implementation) and add it to the profile
conformance tests. Use `kind: "text_only"` when the model has no tool-call protocol.

```text
  ╭──────────────────────────────────────────────────────────╮
  │                                                          │
  │   ⚡ Mivi Agent Engine                                   │
  │   Lightweight, Fast & Sandboxed Local Agent Server       │
  │                                                          │
  │   • Model:      <loaded model or mivi alias>              │
  │   • Context:    16K default, 64K maximum                  │
  │   • Local API:  http://127.0.0.1:8080/v1                 │
  │                                                          │
  │   OpenAI-compatible endpoints:                           │
  │   POST /v1/chat/completions (SSE streaming)              │
  │   POST /v1/mivi/agent       (Autonomous loop)            │
  │   GET  /metrics             (cumulative counters)         │
  │                                                          │
  ╰──────────────────────────────────────────────────────────╯
```

Inference endpoints require a loaded GGUF model; otherwise they return `503 Service Unavailable`.
The default server does not enable cross-origin browser requests. Configure an explicit deployment
proxy/allowlist if a browser client is required.

OpenAI-compatible requests support validated sampling parameters (`temperature`, `top_p`, `top_k`, `min_p`,
`repetition_penalty`, presence/frequency penalties, and `seed`), custom stop sequences, `none`/`auto`/
`required`/named tool choice, and non-streaming `response_format: {"type":"json_object"}`. JSON Schema
responses and JSON streaming are rejected explicitly. `tool_choice: "required"` is validated after generation;
models without constrained tool decoding may return an explicit inference error when they do not emit a tool call.
Anthropic `/v1/messages` supports validated sampling, `stop_sequences`, and structured streaming tool-use blocks.

Agent context documents must be relative to the configured `--workspace` and are size-bounded before being
added to the prompt. On Unix, reads use descriptor-relative no-follow traversal to prevent symlink swaps.
Timed-out built-in tool handlers receive a cooperative cancellation signal; custom legacy handlers may
continue until they return, subject to the broker's concurrency limit.

`GET /metrics` returns process-local JSON counters for accepted/rejected inference requests, generation
latency, token totals, inference errors, and timed-out tools. Counters reset when the process restarts.

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
    "max_steps": 5,
    "tool_choice": "required",
    "tool_call_retries": 1,
    "allowed_tools": ["calculator", "write_file"]
  }'
```

The internal agent accepts `tool_choice: "auto"` (the default) or `"required"`. Required mode fails closed if
the model never emits a tool call and may retry that missing-call response up to the configured
`tool_call_retries` value (bounded by the server); retries do not consume agent action steps. Text-only model
profiles reject this endpoint because they do not define a tool-call protocol.
The endpoint also accepts the validated sampling controls `temperature`, `top_p`, `top_k`, `min_p`,
`repetition_penalty`, `presence_penalty`, `frequency_penalty`, and `seed`; these options are applied to each
generation step without embedding model-specific values in the agent route.

---

## 🧪 Two-Engine Verification Strategy

Mivi employs a strict **Two-Engine Verification Strategy**:
1. **PyTorch Oracle Engine** ([`reference/reference_engine.py`](reference/reference_engine.py)): Ground-truth reference implementation.
2. **Rust Production Engine** ([`crates/mivi-model`](crates/mivi-model)): High-performance native SIMD implementation.

Every layer forward pass (RMSNorm, RoPE, Attention, ShortConv, SwiGLU) is cross-checked against PyTorch golden outputs.

Run the test suite:

```bash
just test
# Or: cargo test --workspace --jobs 1
```

While iterating, prefer a focused package/test command such as
`cargo test -p mivi-server --lib --jobs 1 -- --test-threads=2`. Test counts change as coverage
evolves, so treat the command result—not a fixed example count—as the source of truth.

---

## 🛠️ Justfile Command Reference

| Command | Description |
|---|---|
| `just build` | Compile workspace in debug mode (one job) |
| `just build-release` | Compile optimized release binary |
| `just test` | Run the workspace test suite with one Cargo job and two test threads |
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
