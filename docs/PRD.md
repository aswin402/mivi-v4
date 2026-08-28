# mivi-v4 — Product Requirements Document (PRD)

**Version:** 1.0  
**Date:** 2026-08-28  
**Status:** Draft  
**Owner:** Aswin  

---

## 1. Executive Summary

mivi-v4 is a purpose-built agentic Small Language Model (350M parameters) with a Mixture of LoRA Experts, served by a high-performance Rust inference engine. It delivers AI agent capabilities (tool calling, reasoning, routing, code generation) on consumer hardware with <1GB RAM, no GPU required.

The deliverable is a **single Rust binary** that loads a model file and exposes an OpenAI-compatible HTTP API.

---

## 2. Target Users

### Primary: AI Agent Developers
- Building agent frameworks that need a local, fast, reliable LLM backend
- Want to avoid cloud API costs and latency
- Need structured tool calling, not just chat
- Deploy on edge devices, CI runners, or developer laptops

### Secondary: Embedded/IoT ML Engineers
- Running inference on resource-constrained devices (Raspberry Pi, NUC, ARM SBCs)
- Need a model + runtime that fits in <1GB
- Require deterministic, predictable memory usage

### Tertiary: Open-Source AI Researchers
- Studying efficient MoE architectures at small scale
- Experimenting with agentic fine-tuning of sub-1B models
- Need a hackable, readable inference engine

---

## 3. User Stories

### US-1: Basic Inference
> As an agent developer, I want to run `mivi serve --model ./mivi-v4.gguf` and get an HTTP API endpoint so that my agent can send chat requests and receive responses.

**Acceptance Criteria:**
- Single command starts the server
- Server binds to configurable host:port
- `/v1/chat/completions` endpoint accepts OpenAI-format requests
- Responses stream via SSE
- Cold start < 3 seconds

### US-2: Tool Calling
> As an agent developer, I want to define tools in the system prompt and receive structured `tool_call` responses so that my agent can execute functions reliably.

**Acceptance Criteria:**
- Tool definitions in OpenAI function calling format
- Model outputs `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` 
- JSON output is always valid (grammar-constrained decoding)
- Tool results can be sent back in `tool` role messages
- Multi-step tool chains work (call → result → call → result → answer)

### US-3: Thinking / Reasoning
> As an agent developer, I want the model to reason step-by-step before acting so that I can see its thought process and get better answers.

**Acceptance Criteria:**
- Model produces `<think>...</think>` blocks before responses/tool calls
- Think blocks are streamed but can be optionally hidden from end users
- Thinking improves tool selection accuracy and reduces errors
- Thinking can be disabled via `thinking: false` parameter for speed

### US-4: Expert Routing
> As an agent developer, I want the model to automatically route to specialized experts for different task types so that code tasks get code-expert quality and reasoning tasks get reasoning-expert quality.

**Acceptance Criteria:**
- Model loads multiple LoRA expert adapters at startup
- Per-token, per-layer gating routes to Top-2 experts
- Code/tool tasks measurably improve vs single-model baseline
- Routing is transparent (no user configuration needed)

### US-5: Low-Resource Deployment
> As a developer with an older laptop (4GB RAM, no GPU), I want mivi-v4 to run without issues so that I can develop AI agents locally.

**Acceptance Criteria:**
- Peak RAM < 400MB (typical), < 700MB (worst case with extended context)
- CPU-only inference (x86_64 and aarch64)
- Decode speed > 10 tok/s on a 4-core CPU
- No GPU driver dependencies
- Works on Linux, macOS, Windows (WSL2)

### US-6: Agent Memory & Context
> As an agent developer, I want mivi-v4 to understand agent concepts (memory, skills, context windows, tool schemas) so that it can be an effective agent brain.

**Acceptance Criteria:**
- Model understands system prompts defining agent personas and capabilities
- Model can parse and reference tool schemas accurately
- Model handles multi-turn conversations with tool result history
- Model can route between "I know this" vs "I need to use a tool for this"

---

## 4. Functional Requirements

### FR-1: Rust Inference Engine

| ID | Requirement | Priority |
|---|---|---|
| FR-1.1 | Load GGUF model files (Q4_0, Q4_K_M, Q5_K_M, Q8_0 quantization) | P0 |
| FR-1.2 | Memory-mapped weight loading via `mmap` | P0 |
| FR-1.3 | Pre-allocated RunState arena (zero heap allocation during decode) | P0 |
| FR-1.4 | SIMD-accelerated quantized matrix-vector multiplication (AVX2, NEON) | P0 |
| FR-1.5 | Transformer forward pass: RMSNorm, GQA, SwiGLU FFN, RoPE | P0 |
| FR-1.6 | SSM/Convolution blocks for hybrid LFM architecture | P0 |
| FR-1.7 | BPE tokenizer with SIMD acceleration | P0 |
| FR-1.8 | Sampling: temperature, top-p, top-k, repetition penalty, min-p | P0 |
| FR-1.9 | KV cache: pre-allocated, with sliding window support | P0 |
| FR-1.10 | Stop token detection and generation halt | P0 |
| FR-1.11 | Batch prefill for prompt processing | P1 |
| FR-1.12 | Speculative decoding with draft model | P2 |

### FR-2: HTTP API Server

| ID | Requirement | Priority |
|---|---|---|
| FR-2.1 | `POST /v1/chat/completions` — chat completion with streaming | P0 |
| FR-2.2 | `GET /v1/models` — list loaded models | P1 |
| FR-2.3 | `POST /v1/completions` — raw text completion | P2 |
| FR-2.4 | `GET /health` — health check endpoint | P1 |
| FR-2.5 | `GET /metrics` — inference metrics (tokens/sec, memory, requests) | P2 |
| FR-2.6 | SSE streaming with OpenAI-compatible chunk format | P0 |
| FR-2.7 | Request queuing with configurable concurrency | P1 |
| FR-2.8 | CORS support for browser-based agents | P1 |
| FR-2.9 | API key authentication (optional) | P2 |

### FR-3: Tool Calling System

| ID | Requirement | Priority |
|---|---|---|
| FR-3.1 | Parse tool definitions from `tools` field in request | P0 |
| FR-3.2 | Inject tool schemas into system prompt using ChatML format | P0 |
| FR-3.3 | Detect `<tool_call>` tags in generation stream | P0 |
| FR-3.4 | Grammar-constrained decoding for JSON within tool_call tags | P0 |
| FR-3.5 | Validate tool call JSON against provided schemas | P1 |
| FR-3.6 | Support `tool_choice: "auto" | "none" | {"name": "..."}` | P1 |
| FR-3.7 | Support parallel tool calls (multiple tool_call blocks) | P1 |
| FR-3.8 | Incremental JSON streaming (buffer until complete) | P0 |

### FR-4: Thinking / Reasoning

| ID | Requirement | Priority |
|---|---|---|
| FR-4.1 | Detect `<think>` blocks in generation stream | P0 |
| FR-4.2 | Stream think blocks with separate delta field or role | P0 |
| FR-4.3 | Support `thinking: true/false` parameter to enable/disable | P1 |
| FR-4.4 | Think token budget limit (max thinking tokens) | P1 |
| FR-4.5 | Measure thinking efficiency (accuracy vs think tokens used) | P2 |

### FR-5: Mixture of LoRA Experts (MoLE)

| ID | Requirement | Priority |
|---|---|---|
| FR-5.1 | Load multiple LoRA adapter files at startup | P0 |
| FR-5.2 | Per-layer Top-K gating router computation | P0 |
| FR-5.3 | Fused LoRA forward: `h = Wx + Σ g_i(α/r)(B_i A_i x)` | P0 |
| FR-5.4 | Load-balanced routing (auxiliary loss during training) | P0 |
| FR-5.5 | Expert utilization metrics (which experts activate for what) | P1 |
| FR-5.6 | Hot-swap individual LoRA adapters without restart | P2 |

### FR-6: CLI Interface

| ID | Requirement | Priority |
|---|---|---|
| FR-6.1 | `mivi serve --model <path> [--port N] [--host H]` — start API server | P0 |
| FR-6.2 | `mivi chat --model <path>` — interactive CLI chat | P1 |
| FR-6.3 | `mivi info --model <path>` — display model metadata | P1 |
| FR-6.4 | `mivi bench --model <path>` — run inference benchmarks | P2 |
| FR-6.5 | `mivi quantize --model <path> --type Q4_K_M` — quantize models | P2 |

---

## 5. Non-Functional Requirements

### NFR-1: Performance

| Metric | Requirement | Stretch Goal |
|---|---|---|
| Decode speed (1 thread, x86_64) | ≥ 15 tok/s | ≥ 25 tok/s |
| Decode speed (1 thread, aarch64) | ≥ 10 tok/s | ≥ 18 tok/s |
| Prefill speed (prompt processing) | ≥ 100 tok/s | ≥ 200 tok/s |
| Cold start (model load to first token) | ≤ 3 seconds | ≤ 1 second |
| Time to first token (after prompt) | ≤ 500ms | ≤ 200ms |
| API latency overhead (HTTP + JSON) | ≤ 5ms | ≤ 2ms |

### NFR-2: Memory

| Metric | Requirement | Stretch Goal |
|---|---|---|
| Peak RAM (base model, 2K context) | ≤ 400MB | ≤ 300MB |
| Peak RAM (MoE + router, 2K context) | ≤ 500MB | ≤ 400MB |
| Peak RAM (MoE + 8K context) | ≤ 700MB | ≤ 550MB |
| Peak RAM (absolute maximum) | ≤ 1000MB | ≤ 800MB |
| Heap allocations during decode loop | 0 | 0 |
| Binary size (static, stripped) | ≤ 30MB | ≤ 15MB |

### NFR-3: Reliability

| Metric | Requirement |
|---|---|
| Tool call JSON validity | ≥ 95% (with grammar constraint) |
| Tool call semantic accuracy | ≥ 85% (correct tool + params) |
| Crash-free operation | No panics during normal operation |
| Memory safety | Zero `unsafe` in hot path (use safe SIMD abstractions) |
| Graceful degradation | OOM → reduce context, not crash |

### NFR-4: Compatibility

| Platform | Support Level |
|---|---|
| Linux x86_64 | Tier 1 (primary) |
| Linux aarch64 | Tier 1 |
| macOS x86_64 | Tier 2 |
| macOS aarch64 (Apple Silicon) | Tier 1 |
| Windows x86_64 (WSL2) | Tier 2 |
| WASM (browser) | Tier 3 (future) |

---

## 6. Feature Priority Matrix

```
                        IMPACT
              Low            Medium           High
         ┌──────────┬───────────────┬────────────────┐
    Low  │ WASM     │ Batch API     │ Speculative    │
  EFFORT │ support  │ endpoint      │ decoding       │
         ├──────────┼───────────────┼────────────────┤
  Medium │ API key  │ Hot-swap LoRA │ Grammar-       │
         │ auth     │ adapters      │ constrained    │
         │          │               │ decoding       │
         ├──────────┼───────────────┼────────────────┤
    High │ Windows  │ Semantic      │ Rust engine    │
         │ native   │ router        │ + MoLE         │
         │          │ (MiniLM)      │ inference      │
         └──────────┴───────────────┴────────────────┘
```

---

## 7. API Specification

### Request Format
```json
{
  "model": "mivi-v4",
  "messages": [
    {
      "role": "system",
      "content": "You are MIVI, an agentic AI assistant."
    },
    {
      "role": "user",
      "content": "Search for the latest Rust release"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "web_search",
        "description": "Search the web for current information",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {"type": "string", "description": "Search query"}
          },
          "required": ["query"]
        }
      }
    }
  ],
  "tool_choice": "auto",
  "temperature": 0.7,
  "max_tokens": 2048,
  "stream": true,
  "thinking": true
}
```

### Streaming Response Format (SSE)
```
data: {"id":"mivi-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","thinking":"I need to search..."},"finish_reason":null}]}

data: {"id":"mivi-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"web_search","arguments":"{\"query\": \"latest Rust version\"}"}}]},"finish_reason":"tool_calls"}]}

data: [DONE]
```

### Tool Result Message
```json
{
  "role": "tool",
  "tool_call_id": "call_1",
  "content": "{\"results\": [{\"title\": \"Rust 1.84\", \"url\": \"...\"}]}"
}
```

---

## 8. Model Specifications

| Property | Value |
|---|---|
| **Model name** | `mivi-v4` |
| **Base architecture** | LiquidAI LFM2.5-350M (Hybrid SSM+GQA) |
| **Total parameters** | ~350M (base) + ~32M (4 LoRA experts) + ~1M (router) |
| **Quantization** | GGUF Q4_K_M (primary), Q8_0 (quality), Q2_K (extreme compression) |
| **Native context** | 32,768 tokens |
| **Vocabulary** | 65,536 tokens (byte-level BPE) |
| **Special tokens** | `<think>`, `</think>`, `<tool_call>`, `</tool_call>`, `<|im_start|>`, `<|im_end|>` |
| **Chat format** | ChatML with extensions |
| **Training data** | Agentic trajectories, tool calling, CoT reasoning, code, conversation |

---

## 9. Evaluation & Success Metrics

### Functional Benchmarks

| Benchmark | Target | What It Measures |
|---|---|---|
| BFCL (Berkeley Function Calling) | ≥ 75% | Tool calling accuracy |
| GSM8K (Grade School Math) | ≥ 30% | Mathematical reasoning |
| HumanEval (Code Generation) | ≥ 25% | Code generation quality |
| IFEval (Instruction Following) | ≥ 50% | Instruction adherence |
| Custom Agent Benchmark | ≥ 70% | Multi-step agentic tasks |

### System Benchmarks

| Benchmark | Target |
|---|---|
| Tokens/sec (decode, x86_64 4-core) | ≥ 15 |
| Tokens/sec (decode, Apple M1) | ≥ 20 |
| Peak RSS memory (ps aux) | ≤ 400MB |
| TTFT (Time to First Token) | ≤ 500ms |
| Valid JSON rate (tool calls) | ≥ 95% |

---

## 10. Release Milestones

| Milestone | Target Date | Deliverable |
|---|---|---|
| **M1: Engine Alpha** | Week 3 | GGUF loading + text generation CLI |
| **M2: API Server** | Week 4 | HTTP API with streaming |
| **M3: Tool Calling** | Week 5 | Grammar-constrained tool calling |
| **M4: Base Model** | Week 7 | Fine-tuned mivi-v4-base.gguf |
| **M5: MoE Integration** | Week 10 | 4-expert MoLE in Rust engine |
| **M6: Production** | Week 12 | Benchmarked, documented, CI/CD |
| **M7: v4.1** | Week 14 | Semantic router, speculative decoding |

---

## 11. Out of Scope (v4)

- Training infrastructure (uses existing PyTorch/HF ecosystem)
- GPU inference (CPU-only for v4)
- Vision/multimodal (text-only for v4)
- Model marketplace or registry
- Fine-tuning API
- Batched inference (single request at a time for v4)
