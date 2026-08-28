# mivi-v4 — Core Idea

> **One binary. One command. An AI agent brain that runs anywhere.**

---

## The Thesis

Every AI agent framework today faces the same bottleneck: **the model**.

- Cloud APIs are expensive, slow, and a single point of failure
- Open-source models are bloated — 7B+ parameters, 16GB+ RAM, GPU required
- Small models exist but are trained as "mini chatbots" — they can talk, but they can't *think*, *plan*, or *use tools*

**mivi-v4 is a different animal.** It's not a chatbot shrunk down. It's a purpose-built *agent brain* — a model that was born to:
- Parse tool schemas and call functions correctly
- Think step-by-step before acting
- Route complex tasks to specialized experts
- Use the internet for knowledge instead of memorizing Wikipedia

All in **<400MB of RAM**, on any laptop, without a GPU.

---

## The Paradigm Shift

### ❌ The Old Way: Model as Oracle
```
User → [Giant Model (70B)] → Answer
         ↑
    "knows everything"
    "does everything"
    16GB+ RAM, GPU required
```

The old paradigm stuffs all knowledge into model weights. This is why models are huge — they're memorizing the internet.

### ✅ The mivi-v4 Way: Model as Tool-Using Engineer
```
User → [mivi-v4 (350M)] → Think → Route → Use Tools → Answer
              ↑                        ↑
    "knows HOW to do things"    "uses tools for WHAT things"
    350MB RAM, CPU only          Internet, files, code, APIs
```

mivi-v4 doesn't try to know everything. It knows how to:
1. **Understand** what the user wants
2. **Think** about how to approach it
3. **Route** to the right expert (code, reasoning, tools, chat)
4. **Call tools** to get information and take actions
5. **Synthesize** results into a clear answer

This is how real engineers work. You don't memorize API docs — you look them up. You don't do math in your head — you use a calculator. You don't write code blindly — you test it.

---

## The Four Pillars

### 🧠 Pillar 1: Agent-Native
mivi-v4 is not a general-purpose chatbot fine-tuned for tool use. It's built from the ground up for agentic workflows:

- **Native `<think>` blocks** — The model reasons internally before producing output
- **Native `<tool_call>` format** — Tool invocation is a first-class citizen, not an afterthought
- **Multi-turn state tracking** — Understands conversation history, tool results, and error recovery
- **Self-correction** — When a tool call fails, it reasons about why and retries differently

```
[Input] → <think>reasoning</think> → <tool_call>action</tool_call> → [result] → response
```

### ⚡ Pillar 2: Resource-Frugal
Everything is designed for the constraint: **<1GB RAM, CPU-only, no GPU**.

| What | How |
|---|---|
| Model weights | Q4_K_M quantization → 195MB |
| Architecture | Hybrid SSM+GQA → sub-quadratic KV cache |
| Memory model | Pre-allocated arena → zero heap allocations during inference |
| Weight loading | `mmap` → OS manages paging, only active pages in RAM |
| Tokenizer | SIMD-accelerated → sub-microsecond |
| Runtime | Pure Rust → no Python, no framework overhead |

### 🎯 Pillar 3: Expert-Routed
One model can't be the best at everything. But one model with four specialized experts can be great at four things:

```
                    Input
                      │
              ┌───────▼───────┐
              │  Gating Router │
              │  (learned, per │
              │   layer)       │
              └───┬───┬───┬───┘
                  │   │   │
         ┌────────┘   │   └────────┐
         ▼            ▼            ▼
    ┌─────────┐ ┌─────────┐ ┌─────────┐
    │Reasoning│ │Code+Tool│ │  Chat   │
    │  Expert │ │  Expert │ │  Expert │
    │(LoRA 8MB)│(LoRA 8MB)│(LoRA 8MB)│
    └─────────┘ └─────────┘ └─────────┘
         │            │            │
         └────────────┼────────────┘
                      ▼
              Weighted Sum Output
```

Each expert is a lightweight LoRA adapter (~8MB) trained on domain-specific data. The router learns to activate the right experts per-token, per-layer. Total overhead: ~32MB for four experts.

### 🔧 Pillar 4: Tool-Grounded
mivi-v4 treats tools as its primary knowledge source:

| Need | Tool | Not |
|---|---|---|
| Current information | `web_search` | Memorized training data |
| Code execution | `execute_code` | Hallucinated output |
| File contents | `read_file` | Guessing file structure |
| Calculations | `calculator` | Mental arithmetic |
| Database queries | `sql_query` | Memorized schemas |

This means:
- **Smaller model** — doesn't need to memorize facts
- **More accurate** — verified through tool execution
- **Always current** — internet access for real-time data
- **Trustworthy** — can show its work (tool results are evidence)

---

## What Makes mivi-v4 Different

| Dimension | Other Small Models | mivi-v4 |
|---|---|---|
| **Purpose** | General chatbot | Agent brain |
| **Architecture** | Pure transformer | Hybrid SSM+GQA + MoLE |
| **Knowledge** | Memorized in weights | Retrieved via tools |
| **Tool calling** | Bolted on | Native, grammar-enforced |
| **Reasoning** | Implicit | Explicit `<think>` blocks |
| **Experts** | Single model | 4 specialized LoRA experts |
| **Runtime** | Python (PyTorch) | Pure Rust, zero-heap |
| **Memory** | 2-16GB | <400MB |
| **Deployment** | Docker + Python + CUDA | Single binary, one command |
| **Target** | Cloud GPUs | Any laptop CPU |

---

## The Name

**mivi** = **M**iniature **I**ntelligent **V**ersatile **I**nference

**v4** = Fourth iteration of the architecture

The model name `mivi-v4` represents the convergence of:
- Miniature (350M params, <400MB RAM)
- Intelligent (thinks before acting, routes to experts)
- Versatile (coding, reasoning, tools, chat)
- Inference (optimized for fast, efficient generation)

---

## Success Criteria

mivi-v4 succeeds when:

1. **An AI agent can use mivi-v4 as its brain** — passing tool definitions, receiving structured tool calls, getting reasoned responses
2. **It runs on a $300 laptop** — no GPU, 4GB RAM, i3 processor
3. **Tool calls are reliable** — >90% valid JSON, correct parameter types, appropriate tool selection
4. **Thinking is visible** — `<think>` blocks show the reasoning process
5. **It's one command** — `mivi serve` and you have an API endpoint
6. **It's fast enough** — >15 tokens/sec decode, <2s cold start
7. **The experts add value** — MoE routing measurably improves code, reasoning, and tool tasks vs single model
