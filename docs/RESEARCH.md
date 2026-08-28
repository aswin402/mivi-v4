# Mivi-v4 — Deep Research Findings

> Consolidated research from 25+ reference projects, feasibility analysis, and architecture refinements.

---

## Executive Summary

The research validates the core mivi-v4 concept but recommends one critical change:

> **Do not try to make one 350M model simultaneously become a great general chatbot, coder, reasoner, planner, researcher, and orchestrator through ordinary fine-tuning. Instead, build a small agent-native foundation model + routed specialists + a Rust agent runtime.**

The product definition becomes:

> **Mivi-v4 is a CPU-first, low-memory, agent-native AI system built around an LFM2.5-350M-class model, specialized through post-training and lightweight experts, and extended through tools, memory, retrieval, recursive context, and a highly optimized Rust runtime.**

---

## 1. Base Model Validation: LFM2.5-350M

### Architecture

- 350M parameters, 16 layers
- 10 double-gated convolution blocks + 6 GQA blocks (hybrid SSM+GQA)
- 65,536-token vocabulary
- 32K native context window
- Trained on 28T tokens — enormous for this size class

### Published Performance (Liquid AI)

| Hardware | Decode Speed | Notes |
|---|---|---|
| AMD Ryzen AI Max 395+ CPU | 313 tok/s | llama.cpp Q4 |
| Snapdragon Gen4 | 188 tok/s | — |
| iPhone 13 Mini | 88 tok/s | Cactus INT8 |
| Pixel 6a | 42 tok/s | — |
| Raspberry Pi 5 | 30 tok/s | — |

**Peak memory: ~434 MB** in edge benchmarks — sub-1GB is not a fantasy.

### Critical Limitation

Liquid explicitly states the stock LFM2.5-350M is:
- ✅ Recommended for: tool use, data extraction, structured output
- ❌ Not recommended for: programming, knowledge-intensive tasks, math, creative writing

**This is actually useful** — it means we shouldn't ask the base model to be everything. We should specialize it toward being an agent brain.

### Published Benchmarks (Baseline)

| Benchmark | Score |
|---|---|
| GPQA Diamond | 30.64 |
| MMLU-Pro | 20.01 |
| IFEval | 76.96 |
| IFBench | 40.69 |
| Multi-IF | 44.92 |
| BFCLv3 | 44.11 |
| BFCLv4 | 21.86 |

---

## 2. The Paradigm: Agent Operating System, Not Just a Model

The real product is:

```
an on-device Agent OS / Agent Engine with a small specialized LM inside it.
```

### What the MODEL should learn

- English, conversation, instruction following
- Agent state, tool schemas, tool selection, tool arguments
- Planning, decomposition, verification
- Concise reasoning, coding fundamentals
- Debugging behavior, testing behavior, routing
- When NOT to use a tool, when to ask for clarification
- How to interpret tool results, how to continue after tool failure

### What the RUNTIME should handle

- Actual tool execution (web, filesystem, shell, git, code execution)
- Memory, vector search, context management
- Retries, timeouts, authentication, concurrency, sandboxing
- Model routing, recursion, long-context storage
- API serving

**This separation is extremely important.**

---

## 3. Architecture: Adapter-MoE, Not Full MoE

### Problem with naive MoE

```
8 × 350M experts = 2.8B total parameters
```

Even quantized, this moves away from the tight memory objective.

### Solution: Shared Backbone + LoRA Adapters

```
             Shared LFM2.5-350M
                    │
        ┌───────────┼───────────┐
        │           │           │
      LoRA        LoRA        LoRA
      Code        Agent       Chat
        │           │           │
      LoRA        LoRA        LoRA
     Debug       Frontend    Backend
        │           │           │
      LoRA        LoRA        LoRA
     Testing    Architect   Research
```

The 350M backbone is shared. Each expert is a very small adapter.

### Adapter Composition

The runtime can combine adapters:

```
agent + backend
agent + coding
agent + frontend
agent + debug + testing
```

This creates sparse expert behavior without duplicating the entire model.

---

## 4. Two-Level Routing

### Level 1 — Task Family

```
chat → CHAT
agent task → AGENT
code work → CODE
research → RESEARCH
reasoning → GENERAL
```

### Level 2 — Specialization

```
CODE
 ├── implementation
 ├── debugging
 ├── testing
 ├── frontend
 ├── backend
 └── architecture
```

This reduces routing complexity and makes training much easier.

---

## 5. Expert Set (Revised)

### Initial 6 experts (not 4)

```
1. GENERAL
2. AGENT
3. CODE
4. DEBUG/TEST
5. RESEARCH
6. CHAT
```

### Second wave

```
FRONTEND
BACKEND
ARCHITECTURE
TESTING
PLANNING
TOOL_ROUTING
```

**Six strong experts are better than twenty weak ones.**

---

## 6. Compact Structured Reasoning (Not Giant CoT)

The model is only 350M. Don't train it to emit massive reasoning traces.

### Instead of

```
500-token chain of reasoning
```

### Train

```
<plan>
1. inspect request
2. identify required tool
3. gather missing information
4. execute
5. verify result
6. answer
</plan>
```

Or even more compact:

```
INTENT → PLAN → ACTION → OBSERVATION → VERIFY → ANSWER
```

> **Think enough to act correctly, not think as many tokens as possible.**

ThinkingCap research confirms: carefully fine-tuned models can dramatically reduce reasoning-token consumption while maintaining quality.

---

## 7. 64K Context Strategy: Runtime, Not Architecture

### Phase A: 32K native + 64K runtime

```
64K document
      ↓
context store
      ↓
retrieve relevant regions
      ↓
model sees 4K–12K
      ↓
inspect another region
      ↓
recursive call
      ↓
summarize
      ↓
final answer
```

**64K effective context ≠ 64K raw active transformer context**

### Phase B: Train context extension

```
32K → 48K → 64K
```

### Phase C: Benchmark at each length

Do not assume that simply changing a RoPE parameter means the model now "understands 64K".

---

## 8. RLM-Style Context VM

The most important context innovation from the research:

### Mivi Context VM Operations

```
SEARCH(query)       — semantic search over context
SLICE(source, s, e) — extract region
FILTER(condition)   — filter by metadata
RANK(criteria)      — rank by relevance
SUMMARIZE(source)   — compress context
COMPARE(a, b)       — compare two regions
MAP(op, chunks)     — apply to each chunk
REDUCE(op, results) — aggregate
RECURSE(task, ctx)  — recursive sub-call
STORE(key, value)   — persist to memory
LOAD(key)           — retrieve from memory
PIN(block)          — prevent eviction
EVICT(block)        — free context block
```

The model emits typed context operations; the Rust runtime executes them safely.

**λ-RLM research (March 2026)** shows typed functional combinators are better than open-ended generated control code — safer, more predictable, lower latency.

---

## 9. Mivi-Nano Companion (20-60M)

Inspired heavily by Needle (26M params, 6000 tok/s prefill, 1200 tok/s decode):

```
               MIVI-NANO
                  20–60M
                    │
       ┌────────────┼────────────┐
       │            │            │
     route       tools        context
       │            │            │
       └────────────┼────────────┘
                    ↓
              MIVI 350M
```

### Mivi-Nano handles

- Intent classification
- Tool filtering
- Route prediction
- Context ranking
- Skill selection
- Speculative decoding

### This enables three-stage intelligence

```
Stage 1 — Mivi Nano: cheap decision
Stage 2 — Mivi 350M: reason / act
Stage 3 — tools / external models: specialized capability
```

---

## 10. Training Pipeline (Revised: 10 Stages)

```
LFM2.5-350M-Base (start from Base, not Instruct)
       │
Stage 0: Frozen baseline benchmark
       │
Stage 1: English + instruction foundation
       │
Stage 2: Agent language (state machine training)
       │
Stage 3: Tool calling (including errors/failures)
       │
Stage 4: Planning (simple → medium → complex)
       │
Stage 5: Coding (with separate expert adapter)
       │
Stage 6: Reasoning efficiency (ThinkingCap-style)
       │
Stage 7: Expert adapters (6 initial)
       │
Stage 8: Router training
       │
Stage 9: Preference optimization (DPO)
       │
Stage 10: RL (GRPO on agent tasks)
       │
Quantization → Rust deployment
```

### Critical: Start from Base, not Instruct

We're defining a new behavior distribution (AgentLM). We want control over chat, tools, planning, reasoning, coding, memory, and routing.

---

## 11. Agent State Machine

The model must learn the canonical agent state:

```
USER → SYSTEM → TASK → PLAN → ACTION → TOOL_CALL →
TOOL_RESULT → OBSERVATION → MEMORY → ERROR →
VERIFICATION → ANSWER
```

### Tool error training (critical)

```
single tool, multiple tools, wrong tool
tool unavailable, tool error, invalid arguments
missing argument, tool result contradiction
tool result empty, tool result huge, tool timeout, tool retry
```

The last few are extremely important for real agents.

---

## 12. Verification as Core Capability

Every important action → VERIFY:

```
code generated → run tests
research answer → validate sources
tool result → check schema
calculation → recompute
file patch → inspect diff
```

**Act → Verify**, not **Generate → Hope**.

---

## 13. Failure Trajectories as First-Class Training Data

### Mivi Failure Corpus

```
tool timeout → recover
bad arguments → repair
wrong file → inspect → correct
search result poor → reformulate
wrong result → reassess
test failure → diagnose → patch → retest
context miss → retrieve more
```

Most models only train on successful trajectories. Training on failures produces much more robust agents.

---

## 14. Memory Architecture

### Three layers

```
L1 — working memory (current task)
L2 — semantic memory (embeddings + metadata)
L3 — persistent knowledge (files/OKF-style documents)
```

### Mivi Knowledge Format (.mivi/ directory)

```
.mivi/
├── memory/
├── knowledge/
├── projects/
├── skills/
├── context/
└── indexes/
```

Using OKF-inspired YAML frontmatter + Markdown — human-readable, git-friendly, portable, model-independent.

---

## 15. Tool Architecture

### Tool Broker (model never directly executes)

```
model
  ↓
structured tool request
  ↓
Rust tool broker
  ↓
permission check
  ↓
sandbox
  ↓
tool execution
```

### Dynamic Tool Discovery

Don't waste 5,000 tokens on huge tool definitions. Liquid itself warns that large tool lists consume significant context.

```
request → tool retrieval → 5–12 relevant tools → Mivi
```

### Core tool set

```
web.search, web.fetch
memory.search, memory.write
filesystem.list, filesystem.read, filesystem.write
shell.exec
git.status, git.diff, git.apply
code.run, tests.run
http.request
calculator
python.exec
```

---

## 16. RAM Budget (Revised)

```
INT4 model                  ~180–250 MB
Runtime / buffers            50–120 MB
KV cache                     50–200 MB
Tokenizer                    10–30 MB
Expert adapters               10–80 MB
Server / agent runtime       40–120 MB
-----------------------------------------
Normal total                ~340–800 MB
```

**< 1 GB single-user local agent is realistic.** Liquid already demonstrates this.

---

## 17. Optimization Metric (Revised)

### Primary metric: Agent Intelligence per MB

```
              TASK SUCCESS
───────────────────────────────────────
RAM × latency × tokens × unnecessary actions
```

### Secondary metrics

```
agent task completion
tool-call accuracy
argument accuracy
planning quality
recovery rate
verification rate
tokens/task
seconds/task
RAM peak
cold-start time
```

> Don't judge Mivi by "How close is it to a 7B model?" Judge it by "How much useful agent work can it complete per MB of RAM and per second of CPU time?"

---

## 18. What We Borrow from Each Reference

| Reference | Inspiration | Mivi Implementation |
|---|---|---|
| **LFM2.5-350M** | Edge hybrid architecture | LFM2 backbone |
| **LFM2.5-2.6B** | Agent-centric training direction | Agent curriculum |
| **RLM** | External/recursive context | Mivi Context VM |
| **λ-RLM** | Typed context combinators | Safe context operations |
| **ToolOrchestra** | Orchestrator + efficiency reward | Mivi router/orchestrator |
| **Harness-R1** | Model + harness co-training | Mivi Agent Harness |
| **Needle** | Tiny specialist models | Mivi-Nano (20-60M) |
| **Colibrì** | Memory hierarchy + predictive prefetch | Adapter cache management |
| **Kimi-K3-in-C** | Deterministic testing + oracle comparison | Tiny fixtures, reference vs optimized |
| **llama2.c** | Minimal full-stack inference | Model-specific Rust engine |
| **Cactus** | Edge engine + tool/RAG integration | Engine architecture reference |
| **rustbpe** | Lightweight Rust tokenizer | Fast tokenizer layer |
| **GigaToken** | SIMD tokenization | System-level tokenizer optimization |
| **TurboQuant** | KV compression | Pluggable KV backend |
| **Bonsai** | Aggressive quantization quality curves | Task-preserving compression |
| **Ling** | Separate total capacity from active compute | Adapter experts |
| **GPT-X2.5-135M** | Small-model architecture ablations | Architecture experiment program |
| **Supra2-100M** | Small-model data/curriculum | Data mixture research |
| **MiniLM-L6-v2** | Embeddings | Local retrieval baseline |
| **OKF** | Portable open knowledge | .mivi/ knowledge format |
| **LongCat-2.0** | Long-context agentic coding | Repository agent benchmarks |
| **ThinkingCap** | Reasoning token efficiency | Compact reasoning training |
| **AirLLM** | Model storage abstraction | Future expert streaming |
| **GPT-2.5** | Full-stack model project | One repository lifecycle |
| **nanoGPT** | Experiment simplicity | Minimal training code |
| **Pokee-Isaac** | Useful vs advertised context | Context quality benchmarks |

---

## 19. Key Research Warnings

### KV compression caveat (from TurboQuant)

High attention similarity does NOT necessarily mean correct text generation. Some 3-4 bit KV configurations achieved good attention metrics but failed to reproduce target text. **Always evaluate against task success, not proxy metrics.**

### Context extension caveat

Do not assume that changing a RoPE parameter means the model "understands 64K." Benchmark retrieval, code, agent history, tool traces, and instruction retention at each context length.

### Quantization caveat (from Bonsai)

Treat quantization as a quality curve. INT4 is the safe primary target. INT3/INT2 are experimental research targets that must be validated against the agent task suite.

### Disk streaming caveat (from Colibrì)

Extreme disk streaming can become disk-bound (0.05-0.1 tok/s for huge models). For Mivi's tiny experts, keep them RAM-resident whenever possible.

---

## 20. Development Strategy: Two Engines

### During development, maintain both:

**Reference engine** (Python/PyTorch):
```
prompt → PyTorch → oracle output
```

**Production engine** (Rust):
```
prompt → Rust → candidate output
```

Compare logits, KV, token IDs, sampling, tool output.

Only optimize the Rust engine after correctness is established against the reference. This philosophy comes directly from Kimi-K3-in-C's testing approach.

---

## 21. Revised Milestone Roadmap

| Milestone | Deliverable |
|---|---|
| M1: Baseline | LFM2.5-350M loader, Rust inference, tokenizer, benchmark: `mivi chat` |
| M2: Engine | INT4, KV cache, sampling, streaming, OpenAI API: `mivi serve` |
| M3: Tools | Tool schema, parser, broker, execution, error handling |
| M4: Agent | Planning, multi-step loop, verification, recovery |
| M5: Memory | Embeddings, persistent storage, retrieval, context management |
| M6: Research | Web search, fetch, source ranking, citations |
| M7: Experts | LoRA adapters, router, 6 initial experts |
| M8: Context VM | RLM-style recursive context, paging, compaction |
| M9: Agent RL | GRPO environment, reward, trajectory evaluation |
| M10: Optimization | Kernel optimization, KV compression, speculative decoding |

---

## 22. North-Star Test

Give Mivi:
- A real repository
- A bug
- A failing test
- Web access
- Filesystem
- Shell

Ask: "Find the problem, research the relevant API if necessary, patch the project, run the tests, and explain what you changed."

Success means:
```
inspect → understand → search when necessary → edit →
test → debug → retest → verify → answer
```

**That is the benchmark we optimize around.**

---

## 23. Feasibility Assessment

| Goal | Feasibility |
|---|---|
| 350M CPU model | **Very high** |
| < 1 GB local inference | **High** |
| Fast laptop inference | **High** |
| OpenAI-compatible Rust server | **Very high** |
| Reliable basic tool calling | **High** |
| Agent loop | **High** |
| Memory/RAG | **Very high** |
| Web research orchestration | **High** |
| 64K effective context | **High** |
| 64K native model context | **Medium** |
| Expert adapters | **High** |
| Good coding on small tasks | **Medium-high** |
| Strong repository agent | **Medium** |
| Frontier-quality reasoning | **Low** |
| Frontier-quality coding | **Low** |
| Excellent capability/MB | **Very high potential** |

---

## 24. Final Architecture

```
                         ┌────────────────────┐
                         │    ANY AGENT       │
                         │ IDE / CLI / APP     │
                         └─────────┬──────────┘
                                   │
                          OpenAI-compatible API
                                   │
                                   ▼
                    ┌───────────────────────────┐
                    │       MIVI RUNTIME        │
                    │          Rust             │
                    └─────────────┬─────────────┘
                                  │
                  ┌───────────────┼────────────────┐
                  │               │                │
                  ▼               ▼                ▼
             Mivi-Nano        Context VM       Tool Broker
                  │               │                │
               routing          RLM              web
               filtering        paging           shell
               selection        retrieval        git
                                  │              code
                                  ▼
                             Memory/RAG
                                  │
                                  ▼
                            Mivi Router
                                  │
                     ┌────────────┼────────────┐
                     ▼            ▼            ▼
                  AGENT         CODE       RESEARCH
                     │            │            │
                     └────────────┼────────────┘
                                  ▼
                         LFM2.5-350M Core
                                  │
                         INT4 / optimized CPU
                                  │
                                  ▼
                              ACTION
                                  │
                                  ▼
                              VERIFY
                                  │
                                  ▼
                              RESULT
```
