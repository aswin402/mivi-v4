# Changelog

All notable changes to **Mivi-v4** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v0.2.4] - 2026-09-01

### BF16 Matvec Dispatch, 262k Token Bitmask, Polymorphic Message Payloads, XML Control Stripping & Diagnostic Doctor

#### 💡 Ideas, Inspirations & Sources
- **BFloat16 & Half-Precision Linear Algebra (`mivi-quant`)**:
  - *Full BF16 Dispatch*: Implemented `try_matvec_bf16` and `matvec_bf16` with checked overflow bounds, and wired `GgmlType::BF16` into `quantized_matvec`.
- **Large-Vocabulary Grammar Masking (`mivi-model`)**:
  - *262k Token Coverage*: Scaled `BITMASK_WORDS` to 4,096 words (covering up to 262,144 tokens), preventing tokens $\ge 65,536$ in modern models (LLaMA 3, Qwen 2.5, Gemma 2) from bypassing JSON/tool grammar constraints.
- **OpenAI Multi-Part Content Specification (`mivi-server`)**:
  - *Polymorphic Content Deserialization*: Added `deserialize_polymorphic_content` to `MessageDto` to accept both plain strings and arrays of content parts (`[{"type": "text", "text": "..."}]`) from LangChain, Cursor, and modern SDKs.
- **W3C XML 1.0 Specification & Defensive Encoding (`mivi-agent`)**:
  - *Control Character Stripping*: Filtered invalid non-whitespace ASCII control characters (`\x00`–`\x08`, `\x1F`) in `escape_xml_common` to prevent downstream parser crashes.
- **Memory Record Formatting Integrity (`mivi-memory`)**:
  - *Preserved Indentation*: Fixed frontmatter delimiter stripping in `load_record` to preserve code block and YAML indentation. Added deterministic sorting to `list_records`.
- **Defensive Filesystem Sandboxing (`mivi-tools`)**:
  - *Bounded File Writes & Device File Rejection*: Enforced `MAX_FILE_WRITE_BYTES = 5MB` and verified `meta.is_file()` in `handle_read_file` to prevent device hangs.
- **System Environment Discovery (`mivi-cli`)**:
  - *Comprehensive `mivi doctor`*: Expanded system diagnostic suite to report AVX2, FMA, AVX-512F, ARM64 NEON, thread pool settings, `.mivi` workspace state, and discovered GGUF models.

---

## [v0.2.3] - 2026-09-01

### Universal Layer Norm Dequantization, Pratt Parser Sandboxing, Tool Feedback & API Polish

#### 💡 Ideas, Inspirations & Sources
- **Pratt Parsing & Defensive Compiler Engineering**:
  - *Recursion Depth Guarding*: Enforced `MAX_PARSER_DEPTH = 128` recursion limit in `mivi-tools::builtins::calc_parser`, neutralizing potential stack exhaustion crashes from deeply nested malicious or hallucinated parentheses.
- **Universal GGUF Quantization Handling**:
  - *Multi-Type Norm Dequantization*: Replaced raw float slice assumption in `mivi-model::loader::resolve_f32_vec` with `mivi_quant::dequantize_slice`, enabling correct weight loading across GGUF models with F32, F16, or BF16 layer norms.
- **Anthropic Messages Specification & Polymorphic Content**:
  - *Polymorphic System Field*: Added dual parsing support in `mivi-server::routes::anthropic` for both raw `String` and structured `[{"type": "text", "text": "..."}]` content blocks as emitted by official Anthropic SDKs.
  - *SSE Keep-Alive & Dynamic Token Estimation*: Configured SSE keep-alive heartbeat and dynamic input token estimation.
- **Defensive Agent Sandboxing & Error Recovery**:
  - *Resource Bounded Filesystem Tools*: Capped file reading (`MAX_FILE_READ_BYTES = 5MB`) and directory listings (`MAX_DIR_ENTRIES = 500`) to prevent out-of-memory crashes.
  - *Actionable Tool Syntax Feedback*: Intercepted `__parse_error` in `ToolBroker::execute` to route explicit JSON syntax errors back to the model, allowing autonomous error recovery.
- **Modern OpenAI SDK Compatibility**:
  - *`max_completion_tokens` Field Alias*: Added `#[serde(alias = "max_completion_tokens")]` to `ChatCompletionRequest`.

#### 🛠️ Features, Fixes & Polish
- **Harmonized BOS Injection (`mivi-model`)**: Aligned `generate_tokens_incremental` with ChatML heuristics to prevent accidental leading BOS tokens.
- **Dynamic Output Norm Epsilon (`mivi-model`)**: Passed `cfg.rms_norm_eps` in final output projection norm.
- **Banner & Route Transparency (`mivi-cli`)**: Added `POST /v1/messages` to the server startup banner.
- **Adaptive Rayon Thread Pool (`main.rs`)**: Sized Rayon worker threads dynamically from available CPU cores while respecting `MIVI_THREADS` / `RAYON_NUM_THREADS` and `RUST_LOG`.
- **92 Total Passing Tests (100% Pass Rate)**: Verified across all 13 workspace crates.

---

## [v0.2.2] - 2026-09-01

### Codebase Hardening, Full Specification Compliance, Panic Elimination & Audit Fixes

#### 💡 Ideas, Inspirations & Sources
- **RFC 8259 (The JavaScript Object Notation Data Interchange Format)**:
  - *Full Numerical Grammar Compliance*: Updated `JsonGrammar` literal matching to properly accept decimal points (`.`), signs (`+`, `-`), and exponential notations (`e`, `E`), preventing premature grammar rejection when models generate floating-point and scientific numbers.
- **Anthropic Messages SSE Streaming Protocol Specification**:
  - *Complete Event Stream Lifecycle*: Implemented the full Anthropic streaming event lifecycle (`message_start` $\to$ `content_block_start` $\to$ `content_block_delta` $\to$ `content_block_stop` $\to$ `message_delta` $\to$ `message_stop`), ensuring compatibility with official Anthropic SDKs (Python, TypeScript), Claude Code, and Cursor.
  - *`x-api-key` & CORS Support*: Added native `x-api-key` authentication header extraction alongside Bearer tokens and added Anthropic header support to CORS preflight headers.
  - *Multi-Turn Tool Call & Result Handling*: Preserved `tool_use` and `tool_result` content blocks when converting Anthropic requests to ChatML.
- **OpenAI API Tool Calling Specification**:
  - *Serialized String Arguments*: Enforced that `function.arguments` is strictly emitted as a JSON-serialized `String` rather than a raw JSON object, adhering to standard OpenAI client deserialization requirements.
- **Unicode Standard & Rust UTF-8 Safety**:
  - *Character-Boundary Slicing*: Replaced raw byte slicing in `mivi-server::logging` with `summarize_prompt` using safe character iterators, eliminating runtime panics on multi-byte UTF-8 user prompts.
- **LMCache & Adaptive Engine Design**:
  - *Running Counter $O(K)$ Elastic Memory Pruning*: Added `total_memory_bytes` tracking in `PrefixCache` to eliminate $O(N)$ per-chunk recalculations during eviction.

#### 🛠️ Bug Fixes & Code Quality Upgrades
- **Grammar Floating-Point Parsing (`mivi-model::grammar`)**: Fixed floating-point number rejection by adding `.`, `+`, `e`, `E` to literal patterns.
- **Safe UTF-8 Slicing (`mivi-server::logging`)**: Eliminated potential byte-slicing panics on non-ASCII prompts.
- **Speculative Decoding Boundary (`mivi-model::pld`)**: Fixed PLD search boundary to propose continuation tokens on adjacent repeating n-grams.
- **BPE Byte Fallback Reverse Mapping (`mivi-tokenizer::bpe`)**: Fixed byte fallback to reverse GPT-2 mapped unicode characters to original byte tokens.
- **Context Store Bounded Memory (`mivi-context::store`)**: Enforced strict capacity bounds when 100% of blocks are pinned.
- **Deterministic FS Tool Output (`mivi-tools::builtins::fs`)**: Sorted directory listing output for deterministic reproducibility.
- **Agent Loop Error Tracking (`mivi-agent::engine`)**: Added `status="error"` attribute to tool results on failure and set phase to `Observing`.
- **Dynamic RMSNorm Epsilon (`mivi-model`)**: Passed `cfg.rms_norm_eps` instead of hardcoded default across SSM and Transformer modules.
- **Clamped Agent Steps (`mivi-server::routes::agent`)**: Clamped `max_steps` to `MAX_AGENT_STEPS_LIMIT = 50` to prevent unbounded execution loops.
- **Re-exported Tensor Module (`mivi-core`)**: Re-exported `tensor.rs` primitives in `mivi-core::lib`.
- **91 Total Passing Tests (100% Pass Rate)**: Verified full test suite across all 13 workspace crates.

---

## [v0.2.1] - 2026-09-01

### FreeToken Semantic Anchors, Elastic Memory, Grammar Logit Masking, PLD & Anthropic API Compatibility

#### 💡 Ideas, Inspirations & Sources
- **FreeToken** (*FlashML / UC Berkeley Sky Computing / MIT HAN Lab*, [arXiv:2608.16157](https://arxiv.org/abs/2608.16157), [GitHub](https://github.com/FlashML-org/FreeToken)):
  - *Semantic Anchor Checkpointing*: Snapshots recurrent and KV states at natural structural boundaries (`<|im_start|>`, `<think>`, `<tool_call>`), avoiding cache invalidation when agents trim reasoning traces or edit tool results.
  - *Elastic Memory Management*: Dynamically prunes cached chunks under high RAM pressure without restarting the engine.
  - *Double-Buffered Layer Execution*: Zero-copy ping-pong activation streaming (`x_ping` $\leftrightarrow$ `x_pong`) keeping L1/L2 caches hot.
- **llguidance (Microsoft) & Outlines (dottxt)**:
  - *Grammar-Constrained Decoding*: Deterministic Pushdown Automata (PDA) tracking JSON `{`, `}`, `[`, `]`, literals, and escaping, combined with a 65,536-bit zero-allocation stack bitset (`TokenBitMask`) setting invalid token logits to $-\infty$ before softmax.
- **Prompt Lookup Decoding (Google Research / Apoorv Saxena)**:
  - *Prompt Lookup Proposer*: 3-gram n-gram context matching proposing speculative draft continuation slices in $< 5\text{ µs}$ with zero extra parameter overhead.
- **Anthropic Messages Specification**:
  - *Claude Code & OpenCode Compatibility*: Drop-in support for `POST /v1/messages` with structured `tool_use` blocks and SSE streaming.

#### 🌟 Features & Upgrades
- **Semantic Anchor Checkpoints (`mivi-kv::semantic`)**: Added `SemanticAnchorCache` supporting $O(1)$ state rollback on agent thinking trace trims and tool result insertions.
- **Elastic RAM Watchdog Pruning (`mivi-kv::prefix` & `mivi-server`)**: Added `PrefixCache::prune_to_bytes` connected to `RamWatchdog` memory monitoring.
- **Double-Buffered Layer Ping-Pong (`mivi-core::arena`)**: Added `x_pong` preallocated buffer to `RunState` for zero-allocation layer streaming.
- **Grammar-Constrained Logit Masking (`mivi-model::grammar`)**: Built `TokenBitMask`, `JsonGrammar`, and `ToolCallGrammar` guaranteeing 100% syntactically valid JSON output.
- **Prompt Lookup Speculative Decoding (`mivi-model::pld`)**: Implemented `PromptLookupProposer` for rapid draft generation.
- **Anthropic `/v1/messages` Endpoint (`mivi-server::routes::anthropic`)**: Exposed native Claude Code / OpenCode compatible endpoint.
- **89 Passing Tests**: Comprehensive test suite verified with 100% pass rate across all 12 workspace crates.

---

## [v0.2.0] - 2026-09-01

### LMCache-Inspired Prefix Caching, Hybrid State Serialization & Persistent Disk Cache (.kvc)

#### ⚡ Chunk-Based Prefix Caching (`mivi-kv::prefix`)
- **64-Token Chunk Partitioning**: Implemented `PrefixCache` that partitions input token sequences into 64-token chunks and computes hierarchical 64-bit FNV-1a rolling hashes.
- **$O(1)$ Instant Time-To-First-Token (TTFT)**: If an incoming prompt shares a prefix (such as a system prompt, tool schemas, or multi-turn history), the engine matches the prefix chunks, restores the recurrent states in sub-milliseconds, and skips forward-pass computation for all matched tokens.
- **Bounded In-Memory LRU Eviction**: Manages cached chunk snapshots with LRU eviction under a fixed memory budget.

#### 🧠 Hybrid SSM + Attention State Snapshotting (`mivi-model`)
- **Dual-State Serialization**: Designed `HybridStateSnapshot` capturing both the **6 GQA Attention KV layers** and the **10 Gated ShortConv SSM 1D convolution rolling buffers** (`conv_states`).
- **Seamless Prompt Prefill Hook**: Integrated `find_longest_prefix` into `Model::generate_tokens_incremental`, enabling automatic caching during prefill and zero-compute restoration on cache hits.

#### 💾 Persistent On-Disk KV Cache (`.mivi/cache/*.kvc`)
- **High-Performance Binary Format**: Designed `.kvc` disk format with `MIVIKVC1` magic headers, model checksum verification, token sequences, and raw float buffer storage.
- **Instant Cross-Process State Loading**: Allows large documents, codebase context, and fixed system prompts to be saved to disk and loaded across server/CLI restarts without running forward passes.

#### 🛠️ CLI & Task Runner Integration
- **`mivi cache list` / `just cache-list`**: Displays all persisted `.kvc` files, token lengths, model hashes, and file sizes.
- **`mivi cache clear` / `just cache-clear`**: Clears on-disk cache files to reclaim disk space.
- **78 Total Passing Tests**: Expanded test suite with unit and integration tests covering chunk hashing, LRU eviction, state import/export, and disk roundtrips.

---

## [v0.1.2] - 2026-08-31

### Hono-Style Minimal Logs, Resource Safety Watchdog, BOS Alignment & AI Agent Test Suite

#### 🎨 Hono-Style Minimal Terminal Logging & Telemetry
- **Minimal Borderless Startup Banner**: Replaced boxed ASCII banners with a clean, borderless startup display featuring model name, listening address, registered tool counts, real-time RSS memory, and active route tables.
- **Colorized Real-Time Request Logging Middleware (`mivi_log_middleware`)**: Built a zero-dependency ANSI logging middleware logging HTTP methods (`GET` in green, `POST` in cyan), routes, colored status codes (2xx green, 4xx yellow `⚠`, 5xx red `✗`), high-resolution latencies (`µs`, `ms`, `s`), truncated user prompt previews, token counts (`prompt→completion`), and tool call markers (`🔧`).
- **Prompt & Usage Metadata Propagation**: Attached `LogMetadata` to Axum response extensions in `/v1/chat/completions` (blocking & streaming) and `/v1/mivi/agent`.

#### 🛡️ Resource Safety Watchdog (`Safelock`)
- **Background RAM Monitoring**: Added `ResourceWatchdog` supervisor polling `/proc/self/statm` process RSS physical memory every 3 seconds to protect host systems from OOM or resource starvation.
- **Two-Tier Threshold Enforcement**: Emits yellow warning logs when memory crosses 700 MB and triggers an automatic graceful shutdown at 900 MB before system freeze.
- **Configurable CLI Flags**: Added `--max-memory`, `--warn-memory`, and `--no-safelock` options to `mivi serve`.

#### 🧠 Inference Accuracy & BOS Positional Embedding Anchor
- **Unconditional BOS `<|startoftext|>` Insertion**: Fixed position-0 BOS token injection for ChatML templates in `crates/mivi-model/src/model.rs`, ensuring proper attention head initialization on `LFM2.5-350M`.
- **Conversational ChatML Formatting**: Removed intrusive default `<think>` system prompts from standard conversational chat, enabling fluent multi-turn chat responses without repetitive echo patterns. Added `--thinking` and `--system` (`-s`) CLI options to `mivi chat`.

#### 🧪 Real-World AI Agent Testing Suite
- **OpenAI Python SDK Integration**: Created `scripts/test_agents/01_openai_sdk.py` verifying drop-in OpenAI API compatibility.
- **Tool-Calling Agent Loop**: Created `scripts/test_agents/02_agent_loop.py` demonstrating iterative tool execution with the built-in calculator.
- **Native Autonomous Agent**: Created `scripts/test_agents/03_native_agent.py` testing the `/v1/mivi/agent` multi-step SSE streaming endpoint.
- **Test Suite Expansion**: Added unit tests for logging utilities and watchdog state transitions; **74 total tests passing across all 12 workspace crates**.

---

## [v0.1.1] - 2026-08-30

### Enterprise Security, Panic Elimination, SIMD Dispatch, RegexSet Router & Robustness Hardening

#### 🛡️ Security & Safety Hardening
- **GQA Head Divisibility Validation**: Added `!self.n_heads.is_multiple_of(self.n_kv_heads)` check inside `ModelConfig::validate()`, rejecting misconfigured model configurations at load time.
- **Fallible `GgmlType::block_size()`**: Changed `block_size()` to return `Option<usize>` and added `block_size_checked()` returning `Result<usize, QuantError>`, preventing panics on unsupported or unknown quantization types.
- **Prompt Injection Defense**: Sanitized user task prompts via XML escaping (`mivi_agent::escape_xml_content`) before interpolating into agent system templates.
- **Localhost-Restricted CORS**: Replaced permissive CORS with strict origin validation restricted to `localhost` and `127.0.0.1`.
- **Constant-Time API Key Comparison**: Integrated `subtle::ConstantTimeEq` for timing-attack-safe Bearer token verification.

#### 🛑 Panic Elimination & Fallible APIs
- **Safe Fallible LoRA**: Implemented `LoraWeightPair::try_apply()` returning typed errors on shape mismatches without crashing the runtime.
- **Checked Quantized MatVec**: Added `validate_matvec_args()` and fallible `try_matvec_f16()`, `try_matvec_q8_0()`, `try_matvec_q4_k_m()` with full input/output bounds checks.
- **Checked RoPE Cache**: Added `RopeCache::try_apply()` and `try_rotate_heads()` returning `Result<(), RopeError>` for safe rotary position embedding application.
- **Model Dimension Invariant Checks**: Enforced `final_norm` dimension matching against model dimension with `ModelError::DimMismatch`.
- **Engine Actor Graceful Recovery**: Gracefully closed communication channels and logged errors if runtime spawning fails.

#### ⚡ Performance & Routing Optimizations
- **Single-Pass `RegexSet` Classifier**: Replaced sequential regex scans with `RegexSet` in `IntentClassifier` for fast, single-pass intent routing.
- **SIMD Function Pointer Dispatch**: Optimized `matvec_f32()` to dispatch via a pre-resolved `LazyLock<MatvecFn>` pointer, bypassing runtime branching.
- **Zero-Allocation Error Responses**: Optimized `AppError::into_response()` to consume `self` by value, eliminating heap string clones.
- **Vocab Buffer Reuse**: Preallocated string buffers during vocabulary extraction in GGUF loader.

#### 🧹 Deduplication & Architecture Polish
- **Layer & Weight Resolvers**: Centralized GGUF block naming patterns into `layer_tensor_name()` and `layer_module_name()`.
- **Deduplicated QKV Projections**: Extracted unified linear projection closure in `compute_qkv()`.
- **SSE Error Helper**: Standardized SSE stream error payloads with `create_error_chunk_event()`.
- **Named Constants**: Centralized stop token constants (`DEFAULT_STOP_TOKEN_IM_END`, `DEFAULT_STOP_TOKEN_ENDOFTEXT`) and GGUF metadata keys.

#### 🧪 Verification & Hygiene
- **44 comprehensive workspace unit and integration tests passing**.
- **0 Clippy warnings** with `-D warnings` on all targets.
- **100% `cargo fmt` formatting compliance**.

---

## [v0.0.4] - 2026-08-29

### Engine Actor Concurrency, Stateful PRNG, BPE Merge Ranks, API Key Auth & Enterprise Hardening

#### 🧵 Concurrency & Architectural Safety
- **Dedicated `EngineActor` OS Compute Thread**: Replaced `Arc<Mutex<Model>>` shared locks with an actor architecture running model compute on a dedicated OS thread `"mivi-engine-actor"` communicating via non-blocking Tokio `mpsc` channels (`EngineHandle`), eliminating lock contention during parallel HTTP requests and streaming.
- **Optional API Key Authentication**: Added `require_api_key` Axum middleware for `/v1/*` routes checking `Bearer` authorization headers against the `MIVI_API_KEY` environment variable.
- **Strict Server Body & Token Limits**: Enforced `DefaultBodyLimit::max(2MB)` and clamped `max_tokens` (1 to 8192) on chat completions with structured OpenAI error responses (`invalid_request_error`).
- **Structured Error Handling (`AppError`)**: Implemented `IntoResponse` for `AppError` returning RFC-compliant `OpenAiErrorResponse` JSON with HTTP status codes (400, 401, 500, 503).
- **Non-Blocking Tool Execution**: Wrapped tool execution in `tokio::task::spawn_blocking` and enforced a default 30-second execution timeout via `tokio::time::timeout`.

#### 🧠 Mathematical Correctness & Tokenizer Precision
- **Stateful Xorshift64* PRNG**: Replaced per-sample `SystemTime::now()` non-deterministic RNG with a fast, seedable 64-bit linear state machine (`Sampler::with_seed()`), ensuring reproducible sampling and low overhead.
- **True BPE Merge Rank Ordering**: Updated BPE encoder to look up merges against a precomputed `merge_ranks: HashMap<Vec<u8>, usize>` rather than relying on vocabulary IDs.
- **UTF-8 Multi-Byte Fallback Decoding**: Replaced raw `char` casting with a raw byte accumulator (`Vec<u8>`) and `String::from_utf8_lossy`, properly decoding multi-byte UTF-8 Unicode characters without corruption.
- **GPT-4 / LLaMA Style Regex Pre-Tokenization**: Integrated regex pre-tokenization into BPE encoding to match standard language model tokenization boundaries.
- **Context Length Overflow Guard**: Added explicit sequence length check in `generate()` loop, cleanly terminating generation before exceeding model context capacity.
- **Zero-Allocation Logits Scratchpad**: Added `logits_scratch: Box<[f32]>` to `RunState`, eliminating per-token heap allocations during model output projection.
- **Error Propagation Across Forward Pass**: Updated `ssm_forward` and `attention_forward` to return `Result<()>` and propagate quantization and KV cache errors with `?`.
- **Result-Returning KV Cache Accessors**: Refactored `KvCache::get_k` and `get_v` to return `Result<&[f32]>` with out-of-bounds error verification.

#### 🛡️ Tool Robustness & Feedback
- **Malformed Tool Call Error Feedback**: Updated markup parser to generate structured `__parse_error` synthetic tool calls when invalid JSON or missing fields are produced by the model.
- **Dynamic SSM Sizing**: Dynamically sized fallback state vectors from model configuration (`ssm_state_dim`, `ssm_conv_kernel`).
- **LoRA Rank-0 Guard**: Added explicit zero-rank protection and dimension assertions to LoRA adapter computations.
- **Dynamic Sysconf Page Size**: Calculated Linux memory RSS telemetry dynamically via `sysconf(_SC_PAGESIZE)`.

#### 📦 Build Hygiene & Expanded Test Suite
- **Modernized Dependencies**: Migrated from unmaintained `serde_yaml` to `serde_yaml_ng` v0.10. Trimmed unused dependencies across 6 workspace crates (`mivi-core`, `mivi-model`, `mivi-server`, `mivi-agent`, `mivi-tools`, `mivi-memory`).
- **Isolated Test Environments**: Replaced shared temp directories with isolated `tempfile::tempdir()` across tests.
- **Robust Integration Testing**: Added tests for agent step exhaustion, stagnation detection, unknown tool handling, and verified SSE stream chunk payloads.
- **Manifest-Relative Pathing**: Updated oracle and integration tests to resolve test fixtures relative to `CARGO_MANIFEST_DIR`.
- **26 comprehensive unit, integration, and golden oracle tests passing**.

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
