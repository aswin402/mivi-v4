# Hybrid Chunked Prefill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Keep each task small, run the targeted verification listed in the task, and do not widen Cargo commands beyond the named package.

**Goal:** Reduce first-token latency for long prompts and agent requests by adding a model-agnostic, correctness-verified chunked prefill path that reuses quantized weight rows across several prompt tokens while preserving hybrid SSM recurrence, attention causality, KV state, prefix-cache snapshots, and the existing token-by-token fallback.

**Architecture:** Keep decode and the current `forward_step` path unchanged. Add a runtime `PrefillStrategy` to `mivi-model`, a separate tile activation/state module, and a batch quantized-matrix API in `mivi-quant`. A tile is processed in model-layer order because `ModelConfig::block_types` may interleave `SSM` and `Attention` layers. Linear projections operate over the tile; SSM convolution/recurrent updates and attention causal state are still advanced in token order where the mathematics requires it. The strategy is selected from runtime configuration, never from a model name or a hardcoded LFM2.5 branch. The default remains the proven token path until all equivalence and benchmark gates pass.

**Tech Stack:** Rust workspace, GGUF memory-mapped tensors, existing Q4_K/Q6_K/Q8_0/F16 quantized kernels, `RunState`, selective `KvCache`, `PrefixCache`, Rayon with the project’s low-thread runtime setting, and the existing CLI benchmark.

## Global Constraints

- Treat the existing uncommitted v0.2.32 profiling changes as user work; do not reset, checkout, or overwrite them.
- Compile and test with `--jobs 1` only. Runtime measurements use `MIVI_THREADS=2` unless a comparison explicitly says otherwise.
- Never run workspace-wide `cargo check`, `cargo build`, or `cargo test`; use only the targeted commands listed below.
- Do not add model-name checks, LFM-specific dimensions, fixed vocabulary IDs, or fixed layer layouts. All behavior must use loaded `ModelConfig`, tensor metadata, and the selected runtime strategy.
- Do not claim a speedup until the cold-prefill and warm-prefix benchmarks show it. A reference implementation that merely calls `forward_step` in a loop is not a chunked optimization and must not be presented as one.
- Keep generation/decode on the current single-token path while prefill is being changed.
- Do not update the version or changelog until the implementation and verification tasks are complete. The release update will increment the current `0.2.32` by `0.0.1` to `0.2.33` and document the inspirations and sources.

---

## Task 1: Establish the prefill strategy contract and a reproducible baseline

**Files:** `crates/mivi-model/src/config.rs`, `crates/mivi-model/src/model.rs`, `crates/mivi-model/src/lib.rs`, `crates/mivi-cli/src/runners/bench.rs`.

- [x] Add a serializable, model-agnostic runtime type, for example:

  ```rust
  #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
  pub enum PrefillStrategy {
      Token,
      Chunked { tile_tokens: usize },
  }
  ```

  Validate `tile_tokens > 0` and expose a safe constructor/default. Do not encode a model family in this type.
- [x] Add `Model::set_prefill_strategy`, `Model::prefill_strategy`, and a private dispatch point used by `generate_tokens_incremental_with_cancel`. Preserve the current loop as the `Token` implementation.
- [x] Make cancellation, context overflow, BOS handling, `compute_logits`, and the `start_prefill_idx` prefix-cache behavior explicit inputs to the dispatch path rather than duplicating prompt preprocessing.
- [x] Extend the focused model tests with invalid tile-size rejection and a test proving the default strategy is `Token`.
- [x] Add a benchmark option or environment setting that selects the strategy and tile size without changing model code. The benchmark must print the selected strategy, effective prompt tokens, TTFT, prefill tok/s, first output latency, and current profile stage totals.

**Verification:**

```text
cargo test -p mivi-model --lib --jobs 1
cargo test -p mivi-cli --lib --jobs 1
cargo check -p mivi --jobs 1
```

## Task 2: Add a real batched quantized projection primitive

**Files:** `crates/mivi-quant/src/lib.rs`, `crates/mivi-quant/src/types.rs`, `crates/mivi-quant/src/q4_k_m.rs`, `crates/mivi-quant/src/q6_k.rs`, `crates/mivi-quant/src/q8_0.rs`, `crates/mivi-quant/src/f16.rs`, and their focused tests.

- [x] Define a checked API with an unambiguous row-major layout, such as:

  ```rust
  pub fn quantized_matmul_rows(
      out: &mut [f32],       // [batch, rows]
      quant_type: GgmlType,
      weights: &[u8],        // [rows, cols] in GGUF row layout
      inputs: &[f32],        // [batch, cols]
      batch: usize,
      rows: usize,
      cols: usize,
  ) -> Result<()>
  ```

  Validate all shape/overflow conditions before touching buffers. Reuse the existing tensor block-size and row-byte rules rather than duplicating format constants.
- [x] Implement the first version for every quantization type accepted by `quantized_matvec`. The implementation must perform a matrix-times-matrix operation over the tile, not call the public matvec once per input row. The initial kernel may use a clear scalar/reference inner loop when a format lacks a batch kernel, but it must have tests and a benchmark label identifying that fallback.
- [x] Preserve existing SIMD matvec behavior and exports. Add per-format tests comparing each batched result with independently computed scalar/matvec results, including batch sizes `1`, `2`, and a non-power-of-two tile.
- [x] Add malformed dimension, short input, short output, and short weight-buffer tests. Ensure no unsafe slice is formed before checked validation.

**Verification:**

```text
cargo test -p mivi-quant --lib --jobs 1
cargo check -p mivi-model --jobs 1
```

## Task 3: Add tile activation storage and correctness helpers

**Files:** new `crates/mivi-model/src/prefill.rs`, `crates/mivi-model/src/lib.rs`, optionally `crates/mivi-core/src/arena.rs` only if a reusable allocation is justified.

- [x] Introduce a tile-owned activation structure with explicit `[token, dim]` and `[token, hidden]` layouts, ping-pong buffers, and reusable projection scratch. Size it from `ModelConfig`; do not allocate per token.
- [x] Add helpers for embedding a token range, RMS-normalizing rows, residual addition, SwiGLU, and final-row-only output projection. Each helper must state whether it preserves all rows or only the last row.
- [x] Keep recurrent SSM buffers and the existing `RunState` separate from tile activations. Do not silently reuse one token’s `RunState` arrays for several tokens.
- [x] Add small synthetic tests for layout/indexing, tile lengths of `1`, `2`, and `64`, and final-row logits selection. These tests should not load a production model.

**Verification:**

```text
cargo test -p mivi-model --lib --jobs 1
```

## Task 4: Implement SSM tile execution without changing recurrence semantics

**Files:** `crates/mivi-model/src/ssm.rs`, new `crates/mivi-model/src/prefill.rs`, and focused model tests.

- [x] Add a tile forward function that accepts a tile of hidden rows, one `SsmLayerWeights`, the loaded tensor bytes, adapters, and the absolute starting position. It must update the same per-layer `conv_states` and `ssm_states` owned by `RunState`.
- [x] Batch the independent projections (`in_proj`, `out_proj`, and FFN gate/up/down) with `quantized_matmul_rows` where dimensions permit.
- [x] Advance B/C/X elementwise operations, depthwise causal convolution, and any recurrent hidden-state update in increasing token order. The state after token `i` must be the input state for token `i+1`; no parallel reorder is allowed.
- [x] Apply norms, residuals, LoRA, and SwiGLU with the same operation order and numerical conventions as `ssm_forward`. Active LoRA currently uses the token fallback rather than being silently dropped.
- [x] Add a test-only comparison helper that runs one tile through the tile function and the same tokens through `ssm_forward`, then compares every output row and exported SSM state within a documented tolerance.

**Verification:**

```text
cargo test -p mivi-model --lib ssm --jobs 1
cargo test -p mivi-model --lib --jobs 1
```

## Task 5: Implement attention tile execution with causal KV correctness

**Files:** `crates/mivi-model/src/transformer.rs`, `crates/mivi-kv/src/cache.rs` if a safe range-store/read API is required, `crates/mivi-model/src/prefill.rs`, and focused tests.

- [x] Add a tile attention function that computes Q/K/V rows for the tile, applies the existing optional Q/K norms and RoPE at absolute positions, and stores K/V at exactly the same cache positions as `compute_qkv`.
- [x] For each query row, attend only to positions `0..=absolute_position`. Current cached tokens and earlier rows in the tile must be visible; later rows must not be visible. Preserve GQA head mapping and the active KV precision.
- [x] Reuse the existing numerically stable online-softmax logic or extract it into a tested range helper. Do not materialize an unbounded dense attention matrix.
- [x] Batch output projection and shared FFN projections after attention, preserving residual and adapter order.
- [x] Compare tile attention output, KV current position, and exported KV state with token-major execution for prompt lengths `1`, `2`, `8`, and a tile crossing an attention layer.

**Verification:**

```text
cargo test -p mivi-kv --lib --jobs 1
cargo test -p mivi-model --lib transformer --jobs 1
cargo test -p mivi-model --lib --jobs 1
```

## Task 6: Orchestrate hybrid tiles and preserve prefix-cache boundaries

**Files:** `crates/mivi-model/src/model.rs`, `crates/mivi-model/src/prefill.rs`, `crates/mivi-kv/src/prefix.rs` only if metadata needs a safe assertion, and model integration tests.

- [x] Implement the chunked dispatch using `config.block_types` and `weights.layers` in their loaded order. For each tile: embed rows, run every layer in order, and compute final norm/output logits only for the final prompt row when requested.
- [x] Keep decode calls on `forward_step` and ensure the next sampled token observes the same `RunState`, KV position, and SSM state as the token path.
- [x] Make tile selection stop at a prefix-cache snapshot boundary when necessary. A snapshot at position 64/128/etc. must be exported after exactly that token, even when the requested tile size is 32, 64, or 128. Never store a snapshot from the middle of a tile under the wrong position.
- [x] Preserve cancellation and progress reporting. Cancellation between tiles must leave the model in a documented state; cancellation inside a tile must either be checked at safe token boundaries or return without exposing partially committed cache state.
- [x] Add an integration test using `MIVI_TEST_MODEL` (ignored when unset) and a small synthetic/fake-weight test where possible. Compare token and chunked execution for tile sizes `1`, `2`, `8`, and `64`, prompt lengths `63`, `64`, `65`, and a prompt with a cached 64-token prefix.
- [x] Compare logits, generated token IDs with a fixed seed, `current_pos`, exported KV state, exported convolution state, and exported SSM state. The current integration gate uses a `1e-3` F32 tolerance; quantized-KV coverage remains a benchmark follow-up.

**Verification:**

```text
cargo test -p mivi-model --lib --jobs 1
MIVI_TEST_MODEL=models/mivi-tiny-test.gguf cargo test -p mivi-model test_chunked_prefill_matches_token_path --lib --jobs 1 -- --ignored
```

## Task 7: Benchmark, choose the safe default, and document operational use

**Files:** `crates/mivi-cli/src/runners/bench.rs`, relevant README/docs, `docs/KV_QUANT_AND_CHUNKED_PREFILL_RESEARCH.md`, and `docs/IMPLEMENTATION_PLAN_64K_LONG_CONTEXT_AND_HYBRID_SCALING.md`.

- [x] Run and retain a complete successful cold and warm-prefix sweep for the existing LFM2.5 Q4 model and at least one other model in `models/`, using `MIVI_THREADS=2`, with tile sizes `1`, `2`, `8`, `32`, and `64` where supported. The final-code matrix completed for LFM2.5 1.2B and the separate Mivi Q4 model; results and tile sensitivity are documented in the changelog and research notes.
- [x] Run the same prompts through token and chunked strategies and verify identical deterministic output. The ignored real-model equivalence test covers lengths `63`, `64`, and `65`, tile sizes `1`, `2`, `8`, and `64`, prefix-cache state, KV state, convolution state, and continuation IDs.
- [x] Keep `Token` as the default because the measured LFM cold result is effectively tied: `9.08` tok/s token-major versus `9.04` tok/s chunked tile-64 with `MIVI_THREADS=2`. The benchmark environment selector provides an explicit chunked/token override.
- [x] Update research and implementation docs to distinguish measured results from targets. Explain that SSM recurrence remains ordered and that the optimization comes from safe batching/reuse around it, not from assuming all layers are Transformer attention.
- [x] Run only the targeted final checks:

  ```text
  cargo test -p mivi-quant --lib --jobs 1
  cargo test -p mivi-kv --lib --jobs 1
  cargo test -p mivi-model --lib --jobs 1
  cargo test -p mivi-cli --lib --jobs 1
  cargo check -p mivi --jobs 1
  cargo build -p mivi --release --bin mivi --jobs 1
  ```

## Task 8: Release bookkeeping

**Files:** `Cargo.toml`, `Cargo.lock`, `CHANGELOG.md`.

- [x] Increment the workspace/package version from `0.2.32` to `0.2.33` and update only the matching lockfile package entries.
- [x] Add a changelog entry dated with the actual release date. Include the user-visible behavior, compatibility/fallback behavior, benchmark evidence, correctness gates, and known limitations.
- [x] Explicitly mention the ideas and sources that informed the work: the project’s `KV_QUANT_AND_CHUNKED_PREFILL_RESEARCH.md`, the existing 64K/chunked-prefill implementation plan, the LMCache prefix-cache concept, and llama.cpp’s batched/ubatch prefill approach. Link to public sources where used, and label any expected speedup as a measured result or future target.
- [x] Run `git diff --check`, inspect the final diff for unrelated formatting, and report the exact targeted commands and results. Do not push until the user explicitly requests the push.

## Completion gates

- Token-major and chunked paths produce equivalent logits/state/output on the required test matrix.
- Prefix-cache snapshots restore at exact chunk boundaries for every supported tile size.
- No model-family-specific conditionals or hardcoded architecture dimensions were added.
- Quantized batch kernels pass malformed-input tests and preserve existing matvec tests.
- Agent-style prompts with tool definitions receive a response under the selected strategy; latency and formatting are measured separately from model capability.
- The release is not called complete if only the compiler passes; targeted real-model tests and benchmark evidence are required.
