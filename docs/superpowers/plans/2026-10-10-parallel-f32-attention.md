# Parallel F32 Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Test whether independent F32 query heads can execute faster without changing any attention result or production default.

**Architecture:** Retain the current serial GQA function as an independent reference. A default-off diagnostic flag in RunState selects a safe F32 candidate, partitioning disjoint outputs using the existing Rayon pool. Model diagnostics compare fixed input IDs, logits and hybrid state before measuring latency.

**Tech Stack:** Rust, existing workspace Rayon, feature-gated mivi-core/mivi-model diagnostics.

## Global Constraints

- Approved spec: `docs/superpowers/specs/2026-10-10-parallel-f32-attention-design.md`.
- Cargo jobs1; Rayon threads2 for measurement; serial test harness; no concurrent Cargo/inference commands.
- No new unsafe code, dependency download, model-name dispatch, hidden tuning or production promotion.
- Unsupported KV precisions, one worker and one head retain established serial arithmetic.
- Each child wall limit180s and total measurement session1200s; retain failures and partial results.
- Preserve and exclude the user's .gitignore edit. Completion increments patch version once and records sources before publishing the existing branch.

## Task 1: Safe independent-head operator

**Files:** `crates/mivi-core/Cargo.toml`, `crates/mivi-core/src/arena.rs`, `crates/mivi-model/Cargo.toml`, `crates/mivi-model/src/transformer.rs`; new `crates/mivi-model/src/transformer/parallel_attention.rs` and its `tests.rs`.

**Interfaces:** candidate `compute(state: &mut RunState, kv: &KvCache, layer: usize, pos: usize, cfg: &ModelConfig) -> Result<()>`; `RunState.parallel_attention_enabled: bool` defaults false, `parallel_attention_calls: usize` records actual parallel dispatch. Model feature forwards the core flag and enables optional existing Rayon.

- [x] Write nonuniform F32 MHA/GQA output-bit tests, invalid-dimension/position/layer sentinels and a bounded ignored operator timing test. Begin with a compilable candidate stub returning ExecutionFailed, and observe the parity test fail.
- [x] Run `CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 cargo test -p mivi-model --lib --release --offline --jobs 1 --features parallel-attention-experiment parallel_attention -- --test-threads=1`.
- [x] Implement checked query/output products, divisibility and KV width/position/layer validation before writes. Validate checked getter access to first/last causal KV rows. Within each head use ascending positions and the existing softmax arithmetic; checked access errors map to ModelError.
- [x] Partition with `heads_per_task = cfg.n_heads.div_ceil(rayon::current_num_threads().min(cfg.n_heads));` and `state.attn_out[..query_len].par_chunks_mut(heads_per_task * cfg.head_dim).enumerate().try_for_each(...)`. Queries and KV remain immutable; no pool creation. Join all tasks before returning.
- [x] Add guarded dispatch before the existing serial loop: feature enabled, explicit selector true, F32, at least two heads/workers. Other cases use the untouched function. Increment the counter only after successful candidate execution.
- [x] Re-run the focused filter and `transformer::tests` with the feature, then the existing transformer filter without it. Check rustfmt on touched Rust files and `git diff --check`.

## Task 2: Model parity and bounded measurements

**Files:** `crates/mivi-model/src/model.rs`; new `crates/mivi-model/src/model/parallel_attention_tests.rs`.

**Interfaces:** feature-gated `Model::set_parallel_attention_experiment(&mut self, enabled: bool)` sets the RunState selector. Reset preserves selector but clears dispatch count. Tests use `Model::generate_tokens_incremental`, `Model::forward`, KV export and recurrent-state arrays.

- [x] Add tiny-fixture reset/continuation/multi-tile/prefix-restoration controls. Compare complete logits, generated IDs, KV exports and convolution/SSM bits between baseline/candidate. Verify selector defaults false and actual dispatch occurs under a two-thread pool; Q8/TurboQuant fallback counter stays zero.
- [x] Add one ignored live measurement accepting only an absolute `MIVI_TEST_MODEL` path. Tokenize fixed synthetic workspace text once, select 512/2048 IDs, use F32, context4096, tile64, greedy sampling and one emitted-token cap. Model loading, parity/state comparisons and logging are outside measured generation timers.
- [x] Run the operator pilot before live model measurements. Run the live test sequentially with external180s child bounds, explicit prompt-size/route/pair selectors and session1200s. Three alternating unprofiled pairs per size; profile controls separately. Compare identical work, preserve original failures and do not treat a missing first emission as success.
- [x] Report raw times and paired ratios, cache state/work counts, complete parity results and limitations. If child/session caps prevent completion, report incomplete evidence without extending budgets or promoting the candidate.

## Task 3: Review and publication

**Files:** `docs/PARALLEL_F32_ATTENTION_EVIDENCE_2026-10-10.md`, approved spec/plan, `CHANGELOG.md`, workspace `Cargo.toml`, generated Cargo.lock.

- [x] Review diff for new unsafe code, default dispatch, unsupported precision behavior, out-of-bounds handling, test independence and measurement confounders.
- [x] Record completed versus pending gates; keep candidate default-off irrespective of pilot outcome. No actual15K agent improvement is inferred from the small workload.
- [x] Increment0.2.82 to0.2.83 only after the implemented slice passes its required verification. Changelog includes local source paths and evidence inspirations, without claiming new internet research.
- [ ] Stage only explicit task files, commit and non-force push `feat/runtime-parity-profiling`; inspect final status to confirm .gitignore remains unstaged.

## Execution choice

The user approved continuing the design on the current feature branch. Execute
inline with checkpoints and preserve the existing checkout/cache to avoid
unnecessary compilation. A separate worktree or subagent review is optional,
not permission to change models or increase resource limits.
