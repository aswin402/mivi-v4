# Q4 Cached Sums Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure a faithful, opt-in Q4 FFN cached-sum candidate without changing default inference.

**Architecture:** A checked scratch-taking Q4 API reuses activation sums across weight rows. Feature-gated RunState scratch and a default-off switch connect this API to FFN projections. The existing kernels remain the paired baseline.

**Tech Stack:** Rust, existing Rayon and AVX2/FMA dispatch, GGUF fixtures.

## Global Constraints

- Cargo one job; inference two threads; offline scoped tests only.
- Preserve F32 arithmetic, route-specific reduction order, and portable fallback.
- No model-name dispatch, new dependencies, mutable globals, or decode allocation.
- Default kernel dispatch stays unchanged; no production promotion from this pilot.
- Preserve the unrelated `.gitignore` edit.

### Task 1: Checked experimental kernel

Files: `crates/mivi-quant/Cargo.toml`, `src/q4_k_m.rs`, and new `src/q4_k_m/cached_sums.rs`.

Interface: `try_matvec_q4_k_m_cached(out: &mut [f32], weights: &[u8], x: &[f32], n: usize, d: usize, scratch: &mut [f32]) -> crate::types::Result<()>`. Scratch requires `d / 32` floats after Q4 block-alignment validation.

- [x] Add feature and a stub returning an error; write exact route-specific output tests and rejection-without-write tests.
- [x] Run `CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 cargo test -p mivi-quant --lib --offline --features q4-cached-sums-experiment cached_sums -- --test-threads=1`; observe missing behavior fail.
- [x] Implement validated scratch preparation and row kernels retaining baseline Q4 expressions. AVX2 sum preparation uses four eight-lane additions followed by the existing horizontal-sum helper. Scalar preparation folds 32 values in order.
- [x] Repeat the command; include odd/parallel row counts, multiple block widths, changed-input scratch reuse, zero dimensions, and output/scratch tails.

### Task 2: Opt-in FFN integration and fixed-work evidence

Files: `crates/mivi-core/Cargo.toml`, `src/arena.rs`; `crates/mivi-model/Cargo.toml`, `src/ffn.rs`, `src/model.rs`.

Interface: feature-gated RunState fields `q4_activation_sums: Box<[f32]>` and `q4_cached_sums_enabled: bool`, initially false. Scratch capacity uses the larger model activation dimension rounded to 32; scratch contents are transient, not prefix-cache state.

- [x] Add arena capacity/default/reset controls; observe missing behavior before implementation.
- [x] Wire the feature through quant/model; route Q4 FFN projections only when the switch is true. Apply LoRA after the candidate just as after baseline.
- [x] Add ignored `q4_cached_sums_decode_parity_and_measurement`, sharing identical public synthetic prompt and continuation work between candidate and baseline. Compare bit patterns of all logits and recurrent/KV values, not sampled text alone.
- [x] Run only that release test with `MIVI_TEST_MODEL` set to the absolute local GGUF, `CARGO_BUILD_JOBS=1`, `RAYON_NUM_THREADS=2`, `--offline`, `--features q4-cached-sums-experiment`, and `-- --ignored --test-threads=1 --nocapture` under a 300-second timeout.
- [x] Record every measured pair, declare the unreported warmup timings, paired ratios, eligible FFN tensors, cache state, work count, and limits. Keep candidate opt-in regardless of pilot outcome.

### Task 3: Verification and publication

Files: `docs/Q4_CACHED_SUMS_EVIDENCE_2026-10-07.md`, `CHANGELOG.md`, root `Cargo.toml`, `Cargo.lock`.

- [x] Run targeted existing Q4 controls without the feature, format touched Rust files, and run `git diff --check`.
- [x] Document measured results and remaining end-to-end/RSS/model-coverage gates; mark completed steps here only after verification.
- [x] Bump workspace package version to `0.2.74` and update the fourteen local lockfile package versions; add changelog sources and scope.
- [ ] Commit only scoped files and push `feat/runtime-parity-profiling`; leave `.gitignore` untouched.
