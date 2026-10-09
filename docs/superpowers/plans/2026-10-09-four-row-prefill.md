# Four-row Prefill Implementation Plan

> **For agentic workers:** Execute task-by-task with executing-plans and TDD;
> use a GPT-6 Luna/high read-only reviewer before publication.

**Goal:** Evaluate faithful four-row accumulation against ordinary and scratch-only batch controls.

**Architecture:** Isolated feature-gated core SIMD module, explicit quant candidate
and four-row constructor; no default model/server wiring or control-kernel refactor.

**Tech Stack:** Rust, existing AVX2/FMA runtime detection, Rayon, existing GGUF reader.

## Global constraints

Cargo one job (`CARGO_BUILD_JOBS=1`), inference at most two threads
(`RAYON_NUM_THREADS=2`); offline scoped tests, no full-workspace commands. Work
in place on existing feature branch; preserve unrelated `.gitignore`. Patch bump
and sourced changelog/publication after verification. No new dependency/model-name dispatch.

## Task 1: Core arithmetic

Files: `crates/mivi-core/src/simd/four_row.rs`, its `avx2.rs` child,
`crates/mivi-core/src/simd/mod.rs`, `crates/mivi-core/Cargo.toml`.

Interface: `accumulate(out: &mut [f32], weights: &[f32], inputs: &[f32], batch: usize, cols: usize)`;
four contiguous rows; bounds/overflow checked before mutation; panic contract like existing pair helper.

- [x] Add a stub and test comparing complete bits to two calls of the existing
      paired helper, including nonzero initial outputs and scalar/vector tails.
- [x] Run `cargo test -p mivi-core --lib --offline --features batch-four-row-experiment four_row -- --test-threads=1`; observed numerical assertion failure at batch=1, cols=1; exit 101. One Cargo job/two-thread environment.
- [x] Obtain explicit user approval for new bounded `unsafe` AVX2/FMA intrinsics,
      as required by Rust Core Specialist. User approved the bounded kernel on 2026-10-09.
- [x] Implement AVX2 sixteen-lane/four-row register block with column panels 128,
      eight-lane/scalar tails and checked fallback using existing pair helpers.
- [x] Repeat targeted core tests in debug and release; three passed in each.

## Task 2: Explicit quant projection

Files: `crates/mivi-quant/src/batch_scratch/four_row.rs`, parent registration and Cargo feature.

Interfaces: `BatchProjectionScratch::new_four_row(batch, rows, cols, workers) -> Result<Self>`;
`quantized_matmul_rows_four_row(out, kind, weights, inputs, batch, rows, cols, scratch) -> Result<()>`.

- [x] Stub the API and observe RED on a projection parity test against the ordinary API.
- [x] Allocate four decoded rows per worker with checked/fallible ownership;
      validate all buffers/capacities before writes, retain baseline partitions.
- [x] Compute full groups of four using core; partial groups/small batches reuse
      the existing scratch-only helper. Never grow storage during calls.
- [x] Run six-format/boundary/one-two-thread exact parity, reuse and invalid-buffer controls.

## Task 3: Operator evidence and publication

Files: test-only `crates/mivi-quant/src/batch_scratch/four_row/measurement.rs`,
`docs/FOUR_ROW_PREFILL_EVIDENCE_2026-10-10.md`, sourced changelog, root version/lockfile.

- [x] Add bounded ignored three-control fixed-work diagnostic; generate finite
      activations, use full matrices, check bits/finiteness outside timers.
- [x] Run release diagnostic with local GGUF and synthetic Q8_0; one warmup and
      six measured permutations with three calls/member; retain all raw times.
- [x] Review with Luna/high while recording performance/evidence limits locally.
      Arendt approved static arithmetic/API and subsequent diagnostic/evidence review;
      no findings, no reviewer edits or Cargo commands.
- [x] Run scoped enabled/disabled regression filters, formatting and diff checks.
- [x] Bump 0.2.76 to 0.2.77; document measured positives/negatives and sources.
- [x] Commit and push scoped files; implementation `c7cc24d` was pushed to
      `origin/feat/runtime-parity-profiling`. Unrelated `.gitignore` edits excluded.

Default promotion/model integration remains a subsequent decision requiring model
state/logit parity, agent-sized timing, RSS and short-case regression gates.

## Verification log

- Core RED: no-op stub failed numerical parity. Quant RED: stub constructor
  failed valid-shape parity. Measurement RED: shape stub rejected valid matrices.
- Before publication bump: core debug/release controls **3 passed** each;
  quant four-row filter **6 passed, 1 intentionally ignored**; separate live
  diagnostic **1 passed**, 16.69s, all 105 triples retained in evidence.
- v0.2.77 release quant batch filter, `--features batch-four-row-experiment`:
  **18 passed, 1 intentionally ignored**.
- v0.2.77 release quant batch filter, `--features batch-scratch-experiment`:
  **12 passed**, four-row candidate disabled.
- v0.2.77 release quant batch filter, no experimental features: **6 passed**.
- v0.2.77 core release four-row filter: **3 passed**.
- v0.2.77 quant debug four-row filter: **6 passed, 1 intentionally ignored**.
- Scoped rustfmt check and `git diff --check` passed. Root manifest and all 14
  local lockfile package versions are 0.2.77; registry dependencies unchanged.
- Scoped Clippy: `cargo clippy -p mivi-core -p mivi-quant --lib --offline --features
  batch-four-row-experiment -- -D warnings -A clippy::chunks_exact_to_as_chunks`
  **passed**. Strict first attempts flagged the established eight-argument pair
  API (now narrowly annotated) and two existing `chunks_exact` style lints in
  ordinary quant control code. That specific style lint is allowed for this run;
  measured control math is not rewritten. Not an unqualified all-workspace lint claim.
- All Cargo commands use `CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2`; targeted scope only.
