# Faithful Projection Locality Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bounded, benchmark-only comparison of AVX2 transposed projection token-block widths while preserving each output's FMA accumulation order and leaving default dispatch unchanged.

**Architecture:** Keep the existing SIMD and quantized matmul entry points as the baseline. Add a typed, explicit experimental token-block selector in `mivi-core`, thread it through a feature-gated quantized projection entry point, and extend the existing private projection-measure case/driver to run matched baseline-versus-variant pairs. First prove bitwise parity, then measure the same selected projection cases; no runtime auto-tuning or production promotion is included.

**Tech Stack:** Rust, existing AVX2/FMA and Rayon paths, existing projection-diagnostics example, Python standard library process supervisor. No new dependencies.

## Global Constraints

- Cargo jobs 1; Rust test threads 1; inference/Rayon threads 2. One build or measurement workload at a time. No full-workspace check/build/test.
- Preserve the public checked `quantized_matmul_rows` API, normal batch thresholds, row partitioning, default tile traversal, activation precision, and default dispatch.
- Experimental widths are explicit benchmark parameters, not model-, GGUF-, tensor-, or host-name rules.
- Every output lane must consume columns in the same order with the same FMA operation; require exact output-bit equality against baseline.
- Use only the previously selected synthetic and real-weight cases; synthetic activations remain synthetic. No model download or full-model inference.
- Measurement uses the existing private I/O and child supervisor limits: one child at a time, 180 seconds per child, sampled RSS at most 2 GiB, 64 MiB artifact cap, and 900 seconds per session. Keep private paths, model/tensor identifiers, hashes, activations, and raw logs outside Git.
- Preserve failed attempts and the user's unstaged `.gitignore` change. Stage only task-owned files.
- Do not promote the experimental path or claim speedup/agent quality from operator measurements. Promotion requires a later explicit decision and the P1-A end-to-end gate.
- After the experiment is accepted, bump the current patch version once, document inspirations/sources and limitations in `CHANGELOG.md`, run scoped validation, and push non-force to the existing branch. Planning commits do not change the release version.

---

## File structure and ownership

| File | Responsibility |
| --- | --- |
| `crates/mivi-core/src/simd/mod.rs` | Define a typed token-block selector; preserve the existing wrapper and add explicit experimental single/pair accumulation entry points. Unit tests compare exact bits and cover tails. |
| `crates/mivi-core/src/simd/avx2.rs` | Implement the selectable output-token traversal widths while retaining same-column-order FMA updates per lane. Leave existing entry points/default behavior intact. |
| `crates/mivi-quant/Cargo.toml` | Declare an opt-in `projection-locality-experiment` feature, separate from normal builds. |
| `crates/mivi-quant/src/lib.rs` | Add a feature-gated checked experimental matmul entry point; dispatch only when explicitly invoked with a validated selector. Existing API remains the baseline. |
| `crates/mivi-quant/src/projection_diagnostics.rs` | Add a feature-gated profiled experimental entry point so profiled variant samples time the selector they declare. |
| `crates/mivi-quant/tests/projection_locality.rs` | Check exact output bits across baseline and every supported selector for supported formats, batches, odd rows, and errors, including the profiled selector path. |
| `crates/mivi-model/Cargo.toml` | Forward the opt-in quant feature through a model-level `projection-locality-experiment` feature while preserving the example's diagnostics requirement. |
| `crates/mivi-model/examples/projection_measure.rs` and `crates/mivi-model/examples/projection_measure/case.rs` | Accept an explicit experimental selector in a private measurement case, invoke baseline or experiment, and emit the chosen case identity without leaking local paths. |
| `scripts/runtime_compare/projection_measure.py` | Validate paired comparison cases, alternate baseline/variant order, require matching settings and outputs, and report every attempt and matched full-call timing. |
| `scripts/runtime_compare/test_projection_measure.py` | Regression tests for manifest validation, pairing, order, incomplete/failed pairs, output parity, and report statistics. |
| `docs/PROJECTION_LOCALITY_EVIDENCE_2026-10-06.md` | Redacted results, failed attempts, host and methodology caveats, and an evidence-based next decision. |
| `docs/superpowers/plans/2026-10-06-faithful-projection-locality.md` | Task tracking, final validation record, and release references. |
| `docs/superpowers/plans/2026-10-02-native-cpu-agent-improvements.md` | Mark only the locality measurement subtask complete; leave reference coverage, scratch reuse, blocked-path implementation, and promotion gates unchecked unless separately completed. |
| `CHANGELOG.md`, root `Cargo.toml`, `Cargo.lock` | Patch release and source-attributed changelog, only after accepted implementation and experiment. |

## Task 1: Add typed experimental AVX2 traversal without changing the default

**Files:** `crates/mivi-core/src/simd/mod.rs`, `crates/mivi-core/src/simd/avx2.rs`.

**Interfaces:**

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionTokenTile {
    Tokens32,
    Tokens64,
    Tokens128,
}

pub fn matmul_accumulate_transposed_with_tile_simd(
    out: &mut [f32],
    weights: &[f32],
    transposed_inputs: &[f32],
    batch: usize,
    cols: usize,
    tile: ProjectionTokenTile,
);
```

Add an equivalent pair-row entry point with `out0`, `out1`, `weights0`, and `weights1`. Keep `matmul_accumulate_transposed_simd` and `matmul_accumulate_transposed_pair_simd` as compatibility wrappers that select `Tokens64` and call the shared implementation; this preserves their observable arithmetic/traversal while avoiding duplicated FMA/tail logic. The experiment entry points accept only the enum, so arbitrary invalid widths cannot enter the kernel.

- [x] Add tests in `simd/mod.rs` for selector widths 32/64/128, batches 32/33/63/64/65/127/128/129, column counts 0/1/17/257, nonzero initial accumulators, sentinel outputs, and both one-row and paired-row calls. Compare `to_bits()` for each output with the existing baseline function.
- [x] Run the focused new test before implementation and verify it fails to compile because the experimental API does not exist.
- [x] Implement one shared AVX2 traversal selected by `ProjectionTokenTile`; have both existing compatibility wrappers select `Tokens64`. For each token lane, load its current output, visit all `cols` in ascending order using the existing `_mm256_fmadd_ps`, then store only its valid token range. Handle tails with the existing scalar/FMA behavior and preserve the non-AVX2 fallback semantics. Avoid a second copy of the accumulation/tail logic.
- [x] Run `CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 cargo test -p mivi-core transposed_batch_accumulation -- --test-threads=1`; 3 passed, 1 ignored benchmark, 0 failed.

## Task 2: Thread the explicit variant through checked quantized matmul

**Files:** `crates/mivi-quant/Cargo.toml`, `crates/mivi-quant/src/lib.rs`, new `crates/mivi-quant/tests/projection_locality.rs`.

**Interface:** Under `projection-locality-experiment`, add:

```rust
pub fn quantized_matmul_rows_with_token_tile(
    out: &mut [f32],
    ggml_type: GgmlType,
    weights: &[u8],
    inputs: &[f32],
    batch: usize,
    rows: usize,
    cols: usize,
    tile: mivi_core::simd::ProjectionTokenTile,
) -> Result<()>;
```

It must use the same argument validation, dequantization, transposition, output layout, row partitioning, and row-pair grouping as `quantized_matmul_rows`. Only the large-batch accumulation call changes to the explicitly requested core entry point. Batch-one and small-batch behavior use the existing path. The ordinary API never reads a clock, selector, environment variable, or feature state.

- [x] Add a focused local integration-test fixture builder (existing nonzero-weight helpers are private to other test modules) for deterministic valid nonzero F32, F16, BF16, Q8_0, Q4_K, and Q6_K rows; compare output bits between ordinary baseline and all three selectors.
- [x] Cover supported block-width columns, batches 32/33/64/65, odd row counts including an odd final row, short and long column counts, too-small input/weight/output buffers, misaligned quantized columns, and size-overflow validation where representable.
- [x] Run `CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 cargo test -p mivi-quant --features projection-locality-experiment --test projection_locality -- --test-threads=1` before adding the implementation and verify the expected missing-API failure.
- [x] Add the feature-gated API; return existing `QuantError` values for invalid shapes and never panic on caller-controlled lengths.
- [x] Run the focused integration test above (2 passed), then `CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 cargo test -p mivi-quant matmul -- --test-threads=1` (16 passed); no workspace-wide test.

## Task 3: Extend the private measurement case and paired supervisor

**Files:** `crates/mivi-model/Cargo.toml`, `crates/mivi-model/examples/projection_measure.rs`, `crates/mivi-model/examples/projection_measure/case.rs`, `crates/mivi-quant/src/projection_diagnostics.rs`, `crates/mivi-quant/tests/projection_locality.rs`, `scripts/runtime_compare/projection_measure.py`, `scripts/runtime_compare/test_projection_measure.py`.

**Case identity:** Each logical workload has a baseline member (`token_tile: null`) and explicit variant members (`token_tile: 32`, `64`, or `128`); members have unique safe names and a shared safe `comparison_group`. Reject unknown selectors and non-null selectors unless the example is built with `projection-locality-experiment`. Compare by logical group, source/tensor identity, format, rows, columns, batch, thread count, repetition, and profiled/unprofiled mode. Alternate baseline/variant order per repetition while retaining the driver's existing profiled/unprofiled pairing. A comparison is accepted only if all relevant children succeed, every output bit vector matches, and all non-selector settings match.

- [x] Add Python tests for selector validation, group membership/mismatches, failed/timed-out children, unequal output bits, and baseline/variant order alternation.
- [x] Add model feature `projection-locality-experiment = ["projection-diagnostics", "mivi-quant/projection-locality-experiment"]`; preserve strict fields, duplicate-key rejection, bounded JSON, private paths, child supervision, complete attempt retention, and preflight artifact budgeting across selector×mode×repetition.
- [x] Route unprofiled baseline and explicit selectors in the example using validated case fields, with generic/redacted errors.
- [x] Add `projection_diagnostics::quantized_matmul_rows_profiled_with_token_tile` and route profiled variants through it; baseline profile cases retain the existing API.
- [x] Add integration coverage for exact profiled output parity and profile branch/worker data for all selectors.
- [x] Validate finite result values and compare every `u32` output bit; report baseline/per-selector full-call timing, paired ratios, diagnostic stages, failures, and parity.
- [x] Run the quant locality test with both features enabled: 3 passed.
- [x] Run `python3 -m unittest scripts.runtime_compare.test_projection_measure` (41 passed) and focused supervisor tests (6 passed, from `scripts/runtime_compare`).
- [x] Build only the opt-in release example offline (succeeded without warnings); example unit tests: 15 passed.

## Task 4: Run the bounded paired locality pilot

**Files:** private temporary directory only; final public report `docs/PROJECTION_LOCALITY_EVIDENCE_2026-10-06.md`.

- [x] Rerun the bounded pilot after restoring the exact pre-experiment ordinary SIMD paths. The earlier 48-child session is withdrawn, retained privately, and excluded from evidence.
- [x] Rebuild the opt-in release example from the corrected source. Reverify the synthetic control and metadata-selected Q6_K batch-64 real-weight descriptor (2,048 × 8,192) read-only; use a fresh private session and retain all prior sessions.
- [x] Validate the fresh manifest and resource limits before launch: artifact cap 64 MiB, one child at a time, 180 seconds per child, 2 GiB sampled RSS, 900 seconds total.
- [x] Run three paired repetitions with alternating member order and one warmup per case; collect profiled diagnostics and unprofiled full-call timing at two threads, without flushing caches or overlapping workloads.
- [x] Inspect every attempt and verify settings and exact output bits. All 48 attempts completed and were reaped, with all profile/member pairs accepted and exact parity. Retain the excluded first-run artifacts privately; no artifacts removed.
- [x] Update the redacted [pilot evidence report](../../PROJECTION_LOCALITY_EVIDENCE_2026-10-06.md) from the corrected fresh session, including host caveats, counts, timings, parity and artifact audit.
- [x] Decision: no promotion. Every explicit selector was slower than the restored production path; Q6_K selector medians were 2.34–2.39× baseline. Keep the production default unchanged and diagnose the experimental path before any subsequent optimization evaluation.

## Task 5: Review, record roadmap/release, validate, and publish

**Files:** report, this plan, `docs/superpowers/plans/2026-10-02-native-cpu-agent-improvements.md`, `CHANGELOG.md`, root `Cargo.toml`, `Cargo.lock`.

- [x] Review changes against the approved spec; independent release review approved the restored ordinary SIMD paths, explicit experimental routing, corrected report, and no-promotion decision. No model-specific hardcoding or default promotion.
- [x] Update the roadmap only for work actually completed; keep default scratch reuse, broad reference matrix, model-level parity, and optimization promotion unchecked.
- [x] Increment the current patch version once, from verified `0.2.69` to `0.2.70`; update only local workspace lockfile package versions and add a source-attributed changelog entry covering the negative result and inspirations.
- [x] Run the scoped validations sequentially: core SIMD (3 passed, 1 ignored); quant locality with both required features (3 passed after restoring ordinary paths); matmul regressions (16 passed after restoration); measurement example tests (15 passed after restoration); Python driver (41 passed); supervisor (6 passed); corrected-source opt-in release example build succeeded; offline locked metadata succeeded. All Cargo used one job, tests one thread, Rayon two threads. No full-workspace check/build/test ran.
- [ ] Run `git diff --check`; inspect staged files and verify `.gitignore` remains unstaged. Commit task-owned changes in coherent commits, then push the existing branch non-force and verify the remote commit hash matches local HEAD.
- [ ] Final response reports actual experiment outcome, validation commands/counts, release version and commit/push hash, and remaining P1-A gates without overstating performance.

## Plan self-review

- Spec coverage: benchmark-only variants, same-order FMA parity, unchanged default API/dispatch, selected workload reuse, private bounded paired measurement, failed-attempt retention, no promotion, source-attributed reporting, and later end-to-end promotion gate are covered above.
- Placeholder scan: no TBD/TODO placeholders are present; all commands, paths, enum values, and expected validation targets are specified.
- Type consistency: the typed `ProjectionTokenTile` enum is passed unchanged from the quant feature-gated API to core SIMD; Python uses `null`/integer serialization only at the private JSON boundary and validates it before Rust dispatch.
