# Faithful Projection Column-Panel Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the ineffective experimental token-width selector with an opt-in column-panel-width comparison, preserving the ordinary AVX2 production kernel and its paired 32-token microkernel.

**Architecture:** Add a typed `ProjectionColumnTile` selector and a feature-gated paired-row quantized API. The experimental kernel reuses production's four AVX2 vectors per row and ascending-column FMA order, changing only the const-generic column panel width. Rename the private measurement field and bump its manifest/child/report schema to 3, then rerun the existing bounded paired pilot against an implementation commit. Production APIs and dispatch remain unchanged.

**Tech Stack:** Rust workspace, AVX2/FMA and Rayon already used by Mivi, serde JSON measurement example, Python standard-library supervisor. No new dependencies.

**Approved design:** [Faithful projection column-panel experiment redesign](../specs/2026-10-07-faithful-column-panel-experiment-design.md).

## Global Constraints

- Cargo jobs 1; Rust test threads 1; inference/Rayon threads 2. One build or measurement workload at a time. No full-workspace check/build/test.
- Keep production dispatch unchanged; `quantized_matmul_rows` and ordinary SIMD entry points must continue through their existing implementation.
- The experimental selectors are explicit benchmark parameters, not model-, GGUF-, tensor-, or host-name rules.
- Every output lane must consume columns in ascending order with the same FMA operation; require exact output-bit equality against baseline.
- Use only the previously selected synthetic and real-weight cases; synthetic activations remain synthetic. No model download or full-model inference.
- Measurement uses one child at a time, 180 seconds per child, sampled RSS at most 2 GiB, 64 MiB artifact cap, and 900 seconds per session. Keep private paths, model/tensor identifiers, hashes, activations, and raw logs outside Git.
- Preserve the user's unstaged `.gitignore` change. Stage only task-owned files.
- Do not promote an experimental path or claim speedup/agent quality from operator measurements. Promotion requires a later explicit decision and the P1-A end-to-end gate.
- At release time, re-read workspace version; expected `0.2.70` → `0.2.71` if unchanged. Add a source-attributed changelog entry, run scoped validation, and push non-force to the existing branch. Planning and implementation commits do not bump the release version.

---

## File structure and ownership

| File | Responsibility |
| --- | --- |
| `crates/mivi-core/src/simd/mod.rs` | Replace `ProjectionTokenTile` with `ProjectionColumnTile`; expose explicit paired-row column-panel entry point; keep ordinary wrappers on the existing AVX2 path; test exact parity. |
| `crates/mivi-core/src/simd/avx2.rs` | Share the existing paired-row kernel through const-generic panel widths 32/64/128; preserve the default 128-column path and the 32-token unroll; remove the superseded generic token-tile kernel. |
| `crates/mivi-quant/src/lib.rs` | Rename the feature-gated checked API and internal selector plumbing; leave the odd final row on the ordinary single-row API. |
| `crates/mivi-quant/src/projection_diagnostics.rs` | Rename the profiled feature-gated API and preserve selector-aware profile routing. |
| `crates/mivi-quant/tests/projection_locality.rs` | Exact bit/validation/profile tests for all supported formats, selectors, batch tails, and odd row counts. |
| `crates/mivi-model/examples/projection_measure/case.rs` | Enforce child input schema 3, strict `column_tile` field, and supported values. |
| `crates/mivi-model/examples/projection_measure.rs` | Route null to baseline and explicit panel widths to profiled/unprofiled experimental APIs; emit schema-3 results. |
| `scripts/runtime_compare/projection_measure.py` | Enforce manifest/result/report schema 3, validate and compare `column_tile`, and keep exact parity/pair-order behavior. Profile-call schema remains 1. |
| `scripts/runtime_compare/test_projection_measure.py` | Test schema migration, legacy-field rejection, selectors, comparison matching, parity, failures, and order. |
| `docs/PROJECTION_LOCALITY_EVIDENCE_2026-10-06.md` | Replace the previous corrected token-width result with a redacted fresh column-panel pilot; preserve its historical conclusions in the document. |
| `docs/superpowers/plans/2026-10-07-faithful-column-panel-experiment.md` | Track implementation, measurement, release validation, and publication. |
| `docs/superpowers/plans/2026-10-02-native-cpu-agent-improvements.md` | Update only the locality experiment evidence; leave broad P1-A gates unchecked. |
| `CHANGELOG.md`, root `Cargo.toml`, `Cargo.lock` | One patch release after the implementation and corrected pilot are accepted. |

## Task 1: Add the typed column-panel SIMD variant

**Files:** `crates/mivi-core/src/simd/mod.rs`, `crates/mivi-core/src/simd/avx2.rs`.

**Interface:**

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionColumnTile {
    Columns32,
    Columns64,
    Columns128,
}

pub fn matmul_accumulate_transposed_pair_with_column_tile_simd(
    out0: &mut [f32],
    out1: &mut [f32],
    weights0: &[f32],
    weights1: &[f32],
    transposed_inputs: &[f32],
    batch: usize,
    cols: usize,
    tile: ProjectionColumnTile,
);
```

- [x] Add a core unit test that calls the missing typed API for all three variants; compare output `to_bits()` against `matmul_accumulate_transposed_pair_simd` for batches `32, 33, 63, 64, 65, 127, 128, 129`, column counts `1, 17, 257`, nonzero initial accumulators, odd vector tails, and sentinel elements after `batch`.

```rust
for tile in [Columns32, Columns64, Columns128] {
    let mut expected0 = vec![0.25_f32; batch + 1];
    let mut expected1 = vec![-0.5_f32; batch + 1];
    let mut actual0 = expected0.clone();
    let mut actual1 = expected1.clone();
    matmul_accumulate_transposed_pair_simd(
        &mut expected0, &mut expected1, &weights0, &weights1, &inputs, batch, cols,
    );
    matmul_accumulate_transposed_pair_with_column_tile_simd(
        &mut actual0, &mut actual1, &weights0, &weights1, &inputs, batch, cols, tile,
    );
    for (expected, actual) in [(&expected0, &actual0), (&expected1, &actual1)] {
        for index in 0..batch {
            assert_eq!(expected[index].to_bits(), actual[index].to_bits());
        }
        assert_eq!(expected[batch].to_bits(), actual[batch].to_bits());
    }
}
```

- [x] Run `CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 cargo test -p mivi-core projection_column_tile --locked --offline -- --test-threads=1`; confirm the expected unresolved-type/function failure before implementation.
- [x] Refactor the existing paired-row AVX2 loop into a const-generic panel helper without changing its four vector accumulators per row, shared input-vector loads, column-order FMA operations, or 32-token advancement. Route ordinary `matmul_accumulate_transposed_pair_simd` through `<128>`; route the explicit selector through `<32>`, `<64>`, or `<128>`.
- [x] Give the AVX2 helper this exact shape and move the existing paired loop body into it, replacing only `COLUMN_TILE` with the const parameter:

```rust
unsafe fn pair_panel_avx2<const COLUMN_TILE: usize>(
    out0: &mut [f32],
    out1: &mut [f32],
    weights0: &[f32],
    weights1: &[f32],
    transposed_inputs: &[f32],
    batch: usize,
    cols: usize,
)
```

  Preserve `#[target_feature(enable = "avx2", enable = "fma")]` and the existing safety preconditions on the helper.

- [x] Use this fixed-shape dispatch so widths are compile-time constants in the inner loop:

```rust
match tile {
    ProjectionColumnTile::Columns32 => unsafe {
        pair_panel_avx2::<32>(out0, out1, weights0, weights1, inputs, batch, cols)
    },
    ProjectionColumnTile::Columns64 => unsafe {
        pair_panel_avx2::<64>(out0, out1, weights0, weights1, inputs, batch, cols)
    },
    ProjectionColumnTile::Columns128 => unsafe {
        pair_panel_avx2::<128>(out0, out1, weights0, weights1, inputs, batch, cols)
    },
}
```

- [x] Remove the superseded `ProjectionTokenTile` and token-tiled single/pair SIMD entry points after all references move to the column-panel pair API.
- [x] Keep the odd final row outside this paired helper; it remains routed through the ordinary single-row function by `mivi-quant`.
- [x] Run the focused core test and the existing `transposed_batch_accumulation` filter. Expect all parity tests to pass; no production dispatch changes.
- [x] Commit only the core SIMD changes and tests.

## Task 2: Rename the checked quantized and profiled APIs

**Files:** `crates/mivi-quant/src/lib.rs`, `crates/mivi-quant/src/projection_diagnostics.rs`, `crates/mivi-quant/tests/projection_locality.rs`.

**Interfaces (both remain behind `projection-locality-experiment`):**

```rust
pub fn quantized_matmul_rows_with_column_tile(
    out: &mut [f32],
    ggml_type: GgmlType,
    weights: &[u8],
    inputs: &[f32],
    batch: usize,
    rows: usize,
    cols: usize,
    tile: mivi_core::simd::ProjectionColumnTile,
) -> Result<()>;

pub fn quantized_matmul_rows_profiled_with_column_tile(
    out: &mut [f32],
    ggml_type: GgmlType,
    weights: &[u8],
    inputs: &[f32],
    batch: usize,
    rows: usize,
    cols: usize,
    tile: mivi_core::simd::ProjectionColumnTile,
) -> Result<ProjectionProfile>;
```

- [x] Rename integration test imports and add/adjust assertions for all six supported formats, selectors 32/64/128, batch 32/33/64/65, odd row count, exact bits, checked invalid buffers, and profiled branch/worker output.
- [x] Run `CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 cargo test -p mivi-quant --features 'projection-locality-experiment projection-diagnostics' --test projection_locality --locked --offline -- --test-threads=1`; confirm the new names fail to resolve before implementation.
- [x] Rename `token_tile` plumbing to `column_tile`; retain ordinary validation, decoding, transpose, row grouping, and output layout. For paired rows use the core explicit panel API. For an odd final row, call the unchanged ordinary single-row function.
- [x] Rename the profiled API and route explicit profile cases through the same typed panel selector; keep null-selector profiles on the existing baseline API. Do not change profile-call schema 1.
- [x] Run the locality integration test with both features, then `CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 cargo test -p mivi-quant matmul --locked --offline -- --test-threads=1`; expect exact parity and checked-error tests to pass.
- [x] Commit only the quant API, diagnostics, and integration-test changes.

## Task 3: Migrate the private measurement protocol to schema 3

**Files:** `crates/mivi-model/examples/projection_measure/case.rs`, `crates/mivi-model/examples/projection_measure.rs`, `scripts/runtime_compare/projection_measure.py`, `scripts/runtime_compare/test_projection_measure.py`.

- [x] Add Python tests before changing the driver: a schema-3 manifest with `column_tile` accepts null/32/64/128; schema 2 and legacy `token_tile` are rejected; unknown tile 16, mismatched groups/settings, unequal full output bits, failed children, and order alternation retain existing behavior.
- [x] Run `python3 -m unittest scripts.runtime_compare.test_projection_measure`; verify the new migration tests fail for schema/field mismatch, not because of unrelated test setup.
- [x] Change `CASE_FIELDS`/`RESULT_FIELDS` and normalized case/result/comparison/sample records from `token_tile` to `column_tile`. Require manifest, child input/output, and session report schema 3; leave profile call/worker schema at 1. Reject legacy fields strictly and do not reinterpret old manifests.
- [ ] Change Rust `CaseInput` to schema 3 and `column_tile: Option<u32>`, validate only `32, 64, 128`, invoke the renamed profiled/unprofiled API, and emit schema-3 results with `column_tile`.
- [x] Ensure selector comparisons can exercise distinct panels: require batch ≥32, at least two output rows, and more than 128 columns; validate GGUF dimensions after descriptor-only preflight. Have the child reject unavailable AVX2/FMA routes, report its kernel route, and have the driver validate that route.
- [x] Update all example fixtures and tests to schema 3; add explicit assertions that old `token_tile` and schema-2 inputs fail and each explicit selector reaches the matching route.
- [x] Run the Python driver tests, `python3 -m unittest test_supervisor_limits` from `scripts/runtime_compare`, and `CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 cargo test -p mivi-model --example projection_measure --features projection-locality-experiment --locked --offline -- --test-threads=1`, sequentially. Expect all existing and migration tests to pass.
- [x] Commit only the example and Python measurement protocol changes.

## Task 4: Commit the implementation and run a fresh bounded pilot

**Files:** private temporary directory only until the final redacted evidence update.

- [x] Review the staged implementation to verify normal production calls use the original ordinary kernels; ensure the explicit 128-column pair variant is generated from the same const-generic loop and is the reference control.
- [x] After the scoped commits from Tasks 1–3, record the full `HEAD` SHA containing the complete implementation; build and manifest revision must identify that committed code.
- [x] Build only `projection_measure` in release mode with `projection-locality-experiment`, Cargo jobs 1 and Rayon threads 2. No other compile or benchmark may overlap.
- [x] Create a fresh private session directory without deleting or overwriting old artifacts:

```sh
umask 077
PROJECTION_SESSION_DIR="$(mktemp -d /tmp/mivi-column-panel-XXXXXX)"
chmod 700 "$PROJECTION_SESSION_DIR"
```

  Keep its model/tensor/binary identities private; retained files must be mode 0600 and directories mode 0700.
- [x] Create a schema-3 manifest from the retained private workload configuration, changing only the schema and selector field names; use a new binary path identity and the implementation commit SHA. Include exactly the accepted top-level fields `schema`, `binary`, `revision`, `repetitions`, `wall_seconds`, `session_seconds`, `rss_bytes`, `artifact_bytes`, `buffer_limit_bytes`, `model_limit_bytes`, and `cases`. Each case has exactly `name`, `comparison_group`, `column_tile`, `batch`, `source`, `warmup_calls`, and `measured_calls`.
- [x] Select the same synthetic Q8_0 batch-64 control and descriptor-verified Q6_K batch-64 real-weight case. Each group has a null baseline and panels 32/64/128, three repetitions, one warmup, two threads, and profiled/unprofiled calls. Keep case/source identity private.
- [x] Run `python3 scripts/runtime_compare/projection_measure.py --manifest "$PROJECTION_SESSION_DIR/manifest.json" --output-dir "$PROJECTION_SESSION_DIR/run" --validate-only`; verify the selector×mode×repetition attempt count and predicted artifact bytes are within 64 MiB before launching children.
- [x] Run the same command without `--validate-only`. Respect one child at a time, 180 seconds per child, 2 GiB sampled RSS, and 900 seconds total. Do not flush system caches or start concurrent inference/build work.
- [x] Inspect every attempt and pair. Accept only exact full-vector output parity with identical non-selector settings and matching selector identity. Retain failed/rejected artifacts privately; do not manually salvage samples.
- [ ] If exact parity fails, stop: do not publish variant performance as valid; diagnose the code first. If the 128-panel paired timing falls outside the baseline's paired variation, inspect generated control flow and report the discrepancy before accepting the rest of the results.
- [x] Update `docs/PROJECTION_LOCALITY_EVIDENCE_2026-10-06.md` from only the fresh schema-3 session. Report all selector results, paired unprofiled timings, profile diagnostics, host caveats, exact shapes, parity, counts, artifact/permission audit, and no-promotion decision. Retain the withdrawn token-width measurements as historical context, not evidence for this comparison.

## Task 5: Release, scoped verification, review, and push

**Files:** `CHANGELOG.md`, root `Cargo.toml`, `Cargo.lock`, evidence report, this plan, and the locality line in `docs/superpowers/plans/2026-10-02-native-cpu-agent-improvements.md`.

- [x] Add a changelog entry with the current date, one patch increment, the experiment and result, explicit production/default status, limitations, and ideas/inspirations/source links to Colibri's benchmark methodology and the pinned GGML CPU implementation. Note the experimental token API/manifest schema correction; no code was copied.
- [x] Re-read root metadata and increment the patch version exactly once (expected 0.2.70 → 0.2.71); update only local workspace package entries in Cargo.lock with `CARGO_BUILD_JOBS=1 cargo update --workspace --offline`. Inspect the lockfile diff to ensure dependencies did not change.
- [x] Update the master roadmap only with the corrected experiment result. Leave reference-matrix, scratch-reuse, model-level parity, end-to-end evaluation, and optimization-promotion gates unchecked.
- [x] Run focused validations sequentially: core SIMD panel/parity filters; quant locality with both required features and matmul filter; Python driver and supervisor tests; model measurement-example tests; opt-in release example build; `CARGO_BUILD_JOBS=1 cargo metadata --no-deps --format-version 1 --locked --offline`. Cargo jobs 1, Rust tests one thread, Rayon two threads. No full-workspace check/build/test.
- [x] Run `git diff --check`; inspect the staged diff and verify `.gitignore` remains unstaged. Obtain an independent release review; correct all Important/Critical findings and rerun affected tests.
- [ ] Commit only task-owned implementation, report, plan, changelog, roadmap, and version files. Push the existing branch non-force; verify pushed remote ref equals local HEAD.
- [ ] Final handoff states whether panel widths improved or regressed, exact validation results, patch version, commit/push hash, and remaining P1-A gates. Do not claim model quality or end-to-end speedup from operator measurements.

## Plan self-review

- Spec coverage maps to Tasks 1–5: typed panel API and unchanged baseline; checked/profiled routing and odd-row baseline behavior; strict schema-3 migration; bounded fresh pilot and exact parity; source-attributed patch release and push.
- The panel-128 control is explicit and all profile/result/report schema versions are distinguished; profile schema remains 1.
- The old public API is explicitly experimental and feature-gated; the plan removes it rather than silently mapping the old token-width meaning to a new column-width meaning.
- No model-specific dispatch, full-workspace command, raw artifact publication, or placeholder remains.
