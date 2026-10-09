# Q4 Projection Stage Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for the established inline workflow. GPT-6 Luna/high performs read-only review; main runs Cargo serially. Track steps with checkboxes.

**Goal:** Attribute Q4 batch-32 replay costs while quantifying instrumentation disturbance.

**Architecture:** Test-only stage replay mirrors existing group order/partitions and calls unchanged decode/SIMD helpers. Complete ordinary-reference bit gates precede accepted measurements. Existing capture preparation is shared privately without changing production code.

**Tech Stack:** Rust, Rayon, existing quant scratch and memory-only GGUF/model fixtures.

## Global Constraints

Approved [design](../specs/2026-10-10-q4-stage-replay-design.md).
Cargo jobs1, Rayon threads2, serial harness, offline scoped commands only.
No production kernels/defaults/new unsafe/dependencies/model-name rules.
Preserve user .gitignore; no full workspace check/build/test.
Context65; effective IDs32/64; prompt <=64KiB UTF-8; eligible full Q4_K/Q6_K matrices.
Stage replay requires two positive worker partitions, each a multiple of four rows.
No prompt IDs/weight/activation/output contents logged; retain every timing/negative.
Publish sourced 0.2.79 only when the whole diagnostic is verified.

## Task 1: Checked stage replay and accounting

Files:
- Create `crates/mivi-quant/src/batch_scratch/four_row/faithful_diagnostics/captured/stages.rs`.
- Modify `crates/mivi-quant/src/batch_scratch/four_row/faithful_diagnostics/captured.rs`: add `mod stages;`.

Interfaces defined in the new module:
```rust
struct Shape { kind: GgmlType, batch: usize, rows: usize, cols: usize,
               chunk_rows: usize, row_bytes: usize }
impl Shape {
    fn new(kind: GgmlType, batch: usize, rows: usize, cols: usize) -> EvalResult<Self>;
}
#[derive(Debug, Default, Clone, Copy)]
struct Work { rows: usize, groups: usize, decode_calls: usize,
              helper_calls: usize, decode_ns: u64, zero_ns: u64, accumulate_ns: u64 }
#[derive(Debug, Default)]
struct Profile { validation_ns: u64, transpose_ns: u64, rows_wall_ns: u64,
                 layout_ns: u64, call_wall_ns: u64, residual_ns: u64,
                 workers: [Work; 2] }
fn residual(total: u64, stages: &[u64]) -> EvalResult<u64>;
fn replay(out: &mut [f32], weights: &[u8], inputs: &[f32],
          shape: &Shape, scratch: &mut BatchProjectionScratch,
          group: usize) -> EvalResult<Profile>;
```

- [x] Register module and write compiling stubs returning errors. Add valid-case RED controls:
```rust
#[test]
fn stage_replay_shape_and_wall_accounting() {
    assert!(Shape::new(GgmlType::Q4_K, 32, 256, 256).is_ok());
    assert_eq!(residual(100, &[10, 20, 30]).unwrap(), 40);
}
```
- [x] Run `CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 cargo test -p mivi-quant --lib --offline --features batch-four-row-experiment stage_replay -- --test-threads=1`.
  Expect assertion failure, not compiler error.
- [x] Implement shape validation (Q4/Q6 only; batch32/64; aligned cols1..16384;
  rows at least parallel threshold and <=8192; ceil-half and final partition positive
  multiples4); checked row bytes/products. Implement checked duration conversion,
  addition and residual:
```rust
fn residual(total: u64, stages: &[u64]) -> EvalResult<u64> {
    let used = stages.iter().try_fold(0u64, |a, b| a.checked_add(*b))
        .ok_or("stage wall sum overflow")?;
    total.checked_sub(used).ok_or_else(|| "stage wall exceeds call".into())
}
```
- [x] Add replay valid-case RED fixture: ancestor test weights generate finite nonzero
  Q4/Q6 rows256/cols256; inputs generated finite; compare every ordinary-reference
  bit at both batches/group2+4 in an explicit two-thread pool. Stub replay must fail.
- [x] Implement replay validation before writes: exact external lengths; group2/4;
  current pool2; scratch shape/worker/decoded capacity. Prevalidate all active buffers.
  Record validation and transpose wall, parallel region wall, output layout wall,
  call wall and checked residual. Reuse existing scratch fields without resizing.
- [x] Worker body loops groups in ascending rows. For group2 decode row0, time output
  zero, decode row1; for group4 decode all rows then zero. Time existing helper call;
  accumulate checked per-worker ns/counts. Parallel chunks and mutable worker/profile
  records stay disjoint. Return two reports, never sum worker time into wall.
- [x] Expand tests: unsupported/empty/misaligned/overbudget/nonmultiple partitions;
  short output/input/weights, group invalid, undersized decoded capacity, pool1;
  failures preserve output and all scratch bits; poisoned reuse preserves allocation
  identities; exact worker rows/groups/decode/helper counts; zero wall stages accepted,
  residual overshoot/overflow rejected. Rerun scoped debug/release controls.
- [x] Luna/high read-only review the checked replay before live evaluation.

## Task 2: Shared capture and four-route measurement

Files:
- Modify `faithful_diagnostics/captured.rs` above; extend `captured/stages.rs`.
Interfaces:
```rust
fn prepare_capture() -> EvalResult<(Model, Vec<u32>)>;
fn gated_captures(model: &mut Model, ids: &[u32], batch: usize)
    -> EvalResult<Vec<Capture>>;
```

- [x] Extract existing model/prompt/BOS preparation into `prepare_capture`; extract
  production prefill/walker/final-logit gate into `gated_captures`. Existing ignored
  `four_row_captured_activation` calls both and keeps old replay order/three-call totals.
- [x] Add measurement-order control checking every position count=2 and each
  pairwise relative order balanced=4 among the eight design permutations.
- [x] New ignored `four_row_stage_replay_measurement` requires release/Rayon2.
  For batch32/64, get gated captures; call Shape::new on complete matrices. Create
  S/F/PS/PF scratch and outputs plus ordinary reference outside timers. Gate both
  untimed stage replay outputs exactly before accepted timing.
- [x] One warmup quadruple + eight measured design permutations, one call/member.
  Route0 existing scratch; route1 existing four-row; routes2/3 replay group2/4.
  Use black_box inputs/out; compare complete finite output after timer.
  Print original ns per member plus stage/per-worker reports for PS/PF. Print
  only metadata and counters; check accounting/counts before accepting a profile.
- [x] Run controls and bounded live command:
```sh
CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 \
MIVI_TEST_MODEL=/absolute/path/to/existing.gguf timeout 240s \
cargo test -p mivi-quant --lib --release --offline \
  --features batch-four-row-experiment four_row_stage_replay_measurement \
  -- --ignored --test-threads=1 --nocapture
```
  Expect 4 cases, 144 full timed projections, complete bit gates; retain failures
  without relaxing equality. Rerun existing capture diagnostic after helper extraction.
- [x] Summarize within-quadruple F/S, PS/S and PF/F medians/ranges; profile wall stages
  and separate worker work. Interpret profiler disturbance explicitly.

## Task 3: Evidence, final verification and publication

Files: new `docs/Q4_STAGE_REPLAY_EVIDENCE_2026-10-10.md`;
`CHANGELOG.md`, `Cargo.toml`, `Cargo.lock`, this plan and master CPU plan.

- [x] Retain complete raw warmup/measured records, host/model identity, timing/cache
  contracts, origin/output gates and profiler limits. No universal performance or
  hardware-root-cause claim; default remains off.
- [x] Luna/high source/evidence follow-up; address important issues before publishing.
- [x] Bump root workspace0.2.78->0.2.79, let targeted Cargo update local lock versions;
  add sourced changelog with Mivi/Colibri inspiration links and measured negatives.
- [x] Final scoped release feature-enabled batch filter, feature-off batch filter,
  debug stage/capture controls; nonzero counts, Cargo1/Rayon2 serial. Check changed
  Rust files with rustfmt and tracked/new staged diffs with git diff --check.
- [ ] Stage only task files, commit/push existing feature branch without force;
  preserve .gitignore. Record publication after confirming push/sync.

Final v0.2.79 verification: feature-enabled release batch filter26 passed/4 ignored;
feature-off release batch filter6 passed; debug diagnostic filter8 passed/3 ignored.
Live stage diagnostic7.30s and original capture regression9.46s passed before the
publication bump. All measured math/timing unchanged by later fallible buffer
construction and extra rejection tests. Luna/high source/evidence reviewed and
independently recomputed summaries; no Critical/Important findings. No Clippy or
full-workspace success claim. Changed-source rustfmt and diff whitespace checked.
