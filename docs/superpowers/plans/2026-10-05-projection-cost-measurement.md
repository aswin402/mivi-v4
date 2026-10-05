# Projection Cost Measurement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Identify allocation, layout conversion, decoding and computation costs in the existing projection kernel without promoting an optimization.

**Architecture:** An opt-in, monomorphized kernel diagnostic path shares the current arithmetic and reports serial wall stages separately from summed worker work. A diagnostic model example loads explicitly selected GGUF tensors or synthetic weights, runs matched calls, and is driven by the existing Python process supervisor. A bounded pilot produces a redacted decision, not a production speedup claim.

**Tech Stack:** Rust, existing Rayon/serde/serde_json/GGUF APIs, Python standard library and `scripts/runtime_compare` supervision/private I/O. No new third-party dependencies.

**Approved spec:** [projection cost measurement](../specs/2026-10-05-projection-cost-measurement-design.md).

Implementation status: Tasks 1–3 are complete and independently reviewed through
`281222b`. Live pilots exposed and fixed nested-timer and transpose-availability
validation bugs. Task 4 has 15 accepted pairs across five cases. Final package
review accepted the measurement-only slice at `a75bb09` with no findings;
v0.2.69 metadata is prepared, and publication remains pending.
The original failed pilot is retained and counted toward the artifact budget.

## Global Constraints

- Cargo jobs 1; Rust test threads 1; inference/Rayon threads 2. One build or measurement workload at a time. No full-workspace check/build/test.
- Supervisor pilot limits: 180 seconds per child, 2 GiB sampled child-tree RSS, 64 MiB aggregate artifacts, 15 minutes per explicitly selected pilot session.
- Private experiment directories/files use 0700/0600. Raw weights, activations, IDs, text, local paths and logs stay outside Git.
- Preserve current arithmetic, batch thresholds, row partitioning and checked API. No activation quantization, mutable global scratch, model-name dispatch, default tile changes or provider changes.
- Default calls must not read clocks or collect measurements. Parallel work durations are not additive elapsed wall time.
- Synthetic activations on real tensor shapes are not real-model captured activations. No model download or full-model activation collection.
- Retain unsuccessful experiments. Do not infer speedups or agent quality from operator timings.
- Preserve the user's unstaged `.gitignore` change. Stage explicit task-owned files only.
- If delegating, request `gpt-6-luna` with high reasoning; do not substitute a model silently or permit nested delegation. Serialize builds/measurements across workers.
- Planning commits do not increment a release version. Once this measurement slice passes acceptance, increment the then-current patch version once, attribute changelog sources, and push non-force to the existing branch.

## File ownership and order

| Task | Files | Responsibility |
| --- | --- | --- |
| 1 | `crates/mivi-quant/Cargo.toml`, `crates/mivi-quant/src/lib.rs`, new `crates/mivi-quant/src/projection_diagnostics.rs`, new `crates/mivi-quant/tests/projection_diagnostics.rs` | Shared arithmetic and opt-in stage measurements |
| 2 | `crates/mivi-model/Cargo.toml`, new `crates/mivi-model/examples/projection_measure.rs`, new `crates/mivi-model/examples/projection_measure/case.rs`, existing `crates/mivi-model/examples/runtime_replay/io.rs` | One bounded operator case, safe private JSON and GGUF selection |
| 3 | new `scripts/runtime_compare/projection_measure.py`, new `scripts/runtime_compare/test_projection_measure.py` | Manifest, paired order, process supervision and reports |
| 4 | new `docs/PROJECTION_COST_EVIDENCE_2026-10-05.md`, this plan, master roadmap, `CHANGELOG.md`, `Cargo.toml`, `Cargo.lock` | Bounded measurements, decision, review and release |

Tasks consume the preceding interfaces; no model forward-pass wiring or scratch API is introduced here.

## Task 1 — Shared kernel diagnostics and numerical controls

**Consumes:** `quantized_matmul_rows`, `quantized_matvec`, `validate_matmul_args`, `dequantize_slice`, current batch/parallel thresholds in `crates/mivi-quant/src/lib.rs` and `types.rs`.

**Produces:** feature `projection-diagnostics`, module `projection_diagnostics`, and this feature-only API:

```rust
pub fn quantized_matmul_rows_profiled(
    out: &mut [f32], ggml_type: GgmlType, weights: &[u8], inputs: &[f32],
    batch: usize, rows: usize, cols: usize,
) -> Result<ProjectionProfile>;

#[derive(Debug, Clone, Default)]
pub struct WorkerWork {
    pub scratch_init_ns: u64,
    pub decode_ns: u64,
    pub accumulate_ns: u64,
    pub zero_copy_ns: u64,
    pub rows: usize,
}

#[derive(Debug, Clone)]
pub struct ProjectionProfile {
    pub schema: u32,
    pub branch: &'static str,
    pub call_wall_ns: u64,
    pub validation_ns: u64,
    pub buffer_init_ns: Option<u64>,
    pub input_transpose_ns: Option<u64>,
    pub rows_wall_ns: Option<u64>,
    pub output_layout_ns: Option<u64>,
    pub delegated_matvec_ns: Option<u64>,
    pub unclassified_wall_ns: u64,
    pub workers: Vec<WorkerWork>,
}
```

`branch` is `empty`, `matvec`, `per_input_dot`, `across_batch`, or `across_batch_pair`.
`None` means unavailable/not executed, not measured zero. Errors return the
existing `QuantError`; they do not expose a partial success profile.

- [x] Add the feature and diagnostic module behind it. Start with the public API and failing integration tests; do not change arithmetic to make the tests pass.

```toml
[features]
projection-diagnostics = []
```

```rust
#[cfg(feature = "projection-diagnostics")]
pub mod projection_diagnostics;
```

Create `tests/projection_diagnostics.rs` with crate-level `#![cfg(feature = "projection-diagnostics")]` and this first contract test:

```rust
use mivi_quant::{GgmlType, quantized_matmul_rows};
use mivi_quant::projection_diagnostics::quantized_matmul_rows_profiled;

#[test]
fn projection_profile_preserves_f32_bits() {
    let cols = 17;
    let rows = 3;
    let weights: Vec<u8> = (0..rows * cols)
        .flat_map(|i| (((i % 19) as f32 - 9.0) * 0.0625).to_le_bytes())
        .collect();
    for batch in [1, 2, 8, 9, 32, 64, 65] {
        let inputs: Vec<f32> = (0..batch * cols)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.125).collect();
        let mut baseline = vec![f32::NAN; batch * rows];
        let mut profiled = baseline.clone();
        quantized_matmul_rows(&mut baseline, GgmlType::F32,
            &weights, &inputs, batch, rows, cols).unwrap();
        let p = quantized_matmul_rows_profiled(&mut profiled, GgmlType::F32,
            &weights, &inputs, batch, rows, cols).unwrap();
        assert_eq!(p.schema, 1);
        assert!(baseline.iter().chain(&profiled).all(|x| x.is_finite()));
        assert!(baseline.iter().zip(&profiled)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
        if batch == 1 {
            assert!(p.delegated_matvec_ns.is_some());
            assert!(p.rows_wall_ns.is_none());
            assert!(p.workers.is_empty());
        } else {
            assert_eq!(p.workers.iter().map(|w| w.rows).sum::<usize>(), rows);
        }
    }
}
```

- [x] Run the red phase:

```bash
CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 cargo test --offline -p mivi-quant --features projection-diagnostics --test projection_diagnostics projection_profile -- --test-threads=1
```

Expected initial failure: diagnostic API is not implemented. Record the actual failure; missing dependencies require permission rather than a fabricated test result.

- [x] Extract the current body into private `quantized_matmul_rows_impl<const PROFILE: bool>` returning `Result<Option<ProjectionProfileInternal>>`; keep the original wrapper's signature and delegate to `<false>`. Use an always-compiled private internal record or equivalent no-op collector so default builds do not depend on the feature-gated public type. The feature-only wrapper invokes `<true>` and converts the record into `ProjectionProfile`. Do not duplicate arithmetic into a benchmark kernel.
- [x] Keep `compute_batched_rows` row iteration, SIMD calls, partition size, decode pairing and output layout unchanged. Add the same constant profiling parameter and return one worker record per existing row chunk. Retain `.try_for_each` on the unprofiled parallel path; only profiled execution collects per-worker records. Separate instantiations must not add default-path profile-vector allocations.

Use this internal helper around existing operations; it returns the operation's
result unchanged and reads the clock only inside the constant guard:

```rust
fn measured<const PROFILE: bool, T>(operation: impl FnOnce() -> T)
    -> (T, Option<u64>)
{
    let started = if PROFILE { Some(std::time::Instant::now()) } else { None };
    let value = operation();
    let elapsed_ns = started.map(|t| {
        u64::try_from(t.elapsed().as_nanos()).unwrap_or(u64::MAX)
    });
    (value, elapsed_ns)
}
```

Use saturating additions for work counters. Derive unclassified wall time by
saturating subtraction of serial top-level stages, including `rows_wall_ns`;
exclude summed worker times from that subtraction. Record worker output zeroing
and row-copy work separately from accumulation. Batch 1 measures delegated
matvec wall time only. Empty work has no compute timers. Validate supported
formats before executing any output-writing branch; unsupported formats fail
without mutating caller output.

- [x] Extend the fixture matrix with finite nonzero F16/BF16/Q8_0/Q4_K/Q6_K weights, columns 512 for K-quants, and rows `RAYON_PARALLEL_THRESHOLD + 1` to cover odd parallel tails. Reuse the existing serialized nonzero quant-block construction in the unit tests; add Q6_K scale/bit-plane cases rather than all-zero weights. Run in local Rayon pools of 1 and 2 threads; compare bits only between profile modes in the same pool.
- [x] Add error tests for short out/input/weights, unsupported `Q4_0`, K-quant columns 255, and `usize::MAX` products. Initialize out with a finite sentinel and require it unchanged on error. Add empty batch/rows and exact supported branch-label checks. Require the six-format matrix to reject nonfinite output explicitly. Cross-branch matvec comparisons report maximum absolute/scaled error under the existing scale-aware test policy; never relax tolerances to silence a failure.
- [x] Run the green command above, followed sequentially by the default-feature regression command:

```bash
CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 cargo test --offline -p mivi-quant matmul -- --test-threads=1
```

Both require nonzero executed test counts. Inspect the default wrapper/constant guards to confirm clocks and collection are absent from default execution; do not claim compiled assembly verification without inspecting it.
- [x] Review the extraction against the pre-change kernel and commit only the four Task 1 files after `git diff --check`.

## Task 2 — Bounded operator executable and private I/O

**Consumes:** Task 1 API; `mivi_model::gguf::GgufFile::{open,get_tensor_data}`; existing `runtime_replay/io.rs::PrivateOutput` and descriptor-pinned directory walker.

**Produces:** `projection_measure --input ABS_JSON --output ABS_JSON`, available only with model feature `projection-diagnostics`. One child prepares one case, runs one mode and returns a bounded private record. Model loading and input generation remain outside kernel timers.

- [x] Add feature forwarding and a gated example:

```toml
# In the existing [features] table:
projection-diagnostics = ["fixture-diagnostics", "mivi-quant/projection-diagnostics"]

[[example]]
name = "projection_measure"
required-features = ["projection-diagnostics"]
```

Forwarding fixture diagnostics permits reuse of the existing private example I/O
without moving production modules. It does not wire diagnostics into the server.

- [x] Add generic `read_json<T: serde::de::DeserializeOwned>(path: &Path) -> io::Result<T>` to the existing I/O module using its bounded regular-file read. Preserve `read_input` as a validating ReplayInput wrapper. For the new generic reader pin the parent descriptor, open the basename with no-follow flags, reject nonregular/oversize/traversal inputs, and preserve existing replay tests. Reuse `PrivateOutput` via `#[path = "runtime_replay/io.rs"] mod private_io;` in the new example. Do not overwrite outputs or emit weights/activations to stdout.
- [x] Define the case input in `examples/projection_measure/case.rs` with `deny_unknown_fields`:

```rust
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaseInput {
    pub schema: u32,
    pub source: Source,
    pub batch: usize,
    pub threads: usize,
    pub profile: bool,
    pub warmup_calls: usize,
    pub measured_calls: usize,
    pub buffer_limit_bytes: usize,
    pub model_limit_bytes: u64,
}
#[derive(serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Source {
    Synthetic { ggml_type: u32, rows: usize, cols: usize },
    Gguf { model_path: std::path::PathBuf, tensor: String },
}
```

Set schema 1, threads 2 for pilot (unit tests may use 1), batch at most 65,
warmup 0–1 and measured calls 1–32. Reject zero rows/cols in executable cases
(empty kernel behavior is tested in Task 1). Require positive checked allocation
limits. Reject unknown fields, bool-as-count JSON and unsupported formats.
Reject model files larger than `model_limit_bytes` before mapping; validate GGUF
rank exactly 2, `cols=info.dims[0]`, `rows=info.dims[1]`, block alignment and
checked byte spans. Use the real descriptor type, never a filename-derived type.
Record tensor identity and paths only in private output.

- [x] Compute conservative buffer requirements before allocation: inputs, caller out, kernel row-major out, optional transposed inputs, and per-worker decoded/output scratch. Count up to `min(rows,threads)` workers. Use checked multiplication/addition, include synthetic weight bytes when allocated, and reject over-budget sizes before allocating. Mapping size is recorded separately from heap estimates; watchdog RSS remains authoritative.
- [x] Generate inputs with the exact deterministic rule below; weights use finite F32/F16/BF16 values or well-formed nonzero quant blocks from the Task 1 fixtures, never random bytes interpreted as floating scales:

```rust
let inputs: Vec<f32> = (0..batch * cols)
    .map(|i| ((i % 23) as f32 - 11.0) * 0.125)
    .collect();
```

Keep synthetic weights private when generated at run time. Create a local Rayon
pool from validated threads. Run optional warmup outside measured totals; then
run the requested mode for each measured call. `std::hint::black_box` the inputs
and outputs; reject nonfinite results. Store output `to_bits()` values privately
for paired comparison, bounded by the driver-selected artifact budget. Record
setup time, individual call wall times, feature profile fields when available,
format/shape/branch/threads and `activation_source="synthetic_f32"`.

Keep only one bounded copy of output bits and verify every measured call matches
it within the child. Emit that reference output plus `all_calls_bit_identical`;
the driver compares the two mode references only when that flag is true. Before
computation, bound JSON output conservatively with 11 bytes per output-bit value
plus 256 KiB for counters/metadata against the existing 4 MiB private writer cap.
If the selected shape exceeds that bound, reject it and explicitly select a
smaller batch for a new case; do not silently truncate or alter work. Pilot
`measured_calls` is 1; higher counts remain bounded and are not independent samples.

- [x] Start with unit tests asserting short/duplicate CLI arguments, unknown input fields, invalid counts/formats, budget overflow, missing/rank-invalid tensor and output collision fail. Add a small synthetic F32 round trip and finite quantized output tests. Add parent-symlink/output-symlink refusal and private output permission tests. Run red before implementing each contract.

```bash
CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 cargo test --offline -p mivi-model --features projection-diagnostics --example projection_measure -- --test-threads=1
CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 cargo test --offline -p mivi-model --features fixture-diagnostics --example runtime_replay cli_io_tests -- --test-threads=1
```

Expected green: nonzero test counts in each command, existing replay I/O behavior
preserved. Do not run a full model or all model tests for this I/O-only slice.
- [x] Review the new executable privacy/bounds, run `git diff --check`, then commit explicit Task 2 files.

## Task 3 — Private manifest, paired supervision and report tests

**Consumes:** Task 2 CLI; `private_io::{PrivateDirectory,read_bounded_json,validate_regular_file,open_regular_file}` and `process_supervisor::run_child` in `scripts/runtime_compare`.

**Produces:** Python CLI `projection_measure.py --manifest ABS_JSON --output-dir ABS_DIR [--validate-only]`, functions `validate_manifest(value: dict) -> dict` and `run_session(settings: dict, output_dir: Path) -> dict`, and schema-1 report with all attempted samples and comparison status.

- [x] Define strict manifest fields: `schema`, absolute `binary`, 40-character `revision`, `repetitions`, `wall_seconds`, `session_seconds`, `rss_bytes`, `artifact_bytes`, `buffer_limit_bytes`, `model_limit_bytes`, `cases`. Cases have a unique safe `name`, `batch`, tagged `source` matching Task 2, `warmup_calls`, `measured_calls`; mode/threads are driver-owned. GGUF cases additionally require outer-case `expected_model_sha256`. Reject duplicate JSON keys while parsing and reject bool-as-integer, unknown fields, duplicate names, relative/traversing/symlink paths and unsupported types.
- [x] Require repetitions 3 for the pilot, wall at most 180s, session at most 900s, RSS at most 2147483648 bytes, artifacts at most 67108864 bytes and positive heap/model limits. Limit manifest/input JSON to 64 KiB, case count to 16 and measured calls to 32. Bound predicted output-bit payload plus logs and reports before launching; reject sessions that cannot reserve at least 1 MiB for final reports. Explicitly selected smaller limits are valid; larger sessions need a separate reviewed decision.
- [x] Write red validation tests with a synthetic case and mocked executable validation:

```python
def test_boolean_batch_rejected(self):
    manifest = self.valid_manifest()
    manifest["cases"][0]["batch"] = True
    with self.assertRaises(ValueError):
        projection_measure.validate_manifest(manifest)

def test_validate_only_creates_no_output(self):
    with unittest.mock.patch.object(projection_measure, "run_child") as child:
        self.invoke_validate_only()
        child.assert_not_called()
        self.assertFalse(self.output_dir.exists())
```

Implement test helpers `valid_manifest`, `invoke_validate_only` and temporary
`output_dir` in this test class; they construct explicit private temp paths and a
64-column, 3-row, batch-9 F32 synthetic case. No model is needed for unit tests.

- [x] Implement validation-only without creating directories, spawning a child or loading/mapping model weights. Live preflight hashes executable/models through pinned regular-file descriptors, records actual hashes and revision, and refuses a model-hash mismatch. Do not assert a supplied revision proves binary provenance: retain the binary hash and the build command in private provenance.
- [x] Implement order `unprofiled/profiled`, `profiled/unprofiled`, `unprofiled/profiled` for repetitions 0/1/2. Spawn one child per mode/case/repetition, never parallel models. Allocate private sample directories and bounded input JSON via `PrivateDirectory`. Use existing `run_child`, with a session deadline and reserved cleanup/report allowance:

```python
allowance = min(settings["wall_seconds"], deadline - time.monotonic())
result = run_child(
    [str(settings["binary"]), "--input", str(input_path),
     "--output", str(result_path)],
    allowance, settings["rss_bytes"], 65536,
    artifact_size=lambda: root.size(settings["artifact_bytes"]),
    artifact_limit_bytes=child_artifact_limit,
)
```

Do not launch if allowance cannot cover cleanup; retain a `session_timeout`
sample instead. Derive `child_artifact_limit` from total cap minus report reserve.
Apply existing `limit_retained_artifacts` on failures; bound persisted logs as
well as drained channels. Retain supervision status/RSS scope/cleanup independently
of runner status. Stop launching after cleanup failure; never relabel failure
as a missing or successful sample. No shell commands or stdin protocol.

- [x] Parse bounded child results with exact schema/type/length checks and a 4 MiB per-result cap. Require finite times, finite output floats decoded from bits, compatible shape/source/branch/threads, `all_calls_bit_identical=true` in both modes, and matching reference output bits. Incompatible/failed samples do not contribute to successful medians. Summarize per-repetition call aggregates, then medians/ranges across three repetitions; repeated inner calls are not independent samples.
- [x] Report profiled/unprofiled total-call ratios only for matched work and compatible timer boundaries. Label instrumentation perturbed or uncertain when the absolute median difference exceeds 5% or observed noise prevents attribution; faster profiled calls do not prove negative overhead. The 5% threshold is a reporting policy, not proof of accurate clocks. Keep worker sums under `worker_work_ns` and serial elapsed fields under `wall_ns`; never calculate wall shares from worker sums. Batch-1 stage availability stays null.
- [x] Add mocked-child tests for order, bit mismatch, nonfinite output, malformed schema, overlarge result, timeout, RSS/artifact failure, collision/symlink/privacy, report-reserve exhaustion and cleanup failure halting launches. Exercise the existing supervisor's real timeout/cleanup tests sequentially as regression coverage.

```bash
python3 -m unittest discover -s scripts/runtime_compare -p 'test_projection_measure.py'
python3 -m unittest discover -s scripts/runtime_compare -p 'test_supervisor_limits.py'
```

Expected green: nonzero test counts, no models loaded, no leaked mock children.
If sandbox permissions deny a required mock operation, retry via the approval
mechanism and disclose the failed attempt separately.
- [x] Review validation, deadline/accounting and privacy; commit only the two new Python files after `git diff --check`.

## Task 4 — Bounded pilot, decision and completed release

**Consumes:** Tasks 1–3 passing focused tests, approved explicit resource limits and local model. **Produces:** private paired results and a redacted evidence report; a decision for the next P1-A slice, not an optimized kernel.

- [x] Build just the new example:

```bash
CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 cargo build --offline -p mivi-model --features projection-diagnostics --example projection_measure --release -j 1
```

Expected: successful scoped example build. Record compiler/Cargo versions,
commit, executable hash and feature flags privately. Do not rebuild llama.cpp or
run a full-model comparison for diagnostic-only instrumentation.
- [x] Create a private session with `mktemp -d`; use existing `PrivateDirectory` for manifests/results. Inspect local GGUF descriptors to choose distinct compatible projection and FFN shapes/types explicitly. Do not assume architecture or tensor names. Start with a small synthetic serial and parallel-tail case; select real weight tensors only after bounds are computed. Use batch 64 plus one small or tail batch for selected shapes within the 15-minute budget; do not run the complete combinatorial test matrix as a live benchmark.
- [x] Record CPU/features/RAM/OS, swap/load/thermal caveats, private model/binary hashes, descriptor shape/type and exact case/repetition/warmup conditions. Choose `model_limit_bytes` from the actual file size plus a checked allowance, not an architecture constant. Generate a private manifest for the new driver; set all pilot limits explicitly. No system cache flush.
- [x] Validate then run using user-created private paths (these paths are examples, not presumed existing):

```bash
python3 scripts/runtime_compare/projection_measure.py --manifest /tmp/mivi-projection-session/manifest.json --output-dir /tmp/mivi-projection-session/results --validate-only
python3 scripts/runtime_compare/projection_measure.py --manifest /tmp/mivi-projection-session/manifest.json --output-dir /tmp/mivi-projection-session/results
```

Require validation creates no output/child. Retain all actual statuses and
cleanup evidence. Count completed pairs explicitly; a failed/budget-limited
pilot is evidence, not permission to silently raise limits or omit failures.
- [x] Verify profile/unprofiled bit agreement, nonfinite rejection, default-path tests and profiler disturbance. Publish median/range call times and separate serial/parallel clock interpretation. Discuss which allocation/transpose/decode/compute costs are measurable and which remain uncertain; do not reuse Phase 0 operator percentages as new measurements.
- [x] Write `docs/PROJECTION_COST_EVIDENCE_2026-10-05.md` with redacted provenance, results, failures, instrumentation caveats and exactly one next decision: caller-owned scratch experiment, faithful locality experiment, or insufficient evidence. Keep master P1-A optimization/promotion boxes unchecked; mark only cost measurement complete if the actual data supports that status.
- [x] Self-review the diff, private file modes, nonzero test counts, bounds/cleanup and profile type/clock claims. Request Luna/high review if available; disclose capacity failures without claiming independent approval. Fix introduced findings, rerun affected focused commands and run `git diff --check`.
- [x] After acceptance, increment the then-current workspace patch version once, update all 14 local lockfile package versions without changing third-party dependencies, and add changelog sources: approved spec/Phase 0 evidence, Colibri methodology and pinned GGML numerical-path context. State measurement-only scope, unresolved cross-engine divergence and no default kernel promotion. Do not bump at intermediate task commits.
- [ ] Stage explicit task-owned docs/release files, inspect `git diff --cached --check` and staged paths, commit, then non-force push `feat/runtime-parity-profiling`. Verify remote branch hash equals local HEAD; leave `.gitignore` unstaged.

## Acceptance/handoff

Task 1 delivers shared-arithmetic diagnostics with matched bits and error/branch
coverage. Task 2 delivers bounded safe operator cases without model inference.
Task 3 delivers retained, supervised paired results. Task 4 completes the
measurement slice only after actual bounded evidence, review and release.

No scratch reuse, kernel locality rewrite, model integration or approximation is
implemented in this plan. The next plan consumes the measured decision and
requires its own numerical and end-to-end request regression gates. Existing
short/medium cross-engine divergence remains a separately disclosed limitation.
