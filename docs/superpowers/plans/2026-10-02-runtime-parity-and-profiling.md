# Runtime Parity and Profiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce reproducible, bounded evidence identifying Mivi's CPU latency costs and numerical uncertainty before selecting production optimizations.

**Architecture:** Extend existing feature-gated diagnostic types and use a feature-required model replay example plus a private Python comparison driver. Keep model-level, router-level, and reference-engine boundaries distinct. Independent tiny oracles validate serialized quantized weights and carried hybrid state; reference HTTP scores are supplemental evidence, not a full-state oracle.

**Tech Stack:** Existing Rust model/server/quant crates, serde_json, Rayon, Python standard library, existing Python reference/export code, optional pinned local llama.cpp executable.

## Global Constraints

- Read the [master roadmap](2026-10-02-native-cpu-agent-improvements.md) and [research evidence](../../AGENT_CPU_RUNTIME_RESEARCH_AND_PLAN.md) before execution.
- Cargo jobs1; Rust test-threads1; Rayon/inference threads2; inference concurrency1. No workspace-wide commands or overlapping builds/model runs.
- Preserve unrelated edits, especially `.gitignore`. Use existing checkout and explicit file staging.
- Diagnostic code is feature-gated or test-only. Default server behavior and public API must not acquire raw-prompt capture or hidden benchmarking.
- No model downloads needed. Require explicit model/reference paths at execution; private `/tmp` paths from earlier experiments are not production configuration.
- Use private exclusive artifact creation (directories0700, files0600 on Unix); reject symlink/existing-output collisions and fail closed on nonprivate destinations.
- Bounds: manifest/input/result file ≤4MiB each; retained text ≤64KiB per channel; prompt IDs ≤4096; generated IDs ≤64; teacher-forced probe positions ≤16; selected logit IDs ≤64; result artifact budget ≤64MiB per session. Validate before loading weights or launching subprocesses.
- Initial measured matrix: three workloads × three paired repetitions, plus separate profiling controls; alternate engine order. Per-run wall ceiling180s including startup/cleanup; session ceiling45min; observed child RSS ceiling2GiB, sampled at100ms on Linux. Kill owned process group on breach and retain a failure record. RSS polling is a guard, not a guarantee against transient spikes; label other platforms' enforcement unavailable.
- These harness bounds are adjustable explicit diagnostic configuration, not model-dependent runtime constants. Require a new reviewed budget for larger contexts/models.
- Local reference binds127.0.0.1 only, no API key. No telemetry/upload or full workspace reads in fixtures.
- For completed implementation releases: one patch increment, changelog attribution/limitations, scoped verification and review, nonforce GitHub push under standing policy. Do not bump once per unfinished task.

---

## Files and boundaries

| File | Action / responsibility |
| --- | --- |
| `crates/mivi-model/src/fixture_diagnostics/replay.rs` | Create: typed replay input/output validation, numeric profile export, bounded logit comparison |
| `crates/mivi-model/src/fixture_diagnostics.rs` | Modify: feature-gated `pub mod replay`; reuse existing capture types |
| `crates/mivi-model/examples/runtime_replay.rs` | Create: explicit private diagnostic replay, no server CLI change |
| `crates/mivi-model/Cargo.toml` | Modify: example requires `fixture-diagnostics`; scoped Rayon dev dependency for two-thread pool |
| `scripts/runtime_compare/compare.py` | Create: standard-library bounded paired-process workflow and numerical report |
| `scripts/runtime_compare/test_compare.py` | Create: synthetic subprocess/HTTP/report tests; no model required |
| `crates/mivi-server/src/fixture_diagnostics.rs` | Modify: optional numeric profile/phase fields only where existing captures lack them |
| `crates/mivi-server/src/fixture_diagnostics/fixtures.rs` | Modify: targeted boundary/profiling-control tests using existing actor lifecycle |
| `training/export/generate_adversarial_fixture.py` | Create: separate tiny synthetic fixture export; never overwrite existing tiny model |
| `tests/fixtures/hybrid_adversarial.json` | Create: synthetic oracle trace/provenance, no real-model weights |
| `tests/hybrid_adversarial_oracle.rs` | Create: independent hybrid/reset/chunk-continuation comparison |
| `crates/mivi-quant/src/q4_k_m.rs`, `q6_k.rs` | Modify: independently derived block-layout arithmetic tests |
| `docs/KV_QUANT_AND_CHUNKED_PREFILL_RESEARCH.md`, `docs/AIRLLM_AND_COLIBRI_RESEARCH.md` | Modify: correct dated claims and link measured findings |

Never add a raw model-state dump to normal inference. Existing `ForwardProfileSnapshot`, `ModelRecorder`, and `FixtureSession` remain owners of their current measurements.

## Task 1 — Typed bounded replay contract and substage export

**Files:** create `crates/mivi-model/src/fixture_diagnostics/replay.rs`; modify `crates/mivi-model/src/fixture_diagnostics.rs`.

**Interfaces:** consumes `ForwardProfileSnapshot` and existing `CaptureLimits`. Produces the following feature-gated types. All JSON floats must be finite; use checked arithmetic before allocations.

```rust
use serde::{Deserialize, Serialize};
use crate::ForwardProfileSnapshot;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayInput {
    pub prompt_ids: Vec<u32>,
    pub context: usize,
    pub tile: usize,
    pub max_tokens: usize,
    pub profile: bool,
    pub split_prefill: bool,
    pub teacher_forced_ids: Vec<u32>,
    pub logit_ids: Vec<u32>,
}

#[derive(Debug, Serialize)]
pub struct ProfileMicros {
    pub tokens: usize,
    pub embedding: u64,
    pub attention: u64,
    pub ssm: u64,
    pub logits: u64,
    pub attention_substages: [u64; 5],
    pub ssm_substages: [u64; 5],
}

impl ReplayInput {
    pub fn validate(&self) -> Result<(), &'static str>;
}
impl ProfileMicros {
    pub fn from_snapshot(s: ForwardProfileSnapshot) -> Result<Self, &'static str>;
}
```

The method declarations describe required signatures, not paste-ready Rust implementations. Validation: nonempty prompt, context1..4096, tile1..128, output0..64, prompt+output+teacher sequence within context using checked addition, teacher/logit limits above, unique logit IDs, and no teacher-forced sampling mixture (`teacher_forced_ids` nonempty requires `max_tokens == 0`). Token IDs must subsequently be validated against the loaded vocabulary; BOS-adjusted length must also fit context.

Array order is fixed: attention `[norm, qkv_projection, causal_attention, output_projection, ffn]`; SSM `[norm, input_projection, convolution, output_projection, ffn]`. Convert `Duration::as_micros()` with `u64::try_from`, never wrapping casts. Record aggregate/substage overlap; do not add them together as independent total time.

- [x] Add `pub mod replay;` under the existing feature-gated diagnostic module; write `replay::tests::replay_rejects_unbounded_input` and `replay::tests::profile_export_preserves_substages`.

```rust
#[test]
fn profile_export_preserves_substages() {
    let mut s = crate::ForwardProfileSnapshot::default();
    s.attention_stages.qkv_projection = std::time::Duration::from_micros(17);
    s.ssm_stages.convolution = std::time::Duration::from_micros(23);
    let p = ProfileMicros::from_snapshot(s).unwrap();
    assert_eq!(p.attention_substages, [0, 17, 0, 0, 0]);
    assert_eq!(p.ssm_substages, [0, 0, 23, 0, 0]);
}
```

- [x] Run RED: `CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 cargo test -p mivi-model --features fixture-diagnostics replay::tests -- --test-threads=1`. Expected: missing types/methods or failing bound assertions, not unrelated compiler errors.
- [x] Implement validation/profile conversion; add exact-limit, limit+1, arithmetic-overflow, duplicate-logit, and unknown-field cases.
- [x] Repeat command GREEN; require nonzero test count and all selected tests pass.
- [x] Commit Task1's reviewed model diagnostic files with the planning/release documentation under the standing patch-release policy. Include the approved arithmetic cleanup resolving the verification checkpoint.

Execution note (2026-10-02): Task1's code and selected verification are complete;
GPT-6 Luna/high approved spec compliance and code quality; its minor input-reader
documentation finding was addressed. Baseline16 diagnostic tests passed; the new11-test
filter had six intended RED failures with method stubs, then eleven GREEN passes.
The combined diagnostic filter passed27 tests. Commands additionally used `--lib`
to avoid unrelated targets. Default-feature `mivi-model --lib` check passed.
Extreme `usize` inputs reject before addition; checked context sums also remain
defensive. Profile overflow is checked independently for all14 duration fields.
No model loads or performance measurements have occurred during Task1.

Verification checkpoint resolved after the user's continuation: replace26 KV
and three model/transformer manual-rounding/divisibility patterns with native
integer operations. A new extreme-dimension regression exposed round-up
overflow; checked Q8 byte-size multiplication now returns `AllocationOverflow`.
The corrected regression and partial-block storage test pass. Strict scoped
model-library Clippy including dependencies now passes without suppressions.
Additional small cache/prefix/model-prefix/attention filters passed7/13/6/2
tests; six live model-prefix tests remained ignored. Large memory tests were
excluded. GPT-6 Luna/high approved the arithmetic cleanup with no findings;
all55 selected tests passed after cleanup. Release0.2.63 is scoped to this
reviewed increment on `feat/runtime-parity-profiling`, not a completed Phase0
or a main-branch merge. Task2's execution is recorded below.

## Task 2 — Private model replay and same-engine split parity

**Files:** create `crates/mivi-model/examples/runtime_replay.rs`; modify `crates/mivi-model/Cargo.toml`; extend replay contract tests in Task1's module.

**Consumes:** `ReplayInput`, `ProfileMicros`, `Model::load_with_options`, `Model::forward`, `Model::generate_tokens_incremental_with_cancel`, existing recorder methods, `PrefillStrategy`. **Produces:** `runtime_replay --model PATH --input PATH --output PATH`; stdout only redacted status/path, never prompt or decoded response.

Example registration and dependency:

```toml
[dev-dependencies]
rayon.workspace = true

[[example]]
name = "runtime_replay"
required-features = ["fixture-diagnostics"]
```

Output JSON schema1: model path/identity metadata, effective context/tile/KV=`F32`, temperature0, seed7, repetition1, presence/frequency0, normalized actual prompt IDs, generated content IDs, observed terminal IDs separately, capture truncation flags, model outcome, profile before decode, raw/delivered durations in microseconds, worker/process-return status, and optional teacher-forced score records. Load time is separate from inference time. Reuse existing capture snapshots instead of inventing conflicting raw/visible clocks.

Teacher-forced score records contain position, supplied next-token ID, selected logits, top1/top2 IDs and margin. No full 65536-wide logit arrays or hidden-state export. Stop at16 probe positions. Compare real-model scores only if the reference exposes equivalent pre-sampler values; probability-only results must be labeled as probabilities.

- [x] Add tests for existing-file refusal, oversized input rejection before load, and missing/invalid arguments. Test the private writer against a temporary directory created by the test, not an existing user's file.
- [x] Establish RED before implementation. Execution used focused failing contract tests with stubs rather than the planned missing-file build failure; see the execution note below.
- [x] Implement explicit argument parsing, bounded reads, private exclusive output creation, and model setup. Configure a scoped `rayon::ThreadPoolBuilder::new().num_threads(2).build()`; do not silently inherit all laptop cores.
- [x] Use this generation sequence for split mode inside the pool; normal mode makes one call with the same settings. Reset sampler/profile/cache between modes, and capture the effective BOS-adjusted IDs.

```rust
model.reset_context();
model.prefix_cache.clear();
model.sampler.config.temperature = 0.0;
model.sampler.config.seed = Some(7);
model.sampler.config.repetition_penalty = 1.0;
model.sampler.config.presence_penalty = 0.0;
model.sampler.config.frequency_penalty = 0.0;
model.sampler.set_seed(7);
model.generate_tokens_incremental_with_cancel(
    &input.prompt_ids, 0, 0, |_, _| true, || deadline_reached()
)?;
let prefill_profile = model.forward_profile();
let pos = model.current_pos();
let (text, ids) = model.generate_tokens_incremental_with_cancel(
    &[], pos, input.max_tokens, |_, _| true, || deadline_reached()
)?;
```

Here `deadline_reached` is a local closure over an `Instant` and the explicit180s diagnostic limit, created before model load; the outer driver enforces startup/physical termination as well. Do not infer successful generation solely from a returned text value after cancellation.

- [x] Implement teacher-forcing with `Model::forward(token_id, pos)` on the same normalized prefix in token-major mode. Keep teacher-mode timing out of chunked performance results.
- [x] Repeat scoped build GREEN; run the replay-module unit tests from Task1.
- [x] Add an ignored model-required split-parity test named `replay_split_prefill_matches_single_call`, taking `MIVI_TEST_MODEL` and testing short and long synthetic prefixes sequentially. Check generated IDs, delivered text, terminal handling and final position; save failures privately.
- [x] Run only that live test after an absolute model path and budget are validated: `CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 MIVI_TEST_MODEL=/absolute/path/model.gguf timeout --signal=TERM --kill-after=5s 600s cargo test -p mivi-model --example runtime_replay --release --features fixture-diagnostics replay_split_prefill_matches_single_call --offline -- --ignored --test-threads=1`. Expected: two parity cases pass, not simply successful processes.
- [x] Review privacy/default feature isolation and include Task2 files explicitly in the patch-release commit: `test: add private model runtime replay (v0.2.64)`.

Execution note (2026-10-02): split implementation into the example entry point,
`runtime_replay/io.rs` and `runtime_replay/generation.rs`; add feature-gated direct
capture timing, terminal/stop observations and prefill-only profile snapshots.
Update three server test initializers for capture compatibility only. TDD used
four failing I/O-contract tests, three failing executor/traversal tests, then a
failing repeated-write regression; all passed after implementation. Review added
two RED/GREEN regressions for malformed BOS metadata and entry-clock labeling.
Fifteen non-live example tests, 28 model diagnostic tests and 24 server fixture
compatibility tests passed (two unrelated server live fixtures remained ignored).
Strict model/example Clippy and the default-feature model-library check passed.
GPT-6 Luna/high approved the integrated
implementation and follow-up fixes; the security review prompted pinned directory
walking and single-use output rather than path-based permission checks.

The release example built with one Cargo job. Tiny synthetic-model runs checked
two finite teacher probes and normal/split output equivalence; private results
were mode0600 within a mode0700 directory outside Git. The real local
LFM2.5-1.2B-Instruct-Q4_K_M fixture passed both 110/2636-prefix parity cases in
180.58s total with two inference threads, an external600s timeout and no concurrent
Cargo/model workloads. An initial relative-path launch failed before loading;
use an absolute `MIVI_TEST_MODEL` because Cargo changes the test working directory.
No paired-reference speedup, agent-quality result, hard RSS enforcement or
independent numerical parity is claimed. Those remain Tasks3–6.

## Task 3 — Bounded paired comparison driver

**Files:** create `scripts/runtime_compare/compare.py`, `scripts/runtime_compare/test_compare.py`,
`private_io.py`, `process_supervisor.py`, `gguf_metadata.py` and `test_gguf_metadata.py`
in that directory, plus focused post-exit/retention regressions in
`test_supervisor_limits.py`. The helper split follows implementation inspection: metadata
preflight, pinned private I/O and continuous process supervision are separate
safety boundaries, not additional runtime features.

**Consumes:** Task2 binary/output; optional explicitly supplied local llama.cpp executable; private JSON manifest. **Produces:** `python3 scripts/runtime_compare/compare.py --manifest PATH --output-dir PATH`; private JSON/Markdown reports with schema1.

Manifest schema1 fields: `model_path`, `model_sha256`, `mivi_binary`, `reference_binary`, `reference_revision`, `context`, `tile`, `max_tokens`, `repetitions`, `wall_seconds`, `session_seconds`, `rss_bytes`, `artifact_bytes`, `cases`. Each case has unique `name` and bounded `prompt_ids`; no server instruction text mutation. Validate executables/files, SHA256 streaming, all bounds, and all cases before starting either engine. Refuse unknown fields and reused output directories. Store explicit effective settings and reference version output.

Launch each validated argv list with `subprocess.Popen(argv, start_new_session=True)`, never a shell. Retain only bounded logs; poll time/RSS/file growth; terminate then kill only the owned process group, wait for child, and record cleanup outcome. Default test matrix uses synthetic short/medium/approximately2636-ID prompts from the loaded tokenizer; do not use arbitrary IDs outside its vocabulary. Count actual normalized IDs, not approximate token counts.

Reference configuration: CPU-only, threads2, context4096, parallel1, F32 K/V, microbatch/tile64, flash attention off, fitting off, warmup disabled, default repacking retained. Discover/verify flags against the pinned executable; unsupported flags abort instead of silently comparing different settings. `/completion` takes numeric IDs, greedy temperature0, seed7, repetition1, frequency/presence0, output48; explicitly disable slot reuse and verify timings/processed count. Match stop/BOS/EOS policy where possible and report any unmatchable boundary.

Required pure reporting interface:

```python
def first_difference(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))

def summarize(samples: list[dict]) -> dict:
    import math
    import statistics
    from collections import Counter

    counts = Counter(sample["status"] for sample in samples)
    values = [sample["prefill_ms"] for sample in samples
              if sample["status"] == "complete"]
    if any(not isinstance(value, (int, float)) or isinstance(value, bool)
           or not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("invalid completed prefill duration")
    summary = None if not values else {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }
    return {"status_counts": dict(counts), "prefill_ms": summary}
```

The following executable test fixes report semantics. Empty successes produce `None` summaries, not zeros; add that case alongside the example.

- [x] Write failing standard-library tests before adding functions:

```python
import unittest
from compare import first_difference, summarize

class ReportTests(unittest.TestCase):
    def test_first_divergence_and_shortened_output(self):
        self.assertIsNone(first_difference([1, 2], [1, 2]))
        self.assertEqual(first_difference([1, 2], [1, 3]), 1)
        self.assertEqual(first_difference([1], [1, 2]), 1)

    def test_timeouts_stay_in_report(self):
        result = summarize([
            {"status": "complete", "prefill_ms": 10.0},
            {"status": "complete", "prefill_ms": 20.0},
            {"status": "timeout", "prefill_ms": None},
        ])
        self.assertEqual(result["status_counts"], {"complete": 2, "timeout": 1})
        self.assertEqual(result["prefill_ms"], {"median": 15.0, "min": 10.0, "max": 20.0})
```

- [x] Run RED: `python3 -m unittest discover -s scripts/runtime_compare -p 'test_*.py'`; expected missing module/functions or failed assertions.
- [x] Implement validation, supervision, alternating pairs, and report generation. Add mocked child tests for timeout/RSS/oversized logs, startup failure, collision/symlink refusal, `[DONE]`/EOF disagreement, and cleanup failure. No model loads in these tests.
- [x] Repeat GREEN and require all selected tests pass.
- [x] For the first divergent position, replay the identical prefix in both engines; compare matched selected scores if available, top-token margin, stop policy and quantized-activation behavior. Mark absent raw logits or intermediate states as unavailable, not equal.
- [x] Require reports to distinguish model-callback TTFT from reference network-visible TTFT. Never compute a ratio between mismatched boundaries without flagging it.
- [x] Review the script/helper files and associated usage/release documentation,
  excluding private artifacts and unrelated changes; commit the completed
  increment: `test: add bounded paired runtime comparison`.

Task3 execution: seven standard-library modules/helpers/tests implement the
bounded driver. Initial missing-module RED and focused safety/protocol regressions
preceded GREEN; independent review identified probe-cleanup continuation and
fast-exit artifact bypass, then deadline/source-reporting gaps. Each was repaired
and re-reviewed; native default stop strings are carried from verified output,
not duplicated in the adapter. Fresh scoped54 tests passed. Validation-only
created no output/child; tiny native CLI completed and reaped successfully with
private0700/0600 artifacts, reference unavailable, partial/exit2 and no parity
claim. Artifact/RSS caps are sampled; unsafe cleanup refuses retention. Discarded
diagnostic suffixes are explicitly counted. Actual paired measurements remain
Task6 because no reference executable is installed. Task3 release0.2.65 uses a
scoped default-feature model-library check with jobs1; normal server is unchanged.

## Task 4 — Router timing and profiling controls

**Files:** modify `crates/mivi-server/src/fixture_diagnostics.rs`, `crates/mivi-server/src/fixture_diagnostics/fixtures.rs`; touch `generation.rs`/`engine_actor.rs` only for a missing feature-gated numeric boundary hook.

**Consumes:** existing `FixtureRecord`, `EngineTerminal`, `ModelCapture`, `try_spawn_fixture` lifecycle and Task1 profile representation. **Produces:** numeric optional boundary fields in existing private records and test `fixture_profile_control_parity`.

- [x] Write mocked tests enforcing heartbeat/empty-envelope versus useful delta distinction, monotonic timestamps, missing-boundary `None`, and physical returned/cancelled/receiver-closed status. Verify bounded captured output.
- [x] Run focused `fixture_profile` tests with Cargo jobs1 and one test thread; eight selected tests passed and the one model-required control was ignored. Both contention regressions were first observed failing, then passed after repair.
- [x] Export existing optional model profile into the record without duplicate per-layer timers. Queue/render/tokenize/prefill/first raw/first visible/decode/parse-finish/worker return are recorded only when observed; unavailable prompt-render timing stays `None`.
- [x] Add an ignored live `fixture_profile_control_parity` fixture with sequential identical seeded settings, truncation/completion checks, private artifact writer and owned actor teardown.
- [x] Build only the `mivi-server` release library test target with one Cargo job. Tiny synthetic GGUF off/on control passed: identical captured output, profile observed only when enabled, worker return and actor cleanup observed.
- [x] Capture integration changed only in the feature-gated diagnostics path; focused profile controls cover it. No broad model-required test sweep.
- [x] Verify default-feature `mivi-server` library isolation; prepare explicit Task4 release files for `test: expose fixture profile and lifecycle boundaries`.

## Task 5 — Independent adversarial arithmetic and hybrid oracle

**Files:** create `training/export/generate_adversarial_fixture.py`, `tests/fixtures/hybrid_adversarial.json`, `tests/hybrid_adversarial_oracle.rs`; modify `crates/mivi-quant/src/q4_k_m.rs`, `q6_k.rs`; modify `reference/reference_engine.py` only if an independently tested missing operation is needed.

**Consumes:** existing `GgufWriter`, supported GGUF tensor layouts, independent Python reference operations. **Produces:** new synthetic tiny GGUF in a private output directory and schema1 oracle traces of all64 logits over bounded hybrid sequences. The committed JSON contains synthetic data/provenance only, never real-model tensor dumps.

Fixture contract: vocabulary64, context128, alternating SSM/attention layers, nonuniform bounded weights, GQA ratio2, nonzero positional steps, nonzero convolution carry. Use dimension64/F32 for the hybrid graph and separate width256/512 Q4_K/Q6_K block tests for their actual format constraints. Do not pad a64-wide layer into a purported valid Q4_K fixture.

Trace JSON includes exporter/reference revision, config, tensor byte hashes, cases, token IDs, all64 logits, top token, and numerical policy. Python must evaluate **decoded serialized bytes**, not original pre-quantized weights. Independent layout equations must not call Mivi's Rust decoder or simply duplicate its helper output.

- [x] Add scalar quant block tests named `adversarial_q4_layout_matches_independent_values` and `adversarial_q6_layout_matches_independent_values`: low/high nibble isolation, nontrivial scale/min packing, signed scales, zeros, two blocks, odd row counts, and partial output-row tiles. Use exact binary-friendly values for layout tests and a separately documented scale-aware accumulation tolerance for dot tests.
- [x] Run focused quant RED/GREEN verification one operation at a time with Cargo jobs1, Rayon2, and Rust test threads1; final scoped filter passes 2 tests. The new decoder tests found no arithmetic discrepancy, so production quant kernels remain unchanged.
- [x] Extend the independent generator with explicit `--output-dir PATH` and refuse collisions. Export both specified teacher sequences plus a changed-prefix case, reset the oracle between cases, and derive tensor values/traces from serialized GGUF bytes.
- [x] Add Rust tests comparing token-major, chunked tiles1/2/3/8, split continuation, reset, and cached changed-prefix execution against all-logit traces. Warm a shared cached chunk, retain it over context reset, and diverge afterward. Assert every logit is finite; use absolute+relative `1e-4` checks and an error-composed top-margin bound.

```rust
fn assert_close(actual: f32, expected: f32, atol: f32, rtol: f32) {
    assert!(actual.is_finite() && expected.is_finite());
    let limit = atol + rtol * expected.abs();
    assert!((actual - expected).abs() <= limit,
        "actual={actual}, expected={expected}, limit={limit}");
}
```

Start F32 tiny-graph acceptance at `atol=1e-4`, `rtol=1e-4`. Any failure requires first-divergence analysis; do not loosen tolerances merely to pass. Quantized same-byte accumulation needs its own justified scale-aware policy, not transfer of the old blanket0.05 logit bound.

- [x] Run scoped integration RED/GREEN with Cargo jobs1/Rayon2/test threads1. Missing `MIVI_ADVERSARIAL_FIXTURE` fails actionably; explicit private GGUF path passes the integration test.
- [x] Review oracle independence, provenance/source links, reset/carried-state coverage; prepare Task5 implementation plus the required v0.2.67 release metadata for `test: add independent adversarial hybrid oracle`.

## Task 6 — Execute bounded evidence run, correct docs, and hand off

**Files:** modify research report and affected research notes; update this plan's checkboxes only after evidence exists. Measurements remain private. Release files change only if completed implementation is being released.

**Consumes:** Tasks1–5 passing tests and explicit paths/budget. **Produces:** private paired report plus a redacted committed findings summary identifying the next optimization or correctness fix.

- [ ] Record CPU/features/RAM, binary revisions/hashes, model SHA256, tensor-format inventory, OS/thermal caveats, sampler and terminal policies. Reference baseline model hash is `b1b3de114215d9507409a662a501a631095a479a419584e8a2ded6304b19b4f5`; a mismatch is a different experiment, not the same baseline.
- [ ] Validate manifest via `python3 scripts/runtime_compare/compare.py --manifest /tmp/mivi-runtime-session/manifest.json --output-dir /tmp/mivi-runtime-session/results --validate-only`. These are example user-created private paths; do not presume they already exist. Expected: validated budgets/paths, no model loaded/process started.
- [ ] Run the same command without `--validate-only` after approval of live resource usage. Measure three paired repetitions for each short/medium/long workload; order Mivi/reference, reference/Mivi, Mivi/reference. Retain failures and run separate profile controls, never substitute profiled samples into unprofiled medians.
- [ ] Record cold inference state separately from OS page-cache warmth; no system-wide cache flush. Decode comparisons require matched output lengths or per-token disclosure, not equal total time assertions.
- [ ] Complete first-divergence score/terminal diagnosis; use Task5 to distinguish graph/state/layout errors from near-tied or changed-arithmetic decisions. Record unresolved causes explicitly.
- [ ] Correct `Q8 lossless` claims to lossy, label6×–10× historical goals as unmeasured, and correct F32 K+V4K arithmetic to `2*6*4096*512*4 = 100663296 bytes = 96MiB`, excluding SSM/allocator/prefix overhead. Annotate historical prefill defaults by date.
- [ ] Select exactly one next package from the master roadmap using measured substage costs. Publish relative contributions and uncertainty, not “attention is slow” from whole-block aggregates.
- [ ] Self-review privacy, nonzero test counts, bounds/cleanup, correctness gaps and claimed performance. Run `git diff --check`; stage only reviewed task-owned documentation.
- [ ] If releasing completed diagnostic implementation, apply the master release checklist once to that increment; otherwise leave release status explicitly pending. Changelog sources: Kimi fixtures, Colibri protocol, GGML quant/repack, relevant hybrid-cache guidance, and Liquid tool card where used.

## Acceptance and handoff

P0 is complete only when bounded tooling and selected tests pass, actual repeated results exist, timing boundaries/settings are honest, independent numerical coverage is documented, and output divergence is explained or explicitly unresolved. Unresolved divergence blocks promoting approximate modes or claiming parity; it does not justify an indefinite speculative rewrite.

Choose P1's first operator only after this gate. P2/P3 can receive their own focused plans using the contracts and evidence established here. Do not mark any implementation checkbox complete merely because this document has been written.
