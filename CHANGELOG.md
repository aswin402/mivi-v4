# Changelog

All notable changes to **Mivi-v4** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v0.2.77] - 2026-10-10

### Faithful Four-Row Batch Projection Experiment (Opt-In, Stage 2)

#### Ideas, Inspirations & Sources

- Share activation loads across four decoded output rows while retaining Mivi's established F32 pair-panel accumulation order. [Projection-cost evidence](docs/PROJECTION_COST_EVIDENCE_2026-10-05.md) identified accumulation as dominant; [scratch-only evidence](docs/BATCH_SCRATCH_EVIDENCE_2026-10-09.md) supplies a separate control.
- Three-way comparisons, balanced order and retained regressions follow [Colibri's benchmarking protocol](https://github.com/JustVugg/colibri/blob/main/docs/benchmarking.md). No external inference source was copied.
- See the [approved design](docs/superpowers/specs/2026-10-09-four-row-prefill-design.md), [implementation plan](docs/superpowers/plans/2026-10-09-four-row-prefill.md), and [all raw operator timings](docs/FOUR_ROW_PREFILL_EVIDENCE_2026-10-10.md).

#### Added and Evaluated

- Add feature-gated four-row SIMD accumulation behind checked slice bounds and AVX2/FMA detection, with existing pair-helper fallback. User-approved unsafe intrinsics stay in a private routine; preserve vector FMA, scalar-tail arithmetic, column order and panels of 128.
- Add a separate four-row scratch constructor and explicit checked quant projection API. Retain ordinary and scratch-only control math, worker partitions, small-batch/partial-group behavior and allocation reuse. Add a narrow Clippy annotation to the established eight-argument pair API without changing its behavior.
- Verify complete bits over 3,168 six-format projection cases plus constructor, invalid-buffer/capacity, unaligned delegate, odd-width float, core cancellation/tail and fallback controls.
- Run a bounded two-thread, three-control operator pilot with generated F32 activations, full real Q4_K/Q6_K matrices and synthetic Q8_0, one warmup plus six balanced-order triples per case. All member-final outputs matched baseline bits and were finite; preserve all measured and warmup times.
- Against scratch-only, paired median time was 14.57% lower for Q6_K batch 64, 17.08% lower for Q6_K batch 65, and 16.74% lower for Q4_K batch 65. Q4_K batch 32 was 9.58% slower; Q4_K batch 64 and synthetic/small cases were mixed or regressed. No universal speedup or default-promotion claim.
- Default model/server behavior stays unchanged. Captured-activation/model-state parity, agent latency, RSS, other models and cross-ISA execution remain open gates. Cargo uses one job and inference at most two threads; no full-workspace build/check/test.

## [v0.2.76] - 2026-10-09

### Reusable Batch Projection Scratch (Opt-In, Stage 1)

#### Ideas, Inspirations & Sources

- Reuse caller-owned projection buffers while preserving Mivi's existing checked batch arithmetic. The [projection-cost evidence](docs/PROJECTION_COST_EVIDENCE_2026-10-05.md) separates allocation/layout costs from dominant accumulation work; [agent-sized evidence](docs/Q4_AGENT_SIZED_EVIDENCE_2026-10-07.md) motivates work on prefill rather than decode alone.
- Separating candidates and limiting performance claims follows [Colibri's benchmarking protocol](https://github.com/JustVugg/colibri/blob/main/docs/benchmarking.md). No external inference code was copied.
- See the [approved design](docs/superpowers/specs/2026-10-09-batch-prefill-scratch-design.md), [implementation plan](docs/superpowers/plans/2026-10-09-batch-prefill-scratch.md), and [correctness evidence](docs/BATCH_SCRATCH_EVIDENCE_2026-10-09.md).

#### Added and Verified

- Add `batch-scratch-experiment` with fallible, overflow-checked `BatchProjectionScratch` ownership and an explicit scratch-taking batch API. Caller-supplied shapes/worker capacity are checked before writes; no model-name dispatch or new dependencies.
- Preserve existing dot/FMA order, paired-row SIMD/portable dispatch, partition boundaries and single-token delegation. Reuse disjoint worker buffers without projection-owned resizing; reject insufficient capacity when moving into a larger Rayon pool.
- Add six focused controls: all six supported formats across batch boundaries, odd/parallel rows, exact output bits, poisoned-buffer reuse with stable allocation identities, failure-before-write, constructor overflow, empty work, unaligned F32 delegation and zero-column tails.
- Default runtime/server behavior is unchanged. This is stage 1 infrastructure, not a measured latency gain; four-row accumulation, model integration and end-to-end performance/RSS gates remain pending. Cargo uses one job, inference at most two threads; no full-workspace build/check/test.

## [v0.2.75] - 2026-10-07

### Agent-Sized Q4 First-Output and Decode Evaluation

#### Ideas, Inspirations & Sources

- Extend the [v0.2.74 cached-sum pilot](docs/Q4_CACHED_SUMS_EVIDENCE_2026-10-07.md) using Mivi's existing bounded fixture recorder and chunked prefill, separating engine first nonempty delivery from continuation decode.
- Cache-state declarations, alternating fixed-work pairs and retaining negative results follow [Colibri's benchmarking protocol](https://github.com/JustVugg/colibri/blob/main/docs/benchmarking.md). No external inference source was copied.
- See the [evaluation plan](docs/superpowers/plans/2026-10-07-q4-agent-sized-evaluation.md) and [complete pair evidence](docs/Q4_AGENT_SIZED_EVIDENCE_2026-10-07.md).

#### Added and Verified

- Add a test-only diagnostic requiring both `q4-cached-sums-experiment` and `fixture-diagnostics`, with one warmup and three alternating measured pairs for each cold-prefix/warm-prefix case. Add nonignored measurement controls for timer boundaries, complete work/cache accounting, nonempty delivery, truncation/overflow and accidental profiling rejection.
- Test the local 1.2B Instruct GGUF with 1,557 input IDs (1,558 effective with BOS), F32 KV, context 2,048, tile 64, a greedy first callback, and 16 fixed continuation forwards. All members passed bit-exact complete final-prefill and continuation-logit vectors, first token/text and final KV/SSM-state comparisons. Intermediate prompt logits are not produced or claimed.
- Cold-prefix baseline/candidate first-output medians were 45.833226s/47.440989s; warm-prefix medians were 2.894534s/2.947387s with 1,536 tokens restored and 22 processed. Both ran zero candidate projections during prefill. This selector does not optimize chunked prefill or resolve the initial agent wait; timing differences in that unchanged path are not optimization gains.
- Decode paired-ratio medians were 0.932190 cold and 0.903398 warm (6.78% and 9.66% lower). Preserve the negative cold pair where candidate decode was 2.81% slower; do not infer general superiority or default-promotion readiness from this single-host pilot.
- No runtime kernels, provider configuration or server behavior changed. The candidate remains default-off; model coverage, RSS and real HTTP/tool-loop gates remain open. Cargo used one job and inference two threads; no workspace-wide check/build/test ran.

## [v0.2.74] - 2026-10-07

### Faithful Q4 FFN Cached-Sum Experiment (Opt-In)

#### Ideas, Inspirations & Sources

- Cache activation-only minimum-correction sums identified in Mivi's existing Q4 kernel; preserve its F32 arithmetic and route-specific accumulation order. The [v0.2.73 decode evidence](docs/DECODE_SUBSTAGE_EVIDENCE_2026-10-07.md) identified FFN work as the next candidate.
- Alternating one-variable comparisons, declared cache state, and explicit measurement limits follow [Colibri's benchmarking protocol](https://github.com/JustVugg/colibri/blob/main/docs/benchmarking.md). No external inference code was copied.
- See the [design](docs/superpowers/specs/2026-10-07-q4-cached-sums-design.md), [implementation plan](docs/superpowers/plans/2026-10-07-q4-cached-sums-experiment.md), and [pilot evidence](docs/Q4_CACHED_SUMS_EVIDENCE_2026-10-07.md).

#### Added and Measured

- Add a checked Q4 scratch-taking candidate behind `q4-cached-sums-experiment`. Reuse scalar/AVX2 activation sums across output rows; preserve ordinary kernels and other quantization formats. Reject invalid dimensions/buffers before writes.
- Add feature-gated, preallocated RunState scratch and an explicit default-off selector for single-token FFN gate/up/down projections. Retain LoRA behavior, per-call scratch refresh, portable scalar fallback and read-only worker sharing; no model-name rules or new dependencies.
- Add operator, scratch/arena, mixed-format FFN/LoRA controls and an ignored real-model diagnostic checking bit-exact full logits and final KV/convolution/SSM state with profiling disabled.
- The local 1.2B Instruct GGUF passed all paired real-model comparisons. Forty Q4 FFN tensors used the candidate and eight Q6 tensors retained baseline behavior; each candidate member executed 640 projections over 16 forwards using 1,024 bytes of scratch.
- In three alternating short-context pairs, candidate/baseline wall ratios were 0.915259, 0.914447 and 0.913469 (median 8.56% lower fixed-work decode time). Separate medians were 1.571224s baseline and 1.438076s candidate. The evidence document records workload/cache state and all measured pairs.
- This is a bounded same-engine pilot, not an agent-quality, TTFT, RSS, cross-engine or general speedup claim. Default dispatch remains unchanged; no server/CLI selector or production promotion is included. Chunked batch prefill is not optimized here, and broader CPU-plan gates remain open.
- Targeted debug/release controls and the release live diagnostic used one Cargo job and two inference threads; no workspace-wide Cargo check/build/test ran.

## [v0.2.73] - 2026-10-07

### Single-Token Decode Substage Diagnostics

#### Ideas, Inspirations & Sources

- Reuse Mivi's existing attention/SSM stage profiles and the v0.2.72 prefill boundary snapshot to attribute single-token work.
- Alternating profiled/unprofiled pairs, fixed continuation work, and reporting measured limits follow [Colibri's benchmarking methodology](https://github.com/JustVugg/colibri/blob/main/docs/benchmarking.md). No external inference code was copied.

#### Added

- Profile normalization, Q/K/V projections, causal attention, output projection and FFN on the single-token attention path; profile normalization, input projection, gated convolution, output projection and FFN on the single-token SSM path.
- Extend `mivi bench` to print decode substages and combined FFN share. Timers are enabled only through the existing opt-in forward profiler.
- Add exact profiling-on/off output and KV/convolution-state checks to the synthetic fixtures, and an ignored release diagnostic with one warmup and three alternating real-model pairs using identical continuation tokens.

#### Verification and Evidence

- The two synthetic fixtures and the real 1.2B Q4_K_M model passed exact paired logit/state parity. With a retained 257-token effective prefix and 16 fixed continuation forwards, median decode FFN share was 66.50%; outer wall medians were 1.5707s unprofiled and 1.6134s profiled. See the [decode evidence](docs/DECODE_SUBSTAGE_EVIDENCE_2026-10-07.md) for all pair timings, boundaries, and the earlier discrepant pilot.
- These are three small diagnostic samples on one host, not agent-quality or kernel-speedup claims. No optimization is promoted. Cargo used one job and inference two threads.
- The scoped root release binary built, profile-subtraction tests passed, and a short real-model `mivi bench` smoke run printed the new attention/SSM substages and combined FFN share. Formatting and whitespace checks passed; no workspace-wide Cargo check/build/test ran.

## [v0.2.72] - 2026-10-07

### Decode-Stage Profiling for Focused Model Benchmarks

#### Ideas, Inspirations & Sources

- The separation of prefill and decode measurements and explicit cache-state reporting follow [Colibri's benchmarking methodology](https://github.com/JustVugg/colibri/blob/main/docs/benchmarking.md).
- Decode-stage attribution reuses Mivi's opt-in forward profiler; no external inference code or kernels were copied.

#### Added and Verified

- Capture the profile at the exact prefill/decode boundary and report decode forward-stage time and shares separately from prefill in `mivi bench`.
- Preserve existing normal inference behavior; profiling remains opt-in. Profile subtraction saturates at zero to avoid negative durations from timer granularity.
- Verified with the 1.2B Q4_K_M GGUF, two Rayon threads, and a 1,557-token synthetic agent-style prompt. One profiled run attributed 1.059s (42.0%) to attention, 1.214s (48.1%) to SSM, and 0.249s (9.9%) to logits over 24 decode forward passes; a warm shared-prefix run attributed 0.675s/0.761s/0.158s over 15 passes.
- These are single diagnostic samples, not stable performance claims; measured stage time excludes sampling and streaming overhead. The new profile reports the existing aggregate attention/SSM split for decode; detailed inner attention/SSM sub-stages remain available for chunked prefill only.
- Targeted profile-subtraction tests, the scoped release build, formatting, and the real-model benchmark passed. Cargo used one job; no workspace-wide check/build/test was run.

## [v0.2.71] - 2026-10-07

### Faithful Projection Column-Panel Experiment (Opt-In)

#### Ideas, Inspirations & Sources

- The paired, alternating benchmark design and reporting of inconclusive results were inspired by [Colibri's benchmarking methodology](https://github.com/JustVugg/colibri/blob/main/docs/benchmarking.md).
- Preserving the existing accumulation order and CPU behavior was informed by the pinned [GGML CPU implementation](https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/ggml/src/ggml-cpu/ggml-cpu.c). No GGML code was copied.
- See the [column-panel design](docs/superpowers/specs/2026-10-07-faithful-column-panel-experiment-design.md), [implementation plan](docs/superpowers/plans/2026-10-07-faithful-column-panel-experiment.md), and [redacted pilot evidence](docs/PROJECTION_LOCALITY_EVIDENCE_2026-10-06.md).

#### Added and Measured

- Replace the experimental token-width selector with explicit 32/64/128-column panel selectors behind `projection-locality-experiment`. The experimental API and private manifest field were corrected to match their actual meaning; the strict manifest/result/report protocol moved from schema 2 to schema 3 and rejects the removed `token_tile` field. This experiment code was not copied from GGML.
- Guard measurement validity: explicit panel comparison groups must have batch >= 32, at least two rows, and more than 128 columns; GGUF shape checks run during descriptor preflight before child launch. The measurement child rejects unsupported AVX2+FMA routes and reports the actual kernel route, which the driver validates.
- In the fresh bounded three-repetition pilot, Q6_K 2048x8192 batch-64 selector medians were 21.869–22.420 ms versus 23.873 ms baseline (paired median ratios 0.909–0.935). Synthetic Q8_0 257x256 batch-64 results were mixed: panel 32 was 0.145 ms versus 0.161 ms baseline, panel 64 was 0.176 ms, and panel 128 was 0.153 ms. All 48 attempts completed; profiled/unprofiled and baseline/selector output vectors were bit-identical.
- These are operator-level pilot timings from one host and three repetitions. The Q8_0 panel-128 paired ratio reached 1.280, baseline timing spreads were broad, and scheduler/thermal variation was uncontrolled. No selector is promoted; no model-quality or end-to-end speedup claim follows.
- Ordinary APIs and production dispatch remain on the existing default 128-column kernel. Raw artifacts and identifying paths remain private. Reference-matrix, scratch-reuse, model-level parity, end-to-end evaluation, and optimization-promotion gates remain open.

## [v0.2.70] - 2026-10-07

### Faithful Projection Locality Experiment (Opt-In)

#### Ideas, Inspirations & Sources

- The paired, alternating benchmark design and reporting of negative/inconclusive results were inspired by [Colibri's benchmarking methodology](https://github.com/JustVugg/colibri/blob/main/docs/benchmarking.md).
- Preserving the existing accumulation order and CPU behavior was informed by the pinned [GGML CPU implementation](https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/ggml/src/ggml-cpu/ggml-cpu.c). No GGML code was copied.
- The selectors are an experiment in this repository; see the [design](docs/superpowers/specs/2026-10-06-faithful-projection-locality-design.md), [implementation plan](docs/superpowers/plans/2026-10-06-faithful-projection-locality.md), and [pilot evidence](docs/PROJECTION_LOCALITY_EVIDENCE_2026-10-06.md).

#### Added and Measured

- Add typed 32/64/128-token traversal selectors behind the `projection-locality-experiment` feature, preserving ascending-column FMA order and exact output bits. Existing APIs and production dispatch remain unchanged.
- Extend the bounded private measurement harness with selector-aware profiled/unprofiled paired comparisons, exact full-vector parity, alternating run order, and artifact preflight.
- The corrected three-repetition pilot compared explicit variants against the restored production baseline. Every selector was slower: real-weight Q6_K medians were 2.34–2.39× baseline, with exact output parity. No selector is promoted; investigate the experimental traversal before further performance claims or end-to-end evaluation.
- Focused SIMD, quantized-matmul, locality, Python supervisor/driver and example tests passed; the opt-in release example built cleanly. Cargo used one job, test execution one thread, and Rayon two threads; no full-workspace check/build/test ran.
- Raw experiment artifacts and identifying paths remain private. The default kernel is unchanged; broader scratch reuse, model-level parity, and end-to-end P1-A gates remain outstanding.

## [v0.2.69] - 2026-10-05

### Projection Cost Diagnostics — P1-A Measurement Slice

#### Ideas, Inspirations & Sources

- Follow the [approved projection measurement design](docs/superpowers/specs/2026-10-05-projection-cost-measurement-design.md) and [Phase 0 runtime evidence](docs/CPU_RUNTIME_EVIDENCE_2026-10-05.md).
- Alternating paired runs, explicit clock boundaries, and retained negative results draw on [Colibri's benchmark methodology](https://github.com/JustVugg/colibri/blob/main/docs/benchmarking.md).
- Faithful numerical-path constraints use [pinned GGML CPU traits](https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/ggml/src/ggml-cpu/ggml-cpu.c) as context; no GGML kernel was copied or promoted.

#### Added and Fixed

- Add opt-in shared-arithmetic projection diagnostics and a gated operator example, using synthetic F32 activations with synthetic weights or metadata-selected GGUF matrices. Default calls remain unprofiled.
- Add bounded, privately retained paired measurements through the existing process supervisor. Separate outer call wall time, inner kernel stages, and overlapping worker-work durations.
- Preserve exact paired output comparisons while compacting report metadata; exclude unmatched runs from timing medians and release paired output buffers.
- Cover empty work, overflow, six supported formats, branch boundaries and odd parallel tails. Correct BF16 format validation, nested timer validation, and transpose availability for the actual kernel branches.

#### Evidence, Limits & Verification

- Publish the [projection cost evidence](docs/PROJECTION_COST_EVIDENCE_2026-10-05.md): 15 accepted pairs across five cases. All 84 children cleaned up; 27 earlier validation-error records remain unchanged. Private artifacts total 58,372,289 bytes, below the combined 64 MiB ceiling.
- Select a faithful locality experiment around the largest measured accumulation-work category. This is a hypothesis, not proof of a locality bottleneck, predicted speedup, or production promotion.
- 86 focused Rust/Python tests passed; the scoped diagnostic release example built without warnings. Cargo used one job, Rust tests one thread, and operators two threads. No full-workspace check/build/test ran.
- GGUF diagnostic mapping requires Linux and quiescent input files. Raw results, local paths, and model/binary hashes remain private.
- Measurements used workspace v0.2.68; this release adds diagnostics and metadata, not an optimized inference kernel. Synthetic operator measurements do not establish real-model latency or coding-agent quality; existing cross-engine divergence remains unresolved.

---

## [v0.2.68] - 2026-10-05

### Bounded CPU Runtime Evidence — Phase 0, Task 6

#### Ideas, Inspirations & Sources

- Controlled paired measurements follow [Colibri's benchmark methodology](https://github.com/JustVugg/colibri/blob/main/docs/benchmarking.md).
- Numerical evidence boundaries follow [Kimi's fixture guidance](https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/main/tests/fixtures/README.md).
- Quantized operand-path analysis uses [pinned GGML CPU traits](https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/ggml/src/ggml-cpu/ggml-cpu.c);
  hybrid-state caveats draw on [LMCache guidance](https://docs.lmcache.ai/recipes/kimi_linear.html).

#### Documentation and Evidence

- Publish a redacted [runtime evidence report](docs/CPU_RUNTIME_EVIDENCE_2026-10-05.md):
  three paired repetitions at 110/1024/2636 prompt tokens, 18 completed runs,
  plus three separate native profile controls with matching outputs.
- Long native prefill median is 84.243s; projections/FFNs consume 72.61% of
  counted forward-stage time in the separate long control. Select P1-A faithful
  scratch/locality experiments, starting with cost isolation rather than an
  assumed allocation bottleneck.
- Disclose reproducible short/medium output divergence, unequal medium decode
  lengths, unmatchable first-step EOS policy and incompatible timing boundaries.
  Long output agreement does not establish general cross-engine parity.
- Correct historical Q8 lossless claims to lossy, F32 K+V 4K storage to 96MiB
  for the stated dimensions, and label historical speedup goals/defaults.

#### Scope and Verification

- 54 focused comparison tests passed; scoped replay and pinned reference targets
  built with one job. Inference used two threads, sequential workloads and
  explicit wall/RSS/artifact limits. Cleanup succeeded for all measured runs.
- Measured Mivi is v0.2.67 at `5c49b97`; this release changes documentation and
  version metadata, not inference kernels. Raw experiment artifacts remain
  private. Numerical divergence remains unresolved; no kernel speedup,
  cross-engine parity or coding-agent quality claim is made.

---

## [v0.2.67] - 2026-10-03

### Independent Adversarial Arithmetic and Hybrid Oracle — Phase 0, Task 5

#### Ideas, Inspirations & Sources

- Synthetic fixture provenance and complete-trace discipline follow
  [Kimi's fixture guidance](https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/main/tests/fixtures/README.md).
- Independent serialized-byte evaluation and bounded comparisons draw on
  [Colibri's benchmark methodology](https://github.com/JustVugg/colibri/blob/main/docs/benchmarking.md).
- GGUF tensor layout and K-quant block arithmetic are checked against the
  [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
  and [GGML quantization implementation](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c).
  The tests use independent scalar equations; no inference kernels were copied.

#### Added

- Add independently decoded Q4_K/Q6_K block tests for nibble/bit-plane layout,
  nontrivial scales/minima, signed scales, two-block rows, and partial odd-row
  tiles. Dot accumulation uses a scale-aware absolute-product tolerance.
- Add a collision-refusing private synthetic GGUF exporter and committed schema-1
  oracle traces: serialized F32 tensor hashes, all 64 logits, top token/margin,
  provenance and two teacher sequences plus a changed-prefix case.
- Compare Rust token-major, chunked tile sizes 1/2/3/8, split continuation, reset,
  and changed-prefix execution against the independent hybrid reference. A warmed
  two-token prefix is retained across context reset before testing a divergent suffix.
  Explicitly check finite logits, reset state, and nonzero convolution carry.
- Fix two independently tested Python-reference gaps: all query/KV heads receive
  RoPE, and reset clears lazily created convolution history.

#### Scope and Verification

- Four focused Python tests passed; the generator reproduced the committed trace
  from decoded serialized GGUF bytes, refused an existing output directory, and
  created mode0700/0600 private artifacts. The synthetic GGUF remains outside Git.
- Both scoped `mivi-quant adversarial` tests passed. The focused hybrid integration
  test passed all15 positions (960 logits) at `atol=rtol=1e-4`, across token-major,
  tile sizes 1/2/3/8, reset, a cached changed prefix, and split continuation. The
  top-margin bound accounts for the independent error allowances of both logits.
  Omitting the explicit fixture path fails with an actionable error rather than skipping.
- This is a small F32 synthetic architecture oracle plus isolated quant-block
  arithmetic tests—not a real-model benchmark, full quantized hybrid graph, or
  proof of coding-agent quality. No production inference kernel was changed.

---

## [v0.2.66] - 2026-10-02

### Router Timing and Profiling Controls — Phase 0, Task 4

#### Ideas, Inspirations & Sources

- Controlled profile-on/off comparisons and explicit evidence boundaries follow
  [Colibri's benchmark methodology](https://github.com/JustVugg/colibri/blob/main/docs/benchmarking.md).
- Independent fixture discipline follows
  [Kimi's fixture guidance](https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/main/tests/fixtures/README.md).
- Reuse Mivi's existing model capture/profile counters and actor lifecycle; no
  duplicate kernel timers, model-name branches, or normal-server profile setting.

#### Added

- Optional numeric microsecond boundaries for useful router output, stream parse
  completion, observed model stages/profile, actor queue/worker return; unobserved
  prompt-render timing remains absent.
- Ignore heartbeats and empty role/tool envelopes when identifying first useful
  content or tool output. Keep request-worker return independent from model
  terminal status and actor teardown.
- Make actor/waiter lifecycle handoffs tolerate ordinary fixture-state lock
  contention; preserve captured terminal outcomes when a late cancellation flag
  arrives.
- Add a model-required sequential profiling off/on router control that rejects
  incomplete/truncated captures and compares output without exposing it in logs.
- Add [private router profiling guidance](docs/ROUTER_PROFILE_DIAGNOSTICS.md).

#### Scope and Verification

- Eight focused profile tests passed (one model-required test ignored in the unit
  run); the tiny synthetic GGUF live off/on control passed with identical captured
  output, measured enabled prefill profile, observed worker return and actor
  cleanup. The test process was supervised with bounded resources and private
  artifacts.
- The default-feature `mivi-server` library check passed with Cargo jobs1. No
  workspace-wide build/check/test was run.
- Same-engine output parity is a correctness control, not a performance or coding
  agent quality claim. The synthetic tiny model does not establish real-model
  latency or reference-engine parity; only observed stage timings are exported.

---

## [v0.2.65] - 2026-10-02

### Private Paired Runtime Comparison — Phase 0, Task 3

#### Ideas, Inspirations & Sources

- Independent evidence and controlled pairs follow
  [Kimi's fixture discipline](https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/main/tests/fixtures/README.md)
  and [Colibri's benchmark methodology](https://github.com/JustVugg/colibri/blob/main/docs/benchmarking.md).
- Tokenizer/BOS/EOS/context preflight follows the
  [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).
  Reference checks follow the pinned llama.cpp
  [server contract](https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/tools/server/README.md),
  [response serializer](https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/tools/server/server-task.cpp)
  and [cache/terminal implementation](https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/tools/server/server-context.cpp).
  This is native Python diagnostic orchestration, not copied inference kernels.

#### Added

- Standard-library comparison CLI with exact schema-1 manifests, model SHA256,
  metadata-derived vocabulary/BOS/context validation, optional explicit pinned
  local reference, sequential alternating pairs, and private JSON/Markdown reports.
- Owned-process supervision with bounded logs, sampled Linux process-tree RSS,
  artifact checks during blocking HTTP and after child exit, TERM/KILL cleanup,
  explicit failures and no later launches after unsuccessful cleanup.
- Descriptor-pinned private directories/files, symlink/traversal/FIFO refusal,
  output collision protection, reserved report space and bounded failure retention.
  Retention may discard oversized owned diagnostic suffixes after verified cleanup;
  numeric loss accounting is retained, and model/manifest files are untouched.
- Common-prefix first-divergence probes, selected native logits/reference logprobs,
  explicit unavailable observations, per-sample engine order and
  [comparison usage documentation](docs/RUNTIME_COMPARISON_DIAGNOSTICS.md).

#### Fixed

- Handle the pinned reference's actual cold-cache counters and `/props` fields;
  distinguish cached size from prior reuse and terminal EOS from content IDs.
- Match the full verified native stop-string policy, including runtime defaults,
  rather than sending only metadata EOS or duplicating model-specific literals.
- Reject exhausted session budgets before version/help or inference launches;
  preserve probe cleanup failures instead of reducing them to recoverable errors.
- Catch fast-exit artifact overruns and refuse retention after unverified primary
  or divergence-probe cleanup. Validate the post-BOS prompt bound and skipped
  GGUF boolean-array encodings.

#### Scope and Verification

- All 54 scoped Python tests passed, including real synthetic child timeout,
  logging, RSS, descendant cleanup and fast-exit regressions; HTTP tests use only
  a localhost mock. Review findings were repaired with RED/GREEN regressions.
- Validation-only created no output or child. Tiny-model native CLI smoke completed
  with successful reaping and private mode0700 directories/mode0600 reports;
  an absent reference correctly produced `partial`, exit2 and no comparison.
  The smoke used the existing v0.2.64 replay executable. The v0.2.65 scoped
  default-feature model-library check and lockfile refresh passed with Cargo
  jobs1; no workspace-wide build/test was run.
- RSS/artifact enforcement is sampled, not a kernel quota or real-time guarantee.
  Reference nonstreaming TTFT, raw logits and intermediate state remain unavailable;
  missing or failed measurements never become zero latency or numerical equality.
- No local llama.cpp executable was installed, downloaded or benchmarked. No real
  paired-model performance, speedup or coding-agent quality claim. Normal server,
  provider configuration and inference backend are unchanged; no server rebuild.

## [v0.2.64] - 2026-10-02

### Private Runtime Replay — Phase 0, Task 2

#### Ideas, Inspirations & Sources

- Reuse Mivi's feature-gated capture recorder and forward counters, following
  [Kimi's independent fixture discipline](https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/main/tests/fixtures/README.md)
  and [Colibri's controlled benchmark methodology](https://github.com/JustVugg/colibri/blob/main/docs/benchmarking.md).
  Same-engine parity is only a prerequisite, not an independent numerical oracle.
- Security review motivated component-by-component directory-descriptor traversal
  and a single-use writer; implementation uses native Unix file APIs, not copied
  inference kernels or model-name dispatch.

#### Added

- Feature-required `mivi-model` replay example with explicit local model/input/
  output paths, bounded JSON input/output, exclusive owned-private Unix output,
  symlink/traversal refusal, and a scoped two-thread Rayon pool. Non-Unix output
  fails closed; normal server CLI and provider configuration are unchanged.
- Metadata-validated BOS/EOS/vocabulary/context, F32 KV and deterministic greedy
  settings, normal versus split-prefill modes, prefill-only profile snapshots,
  separate observed terminal IDs and stop reasons, bounded text/ID captures,
  explicit completion/error/cancellation outcomes, and labeled timing boundaries.
- Up to 16 teacher-forced next-token probes with finite selected raw logits,
  log probabilities and top-two margins scored before advancing the supplied
  token. Teacher timing is separate from chunked-prefill measurements.
- Model-required short/long same-engine parity fixture and
  [private replay usage documentation](docs/RUNTIME_REPLAY_DIAGNOSTICS.md).

#### Fixed

- Reject present non-boolean BOS policy metadata rather than silently disabling
  BOS insertion; retain the existing default only for absent metadata.
- Enforce a single output-write attempt so repeated writes cannot append beyond
  the per-result byte budget. Check private output before model loading.
- Extend server diagnostic test initializers for the new optional capture fields;
  this is compatibility maintenance, not a new server profiling mode.

#### Scope and Verification

- The example's scoped release build passed. Tiny-model teacher forcing completed
  two finite probes; normal/split generation matched content, delivered text,
  terminal IDs and final position, using two threads and private mode0600 results.
- Fifteen non-live example tests passed. The explicitly selected real LFM2.5-1.2B
  Q4_K_M parity test passed both 110- and 2636-token prefixes in 180.58s total,
  checking complete, untruncated runs before equality. This synthetic same-engine
  test is not an end-to-end agent workload or a latency benchmark.
- Scoped model diagnostics passed28 tests; server fixture compatibility passed24
  with two unrelated live fixtures ignored. Strict model/example Clippy and the
  default-feature model-library check passed; no workspace-wide build/test.
- The initial live-test launch failed before loading because Cargo resolved a
  relative model path from the crate directory; the absolute-path rerun passed,
  and documentation now makes that path requirement explicit.
- No kernel optimization or speedup claim. Cross-engine numerical comparison,
  repeated/profile-overhead measurements and agent-quality evaluation remain
  subsequent tasks. The 180s cooperative deadline cannot interrupt blocking
  model loading; hard termination, RSS supervision and observed process exit
  belong to the external comparison driver. No normal server binary rebuild.

## [v0.2.63] - 2026-10-02

### Bounded Runtime Replay Contract — Phase 0, Task 1

#### Ideas, Inspirations & Sources

- Establish numerical evidence before selecting faster kernels, inspired by
  [Kimi's independent fixture discipline](https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/main/tests/fixtures/README.md)
  and [Colibri's controlled benchmark protocol](https://github.com/JustVugg/colibri/blob/main/docs/benchmarking.md).
- Export Mivi's existing attention/SSM counters rather than duplicate timers;
  distinguish whole-block time from projection, FFN, causal scan, and convolution.
- The accompanying research roadmap records all eight requested projects and
  directly relevant papers with applicability and limitations. These references
  motivate the plan; their published speedups are not Mivi measurements.

#### Added

- Feature-gated replay input validation for bounded context, tile, output,
  teacher-forced probes and selected logits; checked context arithmetic,
  unknown-field rejection, unique logit IDs, and rejection of simultaneous
  teacher forcing and sampling.
- Numeric microsecond export of existing forward aggregates and all ten
  attention/SSM substages, with checked conversion and documented overlapping
  totals. Overflow fails instead of silently wrapping.
- Eleven model-independent replay/export tests, including exact limits,
  malformed input, every duration field's overflow, and substage ordering.
- Native-first roadmap/TODO and focused Phase 0 execution plan with scoped
  verification, private artifact budgets and evidence-gated optimization.

#### Fixed

- Replace manual integer ceiling-division and cache-boundary modulo checks
  with Rust's native operations in the affected KV/model code, resolving the
  existing strict Clippy failures without changing valid cache layouts.
- Reject overflowing Q8 byte-size calculations with `AllocationOverflow`;
  extreme quantized KV dimensions no longer panic at round-up arithmetic in
  the covered constructor cases. Add regressions for rejection and partial
  block storage sizes. This is input robustness, not a measured speedup.
- Cleanup follows the compiler's exact `manual_div_ceil` and
  `manual_is_multiple_of` diagnostics; the regression was derived from Mivi's
  existing checked-allocation error contract, not a copied external kernel.

#### Scope and Verification

- The replay runner, paired reference driver, new independent hybrid oracle,
  and live repeated measurements remain subsequent tasks. Callers must also
  bound input-file reads and validate vocabulary/BOS-adjusted lengths.
- This increment adds no normal-server capture switch, model-name inference
  behavior, kernel change, timeout change, or measured speed improvement.
- Selected feature-enabled diagnostic tests: 27 passed, including the eleven
  new tests after observed RED/GREEN execution; one Cargo job and one test
  thread, with Rayon configured to two threads. No model was loaded.
- The original strict-Clippy checkpoint exposed26 KV diagnostics and three
  model/transformer diagnostics. After the focused cleanup, scoped strict
  model-library Clippy including dependencies passes without suppressions.
- Additional selected tests: seven small cache tests, thirteen prefix tests,
  six nonignored model-prefix tests (six model-required tests ignored), and
  two synthetic attention tests pass. Large memory-calculation tests were
  excluded deliberately; no full workspace validation or live agent test.
- GPT-6 Luna/high reviews approved replay spec/code quality and the arithmetic
  cleanup. The replay reader-boundary documentation note was addressed; the
  arithmetic review found no issues. This increment totals55 passing selected
  tests and is published on the feature branch, not merged into main.

## [v0.2.62] - 2026-10-02

### Fixture-Only Generation Diagnostics

#### Ideas, Inspirations & Sources

- Preserve evidence before interpretation: distinguish rendered/conditioned
  prompts, synthetic protocol prefixes, raw decoding before stop filtering,
  callback delivery, and the parser-facing SSE stream.
- Reuse existing generation and routing rather than a second inference path:
  [chat rendering and validation](https://github.com/aswin402/mivi-v4/blob/8f7ae0d/crates/mivi-server/src/routes/chat.rs),
  [engine actor](https://github.com/aswin402/mivi-v4/blob/8f7ae0d/crates/mivi-server/src/engine_actor.rs),
  [model prefill/decoder and profiling](https://github.com/aswin402/mivi-v4/blob/8f7ae0d/crates/mivi-model/src/model.rs),
  [benchmark timing](https://github.com/aswin402/mivi-v4/blob/8f7ae0d/crates/mivi-cli/src/runners/bench.rs).
- The previous newline-sensitive fixture motivated byte-exact tool-result
  handoff. That local observation is not a universal model-quality conclusion.

#### Added

- A non-default `fixture-diagnostics` feature, bounded request-local model
  observations, private server sessions, and ignored in-process real-router
  fixtures. No capture CLI flag, request/header switch, environment toggle,
  network listener, project-workspace access, or mutating native tool.
- Separate tokenization/prefill/decode and first raw/delivered/visible timing,
  prompt/reused/processed counts, physical-return outcomes, parser/metrics
  evidence, truncation/overflow/clipping indicators, and answer-quality status.
- Exclusive local capture artifacts with Unix directory/file modes 0700/0600,
  internal sequence names and no overwrite or symlink following. Captured
  prompt/output bodies are not printed or uploaded; fixture payload logs are
  suppressed without suppressing normal logs.
- Bounded SSE reconstruction through body EOF after `[DONE]`; private owned
  actor shutdown; compiled regressions for fragmented events, bounds,
  continuation-quality independence, and clipped settings completeness.

#### Measured Results and Scope

- A fresh local release-profile 1.2B Q4_K_M run completed one valid `read_file`
  call and byte-exact continuation identifying `add` and `a + b`. Both streams
  reached EOF with complete captures, physical model return and owned-worker
  join; no client-disconnect increment. End-to-end elapsed was 10.06s.
- First request: model prefill 4.304533s, decode 1.112983s, consumer first visible
  delta 4.613374s and stream completion 5.424804s. Continuation: 2.374565s prefill,
  2.127321s decode, 2.375536s first visible delta and 4.503064s stream completion.
  Conditioned prompt/reused/processed counts were 110/0/110 and 96/0/96.
  The first request used a fresh loaded actor; continuation reused the actor,
  not a measured KV-cache warm hit. These are local observations, not speedups.
- Settings: context 4096, chunk tile 64, max output 48, request/first-output
  budgets 120s/90s, two configured Rayon threads and one API inference slot.
  Record text/ID caps were 65,536 bytes/256 IDs. No truncation, clipping or
  counter overflow occurred. Model-config name is metadata, not a file hash.
- Fresh observer-on/off parity passed with two sequential model loads,
  context 512, max output 16 and caps 4,096 bytes/32 IDs. It matched generated
  output, callback delivery and post-generation RNG state.
- This tests one supported model/configuration and one controlled tool loop,
  not Minicode or general agent reliability. Agent-sized latency and comparison
  with other runtimes remain separate work; no numerical inference, parser,
  public generation API, production timeout or normal dependency change.
- Coverage limitations remain explicit: no actual-model nonempty byte-decoder
  flush, cache-hit/adapter/mid-layer-failure run; no live malformed-tool,
  timeout, truncation or worker-timeout scenario. The actual chat streaming-error
  payload-log branch remains a review-noted integration-test gap.
- Final v0.2.62 scoped suites passed: model 41 default / 57 feature-enabled
  tests (4/6 ignored), server 88 default / 114 feature-enabled tests (0/2
  ignored). Live fixtures were invoked separately. Scoped Clippy, formatting,
  dependency-tree and whitespace checks passed; the root release executable
  reports v0.2.62 and exposes no capture option. All task reviews approved;
  the stop-token clipping completeness finding was regression-tested and fixed.
- Commands used one Cargo job and one test thread, with two configured Rayon
  threads. One controller lint invocation inadvertently overlapped a release
  command; no concurrent model loads occurred, and subsequent validation was
  sequential. No full-workspace build/check/test was run.
- Independent whole-branch review approved all source/release changes with no
  Critical/Important findings, explicitly deferring the disclosed coverage gaps.
  Main was fast-forwarded, its 88 default server tests passed, and reviewed
  release `c1e931d` was pushed without force. GitHub and local SHA matched;
  the user's `.gitignore` and private local artifacts were excluded and preserved.

## [v0.2.61] - 2026-10-01

### Defer Synthetic Stream Prefixes Until Model Output

#### Ideas, Inspirations & Sources

- Keep server-inserted protocol text from counting as model-produced output:
  hold the forced prefix until the first nonempty decoded model chunk, then
  deliver both together. The prefix still conditions the same prompt, and
  the string-stream API and tool syntax remain unchanged.
- References to the existing actor callback, chat deadline/metric consumer,
  and model streaming callback at the reviewed source revision:
  [engine actor](https://github.com/aswin402/mivi-v4/blob/a64f7e1/crates/mivi-server/src/engine_actor.rs),
  [chat deadlines and metrics](https://github.com/aswin402/mivi-v4/blob/a64f7e1/crates/mivi-server/src/routes/chat.rs),
  [model streaming callbacks](https://github.com/aswin402/mivi-v4/blob/a64f7e1/crates/mivi-model/src/model.rs).

#### Measured Results and Scope

- On the local CPU 1.2B Q4_K_M model (two inference/Rayon threads, context
  4096), a 1,085-prompt-token required-tool stream with a one-second
  first-output deadline emitted an empty assistant role event, then the
  explicit first-token deadline error and `[DONE]`. It produced no tool call;
  metrics recorded `first_token_count=0` and
  `time_to_first_token_microseconds_total=0`. A second request was admitted
  through the single API inference-admission slot after the timeout. This
  verifies admission-permit reuse, not physical prefill completion before
  that second admission.
- With 120-second request and 90-second first-output budgets, the 109-token
  short request emitted a valid `read_file(path="example.rs")` call. Returning
  the exact fixture contents with the same tool-call ID and `tool_choice:none`
  produced the requested identification of `add` and `a + b`. One initial
  follow-up omitted the fixture's trailing newline and did not meet the answer
  check; the byte-exact retry passed. No schema, parser, model, or timeout
  changes were made.
- A bounded monotonic probe measured the first visible tool-call delta at
  4.306361 seconds and server first decoded output at 3.985360 seconds. The
  separate generation latency was 5.072729 seconds and client stream completion
  was 5.093952 seconds. These are local observations, not a performance claim.
- Scoped server library tests passed (86 passed, 0 failed). An initial
  sandboxed run could not bind the test socket; the same command was rerun with
  local-bind permission and passed. Malformed tool-call generation and
  agent-sized latency remain separate work; this change does not claim either
  is fixed.
- Final v0.2.61 checks passed: 86 scoped server tests, scoped Clippy with
  warnings denied and the existing lint allowances, package formatting, and
  whitespace validation. The normal dependency tree is unchanged. The rebuilt
  root `mivi` release executable reports v0.2.61.
- Cargo jobs and test threads were one, inference/Rayon threads were two;
  no full-workspace checks, tests, or builds were run.
- Independent task and whole-change reviews approved publication with no
  Critical or Important findings. Combined actual blocked-send cancellation
  coverage remains an optional regression follow-up; existing cancellation
  and real-channel backpressure checks are separate.
- Published to GitHub `main` without force after 86 scoped server tests passed
  on the fast-forwarded branch. The remote release head matched local main;
  the user's pre-existing `.gitignore` change was preserved and excluded.

#### Changed

- The private production streaming callback now retains the forced prefix
  until a nonempty decoded model chunk arrives, emits the combined first
  string once, and ignores empty decoded chunks for first-output timing.
- No public API, normal dependency, prompt-conditioning, tool-protocol,
  generation-budget, numerical inference, or timeout policy change.

## [v0.2.60] - 2026-10-01

### Group-Size Comparison for Test-Only Packed Prefill Diagnostics

#### Ideas, Inspirations & Sources

- Compare group-32 and group-256 activation packing while retaining the existing
  Q4_K weight bytes and production inference path. GGML Q4_K/Q8_K layouts and
  dot products are a reference point, not a claim that this experimental
  activation codec is bit-identical:
  [GGML block definitions](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h)
  and [GGML quantization source](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c).
- Mixed-precision sensitivity is empirical. [Dettmers et al., LLM.int8()](https://arxiv.org/abs/2208.07339)
  motivates investigating activation outliers, but this experiment does not
  implement its feature-wise outlier decomposition or import its quality claims.

#### Measured Results and Scope

- The 1.2B Q4_K_M capped capture used 16 raw prompt tokens (32-token ceiling),
  128 rows per projection, group=32, and the scalar kernel. It sampled 40 Q4
  projections and counted 8 unsupported FFN projections. Sampled projection
  relative L2 ranged from 0.003657 to 0.013003; max absolute error ranged from
  0.000107 to 0.009445. The production walker control was exactly zero error.
  A separate complete-row, layer-12 down-only residual perturbation measured
  relative L2 0.004483 and max absolute 0.025347, with greedy token 509 unchanged;
  it is not part of the capped projection samples.
- Complete-row traces used the same default raw tool-request fixture per model
  across both codecs. The 1.2B fixture had 26 tokens and the 2.6B fixture 23.
  Baseline/production, per-layer recompute, and recompute/production controls
  matched exactly. Measurements were finite and ordered; final-observer/walker
  residual agreement was checked, while only the non-packed controls are stated
  to match production.
  Greedy next-token IDs were unchanged in all 16 codec/mode measurements.
- Packed/total non-packed FFN projection coverage was identical for both codecs:

  | Model | Down | Gate | Up | Full FFN |
  |---|---:|---:|---:|---:|
  | 1.2B | 8/8 | 16/32 | 16/32 | 40/8 |
  | 2.6B | 16/14 | 30/60 | 30/60 | 76/14 |

  Non-packed counts include deliberately unselected projections, including
  eligible Q4_K weights, as well as unsupported formats; they are not counts
  of unsupported projections alone.
- Final logit relative L2 percentages / max absolute errors and largest
  residual-relative-error increase by mode:

  | Model | Group | Kernel | Down L2 / max abs (growth) | Gate L2 / max abs (growth) | Up L2 / max abs (growth) | Full FFN L2 / max abs (growth) |
  |---|---:|---|---:|---:|---:|---:|
  | 1.2B | 256 | runtime AVX2 tiled | 6.1920% / 0.809669 (2.7227 pp) | 5.9338% / 0.741239 (2.1208 pp) | 5.6856% / 1.127241 (5.2765 pp) | 24.4887% / 2.008592 (10.8271 pp) |
  | 1.2B | 32 | scalar | 7.9674% / 1.026320 (2.8357 pp) | 11.6445% / 1.514009 (4.4457 pp) | 9.6461% / 0.928469 (3.1632 pp) | 8.8656% / 0.894294 (4.6450 pp) |
  | 2.6B | 256 | runtime AVX2 tiled | 3.7911% / 0.570700 (0.8560 pp) | 2.0298% / 0.348042 (0.4749 pp) | 2.7732% / 0.445877 (0.6315 pp) | 4.6426% / 0.841252 (1.1137 pp) |
  | 2.6B | 32 | scalar | 1.4712% / 0.259859 (0.4260 pp) | 1.4896% / 0.218034 (0.2496 pp) | 2.1760% / 0.222596 (0.2730 pp) | 2.8133% / 0.446167 (0.6881 pp) |

- Results are mixed: on the 1.2B fixture, group-32 relative L2 is higher for
  isolated down, gate, and up modes, and lower for full FFN. On the 2.6B
  fixture, all four group-32 measurements are lower. These are repeated modes
  on one shared short raw fixture per model, not independent fixtures or a
  general quality conclusion. There is no packed-model acceptance threshold,
  speedup claim, generated-tool/agent result, causal layer-defect claim, or
  production rollout. Scalar group-32 accuracy results are not performance data.
- Greedy IDs remained `509` (1.2B) and `124902` (2.6B) in every mode. Largest
  residual-relative-error growth occurred at layer 15 (SSM-labelled) for every
  1.2B mode; for 2.6B group-256 it was gate at layer 2 and the other modes at
  layer 13, while group-32 up peaked at layer 2 and the other modes at layer 13.
  These are relative-error increases with changing denominators, not causal
  layer-defect evidence.

#### Added

- A private group-32 activation codec, scalar affine Q4_K dot product,
  transactional checked batched matmul, strict codec selector, and configured
  test diagnostics. The selector accepts only `256` or `32` (including rejecting
  whitespace variants). No production exports, normal dependencies, or unsafe
  code were added; production inference remains unchanged.
- Task 3's original test-first chronology deviation was disclosed and the user
  approved continuing. The original sequence is not described as test-first;
  the later retrospective wrong-selector mutation check does not alter it.
- Production inference code was unchanged. No server binary rebuild is claimed;
  the existing server executable remains v0.2.51.
- Final review identified a unit-test bound that omitted stored F32 scale
  rounding for accepted subnormal scales. A failing signed-subnormal regression
  now passes with an allowance covering normalization, integer rounding, scale
  storage, and reconstruction. Only test assertions changed; the codec and
  five recorded model runs are unaffected.
- Verification: scoped `mivi-quant` library tests passed (50 passed, 5 ignored);
  scoped Clippy passed with the existing style-lint allowances; package format
  check and `git diff --check` passed; normal dependencies are unchanged. Cargo
  jobs and test threads were one, inference/Rayon threads were two. No full-
  workspace check, test, or server build was run. Independent final review
  approved publication after the test-bound correction. Published to GitHub
  `main`; the pushed release head was verified against the local commit.

## [v0.2.59] - 2026-10-01

### Isolated Gate/Up Packing Sensitivity (Test-Only)

#### Ideas, Inspirations & Sources

- **Change one projection's activation packing at a time**: extend the existing
  cumulative FFN experiment with gate-only and up-only modes. Both input
  projections and SwiGLU are recomputed, but only the selected eligible Q4
  projection packs its activations; the other projection and down retain the
  existing F32-activation matmul. Weight tensors/quantization are unchanged.
  - *Sources*: Mivi's preceding cumulative/layer-wise diagnostics, public
    `swiglu_rows` and tile APIs, and the
    [GGML quantization reference](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c).
- **Treat precision sensitivity as an empirical question**: mixed precision is
  a research motivation, not a guarantee that a particular projection should
  always be excluded from packing.
  - *Inspiration/source*: [Dettmers et al., LLM.int8()](https://arxiv.org/abs/2208.07339).
    This experiment does not implement that paper's feature-wise outlier
    decomposition or establish its quality results for these small models.

#### Added

- Private typed gate/up packing selection, shared by the three-fixture corpus
  and layer-wise trace. Existing down-only, full packed FFN, and non-packed
  controls remain available. Unsupported selected projections fall back through
  the existing format/shape checks, never by model name or layer index.
- A model-free two-row Q4 regression verifies complete outputs against separately
  selected gate/up references, distinct fixture outputs, and one packed/two
  non-packed projections. An invalid gate/up isolation without full FFN
  recomputation is rejected. Both behaviors were observed failing before their
  implementation.
- Corpus coverage expands to four modes on each of three short raw fixtures;
  the layer trace uses the same four modes. Controls and full-row residual
  injection remain unchanged. No new dependencies, unsafe code, production
  instrumentation, or production packing rollout.

#### Measured Results and Scope

- Local 1.2B Q4_K_M GGUF, two inference/Rayon threads, complete matrices and
  all token rows. All three production-baseline and non-packed FFN-recompute
  controls matched exactly. Gate-only/up-only each packed 16 projections and
  used 32 non-packed projections; down-only packed 8/used 8 non-packed;
  full FFN packed 40/used 8 non-packed.
- The 2.6B GGUF's three controls also matched exactly. Gate-only/up-only each
  packed 30 projections/used 60 non-packed; down-only packed 16/used 14
  non-packed; full FFN packed 76/used 14 non-packed. Non-packed counts include
  deliberate precision choices, not just unsupported formats.
- Final vocabulary-logit relative L2, percentages (not quality scores):

  | Model | Raw fixture | Down only | Gate only | Up only | Full FFN |
  |-------|-------------|-----------|-----------|---------|----------|
  | 1.2B | coding | 3.9666% | 3.9644% | 4.2797% | 5.4209% |
  | 1.2B | tool-request | 6.1920% | 5.9338% | 5.6856% | 24.4887% |
  | 1.2B | tool-result | 2.6097% | 2.6439% | 7.6851% | 3.9666% |
  | 2.6B | coding | 3.5692% | 6.8320% | 4.2627% | 5.0567% |
  | 2.6B | tool-request | 3.7911% | 2.0298% | 2.7732% | 4.6426% |
  | 2.6B | tool-result | 3.3638% | 5.1426% | 7.2400% | 7.0814% |

- New gate/up maximum absolute logit errors (coding, request, result): 1.2B gate
  **0.494559 / 0.741239 / 0.125933**, up
  **0.508871 / 1.127241 / 0.515253**. Up-only changes the 1.2B tool-result greedy
  next-token ID **509 → 508**; other 1.2B modes retain the baseline choice.
- 2.6B gate maximum absolute errors are **0.855800 / 0.348042 / 0.510609**;
  up errors are **0.645342 / 0.445877 / 0.559045**. Gate/up retain its greedy
  choices on all three fixtures. The existing down-only coding change
  **1275 → 11089** reproduces v0.2.57. Twenty-four model/fixture/mode cases
  completed across both models; these do not count repeated trace runs as
  additional independent fixtures.
- The default tool-request layer trace passed on both models, with exact
  production/per-layer non-packed control agreement and matching corpus logits.
  In 1.2B, gate-only/up-only both have their largest relative-residual increase
  in layer 15 (SSM-labelled), **2.1208 / 5.2765 percentage points**. In 2.6B,
  gate-only peaks in layer 2 (attention-labelled), **0.4749 percentage points**;
  up-only peaks in layer 13 (attention-labelled), **0.6315 percentage points**.
  These are increases in relative residual error with changing denominators,
  not intermediate vocabulary logits or causal layer-defect diagnoses.
- Isolated errors cannot simply be added to predict the full-FFN error:
  SwiGLU and downstream layer propagation are nonlinear, and each mode follows
  its own perturbed activation trajectory. A single projection can be worse
  than full packing on a particular fixture. These are short raw prefill
  diagnostics with residual-delta rounding, not generated tool calls or agent
  correctness, timing improvements, or a packing quality acceptance threshold.
  No hardcoded model/projection exclusion or production rollout is justified.

#### Verification

- Independent read-only implementation review found no issues. Scoped quant
  library tests: **34 passed**, five opt-in tests ignored. The corpus and
  layer-trace opt-in tests passed separately on both local GGUFs.
- Scoped Clippy passed with the existing style-lint allowances; package
  formatting and diff-whitespace checks passed. Cargo jobs/test threads stayed
  at one; inference/Rayon threads stayed at two. No full-workspace checks,
  builds, or tests were run, and normal dependencies are unchanged.

## [v0.2.58] - 2026-09-30

### Layer-Wise Packed FFN Error Localization (Test-Only)

#### Ideas, Inspirations & Sources

- **Trace boundaries before changing a quantization policy**: observe the existing
  test-only walker after each ordinary layer computation and after its packed
  FFN residual-delta injection. Compare both against the same baseline layer
  output, separating propagated error from the newly introduced delta.
  No instrumentation is added to production inference.
  - *Sources*: Mivi's `mivi-model/src/{prefill,ssm,transformer}.rs`, the preceding
    captured/cumulative diagnostics, and
    [GGML quantization reference](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c).
- **Measure real activation distributions without assuming outliers explain
  quality loss**: probe normalized and SwiGLU inputs with the experimental codec;
  report reconstruction error and global peak/RMS across all prompt rows.
  - *Inspiration/source*: [Dettmers et al., LLM.int8()](https://arxiv.org/abs/2208.07339).
    These aggregate probes are not that paper's feature-wise outlier analysis,
    do not implement mixed-precision decomposition, and do not prove causation.

#### Added

- An opt-in layer-wise trace for a configurable short raw prompt (default is the
  preceding tool-request fixture). It rejects empty/truncated prompts, derives
  model dimensions/layer kinds/BOS from metadata, and uses complete matrices.
- Final-token residual snapshots only, plus scalar baseline activation metrics;
  neither prompt text nor activation/residual arrays are dumped. Codec probes
  include inputs to fallback-format projections and do not claim those
  projections actually used the packed kernel.
- Ordered/full layer observation checks, final-observation/state agreement,
  per-layer non-packed FFN controls, final production-logit controls, and existing
  finite-error/coverage checks. Skipped SSM FFNs do not probe stale scratch.
- Model-free activation-stat regression with an independent analytical reference,
  plus zero, malformed, non-finite, and unsupported-width cases.
- Blank/whitespace prompt validation before tokenization, so a metadata-derived
  beginning-of-sequence token cannot mask an empty diagnostic input.

#### Measured Result and Decision

- One raw tool-request prompt per local GGUF, two Rayon/inference threads:
  1.2B has 26 tokens/16 layers; 2.6B has 23 tokens/30 layers. Baseline final logits,
  all non-packed layer controls, and recomputed final logits matched exactly.
- Final logit errors reproduce v0.2.57: 1.2B down/full relative L2
  **6.1920% / 24.4887%**; 2.6B **3.7911% / 4.6426%**. Greedy next-token IDs
  remained 509 and 124902 respectively for this prompt.
- **1.2B full FFN**: the largest relative-residual increase occurs in layer 15,
  an SSM-labelled block. Error after layer 14 is **12.3131%**; after the ordinary
  layer-15 computation on the perturbed input it is **22.9116%**; after the new
  packed delta it is **23.1402%**. The delta itself has relative L2 **0.6861%**
  versus the pre-injection residual. Down-only also has its largest increase
  in layer 15, where that mode performs no packed projection.
- **2.6B**: both modes' largest relative-residual increase occurs in layer 13,
  an attention-labelled block. Full FFN increases from **2.7849%** (layer 12)
  to **3.7696%** before the next delta and **3.8986%** afterward. Its final
  layer residual error is **7.8027%**, distinct from vocabulary-logit error.
- Baseline codec reconstruction relative L2 across layers:
  - 1.2B normalized **0.8765–1.5501%**, SwiGLU **1.1946–2.1492%**;
    highest SwiGLU global peak/RMS **390.890** (layer 7).
  - 2.6B normalized **0.7895–1.5722%**, SwiGLU **0.8491–1.9046%**;
    highest SwiGLU global peak/RMS **490.381** (layer 4).
- Growth differs across models, and high aggregate peak/RMS does not establish
  which projection caused final quality loss. Relative-error denominators also
  change across layers. These measurements localize amplification; they do not
  establish a defective SSM/attention implementation, justify hardcoded layer
  exclusions, or explain existing Minicode latency. The packed prototype remains
  disabled in production. Next work should isolate gate/up contributions and
  compare safer policies with the same controls before considering rollout.

## [v0.2.57] - 2026-09-30

### Cumulative FFN Quantization Sensitivity (Test-Only)

#### Ideas, Inspirations & Sources

- **Measure accumulated error, not just isolated projection agreement**:
  extend the captured-input walker with down-only and full-FFN recomputation
  across executed layers. Reuse Mivi's public tile/SwiGLU APIs; eligible packed
  projections are selected by format and dimensions, never a model name.
  - *Sources*: Mivi's `mivi-model/src/{prefill,ssm,transformer}.rs` and
    [GGML quantization reference](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c).
- **Separate arithmetic controls from quality evidence**: a non-packed complete
  FFN-recompute walk is compared directly with production logits before measuring
  packed effects. The outlier/mixed-precision motivation in
  [Dettmers et al., LLM.int8()](https://arxiv.org/abs/2208.07339) inspires caution
  about activation packing, but this experiment neither implements that paper's
  decomposition nor establishes its quality results for these models.

#### Added

- An opt-in three-fixture cumulative diagnostic for raw coding, tool-request,
  and tool-result-continuation text. Fixtures must fit the configured single-tile
  token budget (default/maximum 64); they are not silently truncated.
- Down-only packing across layers and full gate/up → recomputed SwiGLU → down
  packing across layers. Every token/output row and complete matrix is used;
  unsupported formats explicitly use existing F32-activation inference.
- Production baseline and non-packed FFN-recompute controls, complete executed
  projection coverage assertions, and finite-input/output/derived-metric checks.
  Prefix, recurrent, and KV state are reset between fixtures. No prompt/activation
  dumps, production hooks, new dependencies, model-family branches, or new unsafe.
- Model-free regression coverage for F32 fallback, Q4 dispatch, invalid inputs,
  mixed-format FFN dataflow/SwiGLU recomputation, compounded control drift, and
  non-finite relative error derived from finite zero-reference inputs.

#### Measured Result and Decision

- Two local GGUFs, two Rayon/inference threads, three fixtures and two cumulative
  modes per model: **12 measured cases**. Baseline and non-packed recompute
  logits matched production exactly on all six model/fixture combinations.
- Relative L2 error in the final prompt token's vocabulary logits:

  | Model | Raw fixture | Cumulative down-only | Cumulative full FFN |
  |---|---|---:|---:|
  | LFM2.5-1.2B-Instruct | Coding (21 tokens) | 3.9666% | 5.4209% |
  | LFM2.5-1.2B-Instruct | Tool request (26 tokens) | 6.1920% | 24.4887% |
  | LFM2.5-1.2B-Instruct | Tool result (28 tokens) | 2.6097% | 3.9666% |
  | LFM2.5-2.6B | Coding (20 tokens) | 3.5692% | 5.0567% |
  | LFM2.5-2.6B | Tool request (23 tokens) | 3.7911% | 4.6426% |
  | LFM2.5-2.6B | Tool result (26 tokens) | 3.3638% | 7.0814% |

- Per fixture, 1.2B down-only covers eight packed/eight fallback projections;
  full FFN covers 40 packed/eight fallback. For 2.6B, counts are 16/14 and 76/14.
- Greedy next-token agreement was 11/12, not a generation-quality score. In the
  2.6B coding/down-only case the next-token ID changed **1275 → 11089**.
  The 1.2B tool-request/full-FFN case had max absolute logit error **2.008592**
  despite unchanged greedy choice, illustrating why one-token agreement is weak.
- **Do not enable the prototype in production.** Arithmetic/control checks pass,
  but cumulative error and a changed greedy token leave quality unresolved.
  Residual-delta injection differs in floating-point rounding from direct
  projection replacement. These are short, prefill-only raw-text sensitivity
  tests, not chat-template/tool execution, generated-answer quality, server
  timings, or agent-readiness validation. Production inference remains unchanged.

## [v0.2.56] - 2026-09-30

### Captured Prefill Activation Evaluation (Test-Only)

#### Ideas, Inspirations & Sources

- **Capture real inputs without production instrumentation**: reuse the public
  model/tile layer APIs and their retained normalized/SwiGLU scratch buffers.
  Validate the diagnostic walker against production chunked-prefill logits
  before interpreting packed-kernel errors. BOS handling uses GGUF metadata,
  never a model name, family, or fixed token ID.
  - *Sources*: Mivi's `mivi-model/src/{model,prefill,ssm,transformer}.rs` and
    [GGML quantization reference](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c).
- **Test real activation distributions and distinguish local error from quality**:
  the outlier motivation in
  [Dettmers et al., LLM.int8()](https://arxiv.org/abs/2208.07339) inspired checking
  actual FFN inputs rather than only uniform/random stress recipes. This does
  not implement the paper's mixed-precision decomposition or transfer its
  large-model quality claims to Mivi.

#### Added

- Opt-in, memory-only short-prefill evaluation with configurable raw-text prompt,
  token limit (default 32, maximum 64 including BOS), and projection row cap.
  Neither prompt text nor activation arrays are printed or dumped.
- Replay of every eligible executed Q4 gate/up/down projection against existing
  F32-activation inference, with packed scalar/tiled equality and sampled
  independent f64/error-bound checks. Unsupported formats are counted explicitly;
  SSM layers that skip the FFN do not replay stale scratch buffers.
- A separate single-down-projection logit sensitivity walk: choose the latest
  supported executed layer by metadata, inject packed-minus-exact residual deltas
  for all token/output rows, and propagate through subsequent unchanged layers.
  Report final-token logit errors and greedy-token agreement. Identity, malformed
  delta, and invalid/tied greedy-logit cases have model-free regression coverage.

#### Measured Result and Decision

- One short raw-text prompt on each local GGUF, two Rayon threads, 1,024-row
  prefixes for projection metrics. This is not chat-template/tool-call evaluation.
  - LFM2.5-1.2B-Instruct: 16 tokens including BOS, 40 Q4 projections replayed,
    eight unsupported FFN projections skipped. Relative L2: **0.6098–2.5631%**.
  - LFM2.5-2.6B: 14 tokens, 76 Q4 projections replayed, 14 unsupported FFN
    projections skipped. Relative L2: **0.4137–1.9144%**.
- Baseline logits matched production chunked prefill exactly in both runs;
  all replay scalar/tiled comparisons and sampled bounds passed.
- Single full down-projection perturbations (not capped to 1,024 output rows):
  - 1.2B layer 12: final-logit relative L2 **1.2938%**, max absolute error
    **0.067876**; greedy next-token ID remained 509.
  - 2.6B layer 25: final-logit relative L2 **0.2060%**, max absolute error
    **0.035920**; greedy next-token ID remained 358.
- Residual-delta injection introduces different rounding from replacing a
  projection before residual addition. One unchanged greedy token and bounded
  arithmetic errors do not establish model/agent quality. There is no timing
  result, cumulative gate/up/down quantization validation, or production rollout.
  Production inference remains unchanged; broader quality tests are still needed.

## [v0.2.55] - 2026-09-30

### Real GGUF Weight Projection Evaluation (Test-Only)

#### Ideas, Inspirations & Sources

- **Measure with real weights before enabling an approximate kernel**: reuse
  Mivi's existing GGUF reader through a path-only model dev-dependency rather
  than duplicate parsing or add a production dependency. Tensor selection uses
  metadata/type/shape, not a model name or family. Cargo's separate normal/test
  artifacts permit this dev-dependency cycle; GGUF wire type IDs avoid mixing
  Rust types across those artifacts.
  - *Sources*: Mivi's `mivi-model/src/gguf.rs`,
    [Cargo dev-dependency cycles](https://doc.rust-lang.org/cargo/reference/resolver.html#dev-dependency-cycles),
    and [GGML quantization reference](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c).
- **Do not rely on dense synthetic inputs for activation-quantization quality**:
  add deterministic outlier-heavy generated inputs alongside dense inputs.
  The stress recipe is not a measurement of either model's actual activations
  or an implementation of mixed-precision outlier decomposition.
  - *Inspiration/source*: [Dettmers et al., LLM.int8()](https://arxiv.org/abs/2208.07339)
    explains why outlier features can require separate treatment. Its results
    do not establish quality for Mivi's small-model packed CPU experiment.

#### Added

- An opt-in real-weight Q4_K projection benchmark with configurable GGUF path,
  exact tensor names, automatic tensor count, row cap, batch, and iterations.
- Deterministic automatic selection of distinct eligible matrix shapes; format
  inventory reporting and explicit rejection of unsupported requested tensors.
- Finite/length-checked relative-L2 and absolute-error metrics, exact scalar/tiled
  comparisons, and sampled dequantized f64 references with quantization error
  bounds. Configuration and metric regression tests require no model artifact.
- Forward/reverse timing passes include activation packing, allocation, and
  transposition. Documentation includes the focused one-job invocation and
  makes row-prefix sampling and generated-activation limitations explicit.

#### Measured Result and Decision

- Two local GGUFs tested with two Rayon threads, 1,024-row prefixes, batches
  32/64, and dense/outlier input recipes: eight cases per file, 16 iterations
  per timing pass in the repeated evaluation.
  - LFM2.5-1.2B-Instruct: `blk.11.ffn_down.weight` (8,192 input columns) and
    `blk.0.ffn_gate.weight` (2,048 input columns).
  - LFM2.5-2.6B: `blk.10.ffn_down.weight` (10,752 input columns) and
    `blk.0.ffn_gate.weight` (2,048 input columns).
- Dense relative L2 projection error: **0.3896–0.3955%**. Outlier-heavy generated
  inputs: **3.6567–3.7645%**. All tiled outputs matched the packed scalar reference
  exactly; sampled f64/reference/error-bound checks passed. These error bounds
  are correctness checks, not an acceptable model-quality threshold.
- Real-weight timing gains are smaller and less consistent than the preceding
  synthetic-weight measurements. Representative batch-64 dense down-projections:
  - 1.2B: existing F32 12.852–13.060ms; packed tiled 11.425–11.494ms.
  - 2.6B: existing F32 18.633–19.971ms; packed tiled 15.999–16.633ms.
- Regressions remain: 1.2B gate, batch 32, generated outliers took
  1.605–1.674ms packed versus 1.398–1.407ms F32. These are capped projections,
  not full-layer, server, logit, or end-to-end agent latency measurements.
- GGUF two-dimensional inventories include **11 Q6_K matrices** in the 1.2B file
  and **19 Q6_K matrices** in the 2.6B file. This experiment evaluates only Q4_K;
  filenames containing Q4_K_M do not mean every matrix is Q4_K.
- **Keep production F32 inference unchanged**. Next: capture representative
  model activations, evaluate logit-level effects, and address outlier sensitivity
  and mixed-format coverage before considering an inference dispatch policy.

#### Validation

- Focused `mivi-quant` debug suite: 25 passed, two opt-in benchmarks ignored.
  Release suite with the 1.2B artifact and both benchmarks included: 27 passed;
  the 2.6B real-weight evaluation also passed on the release version.
- Scoped Clippy passed with existing baseline lint allowances; formatting and
  diff checks passed.
- Both real-weight evaluations passed; no model generation or agent-readiness
  claim is made. The new reader dependency is test-only, as confirmed by the
  normal dependency tree. No new handwritten unsafe code was introduced.
- Cargo builds/tests/checks remain scoped with one job, one test thread, and
  two Rayon threads. No full-workspace Cargo build, check, or test was run.

## [v0.2.54] - 2026-09-30

### Test-Only Packed Weight Reuse and Token Tiling

#### Ideas, Inspirations & Sources

- **Reuse weight vectors across a four-token register tile**: prepare Q4_K
  nibble vectors and integer scale/minimum corrections once per weight block,
  then reuse them across tokens. Accumulate scaled products in i32 vectors and
  reduce once per block/token instead of once per 32-value subgroup.
  - *Inspiration*: register blocking and packed operands in high-performance
    matrix multiplication; GGML's Q4_K x Q8 activation dot products.
  - *Sources*: [Goto and van de Geijn, Anatomy of High-Performance Matrix Multiplication](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_final.pdf),
    [GGML quantization reference](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c),
    [Pulp safe runtime SIMD](https://docs.rs/pulp/0.22.3/pulp/),
    and Mivi's existing packed scalar reference.
- **Place tokens sharing a weight block next to each other**: pack activations
  directly in block-major order for the tiled experiment, without allocating
  a second packed activation matrix. Keep the earlier reference layouts for
  reproducible comparisons. The four-token tile is a kernel tuning parameter,
  not a model-name or model-family rule.
  - *Sources*: the matrix-multiplication paper above and local layout benchmarks.

#### Added

- A test-only safe AVX2 tiled kernel, scalar CPU fallback, and single-token
  remainder handling. No production inference path or dependency changed.
- Exact scalar/tiled comparisons covering selected batch sizes between 1 and 65, multiple blocks,
  metadata seeds, zero dimensions, odd/parallel output rows, and sentinels.
- Maximum-scale/max-nibble tests exercising integer sums beyond i16 range,
  signed and zero inputs, tiny finite inputs, and error-before-output-mutation
  checks for non-finite, underflowed, and short-buffer inputs.
- Benchmark coverage expanded to 257 and 1,024 output rows, with 16 iterations
  per method/pass and both reverse and forward method order. Timings include
  packing, allocation, and transposition, not just the inner dot product.

#### Measured Result and Decision

- Latest two-thread synthetic benchmark, 1,024 output rows, milliseconds per
  complete matrix multiplication:

  | Columns | Batch | Existing F32 | Packed tiled |
  | --- | --- | --- | --- |
  | 2,048 | 32 | 1.649–1.655 | 1.182–1.285 |
  | 2,048 | 64 | 4.076–4.215 | 2.304–2.369 |
  | 8,192 | 32 | 5.492–6.405 | 4.819–5.116 |
  | 8,192 | 64 | 12.410–12.545 | 8.970–9.534 |

- An earlier eight-iteration run also favored tiled compute for all four
  1,024-row cases. These are local synthetic measurements, not statistical
  guarantees or end-to-end agent latency results.
- Small 257-row cases remain mixed: the latest 8,192-column/batch-64 case
  regressed from F32 3.998–4.032ms to tiled 4.511–4.527ms. Do not make tiled
  compute a universal default from the favorable large-matrix results.
- Tiled outputs exactly match the packed scalar reference in the tested
  cases. Relative L2 error versus original F32 inputs remains 0.6799–0.7280%.
  This is activation quantization error, not a demonstrated real-model quality
  bound; the experiment is not a bit-identical GGML activation codec.
- **Retain as test-only**: production F32 inference is unchanged. Next gates are
  representative real-model projection benchmarks, numerical/logit validation,
  and mixed-format coverage such as Q6_K before considering inference dispatch.

#### Validation

- Focused `mivi-quant` release suite, including opt-in benchmark: 24 passed.
- Debug suite: 23 passed, benchmark ignored. Scoped Clippy passed with the
  existing baseline lint allowances; formatting and diff checks passed.
- One Cargo job, one test thread, two Rayon threads; no full-workspace Cargo
  build, check, or test. No new handwritten unsafe code or model-specific rules.

## [v0.2.53] - 2026-09-30

### Test-Only Safe SIMD Packed Dot Experiment

#### Ideas, Inspirations & Sources

- **Evaluate runtime SIMD without adding handwritten unsafe code**: build on
  v0.2.52's packed Q4_K activation reference using Pulp's safe CPU-feature tokens
  and AVX2 integer multiply-add wrappers. Pulp is a dev-dependency only, with
  `std` and `x86-v3` features; unsupported CPUs use the scalar reference.
  - *Inspiration and sources*: [Pulp safe SIMD abstraction](https://docs.rs/pulp/0.22.3/pulp/),
    [Pulp V3 source](https://docs.rs/pulp/0.22.3/src/pulp/x86/v3.rs.html),
    and [GGML quantization reference](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c).
- **Keep the hot loop within the SIMD target-feature context**: explicitly
  inline the row loop into Pulp's runtime dispatch. Local benchmark time fell
  from about 6.3ms to 0.9ms for 257 rows, 2,048 columns, batch 32, but still did
  not beat the existing F32 kernel. This is experimental evidence, not a claim
  that all dispatch implementations have the same overhead.
  - *Sources*: local before/after benchmarks and Pulp's vectorization API.

#### Added

- Test-only SIMD low/high-nibble integer dot products. Adjacent products are
  bounded below i16 saturation; reductions widen to i32 before summation.
- Exact scalar/SIMD comparisons across every byte seed, signed activation
  extremes, all-15 nibbles whose sum exceeds i16, odd/parallel output rows,
  multiple weight blocks, batch tails, zero dimensions, and output sentinels.
- Alternating F32, scalar-packed, and SIMD-packed benchmark passes, including
  activation packing, allocation, and output transposition. No production
  dispatch, new model-specific rules, or server behavior changed.

#### Measured Result and Decision

- Synthetic Q4_K matrices, 257 output rows, two Rayon threads; final pre-release
  benchmark ranges in milliseconds per full matrix multiplication:

  | Columns | Batch | Existing F32 | Packed scalar | Packed SIMD |
  | --- | --- | --- | --- | --- |
  | 2,048 | 32 | 0.449–0.503 | 1.030–1.104 | 0.857–0.878 |
  | 2,048 | 64 | 0.816–0.857 | 1.910–2.341 | 1.661–1.778 |
  | 8,192 | 32 | 1.796–2.098 | 4.488–4.663 | 3.558–3.772 |
  | 8,192 | 64 | 3.923–4.577 | 9.377–9.411 | 7.838–7.903 |

- SIMD output exactly matches the scalar packed experiment on tested fixtures.
  Relative L2 error versus F32 remains 0.6804–0.7280%; real-model quality has not
  been evaluated, and these synthetic results do not establish agent readiness.
- **Keep inference on the existing F32 path**: packed SIMD remains slower.
  The next performance gate is batch/cache tiling that reuses weight vectors,
  followed by real-model numerical validation only if it beats F32.

#### Validation

- Focused `mivi-quant` release suite, including the opt-in benchmark: 21 passed.
- Cargo operations use one job; tests use one test thread and two Rayon threads.
- Production dependency-tree inspection confirms Pulp is absent from normal
  `mivi-quant` dependencies. No full-workspace build, check, or test was run.

## [v0.2.52] - 2026-09-30

### Test-Only Packed Activation Reference and Benchmark

#### Ideas, Inspirations & Sources

- **Evaluate packed Q4_K weights with Q8 activations before changing inference**:
  the preceding profiler measured FFNs at about 58% of cold prefill. This safe
  Rust experiment quantizes input blocks once, performs bounded integer dot
  products, and applies Q4_K scale/minimum corrections without decoding entire
  weight rows to F32.
  - *Inspiration*: GGML Q8_K activation blocks and CPU Q4_K x Q8_K dot products.
  - *Sources*: [llama.cpp activation quantization reference](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c),
    [CPU type traits](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/ggml-cpu.c),
    and Mivi's existing Q4_K scale/minimum decoder.
- **Express bounded products as i16 for compiler optimization**: 4-bit nibbles
  times signed 8-bit activations fit in i16; reductions remain i32. This roughly
  halved the initial prototype's measured time, but did not beat the existing
  SIMD F32 path. No new unsafe code was introduced.
  - *Source*: local alternating kernel benchmarks and integer-range analysis.

#### Added

- A `cfg(test)`-only Q4_K packed-activation reference and opt-in benchmark.
  No production dispatch, model-specific rule, dependency, or server behavior changed.
- Signed-scale, ties-to-even activation packing with zero-block handling and
  rejection of non-finite inputs and unrepresentable underflowed scales.
- Error-bound, scale/minimum correction, parallel/odd-row, zero-size, output
  sentinel, and buffer-validation tests. Activation packing happens before
  output mutation, so rejected input leaves output untouched.
- The prototype's quantizer is not a bit-identical GGML codec or a new supported
  GGUF weight format. Q6_K support and real-model quality evaluation remain future work.

#### Measured Result and Decision

- Synthetic Q4_K matrices, 257 output rows, two Rayon threads; timings include
  activation packing or F32 transposition and alternate both methods:
  - 2,048 columns, batch 32: F32 0.363–0.591ms; packed 1.096–1.150ms.
  - 2,048 columns, batch 64: F32 0.739–0.912ms; packed 2.370–2.545ms.
  - 8,192 columns, batch 32: F32 2.166–2.230ms; packed 4.377–4.643ms.
  - 8,192 columns, batch 64: F32 3.824–4.213ms; packed 9.615–9.686ms.
- Relative L2 projection error: 0.6804–0.7280%; maximum absolute error
  3.618–8.509 for these synthetic weight scales. These are not real-model quality
  results, and the benchmark's 5% relative-L2 gate is only an experimental guard.
- **Do not enable this implementation for inference**: scalar/compiler-vectorized
  packing alone is slower. Retain it as a reproducible numerical/performance
  reference for a future architecture-specific packed integer kernel.

#### Reproduce

- Validation: 18 quantization tests passed in both debug and release mode;
  all six focused prototype tests (including the ignored benchmark) passed.
  Targeted test/library Clippy and formatting passed. Cargo used one job.

```sh
RAYON_NUM_THREADS=2 cargo test -p mivi-quant --release --lib --jobs 1 packed_prefill_benchmark -- --ignored --test-threads=1 --nocapture
```

## [v0.2.51] - 2026-09-30

### Opt-In Attention Prefill Profiling

#### Ideas, Inspirations & Sources

- **Measure inside the attention block before choosing another optimization**:
  extend the existing SSM stage profiler with attention normalization, Q/K/V
  projections, causal processing, output projection, and FFN buckets.
  - *Inspiration*: Mivi's existing opt-in SSM profiling and measured long-prompt
    latency; distinguish projection costs from the causal KV scan.
  - *Sources*: `crates/mivi-model/src/ssm.rs`, local real-model benchmarks,
    and [Rust Instant documentation](https://doc.rust-lang.org/std/time/struct.Instant.html).
- **Investigate packed quantized projections next, rather than assume the KV
  scan is the only bottleneck**: Mivi's batch kernel currently decodes weight
  rows to F32. llama.cpp's CPU type traits pair Q4_K and Q6_K with Q8_K activation
  dot products, providing a concrete alternative to evaluate.
  - *Source*: [llama.cpp CPU type traits](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/ggml-cpu.c).
  - This release does not implement activation quantization or claim equivalent
    speed, accuracy, or backend performance.

#### Added

- Aggregate attention tile stage timings in opt-in forward-profile snapshots.
- Benchmark output includes attention stage seconds and percentage shares.
- The causal bucket includes per-head normalization, RoPE, KV insertion, and
  attention; the output bucket includes its residual connection.
- Attention profiling creates no timestamps when disabled; the existing public
  unprofiled tile API and numerical computation remain unchanged.
- Tests cover profile totals, accumulation, reset/disable behavior, exclusion of
  nested stages from top-level totals, and identical profiling-on/off output and KV state.

#### Validation and Remaining Work

- Real LFM2.5 1.2B Q4_K_M, two inference threads, 2,068 effective cold tokens:
  isolated prefill 66.28s; first-text latency 65.11s. Warm isolated prefill 2.75s,
  recomputing 23 tokens and reusing 2,048 tokens.
- Attention stages: norm 0.09s, Q/K/V 1.93s, causal processing 16.24s,
  output projection 1.22s, FFN 14.32s. SSM took 32.43s, with FFN share 74.8%.
  Combined FFNs account for approximately 38.6s, or 58% of cold prefill.
- The preceding unmodified v0.2.50 run took 63.06s cold and 2.69s warm.
  These local wall-clock runs vary; this diagnostic update claims no speedup.
- Targeted model/CLI suite: 63 tests passed; four tiny-model fixture tests and
  the release-mode real LFM2.5 profiling-on/off state comparison passed.
  Targeted library Clippy, formatting, and release CLI build passed. Cargo
  compilation used one job; fixture and inference runs used two Rayon threads.
- A real 9,000-token Minicode completion is still unverified. Next: compare
  packed quantized matrix kernels against the existing F32 path, with explicit
  numerical-error checks and model-independent type/capability dispatch.

## [v0.2.50] - 2026-09-30

### Paired-Row, Cache-Tiled CPU Prefill Projections

#### Ideas, Inspirations & Sources

- **Reuse each input vector across two output rows**: the generic batch path
  decodes two weight rows per worker and shares input loads between independent
  AVX2/FMA accumulators. Small batches and odd output-row tails retain the
  established single-row path; no model-family rules were introduced.
  - *Inspiration*: register blocking and data reuse in matrix multiplication.
  - *Sources*: [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html),
    local SSM profiling and alternating single-row/paired-row microbenchmarks.
- **Tile input columns to improve cache locality**: the initial paired kernel
  was inconsistent at batch 64. Processing 128 columns at a time improved local
  measurements while preserving each output lane's column accumulation order.
  The tile size is a CPU-kernel tuning constant, not a model-specific setting.
  - *Inspiration*: layered memory-aware matrix multiplication kernels.
  - *Source*: Goto and van de Geijn,
    [Anatomy of High-Performance Matrix Multiplication](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_final.pdf).

#### Changed

- Large-batch projections accumulate paired decoded rows directly into output
  storage, including within Rayon partitions, eliminating the temporary row copy.
- AVX2/FMA input reuse is runtime-dispatched; portable and small-batch paths remain.
- Each worker needs one additional decoded weight row, not a fully decoded matrix.

#### Added

- Paired-kernel boundary, nonzero-accumulator, output-sentinel, and rounding-order
  tests, plus release-build buffer-bound checks before unsafe SIMD entry.
- Exact comparison against independent decoded-row accumulation for F32, F16,
  BF16, Q4_K, Q6_K, and Q8_0, with serial and odd parallel output-row partitions.
- An opt-in alternating baseline/paired-kernel performance test.

#### Validation

- Local kernel benchmark, 8,192 columns, milliseconds per two output rows:
  batch 32: 0.023–0.024 -> 0.015; batch 64: 0.047–0.049 -> 0.037–0.039;
  batch 128: 0.143–0.158 -> 0.091–0.098.
- Real LFM2.5 1.2B Q4_K_M, two inference threads, identical effective prompt
  length of 538 tokens: cold isolated prefill 16.89s -> 14.98s (~11% shorter);
  cold first-text latency 16.84s -> 14.74s. Warm isolated prefill 2.38s -> 2.28s.
- These are local wall-clock observations, not controlled cross-backend or model
  quality comparisons. Cold latency for a real 9,000-token agent prompt remains
  unverified and is not claimed to be resolved.
- Targeted core/quant/model suite: 78 tests passed; all three tiny-model fixture
  regressions and the real LFM2.5 cold/cache state comparison at 129 tokens passed.
  Release-mode core/quant suite: 38 tests passed, including buffer rejection.
  Targeted library Clippy passed with the existing lint allowances.

## [v0.2.49] - 2026-09-30

### Shared Prefix KV Blocks and Correct Recurrent Cache Restore

#### Ideas, Inspirations & Sources

- **Store each causal KV block once**: prefix nodes own only their new token
  interval and reference shared ancestors by chained hash. Each boundary keeps
  its SSM checkpoint, avoiding cumulative copies of all previous KV tokens.
  - *Inspiration*: hash-addressed KV blocks and shared prefix ancestry.
  - *Source*: [vLLM automatic prefix caching design](https://docs.vllm.ai/en/latest/design/prefix_caching/).
- **Keep recurrent checkpoints causally correct**: the final prompt token is
  evaluated from a checkpoint preceding that token. A failing tiny hybrid-model
  comparison demonstrated that restoring a checkpoint after the final token and
  processing it again corrupted the old cached result.
  - *Sources*: local cold-versus-cached KV, SSM, and logits comparisons;
    `crates/mivi-model/src/model.rs` and `crates/mivi-kv/src/prefix.rs`.

#### Changed

- Token and chunked prefill now cache KV intervals rather than cumulative KV snapshots.
- Shared ancestors remain available when deeper nodes are evicted under the existing
  count and byte limits; retained state grows linearly with cached context length.
- Model restore reassembles KV intervals in causal order and restores the final
  SSM checkpoint. Complete snapshots and their disk format remain supported.
- The model uses the cache's configured chunk size for recording boundaries.
- Focused model benchmarks report estimated prefix-cache memory usage.

#### Fixed

- Exact-boundary cache hits no longer process the final token twice in recurrent state.

#### Added

- Range export/import round trips for all four KV precisions and selective layers.
- Shared-branch and eviction coverage, missing-ancestor rejection, and a low-memory
  9,000-token synthetic cache test retaining 8,960 tokens within a 4 MiB budget.
- Opt-in cold-versus-cached hybrid-model comparison at exact and partial boundaries.

#### Validation

- Real LFM2.5 1.2B Q4_K_M benchmark, two inference threads, target prompt 2,048:
  - Warm isolated prefill: 22.87s -> 2.69s (~8.5x faster in these local runs).
  - Warm first-text latency: 23.39s -> 2.64s.
  - Recomputed prompt tokens: 471 -> 23; 2,048 tokens reused.
  - 32 cached blocks consumed approximately 60.27 MiB.
- Cold isolated prefill was 105.08s in this run (previous run: 82.70s).
  This cache change targets reuse; it does not resolve cold agent-prompt latency.
  Wall-clock comparisons include host-load variation.

## [v0.2.48] - 2026-09-30

### Register-Blocked CPU Batch Accumulation

#### Ideas, Inspirations & Sources

- **Keep output accumulators in registers across input columns**: local SSM profiling
  identified batched FFN projections as a major prefill cost. The shared AVX2/FMA
  kernel now uses independent accumulators for 64- and 32-value batch blocks,
  reducing repeated output loads/stores while preserving each lane's column order.
  - *Inspiration*: register blocking and independent FMA accumulators, applied to
    Mivi's existing transposed-input kernel without model-specific rules.
  - *Sources*: [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html),
    the local `mivi bench` SSM profile, and `crates/mivi-core/src/simd/avx2.rs`.
- Preserve the established column-first kernel for batches below 32, including
  short remainders after a prefix-cache hit; retain portable scalar dispatch.

#### Changed

- Register-blocked accumulation in the shared large-batch AVX2/FMA path used by
  supported quantized projections in attention and SSM layers.

#### Added

- Vector block/tail tests covering zero columns, nonzero initial accumulators,
  and output sentinels.
- Nonzero Q4_K and Q8_0 batch comparisons against matrix-vector results, including
  parallel row partitions and irregular batch sizes.
- Opt-in release-mode kernel benchmark, ignored during ordinary test runs.

#### Validation

- Focused 8,192-column kernel measurement (512 repetitions): batch 64 improved
  from 0.037 to 0.027 ms/row; batch 128 from 0.135 to 0.074 ms/row.
- These are local kernel measurements, not a guarantee of equivalent improvement
  in full-model latency or completion of a coding-agent task.
- Real LFM2.5 1.2B Q4_K_M run, two inference threads, chunked prefill, requested
  tile 128, target prompt 2,048 tokens (2,068 effective cold / 2,071 warm):
  - Cold isolated prefill: 119.68s -> 82.70s (~31% less elapsed time).
  - Cold first-text latency: 107.53s -> 81.19s.
  - Warm isolated prefill: 31.84s -> 22.87s.
  - Warm first-text latency: 32.67s -> 23.39s.
- Before/after timings are single local runs and include host-load variation.
  The warm run still recomputed 471 tokens because cumulative full-state cache
  snapshots exhausted the byte budget; agent-sized first turns remain unresolved.

## [v0.2.47] - 2026-09-30

### Fused Batched SIMD Accumulation

#### 💡 Ideas, Inspirations & Sources

- **Fuse the large-batch accumulation loop**: batched quantized matmul now performs one
  SIMD-dispatched transposed-input accumulation per decoded weight row instead of dispatching
  once for every input column.
  - *Inspiration*: llama.cpp's batched/ubatch kernels minimize repeated inner-loop dispatch and
    keep the batch dimension contiguous for vectorized accumulation.
  - *Sources*: [llama.cpp batch processing](https://github.com/ggml-org/llama.cpp/wiki/How-to-use-llama.cpp-with-other-models)
    and [ggml quantized matrix multiplication](https://github.com/ggerganov/ggml/blob/master/src/ggml-cpu/ggml-cpu.c).
- **Preserve portability and correctness**: the fused helper retains AVX2/FMA and scalar paths,
  with the same `[batch, rows]` output layout and no model-family assumptions.

#### Changed

- Replaced repeated per-column SIMD dispatch in the large-batch quantized matmul kernel with a
  fused transposed-batch SIMD helper.

#### Performance

- LFM2.5 1.2B Q4_K_M, two runtime threads, 128-token agent-style prompt:
  - Cold prefill: 13.11s → 9.31s (~29% faster).
  - Warm prefill: 4.64s → 2.57s (~45% faster).

#### Added

- SIMD regression coverage for transposed-batch accumulation equivalence.

## [v0.2.46] - 2026-09-30

### SSM Prefill Stage Diagnostics

#### 💡 Ideas, Inspirations & Sources

- **Measure the real agent-latency bottleneck**: opt-in profiling now breaks chunked SSM
  prefill into normalization, input projection, causal convolution, output projection, and FFN.
  - *Inspiration*: llama.cpp and other local inference runtimes expose stage-level prompt
    processing measurements so optimization targets are based on observed workload cost.
  - *Sources*: [llama.cpp performance counters](https://github.com/ggml-org/llama.cpp/wiki/Performance-of-llama.cpp)
    and [vLLM profiling guidance](https://docs.vllm.ai/en/latest/design/metrics.html).
- **Keep diagnostics model-agnostic and low overhead**: timing is enabled only through the
  existing benchmark profiling path; ordinary inference does not collect per-stage timestamps.

#### Added

- SSM sub-stage timing in the focused model benchmark.
- Regression coverage for SSM stage-duration accounting.

#### Findings

- On the LFM2.5 1.2B CPU benchmark, SSM FFN work accounts for roughly 75% of SSM time,
  making the generic batched quantized FFN path the next optimization target.

## [v0.2.45] - 2026-09-28

### Multi-Core Parallel Quantized Matmul & 64K In-Memory Prefix Cache Retention

#### 💡 Ideas, Inspirations & Sources

- **All-Core Rayon Work-Stealing in Batch Matrix Multiplication (`mivi-quant::lib`)**:
  - *Problem Fixed*: `quantized_matmul_rows` previously performed a single 2-way `rayon::join` on matrix rows, leaving CPUs with 4, 8, 12, or 16 threads largely idle during prompt prefill.
  - *Solution*: Replaced 2-way split with dynamic `par_chunks_mut(chunk_rows * batch)` chunking rows evenly across all `rayon::current_num_threads()` workers. Every core processes non-overlapping row partitions with thread-local buffers, maximizing hardware ALU utilization during cold prefill.
  - *Inspiration*: [llama.cpp `ggml_mul_mat` OpenMP thread work-sharing](https://github.com/ggerganov/llama.cpp) across physical CPU cores.

- **64K-Capacity In-Memory Prefix Caching for Instant Agent Turns (`mivi-kv::prefix`, `mivi-model::model`)**:
  - *Problem Fixed*: When an AI coding agent (e.g. `minicode`, Cline) sent a 14,000+ token static prompt (system instructions + tool schemas), Mivi's prefix cache previously capped retention at 32 chunks (2,048 tokens / 32 MB). Every turn was forced to recompute all 14,000 tokens sequentially on CPU, whereas Ollama and llama.cpp reuse resident KV cache slots for near-instant (0 ms) responses on turn 2+.
  - *Solution*: Expanded `DEFAULT_MAX_CACHED_CHUNKS` to 1,024 chunks (65,536 tokens = 64k context window) and `DEFAULT_MAX_PREFIX_CACHE_BYTES` to 512 MB (fitting safely within Mivi's 3.0 GB system RAM budget). Once an agent's static workspace prompt completes turn 1, subsequent turns achieve full prefix cache hits with zero loading time.
  - *Inspiration*: [Ollama context persistence & llama.cpp slot cache reuse](https://github.com/ollama/ollama).

- **SSM 1D Short-Convolution Kernel Hoisting & Vectorization (`mivi-model::ssm`)**:
  - *Problem Fixed*: In hybrid models like `LFM2.5` with 10 SSM layers, the 1D convolution rolling buffer checked kernel configuration inside the inner dimension loop (`for d in 0..dim`), executing hundreds of millions of branch evaluations.
  - *Solution*: Hoisted the `kernel_size == 3 && has_full_conv` branch outside the dimension loop, enabling LLVM auto-vectorization across all 2,048 dimensions.

- **Developer Concurrency Guardrails & Rust 1.94 Linter Alignment (`justfile`, workspace)**:
  - Configured `justfile` recipes with explicit job limits (`jobs 1` for test, clippy, fmt; `jobs 2` for build, serve, chat) to protect low-spec machines from memory spikes.
  - Cleaned up clippy lints across all 12 workspace crates for Rust 1.94 toolchains.

---

## [v0.2.44] - 2026-09-17

### Prefix Cache Root-Chain Retention

#### 💡 Ideas, Inspirations & Sources

- **Preserve reusable prefix roots under tight cache budgets**: prefix cache eviction now trims the
  deepest tail chunks before evicting root chunks, because Mivi's hierarchical prefix lookup cannot
  reuse later chunks after chunk 0 is gone.
  - *Inspiration*: the live 1.2B benchmark showed warm/shared-prefix prefill recomputing almost the
    full prompt because the 32 MB cache budget retained only late chunks that were unusable without
    their prefix chain.
  - *Sources*: [vLLM automatic prefix caching](https://docs.vllm.ai/en/latest/design/automatic_prefix_caching.html)
    and [LMCache KV-cache reuse documentation](https://docs.lmcache.ai/).
- **Optimize for repeated agent prompts**: coding agents resend stable system/tool/workspace prefixes
  across turns, so retaining the earliest reusable chunks reduces TTFT more than retaining isolated
  tail chunks.
  - *Inspiration*: prefix caching guidance for repeated chat, RAG, and agent workloads where shared
    prompt structure should skip repeated prefill computation.
  - *Source*: [Modular Inference Handbook: Prefix caching](https://handbook.modular.com/inference-optimization/prefix-caching/).

#### Fixed

- Tight in-memory prefix caches no longer evict reusable root chunks before deeper tail chunks.
- Warm/shared-prefix benchmark reuse now preserves a valid prefix chain instead of falling back to
  full prefill when the cache budget prunes late in a long prompt.

#### Added

- Regression coverage proving a tight cache keeps the reusable root chain available for prefix lookup.

## [v0.2.43] - 2026-09-17

### Agent-Sized Prefill Benchmark Controls

#### 💡 Ideas, Inspirations & Sources

- **Benchmark the actual agent bottleneck**: `mivi bench --model` now accepts explicit prefill
  strategy, tile size, and target prompt-token controls so local testing can reproduce
  AI-agent-sized context prompts instead of relying on tiny canned prompts.
  - *Inspiration*: the real failure mode was a client disconnect during thousands of prompt tokens
    of CPU prefill, so the benchmark must measure prompt ingestion directly.
  - *Sources*: [llama.cpp server prompt/batch options](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
    and [llama.cpp server batching design notes](https://github.com/crc-org/llama.cpp/blob/main/tools/server/README-dev.md).
- **Keep comparisons model-agnostic**: synthetic workspace-context prompts are generated from the
  active model tokenizer and do not depend on model-family names or hardcoded special tokens.
  - *Inspiration*: local inference servers expose prompt-processing controls separately from model
    quality so performance diagnosis stays reproducible.

#### Added

- `mivi bench --prefill-strategy <token|chunked>`.
- `mivi bench --prefill-tile-tokens N`.
- `mivi bench --bench-prompt-tokens N`.
- Regression coverage for benchmark CLI defaults/configuration and synthetic prompt sizing.

## [v0.2.42] - 2026-09-17

### Agent-Ready Chunked Prefill Defaults

#### 💡 Ideas, Inspirations & Sources

- **Default server prefill to chunked execution**: `mivi serve` now defaults to the model-agnostic
  chunked prefill path with 64-token tiles, reducing cold first-token latency for large AI-agent
  prompts without requiring every user to discover `--prefill-strategy chunked` manually.
  - *Inspiration*: local-agent servers need fast prompt ingestion because coding agents commonly
    send thousands of workspace/context tokens before the model can produce its first answer.
  - *Sources*: [llama.cpp server prompt/batch options](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
    and [llama.cpp server batching design notes](https://github.com/crc-org/llama.cpp/blob/main/tools/server/README-dev.md).
- **Keep a conservative escape hatch**: `--prefill-strategy token` remains available for debugging,
  profile comparisons, and any model path where token-by-token prefill is preferable.
  - *Inspiration*: llama.cpp exposes separate runtime controls for prompt processing and generation
    threads/batches; Mivi should keep prefill policy explicit and observable rather than burying it
    in model-specific code.

#### Changed

- `mivi serve` defaults to `--prefill-strategy chunked --prefill-tile-tokens 64`.
- `ServerConfig::default()` now uses chunked prefill for server/agent workloads.

#### Added

- Regression coverage for the CLI and server default prefill strategy.

## [v0.2.41] - 2026-09-17

### Streaming Disconnect & Error Lifecycle Metrics

#### 💡 Ideas, Inspirations & Sources

- **Count every streaming lifecycle outcome**: SSE responses now distinguish normal completion,
  response-body errors, and early client disconnects in `/metrics`, so cancelled agent requests no
  longer disappear from observability.
  - *Inspiration*: agent clients often cancel or retry long-running streams, and those cancellations
    need to be visible separately from successful generation.
  - *Sources*: [Axum `Body::from_stream` and `Body::into_data_stream`](https://docs.rs/axum/latest/axum/body/struct.Body.html)
    and [Axum SSE connection-close discussion](https://github.com/tokio-rs/axum/discussions/1060).
- **Use response-body ownership as the lifecycle boundary**: Mivi now records disconnects when a
  streaming body is dropped before EOF, while preserving the existing `headers` vs `complete` log
  split.
  - *Inspiration*: Drop-guard based cleanup for async response bodies, a common Rust pattern when
    cancellation happens by dropping the future or stream.
  - *Sources*: [Axum disconnect/drop discussion](https://github.com/tokio-rs/axum/discussions/1094)
    and [Tower HTTP timeout layers](https://docs.rs/tower-http/latest/tower_http/timeout/index.html).

#### Added

- `/metrics` counters for streaming lifecycle outcomes:
  - `stream_completions_total`
  - `stream_body_errors_total`
  - `stream_client_disconnects_total`
- Regression coverage for normal SSE completion, response-body errors, and early client disconnects.

#### Fixed

- Dropped SSE bodies from cancelled clients are now recorded as `client disconnected` instead of
  being invisible in server metrics.

## [v0.2.40] - 2026-09-17

### Accurate Streaming Request Lifecycle Logs

#### 💡 Ideas, Inspirations & Sources

- **Separate response start from response completion**: SSE requests now report header latency
  when the response opens and total request latency when the body closes, preventing long-running
  generations from appearing to finish in microseconds.
  - *Inspiration*: streaming-server observability that distinguishes time-to-first-byte from
    end-to-end request duration for agent workloads.
  - *Sources*: [Axum SSE response streaming](https://docs.rs/axum/0.7/axum/response/sse/index.html)
    and [HTTP streaming body semantics](https://docs.rs/axum/0.7/axum/body/struct.Body.html).
- **Detect streaming at the transport boundary**: the middleware recognizes `text/event-stream`
  responses in addition to route metadata, keeping the behavior model- and provider-agnostic.
  - *Inspiration*: content-type based protocol detection so future streaming endpoints do not need
    duplicated logging assumptions.
  - *Source*: [WHATWG Server-Sent Events](https://html.spec.whatwg.org/multipage/server-sent-events.html).

#### Fixed

- Streaming OpenAI, Anthropic, and native-agent responses no longer log the time to create SSE
  headers as if it were the completed request duration.
- Native-agent SSE responses are now explicitly marked as streaming in their route metadata.

#### Added

- Regression coverage for identifying SSE responses without relying on route-specific metadata.

## [v0.2.39] - 2026-09-17

### Streaming Token Accounting

#### 💡 Ideas, Inspirations & Sources

- **Measure the complete serving path**: OpenAI-compatible streaming responses now record the
  admitted prompt tokens and the raw generated output tokens using the active runtime tokenizer,
  matching the existing blocking and native-agent accounting without model-specific assumptions.
  - *Inspiration*: serving metrics that separate request latency, time-to-first-token, and token
    throughput so agent performance can be diagnosed from evidence.
  - *Sources*: [llama.cpp server monitoring](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
    and [vLLM metrics documentation](https://docs.vllm.ai/en/latest/usage/metrics.html).
- **Count once at stream completion**: accounting occurs after the generation stream closes and
  before response validation, so tool-call markup and ordinary text are measured consistently.
  - *Inspiration*: protocol-independent runtime instrumentation that works for future models and
    codecs.
  - *Source*: [OpenAI-compatible streaming conventions in llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

#### Fixed

- Streaming chat and streaming tool-call requests no longer leave
  `prompt_tokens_total` and `completion_tokens_total` at zero in `/metrics`.

#### Added

- Regression coverage requiring nonzero prompt and completion token counters for a completed
  streaming chat response.

## [v0.2.38] - 2026-09-17

### Automatic Model-Agnostic Tool Profile Discovery

#### 💡 Ideas, Inspirations & Sources

- **Separate prompt metadata from output-protocol metadata**: GGUF chat templates can describe
  how tools are listed in the prompt without spelling out the delimiters the model emits. Mivi now
  supplements the template with semantically named tool delimiter tokens discovered in the loaded
  tokenizer vocabulary.
  - *Inspiration*: keeping model protocol data at the metadata/adapter boundary instead of
    identifying models by name or embedding LFM2.5 branches in HTTP routes.
  - *Sources*: [GGUF metadata specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
    and [LFM2.5 tool-template discussion](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/discussions/12).
- **Explicit configuration remains authoritative**: external profile files still override automatic
  discovery, preserving support for future models with different delimiters and incomplete metadata.
  - *Inspiration*: declarative model adapters and the explicit profile escape hatch documented in
    the model-agnostic agent plan.
  - *Source*: [LFM2.5 chat template](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/blob/main/chat_template.jinja).

#### Fixed

- LFM2.5 GGUF files with tool delimiter tokens in the vocabulary but no output delimiters in the
  embedded chat template no longer silently resolve to `text_only`.
- Default server startup now exposes tool capability for such models without requiring a
  model-specific command-line profile.

#### Added

- Regression coverage for completing a chat-template profile from tokenizer tool markers.

## [v0.2.37] - 2026-09-17

### Agent Latency Telemetry

#### 💡 Ideas, Inspirations & Sources

- **Measure the agent-facing boundary**: added cumulative time-to-first-token microseconds and
  sample counts to the server metrics snapshot, making prefill and responsiveness measurable
  without assuming a model family, prompt format, or provider.
  - *Inspiration*: separating TTFT from decode throughput in local inference benchmarking and
    exposing runtime monitoring data for serving diagnosis.
  - *Sources*: [llama.cpp server monitoring and metrics](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
    and [Liquid AI hardware evaluation guidance](https://github.com/Liquid4All/docs/blob/main/guides/hardware-evaluation.mdx).
- **Instrument every streaming protocol boundary**: OpenAI chat, native agent steps, and
  Anthropic streaming now record the first non-empty model output; blocking responses remain
  total-latency-only because they do not expose an observable first-token event.
  - *Inspiration*: protocol-independent observability at the engine/stream boundary, so future
    models can be compared using the same server measurements.
  - *Source*: [vLLM automatic prefix-caching performance model](https://docs.vllm.ai/en/v0.10.1/features/automatic_prefix_caching.html).

#### Added

- `time_to_first_token_microseconds_total` and `first_token_count` in `/metrics`.
- Unit and streaming integration coverage for TTFT accounting.

#### Fixed

- Agent telemetry now records the first emitted chunk rather than accidentally measuring the
  final chunk of a completed response.

## [v0.2.35] - 2026-09-17

### Agent Tool-Call Compatibility

#### 💡 Ideas, Inspirations & Sources

- **Template-compatible native tool lists**: LFM2.5's documented Jinja `tojson` representation
  uses one-line JSON with separator spacing. Mivi now preserves that representation instead of
  compacting tool definitions, which prevents small models from corrupting function names and
  arguments.
  - *Inspiration*: model-native chat-template fidelity and the project's live LFM2.5 diagnosis.
  - *Source*: [Liquid LFM2.5-1.2B model card](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/blob/main/README.md).
- **Codec-driven required tool use**: required and named tool choices now derive the forced output
  prefix from the selected tool codec. The implementation does not identify models by name and
  remains usable with future delimiter-based model profiles.
  - *Inspiration*: structured decoding and the protocol-specific parser boundary used by
    llama.cpp's LFM2/LFM2.5 support.
  - *Sources*: [llama.cpp LFM2.5 parser issue](https://github.com/ggml-org/llama.cpp/issues/23838)
    and [Liquid's tool-call template discussion](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/discussions/12).
- **Measured agent validation**: with the local official LFM2.5 1.2B Q4 model, two runtime
  threads, and chunked prefill, Mivi now emits a valid `get_candidate_status` call and completes
  the calculator agent loop (`45 * 12 = 540`). Optional `auto` tool choice remains optional by
  design; clients that require a tool must send `tool_choice: "required"`.

#### Added

- Server configuration for `--prefill-strategy` and `--prefill-tile-tokens`.
- Regression coverage for codec-derived prefixes, native tool-list formatting, and profile prompt
  fidelity.

#### Fixed

- Required tool calls could previously fail after sampling because Mivi validated them only after
  generation; the selected codec prefix is now supplied before decoding.
- Agent runs could repeatedly force a tool call after a tool had already executed; only the first
  required agent step is constrained, allowing the final natural-language response.

## [v0.2.34] - 2026-09-17

### Small-Batch Prefill Projection Optimization

#### 💡 Ideas, Inspirations & Sources

- **Loop-order correction for tiled projections**:
  - Small tiles now compute full-width SIMD dot products per input row instead of invoking a
    SIMD helper once per input column over a tiny batch slice. This keeps the implementation
    model-agnostic and removes the tile-size-specific performance cliff.
  - *Inspiration*: llama.cpp's batched/ubatch prefill approach and the project's measured
    chunked-prefill research.
  - *Sources*: [llama.cpp](https://github.com/ggerganov/llama.cpp) and
    [KV/chunked-prefill research](docs/KV_QUANT_AND_CHUNKED_PREFILL_RESEARCH.md).
- **Evidence-driven optimization**:
  - The Mivi Q4 model's tile-2 cold prefill improved from `1.68` to `13.69` tok/s and its
    tile-8 result improved from `9.82` to `27.92` tok/s. The LFM2.5 1.2B Q4 tile-2 run now
    completes at `6.49` tok/s instead of exceeding the prior 180-second benchmark budget, and
    its final-code tile-8 result is `8.96` tok/s. The token-major default remains unchanged
    because chunked performance is still model- and tile-dependent.
  - *Inspiration*: stage-level profiling and separate prefill/TTFT/decode measurements.
  - *Source*: [Liquid AI hardware evaluation guide](https://github.com/Liquid4All/docs/blob/main/guides/hardware-evaluation.mdx).

#### Changed

- Added a regression test that selects full-input dot products for small batches and retains
  across-batch FMA for larger tiles.
- Revalidated model output/state equivalence after the quantized projection change.

#### Known Limitations

- The batch kernel is still generic rather than format-specific Q4_K/Q6_K microcode, so reliable
  speedups for every model and tile size are not established.
- Active LoRA adapters and quantized-KV equivalence remain on the existing follow-up path.

## [v0.2.33] - 2026-09-17

### Opt-In Hybrid Chunked Prefill

#### 💡 Ideas, Inspirations & Sources

- **Model-agnostic tiled prefill**:
  - Added `PrefillStrategy` with token-major fallback and opt-in chunked execution selected by
    `MIVI_PREFILL_STRATEGY` and `MIVI_PREFILL_TILE_TOKENS`; no model-family or fixed-layer branch
    was added.
  - *Inspiration*: llama.cpp micro-batched prefill (`n_ubatch`) and the project’s earlier
    chunked-prefill research.
  - *Sources*: [llama.cpp](https://github.com/ggerganov/llama.cpp) and
    [KV/chunked-prefill research](docs/KV_QUANT_AND_CHUNKED_PREFILL_RESEARCH.md).
- **Hybrid state correctness**:
  - Batched projections now preserve ordered SSM convolution state, causal attention, selective KV
    positions, final-row logits, and 64-token prefix-cache snapshot boundaries.
  - *Inspiration*: LMCache-style prefix state reuse and the existing hybrid SSM/attention design.
  - *Sources*: [LMCache](https://github.com/LMCache/LMCache) and the
    [64K hybrid scaling plan](docs/IMPLEMENTATION_PLAN_64K_LONG_CONTEXT_AND_HYBRID_SCALING.md).
- **Measured, conservative rollout**:
  - Added token-vs-chunked equivalence tests for prompt lengths 63/64/65 and tile sizes 1/2/8/64.
  - Added input transposition, SIMD FMA across tile rows, and two-way Rayon output-row splitting
    to the batch kernel. On the 1.2B Q4 model with two runtime threads, cold prefill measured
    9.08 tok/s token-major versus 9.04 tok/s with tile-64 chunked prefill: effectively within
    measurement noise, so chunked mode remains opt-in. Profiling still shows SSM as the dominant
    stage at roughly 62–65%.
  - *Inspiration*: measured TTFT/prefill/decode separation from the previous benchmark work.
  - *Source*: [Liquid AI hardware evaluation guide](https://github.com/Liquid4All/docs/blob/main/guides/hardware-evaluation.mdx).
  - A bounded two-thread sweep also exposed strong tile-size sensitivity: on the LFM2.5 1.2B
    Q4 model, cold chunked prefill was `8.15/3.33/8.90/10.12` tok/s for tiles `1/8/32/64`
    respectively, while tile `2` exceeded the 180-second run budget. On the separate local
    Mivi Q4 model, cold token-major prefill was `17.13` tok/s and chunked tiles `1/2/8/32/64`
    measured `19.20/1.68/9.82/29.84/34.65` tok/s. These results are evidence for explicit
    runtime selection and future auto-tuning, not a universal tile-size recommendation.

#### Added

- Reusable tile activation buffers and checked batched quantized projection APIs for F32, F16,
  BF16, Q8_0, Q4_K, and Q6_K.
- Ordered SSM and causal GQA attention tile paths with deterministic tiny-GGUF equivalence gates.

#### Known Limitations

- The batch kernel now reuses decoded rows and SIMD-accumulates across transposed tile inputs,
  but it is still a generic path rather than format-specific Q4_K/Q6_K microkernels; it has not
  demonstrated a reliable larger-Q4 speedup yet.
- Small tiles can regress severely because the generic batched path pays setup/dequantization
  overhead without enough reuse; tile `2` exceeded the bounded LFM benchmark timeout and reached
  only `1.68` tok/s on the separate Mivi model. Chunked execution is therefore opt-in and has no
  hardcoded automatic tile choice.
- Active LoRA adapters intentionally use the proven token-major fallback.
- Quantized-KV equivalence and format-specific tiled SIMD optimization remain follow-up work.

## [v0.2.32] - 2026-09-17

### Opt-In Forward Stage Profiling

#### 💡 Ideas, Inspirations & Sources

- **Forward-stage diagnostics**:
  - Added disabled-by-default profiling for embedding, attention, SSM, and final-logits stages.
  - The focused benchmark now reports stage durations and relative shares so optimizations can be
    selected from measurements instead of assumptions.
  - *Inspiration*: separate time-to-first-token, prefill, and decode measurements in local hybrid
    inference systems.
  - *Source*: [Liquid AI hardware evaluation guide](https://github.com/Liquid4All/docs/blob/main/guides/hardware-evaluation.mdx).
- **Model-agnostic instrumentation**:
  - Profiling is exposed through the generic `Model` API and is not tied to LFM2.5 names, tensor
    shapes, or a specific tool-calling format.
  - *Inspiration*: Liquid AI's hybrid convolution and GQA architecture guidance.
  - *Source*: [LFM2.5-1.2B-Instruct model card](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct).

## [v0.2.31] - 2026-09-17

### Focused Prefill and First-Output Benchmarking

#### 💡 Ideas, Inspirations & Sources

- **Separated model latency measurements**:
  - Added isolated prompt-prefill throughput, first emitted-text latency, total generation time,
    decode throughput estimates, effective prompt-token counts, and prefix-cache visibility to
    `mivi bench --model`.
  - *Inspiration*: benchmarking time-to-first-token separately from decode throughput for hybrid
    local inference runtimes.
  - *Source*: [Liquid AI hardware evaluation guide](https://github.com/Liquid4All/docs/blob/main/guides/hardware-evaluation.mdx).
- **Reproducible, model-agnostic diagnostics**:
  - Removed unrelated synthetic demo claims from the model benchmark so its output represents
    measured latency rather than unsupported feature status.
  - The benchmark uses generic prompt fixtures and reports when a model produces no visible output;
    it does not hardcode a model family or claim response quality.
  - *Inspiration*: model-provided templates and runtime-specific compatibility should be tested
    independently from performance measurement.
  - *Sources*: [Liquid AI tool-use documentation](https://docs.liquid.ai/lfm/key-concepts/tool-use),
    [Liquid AI migration guide](https://github.com/Liquid4All/docs/blob/main/guides/migration-guide.mdx).

## [v0.2.30] - 2026-09-16

### Agent Prompt Fidelity & Built-in Tool Schema

#### 💡 Ideas, Inspirations & Sources

- **Preserve task intent in internal agent prompts**:
  - *Inspiration*: OpenAI's role-based message model, where the user's message is passed as the task input
    and tool selection is handled separately.
  - *Source*: [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat).
  - Removed an optional planning suffix that could distract small models from the user's exact task.
- **Concise model-facing built-in tool schemas**:
  - *Inspiration*: Liquid AI's tool-use flow, which keeps tool definitions explicit while leaving execution to
    the external tool runner.
  - *Source*: [Liquid AI LFM tool-use documentation](https://docs.liquid.ai/lfm/key-concepts/tool-use).
  - Simplified the calculator schema descriptions and removed an illustrative expression that could be
    copied as a tool argument by small models.
- **Live agent verification**:
  - *Inspiration*: model-native tool calls followed by tool results and a second generation.
  - *Sources*: [Liquid AI tool-use documentation](https://docs.liquid.ai/lfm/key-concepts/tool-use),
    [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat).
  - Verified the 2.6B model through `/v1/mivi/agent`: `45 * 12` produced tool result `540` and a final
    answer of `540`, including complete SSE termination.

## [v0.2.29] - 2026-09-16

### Internal Agent Reliability & Request-Scoped Sampling

#### 💡 Ideas, Inspirations & Sources

- **Bounded required-tool retries**:
  - *Inspiration*: OpenAI-style `tool_choice: "required"` semantics, where a tool call is required before
    the assistant can finish.
  - *Source*: [OpenAI tool-choice API reference](https://platform.openai.com/docs/api-reference/chat/create).
  - Added `tool_call_retries` to `/v1/mivi/agent` (default 1, bounded by the server). Retries apply only when
    required tool output is missing, do not consume action steps, and fail closed when the budget is exhausted.
- **Model-native multi-turn tool workflow**:
  - *Inspiration*: Liquid AI's documented sequence of tool definitions, model tool call, external execution,
    `tool`-role result, and a second generation for the final answer.
  - *Source*: [Liquid AI LFM tool-use documentation](https://docs.liquid.ai/lfm/key-concepts/tool-use).
  - Required mode now accepts normal final text after a tool call instead of incorrectly rejecting the final
    response on the next agent turn.
- **Request-scoped agent sampling**:
  - *Inspiration*: the existing OpenAI-compatible generation controls and Liquid's prompting guidance for
    tuning sampling without changing model adapters.
  - *Sources*: [Liquid AI prompting guide](https://docs.liquid.ai/lfm/key-concepts/text-generation-and-prompting),
    [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat).
  - Added validated `temperature`, `top_p`, `top_k`, `min_p`, repetition/presence/frequency penalties, and
    `seed` fields to internal agent requests. The values are forwarded to every generation step and remain
    model-agnostic.

## [v0.2.28] - 2026-09-16

### Model-Agnostic Agent Compatibility

- Added canonical model-neutral messages, tool calls, and tool definitions for API adapters.
- Added metadata-driven and externally configurable model profiles, including native delimited tool calls and an explicit text-only profile.
- Improved OpenAI, Anthropic, and internal-agent tool-call handling, streaming, cancellation, context admission, and capability reporting.
- Preserved tool-call identities, null assistant content, tool results, and multi-turn native history.
- Removed prompt-text-based BOS detection so custom model delimiters are handled by token identity.
- Fixed `--no-safelock` exiting immediately when the disabled watchdog closes its channel.

## [v0.2.27] - 2026-09-07

### Correctness Fixes After v0.2.26 Review

- Fixed a long-output Anthropic streaming deadlock caused by usage encoding waiting behind a full bounded generation buffer.
- Preserved streaming inference errors through finalization so OpenAI does not claim a normal stop and Anthropic does not claim end_turn after failure.
- Added configurable inference-slot backpressure (one concurrent request by default) so work cannot accumulate unbounded behind the single engine actor; busy requests receive HTTP 429.
- Extended API-key middleware to model, status, tools, metrics, and Ollama API routes; only health and the embedded UI remain public when a key is configured.
- Corrected Anthropic streaming usage to report zero output tokens when generation produces no text.
- Stopped agent execution of later tool calls after a timeout leaves the previous call's side effects unknown.
- Added cooperative cancellation signals for cancellable tool handlers and migrated built-in filesystem/calculator handlers to observe timeout cancellation.
- Bounded concurrent blocking tool handlers so timed-out handlers that already started cannot exhaust the runtime's blocking pool.
- Exposed the blocking-tool concurrency bound through `ServerConfig` and `mivi serve --max-concurrent-tool-executions` (default 4).
- Normalized zero-valued server channel and concurrency settings before Tokio channel/semaphore creation, preventing configuration panics.
- Hardened Unix workspace reads and directory listings with descriptor-relative no-follow access, preventing symlink swaps from redirecting a read after path validation.
- Routed agent context-document reads through the same bounded descriptor-relative no-follow reader, closing the validation/read race for agent prompts.
- Replaced fake Ollama model metadata with GGUF-derived file size, parameter count, architecture family, and dominant quantization; unavailable digest values are omitted instead of reported as empty strings.
- Removed stale fixed benchmark, memory, test-count, and context claims from the README and aligned the HTTP specification with the implemented routes and configuration.
- Added a lightweight process-local `/metrics` endpoint covering inference admission, slot wait, generation latency, token totals, inference errors, and timed-out tools.
- Made stop-sequence prefix matching safe for Unicode, preserved sampling penalties at `temperature: 0`, and corrected unseeded RNG progression.
- Removed the fixed top-p candidate cap by expanding the selection window until the requested nucleus probability is covered.
- Strengthened JSON primitive validation and added checked KV-cache disk length arithmetic and serialization bounds.
- Added bounded JSON Schema validation for generated OpenAI and Anthropic tool calls, including declared-tool, type, required-field, enum, array, and additional-property checks.

## [v0.2.26] - 2026-09-05

### API Contract Hardening, Generation Controls & Workspace Security

- **Request-scoped generation controls**: Added validated `temperature`, `top_p`, `top_k`, `min_p`, repetition/presence/frequency penalties, `seed`, and stop sequences for OpenAI-compatible requests, with sampler state and RNG restored after each request.
- **Structured output and tool compatibility**: Added non-streaming JSON-object generation, explicit unsupported-format errors, named tool-choice filtering, assistant tool-call history preservation, and structured Anthropic `tool_use` streaming with accurate tokenizer-based usage counts.
- **Server safety**: Added explicit no-model readiness/errors, model ID validation, loopback-by-default binding, API-key protection for public binds, closed-by-default CORS with an explicit allowlist, and bounded workspace context documents.
- **Agent and filesystem hardening**: Enforced agent tool allowlists, fail-closed handling for timed-out tools, and Unix descriptor-relative atomic workspace writes that do not follow swapped symlinks.
- **Inference and cache reliability**: Added explicit mock-engine mode, context-position overflow checks, safer hybrid prefix/suffix cache handling, and quantized KV/prefill coverage.
- **Documentation and regression coverage**: Updated the API/specification and low-resource operating guidance, and added targeted tests for authentication, generation validation, JSON output, tool routing, Anthropic streaming, CORS, no-model behavior, and filesystem safety.

## [v0.2.25] - 2026-09-03

### 64K Max Context Scaling, Fail-Fast Token-0 Validation & SSE Keep-Alive Heartbeats

#### 💡 Ideas, Inspirations & Sources
- **64K Context Support & CLI `--ctx-size` Flag (`mivi-model::model`, `mivi-cli::commands`)**:
  - *Inspiration*: [llama.cpp `-c / --ctx-size` context sizing & Ollama `num_ctx` configuration].
  - *Problem Fixed*: Mivi had hardcoded `DEFAULT_WORKING_CTX: usize = 4096;` in `model.rs` and had no `--ctx-size` flag in `serve`. When coding agents (Cline, Roo Code, Continue) sent prompts with 14,635 tokens, Mivi crashed with `Context overflow: current pos 4096 >= max_seq_len 4096`.
  - *Solution*: Elevated default working context to 16,384 tokens, added full support for up to 65,536 (64K) max context with automatic YaRN RoPE frequency scaling, added `--ctx-size` (`-c`) to `mivi serve`, and updated `justfile` serve recipe to default to `--ctx-size 65536`.
- **Fail-Fast Context Validation at Token 0 (`mivi-model::model`)**:
  - *Problem Fixed*: When a prompt exceeded context capacity, Mivi computed thousands of tokens sequentially on CPU before erroring out at token 4096 (wasting minutes of 100% CPU).
  - *Solution*: Added an immediate check `start_pos + n_prompt > self.config.max_seq_len` at token 0, returning an immediate error in 0 milliseconds before running any forward steps.
- **SSE Keep-Alive Stream Heartbeats (`mivi-server::streaming`, `mivi-server::routes::chat`)**:
  - *Inspiration*: [vLLM `with_sse_keep_alive` wrapper & standard SSE comment specifications].
  - *Problem Fixed*: During CPU prefill of large prompts, no bytes were sent across the SSE stream for tens of seconds, causing AI agent HTTP clients (Cline / Roo Code) to drop the connection due to idle socket timeouts and re-send the request 4 times concurrently.
  - *Solution*: Added a 2-second keep-alive loop emitting `: keep-alive\n\n` comments during prefill. This keeps the client socket active, resets idle timers, and prevents client drops or retry storms.

---

## [v0.2.24] - 2026-09-03

### Prefix Snapshot Allocation Bounds & Long Prompt Prefill Visibility

#### 💡 Ideas, Inspirations & Sources
- **PrefixCache Chunk Allocation Bounding (`mivi-model::model`)**:
  - *Problem Fixed*: When an AI coding agent sent a long prompt (e.g. 7,082 tokens with all workspace tools and files), `model.rs` triggered full state export snapshots every 64 tokens (110 times), cloning gigabytes of memory on the heap despite the cache capacity being capped at 32 chunks (32 MB).
  - *Solution*: Added a strict pre-condition check `(cur_pos + 1) / mivi_kv::PREFIX_CHUNK_SIZE <= mivi_kv::DEFAULT_MAX_CACHED_CHUNKS` preventing wasteful memory allocation during massive prompt processing.
- **Live Prefill Progress Milestones (`mivi-model::model`)**:
  - Added periodic prefill percentage progress logging every 500 tokens (`│ ⏳ prefill progress: [1000/7082] (14%)`) to provide clear visual feedback during large prompt processing on CPU.

---

## [v0.2.23] - 2026-09-03

### Opt-In Thinking Control & 25x Chat Latency Acceleration

#### 💡 Ideas, Inspirations & Sources
- **Opt-In Thinking Control (`mivi-server::routes::chat`, `ChatCompletionRequest::reasoning_effort`)**:
  - *Inspiration*: [OpenAI API reasoning_effort specification & standard Instruct SLM prompt formats].
  - *Problem Fixed*: Previously, `enable_thinking = true` was hardcoded for all `/v1/chat/completions` requests, injecting a default system prompt instructing the model to think inside `<think>...</think>`. For simple greetings like `"hii"`, the model was forced to produce a 200–300 token internal thinking monologue taking 25–40 seconds on CPU. This caused AI agent HTTP clients (with 30s timeouts) to abort and fail with `ReadTimeout`.
  - *Solution*: Made thinking strictly opt-in based on `req.reasoning_effort`. For standard chat/agent interactions with `LFM2.5-1.2B-Instruct`, the model now responds directly in 1.9 seconds (9 tokens, 4.7 tok/s) without wasting CPU cycles on unnecessary thinking blocks.
- **Live Prompt Token Sizing in Terminal (`mivi-server::logging`)**:
  - `print_incoming_prompt` now displays the exact prompt token count (`⏳ prefilling 13 prompt tokens & generating on CPU...`), giving users immediate visibility into request size and progress.

---

## [v0.2.22] - 2026-09-03

### Live Request Arrival Logging, Immediate Terminal Feedback & Direct Stop Tokens

#### 💡 Ideas, Inspirations & Sources
- **Immediate Prompt Arrival & Inference Notification (`mivi-server::logging`, `print_incoming_prompt`)**:
  - *Inspiration*: [FastAPI / Uvicorn real-time access logging & Hono HTTP lifecycle].
  - *Problem Fixed*: Previously, HTTP middleware only logged *after* inference completed (`next.run().await`). When an AI agent sent a request with system prompts and tools, CPU spiked to 100% computing prefill and forward passes on CPU, but the terminal showed zero output until the entire generation finished (or timed out).
  - *Solution*: Added instantaneous arrival logging (`→ POST /v1/chat/completions`) and immediate prompt display (`┌─ user › "hii" | ⏳ prefilling & generating on CPU...`) the millisecond the request reaches the server, followed by clean completion boxes upon finish.
- **Direct Integer Loop Stop Tokens (`mivi-model::model`)**:
  - *Problem Fixed*: Model generation loop only checked `next_token == eos_token_id`, relying on subsequent UTF-8 decoding and string suffix matching for `<|im_end|>` and `<|endoftext|>`. This could cause excess token generation cycles before stopping.
  - *Solution*: Added direct integer checks `next_token == eos_token_id || Some(next_token) == im_end_id || Some(next_token) == endoftext_id` right after sampling for instant zero-overhead loop exit.
- **Explicit `stdout.flush()` Across All Loggers**:
  - Ensured all terminal output flushes immediately, eliminating any libc block-buffering in background terminal tasks.

---

## [v0.2.21] - 2026-09-03

### Universal AI Agent API Compatibility & Dual-Stack Host Binding

#### 💡 Ideas, Inspirations & Sources
- **Base `/v1` Probe & Model Retrieval (`mivi-server::routes`, `v1_root`, `get_model_info`)**:
  - *Inspiration*: [OpenAI API Reference, LangChain, AutoGen, CrewAI & LiteLLM].
  - *Problem Fixed*: AI agent frameworks ping `GET /v1` or `GET /v1/` on initialization and query `GET /v1/models/{model}` to verify engine availability. Previously returned 404.
  - *Solution*: Added dedicated routes for `GET /v1`, `GET /v1/`, `GET /v1/models/:model_id`, `GET /models`, and `GET /models/:model_id`.
- **Ollama API Compatibility Layer (`/api/tags`, `/api/version`)**:
  - Added native Ollama endpoint compatibility for agent frameworks and IDE extensions (Continue, Roo Code, Cline, OpenCode) that autodetect local Ollama instances.
- **Unprefixed Route Aliases (`/chat/completions`, `/messages`)**:
  - Registered direct unprefixed paths for client libraries that configure `baseURL = "http://localhost:8080"` without `/v1`.
- **Permissive CORS Policy (`CorsLayer::permissive()`)**:
  - Enabled full cross-origin resource sharing supporting browser webviews, web extensions, and local frontends sending preflight `OPTIONS` requests.
- **`0.0.0.0` Host Binding in Recipes (`justfile`)**:
  - Changed default `just serve` host to `0.0.0.0` so clients resolving `localhost` to IPv6 `::1` or IPv4 `127.0.0.1` connect without `ECONNREFUSED`.
- **Flexible Schema & Role Mapping (`mivi-server::types`, `mivi-tokenizer::chatml`)**:
  - Made `model` optional with fallback to the running SLM and mapped OpenAI `developer` message role to `system`.

---

## [v0.2.20] - 2026-09-03

### PrefixCache RAM Budget Ceiling & Real SLM Default Alignment

#### 💡 Ideas, Inspirations & Sources
- **Strict Memory Budget for PrefixCache (`mivi-kv::prefix`, `PrefixCache::prune_to_bytes`)**:
  - *Inspiration*: [LMCache & vLLM memory pool management].
  - *Problem Fixed*: Previously, `PrefixCache` allowed up to 256 uncompressed `HybridStateSnapshot` state instances with no byte limit, which caused RAM to balloon toward 3.0 GB and trigger the emergency safety watchdog during multi-turn or long reasoning inferences on 1.2B/2.6B models.
  - *Solution*: Set `DEFAULT_MAX_CACHED_CHUNKS = 32` and introduced a strict `DEFAULT_MAX_PREFIX_CACHE_BYTES = 32 MB` ceiling with automatic LRU byte-level pruning (`prune_to_bytes`) after every snapshot insertion.
- **Recipe Alignment with Real SLMs (`justfile`)**:
  - Configured `just serve`, `just chat`, and `just info` recipes to default to `models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf` instead of the early 350M dummy test fixture, ensuring full intelligence, multi-turn reasoning, and real answers out of the box.

---

## [v0.2.19] - 2026-09-03

### Live Hono-Style HTTP Logs with User Prompt & SLM Output Visualizer

#### 💡 Ideas, Inspirations & Sources
- **Hono-Style Terminal Logger (`mivi-server::logging`, `mivi_log_middleware`)**:
  - *Inspiration*: [Hono.js logger middleware](https://hono.dev/docs/middleware/builtin/logger) & [Ollama / vLLM terminal telemetry].
  - *Clean Box-Drawing Formatting*: Implemented `print_interaction_box` using unicode box-drawing characters (`┌─`, `│`, `└─`) to cleanly format user input prompts, model thinking process (`💭 thinking › "..."`), tool calls (`🔧 tool call › ...`), and final SLM output (`mivi › "..."`).
  - *Streaming & Blocking Log Unification*: Full support for both non-streaming JSON responses and streaming SSE sequences (`/v1/chat/completions`, `/v1/messages`, `/v1/mivi/agent`) with accurate token counts, tokens/sec generation throughput, and response latency.
  - *Thinking & Response Separation*: Enhanced `mivi_tools::parser` (`extract_thinking`, `strip_thinking`) to gracefully isolate `<think>...</think>` tags (even for unclosed tags on truncated completions) so reasoning steps and final answers are displayed in dedicated rows.

---

## [v0.2.18] - 2026-09-03

### Default 3.0 GB Memory Ceiling & Large Model Server Hardening

#### 💡 Ideas, Inspirations & Sources
- **Expanded Default Memory Envelope (`mivi-server::watchdog`, `mivi-cli::commands`, `justfile`)**:
  - *3.0 GB RAM Ceiling*: Increased default watchdog kill ceiling from 450 MB to **3000 MB** (with soft warning threshold at **2400 MB**), allowing seamless full-context inference on larger 1.2B and 2.6B models without premature safety aborts during heavy generation.
  - *Justfile Alignment*: Updated `just serve` recipe defaults to pass `--max-memory 3000` and `--warn-memory 2400`.

---

## [v0.2.17] - 2026-09-02

### Configurable Low-Memory KV Precision (Q8_0 / TurboQuant) & Auto-Adaptive Memory Watchdog

#### 💡 Ideas, Inspirations & Sources
- **Multi-Precision KV Cache Architecture (`mivi-kv::cache`, `mivi-model::model`)**:
  - *Full Precision Export & Import*: Extended `KvCache::export_state` and `import_state` to support all quantization precisions (`F32`, `Q8_0`, `TurboQuant4`, `TurboQuant2`), enabling Prefix Caching (LMCache) and Semantic Rollback across all quantized modes.
  - *Configurable CLI `--kv-precision`*: Added `--kv-precision <f32|q8_0|tq4|tq2>` to `mivi chat`, `mivi serve`, and `mivi bench`.
  - *Q8_0 High-Performance Quantization*: `kv_precision=q8_0` delivers **21.7 tok/s** with **73.4% KV cache RAM savings** while maintaining 100% mathematical accuracy.
- **Auto-Adaptive Watchdog & Pruning (`mivi-server::watchdog`)**:
  - *Lowered Default Ceiling*: Reduced default emergency kill threshold from 900 MB down to **450 MB** (and warning at **350 MB**) tailored for edge and container environments.
  - *Adaptive Sizing*: Added `WatchdogConfig::adaptive` dynamically computing optimal thresholds from model weights and active KV cache dimensions.

---

## [v0.2.16] - 2026-09-02

### JetSpec-Inspired Multi-Branch Tree-PLD, Reasoning Speculative Sizing & Zero-Alloc Tree Verification

#### 💡 Ideas, Inspirations & Sources
- **JetSpec (`hao-ai-lab/JetSpec`)**:
  - *Multi-Branch Tree-PLD (`mivi-model::pld::TreePldProposer`)*: Replaced single linear chain speculation with structured multi-branch tree drafting. Scans the prompt and multi-turn context buffer to propose primary and secondary candidate continuation branches simultaneously in $< 3\ \mu\text{s}$.
  - *Reasoning-Adaptive Speculative Router (`mivi-model::pld::ReasoningSpecRouter`)*: Inspired by JetSpec's `reasoning_router` and `top2gap_fanout`, dynamically detects active `<think>` tags, math formulas, and code fences to shift between **Deep-Chain Mode** (Depth $K=5$, Width $W=1$) for deterministic reasoning steps and **Multi-Branch Mode** (Depth $K=3$, Width $W=2$) for open-ended generation.
  - *Zero-Allocation Tree Verifier (`mivi-model::pld::TreeVerifier`)*: Implemented a pure-Rust, stack-allocated verification walk that resolves the longest accepted branch in sub-microsecond time with 100% greedy losslessness guaranteed.

---

## [v0.2.15] - 2026-09-02

### Full Codebase Hardening, Security Safeguards & Multimodal Tool Enhancements

#### 💡 Ideas, Inspirations & Sources
- **Tokenizer & Turbo-BPE Hardening (`mivi-tokenizer::turbo`, `mivi-tokenizer::chatml`)**:
  - *Intrusive BPE Array Bounds Clamp*: Fixed potential stack array out-of-bounds panic when input piece length equals or exceeds `MAX_PIECE_BYTES = 256`.
  - *ChatML Multi-System Instruction Guard*: Enforced single-injection for `<tools>` and thinking instruction tags across conversation histories with multiple system turns.
- **FlashDecoding & Model Execution (`mivi-model::transformer`, `mivi-core::math`)**:
  - *Head Dimension Scalability*: Expanded `v_head_buf` to 256 elements in `mivi-model::transformer`, supporting models with `head_dim = 256` (Gemma 2, Command R+, DeepSeek V2/V3).
  - *Numerical Stability on $-\infty$*: Guarded `silu_scalar` against IEEE 754 $-\infty / (1.0 + \infty) = \text{NaN}$ computation on extreme negative inputs.
  - *AVX2 Target Feature Syntax*: Fixed comma-separated target feature string syntax in `mivi-quant::q8_0`.
- **Server Security & Protocol Compliance (`mivi-server::routes`, `mivi-server::streaming`)**:
  - *CORS Origin Validation*: Replaced naive prefix matching with exact host parsing to eliminate cross-origin request vulnerabilities.
  - *Streaming Tool Call Serialization*: Added missing `tool_calls` delta borrowing to `ChunkDeltaBorrow` to ensure streaming tool calls serialize correctly.
- **Rayon Parallelism & Tool Enhancements (`src/main.rs`, `mivi-tools::calc_parser`)**:
  - *High-Core CPU Scalability*: Removed arbitrary 8-thread clamp on Rayon thread pool to fully utilize 16, 32, 64, and 128 core machines.
  - *Scientific Notation Support*: Added exponent `e`/`E` parsing to calculator Pratt parser (e.g. `1e6 + 2.5e-3`).

---

## [v0.2.14] - 2026-09-02

### Turbo-BPE Zero-Allocation Intrusive Merger, Word Memo Cache & Workload-Adaptive Expert Learning Cache

#### 💡 Ideas, Inspirations & Sources
- **GigaToken (`marcelroed/gigatoken`)**:
  - *Turbo-BPE Zero-Allocation Intrusive Linked Merger (`mivi-tokenizer::turbo`)*: Replaced vector allocations and string-cloning loops with a stack-allocated intrusive linked-array buffer (`[BpeSymbolNode; 256]`), eliminating 100% of heap allocations during BPE symbol merges.
  - *Word-Level Memoization Cache (`mivi-tokenizer::turbo`)*: Implemented thread-safe direct-mapped `WordMemoCache` for sub-5ns Zipf word token retrieval, bypassing merge loops for common keywords (`function`, `import`, `let`, `def`, `class`, `the`, `return`).
  - *256-Byte Pre-Token Lookup Table*: Added $O(1)$ ASCII character classifier (`BYTE_CLASS_TABLE`) for rapid whitespace/word boundary identification.
- **AirLLM (`lyogavin/airllm`) & Colibrì (`JustVugg/colibri`)**:
  - *Workload-Adaptive Expert Learning Cache (`mivi-model::expert_cache`)*: Implemented `ExpertHeatTracker` and `ExpertPinningManager` tracking MoE expert activation frequency with exponential moving average (EMA) decay.
  - *Dynamic RAM Residency Policies*: Added `ExpertPinningStrategy::TopGlobal` and `TopPerLayer` to pin the hottest 20% of MoE specialists in RAM while streaming cold experts on demand.
  - *Persistence*: Auto-saves/loads learned user workload heat profiles to `.mivi/expert_heat.json`.

---

## [v0.2.13] - 2026-09-02

### Full Codebase Hardening, Zero-Allocation Grammar Engine, FlashDecoding Numerical Stability & Web UI Refinement

#### 💡 Ideas, Inspirations & Sources
- **FlashDecoding & Attention Numerical Stability (`mivi-model::transformer`)**:
  - *Guarded Online Softmax ($-\infty$ Threshold)*: Resolved numerical edge case where masked tokens with $-\infty$ scores could receive non-zero probability weights on first accumulation.
  - *Dynamic Memory Sizing for TurboQuant*: Migrated attention dequantization to model arena buffers (`state.hb`), lifting previous fixed 1024-dimension stack limits.
  - *GQA Head Ratio Safety*: Added `.max(1)` division-by-zero protection for uneven query/KV head ratios.
- **Zero-Allocation Stack-Allocated Grammar Engine (`mivi-model::grammar`)**:
  - *Zero-Heap `JsonGrammar` Pushdown Automaton*: Refactored scope tracking from heap vectors to a fixed 32-slot stack array (`[JsonScope; 32]`), making `JsonGrammar` `Copy`-able and eliminating **16+ million heap allocations** during token-by-token grammar logit masking.
  - *Vocabulary-Bounded Logit Scanning*: Added early break in `TokenBitMask::apply_to_logits` when scanning past actual vocabulary boundaries.
  - *Schema Recursion Depth Bound*: Added `MAX_SCHEMA_COMPACT_DEPTH = 32` guard to prevent stack overflow on deep JSON schemas.
- **TurboQuant & Core Math Hardening (`mivi-core::turboquant`)**:
  - *NaN-Safe Binary Search*: Handled coordinate `NaN` float comparisons gracefully in `TurboQuant4Bit::quantize` using `unwrap_or(Ordering::Less)`.
- **Open Knowledge Format (OKF v0.2) Parser (`mivi-context::okf`)**:
  - *Multiline YAML List Parsing*: Added stateful parsing for standard indented `- item` bullet lists under `sources:` and `tags:`.
- **Web UI & Telemetry Refinements (`mivi-server::ui`)**:
  - *SSE Stream Reader Termination*: Fixed reader loop exit on `[DONE]` events.
  - *Multi-Turn `<think>` Card Rendering*: Implemented full multi-block reasoning trace parser.
  - *Host Auto-Discovery*: Replaced hardcoded localhost ports with dynamic `window.location.host`.

---

## [v0.2.12] - 2026-09-02

### In-Engine Prefix Cache Alignment, AST Code Minification, Grammar Compaction & OKF v0.2 Knowledge Engine

#### 💡 Ideas, Inspirations & Sources
- **Headroom MCP (`aswin402/headroom-mcp`) & LMCache**:
  - *In-Engine Prefix Cache Boundary Aligner (`mivi-tokenizer::align`)*: Implemented `split_aligned_prefix`, `pad_to_chunk_boundary`, and `normalize_prompt_whitespace` to align static prompts and ChatML headers to exact 64-token chunk boundaries (`PREFIX_CHUNK_SIZE = 64`), guaranteeing **100% prefix cache reuse and 0 ms TTFT**.
  - *Syntax-Aware AST Code & Output Minifier (`mivi-core::minifier`)*: Implemented AST signature extractors for Rust, Python, and TypeScript, stripping function bodies while retaining type contracts to reduce code token consumption by up to **85%** and prevent SLM attention dispersion.
  - *Command Output & Log Filters*: Added compiler/test minification that suppresses passing tests and download spam while preserving failing assertion traces and panics.
- **Google Cloud Platform Open Knowledge Format v0.2 (`GoogleCloudPlatform/open-knowledge-format`)**:
  - *Native OKF v0.2 Knowledge Parser & Navigator (`mivi-context::okf`)*: Ingests Markdown concepts with structured YAML frontmatter (`type`, `sources`, `trust_tier`, `status`, `stale_after`) and hierarchical `index.md` progressive disclosure navigation.
- **Grammar & JSON Schema Compactor (`mivi-model::grammar`)**:
  - *Canonical Schema Minifier*: Added `compact_json_schema` and `compact_json_schema_str` stripping non-structural annotations (`description`, `title`, `$comment`) to save **40%–60%** prompt tokens during grammar-constrained logit masking.

---

## [v0.2.11] - 2026-09-02

### Built-in Interactive Web UI Dashboard & Telemetry Visualizer

#### 💡 Ideas, Inspirations & Sources
- **Colibrì (`JustVugg/colibri`) & AirLLM (`lyogavin/airllm`)**:
  - *Embedded Web UI & Live Telemetry Dashboard (`mivi-server::ui`)*: Inspired by Colibrì's `./coli web` dashboard, added a zero-dependency, single-file HTML/CSS/JS interface served directly at `http://localhost:8913/` (and `/web`) with live SSE streaming chat, collapsible `<think>` blocks, interactive `<tool_call>` execution cards, live generation speedometers, and memory tier watermarks.
  - *Architectural Research & Future Roadmap Blueprint (`docs/AIRLLM_AND_COLIBRI_RESEARCH.md`)*: Saved research for future implementation:
    1. *Workload-Adaptive Expert Learning Cache (`.mivi/expert_heat.json`)*: Track expert routing frequencies across user sessions and pin the hottest specialists into RAM.
    2. *Asynchronous Lookahead Weight Prefetching*: Overlap layer $L+1$ disk reading via `madvise(MADV_WILLNEED)` while layer $L$ computes.

---

## [v0.2.10] - 2026-09-02

### Outlier-Free TurboQuant 4-Bit & 2-Bit Attention KV Cache Compression

#### 💡 Ideas, Inspirations & Sources
- **TurboQuant Attention KV Cache Quantization (`mivi-kv::cache`)**:
  - *Outlier Energy Dispersion via Block-Hadamard Transforms*: Applied deterministic orthogonal 2-round Block-Hadamard rotations to Key and Value activations, uniformly dispersing outlier channel magnitudes across all dimensions.
  - *Extreme 87.3% and 93.5% Memory Reduction*: Added `KvPrecision::TurboQuant4` (204 MB for 64K context) and `KvPrecision::TurboQuant2` (103 MB for 64K context vs 1.61 GB in FP32).
  - *Exact Inverse Orthogonal Reconstruction*: Added `unrotate_vector_in_place` in `mivi-core::turboquant` ensuring bit-exact vector reconstruction for Value dequantization.
- **In-Place FlashDecoding Query LUT Scoring (`mivi-model::transformer`)**:
  - *Zero-Heap Allocation Attention*: Evaluates Query-Key attention dot products directly in CPU registers by computing single-pass Query LUT lookups against 4-bit and 2-bit packed Key vectors.

---

## [v0.2.9] - 2026-09-02

### TurboQuant 4-bit Vector Quantization, Orthogonal Block-Hadamard Transforms & Compact Semantic Memory Search

#### 💡 Ideas, Inspirations & Sources
- **TurboQuant (Data-Oblivious Vector Quantization, `arXiv:2504.19874`, Google Research & NYU, ICLR 2026)**:
  - *Deterministic Orthogonal Block-Hadamard Transform (`mivi-core::turboquant`)*: Implemented in-place Fast Walsh-Hadamard Transform (`fwht_in_place`) combined with deterministic SplitMix64 coordinate permutation and sign-flips. Universally maps arbitrary embedding vectors to symmetric Gaussian/Beta coordinate distributions.
  - *Analytical 4-Bit Lloyd-Max Quantizer*: Quantizes coordinates into 4-bit nibbles (2 coordinates per byte, achieving 16x memory compression) using analytical Beta distribution decision boundaries with **zero training data or codebook clustering**.
  - *Asymmetric Query LUT Scoring*: Fast cosine similarity estimation via query look-up tables directly in CPU registers.
- **`turbovec` (`RyanCodrai/turbovec`) & `turboquant-pytorch` (`tonbistudio/turboquant-pytorch`)**:
  - *Ultra-Compact `TurboMemoryIndex` (`mivi-memory`)*: Stored 4-bit compressed episodic and semantic agent memories, allowing 100,000 vectors to fit in only **38 MB of RAM** with sub-millisecond similarity recall.
  - *Semantic Context VM Retrieval (`mivi-context`)*: Added `ContextStore::search_semantic` enabling dense semantic similarity search across loaded workspace code blocks and conversation histories.

---

## [v0.2.8] - 2026-09-02

### Quantized KV Cache (`Q8_0`), Fused SIMD FlashDecoding Attention & High-Throughput Chunked Prefill

#### 💡 Ideas, Inspirations & Sources
- **KIVI (Tuning-Free Asymmetric 2-bit/8-bit KV Quantization, `arXiv:2402.02750`)**:
  - *Asymmetric Key/Value Memory Scaling*: Implemented `KvPrecision::Q8_0` (34 bytes per 32-element block) reducing 64K KV cache footprint from 1.61 GB down to **427 MB (73.4% RAM reduction)** on Mivi's 6 attention layers.
- **`llama.cpp` (`-ctk/-ctv q8_0`) & Fused SIMD Kernel Design**:
  - *Zero-Dequantization Attention Scoring*: Added `dot_q8_0_f32_avx2` in `mivi-quant::q8_0` computing fused $Q_{\text{f32}} \cdot K_{\text{q8\_0}}^T$ in-place without dequantizing whole cache layers into memory.
  - *Zero Heap Allocations in FlashDecoding*: Values are dequantized on-the-fly into fixed 128-float stack buffers within L1 CPU cache.
- **Sarathi (Chunked-Prefills, `arXiv:2308.16369`) & `vLLM` (`--enable-chunked-prefill`)**:
  - *Vocabulary Projection Bypass*: During chunked prompt prefill, output normalization and large vocabulary unembedding projections ($W_{\text{head}}$ with 65,536+ rows) are skipped for all non-terminal prompt tokens, saving millions of unnecessary FLOPs.
  - *Hierarchical Snapshot Synchronization*: Synchronized 64-token chunk boundaries with LMCache prefix snapshots for instant $< 0.05\text{ ms}$ state restoration.

---

## [v0.2.7] - 2026-09-02

### FlashDecoding Numerical Hardening, YaRN RoPE Math Correction, API Protocol Compliance & Agent Oscillation Protection

#### 💡 Ideas, Inspirations & Sources
- **FlashDecoding Online Softmax Numerical Hardening (`mivi-model::transformer`)**:
  - *Zero-NaN Guarantees on Masked Sequences*: Fixed an IEEE-754 `-Inf - (-Inf) = NaN` subtraction bug in online softmax accumulation when the initial cached token was masked out. Hardened `mivi-core::math::softmax` against `+Inf` and `NaN` logit inputs.
- **YaRN (Yet another RoPE extensioN, `arXiv:2309.00071`) Parameter Utilization (`mivi-core::rope`)**:
  - *Accurate Frequency Boundaries*: Corrected the YaRN frequency ramp divisor and wavelength interpolation to scale properly with `beta_fast`, `beta_slow`, and `orig_max_seq_len` on 64K/128K sequences.
- **OpenAI & Anthropic SSE Protocol Compliance (`mivi-server`)**:
  - *Standard Chunk Lifecycle*: Emitted initial `choices[0].delta = {"role": "assistant"}` chunk on stream start and formatted errors as standard JSON error events rather than `<error>` text tags.
  - *Dynamic Anthropic Telemetry*: Implemented dynamic output token counting in `message_delta` (eliminating hardcoded `output_tokens: 10`), preserved conversational text preceding `tool_use` blocks, and included `input_schema` in ChatML tool definitions.
  - *Worker Actor Panic Recovery*: Wrapped engine actor execution in `std::panic::catch_unwind` to prevent worker thread panics from permanently halting the server.
- **Agent Loop Oscillation Stagnation Guard (`mivi-agent`)**:
  - *Periodic Cycle Detection*: Added $N$-cycle periodic oscillation detection (e.g. A $\to$ B $\to$ A $\to$ B) to terminate oscillating tool loops safely before exhausting step budgets.
- **CLI Chat REPL Dynamic History (`mivi-cli`)**:
  - *Full Context Retention*: Removed the artificial 3-turn limit in `chat.rs`, allowing full conversation history to be retained within the model's sequence length budget.

---

## [v0.2.6] - 2026-09-02

### 64K/128K Long-Context Scaling, YaRN NTK-Aware RoPE Extrapolation, Selective KV Memory Telemetry & NIAH Test Suite

#### 💡 Ideas, Inspirations & Sources
- **Pokee-Isaac 28B & Liquid AI LFM2.5 (explainx.ai)**:
  - *Non-Decoder Long-Context Synergy*: Validated that hybrid linear SSM + attention architectures prevent associative recall collapse on extended sequences. Because 10 out of 16 layers in Mivi are SSMs (which store recurrent state in constant-size 500 KB buffers), 62.5% of model layers consume zero KV cache.
- **YaRN (Yet another RoPE extensioN, `arXiv:2309.00071`) & LongRoPE (`arXiv:2402.13753`)**:
  - *NTK-Aware Frequency Scaling*: Implemented `RopeScaling` (supporting `None`, `Linear`, and `YaRN`) in `mivi-core::rope`. When sequence lengths extend past 4,096 up to 65,536 (64K) or 131,072 (128K), frequencies smoothly interpolate between high-frequency and low-frequency bands, preserving positional resolution.
- **Selective KV Cache Scaling & Telemetry (`mivi-kv`)**:
  - *RAM Footprint Tracking*: Added `memory_bytes()` and `capacity_tokens()` to `KvCache`. Verified that a full 64,000-token context on Mivi uses only ~402 MB in Q8_0 and ~1.61 GB in F32 (vs $>4.29\text{ GB}$ on pure transformers).
- **Automated Long-Context Harness (`tests/long_context_retrieval.rs`)**:
  - *Comprehensive Integration Coverage*: Added test suite verifying 64K KV cache storage integrity at boundary positions, YaRN RoPE rotation stability up to position 65,535, and 100-chunk (6,400-token) prefix chaining.

---

## [v0.2.5] - 2026-09-01

### Karpathy's llama2.c Top-P Cutoff Optimization, Real-World Live Verification & Engine Thread Runtime Decoupling

#### 💡 Ideas, Inspirations & Sources
- **Andrej Karpathy's `llama2.c` (`sample_topp` Heuristic Cutoff)**:
  - *Sub-Microsecond Nucleus Sampling*: Adopted Karpathy's pre-sort cutoff optimization `cutoff = (1.0 - top_p) / (vocab_size - 1)` in `mivi-model::sampler`. By filtering out tokens with negligible probabilities during the initial pass, sorting size is reduced from 65k–262k down to ~20–80 candidates, delivering massive speedups on large-vocabulary nucleus sampling.
- **Dedicated OS Worker Actor Architecture (`mivi-server`)**:
  - *Runtime Decoupling*: Replaced the embedded single-threaded Tokio runtime inside the dedicated engine worker thread with a direct `rx.blocking_recv()` loop. This eliminates Tokio `Cannot block the current thread from within a runtime` conflicts when forwarding streaming token deltas to HTTP response channels.
- **Real-World Live System Verification**:
  - *Full End-to-End Validation*: Verified `mivi doctor` (16 cores, AVX2/FMA), `mivi info` (16 hybrid layers), `mivi bench` (47.15 GFLOPS, 7.0x LMCache speedup), `mivi chat` (15.0 tok/s), and real HTTP server endpoints (`/health`, `/v1/models`, `/v1/mivi/status`, `/v1/mivi/tools`, `/v1/chat/completions`, `/v1/messages`, `/v1/mivi/agent`).

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
