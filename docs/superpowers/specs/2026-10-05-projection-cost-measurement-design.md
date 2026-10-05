# Projection cost measurement design

Status: written spec approved by the user on 2026-10-05. This is the first
measurement-only slice of P1-A, not approval to promote a new kernel.

## Goal and evidence

Separate the costs inside native batched projections before choosing a faithful
scratch/locality optimization. The [Phase 0 evidence](../../CPU_RUNTIME_EVIDENCE_2026-10-05.md)
attributes 72.61% of counted long-control forward-stage time to projections/FFNs,
but does not distinguish allocation, transposes, weight decoding or computation.
Short/medium cross-engine divergence remains unresolved. This work measures the
existing native arithmetic; it does not claim to resolve that divergence.

## Current code and alternatives

`crates/mivi-quant/src/lib.rs::quantized_matmul_rows` validates shapes, dispatches
batch 1 to matvec, and returns early for empty work. Batches 2–8 use per-input
dots; larger batches transpose inputs and accumulate across batch. Batches at
least 32 decode paired rows. Rayon divides sufficiently large output-row ranges.
The function allocates transposed inputs and row-major outputs; each worker also
allocates decoded-row scratch, plus a reusable row-output buffer on the unpaired path.
It finally converts the output layout into the caller's batch-major buffer.

`crates/mivi-model/src/prefill.rs::TileActivations` already owns reusable model
activation buffers, but not these quant-kernel temporaries.

Chosen approach: instrument the existing work with opt-in diagnostics and compare
profiled/unprofiled calls. Immediate scratch reuse would assume allocation is
important; a blocked-kernel rewrite would combine measurement with arithmetic
and locality changes. Both alternatives are deferred until evidence exists.

## Architecture and boundaries

Keep the existing checked `quantized_matmul_rows` interface and default behavior.
Add non-default quant diagnostics, with a feature-gated diagnostic entry point
sharing the same validation, dispatch and arithmetic implementation. Default
calls must not read clocks or collect measurements. No global profiling flag,
mutable global scratch, allocator replacement or model-name dispatch is allowed.
Keep profiling definitions in a small module if they would obscure the kernel.

Record call wall time, validation/dispatch, input/output-buffer allocation and
initialization, input transpose, worker scratch allocation/initialization,
weight decoding, accumulation, output zeroing/copying and final layout conversion.
Unclassified time remains explicit. Allocation timers include initialization;
they do not isolate allocator internals. Cold/warm process and page-cache
conditions are disclosed separately.

Batch 1 retains its matvec delegation: report delegated wall time, with internal
batch-only stages marked unavailable rather than measured zero. Empty work
reports validation and no executed compute stages.

Parallel worker times are summed work durations, not elapsed request time.
Report the parallel region's wall duration separately; do not add summed worker
durations to serial elapsed stages or interpret them as additive wall shares.
Fine per-row timers can be expensive, so only diagnostic calls collect them and
timer overhead is evaluated against matched unprofiled runs.

The bounded driver accepts an explicit private manifest. It records schema,
format, dimensions, thread count, accumulation branch, binary/revision/model
hashes where applicable, individual sample durations and output agreement.
Existing checked shape logic remains authoritative. Invalid formats, alignment,
buffer lengths and multiplication overflow return errors before output mutation.
Driver limits are checked before large allocation or mapping. Output artifacts
must not overwrite an existing experiment directory.

## Workloads and safety

- Synthetic cases use deterministic nonzero finite values, valid quant blocks,
  odd output rows and batches 1/2/8/9/32/64/65. Cover F32/F16/BF16/Q8_0/Q4_K/Q6_K
  where supported by the existing kernel, including both serial and Rayon paths.
- Real cases use explicit GGUF tensor selection and metadata-derived shapes/types;
  no fixed architecture dimensions, assumed filename format or automatic download.
  Use deterministic synthetic F32 activations at those shapes and label them as
  such, not captured real-model activations. Do not run the full model to collect
  activations in this measurement slice.
- Manifest controls case selection, repetition count and limits. Default pilot:
  three measured pairs per selected case, alternating profiled/unprofiled order.
  Warmup is separately labeled and excluded; no kernel result cache is permitted.
- Cargo jobs 1; Rust test threads 1; inference/Rayon threads 2. One build or
  measurement workload at a time. No full-workspace check/build/test.
- Supervisor pilot limits: 180 seconds per child, 2 GiB sampled child-tree RSS,
  64 MiB aggregate artifacts, 15 minutes per explicitly selected pilot session.
  Reuse the existing supervisor rather than inventing a second process watchdog.
  Retain timeouts/errors and verify cleanup. Limit sampling is not a peak-RSS proof.
- Private experiment directories/files use 0700/0600. Raw weights, activations,
  IDs, text, local paths and logs stay outside Git. Commit only redacted findings
  and synthetic test code/data that do not originate in private model tensors.

## Acceptance and tests

Profiled and unprofiled execution must produce bit-identical outputs on the same
host/dispatch for each valid case: clocks must not change arithmetic or work
ordering. Cross-branch comparisons against matvec use the existing numerical
policy and separately report maximum error; branch outputs are not assumed
bit-identical to one another. Non-finite values cannot silently pass comparisons.

Tests cover empty work, batch-1 delegation, all accumulation branches, odd row
tails, supported formats, short input/output/weight buffers, misalignment and
overflow. Check timing fields are finite/nonnegative, schema is explicit, and
parallel work counters cannot be mistaken for serial wall totals. Default builds
must remain usable without diagnostic features. Driver tests cover manifest
rejection, bounds, collision refusal, privacy and failure retention.

Report three-sample medians/ranges for the pilot, not p95 or universal speedups.
If instrumentation materially changes timings, mark detailed attribution as
perturbed and use matched unprofiled totals for decisions. Do not infer a
dominant cost from timer noise or summed overlapping workers.

The deliverable is scoped tests, an opt-in diagnostic path, a bounded pilot and
one redacted decision: scratch reuse, locality work, or no justified change yet.
Caller-owned scratch integration and production promotion are subsequent tasks
with their own numerical/request regression gates. This slice changes no
activation precision, sampler behavior, default tile size or provider API.

## Release and sources

The design checkpoint is not a completed feature release. Apply the user's
patch-version increment, attributed changelog and non-force GitHub publication
when the implemented measurement slice passes its acceptance gates.

Ideas and sources: existing Mivi checked kernel and activation workspace;
[master P1-A roadmap](../plans/2026-10-02-native-cpu-agent-improvements.md);
[Phase 0 measured findings](../../CPU_RUNTIME_EVIDENCE_2026-10-05.md);
[Colibri measurement methodology](https://github.com/JustVugg/colibri/blob/main/docs/benchmarking.md)
and [pinned GGML operand-path context](https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/ggml/src/ggml-cpu/ggml-cpu.c).
These motivate measurement discipline; no external inference kernel is copied.
