# Q4 batch-32 projection stage replay design — 2026-10-10

## Approval and scope

The user approved test-only stage measurements after reproducing the Q4 batch-32
regression and then approved this written design before the implementation plan/code.
This document makes that approach concrete. No production kernel,
precision, default dispatch, model integration, dependencies or new unsafe code
changes are in scope.

## Problem and alternatives

[v0.2.78 evidence](../../FOUR_ROW_ISOLATION_CAPTURE_EVIDENCE_2026-10-10.md)
contains an 11.61% captured Q4 batch-32 regression relative to scratch-only.
The subsequent read-only repeat passed complete bits but showed a smaller 3.26%
paired median regression, with mixed pairs. Generated 2,048-column/batch32
accumulation was 16.44% slower in that repeat. These observations motivate
stage attribution; neither establishes a hardware root cause.

Approved approach: safe, test-only stage replay using existing dequantization and
SIMD helpers, validated against unchanged quant APIs. It is portable and preserves
default kernels, but per-group clocks perturb work and duplicated orchestration
can drift; strict complete-output/reference checks are mandatory.

Alternative: hardware sampling of existing kernels. It may identify stalls or
compiled instruction behavior, but requires host permissions/tool support and
does not replace source-level stage accounting. Do not change register blocking
based on source broadcast counts or these pilot ratios alone.

## File boundaries

- New test-only child
  `crates/mivi-quant/src/batch_scratch/four_row/faithful_diagnostics/captured/stages.rs`:
  checked stage replay, profile accounting, correctness tests and ignored live test.
- Existing `faithful_diagnostics/captured.rs`: register child and extract private
  capture preparation/origin-gate helpers shared by existing and new live tests.
  Preserve existing prompt byte cap, metadata BOS policy, execution conditions and
  capture selection. Existing replay timing/order semantics remain unchanged.
- New plan/evidence documents and, on completed verification, sourced changelog
  plus workspace/lock patch bump from 0.2.78 to 0.2.79.
- Leave core/production quant/model/server functions untouched and preserve the
  user's unrelated .gitignore edit.

## Capture, formats and bounds

Reuse the memory-only one-tile production-helper walker. Context65, effective
prompt IDs32/64, no adapters, UTF-8 prompt <=64KiB before tokenization. Compare all
complete finite final-logit bits against normal production chunked prefill before
any case timings. Candidates never feed into model inference.

Select first executed eligible Q4_K/Q6_K FFN projections by metadata/wire type;
nonzero columns <=16,384 aligned256, nonzero rows <=8,192. Retain complete matrices.
For the first bounded stage diagnostic, additionally require both two-thread worker
partitions to contain a positive multiple of four rows. Fail explicitly when a
selected shape cannot satisfy this rather than trimming rows. This avoids partial
group ambiguity; tail/other-format support is not claimed. The local selected
8,192x2,048 and 2,048x8,192 matrices satisfy it. No model-name dispatch.

Check supported formats, dimensions, checked products, exact input/output/weight
coverage, and replay buffer capacities before writes. Reuse allocated test-owned
transposed/output and two-/four-row worker buffers throughout each case. No growth
inside accepted calls. Allocation failure/overflow rejects the case.

## Replay stages and arithmetic

Mirror the two existing worker partitions, row order, group sizes and zero timing:
scratch-only decodes first row, zeroes pair output, decodes second row, accumulates;
four-row decodes all four rows, zeroes group output, accumulates. Call unchanged
`dequantize_slice`, pair SIMD and four-row wrappers. Preserve F32 arithmetic and
column order. No new scalar/SIMD implementation or precision conversion.

Record validation wall, input-transpose wall, parallel rows-region wall,
output-layout wall, total call wall and unclassified wall. Each disjoint worker
owns a report containing row/group/decode/helper-call counts and separate elapsed
decode, output-zero and accumulation measurements. Use checked integer duration
conversion/sums; reject overflow/accounting inconsistencies. Buffer construction
stays outside call timing. Include Rayon scheduling in rows-region wall.

Worker elapsed measurements overlap and are not CPU time or wall-time percentages.
Never add worker sums to serial wall stages. Wall-stage accounting must not exceed
total call wall; retain a nonnegative residual. A zero-duration tiny stage is valid,
not an invented timer failure. Timers remain around the operations they name;
logging and exact comparisons occur outside timers.

## Controls, timing and gates

For every accepted case, construct all buffers and compute a finite ordinary
reference outside timing. Both untimed stage-replay outputs must match every
reference bit before accepted timing begins.

Four routes: unchanged scratch-only S, unchanged four-row F, profiled stage replay
PS, profiled stage replay PF. One warmup quadruple and eight measured quadruples
use these balanced orders:
`[0,1,2,3], [3,2,1,0], [1,2,3,0], [0,3,2,1],
[2,3,0,1], [1,0,3,2], [3,0,1,2], [2,1,0,3]`.
Each member makes one complete projection; compare every output bit/finiteness
after its timer. Each route occupies each position twice among measured rounds;
reversal pairs balance relative order. This is 144 timed full projections over
four format/batch cases, plus four references and eight untimed replay gates.

Report S/F uninstrumented paired ratios independently from PS/PF instrumentation.
Quantify profiler disturbance PS/S and PF/F from within-quadruple comparisons.
Profile stages are observations of replay, not direct timings of the unchanged
production functions or proof that their compiled code has identical performance.

Warm model/process pages, resident scratch; no cache flush, prefix restore, thread
pinning or thermal control. Capture/model work is outside projection timers.
Keep all measured/warmup metadata/timings and negative cases; no prompt IDs,
activation values, weight contents or output values are persisted/printed.

## Verification and acceptance

Cargo jobs1, Rayon threads2, serial harness, offline scoped commands; no workspace-
wide check/build/test. One live diagnostic command timeout240s, using an existing
absolute model path; no download. No simultaneous Cargo/live timing command.

TDD controls cover shape/buffer/overflow rejection before writes, complete
two-/four-row replay bits for nonzero finite Q4_K/Q6_K fixtures at batches32/64,
poisoned buffer reuse, worker/count accounting, and timing report consistency.
Observe valid-case RED before implementation. Shared capture helper extraction
also requires scoped existing diagnostic regression checks.

GPT-6 Luna/high performs read-only source/evidence review; main alone runs Cargo.
Publish a patch release only after final-version scoped feature-enabled/off checks,
live exact gates, formatting/diff checks, and a sourced evidence report. A negative
performance result is accepted evidence, not a reason to relax numerical gates.

## Decision boundary and sources

A stage replay may narrow the hypothesis to accumulation, decode or orchestration,
but cannot confirm cache misses, broadcast throughput, spills or instruction counts.
If profiler disturbance is large or stage attribution inconsistent, report it as
inconclusive and retain default-off. Kernel changes/model-workspace integration,
candidate model/KV/SSM parity, agent latency and RSS remain separately approved work.

Ideas/sources: Mivi's unchanged scratch/four-row implementations, production-helper
capture and [projection-cost accounting](../../PROJECTION_COST_EVIDENCE_2026-10-05.md).
Balanced comparisons, cache declarations and retained negatives follow
[Colibri benchmarking](https://github.com/JustVugg/colibri/blob/main/docs/benchmarking.md).
No external inference implementation is copied.
