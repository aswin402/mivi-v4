# Opt-in batch-prefill scratch: approved stage 1

The user approved the two-stage proposal: first reusable caller-owned scratch,
then a separately measured four-output-row accumulation experiment. This document
covers only stage 1. Default model/server dispatch and the old batch API stay intact.

## Contract

- Feature `batch-scratch-experiment` exposes `BatchProjectionScratch::new` and
  `quantized_matmul_rows_with_scratch` in mivi-quant. No model-name selectors.
- Caller declares maximum batch, rows, columns and worker partitions. Allocation
  is fallible, checked for element/byte overflow, and performed only at construction.
- Scratch owns transposed inputs, row-major outputs, and disjoint decoded-weight
  and row-output buffers per partition. Calls never resize these buffers.
- Validate formats, input/output/weight sizes and scratch capacity before writes.
  Empty work and batch-one delegate semantics match the established batch API;
  they do not require scratch capacity. Other calls use the baseline partition
  boundaries, dot/FMA order, paired rows and SIMD/portable helpers unchanged.
- If a caller moves the workspace into a larger Rayon pool, insufficient worker
  capacity is an error, not silently dropped output or an implicit allocation.
- Scratch may contain stale values; active output regions are zeroed as required.
  External output tails and all input buffers remain untouched.

## Gates and limits

Bit-exact comparisons across all six supported formats, batch boundaries,
odd/parallel rows, shape/format reuse, error-before-write and overflow controls.
One Cargo job; Rayon pools of one and two threads only. No large model run yet.
Persistent pointers/capacities plus source inspection verify workspace reuse;
they do not establish zero allocations inside Rayon's scheduler or matvec delegate.
No latency or memory improvement is claimed until separately measured.

## Ideas, inspirations and sources

Derived from Mivi's existing checked batch kernel and
[projection-cost evidence](../../PROJECTION_COST_EVIDENCE_2026-10-05.md).
[Agent-sized evidence](../../Q4_AGENT_SIZED_EVIDENCE_2026-10-07.md) explains why
prefill, not just decode, needs attention. Separating changes and retaining negative
results follows [Colibri benchmarking](https://github.com/JustVugg/colibri/blob/main/docs/benchmarking.md).
No external inference source is copied. Scratch removes repeated buffer ownership
costs, not the dominant accumulation work; four-row blocking remains stage 2.
