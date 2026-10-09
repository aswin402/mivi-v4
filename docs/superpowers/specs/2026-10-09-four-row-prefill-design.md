# Four-row faithful prefill experiment: approved stage 2

Continue the user-approved scratch-then-four-row design. Ordinary batch dispatch,
scratch-only control, precision and model/server behavior remain unchanged.

## Architecture

`batch-four-row-experiment` depends on the existing scratch feature and forwards
to an isolated mivi-core SIMD feature. A checked safe core wrapper accepts four
contiguous output/decoded-weight rows. On AVX2/FMA and batches >=32, a four-row
kernel shares two activation vectors across eight accumulators (16 batch lanes).
Columns advance in ascending order in panels of 128, matching established paired
arithmetic. Eight-lane and scalar tails preserve fused/nonfused baseline behavior.
Portable/small-batch fallback delegates to the existing pair helpers.

The quant candidate uses a separately constructed four-row workspace, preserving
the scratch-only constructor's two-row memory footprint. Full groups of four are
computed within existing worker partition boundaries; partial groups and batches
below 32 use the existing scratch-only compute helper. Validation precedes writes,
including decoded-workspace capacity. No model names, automatic promotion or CLI
switch is introduced.

## Safety approval gate

The register-blocked AVX2 implementation requires new `unsafe` intrinsics. The
Rust Core Specialist skill requires explicit user permission before adding them.
The user approved this bounded kernel on 2026-10-09 after the failing stub test.
Contain intrinsics in a private target-feature function, call it only
after AVX2/FMA detection and checked slice bounds, and document every safety invariant.

## Validation and measurement

Observe a failing core output-parity test against a stub before implementing SIMD.
Observe a failing quant candidate test before implementing the projection API.
Compare every output bit across six formats, full/partial rows and batches including
1/2/8/9/31/32/33/63/64/65, one/two-thread pools, stale scratch and error controls.
Core tests also exercise nonzero initial accumulation, cancellation-sensitive values,
short slices, overflow, zero work and untouched tails.

A bounded ignored operator diagnostic compares ordinary, scratch-only and four-row
paths with generated finite activations, full output parity, one warmup triple and
six measured triples using all six order permutations, three calls per member.
Evaluate synthetic Q8_0 and one metadata-selected real Q4_K/Q6_K matrix per format
from `MIVI_TEST_MODEL`. Declare cache state (warm process/model pages), all raw times,
negative cases and limits. No model inference, agent TTFT, RSS or quality claim.
Credible operator evidence is required before a later expensive model integration.

## Sources and inspiration

Mivi's existing pair-panel kernel and
[scratch evidence](../../BATCH_SCRATCH_EVIDENCE_2026-10-09.md) supply the faithful
control. [Projection-cost evidence](../../PROJECTION_COST_EVIDENCE_2026-10-05.md)
identified accumulation as dominant. Separating variables, alternating order and
retaining negative cases follows [Colibri benchmarking](https://github.com/JustVugg/colibri/blob/main/docs/benchmarking.md).
No external inference source is copied. The register-blocking hypothesis is not
a speedup promise; retain the experiment default-off regardless of this pilot.
