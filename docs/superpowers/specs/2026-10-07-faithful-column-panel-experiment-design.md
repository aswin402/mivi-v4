# Faithful projection column-panel experiment redesign

## Goal

Replace the ineffective experimental token-width selector with an explicit
column-panel-width experiment that preserves the production AVX2 accumulation
geometry. Compare 32-, 64-, and 128-column panels on the existing bounded
synthetic and local-real-weight operator cases. This remains benchmark-only;
production dispatch must not change and no performance gain is assumed.

## Evidence and motivation

The corrected token-width pilot found every variant slower than the restored
production baseline. For the Q6_K 2,048 × 8,192, batch-64 case, its unprofiled
selector medians were 2.34–2.39× baseline. Source inspection explains the
shape: the experimental loop handles one 8-lane vector across all columns at a
time, while the production paired-row kernel handles four vectors per row
together inside 128-column panels. That reduces independent accumulators and
repeats weight broadcasts/full-column traversals. See the corrected
[measurement report](../PROJECTION_LOCALITY_EVIDENCE_2026-10-06.md).

## Approaches considered

1. **Column-panel variants with the production microkernel (recommended).**
   Keep the production 32-token unroll, paired-row input reuse, ascending-column
   FMA order, and vary only the number of columns processed before storing the
   accumulators. This directly tests panel locality without repeating complete
   column scans for each 8-token vector.
2. **Keep token-width variants and add a 32-token microtile.** This can retain
   register use, but for the current batch-64 pilot several settings collapse
   into the same microtile schedule, so the selector may not describe a
   meaningful kernel difference.
3. **Retire the selector experiment.** The previous candidates were slower.
   This is the fallback if the column-panel variant cannot match the baseline
   operation order and performance closely enough to justify keeping it.

## Design

### Kernel and API

- Add a typed `ProjectionColumnTile` with only `Columns32`, `Columns64`, and
  `Columns128`. The private measurement case uses `column_tile: null|32|64|128`.
- Add explicit feature-gated quantized and profiled APIs using that type. Remove
  the newly introduced token-tile API and field rather than retain an
  experimental name whose meaning no longer matches its behavior. The
  experimental feature API is not a stable compatibility promise; document
  this correction in the next patch changelog.
- Keep ordinary `quantized_matmul_rows` and ordinary SIMD entry points on the
  exact existing production path. No selector, environment variable, model
  metadata, tensor name, or model identity may affect ordinary calls.
- Implement the paired-row experimental AVX2 path with the same four
  independent 8-lane accumulators per output row, two-row shared input loads,
  and ascending-column FMA sequence as production. Only the column-panel width
  varies. Keep the baseline 128-column operation as the reference control.
- Handle an odd final row through the unchanged ordinary single-row path; the
  experimental candidate is specifically the paired-row hot path. Do not route
  paired rows through a generic `Option`-branched hot loop.
- For non-AVX2 execution, preserve checked slices and exact operation order;
  selector widths remain explicit experiment parameters, not hardware or model
  heuristics.

### Measurement protocol

- Bump the private measurement manifest schema to 3 and rename the strict case
  field from `token_tile` to `column_tile`. Reject the removed field and unknown
  selectors; update the example, supervisor validation, and tests together.
- Compare a null-selector baseline with all three panel widths, with identical
  format, dimensions, batch, thread count, profile mode, and repetitions.
  Continue alternating case order and require exact full output-vector parity.
- Retain the existing bounded limits, private artifacts, synthetic control,
  and metadata-verified real-weight Q6_K batch-64 workload. No model download,
  real activations, full-model inference, cache flush, or concurrent workload.
- Treat unprofiled full-call paired wall time as primary. Profile stage times
  are explanatory only. A panel-128 result is the control for detecting
  accidental divergence from the production 128-column traversal.

### Tests and acceptance

- First add tests that fail until the renamed selector APIs/field exist. Cover
  all supported quant formats, batch and column tails, odd/even row counts,
  nonzero accumulators, exact `u32` equality, checked invalid inputs, profiled
  route identity, schema migration, unknown/legacy fields, pair ordering, and
  failure retention.
- Require exact output-bit equality for every accepted variant, including the
  128-column control. Check the ordinary API still uses the unmodified
  production path and that the odd final row remains on its ordinary path.
- Run focused core SIMD, quant locality/matmul, example, Python driver and
  supervisor tests sequentially; Cargo jobs 1, test threads 1, Rayon threads 2.
  Build only the opt-in release example. Do not run a workspace-wide
  check/build/test.
- Run the bounded fresh pilot only after parity tests pass. Preserve both prior
  sessions privately; exclude the withdrawn token-width measurements. Report
  all pairs, caveats and any regression. Keep no selector promoted regardless
  of operator-only results; further end-to-end work requires the P1-A gate.
- After accepted implementation and measurement, bump the then-current patch
  version once, update the changelog with the sources and negative/positive
  evidence, run the scoped validations, and push the existing branch non-force.

## Non-goals

No default dispatch or threshold changes, automatic tuning, model-specific
rules, arithmetic/precision changes, cache or scratch ownership redesign,
server/client changes, model-quality claims, or production promotion.

## Inspirations and sources

- The corrected Mivi [projection locality evidence](../PROJECTION_LOCALITY_EVIDENCE_2026-10-06.md)
  supplies the negative result and diagnostic evidence for this redesign.
- The preserved AVX2/FMA operation structure follows Mivi's existing paired
  projection kernel; the pinned [GGML CPU implementation](https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/ggml/src/ggml-cpu/ggml-cpu.c)
  is reference context only. No code is copied.
- Matched alternating comparisons and explicit negative-result reporting draw
  inspiration from [Colibri's benchmark methodology](https://github.com/JustVugg/colibri/blob/main/docs/benchmarking.md).

## Self-review

- Scope is one experimental dimension: column-panel width. Production routing,
  arithmetic, model behavior, and unrelated P1 tasks are explicitly excluded.
- Terminology is consistent at each boundary: typed Rust `ProjectionColumnTile`,
  JSON `column_tile`, with null reserved for the baseline.
- The schema version changes because the strict private manifest field changes;
  old token-width manifests must fail validation instead of being reinterpreted.
- No placeholder, unspecified benchmark case, model-specific constant, or
  unbounded test/build command remains.
