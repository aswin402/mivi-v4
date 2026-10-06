# Faithful projection locality experiment

## Goal

Test whether changing the traversal width of the existing AVX2/FMA large-batch
projection kernel improves locality or measured latency, without changing the
mathematical operation, per-output accumulation order, activation precision, or
default runtime behavior. The experiment is exploratory; it does not promise a
speedup and does not authorize promoting a variant to the production default.

## Context

The bounded projection-cost pilot found accumulation to be the largest measured
worker-work category in selected Q6_K and Q4_K cases. Worker-work sums overlap
across threads, and the pilot did not identify cache locality as the cause.
Activations were synthetic, the case set was narrow, and the host had load and
swap pressure. The pilot therefore motivates a hypothesis test only.

The current large-batch AVX2 kernel traverses the transposed input by token
blocks, retaining independent vector accumulators and processing columns in
order. The experiment will compare alternate token-block widths for batch sizes
that use this path. Each token/output lane must still visit columns in exactly
the same order with the same FMA operation. Remainders must be handled without
dropping or duplicating tokens.

## Approaches considered

1. **Benchmark-only traversal-width variants (recommended).** Add an explicit
   experimental entry point/configuration for the SIMD kernel and compare a
   small predeclared set of widths, including the current width. Keep normal
   dispatch untouched. This isolates traversal width, though generated code and
   register pressure may limit or reverse any locality benefit.
2. **Change the SIMD accumulation strategy.** This could expose more instruction
   level parallelism, but risks changing rounding/bit patterns and confounds the
   locality hypothesis. Deferred.
3. **Collect deeper cache-counter evidence first.** This avoids kernel changes,
   but does not directly test the selected traversal-width hypothesis and may
   not be portable or available on the target host. It can supplement, not
   replace, the controlled variant comparison.

## Design

The experiment adds a benchmark/test-only route to run the existing traversal
and alternate explicit token-block widths on identical inputs. The public
checked `quantized_matmul_rows` API and default production dispatch remain
unchanged. No model-name, GGUF-name, tensor-name, or laptop-specific dispatch is
introduced. The tested widths are benchmark parameters, not inferred model
properties. Only formats and batch cases already supported by this kernel are
included; unsupported widths or shapes must return a clear error or use the
existing baseline route, never silently select a different production path.

Correctness compares every variant with the current baseline for exact output
bits across deterministic finite inputs, supported block-aligned formats,
batch tails, and odd row counts. If the code structure cannot guarantee the same
per-output operation order, the variant is not considered faithful and must not
be benchmarked as such. Tests cover the smallest supported batch for the path,
the block boundary, non-multiple tails, and invalid arguments. The existing
batch-one path and small-batch path remain outside the kernel change.

Performance evaluation uses a fixed manifest and paired baseline/variant order,
with warmups, repetitions, full-call wall time, inner accumulation timing,
thread count held at two, and the existing bounded-child privacy/resource
limits. It reuses only the previously selected synthetic and real-weight cases;
it makes no real-model inference or agent-quality claim. Retain failed attempts
and report all matched pairs, medians/ranges, host conditions, and artifact
validation. Worker-work sums are not treated as wall-time savings. No model
downloads or concurrent benchmark workloads are permitted.

## Gates and outcomes

- Exact bitwise parity for every supported test case, including tails.
- No change to production/default dispatch in this experiment.
- No claim of improvement based only on profiled substage work; paired
  unprofiled full-call measurements are primary.
- Report regressions and inconclusive/noisy results as such. If no improvement
  exceeds observed spread, retain the variant as experimental or discard it.
- Any later promotion requires a separate decision based on end-to-end request
  measurements, representative model behavior, short-case/RSS checks, and the
  roadmap P1-A exit gate.

## Inspiration and sources

- Mivi bounded projection-cost pilot, `docs/PROJECTION_COST_EVIDENCE_2026-10-05.md`
  (local measurement evidence; exploratory conclusion only).
- Existing Mivi large-batch accumulation implementation in
  `crates/mivi-quant/src/lib.rs` and `crates/mivi-core/src/simd/avx2.rs`.
- GGML reference kernel and format/layout conventions, pinned in the existing
  projection measurement plan: <https://github.com/ggml-org/llama.cpp>.
- Colibri's focus on low-level CPU inference kernels and measured kernel
  specialization, used as general inspiration only; no code is copied:
  <https://github.com/JustVugg/colibri>.

## Non-goals

No default tile tuning, model-specific heuristics, activation quantization,
caller scratch redesign, SIMD arithmetic redesign, multi-thread policy change,
server integration, model download, cross-engine parity claim, or production
kernel promotion is included.
