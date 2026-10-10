# Experimental parallel F32 attention

Date: 2026-10-10. Baseline: v0.2.82, b2db472.
Status: proposed design, awaiting written-spec review. No implementation or
production promotion is authorized by this document alone.

## Motivation and scope

The latest existing-binary profile measured cold prefill at 13.76s for538
effective tokens and59.34s for2068. The causal-attention region increased
from0.77s to13.30s. This region includes normalization, RoPE and KV insertion,
not just the scan. One profiled observation per size is not a stable benchmark.

`compute_gqa_attention` currently processes query heads serially and scans each
head's causal prefix. Dot products and value accumulation already use SIMD.
The first experiment changes head scheduling only. Projection kernels, model
templates, tool handling and Minicode are outside scope.

## Alternatives and decision

1. Recommended: an explicit default-off F32-only parallel-head candidate. Heads
   write independent output slices while reading shared immutable Q/K/V. Preserve
   within-head traversal/arithmetic; test the scheduling hypothesis directly.
2. Blocked multi-query attention: potentially better KV reuse, but a larger
   algorithm and floating-point-order change. Defer until this smaller experiment
   establishes a trustworthy comparison boundary.
3. Projection optimization first: addresses the larger measured FFN category,
   but does not investigate the sharply growing attention region. Keep it as the
   separate next experiment, using existing scratch/four-row evidence.

## Candidate architecture

- Add a non-default `parallel-attention-experiment` feature in mivi-model with
  optional use of the existing workspace Rayon dependency. No new external crate.
- Expose selection only through a feature-gated diagnostic model option. Enabling
  the feature alone does not select the candidate. Normal CLI/server defaults and
  the established serial implementation remain unchanged.
- F32 only in this slice. Q8_0/TurboQuant modes use the established serial path;
  TurboQuant currently mutates shared `state.hb` scratch. Do not share that scratch
  across workers or introduce new unsafe code to force concurrency.
- Use safe disjoint mutable output slices and immutable query/KV access. Validate
  nonzero dimensions, head divisibility, required Q/output lengths, layer mapping,
  KV width and causal position before candidate writes. Reuse checked KV getters
  or a safe bounded view; do not add unchecked access or custom Send/Sync impls.
- Partition contiguous query heads into at most the active pool's worker count.
  Use the caller's existing pool, never a new pool per token or layer. Single-worker
  and single-head cases remain serial. No model-name/filename heuristics, hidden
  startup tuning, arbitrary sequence cutoff or thread-count increase.
- Insert the current query's KV before scheduling its heads, join all head work
  before consuming outputs, and retain sequential query-row/recurrent-state order.
- Preserve ascending KV positions, GQA head mapping, online-softmax branches,
  SIMD calls, score scaling and output normalization within each head. Parallelism
  changes ownership/scheduling, not reduction order or precision.
- Keep baseline and candidate independently selectable for comparison. Initially
  retain baseline arithmetic as the reference rather than replacing it with a
  helper shared by both tested paths.

## Correctness and error gates

Use meaningful focused tests with nonuniform finite inputs, GQA and ordinary MHA,
multiple causal positions, nonzero output sentinels, head/worker partition tails,
one-/two-thread pools and repeated calls. Compare every output bit to the current
serial reference on this host, not only argmax or selected elements.

Invalid dimensions/position/layer must fail before candidate output mutation.
Test unsupported-precision fallback independently. No changed KV or scratch state
is allowed from read-only attention evaluation. Same-host bit agreement does not
establish cross-ISA reproducibility or parity with another engine.

Before timing a model-integrated candidate, compare complete logits, generated
IDs and carried KV/SSM state for short multi-tile, reset, continuation and exact
prefix-restoration fixtures. A mismatch blocks promotion and must be investigated
without relaxing tolerances to make the experiment pass.

## Bounded measurement and resource contract

Cargo jobs1, inference pool2, serial harness; no concurrent build/inference runs.
Use focused mivi-model tests/examples only, offline where dependencies permit.
Start with synthetic head-scan parity and a small balanced operator timing pilot.
If correctness passes, use a diagnostic runner with explicit baseline/candidate
selection on the same model and prompt IDs, F32 KV, context4096 and tile64.

Measure 512- and2048-token synthetic prompts, with three alternating paired
unprofiled observations per size and separate profiled controls. Each child has an
external180s wall bound; session bound1200s. Stop on parity failure or exhausted
resource limits, retain all failures/partial results, and report inconclusive
measurements rather than launching an unbounded repeat loop.

Report cold prefix state separately from warm exact-prefix reuse, computed and
reused token counts, scan-region time, total prefill and first emitted text.
Distinguish model callback latency from network/client TTFT. Do not infer agent
success from synthetic output or extrapolate15K latency linearly.

Candidate remains default-off even if the pilot improves. Production promotion
requires repeated end-to-end gain, acceptable short-input behavior and explicit
review. Do not add a silent mature-runtime backend or claim general superiority.

## Completion and TODO

- [ ] Written-spec approval and detailed implementation plan.
- [ ] Feature-gated candidate and independently retained serial reference.
- [ ] Focused bit/state/error/fallback correctness gates.
- [ ] Bounded operator pilot and model-integrated comparison.
- [ ] Evidence report retaining negative and inconclusive outcomes.
- [ ] On completed implementation: patch version+0.0.1, changelog with ideas,
      inspirations and direct sources, scoped verification, commit and push the
      existing feature branch. Preserve and exclude the user's .gitignore edit.

This design-only stage does not bump the application version. No performance
improvement or successful15K Minicode run is claimed.

## Ideas, inspirations and sources

- Mivi's `crates/mivi-model/src/transformer.rs` provides the baseline GQA,
  per-head independence and precision-specific scratch constraints.
- `crates/mivi-core/src/math.rs` confirms the existing SIMD primitives.
- `crates/mivi-kv/src/cache.rs` supplies checked immutable F32 KV access.
- Latest local profile records: `/tmp/mivi-prefill-profile.49QxK4/REPORT.md`.
- [Prior CPU evidence](../../CPU_RUNTIME_EVIDENCE_2026-10-05.md) and
  [captured projection evidence](../../FOUR_ROW_ISOLATION_CAPTURE_EVIDENCE_2026-10-10.md)
  motivate separate correctness, operator and end-to-end gates.

No new online research, copied external code or model-specific dispatch is part
of this design. The brainstorming skill supplied the reviewed-design gate.
