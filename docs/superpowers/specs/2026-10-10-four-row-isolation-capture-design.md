# Approved four-row isolation and captured-activation diagnostics

User approved two diagnostic-only steps after the v0.2.77 mixed operator results:
isolate accumulation, then replay actual Q4/Q6 activations behind exact baseline parity.
No production kernel, precision, default dispatch, model selector or new unsafe code changes.

## Isolation

Test-only diagnostic calls existing two-pair and four-row core helpers on four decoded
rows and pretransposed generated inputs. Decode/transpose/allocations are outside timers;
both paths zero active outputs before each iteration. Use columns 2048/8192, batches
32/64/65, a declared four-row warm working set, 200 iterations/member, one warmup pair
and six alternating measured pairs. Full finite-output bits must agree outside timers.
This isolates small-group arithmetic, not full projection cache/decode effects or
hardware instruction counts. Do not infer root cause from source broadcast counts alone.

## Actual capture

Memory-only one-tile walker calls existing production attention/SSM tile routines,
captures post-FFN norm (gate/up input) and post-SwiGLU gate (down input), then forms
final logits through the existing output norm/head. Use metadata BOS policy and exactly
32/64 effective IDs from a public prompt (or a caller-provided prompt kept private).
Reject active adapters, caller prompts over 64KiB UTF-8 before tokenization, and
unsupported/unbounded shapes. Skip nonexecuted FFNs using
the existing execution condition; never capture stale scratch. Select first executed
eligible Q4_K/Q6_K projection by metadata/type, not model name. Retain full weight matrices.

Normal production chunked prefill runs before a reset and the walker. Require every
element of complete final logit vectors to be finite and bit-identical before timing
any capture. This is a replay-origin gate, not candidate-integrated model-state parity.
The candidates never feed back into the model. Captured prompts, IDs, activations and
weights are neither printed nor persisted.

Replay three unchanged controls: ordinary, scratch-only, four-row. Construction and
capture outside timers; three calls/member, one warmup triple plus six measured
permutations. Check complete member-final output bits/finiteness outside timers and
retain all raw times/negatives. Bound one-tile capture to context 65 and format/shape
caps from the prior diagnostic. Keep default-off regardless of pilot results.

## Verification and sources

Observe RED on strict finite/length/bit gate and isolation shape controls before
implementation; run scoped tests and bounded live diagnostics with Cargo jobs=1,
Rayon threads=2, serial harness. Review through GPT-6 Luna/high, publish sourced patch
release only after completing/reporting required gates.

Sources: Mivi's existing checked SIMD pair/four-row helpers and memory-only capture
walker; [v0.2.77 evidence](../../FOUR_ROW_PREFILL_EVIDENCE_2026-10-10.md) motivates
separating accumulation from projection work. Balanced comparisons and retained
negative results follow [Colibri benchmarking](https://github.com/JustVugg/colibri/blob/main/docs/benchmarking.md).
No external inference code or new research claims are introduced.
