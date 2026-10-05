# CPU runtime evidence — 2026-10-05

Phase 0 Task 6 completed the bounded comparison and selected **P1-A: faithful
scratch/locality experiments for projections and FFNs**. The long-workload native
prefill median is 84.243 seconds; substantial latency remains. Short and medium
cross-engine output differences are reproducible and their cause is unresolved.
This is a measurement/documentation update, not a kernel speedup or coding-agent
quality result.

## Provenance and settings

- Measured Mivi: v0.2.67, commit `5c49b97c436fb2fcb3a3babaea15a3ef657fe49b`.
  Release replay example, fixture-diagnostics enabled, Rust/Cargo 1.94.0.
- Reference: llama.cpp build11146, commit
  `7fe450e19305b828c199d602c23a8337aaa1f03b`, GNU13.3.0, Release CPU build.
  Native CPU features enabled; BLAS, CUDA and OpenMP disabled; default repacking
  retained. Only the server and vocabulary-only tokenizer targets were built.
- Hardware: Ryzen7 7730U, AVX2/FMA, 16 logical CPUs, approximately15GiB RAM.
  Swap was already in use. Affinity, background load and thermal conditions were
  uncontrolled; no system-wide page-cache flush was performed.
- Model: local LFM2.5-1.2B-Instruct GGUF, SHA256
  `b1b3de114215d9507409a662a501a631095a479a419584e8a2ded6304b19b4f5`.
  This matches the earlier baseline. Descriptor inventory: 82 Q4_K, 11 Q6_K and
  55 F32 tensors. The filename does not imply every tensor has the same format.
- Exact normalized prompt lengths: 110/1024/2636. Prompts are slices of a
  tokenizer-derived synthetic context; metadata BOS is included exactly once.
  These synthetic continuations do not evaluate tool calling or programming.
- Both engines: CPU threads2, context4096, F32 K/V, tile/microbatch64,
  temperature0, seed7, repetition penalty1, presence/frequency penalties0,
  output cap48, no reused prompt tokens. Reference parallelism1, flash attention
  off, fitting off, warmup disabled, and bind restricted to localhost.
- Three paired repetitions per case, in Mivi/reference, reference/Mivi,
  Mivi/reference order. Each engine starts with fresh inference state. Weight
  pages may be warm, including from SHA preflight; fresh state is not a cold
  filesystem-cache claim.
- Per-run wall limit180s including supervised startup/cleanup; sampled child-tree
  RSS watchdog2GiB at100ms; artifact cap64MiB. Matrix budget35min plus10min for
  separate controls. The matrix completed in746.26s; all controls also completed.
  RSS sampling cannot rule out transient peaks between observations.

Executable SHA256 values (dependent reference-library hashes also remain in
private provenance):

| Executable | SHA256 |
| --- | --- |
| Mivi replay | `ac6d5e87a1234f0724f5fb394e34a5acf4a346ad040ad4c1df0b6194a8ffda13` |
| Reference server | `b37730ef614e75fb93bd55827a1f1fbb4bd0c7b5e284defad11142047b66e585` |

## Unprofiled observations

Every workload has three completed samples per engine (18 completed runs total).
Every supervised child cleanup succeeded. No main sample failed, timed out, or
tripped an RSS/artifact guard. The short/medium first-divergence probes also
completed and are retained separately.

| Normalized prompt tokens | Mivi prefill median [min,max], seconds | Reference prefill median [min,max], seconds | Content token counts, Mivi/reference |
| --- | --- | --- | --- |
| 110 | 3.832 [3.673,4.027] | 1.283 [1.231,1.376] | 48/48 |
| 1024 | 27.178 [26.215,28.211] | 12.434 [11.466,12.457] | 48/19 |
| 2636 | 84.243 [81.348,96.143] | 32.313 [30.793,35.754] | 48/48 |

These are each engine's reported prefill clocks. Mivi uses model-prefill-only
timing; the reference reports server prompt-evaluation timing. The comparison
report deliberately marks timing-ratio validity false. Reference client-visible
TTFT is unavailable from its nonstreaming completion; Mivi's callback clock is
not an HTTP first-useful-delta clock. Queue/render/router/client durations are
unavailable in this model-level matrix. Three samples do not establish p95/p99.

All native outputs reached the48-content-token cap. Reference medium outputs
contain19 content tokens followed by EOS. Aggregate decode times cannot be
compared as equal work; the retained comparison summary does not expose reference
per-token decode timings. The native median decode duration per emitted content
token is96.64ms/105.83ms/116.40ms for short/medium/long respectively, using its
model decode boundary and including its delivery work.

## Separate profile controls

One profile-enabled control per case completed. All generated IDs, terminal IDs,
raw/delivered/returned text and stopping reasons match the corresponding three
unprofiled native samples. Profiled times (3.416/24.917/79.896s) are excluded from
the medians above. They are faster than the observed unprofiled ranges; the
uncontrolled run conditions and one control per case do not establish profiler
overhead or a profiling speedup.

Percentages below use counted embedding+attention+SSM+logits time. Substage
timers are nested within those aggregates, not additional independent total time.
FFN is a subset of the projections+FFN column; the columns must not be added.

| Tokens | Counted forward stages, seconds | Projections+FFN | FFN subset | Causal attention scan | Convolution |
| --- | --- | --- | --- | --- | --- |
| 110 | 3.415 | 98.09% | 76.43% | 1.02% | 0.16% |
| 1024 | 24.914 | 87.94% | 68.39% | 11.37% | 0.17% |
| 2636 | 79.889 | 72.61% | 56.52% | 26.86% | 0.15% |

The measurements favor projection/FFN work first. They do not isolate allocation,
transpose, weight decoding and dot-product costs inside those stages. That is the
first experiment in P1-A, before proposing caller-owned scratch or a blocked F32
activation path. The causal scan is material on the long prompt; convolution is
small in these controls. These conclusions are specific to this workload/model.

## Numerical and terminal limits

Short outputs first differ at zero-based generated position1 in every pair;
medium outputs first differ at position11. Long outputs match all48 content IDs
in every pair. At the identical shared prefixes, the token-major native probe
chooses the same top token as the ordinary chunked native run.

Both divergent candidates are available in each bounded score probe. Native
candidate-minus-reference-candidate gaps are +1.605/+8.438 logits for
short/medium; reference gaps are -0.086/-1.395 log probabilities. Native top-two
margins are0.266/3.636; reference top-two log-probability margins are0.086/0.931.
The medium disagreement cannot be dismissed as merely a near tie. Raw logits
and log probabilities are different quantities; the report labels them separately.

Reference raw full logits, intermediate hybrid states and quantized activation
traces are unavailable through this workflow, so the cause remains unresolved.
The [pinned GGML CPU traits](https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/ggml/src/ggml-cpu/ggml-cpu.c)
use Q8_K operands for Q4_K/Q6_K dot products; Mivi's decoded-row batch path uses
F32 activations. This establishes a numerical-path difference, not the cause of
the observed rankings. Native first-step EOS suppression also differs from the
reference request policy; the observed divergences occur after that first step.

Task5's independent F32 hybrid oracle and isolated Q4_K/Q6_K layouts passed.
They do not establish full real-model quantized-graph equivalence. No confirmed
graph/layout defect was identified by this experiment. Preserve the existing
native arithmetic as the baseline for P1-A and investigate any new regression
before promotion. Cross-engine parity and approximate-mode promotion remain
blocked by this uncertainty.

## Handoff and reproduction

The next package is **P1-A only**: measure allocation/transpose/decode/compute,
then evaluate scratch ownership and F32 locality with existing checked APIs,
portable fallback, numerical controls and repeated request measurements.
No operator implementation or universal speedup is claimed here.

Use the existing `scripts/runtime_compare/compare.py` with an explicit private
schema1 manifest, verify it with `--validate-only`, then run the same command
without that flag. Supply the actual model/executable paths and pinned reference
revision, exact tokenizer-derived IDs, and the budgets recorded above. Raw IDs,
text, outputs, logs and artifact paths are retained privately rather than committed.
All measurement directories/files were verified mode0700/0600; their total
retained size is540492 bytes, excluding separate source/build scratch.

Verification:54 focused Python comparison tests passed. The scoped release replay
example and pinned reference targets built with one job. Profile controls and
paired runs were sequential; no concurrent Cargo/model workloads. Documentation
arithmetic was independently recalculated and `git diff --check` passed. The
requested Luna/high review was unavailable due to model capacity; local review
covers claims, privacy and arithmetic.

Setup attempts are separate from measured samples: the initial sandbox test run
could not bind its localhost mock and passed after the required permission retry.
An ad hoc version probe omitted its required artifact counter and failed before
model loading; correcting that invocation made version/help validation pass.
Neither attempt was counted as a model timing sample or concealed as a success.

Methodology/inspiration: [Colibri benchmark protocol](https://github.com/JustVugg/colibri/blob/main/docs/benchmarking.md),
[Kimi fixture guidance](https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/main/tests/fixtures/README.md),
[GGML quant arithmetic](https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/ggml/src/ggml-quants.c),
and [LMCache hybrid-state guidance](https://docs.lmcache.ai/recipes/kimi_linear.html).
