# Q4 cached sums: agent-sized first-output evaluation

## Result and decision

All warmup and measured pairs passed bit-exact final-prefill/continuation-logit,
first-token/text and final KV/convolution/SSM-state comparisons. The live test
finished successfully in 466.42s after 28.63s of scoped release compilation.
The measurement code was run on the v0.2.74 base plus these test-only additions,
before the publication version bump to v0.2.75; inference kernels are unchanged.

**This candidate does not solve the initial agent wait.** First-output prefill
ran zero candidate projections in every member. Cold-prefix first output took
roughly 43–49s in measured members; warm-prefix first output took roughly 2.8–3.0s
with 1,536 tokens restored and 22 processed. Both routes execute the same batch
prefill code, so these timing differences are not evidence of a mechanism-level
prefill gain. Keep the candidate default-off and prioritize chunked prefill next.

| Case | Baseline median prefill / first / decode (s) | Candidate median prefill / first / decode (s) | Median paired candidate/baseline ratios |
| --- | --- | --- | --- |
| Cold prefix | 45.833162 / 45.833226 / 1.722731 | 47.440911 / 47.440989 / 1.610125 | 0.973403 / 0.973403 / 0.932190 |
| Warm prefix | 2.894473 / 2.894534 / 1.718238 | 2.947307 / 2.947387 / 1.520549 | 1.027360 / 1.027360 / 0.903398 |

Separate medians and paired-ratio medians are different statistics; the apparent
cold first-output advantage in the paired ratio is not a causal optimization
claim. Candidate first-output separate medians are higher in both cases, and
the candidate does not execute in prefill. Three pairs and uncontrolled host
conditions cannot establish equivalence or a reliable first-output regression.

Decode paired medians were 6.78% lower cold and 9.66% lower warm. Retain the
counterexample: cold pair 1 candidate decode was 1.771148s versus 1.722731s
baseline, a 2.81% increase. All warm pairs had lower candidate decode time, but
this is still a single-host pilot, not default-promotion evidence. Each candidate
member ran 640 projections across 16 forwards, matching 40 eligible Q4 FFN
tensors; baseline ran zero. The existing scratch remains 1,024 bytes.

## Complete pair record

The metric order below is **prefill / first nonempty output / 16-forward decode**.
Warmups are printed separately and excluded from all summaries; no measured
members were failed, excluded or retried. Cold members reused zero and processed
1,558 prompt tokens; warm members reused 1,536 and processed 22.

| Case | Pair | Order | Baseline prefill / first / decode (s) | Candidate prefill / first / decode (s) |
| --- | --- | --- | --- | --- |
| ColdPrefix | warmup | candidate → baseline | 46.033491 / 46.033558 / 1.709248 | 46.688760 / 46.688913 / 1.563872 |
| ColdPrefix | 1 | baseline → candidate | 45.833162 / 45.833226 / 1.722731 | 48.249642 / 48.249727 / 1.771148 |
| ColdPrefix | 2 | candidate → baseline | 48.737165 / 48.737242 / 1.727249 | 47.440911 / 47.440989 / 1.610125 |
| ColdPrefix | 3 | baseline → candidate | 45.627301 / 45.627364 / 1.681305 | 43.189376 / 43.189441 / 1.550478 |
| WarmPrefix | warmup | candidate → baseline | 2.951963 / 2.952029 / 1.730939 | 2.997828 / 2.997922 / 1.550442 |
| WarmPrefix | 1 | baseline → candidate | 2.904455 / 2.904517 / 1.718238 | 2.983919 / 2.983983 / 1.582918 |
| WarmPrefix | 2 | candidate → baseline | 2.894473 / 2.894534 / 1.683144 | 2.943004 / 2.943066 / 1.520549 |
| WarmPrefix | 3 | baseline → candidate | 2.822579 / 2.822641 / 1.720697 | 2.947307 / 2.947387 / 1.503227 |

## Protocol and scope

This follow-up tests the shipped opt-in candidate on the local
`LFM2.5-1.2B-Instruct-Q4_K_M.gguf`, using a public synthetic workspace-context
prompt with 1,557 tokenizer IDs. Model metadata inserts BOS, giving an effective
1,558-token prefix. Context is 2,048, KV precision F32, chunked prefill tile 64,
greedy first-token sampling, and continuation work is 16 fixed teacher-forced
forwards. Both routes use the same loaded mapping and feature-enabled binary.

Cold-prefix members clear engine snapshots before each request. This is **not**
a cold disk/page-cache measurement. Warm-prefix members restore a baseline-primed
prefix. Both cases have one printed warmup pair followed by three alternating
measured pairs. Each pair must report identical reused/processed work counts;
no measured pair is silently excluded. Model state is reset between members.

The existing fixture recorder measures prefill and first nonempty delivery,
with the aggregate/substage profiler disabled. Capture setup, model load,
tokenization, reset and cold-cache clearing are outside the request clock;
prefix restoration is inside. The first callback returns false. Its observed
time is independently bounded by recorder delivery time and physical return.
The fixture must stop before feeding the generated ID back into the model,
so the continuation begins from the same prompt position for both members.

Capture is disarmed before the 16-forward decode timer. That timer includes
finite-logit checks and bit-pattern copies. First-output and decode timings are
separate; their sum is not claimed as an end-to-end request duration. KV exports
and final state comparisons occur outside both timers. Every element of the
final prefill logit vector and all 16 continuation logit vectors is compared
bit-for-bit, together with first generated ID/text and final KV/convolution/SSM
state. Intermediate prompt-token output logits are not generated by ordinary
chunked prefill and are not claimed as tested.

Each candidate member must execute the metadata-derived number of Q4 FFN
projections over its continuation; the first-output prefill path must execute
zero candidate projections. Chunked batch prefill still uses baseline kernels,
including a partial final tile. This is important to interpreting first-output
changes: this selector does not optimize the measured prefill workload.

## Reproduction

```sh
CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 MIVI_TEST_MODEL=/absolute/path/LFM2.5-1.2B-Instruct-Q4_K_M.gguf timeout --signal=TERM --kill-after=5s 600s cargo test -p mivi-model --lib --release --offline --features q4-cached-sums-experiment,fixture-diagnostics q4_cached_sums_agent_prompt_parity_and_measurement -- --ignored --test-threads=1 --nocapture
```

The ignored diagnostic requires explicit release execution. Its nonignored
measurement-control test rejects missing delivery, inconsistent cache/work
counts, partial prefill, truncated output IDs, overflowing counters, invalid
clock ordering and accidental aggregate profiling. The control was observed
failing against a stub, then passed after implementation.

## Limits and sources

This is direct-token engine callback latency, not HTTP TTFT, Minicode task
completion, useful tool-call latency, model quality or cross-engine superiority.
The first output may be a partial text token and the fixed continuation is not
the model's generated response. No server queuing, network, SSE parsing, chat
template or actual agent tool loop is included. Three pairs on one CPU are not
enough for production promotion. Only one model and one agent-sized prompt are
covered; RSS and model-checksum collection remain outside this pilot.

The host is the local x86_64 AMD Ryzen 7 7730U, with two Rayon workers. CPU
frequency, affinity, thermal state and background load are uncontrolled. Near
the first warmup, RAM available was about 5,934 MiB of 14,819 MiB total, with
3,966 MiB swap used. These are host snapshots, not measured per-route peak RSS.

Verification: the new measurement-control test passed in debug and in the
published v0.2.75 release build. Touched Rust files passed scoped formatting
checks, and whitespace checks passed. Luna/high review found no remaining
concrete issue after clarifying final-prefill-vector versus intermediate prompt
logits. No workspace-wide Cargo check/build/test or second live rerun was used.

## Next work

Return to the CPU plan's remaining faithful chunked-projection work, using the
existing [projection-cost evidence](PROJECTION_COST_EVIDENCE_2026-10-05.md) and
[locality pilot](PROJECTION_LOCALITY_EVIDENCE_2026-10-06.md). Evaluate a
caller-owned scratch/blocked F32 path against the current checked batch API
while preserving accumulation order and fallbacks. Require agent-sized
first-output evidence, not only operator or
decode gains, before promotion. Packed activations remain a separate numerical
policy decision; this result does not authorize silently changing precision.

Ideas and sources: the [v0.2.74 cached-sum evidence](Q4_CACHED_SUMS_EVIDENCE_2026-10-07.md),
Mivi's existing fixture recorder and chunked prefill implementation, and
[Colibri's benchmarking protocol](https://github.com/JustVugg/colibri/blob/main/docs/benchmarking.md)
for explicit cache state, one-variable interleaving and retaining negative
results. No external inference code is copied. See the
[evaluation plan](superpowers/plans/2026-10-07-q4-agent-sized-evaluation.md).
