# Faithful Q4 FFN cached sums — bounded pilot

## Result and decision

The v0.2.74 opt-in candidate passed bit-exact paired full-logit and final
KV/convolution/SSM-state comparisons on the local
`LFM2.5-1.2B-Instruct-Q4_K_M.gguf`. All warmup and measured members completed.
Three alternating measured pairs had candidate/baseline wall ratios
0.915259, 0.914447, and 0.913469: a median **8.56% lower fixed-work decode
wall time**. Separate wall medians were 1.571224s baseline and 1.438076s
candidate. These statistics describe this pilot, not a general engine speedup.

Keep the feature and per-state selector opt-in. Do not promote ordinary
dispatch from three short-context pairs or infer faster agent first output,
quality improvement, or superiority to llama.cpp/Ollama.

| Measured pair | Execution order | Baseline seconds | Candidate seconds | Candidate / baseline |
| --- | --- | ---: | ---: | ---: |
| 1 | baseline, candidate | 1.571224 | 1.438076 | 0.915259 |
| 2 | candidate, baseline | 1.577286 | 1.442344 | 0.914447 |
| 3 | baseline, candidate | 1.530871 | 1.398403 | 0.913469 |

No measured pairs were excluded. One warmup pair preceded the table and is
excluded from summaries; its timings were not printed by the harness.

## What changed

Q4's minimum correction repeatedly sums the same 32 activation values inside
every weight row. The scratch-taking candidate prepares those sums once per
projection. Scalar additions retain their sequential order; AVX2 uses the same
four eight-lane additions and existing horizontal reduction as its baseline.
Weight decoding, F32 activation precision, dot accumulation, final expression,
and LoRA application remain unchanged. Scalar and AVX2 each compare against
their own route; this does not assert equality between different CPU routes.

The existing kernels remain untouched and the normal build does not compile
the experiment. `q4-cached-sums-experiment` on `mivi-model` forwards to quant
and core. Even with it compiled, `model.state.q4_cached_sums_enabled` defaults
false. The diagnostic enables it only for candidate single-token FFN forwards.
There is no new server/CLI selector, automatic model-name rule, or global state.

Scratch is preallocated once from the larger activation dimension rounded to
32 elements, recomputed on every projection, shared read-only with Rayon workers,
and excluded from prefix-cache recurrent state. Reset clears scratch and call
counts but preserves the explicit selector. Buffer/dimension errors occur before
output or scratch writes. The candidate kernel allocates no heap buffers per call.

## Actual tensor coverage

Eligibility comes from loaded tensor metadata, not the GGUF filename:

| Format | Output rows | Input columns | FFN tensors |
| --- | ---: | ---: | ---: |
| Q4_K | 8192 | 2048 | 32 |
| Q4_K | 2048 | 8192 | 8 |
| Q6_K | 2048 | 8192 | 8 |

Forty of 48 FFN projections use the candidate; eight Q6 projections retain
baseline behavior. Each candidate member executed 640 experimental projections
over 16 forwards, versus zero for baseline. Scratch capacity is **1,024 bytes**;
this is the buffer size, not a measured RSS delta.

## Reproduction and boundaries

```sh
CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 MIVI_TEST_MODEL=/absolute/path/LFM2.5-1.2B-Instruct-Q4_K_M.gguf timeout --signal=TERM --kill-after=5s 300s cargo test -p mivi-model --lib --release --offline --features q4-cached-sums-experiment q4_cached_sums_decode_parity_and_measurement -- --ignored --test-threads=1 --nocapture
```

The synthetic public prompt supplies 256 tokenizer IDs, with model metadata
adding BOS for an effective prefix of 257. Each member starts from identical
baseline-prepared prefix state, F32 KV, context 1,024, chunked prefill tile 64,
and the same 16 teacher-forced continuation IDs. Profiling stays disabled.
Prefix cache is cleared once before warmup and retained between members.
Prefix preparation is outside the timer. Both routes run in the same
feature-enabled binary and use the same loaded GGUF mapping.

The timer includes forward calls, finite-logit checks, bit-pattern copies and
the harness's logit-vector allocations. Final KV/SSM export and comparison are
outside it. There is no sampling, HTTP, SSE, tool parsing, or agent execution.
The harness allocations are not candidate-kernel allocations.

Host: x86_64 AMD Ryzen 7 7730U, 8 physical cores/16 logical CPUs; two Rayon
workers. Host RAM was approximately 14,819 MiB total and 5,167 MiB available
when inspected; swap was full. Thread affinity, CPU frequency, thermal state,
background load and page-cache eviction were not controlled. No external
engine was benchmarked in this pilot. Model-file checksum and peak RSS were
not collected. Release test compilation took 1m27s; the live test took 23.04s,
including untimed preparation and comparisons.

## Controls and remaining gates

Targeted controls cover route-specific bit equality, zero/odd/parallel row
counts, multiple block widths, changed-input scratch reuse, untouched buffer
tails, validation without writes, arena capacity/reset/default-off behavior,
FFN Q4/F32 fallback and synthetic nonzero LoRA adapters. Portable scalar code
is exercised directly on this x86 host, not cross-compiled for another host.
Same-engine baseline parity is not independent model-correctness validation.

Before any default promotion: repeat beyond three pairs, broaden models and
shapes, inspect scratch/RSS and short-case regressions, and measure agent-sized
cold/warm first useful output and completed tool loops. Chunked prefill's batch
kernels are not optimized by this experiment; slow agent prefill may remain
the dominant latency problem. Existing CPU-plan end-to-end gates stay open.

## Ideas, inspirations and sources

- Mechanism: Mivi's existing [Q4 kernel](../crates/mivi-quant/src/q4_k_m.rs)
  and [v0.2.73 decode attribution](DECODE_SUBSTAGE_EVIDENCE_2026-10-07.md).
- Interleaving one-variable comparisons, declaring cache state and workload,
  and retaining negative results follow
  [Colibri's benchmarking protocol](https://github.com/JustVugg/colibri/blob/main/docs/benchmarking.md),
  reviewed 2026-10-07. This small pilot does not fulfill its full comparative
  reporting protocol.
- See the [approved design](superpowers/specs/2026-10-07-q4-cached-sums-design.md)
  and [implementation checklist](superpowers/plans/2026-10-07-q4-cached-sums-experiment.md).
  No external inference source was copied; the candidate reuses Mivi's own
  Q4 layout and arithmetic.
