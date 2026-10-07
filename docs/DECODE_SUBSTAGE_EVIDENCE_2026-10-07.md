# Decode substage diagnostics — 2026-10-07

The v0.2.73 diagnostic attributes single-token attention and SSM work using the
same arithmetic as ordinary inference. Profiling is opt-in; disabled single-token
paths do not create substage timestamps. `mivi bench` subtracts the cumulative snapshot at
its own generation call's prefill boundary from the final forward profile.

## Result

The local LFM2.5-1.2B-Instruct Q4_K_M GGUF has six attention and ten SSM blocks.
The retained-prefix run completed all warmup and measured members with exact
paired logit/state parity. Its effective prefix was 257 tokens including BOS.
The median combined FFN share was **66.50%** of profiled forward-stage time.
Prioritize FFN linear-projection cost isolation next; this does not establish a
specific faster kernel or justify promoting an existing experiment.

Measured outer wall seconds for 16 identical continuation forward passes:

| Pair | Order | Unprofiled | Profiled |
| --- | --- | ---: | ---: |
| 1 | Unprofiled, profiled | 1.5707 | 1.6134 |
| 2 | Profiled, unprofiled | 1.5205 | 1.5536 |
| 3 | Unprofiled, profiled | 1.6116 | 1.6684 |
| Median | | 1.5707 | 1.6134 |

Each profiled member was about 2.2–3.5% slower than its paired unprofiled member.
This is observed timing variation plus instrumentation cost, not an isolated
estimate of timer overhead. Test execution took 20.34s excluding compilation.

Median milliseconds per continuation forward, aggregated across all blocks of
the indicated type, from three profiled samples:

| Stage | Attention blocks | SSM blocks |
| --- | ---: | ---: |
| Normalization | 0.0514 | 0.0873 |
| Q/K/V or input projection | 3.4876 | 10.5685 |
| Causal attention or gated convolution | 2.5597 | 0.0864 |
| Output projection plus residual | 2.2544 | 3.7016 |
| FFN | 24.6334 | 41.4403 |

Final normalization/output logits took a median 11.5270ms per forward.

The scoped release `mivi` binary also completed a short CLI smoke run with
`--bench-prompt-tokens 64`, printing the new decode substage fields for both cold
and shared-prefix cases. Its variable-length sampled responses are excluded from
the fixed-work medians above.

An earlier completed pilot cleared the prefix cache before every member. It also
passed exact paired parity but showed 3.4163s profiled versus 1.6025s unprofiled
wall medians and 61.65% FFN share. Its discrepancy is unresolved. Retain that
observation rather than treating it as a stable instrumentation-overhead result.
The retained-prefix protocol reduces untimed cold prefill work; its timings are
not an equivalent end-to-end comparison with that initial pilot.

## Measurement and parity protocol

The ignored `decode_substage_profile_parity_and_measurement` test uses an explicit
local GGUF, F32 KV, a 1,024-token context, chunked prefill with tile 64, 256 input
prompt IDs, and 16 fixed teacher-forced continuation IDs. Model metadata still
controls BOS insertion. Continuation positions start at the effective prefill
position, including any BOS. Prompt and continuation text are synthetic public
fixture inputs, not a real workspace or agent conversation.

One warmup pair precedes three measured pairs. Profiling-on/off order alternates.
Each member resets model state, restores the same prefix with profiling disabled,
then times the same 16 forward passes. The prefix cache is cleared once before the
warmup and retained between members to reduce untimed prefill cost. Each pair checks exact
equality of every continuation logit, final convolution/SSM state, exported KV
state, and position. Finite logits, forward counts, and nested-timer bounds are
also checked. There is no inference kernel replacement in this release.

Run only this target with an absolute model path:

```sh
CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 MIVI_TEST_MODEL=/absolute/path/model.gguf timeout --signal=TERM --kill-after=5s 300s cargo test -p mivi-model --lib --release --offline decode_substage_profile_parity_and_measurement -- --ignored --test-threads=1 --nocapture
```

## Stage boundaries

- Attention: pre-normalization; Q/K/V linear projections; per-head normalization,
  RoPE, KV insertion and causal attention scan; output projection plus residual;
  shared FFN.
- SSM: pre-normalization; input projection; gating and convolution state update;
  output projection plus residual; shared FFN.
- Each FFN bucket includes its normalization, gate/up/down projections, SwiGLU,
  and residual. Logits include the final normalization/output projection.

Nested substages overlap their parent attention/SSM totals. Do not add parent and
child values. Component medians need not sum to the median total; the reported
FFN share is computed within each sample before taking its median.

## Limits

The outer diagnostic wall clock includes logit copies and finite-value checks.
It does not measure sampling, streamed delivery, HTTP latency, tool calling, or
agent quality. Profiling changes timing overhead, and three repetitions on one
host are a small sample with uncontrolled scheduling and thermal conditions.
This short-context workload cannot establish long-context attention costs or
general performance across models. Exact profiling parity establishes unchanged
arithmetic for the tested cases, not independent numerical correctness.

Ideas and sources: Mivi's existing opt-in forward profiles; alternating fixed-work
comparisons inspired by [Colibri's benchmarking methodology](https://github.com/JustVugg/colibri/blob/main/docs/benchmarking.md).
No external inference source was copied.
