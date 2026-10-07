# Q4 cached sums: agent-sized evaluation

This follows the approved cached-sum design and the user's approval to test
agent-sized prompts before any default promotion. No kernel or server behavior
changes are in scope. Continue on `feat/runtime-parity-profiling`, preserving
the unrelated `.gitignore` edit.

## Measurement contract

Use the same loaded GGUF, F32 KV, context 2,048, chunked tile 64, greedy first
token and 16 identical teacher-forced continuation IDs. Tokenize a public
synthetic workspace-context prompt before timing and take 1,557 IDs; model
metadata controls BOS. Compare baseline against the explicit cached-sum selector
in the same feature-enabled test binary, with the forward profiler disabled.

Cold-prefix members reset context and clear engine prefix snapshots before
every request. This does not evict OS pages or mapped model weights. Warm-prefix
members restore a baseline-primed prefix; both members must report identical
reused/processed counts. Each case has one printed warmup pair and three measured
pairs with alternating execution order. Never discard a measured pair silently.

Existing fixture diagnostics measure prefill and first nonempty model delivery.
The first callback returns false; require current position to equal the prompt
end, proving that the generated token has not been forwarded. Record the physical
return wall time as a timer bound, not as decode throughput. Disarm capture before
timing continuation forwards. Preserve every element of the final prefill logit
vector and each continuation logit vector, first token/text, and final
KV/convolution/SSM state. Intermediate prompt-token output logits are not
produced by ordinary chunked prefill and are not claimed as parity evidence.

The first-output metric excludes loading, tokenization, HTTP, server queuing,
SSE buffering, agent prompt construction and tool handling. It is engine callback
latency, not real-agent TTFT or tool-loop completion. Teacher-forced continuation
does not measure response quality. Production promotion and RSS/model-coverage
gates remain open regardless of this pilot's outcome.

## Tasks and verification

- [x] Add test-only `model/cached_sums_agent_tests.rs`, gated by both existing
  experiment and fixture-diagnostics features; no ordinary-build overhead.
- [x] Observe the missing measurement validator fail, then verify cold/warm work
  accounting, timer boundaries, missing delivery, partial prefill, truncated IDs,
  counter overflow and accidental profiling rejection with:

  ```sh
  CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 cargo test -p mivi-model --lib --offline --features q4-cached-sums-experiment,fixture-diagnostics agent_first_output_measurement_controls -- --test-threads=1
  ```

- [x] Run only the bounded live release diagnostic:

  ```sh
  CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 MIVI_TEST_MODEL=/absolute/path/LFM2.5-1.2B-Instruct-Q4_K_M.gguf timeout --signal=TERM --kill-after=5s 600s cargo test -p mivi-model --lib --release --offline --features q4-cached-sums-experiment,fixture-diagnostics q4_cached_sums_agent_prompt_parity_and_measurement -- --ignored --test-threads=1 --nocapture
  ```

- [x] Review benchmark boundaries with GPT-6 Luna/high; no parallel Cargo jobs.
- [x] Report all warmup/measured timings, separate metric medians and paired
  ratios, actual prefix reuse, eligible/candidate calls, limitations and decision.
- [x] Verify release measurement controls, formatting and whitespace; update
  changelog with ideas/sources and increment the patch version when complete.
- [ ] Commit/push scoped changes only; verify the branch is synchronized.

Sources: the [approved mechanism design](../specs/2026-10-07-q4-cached-sums-design.md),
[short-prefix evidence](../../Q4_CACHED_SUMS_EVIDENCE_2026-10-07.md), Mivi's existing
fixture recorder, and [Colibri's benchmark protocol](https://github.com/JustVugg/colibri/blob/main/docs/benchmarking.md)
for explicit cache state, one-variable interleaving and reporting failed or negative
results. No external inference code is copied.
