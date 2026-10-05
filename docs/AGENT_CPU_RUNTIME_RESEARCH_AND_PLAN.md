# Mivi CPU agent runtime: research and proposed roadmap

Research date: 2026-10-02. Baseline: Mivi 0.2.62, commit `0fe99c81275ef9dbef1f133ad8f5a59a1584b83f`.
Status (2026-10-05): Phase 0 Tasks 1–5 are implemented and released through
v0.2.67: bounded replay/comparison tooling, profiling/lifecycle boundaries, and
independent arithmetic/hybrid fixtures. Task 6 has completed the bounded evidence
package; see the [measured findings and unresolved numerical limits](CPU_RUNTIME_EVIDENCE_2026-10-05.md).
P1-A is selected for projection/FFN experiments. Cross-engine parity and
approximate-mode promotion remain unestablished. See the [master roadmap and TODO](superpowers/plans/2026-10-02-native-cpu-agent-improvements.md)
and [first execution package](superpowers/plans/2026-10-02-runtime-parity-and-profiling.md).
Optional expansion still needs separate approval.

## Recommendation

Keep the native Rust runtime as the primary engine. Establish numerical and agent-protocol correctness, optimize the measured CPU prefill path, validate exact hybrid-prefix reuse, and then run bounded coding-agent evaluations. Context retrieval and larger-model offloading are separate later projects.

Do not combine all eight reference projects into Mivi. Borrow their measurement discipline, modular boundaries, safe tool contracts, and memory-locality techniques where a measured workload justifies them.

## Evidence and limits

The preceding private synthetic comparison used the same local LFM2.5-1.2B-Instruct Q4_K_M GGUF and identical numeric prompt IDs, two CPU inference threads, context4096, F32 KV, tile/microbatch64, greedy sampling, no repetition penalty, output limit48, and no prefix reuse. Reference: official llama.cpp b11146, commit `7fe450e19305b828c199d602c23a8337aaa1f03b`.

| Prompt tokens | Mivi prefill | llama.cpp prefill | Ratio |
| --- | --- | --- | --- |
|110|3.761s|1.208s|3.11×|
|2636|82.912s|31.935s|2.60×|

The long Mivi profile attributed43.916s to attention blocks and38.971s to SSM blocks. Those totals include their projections and FFNs; they do **not** prove that attention scoring or recurrent convolution alone dominates.

One run per case, no thermal/affinity controls, and profiling enabled only on Mivi prefill: these are diagnostic observations, not stable benchmark ratios. Mivi used a temporary split-prefill/decode driver; its short-case output matched the normal single-call path, while long-case split parity was not independently rerun. Output IDs differed across engines; numerical parity and comparative tool quality remain unestablished. A separate long-prompt tokenizer check matched after removing the extra BOS introduced by the reference check, not by the timed completion.

Raw prompts, outputs and logs remain private outside the repository. Preserve that boundary in follow-up work. The82.9s pre-output delay can explain a client terminating near70s, but the earlier Minicode request was not replayed in this comparison.

## What each requested project contributes

| Project and primary source | Applicable inspiration | Fit and limitation |
| --- | --- | --- |
|[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c), [fixture design](https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/main/tests/fixtures/README.md)|Independent, adversarial tiny oracles; teacher-forced, greedy, and incremental validation; configuration read from checkpoints.|Adopt the gate structure, not Kimi-specific tensor graphs. Its large-model memory claim is not a small-model agent latency benchmark, and its README explicitly distinguishes limited arithmetic parity from output quality.|
|[Colibri](https://github.com/JustVugg/colibri), [benchmark protocol](https://github.com/JustVugg/colibri/blob/main/docs/benchmarking.md)|Controlled end-to-end comparisons; cache-state labels; stage profiles; report negative results and excluded runs.|Immediately useful methodology. Its expert streaming, learned placement and disk hierarchy address large MoE storage workloads; they are not demonstrated remedies for the current resident small-model workload.|
|[AirLLM](https://github.com/lyogavin/airllm)|Layer-wise weight residency, explicit memory/compute tradeoffs, and overlapping loading with computation.|Keep as a future out-of-core design reference. Its own documentation identifies disk loading as a bottleneck in its compression scenario. Reducing resident memory does not establish faster cold prefill for our current GGUF.|
|[Hyper-Extract](https://github.com/yifanfeng97/Hyper-Extract)|Source-attributed structured knowledge, incremental updates and rollback, provider separation.|Later context-management inspiration, not an inference kernel. Knowledge extraction itself incurs model calls and needs quality and freshness validation. Do not import its graph stack into the runtime hot path.|
|[OpenKB](https://github.com/VectifyAI/OpenKB), [configuration implementation](https://github.com/VectifyAI/OpenKB/blob/main/openkb/config.py)|Persist reusable source-linked knowledge; retrieve selected context; immutable per-request configuration bundles.|Potential agent-side memory layer. Compilation/reasoning-based retrieval has costs. Reuse the isolation/configuration principle; avoid silently rewriting user messages or adding a mandatory retrieval service.|
|[LMCache](https://github.com/LMCache/LMCache), [hybrid-state guidance](https://docs.lmcache.ai/recipes/kimi_linear.html)|Cache lifecycle metrics, bounded tiers, connector boundaries, and snapshots covering recurrent state as well as attention cache.|Strengthen Mivi's existing exact-prefix path first. The documented Kimi-Linear integration requires model/backend-specific state handling and does not support CacheBlend there. This is not evidence of a ready-made Mivi/LFM2 connector.|
|[gmgn-minicpm](https://github.com/dvictor357/gmgn-minicpm)|Small-model agent loops with allowlisted tools, argument validation, bounded iterations/output/results, and direct argv execution rather than a shell.|Use a generic synthetic read-only tool loop as an evaluation pattern. Its published live setup uses MiniCPM5 and Metal acceleration; its latency is not transferable to Mivi's two-thread CPU. Do not copy its financial tool set or silently drop unknown arguments.|
|[mistral.rs](https://github.com/EricLBuehler/mistral.rs), [tool constraints](https://docs.mistralrs.dev/guides/agents/tool-calling-basics/)|Loader/capability registries, hardware-aware configuration, separation of schema-constrained arguments from tool-choice policy, bounded agent loops.|Useful Rust architecture and protocol reference. Its CUDA benchmark numbers do not predict this laptop's CPU speed; LFM support was not established by the inspected supported-models page. Borrow interfaces, not an assumed interchangeable runtime.|

Inspection scope: project READMEs, selected official documentation, Kimi fixture documentation, OpenKB configuration code, and GGML CPU kernel/repacking code. These projects were not installed, run, or comprehensively audited. Some raw-source URLs were unavailable; no implementation claims rely on those unavailable files.

## Additional directly relevant sources

- [GGML x86 quantized dot products](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/arch/x86/quants.c) implement packed Q4_K/Q8_K operations with CPU intrinsics. [CPU repacking](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/repack.cpp) interleaves quantized blocks. These are concrete kernel-design references, not proof that repacking explains our gap. Pin revisions and check license/notice requirements before any reuse.
- [FlashAttention](https://arxiv.org/abs/2205.14135) motivates exact attention with IO-aware tiling. Apply memory-locality principles experimentally on CPU; do not transplant GPU speedup claims. Exact mathematical attention does not guarantee bit-identical floating-point accumulation.
- [Sarathi-Serve](https://arxiv.org/abs/2403.02310) addresses prefill/decode interference through chunked scheduling. Mivi already chunks prefill. Continuous multi-request batching is secondary at our current inference concurrency1.
- [LMCache paper](https://arxiv.org/abs/2510.09665) motivates cache movement, reuse, connector modularity and observability. Its serving evaluations do not establish first-request speedups when no reusable prefix exists.
- [Liquid AI's LFM2.5-1.2B model card](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/blob/47c527b619f86234f47382b40f16a5ad56b6fdd5/README.md) specifies Pythonic tool calls with model delimiters and tool-role handoff. It recommends agentic/extraction/RAG use but explicitly does not recommend programming. Engine latency, tool transport, and coding-model suitability must therefore be evaluated separately.

## Existing Mivi components: extend, do not duplicate

- `crates/mivi-model/src/model.rs`: token-major and layer-ordered chunked prefill, cancellation, forward/substage profiling, generation, prefix restoration, and leaving the final restored token for logits reconstruction.
- `crates/mivi-model/src/transformer.rs`, `ssm.rs`: attention/SSM substage counters already exist. Export/inspect them before adding new timers or replacing attention wholesale.
- `crates/mivi-quant/src/lib.rs`: checked batch API; decoded weight rows reused across tokens; SIMD accumulation and Rayon row partitions. It still allocates transposition/output/work buffers per call and decodes rows into floats. These are optimization candidates, not established dominant costs.
- `crates/mivi-quant/src/q4_k_m/packed_prefill.rs` and its nested modules: existing benchmark-only scalar/AVX2/tiled activation-packing, group32 and cumulative projection/FFN experiments. Reuse their tests and evidence; do not mistake them for a production inference path or recreate the experiments.
- `crates/mivi-kv/src/prefix.rs`: causally chained KV ranges and final SSM checkpoints. Arbitrary suffix reuse is intentionally not used by the model generation path.
- `crates/mivi-server/src/model_profile.rs`: metadata/explicit protocol profiles and tool codecs. Protocol-profile flexibility does not imply support for arbitrary GGUF architectures.
- `crates/mivi-server/src/routes/chat.rs`: SSE heartbeats, request/first-output deadlines and cancellation already exist. Heartbeats are transport liveness, not model output; do not use them to satisfy first-answer latency metrics.
- `tests/oracle_comparison_test.rs`: existing two-layer tiny-model oracle checks argmax and sampled logits. Extend evidence to adversarial hybrid sequences and independent quantized-weight comparisons rather than asserting this proves real-model equivalence.

## Three possible approaches

1. **Native-first, evidence-gated improvement — recommended.** Retain current interfaces and scalar/reference fallbacks, address numerical uncertainty, optimize the measured operators, then test cache reuse and agent workflows. Best fit for Mivi's ownership and future models, with more kernel-engineering effort.
2. **Optional mature-runtime backend.** A separately selectable llama.cpp backend could provide a practical compatibility option. It adds packaging, process/FFI, capability, cancellation and version-maintenance costs. Never silently switch engines or claim its speed as a native Mivi improvement. Consider only with explicit approval; no evidence establishes all requested runtimes as drop-in alternatives.
3. **Context/memory platform first.** Retrieval, compiled knowledge and offloading can reduce some workload costs or accommodate larger models. They expand scope and do not remove the measured cold-prefill gap on unchanged IDs. Defer until the native runtime has a correctness/performance gate.

## Proposed phases and TODOs

### Phase 0 — trustworthy comparison and numerical diagnosis

- [x] Turn the existing private diagnostic replay into a bounded, reproducible comparison workflow, retaining non-default diagnostic exposure (Tasks 1–3).
- [x] Verify requested/effective BOS, sampler penalties, context, KV precision, numeric IDs, threads and output cap; retain terminal tokens separately. Task6 documents the unmatchable first-step EOS suppression rather than claiming identical terminal policy.
- [x] Run separate profile controls and unprofiled pairs. Task6 labels model prefill/callback/decode clocks and unavailable router/client TTFT; Task4's separate fixture boundary coverage is not a measured real-agent result.
- [x] Export existing attention/SSM substage aggregates: QKV/input/output projections, FFN, causal attention scan and convolution (Tasks 1/4). Allocation/copy/transpose attribution remains a subsequent measurement.
- [x] Probe short/medium first divergences with identical shared prefixes and bounded selected scores. Task6 reports ranking disagreements and precise missing raw-logit/activation/state evidence; the cause is unresolved and the medium disagreement is not merely a near tie.
- [x] Strengthen tiny independent fixtures: quant-block scales/mins/nibbles, odd row/tile tails, nonuniform weights, carried convolution state, GQA/RoPE positions, reset and continuation (Task 5). The hybrid graph is F32; Q4_K/Q6_K arithmetic is tested independently on serialized quant blocks.

Gate: reproducible report with known confounders, substage breakdown, and explained or explicitly unresolved divergence. Any confirmed correctness bug is fixed before promoting a faster default.

### Phase 1 — one measured native CPU optimization at a time

- [ ] If projections/FFNs dominate, first evaluate reusable per-worker scratch and blocked decoded-weight/F32-activation kernels. Preserve the checked API, supported shapes, F32 semantics, and portable fallback.
- [ ] Evaluate format-specific fused unpack/compute and optional repacking behind quantization/shape/CPU-feature dispatch. Start with formats observed in the loaded model, including mixed Q4_K/Q6_K tensors; do not infer every tensor format from the filename.
- [ ] Treat Q8 activation packing as a separate numerical mode/experiment: it adds rounding and must not silently replace faithful F32 activation processing.
- [ ] If causal scans dominate, evaluate CPU-local Q/K/V blocking, grouped-head reuse and scratch reuse while preserving causal masking and online-softmax stability. Existing online softmax is not a reason to implement another named "FlashAttention" wrapper.
- [ ] Keep recurrent state updates ordered; no approximate scan, skipped blocks, or model-name-specific dimension heuristics.
- [ ] Evaluate tile choices with explicit bounded configuration; an optional tuning result must identify hardware, model shapes, formats, precision and runtime revision. Do not launch hidden benchmarking at server startup.

Gate: scoped reference/parity tests plus repeated end-to-end improvement, no unacceptable short-prompt/quality/memory regression, and an explicit fallback. A faster microbenchmark alone does not qualify.

### Phase 2 — measured exact-prefix reuse for agent sessions

- [ ] Exercise a genuinely shared-prefix multi-turn fixture; report reused/processed tokens, snapshot/restore time and bytes, not just the label "warm". The earlier short router follow-up reused an actor but had zero KV hits.
- [ ] Preserve stable ordering/rendering of unchanged systems/tools; do not canonicalize or reorder arbitrary user content. Agent-side changing workspace headers can break early prefix matches and need direct evidence.
- [ ] Validate combined KV and SSM restore at chunk boundaries, final-token reconstruction, changed-prefix misses and bounded eviction.
- [ ] For persistence or cross-instance sharing, explicitly namespace by model/weights identity, tokenizer/template version where applicable, context/position policy, precision, adapters, runtime/backend state layout and session/privacy policy. Current instance-local cache ownership is not proof of a cross-model leak.
- [ ] Consider disk tiers or an LMCache connector only after measured restore savings exceed copying/I/O costs and a compatible hybrid serialization contract exists.

Gate: cached and fresh runs remain equivalent within the established numerical policy, changed state invalidates reuse, and repeated-session client-visible latency measurably improves. Cold requests still need Phase1.

### Phase 3 — reliable model-agnostic agent contract

- [ ] Cross-check prompt/tool profiles against each supported model's actual metadata/template. Reject unsupported architecture/tensor requirements rather than guessing LFM dimensions; add new families through explicit tested adapters.
- [ ] Audit supported request options and capability reporting. Separate schema validation after generation from actual schema-constrained decoding; reject unsupported strict-schema features clearly rather than pretending support.
- [ ] Cover required/none/auto/named tool choices, streamed tool IDs/arguments, tool-role continuation, malformed calls, unknown names/arguments/enums, truncation and cancellation.
- [ ] Keep externally supplied tools client-executed unless server execution is explicitly configured. Use existing allowlist/path/schema checks and bounded loop/result budgets; never add shell execution merely because another runtime provides it.
- [ ] Align client/server total and first-output deadlines through explicit configuration. Retain honest heartbeat/error/finish behavior and bounded worker cleanup. Increasing a timeout is not a performance fix.
- [ ] Test a generic mock tool agent before actual Minicode: exact file read and follow-up answer, then a small isolated coding task with deterministic expected behavior. Score correctness, not only valid JSON or HTTP200.

Gate: bounded agent completes correct tool/result/final-answer handoffs, failures remain visible, and Minicode integration meets an agreed deadline without leaking actual workspace contents into committed diagnostics.

### Phase 4 — context efficiency and broader models, separately approved

- [ ] Agent-side opt-in retrieval of relevant code/source-linked notes; freshness/invalidation, conflict handling and source attribution, inspired by OpenKB/Hyper-Extract.
- [ ] Budget selected schemas/tool results using exact loaded-model tokenization. Preserve instructions and tool-call/result adjacency; no silent destructive server-side context truncation.
- [ ] Evaluate retrieval's model-call and latency overhead as well as token savings and answer quality. Keep untrusted retrieved text distinct from instructions.
- [ ] Test additional existing local models sequentially, with architecture compatibility established first. Thinking-mode markers/budgets belong to profiles, not global assumptions.
- [ ] Consider out-of-core/MoE heat placement only on a supported model/workload with measured page faults, expert misses or memory pressure. Tiny fixture files establish plumbing, not real-model performance or coding quality.

Gate: new model/context paths have explicit capabilities and independent quality/performance evidence. No claim that any future GGUF works merely by choosing a profile.

## Resource and acceptance policy

All future Cargo operations: jobs1, scoped package/test/filter only; Rust tests: test-threads1. Runtime computation: two configured threads, inference concurrency1, sequential model loads and no overlapping builds/inference. Bound repeats, fixture sizes, output length, RSS, wall time, artifact bytes and cleanup. No new large-model download is needed for Phases0–3.

Start with short, medium and roughly2636-token synthetic cases and existing local models. Use three sequential paired repetitions for initial medians/ranges, alternating engine order; do not claim p95/p99 from three samples. Run independent quality checks and retain failures/timeouts in the report. Distinguish cold inference state from OS weight-page-cache warmth; do not flush system-wide caches on a shared laptop.

Proposed targets, not predictions: first meaningful optimization should show roughly20% lower agent-sized median first-output latency without unacceptable regressions. Longer-term aim: native median cold prefill within1.5× the pinned matched reference and first useful output comfortably inside the configured agent deadline. Agree targets against hardware/workload and measured spread before encoding them in tests; never make these model-specific server constants.

Refresh affected research notes: Q8 quantization is lossy; the idealized6×–10× prefill figures are not current results. The old KV table's F32 4K estimate also conflicts with its formula: `2 × 6 × 4096 × 512 × 4 = 100,663,296 bytes` (96MiB for K+V alone), not24.5MB. Include allocator/SSM/prefix overhead separately. Annotate dated historical defaults instead of treating them as current settings.

For each completed implementation increment: scoped verification, independent review where appropriate, changelog entry describing actual changes and measured limitations with ideas/inspirations/source links, one0.0.1 patch increment, and reviewed nonforce publication per the user's release policy. Research/design drafts alone are not a completed runtime feature and do not justify a speed-fix release claim.

## Immediate decision

Detailed planning is now available in the linked master roadmap and Phase0 package. The first bounded implementation deliverable is the replay contract and existing substage export, followed by matched numerical/timing comparison—not cache infrastructure, offloading, a new model, or a wholesale runtime rewrite. Choose an execution approach before starting implementation.
