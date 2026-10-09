# Native CPU Agent Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make native Mivi a measurably faster, correct, and honest local agent provider without model-name hardcoding or unsupported compatibility claims.

**Architecture:** Retain the Rust runtime and existing protocol, cache, cancellation, and diagnostic boundaries. Establish numerical and timing evidence before selecting individual optimizations; independently validate hybrid-prefix reuse and the agent contract. Retrieval, offloading, and alternative backends remain separately approved projects.

**Tech Stack:** Existing Rust workspace, GGUF, Rayon, serde, Python standard-library comparison tooling, existing Python reference engine, local models, optional pinned llama.cpp reference.

## Global Constraints

- Baseline: version `0.2.62`, commit `0fe99c81275ef9dbef1f133ad8f5a59a1584b83f`; recheck at execution time.
- Cargo jobs **1**; Rust test threads **1**; inference/Rayon threads **2**; inference concurrency **1**.
- Run only named packages, targets, and filters. Never full-workspace check/build/test. No overlapping builds, model loads, or inference runs.
- Preserve the user's `.gitignore` change. Explicitly stage only task-owned files; never `git add .`.
- Localhost needs no API key. Do not weaken nonlocal security or add server-side shell execution.
- Defaults, safety limits, and supported architecture adapters are legitimate configuration/contracts. Do not encode model filenames, laptop measurements, or guessed dimensions as runtime behavior.
- Raw requests, IDs, outputs, weights, state, and logs remain in bounded private artifacts; no normal-server capture flag and no automatic uploads.
- If subagents are selected: `gpt-6-luna`, high reasoning, no nested delegation; verify availability instead of silently substituting.
- Planning does not change runtime version. Each completed implementation increment gets one patch bump, changelog with sources and limitations, scoped verification, review, and a nonforce GitHub push under the standing release policy.

---

## Status and dependencies

This is the master roadmap and TODO tracker. The [Phase 0 plan](2026-10-02-runtime-parity-and-profiling.md) is the first execution package. Later packages below specify scope, files, tests, and promotion gates; write their focused execution plans after the relevant evidence gate, not speculative kernels now.

- [x] Research the eight requested projects and primary papers; document applicability and limitations in [the research report](../../AGENT_CPU_RUNTIME_RESEARCH_AND_PLAN.md).
- [x] Obtain an initial matched-ID, two-thread CPU comparison. **Single samples only; output divergence remains unresolved.**
- [x] P0. Establish repeatable timing, substage attribution, numerical diagnosis, and independent oracle coverage. [Task6 findings](../../CPU_RUNTIME_EVIDENCE_2026-10-05.md) retain unresolved short/medium ranking differences; parity/approximate promotion remains blocked.
- [ ] P1. Select and implement one measured native optimization, then reassess.
- [ ] P2. Validate and improve exact multi-turn prefix reuse.
- [ ] P3. Validate architecture/capability declarations and bounded agent interoperability.
- [ ] P4. Seek separate approval for context efficiency, additional architecture adapters, or memory expansion.

```text
P0 evidence ──> P1 selected optimization ──> repeat measurements
     │                    │
     ├──> P2 exact reuse ──┤
     └──> P3 contract ─────┴──> mock agent ──> isolated Minicode trial
                                              │
                                       separately approved P4
```

Any confirmed correctness bug takes precedence over speed. Protocol tests can proceed independently of kernel work; expensive model runs remain sequential.

## File/responsibility map

| Area | Existing files to extend | Responsibility |
| --- | --- | --- |
| Measurements | `crates/mivi-model/src/fixture_diagnostics.rs`; `crates/mivi-server/src/fixture_diagnostics.rs`; `crates/mivi-server/src/fixture_diagnostics/fixtures.rs` | Bounded opt-in captures and router/worker boundaries |
| Forward profiles | `crates/mivi-model/src/model.rs`; `transformer.rs`; `ssm.rs` in that directory | Existing stage counters; no duplicate timer hierarchy |
| CPU kernels | `crates/mivi-quant/src/lib.rs`; `q4_k_m.rs`; `q6_k.rs`; `types.rs` in that directory | Checked dispatch, portable fallbacks, scratch and locality |
| Existing experiments | `crates/mivi-quant/src/q4_k_m/packed_prefill.rs` and its `group32`/`real_weights` modules | Benchmark-only packed activations and cumulative drift; not production inference |
| Model integration | `crates/mivi-model/src/ffn.rs`; `prefill.rs`; `transformer.rs`; `ssm.rs` | Worker-owned buffers and ordered hybrid computation |
| Exact reuse | `crates/mivi-kv/src/prefix.rs`; `cache.rs`; `crates/mivi-model/src/model.rs` | KV plus recurrent-state restoration and final-token logits |
| Architecture | `crates/mivi-model/src/loader.rs`; `config.rs`; `weights.rs` | Metadata/tensor validation and explicit family support |
| Agent contract | `crates/mivi-server/src/model_profile.rs`; `generation.rs`; `types.rs`; `routes/chat.rs`; `streaming.rs`; `watchdog.rs` | Profiles, capabilities, tool/SSE semantics, deadlines |
| Schema validation | `crates/mivi-tools/src/`; `crates/mivi-server/src/grammar.rs`; `crates/mivi-model/src/grammar.rs` | Existing validation versus actual decoding constraints |
| Independent oracle | `reference/reference_engine.py`; `training/export/generate_fixture.py`; `tests/oracle_comparison_test.rs` | Independent computation over the same serialized weights |
| Agent evaluation | `scripts/test_agents/02_agent_loop.py`; proposed `scripts/test_agents/04_provider_contract.py` | Client-owned deterministic tool loop, no workspace-wide edits |
| Release/docs | `Cargo.toml`; `Cargo.lock`; `CHANGELOG.md`; affected research notes | Actual release changes, attribution, measured limits |

Names without a directory in a table cell belong to the directory of its first full path. Inspect current files again before implementation; do not treat this map as permission for broad restructuring.

## P0 — Trustworthy comparison and numerical diagnosis

**Deliverables:** private replay manifest/results, existing substage export, repeated matched comparison, teacher-forced divergence report, adversarial independent fixtures, corrected research notes.

- [x] Execute Tasks 1–6 in [the detailed P0 plan](2026-10-02-runtime-parity-and-profiling.md).
- [x] Record long-case split-prefill versus normal-call parity, not only the existing short check. Task2's 110/2636-token LFM fixture passed both cases; this is same-engine synthetic evidence only.
- [x] Compare separate profile controls against three unprofiled samples per workload; outputs/stopping match, timing variation is disclosed, and setup failures are identified separately.
- [x] Mark short/medium divergence unresolved with missing reference raw logits and real-model hybrid/activation traces; long outputs match in all three pairs.
- [x] Correct the independently tested Python reference RoPE/reset gaps in Task5. No real-model graph/layout defect is confirmed by Task6; future regressions still block promotion.

**Exit gate:** three paired samples per workload with medians/ranges, matched effective settings and IDs, distinct timing boundaries, substage evidence, independently checked quantized arithmetic/hybrid state, and no hidden failures. Full hidden-state parity with llama.cpp is not assumed available through its HTTP API.

## P1 — Native CPU performance packages

### P1-A: Faithful scratch/locality optimization

**Depends on:** P0 showing projections/FFNs or copy/allocation costs materially affect latency.

**Files:** `crates/mivi-quant/src/lib.rs`, `types.rs`; `crates/mivi-model/src/ffn.rs`, `prefill.rs`, `transformer.rs`, `ssm.rs`. Introduce a focused scratch module only if ownership cannot fit the existing activation workspace.

**Boundary:** keep `quantized_matmul_rows(out, ggml_type, weights, inputs, batch, rows, cols) -> Result<()>` checked and available. A new scratch-taking variant must preserve the old wrapper and use caller-owned buffers, not mutable globals or per-model-name dispatch.

- [x] Measure allocation/transpose/weight decoding separately using short synthetic shapes and observed real tensor shapes. The [bounded projection cost evidence](../../PROJECTION_COST_EVIDENCE_2026-10-05.md) separates allocation-plus-initialization, stage wall time, and worker work; allocation alone is not isolated. Fifteen accepted pairs select a faithful locality experiment as a hypothesis, not a production optimization or speedup claim.
- [x] Complete the bounded, exact-parity [faithful column-panel locality experiment](../../PROJECTION_LOCALITY_EVIDENCE_2026-10-06.md) against the production baseline. Fresh Q6_K medians were 0.909–0.935× baseline; synthetic Q8_0 was mixed, with a noisy 1.280 paired ratio for panel 128. The three-repetition pilot is inconclusive; no selector is promoted and all broader P1-A gates below remain open.
- [x] Write baseline-reference comparisons for batch 1/2/8/9/32/64/65, odd output rows, valid format block widths, and invalid buffers/overflow. The [stage-1 scratch evidence](../../BATCH_SCRATCH_EVIDENCE_2026-10-09.md) covers six formats and one/two-thread pools; real-model and cross-ISA gates remain separate.
- [x] Define caller-owned quant-operator scratch capacity and demonstrate reuse across successive shapes/formats. Stage 1 is explicit and default-off; ownership in model prefill workspaces is not yet integrated.
- [ ] Implement one blocked F32-activation path without changing activation precision.
- [ ] Run `CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 cargo test -p mivi-quant matmul -- --test-threads=1`; require a nonzero matching test count.
- [ ] Run selected model chunked-versus-token parity and the P0 comparison; inspect RSS and short-case regressions.
- [ ] Promote only the winning path; keep a portable fallback and record negative experiments.

### P1-B: Optional packed-activation mode

**Depends on:** P0 numerical policy plus P1-A evidence; separate decision to accept changed arithmetic.

**Files:** reuse `crates/mivi-quant/src/q4_k_m/packed_prefill.rs`, `packed_prefill/group32.rs`, and `packed_prefill/real_weights/captured_activations/cumulative.rs`. Production wiring additionally touches `mivi-model` configuration and projection dispatch only after promotion approval.

- [ ] Review existing scalar/AVX2/tiled and group32 experiments; do not repeat already-completed fixture work.
- [ ] Re-run their relevant nonignored controls with `cargo test -p mivi-quant packed_prefill -j 1 -- --test-threads=1` and `RAYON_NUM_THREADS=2`.
- [ ] Report tensor eligibility and Q6/F32 fallback coverage from actual GGUF metadata, not `Q4_K_M` in a filename.
- [ ] Validate per-projection, cumulative FFN, teacher-forced logit, generated-output, and agent-quality drift separately.
- [ ] Specify a typed opt-in numerical mode with explicit unsupported-shape/CPU fallback; faithful processing stays default until approved evidence supports a change.
- [ ] Benchmark complete requests, including packing, allocation, and mixed-format fallback overhead.

**Stop condition:** insufficient quality/parity or no end-to-end gain means retain benchmark-only status. Faster isolated dots are not a release claim.

### P1-C: Attention/SSM/tile package, only if measured

**Files:** `crates/mivi-model/src/transformer.rs`, `ssm.rs`, `prefill.rs`, `config.rs`; `crates/mivi-core/src/` only for a demonstrated reusable primitive.

- [ ] If causal scan dominates, design grouped-head shared-KV locality and scratch reuse around existing online softmax.
- [ ] Test causal boundaries, GQA mapping, nonzero RoPE positions, finite extreme logits, and uneven final tiles against the scalar path.
- [ ] If convolution dominates, optimize storage/access while preserving token-order recurrence and carried state exactly under the numerical policy.
- [ ] Evaluate explicitly configured tiles 16/32/64/128 sequentially, with matching context/output limits and whole-request results.
- [ ] If tuning persistence is justified, key results by hardware/features, tensor shapes/formats, precision, and runtime revision. Reject incompatible records; never auto-benchmark on server startup.
- [ ] Write a focused execution plan for whichever operator wins attribution; do not implement all branches together.

**P1 exit gate:** agreed end-to-end improvement beyond measured spread; proposed initial target approximately 20% lower agent-sized median first useful output, no unacceptable quality/RSS/short-case regression. Longer-term target within 1.5× the pinned reference is aspirational, not a hardcoded test or promise.

## P2 — Exact hybrid-prefix reuse

**Files:** `crates/mivi-kv/src/prefix.rs`, `cache.rs`; `crates/mivi-model/src/model.rs`; `crates/mivi-server/src/fixture_diagnostics/fixtures.rs`; `model_profile.rs` only if unstable rendering is demonstrated.

**Consumes:** exact rendered/tokenized prompts and P0 numerical policy. **Produces:** measured reuse of causally identical prefixes, not arbitrary suffix matches.

- [ ] Create a two-turn synthetic agent case with unchanged system/tool prefix long enough to cross a real cache chunk boundary.
- [ ] Compare fresh and restored runs at boundary−1/boundary/boundary+1; check KV, convolution/hidden state, and final-token logits.
- [ ] Assert `reused_tokens > 0`, `processed_tokens < prompt_tokens`, and output equivalence for the actual cache-hit case.
- [ ] Add changed-first-token/tool-template, eviction, adapter, and precision invalidation tests where the current ownership permits those changes.
- [ ] Measure snapshot/restore duration and retained bytes; do not call actor reuse a cache hit.
- [ ] Preserve stable unchanged tool rendering without reordering user messages or tool-call/result adjacency.
- [ ] Run `cargo test -p mivi-kv prefix -j 1 -- --test-threads=1`, then `cargo test -p mivi-model prefix_cache -j 1 -- --test-threads=1`, with `RAYON_NUM_THREADS=2`.
- [ ] Run the new release-only shared-prefix router fixture sequentially and report visible latency with cold/warm labels.

**Exit gate:** exact restore within policy, correct misses/eviction, bounded memory, measurable warm-request gain. Current instance-local ownership is not a confirmed cross-model cache leak.

Persistence/cross-instance sharing needs a separate design for weights identity, tokenizer/template revision, context/position semantics, precision, adapters, layout/backend version, and session privacy. No disk tier or LMCache connector before measured copy/I/O savings and compatible complete hybrid-state serialization.

## P3 — Model-agnostic agent contract

### P3-A: Architecture and capabilities

**Files:** `crates/mivi-model/src/loader.rs`, `config.rs`, `weights.rs`; `crates/mivi-server/src/model_profile.rs`, `types.rs`, `generation.rs`, `grammar.rs`.

- [ ] Build a metadata/tensor compatibility matrix from local files; protocol profile selection must not imply architecture support.
- [ ] Add deterministic rejection tests for unsupported architecture, missing/mismatched tensors, contradictory dimensions, and unsupported request options.
- [ ] Keep architecture adapters explicit and tested; never guess LFM dimensions for an arbitrary GGUF.
- [ ] Define capabilities for tools, tool choices, JSON syntax, strict schema constraints, streaming, context, and thinking profiles from actual implementation support.
- [ ] Reject unsupported strict-schema features clearly. Existing post-generation schema validation is not constrained decoding.
- [ ] Verify local no-key requests and ensure nonlocal access policy is unchanged.
- [ ] Run targeted loader/profile/type/grammar filters in `mivi-model` or `mivi-server`, one Cargo operation at a time, jobs1/test-threads1.

### P3-B: Bounded protocol and real agent evaluation

**Files:** `crates/mivi-server/src/fixture_diagnostics/fixtures.rs`, `routes/chat.rs`, `streaming.rs`, `watchdog.rs`; `scripts/test_agents/04_provider_contract.py`; existing `02_agent_loop.py` as a behavior reference.

- [ ] Add mocked transport cases for `auto`, `none`, `required`, and named tool choices; validate streamed call IDs and argument reconstruction.
- [ ] Add unknown tool/argument/enum, malformed/truncated call, tool-result continuation, cancellation, timeout, and worker-cleanup cases.
- [ ] Keep tools client-executed by default. Test an allowlisted synthetic `read_file` fixture with path containment, result-size and turn limits; no arbitrary shell commands.
- [ ] Record HTTP headers, first useful tool/content delta, terminal reason, `[DONE]`, EOF, and physical worker return separately. Heartbeats never count as useful output.
- [ ] Configure explicit client/server deadlines with measured headroom; do not disguise slow inference by only increasing timeout.
- [ ] First complete a generic read-only tool loop; then run Minicode on a disposable tiny project with one specified edit and a deterministic test, not the real repository.
- [ ] Fix model/provider/client settings per trial. Count wrong code, unsupported options, invalid calls, and deadline failures separately from network failures.
- [ ] Compare protocol success and coding quality independently; the LFM card's programming limitation remains relevant even if transport is perfect.

**Exit gate:** correct bounded tool/result/final-answer handoff, truthful capability failures, clean cancellation, and useful output inside the agreed client deadline. Do not publish “Minicode works” based only on HTTP200 or a valid call envelope.

## P4 — Optional expansion, not authorized implementation

- [ ] Ask separately whether agent-side retrieval/context work is wanted after P0–P3.
- [ ] For retrieval: write a focused design with source identities, freshness/invalidation, conflict attribution, opt-in selection, and untrusted-text isolation; use OpenKB/Hyper-Extract as inspiration, not a mandatory graph service.
- [ ] Budget with the actual loaded tokenizer; preserve instructions and tool adjacency. Report schema/result/retrieval overhead and answer quality, not just fewer input tokens.
- [ ] For additional local models: establish architecture support first; load/test one at a time, configure model-specific template/thinking behavior through profiles, and score tools and coding separately.
- [ ] For out-of-core/MoE: require supported architecture plus measured memory pressure/page faults/expert misses before designing residency/prefetch. AirLLM/Colibri are references, not evidence of a cold-prefill fix here.
- [ ] For an alternative llama.cpp backend: seek explicit approval; design process/FFI lifecycle, version pinning, capabilities, and cancellation. Never silently replace the native runtime or attribute reference-engine speed to it.

## Release checklist for each completed increment

- [ ] Review exact diff and confirm unrelated files/model artifacts are absent.
- [ ] Run only affected scoped tests; record actual counts and commands. Performance changes also require paired whole-request measurements and numerical/quality gates.
- [ ] Increment the then-current patch version once; refresh `Cargo.lock` using a scoped Cargo operation, jobs1, not a full build.
- [ ] Add changelog: what changed, defaults/fallbacks, measured evidence and limitations, ideas/inspirations, direct source links, license notices if applicable.
- [ ] Stage explicit task/release files, review staged diff, commit, and push the current branch nonforce to the configured GitHub remote. Request sandbox/network approval when required.
- [ ] Confirm remote commit and publish concise results; preserve `.gitignore` and private artifacts.

Sources and evidence boundaries are maintained in [the research report](../../AGENT_CPU_RUNTIME_RESEARCH_AND_PLAN.md). Prioritize its pinned GGML references, Kimi oracle discipline, Colibri benchmark protocol, LMCache hybrid-state guidance, mistral.rs capability separation, and Liquid AI's actual tool format. Check licenses before copying source, rather than merely borrowing an idea.

## Immediate next action

Tasks1–6 have completed the Phase0 evidence package. The measured runtime is
v0.2.67; this documentation checkpoint does not change its inference arithmetic.
Select P1-A only: first measure projection/FFN allocation, transpose, decode and
compute, then evaluate faithful scratch/locality improvements. Short/medium
cross-engine divergence is unresolved; approximate modes and parity claims remain
blocked. Agent quality still needs the separate P3 evaluation.
