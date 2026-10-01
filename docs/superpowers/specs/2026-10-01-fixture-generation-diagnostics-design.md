# Fixture-Only Generation Diagnostics

Status: implementation, all four task reviews, scoped release verification and
whole-change review completed; v0.2.62 published on GitHub main without force.
Remote/local equality was verified for reviewed release `c1e931d`.
Clipped-metadata completeness was
fixed and regression-tested in `f409922`; its re-review is approved.
Workspace version is v0.2.62. Minor coverage limitations remain documented.
Baseline: published v0.2.61, commit `8f7ae0d`.

## Objective and approved scope

Capture evidence that distinguishes prompt rendering, model generation,
tool parsing, and inference latency when an isolated tool-call fixture fails.
The user selected isolated fixtures only, not real agent traffic, and approved
capture at existing server/model boundaries instead of a duplicated inference
command or HTTP-only logging.

This increment implements capture and bounded fixture verification. Comparing
against another runtime and fixing any demonstrated model/runtime defect are
subsequent work. It must not claim improved agent quality or inference speed.
The first increment covers streaming chat completions only; native agent,
Anthropic, blocking completion, and general CLI diagnostic entry points are
not added.

## Architecture and entry point

- Use a non-default `fixture-diagnostics` Cargo feature, forwarded from
  `mivi-server` to `mivi-model` only for explicit diagnostic builds. The
  implementation plan must identify the minimal feature-gated observer seam;
  existing public generation methods and string-stream consumers remain
  source-compatible. No normal dependency or unsafe code is required.
- A deliberately invoked, ignored server fixture test drives the existing
  Axum router in-process and its real engine actor/model. It creates a fresh
  temporary workspace containing only synthetic files and sends controlled
  streaming `/v1/chat/completions` requests. It does not bind an HTTP listener
  or accept arbitrary live requests, read the project workspace, or invoke
  filesystem mutation tools.
- Capture requires both the feature and an explicitly installed per-run
  diagnostic session from that fixture harness. Normal `serve`, chat,
  benchmark, and API requests cannot enable it through a flag, header,
  request body, environment toggle, or global logger. Capture is absent from
  default builds and inactive without a session in diagnostic builds.
- The harness supplies model path and bounded run settings explicitly.
  Model/profile selection uses existing metadata or the existing explicit
  profile mechanism, never model-name tests, fixed token IDs, or delimiter
  comparisons added for this feature.
- Keep the collector and artifact writer in a focused diagnostics module.
  Do not grow the actor into a general logging subsystem or duplicate prompt
  rendering, sampling, prefill, decoding, or tool parsing.

## Data captured at real boundaries

One request record correlates fixture identity and sequence with:

1. The exact UTF-8 prompt produced by the existing model profile, including
   whitespace and trailing newlines, and the separately identified forced
   prefix. Record the exact conditioned prompt given to the model as well;
   it must equal rendered prompt plus that prefix under current behavior.
2. Bounded raw decoded model text at the decoder boundary, before configured
   stop-sequence trimming, synthetic prefix insertion, and tool parsing.
   Decoder flush text is included. This is decoded UTF-8 text, not a claim of
   capturing original token bytes. Record bounded generated token IDs separately
   when available, using the existing generated-ID sequence (termination
   tokens excluded), without importing another tokenizer's vocabulary.
3. Bounded delivered callback text after existing stop handling but before
   actor prefix insertion, plus the parser-facing result observed by the
   ordinary router. Label these separately; raw output must never include a
   server-inserted prefix or be mistaken for validated tool arguments.
4. Existing model identity, effective sampling options, token budget, context
   limit, prefill strategy/tile size, inference thread configuration, prompt
   token counts including any metadata-directed BOS, and reused-prefix count
   when exposed by the prefill boundary. Never serialize API keys, environment
   dumps, absolute project paths, or private fixture source paths.
5. Physical generation lifecycle and result: successful completion, cancellation,
   model failure, or receiver closure. Preserve the router's finish/error result
   separately. Completion before any decoded output is a valid empty capture,
   not a fabricated first-output sample. Capture truncation and persistence
   failure are diagnostic states, not generation completion reasons.

Output strings are evidence only. Do not repair malformed calls, infer missing
arguments, relax validation, change model profiles, or reinterpret delimiters.

## Timing semantics

Use monotonic elapsed durations from the actual execution boundaries:

- Tokenization and prefill/cache-restore wall time are separate observations.
  The prefill timer begins before prefix-cache lookup and ends after the
  existing prefill call returns; record restored and newly processed tokens
  separately rather than attributing reused tokens to new computation.
  Prefill ends only when the existing prefill path returns, with its success,
  cancellation, or error result recorded. Do not infer this from first output.
- Decode wall time begins after successful prefill and ends when generation
  returns. It includes existing callback delivery/backpressure; do not label
  it pure kernel time or add it to overlapping forward-stage profiles.
- Record time to first nonempty raw decoded text and first nonempty delivered
  callback text separately, both relative to the same physical generation
  entry instant before tokenization. Either may be absent, and stop handling
  can cause them to differ. Headers, role events, synthetic prefixes, and keep-alives
  count as neither.
- Harness-visible first tool/content delta and total stream duration are
  separate client-observation measurements. Compare per-request server metric
  deltas without substituting those metrics for observed SSE timestamps.
- A timed-out API request may release its admission permit before model work
  finishes. Finalize the engine record at actual generation return, not at
  HTTP timeout; distinguish admission reuse from physical cancellation.
- Start with a fresh model per cold fixture run; any warm sequential fixture
  is explicitly labelled. Profiling overhead and shared-host load are disclosed;
  these runs are diagnostics, not cold-cache throughput benchmarks.

## Bounds, privacy, and failures

The harness configures finite request count, prompt/output byte caps, token-ID
cap, stream deadline, and collector capacity. Reject zero, overflow, or
unbounded settings. Retain a UTF-8-safe prefix when a text cap is reached and
mark that field truncated with observed and retained lengths; never silently
present truncated text as byte-exact. Counters must not overflow.

Collectors retain bounded data in memory. No file writes, formatting, or
blocking diagnostic-channel sends occur per model token. Persistence happens
after capture, outside measured generation work. When capacity is exhausted,
mark the capture incomplete; never block inference or silently overwrite a
previous request's evidence.

Write only into a newly created private temporary artifact directory. On Unix,
use directory permissions 0700 and file permissions 0600; create files
exclusively, avoid following user-provided symlinks, and never overwrite
existing artifacts. Use fixture IDs rather than private path names in records.
No raw prompts/output are printed to regular logs, streamed to a service,
committed, or uploaded to GitHub. Report the local artifact directory and
capture status without echoing captured content.

Invalid diagnostic configuration fails before model loading. A capture or
persistence failure remains visible and makes the diagnostic verification
unsuccessful, but must not change model output, error propagation, sampling
checkpoint restoration, cancellation, receiver closure, or channel backpressure.
Use bounded watchdogs and finish/join only harness-owned workers; do not kill
unrelated servers or jobs. Retain partial captures with honest terminal states
where possible, without pretending a still-running engine has finalized.

## Verification and release

Observe behavioral assertion RED/GREEN for enabled capture, bounds, byte-exact
prompts/newlines, prefix separation, lifecycle/timing, and persistence failures.
Test that default and feature-enabled-without-session paths capture nothing.
Compare observer-on/off output under identical deterministic settings to check
that diagnostics do not change generation or sampler state.

Unit tests cover empty output, whitespace, multibyte truncation, receiver
closure, cancellation during prefill/decode, generation errors, collector
overflow, exclusive/private artifact creation, and sequential request isolation.
Avoid scheduling sleeps as evidence of cancellation or timing boundaries.

Run scoped server/model tests and checks only: Cargo jobs=1, test threads=1,
Rayon/inference threads=2, with sequential Cargo commands/model loads. No
full-workspace check/build/test. An explicitly invoked bounded real-model
fixture records a tool request and byte-exact tool-result continuation using
the actual selected model. A malformed result is preserved, not hidden or
counted as a successful tool answer. Raw model evidence and capture correctness
are assessed separately from model answer quality.

After implementation, scoped/live verification, and independent review, update
the changelog with measured evidence, limitations, ideas/inspirations/sources;
bump v0.2.61 to v0.2.62 once and publish without force under the user's standing
authorization. Rebuild only the root `mivi` package when a release executable
is needed; `mivi-cli` alone does not build that executable. Preserve/exclude
the user's `.gitignore`. Any subagents must be GPT-6 Luna, high reasoning.

## Ideas, inspirations, and sources

- Preserve evidence before interpretation: separate rendered/conditioned prompt,
  decoded model output, actor-added prefix, and validated parser result.
- Reuse the project's real router, actor, decoder, and opt-in profiling rather
  than inventing an independent rendering/generation path:
  [chat rendering and validation](https://github.com/aswin402/mivi-v4/blob/8f7ae0d/crates/mivi-server/src/routes/chat.rs),
  [engine actor](https://github.com/aswin402/mivi-v4/blob/8f7ae0d/crates/mivi-server/src/engine_actor.rs),
  [model prefill/decoder and profiling](https://github.com/aswin402/mivi-v4/blob/8f7ae0d/crates/mivi-model/src/model.rs),
  [existing benchmark timing](https://github.com/aswin402/mivi-v4/blob/8f7ae0d/crates/mivi-cli/src/runners/bench.rs).
- The v0.2.61 fixture verification motivates byte-exact tool results and
  separate decoded-output/SSE timing; correcting an omitted trailing newline
  changed that fixture's final answer. This observation is not a universal
  model-quality conclusion or proof of a parser defect.
