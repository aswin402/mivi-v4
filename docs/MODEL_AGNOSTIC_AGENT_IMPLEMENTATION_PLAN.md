# Mivi Model-Agnostic Agent Compatibility Plan

**Status:** In progress
**Scope:** `/v1/chat/completions`, `/v1/messages`, `/v1/mivi/agent`, model loading, tool calling, streaming, context limits
**Constraint:** Never run full-workspace Cargo commands. All verification uses targeted packages and `--jobs 1`.

## Objective

Make Mivi reliable for agent clients while keeping model-specific behavior out of the HTTP server and core agent loop.

The system must support:

- LFM2.5 native prompt and tool syntax.
- Existing generic chat models.
- Future models with different chat templates, tool-call syntaxes, thinking formats, stop tokens, and context limits.
- OpenAI-compatible clients such as Cline, Roo, Continue, and the OpenAI SDK.
- Mivi's own internal agent endpoint and broker tools.

## Architecture rule

Do not solve this by adding more special cases to `routes/chat.rs` or by forcing every model to use one markup format.

Use this translation pipeline:

```text
External API request
  -> protocol adapter
  -> canonical conversation + canonical tools
  -> model capability/profile adapter
  -> model prompt
  -> model output decoder
  -> canonical generation events
  -> protocol response/SSE adapter
```

The HTTP layer must not know whether a model uses ChatML, LFM2.5 Python-style calls, JSON calls, XML calls, or another format.

## Current progress

- [x] Added shared `mivi-protocol` canonical message, tool-call, and tool-definition types.
- [x] Added lossless OpenAI message/tool normalization at the server boundary.
- [x] Added configurable delimited Python-style tool-call decoding.
- [x] Added metadata-driven model profile selection and profile-owned prompt rendering.
- [x] Added validated external JSON model profiles and CLI override support.
- [x] Stream plain text immediately for tool-enabled requests while buffering only undecided tool-call output.
- [x] Emit OpenAI-compatible tool-call headers and monotonic JSON argument deltas for the native codec.
- [x] Add configured streaming deadlines and cooperative cancellation on client disconnect.
- [x] Complete native adapter history handling, including null assistant content and tool-result replay.
- [x] Route Anthropic prompt rendering and tool-output parsing through the selected model profile.
- [x] Route the internal agent loop through the selected tool codec and canonical native tool history.
- [x] Bound each internal agent generation step with configured deadlines and cooperative cancellation.
- [x] Apply first-token, total-generation, and disconnect cancellation behavior to Anthropic streaming.
- [x] Use tokenizer-derived prompt counts in request diagnostics instead of byte-length estimates.

## Canonical internal contracts

Create model-neutral types before changing behavior:

- `Conversation`: ordered messages with role, content parts, optional reasoning, assistant tool calls, and tool results.
- `ToolDefinition`: name, description, and JSON Schema parameters without OpenAI wrapper fields.
- `ToolCall`: stable call id, tool name, typed JSON arguments, and parse/validation status.
- `GenerationEvent`: text delta, reasoning delta, tool-call start, tool-argument delta, tool-call complete, finish, and error.
- `ContextBudget`: prompt tokens, requested output tokens, reserved output space, model capacity, and admission result.

Preserve all information received from clients. In particular, do not discard `tool_call_id`, `content: null`, `parallel_tool_calls`, message content parts, or provider extension fields before an adapter has handled them.

## Model capability/profile design

Add a model capability layer near `mivi-model`/`mivi-tokenizer`.

Suggested responsibilities:

- Read GGUF tokenizer metadata, special-token ids, EOS/BOS behavior, context metadata, and embedded `tokenizer.chat_template` when present.
- Select a profile from metadata or an explicit external model configuration.
- Expose capabilities rather than model-name conditionals:
  - template source and rendering mode;
  - tool-call wire format and parser;
  - reasoning format;
  - supported roles;
  - parallel-call support;
  - context capacity;
  - stop conditions.
- Fail clearly when a model has no compatible template/profile. Do not silently use a guessed template for tool-enabled requests.

Model profiles should be declarative configuration or isolated adapter implementations, not constants spread across server code. A profile may contain special-token strings as model data, but the generic server must never embed those strings.

Preferred resolution order:

1. Explicit profile supplied with the model configuration.
2. Embedded GGUF tokenizer/chat-template metadata.
3. A generic text-only fallback for requests without tools.
4. A clear unsupported-capability error for tool requests when no tool codec exists.

The server accepts an explicit profile with `mivi serve --model-profile PATH`. The checked-in
[LFM2.5 example profile](lfm2.5-profile.json) is configuration data only; future model families
can provide a separate JSON profile without changing the HTTP routes.

## Phased implementation TODO

### Phase 0 — Reproduce and freeze the failure

- [ ] Capture one real request from the failing client, including `stream`, `tools`, `tool_choice`, `parallel_tool_calls`, context size, and timeout.
- [ ] Capture the complete raw SSE response until `[DONE]` or disconnect.
- [x] Add a small red/green fixture test for the current failure:
  - agent tool request;
  - native LFM2.5 tool output;
  - multi-turn assistant tool call plus `tool` result.
- [ ] Separate server response-created, first-token, generation-complete, client-disconnected, and timeout metrics in logs.

No model or server behavior should be redesigned before these fixtures exist.

### Phase 1 — Introduce canonical messages and tools

Primary files:

- `crates/mivi-server/src/types.rs`
- `crates/mivi-tools/src/schema.rs`
- new model-neutral types under `crates/mivi-tokenizer/src/` or a small shared crate if dependency direction requires it

- [ ] Replace lossy `MessageDto -> ChatMessage` conversion with a lossless canonical conversion.
- [ ] Preserve `tool_call_id`, assistant `tool_calls`, `role: tool`, null content, content arrays, and provider metadata.
- [ ] Normalize OpenAI tool wrappers into canonical `ToolDefinition` values at the protocol boundary.
- [ ] Keep OpenAI serialization separate from internal representation.
- [ ] Add unit tests for every supported message shape.

### Phase 2 — Add model profile and template interfaces

Primary files:

- `crates/mivi-tokenizer/src/lib.rs`
- `crates/mivi-tokenizer/src/chatml.rs`
- `crates/mivi-tokenizer/src/special.rs`
- `crates/mivi-model/src/gguf.rs`
- `crates/mivi-model/src/config.rs`
- `crates/mivi-model/src/model.rs`

- [ ] Define `ChatTemplateAdapter`/equivalent interface that accepts canonical messages and tools.
- [x] Define `ToolCallCodec` interface for encoding tool instructions and decoding generated tool calls.
- [x] Move template selection out of `routes/chat.rs`.
- [ ] Load BOS/EOS/special-token ids and context capacity from model metadata where available.
- [ ] Represent missing metadata as an explicit capability state rather than silently inserting a hardcoded value.
- [x] Add an explicit generic text-only profile for models that do not support tools.

### Phase 3 — Implement LFM2.5 as the first isolated adapter

Primary files:

- new isolated LFM2.5 profile/codec module under `crates/mivi-tokenizer/src/` or `crates/mivi-model/src/`
- `crates/mivi-tools/src/parser.rs`
- `crates/mivi-server/src/types.rs`

- [x] Render the official LFM2.5 prompt structure through the adapter, including its BOS behavior and native tool list.
- [x] Render assistant tool-call history in the native LFM2.5 representation.
- [x] Render tool results with the canonical `tool` role mapping.
- [x] Parse native Python-style calls between the model's tool-call delimiters.
- [x] Support strings, escaped strings, numbers, booleans, nulls, arrays, objects, and multiple calls.
- [x] Validate parsed arguments against the canonical schema.
- [x] Retain the old XML/JSON parser only as an explicitly documented legacy codec, not as the default for every model.
- [ ] Add golden prompt/output fixtures generated from the official LFM2.5 examples.

### Phase 4 — Rebuild the OpenAI and Anthropic protocol adapters

Primary files:

- `crates/mivi-server/src/routes/chat.rs`
- `crates/mivi-server/src/routes/anthropic.rs`
- `crates/mivi-server/src/streaming.rs`
- `crates/mivi-server/src/generation.rs`

- [x] Convert external requests into canonical messages/tools before generation.
- [x] Support `tool_choice`: `none`, `auto`, `required`, and named tool choices according to model capability.
- [x] If a model cannot guarantee required tool use, return a precise capability error or apply a documented best-effort policy.
- [ ] Convert canonical tool calls back to valid OpenAI `tool_calls` with stable ids and JSON arguments.
- [ ] Convert canonical tool results back to the requested protocol format.
- [ ] Keep `/v1/chat/completions` as an external tool-calling API; do not mix it with internal broker execution.
- [x] Keep `/v1/mivi/agent` responsible for Mivi's broker and ReAct loop, while reusing the same canonical/model adapter layers.
- [x] Render internal agent prompts and follow-up history through the selected profile for both legacy and native models, advertising only allowed tools.

### Phase 5 — Correct streaming, cancellation, and timeout behavior

Primary files:

- `crates/mivi-server/src/routes/chat.rs`
- `crates/mivi-server/src/streaming.rs`
- `crates/mivi-server/src/routes/mod.rs`
- `crates/mivi-server/src/logging.rs`
- `crates/mivi-server/src/engine_actor.rs`

- [ ] Stream normal text as soon as it is safely classified.
- [ ] Buffer only the undecided tool-call prefix; do not buffer an entire tool-enabled generation unnecessarily.
- [ ] Emit OpenAI-compatible tool-call deltas and finish with `finish_reason: tool_calls`.
- [ ] Emit a terminal error event if parsing, validation, cancellation, or inference fails.
- [x] Add a configured first-token deadline that distinguishes slow prefill from a stalled model.
- [x] Add a configured total-generation deadline that covers the spawned inference task.
- [x] Cancel model generation when the client disconnects or the deadline expires.
- [x] Reject OpenAI requests whose tokenized prompt plus reserved output exceeds model context.
- [x] Reuse model-derived context admission for Anthropic requests.
- [x] Recheck context admission before every internal agent generation step.
- [ ] Log stream completion separately from HTTP header/response creation.
- [ ] Keep heartbeats for transport liveness, but do not treat them as a substitute for model-output events.

### Phase 6 — Context admission and truthful limits

Primary files:

- `crates/mivi-model/src/model.rs`
- `crates/mivi-server/src/config.rs`
- `crates/mivi-server/src/routes/chat.rs`
- model listing/status handlers

- [x] Calculate real prompt token count with the loaded tokenizer, not `prompt.len() / 4`.
- [x] Check `prompt_tokens + reserved_output_tokens <= model_context_capacity` before generation.
- [x] Reserve output space and return a clear `context_length_exceeded` error when admission fails.
- [x] Report the actual context capacity in `/v1/models` and `/v1/mivi/status`.
- [ ] Report `finish_reason: length` when the model reaches context capacity.
- [ ] Add configurable context compaction/truncation policies only after correct admission checks exist.
- [ ] Default agent tests to a realistic context budget rather than accepting a client advertisement such as 204,800 tokens.

### Phase 7 — Future-model support and registry

- [x] Add a model capability report API (`/v1/mivi/status`) showing the selected profile, tool codec, context size, and streaming/tool support.
- [x] Add profile loading from a validated external configuration format so new model families do not require server-route edits.
- [x] Add a profile conformance test suite that every model adapter must pass.
- [ ] Keep model-specific code in separate modules with no cross-model conditionals in HTTP handlers.
- [x] Document how to add a new model adapter and how to declare a text-only model.

## Verification strategy

Use only targeted, single-job checks:

```text
cargo test -p mivi-tokenizer --jobs 1
cargo test -p mivi-tools --jobs 1
cargo test -p mivi-server --jobs 1
cargo test -p mivi-server --test <specific-test> --jobs 1
cargo check -p <changed-package> --jobs 1
cargo build -p <changed-package> --jobs 1
```

Do not run full workspace `cargo check`, `cargo build`, or `cargo test` unless explicitly requested.

Required test layers:

- Unit: canonical message conversion, profile selection, template rendering, tool parsing, argument validation.
- Golden: exact prompt/output fixtures for each model profile.
- Protocol: OpenAI non-streaming and SSE responses, Anthropic responses, tool choice variants, `[DONE]` handling.
- Integration: first tool call, tool result, final answer, multiple calls, malformed calls, client disconnect, timeout, and context overflow.
- Real-model smoke: run a short request against each available GGUF with a small prompt and one deterministic tool.
- Agent replay: replay captured Cline/Roo/Continue requests and verify that the client receives usable text/tool-call events.

## Acceptance criteria

- Normal chat continues to work for existing models.
- LFM2.5 native tool calls become valid OpenAI tool calls.
- A complete multi-turn tool conversation reaches a final answer.
- Tool-enabled streaming produces usable events before client timeout.
- Context overflow is rejected or reported truthfully.
- No server route contains LFM2.5-specific tags or parser logic.
- Adding a future model requires a profile/adapter and tests, not edits across the HTTP and agent loops.
- Internal Mivi broker execution remains separate from external OpenAI tool execution.

## Delivery order

Implement and verify one phase at a time:

1. Phase 0 reproduction fixtures.
2. Phase 1 canonical data model.
3. Phase 2 adapter interfaces and metadata.
4. Phase 3 LFM2.5 adapter.
5. Phase 4 OpenAI/Anthropic conversion.
6. Phase 5 streaming/cancellation.
7. Phase 6 context correctness.
8. Phase 7 future-model registry and documentation.

Only after the relevant targeted tests pass should we update the changelog/version and create a release commit.
