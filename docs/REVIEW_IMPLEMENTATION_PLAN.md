# Review Implementation Plan and TODO

**Scope:** Correctness, API-contract behavior, security, and resource usage identified during the project review.

## Completed in this review

- Default server binding is loopback-only; public binds require `MIVI_API_KEY`.
- Inference routes return an explicit service-unavailable error when no model is loaded. Readiness and model-discovery endpoints no longer advertise a fake model.
- OpenAI sampling options (`temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, presence/frequency penalties, `seed`, and `stop`) are validated and passed through request-scoped engine options. Existing sampler settings are restored after each request, while the RNG advances for unseeded requests and is restored for explicitly seeded requests.
- OpenAI `response_format: {"type":"json_object"}` uses constrained JSON generation for non-streaming requests. Unsupported JSON Schema and JSON streaming are rejected explicitly.
- OpenAI tool choice is validated, named choices restrict the prompt/tool set, tool calls round-trip in blocking and buffered streaming responses, and assistant tool-call history is preserved in subsequent prompts.
- Anthropic sampling and `stop_sequences` now use the same validated generation path. Anthropic streaming emits structured `tool_use` blocks and reports `tool_use` instead of always claiming `end_turn`.
- Anthropic streaming usage reports tokenizer counts exactly, including zero-token output.
- Agent `allowed_tools` is enforced, context documents are workspace-confined and size-bounded, and timed-out tools fail closed because their side-effect status is unknown.
- Agent tool batches stop after the first timeout so later calls cannot add side effects to an already ambiguous run.
- Timed-out agent tools now receive a cooperative cancellation signal when registered as cancellable handlers; legacy handlers remain supported but cannot observe cancellation.
- Blocking tool handlers are bounded by a broker-level execution semaphore; a handler that has already started may finish after timeout, but cannot create unbounded active blocking work.
- When API-key authentication is configured, all API/model/control routes require it; health and the embedded UI remain public for operational access.
- Default CORS is disabled to avoid exposing an unauthenticated local API to arbitrary browser origins.
- Incremental context-position arithmetic is checked for overflow, and hybrid suffix snapshots are not restored without proof that their recurrent state is causally compatible.
- Mock inference is explicit (`EngineActor::spawn_mock()`) and is used only by shape/integration tests.
- Anthropic streaming usage is encoded before generation starts, avoiding bounded-buffer deadlocks on long streams.
- OpenAI and Anthropic streaming failures now prevent normal completion finalization (stop/end_turn) from being emitted after an error.
- Inference routes now hold a configurable semaphore permit for the full request/stream/agent lifetime and reject excess work with HTTP 429.
- Blocking tool execution is bounded by a configurable broker semaphore (`--max-concurrent-tool-executions`, default 4); zero is clamped to one.
- Zero-valued channel and concurrency settings are normalized before Tokio channel/semaphore creation, preventing configuration panics.
- Stop-prefix matching is UTF-8 boundary-safe; zero-temperature sampling still applies penalties; and top-p expands its candidate window until the requested nucleus mass is covered.
- JSON primitive parsing rejects incomplete literals, malformed numbers, and trailing commas instead of relying only on final JSON validation.
- On-disk KV-cache length encoding and payload-size arithmetic are checked before writing or allocating.
- Generated OpenAI and Anthropic tool calls are validated against caller-declared JSON Schemas before being exposed as tool-use responses; malformed declarations are rejected at request validation.
- Unix workspace reads and directory listings now traverse pinned directory descriptors and use no-follow operations; non-Unix builds retain the portable fallback.
- Agent context documents reuse the bounded no-follow workspace reader, so validation and prompt reads cannot be redirected by a swapped symlink on Unix.

## Remaining TODO

### P0 — before exposing the server beyond a trusted local machine

- [x] Add an integration test that verifies API-key protection on both OpenAI and Anthropic routes, including streaming.
- [x] Apply API-key protection consistently to API metadata, status, tools, metrics, and Ollama compatibility routes while retaining public health/UI endpoints.
- [x] Add a configurable CORS allowlist for deployments that genuinely need browser clients; keep the default closed.
- [x] Replace the filesystem check-then-write flow with a Unix descriptor-relative, no-follow, atomic write strategy. Non-Unix builds retain the existing portable fallback and should run only with a trusted workspace owner.
- [x] Route agent context-document reads through the Unix descriptor-relative, no-follow reader and preserve bounded portable behavior on non-Unix builds.

### P1 — compatibility and correctness

- [x] Validate requested model IDs against the loaded model; retain `mivi` as the stable default alias.
- [x] Document `tool_choice: "required"` as unsupported until a real constrained tool-call decoder exists.
- [x] Reject JSON Schema response formats consistently and document the limitation in the API docs.
- [x] Add exact tokenizer-based usage accounting to Anthropic streaming.
- [x] Add route-level tests for invalid sampling, invalid stop sequences, JSON output mode, named tool choice, and Anthropic tool-use streaming.

### P2 — quality and observability

- [x] Populate Ollama-compatible `size`, parameter count, quantization, and family fields from the loaded GGUF; omit digest when no digest is available rather than returning a placeholder.
- [x] Remove or update current documentation claims that are not backed by a reproducible benchmark/configuration; historical proposals are explicitly labeled.
- [x] Add bounded request concurrency/backpressure around inference so many HTTP tasks cannot queue unbounded work behind the single engine actor.
- [x] Bound concurrent blocking tool handlers and expose the limit through server configuration/CLI so timed-out started handlers retain a slot until completion.
- [x] Add process-local metrics for slot wait, generation latency, token counts, rejected requests, inference errors, and tool timeouts.
- [x] Run a dedicated formatting/lint cleanup on the touched Rust modules; unrelated untouched workspace formatting remains out of scope.

## Low-resource verification policy

Use targeted commands only while iterating:

```bash
cargo test -p mivi-server --lib --jobs 2 -- --test-threads=2
cargo test -p mivi-agent --lib --jobs 2 -- --test-threads=2
cargo test -p mivi-model --lib checked_context_end --jobs 2 -- --test-threads=2
cargo test -p mivi --test integration_tests test_http_server_endpoints --jobs 2 -- --test-threads=2
```

Do not use workspace-wide `cargo check`, `cargo build`, or `cargo test` as part of routine review iterations.
