# Review Implementation Plan and TODO

**Scope:** Correctness, API-contract behavior, security, and resource usage identified during the project review.

## Completed in this review

- Default server binding is loopback-only; public binds require `MIVI_API_KEY`.
- Inference routes return an explicit service-unavailable error when no model is loaded. Readiness and model-discovery endpoints no longer advertise a fake model.
- OpenAI sampling options (`temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, presence/frequency penalties, `seed`, and `stop`) are validated and passed through request-scoped engine options. Existing sampler settings are restored after each request.
- OpenAI `response_format: {"type":"json_object"}` uses constrained JSON generation for non-streaming requests. Unsupported JSON Schema and JSON streaming are rejected explicitly.
- OpenAI tool choice is validated, named choices restrict the prompt/tool set, tool calls round-trip in blocking and buffered streaming responses, and assistant tool-call history is preserved in subsequent prompts.
- Anthropic sampling and `stop_sequences` now use the same validated generation path. Anthropic streaming emits structured `tool_use` blocks and reports `tool_use` instead of always claiming `end_turn`.
- Agent `allowed_tools` is enforced, context documents are workspace-confined and size-bounded, and timed-out tools fail closed because their side-effect status is unknown.
- Default CORS is disabled to avoid exposing an unauthenticated local API to arbitrary browser origins.
- Incremental context-position arithmetic is checked for overflow, and hybrid suffix snapshots are not restored without proof that their recurrent state is causally compatible.
- Mock inference is explicit (`EngineActor::spawn_mock()`) and is used only by shape/integration tests.

## Remaining TODO

### P0 — before exposing the server beyond a trusted local machine

- [x] Add an integration test that verifies API-key protection on both OpenAI and Anthropic routes, including streaming.
- [x] Add a configurable CORS allowlist for deployments that genuinely need browser clients; keep the default closed.
- [x] Replace the filesystem check-then-write flow with a Unix descriptor-relative, no-follow, atomic write strategy. Non-Unix builds retain the existing portable fallback and should run only with a trusted workspace owner.

### P1 — compatibility and correctness

- [x] Validate requested model IDs against the loaded model; retain `mivi` as the stable default alias.
- [x] Document `tool_choice: "required"` as unsupported until a real constrained tool-call decoder exists.
- [x] Reject JSON Schema response formats consistently and document the limitation in the API docs.
- [x] Add exact tokenizer-based usage accounting to Anthropic streaming.
- [x] Add route-level tests for invalid sampling, invalid stop sequences, JSON output mode, named tool choice, and Anthropic tool-use streaming.

### P2 — quality and observability

- [ ] Populate Ollama-compatible `size`, `digest`, parameter count, quantization, and family fields from GGUF metadata instead of placeholders.
- [ ] Remove or update documentation claims that are not backed by a reproducible benchmark/configuration.
- [ ] Add bounded request concurrency/backpressure around inference so many HTTP tasks cannot queue unbounded work behind the single engine actor.
- [ ] Add metrics for queue wait, generation latency, token counts, rejected requests, and tool timeouts.
- [ ] Run a dedicated formatting/lint cleanup on the touched modules, separating pre-existing formatting changes from functional changes.

## Low-resource verification policy

Use targeted commands only while iterating:

```bash
cargo test -p mivi-server --lib --jobs 2 -- --test-threads=2
cargo test -p mivi-agent --lib --jobs 2 -- --test-threads=2
cargo test -p mivi-model --lib checked_context_end --jobs 2 -- --test-threads=2
cargo test -p mivi --test integration_tests test_http_server_endpoints --jobs 2 -- --test-threads=2
```

Do not use workspace-wide `cargo check`, `cargo build`, or `cargo test` as part of routine review iterations.
