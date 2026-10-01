# Deferred Streaming Prefix and First-Output Deadline

Status: written spec approved; implementation plan prepared, execution pending.

## Problem and evidence

The engine currently sends a forced output prefix before starting model
generation. The chat route counts every non-empty engine string as first
output, records its latency, and disables the first-token deadline.
Required-tool requests therefore report roughly 1 ms even when prefill and
generation take tens of seconds. The same premature string can affect other
consumers of the engine's streaming interface.

The bounded local 1.2B test used two inference/Rayon threads. A 1,134-token
request delivered its first visible tool delta at 31.16 seconds while the
first-token metric reported 1.12 ms. A 2,670-token request ended with an error
at 76.71 seconds without a usable tool delta, yet the metric reported 1.15 ms.
These were synthetic, prefix-sharing requests, not cold-cache benchmarks.
Repeated requests were faster but retained their tool-format failures.

## Approved approach and alternatives

Delay the synthetic prefix until the first non-empty decoded model chunk and
combine both into one engine message. Keep the existing string-based stream
contract. The prefix remains appended to the model's input prompt, so this
fix changes emission timing, not generation conditioning or tool syntax.

A typed prefix/token event protocol would also distinguish synthetic output,
but changes every consumer and is unnecessary for this bounded fix. Merely
ignoring a matching prefix in HTTP routes would duplicate policy, depend on
text comparisons, and leave other consumers vulnerable.

## Behavior and scope

- Hold the optional prefix locally in the streaming engine actor. No
  prefix-only message may be sent before a real non-empty model chunk.
- Ignore empty decoded chunks for prefix emission and first-output purposes.
  Timing remains time to first non-empty decoded model output, not time to
  the first sampled token that might decode to an empty string.
- On the first non-empty chunk, prepend the pending prefix exactly once and
  send the combined string. Later chunks retain their content and order.
  With no prefix, non-empty model output is unchanged.
- Preserve the existing generation prompt, blocking generation behavior,
  tool-call syntax, generation budget, and string-based engine API. Do not
  introduce model names, token IDs, or delimiter-specific checks.
- Preserve cancellation checks before/after sending, receiver-close handling,
  bounded channel backpressure, and sampling checkpoint restoration. A
  cancelled/failed send must stop generation rather than leave a pending
  synthetic message for later delivery.
- If cancellation, generation failure, or clean completion happens before a
  non-empty chunk, emit no synthetic prefix and record no first-output sample.
  Existing error propagation and required-tool validation remain in force.
- First-output deadlines in all streaming consumers remain active throughout
  prefill and until a non-empty model chunk is received. Initial HTTP headers,
  role events, and SSE keep-alives are not model output.
- Keep existing timeout values; this is a correctness fix, not a speedup or
  a reason to increase agent timeouts.

Prefer a small private callback adapter in `engine_actor.rs`, used by the
real streaming generation path and directly testable without loading a model.
Do not add a public abstraction, normal dependency, or unsafe code.

Malformed generated tool calls are a separate investigation. This fix must
not relax schema validation, guess missing arguments, change model profiles,
enable experimental packed inference, or claim agent correctness.

## Verification and release

Write behavioral regressions first and observe assertion failures before the
implementation. Cover no delivery before the first non-empty model callback,
empty callbacks, once-only prefix emission and ordering, prefix-free streams,
cancellation/closed receivers, and completion/error before real output.
Exercise the production callback adapter, not an unrelated duplicate helper.

Add bounded deadline/metric coverage showing a forced prefix does not satisfy
the first-output deadline, while genuine output does. Existing streaming/tool
conformance behavior must remain intact. Use deterministic synchronization or
controlled time where available rather than timing-sensitive short sleeps.

Run only the affected server tests/checks, with Cargo jobs=1, test threads=1,
and Rayon threads=2. No full-workspace commands or concurrent model loads.
Rebuild only the CLI/server binary needed for live verification, also jobs=1.
Use one isolated loopback server, two inference threads, and a temporary
workspace. Verify a deliberately short first-output deadline fires during
prefill on a required-tool request; separately verify a normal-budget short
tool-call round trip remains functional and its metric is no longer synthetic.
Distinguish raw model-output latency from client-visible decoded tool progress.

After implementation, independent review, and successful scoped/live checks,
update the changelog with actual evidence and source/inspiration links, bump
workspace version once from 0.2.60 to 0.2.61, and publish without force under
the user's standing authorization. Do not claim malformed tool output or
agent-sized latency is fixed. Preserve and exclude the user's `.gitignore`.
All subagents, if used, must be GPT-6 Luna with high reasoning.

## Ideas, inspirations, and sources

- Distinguish server-inserted protocol bytes from model-produced output; defer
  the former rather than treating them as evidence of completed inference.
- The project's existing forced-prefix logic and cancellation-aware model
  callbacks are the implementation reference:
  [engine actor](https://github.com/aswin402/mivi-v4/blob/a64f7e1/crates/mivi-server/src/engine_actor.rs),
  [chat deadlines and metrics](https://github.com/aswin402/mivi-v4/blob/a64f7e1/crates/mivi-server/src/routes/chat.rs),
  [model streaming callbacks](https://github.com/aswin402/mivi-v4/blob/a64f7e1/crates/mivi-model/src/model.rs).
- The local long-context reproduction above motivates the fix; it does not
  attribute tool-format failures to a particular model or parser component.
