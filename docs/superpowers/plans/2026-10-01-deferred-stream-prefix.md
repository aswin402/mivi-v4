# Deferred Streaming Prefix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep first-output metrics/deadlines tied to decoded model output rather than an inserted prefix.

**Architecture:** A private callback factory holds the optional prefix until the first non-empty model chunk, then sends prefix and chunk together once. Keep the string stream API and existing HTTP/agent consumers. Generation conditioning and tool parsing remain unchanged.

**Tech Stack:** Rust, existing Tokio bounded channels and cancellation, Axum SSE, existing model streaming callbacks.

**Spec:** `docs/superpowers/specs/2026-10-01-deferred-stream-prefix-design.md` (approved).

## Global Constraints

- Cargo jobs=1, test threads=1, Rayon/inference threads=2; sequential Cargo/model loads. No full-workspace checks/builds/tests.
- Hold the optional prefix locally in the streaming engine actor. No prefix-only message may be sent before a real non-empty model chunk.
- Ignore empty decoded chunks for prefix emission and first-output purposes. Timing remains time to first non-empty decoded model output, not time to the first sampled token that might decode to an empty string.
- On the first non-empty chunk, prepend the pending prefix exactly once and send the combined string. Later chunks retain their content and order. With no prefix, non-empty model output is unchanged.
- Preserve the existing generation prompt, blocking generation behavior, tool-call syntax, generation budget, and string-based engine API. Do not introduce model names, token IDs, or delimiter-specific checks.
- Preserve cancellation checks before/after sending, receiver-close handling, bounded channel backpressure, and sampling checkpoint restoration.
- If cancellation, generation failure, or clean completion happens before a non-empty chunk, emit no synthetic prefix and record no first-output sample. Existing error propagation and required-tool validation remain in force.
- Keep existing timeout values; this is a correctness fix, not a speedup or a reason to increase agent timeouts.
- No public abstraction, new normal dependencies, unsafe code, profile changes, relaxed validation, or packed-inference rollout. Raw malformed-output capture is a separate follow-up.
- Observe behavioral assertion RED/GREEN against the actual production callback. Missing-symbol compilation errors do not count. Report preparatory characterization/extraction separately from the bug-fix chronology.
- Use `apply_patch`; preserve/exclude user `.gitignore`. No destructive cleanup, unrelated edits, raw-output logging, or secret exposure.
- All subagents, if used, must be GPT-6 Luna with high reasoning; no nested agents. Controller handles explicit staging and publication.
- After scoped/live verification and independent review, bump 0.2.60 to 0.2.61 once, update changelog with actual evidence and ideas/inspirations/source links, and push without force under standing authorization.

## Task 1: Production callback adapter and regressions

**Files:** Modify `crates/mivi-server/src/engine_actor.rs`, including its existing `tests` module. Do not change routes/model logic without a proven regression requiring review.

**Interfaces:** Consumes the existing bounded `mpsc::Sender<Result<String, String>>`, `GenerationCancellation`, and model callback. Produces private `model_stream_callback<'a>(responder: &'a mpsc::Sender<Result<String, String>>, cancellation: &'a GenerationCancellation, prefix: Option<&'a str>) -> Result<impl FnMut(u32, &str) -> bool + 'a, ()>`, used by the real `handle_generate_stream` path.

- [x] Run baseline commands separately; require zero failed tests and record output:

```bash
RAYON_NUM_THREADS=2 cargo test -p mivi-server --offline --lib --jobs 1 engine_actor::tests -- --test-threads=1
RAYON_NUM_THREADS=2 cargo test -p mivi-server --offline --lib --jobs 1 metrics_endpoint_records_streaming_first_token -- --test-threads=1
```

- [x] Characterize existing stream assembly/cancellation, then extract a behavior-preserving callback seam. This preparation must retain the old eager prefix behavior, not fix it yet:

```rust
fn model_stream_callback<'a>(
    responder: &'a mpsc::Sender<Result<String, String>>,
    cancellation: &'a GenerationCancellation,
    prefix: Option<&'a str>,
) -> Result<impl FnMut(u32, &str) -> bool + 'a, ()> {
    if let Some(prefix) = prefix.filter(|prefix| !prefix.is_empty()) {
        if cancellation.is_cancelled()
            || responder.blocking_send(Ok(prefix.to_string())).is_err()
        {
            return Err(());
        }
    }
    Ok(move |_, text| {
        !cancellation.is_cancelled()
            && responder.blocking_send(Ok(text.to_string())).is_ok()
            && !cancellation.is_cancelled()
    })
}
```

In `handle_generate_stream`, preserve `model_prompt`, JSON-stream rejection, generation arguments and error/checkpoint cleanup. Obtain the callback through this factory only for text generation. On factory `Err(())`, restore the checkpoint and return. Pass the returned callback directly to `generate_streaming_with_cancel`, retaining its cancellation closure. Re-run baseline tests; require GREEN for this preparatory refactor. Do not commit it as a completed fix.

- [x] Add the following regression; run the `deferred_prefix` filter and require an empty-channel assertion failure after compilation, not a missing-type error:

```rust
#[test]
fn deferred_prefix_waits_for_nonempty_model_output() {
    let (sender, mut receiver) = mpsc::channel(4);
    let cancellation = GenerationCancellation::new();
    let mut callback = model_stream_callback(&sender, &cancellation, Some("prefix:")).unwrap();
    assert!(matches!(receiver.try_recv(), Err(mpsc::error::TryRecvError::Empty)));
    assert!(callback(0, ""));
    assert!(matches!(receiver.try_recv(), Err(mpsc::error::TryRecvError::Empty)));
    assert!(callback(0, "first"));
    assert_eq!(receiver.try_recv().unwrap().unwrap(), "prefix:first");
    assert!(callback(0, "second"));
    assert_eq!(receiver.try_recv().unwrap().unwrap(), "second");
}
```

```bash
RAYON_NUM_THREADS=2 cargo test -p mivi-server --offline --lib --jobs 1 deferred_prefix -- --test-threads=1
```

- [x] Replace the eager factory with this deferred implementation, keeping the real actor wired to it. Re-run the regression and require GREEN:

```rust
fn model_stream_callback<'a>(
    responder: &'a mpsc::Sender<Result<String, String>>,
    cancellation: &'a GenerationCancellation,
    prefix: Option<&'a str>,
) -> Result<impl FnMut(u32, &str) -> bool + 'a, ()> {
    if cancellation.is_cancelled() || responder.is_closed() {
        return Err(());
    }
    let mut pending_prefix = prefix.filter(|prefix| !prefix.is_empty());
    Ok(move |_, text| {
        if cancellation.is_cancelled() || responder.is_closed() {
            return false;
        }
        if text.is_empty() {
            return true;
        }
        let output = match pending_prefix.take() {
            Some(prefix) => {
                let mut output = String::with_capacity(prefix.len() + text.len());
                output.push_str(prefix);
                output.push_str(text);
                output
            }
            None => text.to_string(),
        };
        !cancellation.is_cancelled()
            && responder.blocking_send(Ok(output)).is_ok()
            && !cancellation.is_cancelled()
    })
}
```

- [x] Extend tests using the same factory/channel pattern: `None` and `Some("")` send only model text; whitespace text with a prefix sends `"prefix: "`; dropping callback/sender without output closes the receiver without a prefix. Cancellation or closed receiver before construction yields `Err(())`; cancellation/closure after construction makes empty/nonempty callbacks return false without enqueueing a prefix. Generation failure before output is represented by dropping the callback then sending `Err("generation failed".to_string())`; the receiver must see that error first, then close, never a prefix. Assert exact items and termination, not merely successful execution.
- [x] Add capacity=1 backpressure coverage using an OS worker and synchronization channels: first combined item fills the queue, draining permits the unchanged second item, and the worker joins. Exercise cancellation after a blocked send and verify the callback stops. Do not call `blocking_send` inside a Tokio runtime. Use bounded outer watchdogs and release/join workers on all failure paths, not scheduling sleeps as proof.
- [x] Add an idle-callback deadline test: create the real factory on an OS worker, signal readiness through a oneshot, and hold model callbacks behind a release gate. A bounded Tokio timeout on `receiver.recv()` must expire before the gate releases real text. Then release text and assert combined delivery. This verifies the engine-stream invariant; actual HTTP metrics/deadlines are covered by Task 2. Always release/join the worker, including assertion-failure paths.
- [x] Run the following sequential scoped checks, require no failed tests/new warnings, and review checkpoint restoration plus unchanged chat/agent metric/deadline consumers:

```bash
RAYON_NUM_THREADS=2 cargo test -p mivi-server --offline --lib --jobs 1 engine_actor::tests -- --test-threads=1
RAYON_NUM_THREADS=2 cargo test -p mivi-server --offline --lib --jobs 1 streaming -- --test-threads=1
RAYON_NUM_THREADS=2 cargo test -p mivi-server --offline --lib --jobs 1 tool -- --test-threads=1
RAYON_NUM_THREADS=2 cargo clippy -p mivi-server --offline --lib --tests --jobs 1 -- -D warnings -A clippy::manual_div_ceil -A clippy::manual_is_multiple_of -A clippy::items_after_test_module -A clippy::field_reassign_with_default
cargo fmt -p mivi-server -- --check
git diff --check
```

- [x] Request independent read-only spec/quality review of real production wiring, idle/error/cancellation/closure behavior, backpressure, ordering and checkpoint cleanup. Resolve findings with regressions/re-review. Commit only reviewed Task 1 files, excluding `.gitignore`.

## Task 2: Live verification and reviewed patch release

**Files:** `Cargo.toml`, `Cargo.lock`, `CHANGELOG.md`, spec status, this plan's checkboxes. Temporary fixtures/probes outside repository, edited via `apply_patch`.

**Interfaces:** Consumes Task 1 through ordinary production CLI/server and existing HTTP metrics/timeouts. Produces measured deadline/metric evidence and reviewed v0.2.61 publication; malformed-call diagnosis remains separate.

- [x] Run `RAYON_NUM_THREADS=2 cargo test -p mivi-server --offline --lib --jobs 1 -- --test-threads=1`; require zero failures. No numerical model/quant corpus reruns for this stream-assembly change.
- [x] Build only the executable package required for live checks, still at 0.2.60: `RAYON_NUM_THREADS=2 cargo build -p mivi --offline --release --jobs 1`. Check `target/release/mivi --version`; record that this binary includes the fix but precedes the patch metadata bump. The supporting `mivi-cli` library was also built once; it does not produce `target/release/mivi`.
- [x] Create an isolated workspace with `mktemp -d /tmp/mivi-prefix-verification.XXXXXX`; retain its exact returned path. Add `example.rs` containing `pub fn add(a: i32, b: i32) -> i32 { a + b }`. Prepare the known successful short tool payload from the preceding smoke test and a medium variant with 64 numbered background records. Required fields: model alias `mivi`, `read_file` function schema requiring string `path`, `tool_choice:"required"`, `temperature:0`, `max_tokens:48`, `stream:true`. Do not execute arbitrary tool expressions, expose project files, or write through native tools.
- [x] Start one fresh loopback server with the actual temporary path: `env -u MIVI_API_KEY MIVI_THREADS=2 RAYON_NUM_THREADS=2 target/release/mivi serve --model models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf --host 127.0.0.1 --port 18560 --workspace <returned-path> --ctx-size 4096 --max-concurrent-tool-executions 1 --request-timeout-secs 30 --first-token-timeout-secs 1`. Substitute the resolved workspace explicitly; do not use an unresolved destructive target.
- [x] Send the medium required-tool SSE request with a bounded client deadline. Require an explicit first-token deadline error during prefill, orderly stream termination, no successful tool call, zero new first-output metric samples, and eventual inference-slot release through cooperative cancellation. Headers, role events and keep-alives do not count as output. If this hardware finishes prefill within one second, increase the fixture within 4,096 context tokens or report the test unexercised; do not fabricate a pass. Stop only the owned server after resolving its exact PID.
- [x] Restart sequentially with request timeout=120 and first-output timeout=90. Send the short required-tool request, reconstruct SSE call fragments by index, and validate `read_file(path="example.rs")`. Return actual file contents with the same `tool_call_id` and `tool_choice:"none"`; require an answer identifying `add` and `a + b`. Record metric deltas, first visible tool/content progress, total timings, and server/model configuration. The metric can precede decoded tool-name emission, but must represent actual decoded model output rather than a pre-inference synthetic prefix. Stop this server too.
- [x] If tool correctness fails, record/capture evidence separately before attributing it to this adapter or modifying parsers/profiles. A new adapter regression blocks release and needs a scoped fix/re-review; do not hide a failed live check. Keep pre-existing malformed-output failures and general agent latency limitations explicit.
- [x] After successful scoped/live verification and review, bump workspace version once from 0.2.60 to 0.2.61. Update changelog with root cause, actual tests/live outcomes, resources, limitations, and pinned actor/chat/model source links from the spec. Refresh lockfile workspace package versions with the next scoped command; normal dependencies must not change. Controller directed this metadata step after the live evidence and before its independent Task 2 review; that review remains pending.
- [x] Re-run final scoped tests/Clippy/format/normal dependency tree/whitespace checks sequentially. Rebuild only package `mivi` once more to deliver an honestly labelled 0.2.61 executable; this metadata rebuild is needed for the released binary, not a full-workspace build. Numerical/stream behavior was already live-verified; re-run live checks only if logic changes during review.

```bash
RAYON_NUM_THREADS=2 cargo test -p mivi-server --offline --lib --jobs 1 -- --test-threads=1
RAYON_NUM_THREADS=2 cargo clippy -p mivi-server --offline --lib --tests --jobs 1 -- -D warnings -A clippy::manual_div_ceil -A clippy::manual_is_multiple_of -A clippy::items_after_test_module -A clippy::field_reassign_with_default
cargo fmt -p mivi-server -- --check
cargo tree -p mivi-server --offline -e normal --depth 1
RAYON_NUM_THREADS=2 cargo build -p mivi --offline --release --jobs 1
target/release/mivi --version
git diff --check
```

- [ ] Obtain final whole-change review using Task 1's behavioral RED/GREEN report and actual live evidence; fix Important findings with regressions and re-review. Update completion docs only after their checks pass. Commit only task files and push the authorized branch without force. For main, compare `git rev-parse HEAD` with `git ls-remote origin refs/heads/main`. Preserve/exclude `.gitignore`. Report source/binary version, corrected metric/deadline behavior, and unresolved malformed-call/agent-quality work.

## Plan self-review

- Prefix timing, empties, ordering, cancellation, receiver closure, errors, backpressure, checkpoint cleanup: Task 1.
- Real first-output deadline/metric behavior and tool compatibility: Task 1's invariant plus Task 2's live checks and existing scoped server tests.
- No conditioning/blocking/protocol/numerical policy change: both tasks' constraints and reviews.
- Version, binary, evidence/changelog/sources, resources and publication: Task 2.
- Raw malformed-output capture remains a separate diagnostic follow-up; no raw-output logging feature is included.

Execution status: Task 1 complete after behavioral RED/GREEN, scoped verification, and two-wave independent spec/quality review with no findings. Task 2 scoped/live verification, v0.2.61 metadata/changelog, final scoped checks, and the fresh root executable are complete. Independent Task 2 re-review approved after evidence-ledger corrections; final whole-change review and publication remain pending. Second-request acceptance demonstrates API inference admission-permit reuse, not physical prefill completion before admission. User `.gitignore` preserved.
