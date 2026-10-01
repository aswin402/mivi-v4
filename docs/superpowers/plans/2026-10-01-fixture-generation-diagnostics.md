# Fixture-Only Generation Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce private, bounded evidence from isolated streaming tool fixtures without changing generation or enabling capture of real agent traffic.

**Architecture:** Feature-gated model observations collect bounded text/token/timing data at the existing generation boundaries. A crate-private server fixture session correlates actor snapshots with the ordinary router's SSE result. An ignored in-process fixture test owns model/workspace/worker lifetime and writes artifacts only after physical generation finishes.

**Tech Stack:** Rust, existing Serde/JSON, Tokio/Axum/Tower, standard-library filesystem and monotonic clocks; no new normal dependencies.

**Approved spec:** `docs/superpowers/specs/2026-10-01-fixture-generation-diagnostics-design.md`.

## Global Constraints

- The first increment covers streaming chat completions only; native agent, Anthropic, blocking completion, and general CLI diagnostic entry points are not added.
- Use a non-default `fixture-diagnostics` Cargo feature, forwarded from `mivi-server` to `mivi-model` only for explicit diagnostic builds.
- Capture requires both the feature and an explicitly installed per-run diagnostic session from that fixture harness.
- No capture activation through normal CLI flags, request fields/headers, environment toggles, or global loggers. A test-model path supplied to the ignored harness is configuration, not a capture toggle.
- Existing public generation methods and string-stream consumers remain source-compatible. No normal dependency or unsafe code is required.
- Keep rendering, sampling, prefill, decoding, stop handling, profiles, parser validation, checkpoint restoration, cancellation, and backpressure unchanged.
- Raw decoded text is captured before text-stop trimming; termination token IDs are excluded just as in the existing generated-ID sequence. Synthetic prefixes are separate data, never model output.
- All captured strings/IDs/SSE data and record counts are bounded; mark truncation, overflow, capture failure, and persistence failure explicitly. No per-token I/O or blocking diagnostic sends.
- Write only private local artifacts in an exclusively created temporary directory; Unix directory/file modes 0700/0600. Never overwrite, follow user-provided symlinks, print raw evidence, or commit/upload artifacts.
- No listener or arbitrary live requests; the harness drives the actual router in-process with synthetic files only. No project-workspace access or mutating native tools.
- Finalize engine evidence at physical generation return, not API timeout. A second admission proves permit reuse, not completed prefill.
- Cargo jobs=1, test threads=1, Rayon/inference threads=2, with sequential Cargo commands/model loads. No full-workspace check/build/test.
- Use `apply_patch`; preserve/exclude user `.gitignore`. Controller handles explicit staging/commits/publication. Any subagents must be GPT-6 Luna, high reasoning, without nested agents.
- Observe compiled behavioral RED/GREEN; missing types, compile errors, and retrospective mutations are not the original behavioral RED.
- After implementation, scoped/live verification, and independent review, update the changelog with measured evidence, limitations, ideas/inspirations/sources; bump v0.2.61 to v0.2.62 once and publish without force under the user's standing authorization.

## File and interface map

| File | Responsibility |
|---|---|
| `crates/mivi-model/Cargo.toml`, `src/lib.rs` | Non-default feature and conditional diagnostics module |
| `crates/mivi-model/src/fixture_diagnostics.rs` | Bounded text/ID collectors, lifecycle and timing observations, snapshot types |
| `crates/mivi-model/src/model.rs` | Small feature-gated hooks in the existing streaming path |
| `crates/mivi-server/Cargo.toml`, `src/lib.rs` | Feature forwarding and private diagnostics module |
| `crates/mivi-server/src/fixture_diagnostics.rs` | Session records, bounded metadata, per-request completion coordination |
| `crates/mivi-server/src/fixture_diagnostics/artifacts.rs` | Exclusive private artifact creation and persistence |
| `crates/mivi-server/src/engine_actor.rs` | Private fixture spawn/join ownership, capture start/finalize wiring |
| `crates/mivi-server/src/routes/chat.rs` | Fixture-only suppression of existing summaries/error payload logs |
| `crates/mivi-server/src/fixture_diagnostics/fixtures.rs` | Ignored real-router fixture, bounded SSE reconstruction, byte-exact handoff |
| `Cargo.toml`, `Cargo.lock`, `CHANGELOG.md`, spec/plan | Reviewed release metadata and actual outcomes |

No production CLI/config/request-schema changes. Model methods/types below exist
only under `fixture-diagnostics`; server session/spawn APIs are `pub(crate)`.

## Task 1: Bounded model collector and lifecycle contract

**Files:** model feature manifest, `src/lib.rs`, new `src/fixture_diagnostics.rs`.

**Interfaces:** Produce `CaptureLimits`, `CapturedText`, `CapturedIds`, `ModelOutcome`,
`StageOutcome`, `ModelCapture`, and `ModelRecorder`. These are feature-only model
module types shared with Task 2 and the server. All snapshot fields derive
Serialize, with no unbounded string errors or filesystem paths.

- [x] Run the default scoped model baseline, separately from every later Cargo command:

```bash
RAYON_NUM_THREADS=2 cargo test -p mivi-model --offline --lib --jobs 1 -- --test-threads=1
```

Require zero failures; record ignored real-model tests and warnings honestly.

- [x] Add feature/module scaffolding and compile a behavior-preserving collector skeleton. This is explicitly preparatory work, not the behavioral fix. Define the contracts below; implement `push` as a no-op only for the initial compiled RED:

```toml
# mivi-model/Cargo.toml
[features]
fixture-diagnostics = []
```

```rust
// mivi-model/src/lib.rs
#[cfg(feature = "fixture-diagnostics")]
pub mod fixture_diagnostics;
```

```rust
use serde::Serialize;
use std::time::{Duration, Instant};

pub const MAX_TEXT_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_CAPTURE_IDS: usize = 65_536;

#[derive(Clone, Copy, Debug)]
pub struct CaptureLimits { pub text_bytes: usize, pub token_ids: usize }
impl CaptureLimits {
    pub fn validate(self) -> Result<Self, &'static str> {
        if self.text_bytes == 0 || self.text_bytes > MAX_TEXT_BYTES
            || self.token_ids == 0 || self.token_ids > MAX_CAPTURE_IDS {
            return Err("invalid fixture capture bounds");
        }
        self.text_bytes.checked_mul(4)
            .and_then(|n| self.token_ids.checked_mul(4).and_then(|ids| n.checked_add(ids)))
            .ok_or("fixture capture budget overflow")?;
        Ok(self)
    }
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct CapturedText {
    pub text: String, pub observed_bytes: usize,
    pub truncated: bool, pub counter_overflow: bool,
    #[serde(skip)] cap: usize,
}
impl CapturedText {
    pub fn new(cap: usize) -> Self {
        Self { text: String::with_capacity(cap), observed_bytes: 0,
            truncated: false, counter_overflow: false, cap }
    }
    // Implement after observing the compiled failing test below.
    pub fn push(&mut self, _chunk: &str) {}
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct CapturedIds {
    pub ids: Vec<u32>, pub observed_tokens: usize,
    pub truncated: bool, pub counter_overflow: bool,
    #[serde(skip)] cap: usize,
}
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
pub enum StageOutcome { Complete, Cancelled, ModelError }
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
pub enum ModelOutcome { Complete, Cancelled, DeliveryStopped, ModelError }
#[derive(Clone, Debug, Serialize)]
pub struct ModelCapture {
    pub raw_decoded: CapturedText, pub delivered: CapturedText,
    pub generated_ids: CapturedIds,
    pub tokenization: Option<Duration>, pub prefill: Option<Duration>,
    pub prefill_outcome: Option<StageOutcome>, pub decode: Option<Duration>,
    pub first_raw: Option<Duration>, pub first_delivered: Option<Duration>,
    pub prompt_tokens: Option<usize>, pub reused_tokens: Option<usize>,
    pub processed_tokens: Option<usize>, pub outcome: Option<ModelOutcome>,
    pub progress_counter_overflow: bool,
}
pub struct ModelRecorder {
    pub snapshot: ModelCapture,
    entry: Option<Instant>, tokenization_start: Option<Instant>,
    prefill_start: Option<Instant>, decode_start: Option<Instant>,
}
```

- [x] Add and run this compiled behavioral RED. It must fail on the retained-text assertion, not compilation:

```rust
#[test]
fn fixture_capture_preserves_newline_and_utf8_prefix() {
    let mut text = CapturedText::new(5);
    text.push("a\n");
    text.push("éé");
    text.push("tail");
    assert_eq!(text.text, "a\né");
    assert_eq!(text.observed_bytes, 10);
    assert!(text.truncated);
    assert!(!text.counter_overflow);
}
```

```bash
RAYON_NUM_THREADS=2 cargo test -p mivi-model --offline --lib --features fixture-diagnostics --jobs 1 fixture_capture_preserves -- --test-threads=1
```

- [x] Implement retention as a prefix of the complete observed stream. Once truncated, later smaller chunks must not be appended into the remaining capacity:

```rust
pub fn push(&mut self, chunk: &str) {
    match self.observed_bytes.checked_add(chunk.len()) {
        Some(n) => self.observed_bytes = n,
        None => { self.observed_bytes = usize::MAX; self.counter_overflow = true; }
    }
    if self.truncated { return; }
    let available = self.cap.saturating_sub(self.text.len());
    let mut keep = available.min(chunk.len());
    while !chunk.is_char_boundary(keep) { keep -= 1; }
    self.text.push_str(&chunk[..keep]);
    self.truncated = keep < chunk.len();
}
```

`CapturedIds::new(cap)` preallocates its vector; `push(id)` checked-increments
observed count, appends only while under cap, and marks truncation otherwise.
Use this exact implementation pattern for IDs:

```rust
impl CapturedIds {
    pub fn new(cap: usize) -> Self {
        Self { ids: Vec::with_capacity(cap), observed_tokens: 0,
            truncated: false, counter_overflow: false, cap }
    }
    pub fn push(&mut self, id: u32) {
        match self.observed_tokens.checked_add(1) {
            Some(n) => self.observed_tokens = n,
            None => { self.observed_tokens = usize::MAX; self.counter_overflow = true; }
        }
        if self.ids.len() < self.cap { self.ids.push(id); }
        else { self.truncated = true; }
    }
}
```

- [x] Add the validated recorder constructor and methods below. Each timing method accepts a supplied `Instant` so unit tests use checked synthetic time, not sleeps:

```rust
impl ModelRecorder {
pub fn new(limits: CaptureLimits) -> Result<Self, &'static str> {
    let limits = limits.validate()?;
    Ok(Self {
        snapshot: ModelCapture {
            raw_decoded: CapturedText::new(limits.text_bytes),
            delivered: CapturedText::new(limits.text_bytes),
            generated_ids: CapturedIds::new(limits.token_ids),
            tokenization: None, prefill: None, prefill_outcome: None,
            decode: None, first_raw: None, first_delivered: None,
            prompt_tokens: None, reused_tokens: None, processed_tokens: None,
            outcome: None, progress_counter_overflow: false,
        },
        entry: None, tokenization_start: None, prefill_start: None, decode_start: None,
    })
}
// Methods on ModelRecorder; private timestamps do not serialize.
pub fn begin(&mut self, now: Instant) {
    self.entry = Some(now); self.tokenization_start = Some(now);
}
pub fn tokenization_done(&mut self, now: Instant) {
    self.snapshot.tokenization = self.tokenization_start
        .map(|start| now.saturating_duration_since(start));
}
pub fn prefill_begin(&mut self, now: Instant, tokens: usize) {
    self.prefill_start = Some(now); self.snapshot.prompt_tokens = Some(tokens);
    self.snapshot.processed_tokens = Some(0);
}
pub fn processed(&mut self, count: usize) {
    let previous = self.snapshot.processed_tokens.unwrap_or(0);
    self.snapshot.processed_tokens = Some(match previous.checked_add(count) {
        Some(total) => total,
        None => { self.snapshot.progress_counter_overflow = true; usize::MAX }
    });
}
pub fn prefill_end(&mut self, now: Instant, reused: usize,
    processed: usize, outcome: StageOutcome) {
    self.snapshot.prefill = self.prefill_start
        .map(|start| now.saturating_duration_since(start));
    self.snapshot.reused_tokens = Some(reused);
    self.snapshot.processed_tokens = Some(processed);
    self.snapshot.prefill_outcome = Some(outcome);
    if outcome == StageOutcome::Complete { self.decode_start = Some(now); }
}
pub fn raw(&mut self, now: Instant, text: &str) {
    if !text.is_empty() && self.snapshot.first_raw.is_none() {
        self.snapshot.first_raw = self.entry.map(|start| now.saturating_duration_since(start));
    }
    self.snapshot.raw_decoded.push(text);
}
pub fn delivered(&mut self, now: Instant, text: &str) {
    if !text.is_empty() && self.snapshot.first_delivered.is_none() {
        self.snapshot.first_delivered = self.entry.map(|start| now.saturating_duration_since(start));
    }
    self.snapshot.delivered.push(text);
}
pub fn finish(&mut self, now: Instant, outcome: ModelOutcome) {
    self.snapshot.decode = self.decode_start.map(|start| now.saturating_duration_since(start));
    self.snapshot.outcome = Some(outcome);
}
}
```

- [x] Add direct tests: exact newline/whitespace preservation; truncation at
each boundary of `é`; no appends after a truncated multibyte chunk; IDs in
order/capped; checked-counter overflow; zero/over-limit configs rejected;
empty raw/delivered events leave first times absent; explicit instants produce
tokenization=2ms, prefill=3ms, first raw=6ms, first delivered=8ms, decode=5ms;
cancelled/error prefill leaves decode absent. Construct the timing fixture with:

```rust
let start = Instant::now();
let at = |ms| start.checked_add(Duration::from_millis(ms)).unwrap();
let mut recorder = ModelRecorder::new(CaptureLimits { text_bytes: 64, token_ids: 8 }).unwrap();
recorder.begin(at(0)); recorder.tokenization_done(at(2));
recorder.prefill_begin(at(2), 7);
recorder.prefill_end(at(5), 2, 5, StageOutcome::Complete);
recorder.raw(at(6), "x"); recorder.delivered(at(8), "x");
recorder.finish(at(10), ModelOutcome::Complete);
assert_eq!(recorder.snapshot.first_raw, Some(Duration::from_millis(6)));
assert_eq!(recorder.snapshot.first_delivered, Some(Duration::from_millis(8)));
assert_eq!(recorder.snapshot.decode, Some(Duration::from_millis(5)));
```

- [x] Require collector GREEN and format; request independent task review. Commit only Task 1 files after findings are resolved:

```bash
RAYON_NUM_THREADS=2 cargo test -p mivi-model --offline --lib --features fixture-diagnostics --jobs 1 fixture_diagnostics -- --test-threads=1
cargo fmt -p mivi-model -- --check
git diff --check
git add crates/mivi-model/Cargo.toml crates/mivi-model/src/lib.rs crates/mivi-model/src/fixture_diagnostics.rs
git commit -m "feat: add bounded fixture diagnostic collectors"
```

## Task 2: Observe the real model path without changing inference

**Files:** `crates/mivi-model/src/model.rs`, collector lifecycle tests.

**Interfaces:** Consume Task 1 types. Feature-only methods on `Model`:
`start_fixture_capture(CaptureLimits) -> Result<(), &'static str>` and
`take_fixture_capture() -> Option<ModelCapture>`. The observer is request-local;
no global/thread-local sink. Existing generate signatures remain unchanged.

- [x] Add a feature-only `fixture_recorder: Option<ModelRecorder>` field,
initialize it to `None` in the model loader, and implement activation/take:

```rust
// Conditional import, Model field, and matching loader initializer:
#[cfg(feature = "fixture-diagnostics")]
use crate::fixture_diagnostics::ModelRecorder;
// In Model:
#[cfg(feature = "fixture-diagnostics")]
fixture_recorder: Option<ModelRecorder>,
// In its existing loader initializer:
#[cfg(feature = "fixture-diagnostics")]
fixture_recorder: None,
```

```rust
#[cfg(feature = "fixture-diagnostics")]
pub fn start_fixture_capture(&mut self,
    limits: crate::fixture_diagnostics::CaptureLimits) -> std::result::Result<(), &'static str> {
    if self.fixture_recorder.is_some() { return Err("fixture capture already active"); }
    self.fixture_recorder = Some(crate::fixture_diagnostics::ModelRecorder::new(limits)?);
    Ok(())
}
#[cfg(feature = "fixture-diagnostics")]
pub fn take_fixture_capture(&mut self) -> Option<crate::fixture_diagnostics::ModelCapture> {
    self.fixture_recorder.take().map(|recorder| recorder.snapshot)
}
```

- [x] Characterize a real-model ignored test with explicit model path and fixed
seed. Run unchanged generation with the observer armed but not yet wired;
assert capture has nonempty raw data whenever the baseline delivered nonempty
text. This must compile and fail on missing observed data; if the selected
fixture generates nothing, report it unexercised and choose a bounded nonempty
fixture, never count an empty run as RED. Invoke only that ignored test with
`MIVI_TEST_MODEL` supplied, `MIVI_THREADS=2`, and the Task 1 scoped command pattern.
Use this bounded test body in the existing model test module:

```rust
#[test]
#[ignore = "requires explicit MIVI_TEST_MODEL; observes real streaming hooks"]
#[cfg(feature = "fixture-diagnostics")]
fn fixture_model_observation() -> std::result::Result<(), Box<dyn std::error::Error>> {
    use crate::fixture_diagnostics::{CaptureLimits, ModelOutcome};
    let path = std::env::var("MIVI_TEST_MODEL")?;
    let mut model = Model::load_with_ctx(std::path::Path::new(&path), Some(512))?;
    model.sampler.config.temperature = 0.0;
    model.sampler.set_seed(7);
    model.start_fixture_capture(CaptureLimits { text_bytes: 4096, token_ids: 32 })?;
    let mut delivered = String::new();
    model.generate_streaming("Return the word hello.", 16, |_, text| {
        delivered.push_str(text); true
    })?;
    assert!(!delivered.is_empty(), "observation fixture did not exercise decoded output");
    let capture = model.take_fixture_capture().ok_or("missing fixture capture")?;
    assert!(!capture.raw_decoded.text.is_empty(), "raw observation missing");
    assert!(capture.delivered.text == delivered, "delivered observation mismatch");
    assert_eq!(capture.outcome, Some(ModelOutcome::Complete));
    assert!(model.take_fixture_capture().is_none());
    Ok(())
}
```

If the error type conversion rejects a static-string collector error, map it
to `std::io::Error::new(InvalidInput, message)` rather than adding a dependency.
Run only this ignored observation test for RED and GREEN:

```bash
MIVI_TEST_MODEL="$PWD/models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf" MIVI_THREADS=2 RAYON_NUM_THREADS=2 cargo test -p mivi-model --offline --lib --features fixture-diagnostics --jobs 1 fixture_model_observation -- --ignored --test-threads=1
```

- [x] Wire `generate_streaming_with_cancel`: begin observation at its entry,
retain `reset_context` and existing delegation, bind the existing result, and
finish the observer at actual return. A feature-only local flag records whether
`should_cancel` ever returned true; wrap that same callback without evaluating
it additional times. Classify errors before cancellation, then delivery stop,
then complete. Do not read a wall-clock in the feature-disabled or unarmed path:

```rust
#[cfg(feature = "fixture-diagnostics")]
if let Some(recorder) = self.fixture_recorder.as_mut() { recorder.begin(Instant::now()); }
// Existing reset_context + incremental call are retained, not reimplemented.
#[cfg(feature = "fixture-diagnostics")]
if let Some(recorder) = self.fixture_recorder.as_mut() {
    use crate::fixture_diagnostics::ModelOutcome;
    let outcome = if result.is_err() { ModelOutcome::ModelError }
        else if cancellation_observed.get() { ModelOutcome::Cancelled }
        else if delivery_stopped.get() { ModelOutcome::DeliveryStopped }
        else { ModelOutcome::Complete };
    recorder.finish(Instant::now(), outcome);
}
```

The two flags above are local, nonserialized `Cell<bool>` values. Wrapping
cancellation in the outer streaming call uses these cells, not a closure
that borrows `self.fixture_recorder` while generation already borrows `self`.
The existing incremental call is selected by `cfg`; there is still exactly
one runtime generation call and no second loop. Use this complete wrapping
pattern (the original parameters are `on_token` and `should_cancel`):

```rust
#[cfg(feature = "fixture-diagnostics")]
let cancellation_observed = std::cell::Cell::new(false);
#[cfg(feature = "fixture-diagnostics")]
let delivery_stopped = std::cell::Cell::new(false);
#[cfg(feature = "fixture-diagnostics")]
let result = {
    let mut on_token = on_token;
    let mut should_cancel = should_cancel;
    self.generate_streaming_incremental_with_cancel(prompt, 0, max_tokens,
        |id, text| {
            let keep_going = on_token(id, text);
            if !keep_going { delivery_stopped.set(true); }
            keep_going
        },
        || {
            let cancelled = should_cancel();
            if cancelled { cancellation_observed.set(true); }
            cancelled
        })
};
#[cfg(not(feature = "fixture-diagnostics"))]
let result = self.generate_streaming_incremental_with_cancel(
    prompt, 0, max_tokens, on_token, should_cancel);
// Finish the recorder using the two Cell::get() values, then return result.
```

Cells are not
timestamps/collectors and do not alter callback evaluation count. The
feature-enabled, unarmed path uses the same return values with no captures.

- [x] In `generate_streaming_incremental_with_cancel`, call
`tokenization_done` after existing encoding/BOS processing. In
`generate_tokens_incremental_with_cancel`, call `prefill_begin` after effective
BOS/context validation and before prefix-cache lookup. Bind the existing
`run_prefill` result, preserve its error/false return exactly, and record its
elapsed wall time/outcome. Count newly processed tokens from actual progress,
not `n_prompt - restored` on a cancelled partial prefill. Add an optional
feature-only progress counter at the existing `run_prefill` completed-token
boundary; update it for token and tiled strategies, without new timestamps
per prefill token. For the token strategy call `recorder.processed(1)` only
after the existing successful forward step. For a successful tile call
`recorder.processed(tile.len())` using that branch's actual completed tile
slice. Neither an attempted nor failed forward step increments the count.
At each boundary the feature-gated hook is:

```rust
#[cfg(feature = "fixture-diagnostics")]
if let Some(recorder) = self.fixture_recorder.as_mut() {
    recorder.processed(completed_count);
}
```

Here `completed_count` is the explicit `1` or completed tile length described
above, not a new inference counter. The hook at completion is:

```rust
#[cfg(feature = "fixture-diagnostics")]
if let Some(recorder) = self.fixture_recorder.as_mut() {
    let processed = recorder.snapshot.processed_tokens.unwrap_or(0);
    recorder.prefill_end(Instant::now(), start_prefill_idx, processed, stage_outcome);
}
```

- [x] Immediately after decoder `feed`, before `pending_text.push_str` and
stop handling, record the raw decoded chunk and that generated token ID.
Record decoder flush text before existing final stop trimming. Before each
of the three existing nonempty `on_token` call sites, record delivered text;
bind callback results and mark delivery stop when false without changing
existing branching (including sites that currently ignore the callback result):

```rust
#[cfg(feature = "fixture-diagnostics")]
if let Some(recorder) = self.fixture_recorder.as_mut() {
    recorder.raw(Instant::now(), &decoded_chunk);
    recorder.snapshot.generated_ids.push(next_token);
}
// At callback sites use the actual emit_str/pending_text slice, never the prefix.
#[cfg(feature = "fixture-diagnostics")]
if let Some(recorder) = self.fixture_recorder.as_mut() {
    recorder.delivered(Instant::now(), emitted_text);
}
```

- [x] Prove observer-on/off parity with two sequential fresh model loads:
identical prompt, context, strategy, seed, temperature, and budget; equal
delivered strings, generated output, and post-run sampler RNG state. Test raw
versus stop-filtered delivered text, flush handling, no token IDs for EOS,
prefill/decode cancellation, error finalization, empty output, and no active
observer allocation/timestamps without a session. Use supplied instants for
collector lifecycle tests and synchronized cancellation gates, not sleeps.

- [x] Run sequential affected model tests with feature enabled and disabled,
then scoped Clippy/format. Require no new warnings, review physical boundary
placement, and independently review before committing model hook files:

```bash
RAYON_NUM_THREADS=2 cargo test -p mivi-model --offline --lib --features fixture-diagnostics --jobs 1 -- --test-threads=1
RAYON_NUM_THREADS=2 cargo test -p mivi-model --offline --lib --jobs 1 -- --test-threads=1
RAYON_NUM_THREADS=2 cargo clippy -p mivi-model --offline --lib --tests --features fixture-diagnostics --jobs 1 -- -D warnings -A clippy::manual_div_ceil -A clippy::manual_is_multiple_of -A clippy::items_after_test_module -A clippy::field_reassign_with_default
cargo fmt -p mivi-model -- --check
git diff --check
git add crates/mivi-model/src/model.rs crates/mivi-model/src/fixture_diagnostics.rs
git commit -m "feat: observe fixture prefill and decoded output boundaries"
```

## Task 3: Private server session, owned worker, and artifact writer

**Files:** server feature/lib declarations, new diagnostics module/artifact
submodule, actor wiring, streaming-chat fixture log suppression.

**Interfaces:** `FixtureSession`, `FixtureLimits`, `FixtureRecord`,
`FixtureEngine`, `ArtifactDirectory` are crate-private and feature-only.
`EngineActor::try_spawn_fixture(Model, &ServerConfig, FixtureSession)
-> io::Result<FixtureEngine>` is also crate-private. Normal spawn entry points
continue returning only `EngineHandle` with diagnostics disabled.

- [x] Add forwarding and private modules:

```toml
# mivi-server/Cargo.toml
[features]
fixture-diagnostics = ["mivi-model/fixture-diagnostics"]
```

```rust
// mivi-server/src/lib.rs
#[cfg(feature = "fixture-diagnostics")]
mod fixture_diagnostics;
```

Before these server edits, record the scoped default server baseline:

```bash
RAYON_NUM_THREADS=2 cargo test -p mivi-server --offline --lib --jobs 1 -- --test-threads=1
```

- [x] Define a validated session config and record contract. Fixture limits
are bounded infrastructure defaults, not model-specific policy. Harness
defaults are request count=4, text bytes=65_536, token IDs=256, collector
capacity=4, stream deadline=120s, physical-generation watchdog=120s. Reject
zero values, request/collector counts above 128, durations above 600s, text/ID
bounds rejected by Task 1, and checked aggregate budget above 64MiB. Budget
each retained record for 16 text-cap slots (including metadata and temporary
SSE reconstruction), its ID cap, and 4096 bytes fixed overhead:

```rust
let per_record = limits.model.text_bytes.checked_mul(16)
    .and_then(|text| limits.model.token_ids.checked_mul(4)
        .and_then(|ids| text.checked_add(ids)))
    .and_then(|n| n.checked_add(4096)).ok_or("fixture budget overflow")?;
let total = per_record.checked_mul(limits.records).ok_or("fixture budget overflow")?;
if total > 64 * 1024 * 1024 { return Err("fixture budget exceeds limit"); }
```

No unbounded generic JSON metadata tree; records have the following contract:

```rust
pub(crate) struct FixtureLimits {
    pub model: mivi_model::fixture_diagnostics::CaptureLimits,
    pub requests: usize, pub records: usize,
    pub stream_deadline: std::time::Duration,
    pub generation_watchdog: std::time::Duration,
}
#[derive(serde::Serialize)]
pub(crate) struct FixtureRecord {
    pub sequence: usize, pub fixture_id: String,
    pub rendered_prompt: mivi_model::fixture_diagnostics::CapturedText,
    pub forced_prefix: mivi_model::fixture_diagnostics::CapturedText,
    pub conditioned_prompt: mivi_model::fixture_diagnostics::CapturedText,
    pub model: Option<mivi_model::fixture_diagnostics::ModelCapture>,
    pub descriptor: serde_json::Value,
    pub effective_settings: serde_json::Value,
    pub router_stream: mivi_model::fixture_diagnostics::CapturedText,
    pub first_visible_delta: Option<std::time::Duration>,
    pub stream_elapsed: Option<std::time::Duration>,
    pub metrics_before: crate::state::MetricsSnapshot,
    pub metrics_after: crate::state::MetricsSnapshot,
    pub router_finish: Option<String>, pub saw_done: bool,
    pub router_parse_error: Option<&'static str>,
    pub answer_quality: QualityOutcome,
    pub engine_terminal: EngineTerminal,
    pub capture_incomplete: bool,
}
#[derive(serde::Serialize)]
pub(crate) enum EngineTerminal { NotStarted, Unobserved, Returned, Cancelled, ReceiverClosed, ModelError }
#[derive(serde::Serialize)]
pub(crate) enum QualityOutcome { NotAssessed, Passed, Failed, ContinuationUnexercised }
```

The descriptor contains bounded model-config name, existing family/quantization
metadata, GGUF size and context limit, not a model path or a claimed file hash.
Settings include actual sampler config after applying request options, model
prefill strategy, thread counts, max tokens, and profile source (metadata or
explicit). Serialize only allowlisted values with bounded strings/arrays;
clip before cloning and mark each clipped value; cap metadata/settings
serialized bytes individually at the text-byte limit. Reject metadata that
cannot be represented in the bounded fixed schema without marking loss.
Do not `to_value` a whole GGUF metadata map or arbitrary profile configuration.

- [x] Implement `FixtureSession::new(FixtureLimits) -> Result<Self, &'static str>`;
`arm(&self, fixture_id: &str) -> Result<usize, &'static str>`;
`begin(&self, sequence, prompt, prefix, settings, descriptor) -> Option<FixtureRecord>`;
`finish_engine(&self, FixtureRecord)`; and
`take_finished(&self, sequence) -> Option<FixtureRecord>`. IDs are 1–64 ASCII
alphanumeric/underscore/hyphen characters and are labels only, not paths.
Use bounded records, one armed request at a time, and an atomic incomplete
flag; `Mutex::try_lock` only at request boundaries. On contention/full/poison,
mark incomplete and return without blocking generation. No lock or Arc clone
on every token. `arm` reserves a monotonically increasing checked sequence;
attempts exceeding configured count fail before another request is sent.
Add `wait_finished(&self, sequence) -> Result<FixtureRecord, &'static str>`
as an async method bounded by `generation_watchdog`, using a session-owned
`tokio::sync::Notify`. `finish_engine` and incomplete states call `notify_one`
after releasing the short request-boundary lock. Register notification before
checking the record, so completion between checking and awaiting is not lost.
Retain a bounded initial record at the begin boundary, updated with effective
settings before generation, for `partial(&self, sequence: usize)
-> Option<FixtureRecord>`. This request-boundary copy is included in the
16-slot aggregate budget and removed on finalization. It does not mirror
model tokens or introduce per-token locks. A watchdog/collector failure
without a physical-return observation uses `EngineTerminal::Unobserved` and
`capture_incomplete = true`, not `NotStarted`, `Returned`, or a fabricated
model outcome. `NotStarted` is reserved for a request known not to have
entered generation. Do not imply that an unobserved worker has been joined.

The incomplete marker follows this exact non-blocking pattern:

```rust
match records.try_lock() {
    Ok(mut records) if records.len() < capacity => records.push(record),
    _ => incomplete.store(true, std::sync::atomic::Ordering::Release),
}
```

- [x] Extract the existing actor spawn body into one shared private inner
constructor returning `(EngineHandle, JoinHandle<()>)`. Default callers
discard the join handle exactly as today and supply no session. Fixture
callers retain the join handle and a completion notification. Preserve the
single existing command loop, prefill setup, and panic handling; no alternate
generation implementation. On the streaming match arm, capture the existing
rendered prompt/forced prefix and effective sampler settings, install the
model recorder, then run unchanged `handle_generate_stream`. Take the model
snapshot and finish the session only after the handler returns, including
early cancellation/closure/JSON rejection/errors. Record a caught panic as an
incomplete capture without serializing its potentially sensitive payload.

`FixtureEngine` owns `handle: Option<EngineHandle>`, actor join handle, and a
completion receiver. The harness drops router/state/handle clones before
shutdown, waits for completion with the configured watchdog, and joins only
after `JoinHandle::is_finished()` is true. Drop actor-owned model/receiver
before sending completion. The last return race is checked with bounded
async yielding after that notification, not an unbounded blocking join.
A timeout is a verification failure, not permission to
claim cleanup or kill an unrelated process. No default-runtime API changes.

- [x] Add `EngineHandle::fixture_capture_active(&self) -> bool` with default
false behavior and a feature-only session check. Use it to suppress only the
captured fixture's incoming prompt, streaming completion box, prompt-summary
extension, and errors that interpolate generated text. Preserve error codes,
metrics, SSE payloads, and nonfixture logging. The branching pattern is:

```rust
if !engine.fixture_capture_active() {
    crate::logging::print_completion_response_box(thinking.as_deref(), tools, Some(&clean_reply));
}
```

Do not suppress all application logs or add an environment logger switch.
Test the predicate for default, feature-without-session, and fixture handles;
exercise error paths with captured synthetic markers and verify the fixture
logging helpers emit no marker content.

- [x] Implement private artifact creation with UUID-named directories under
the platform temp directory, exclusively created (no `create_dir_all`). Use
fixed sequence filenames generated internally. No arbitrary caller path:

```rust
pub(crate) struct ArtifactDirectory { path: std::path::PathBuf }
impl ArtifactDirectory {
    pub(crate) fn create() -> std::io::Result<Self> {
        let path = std::env::temp_dir().join(format!("mivi-fixture-capture-{}", uuid::Uuid::new_v4()));
        let mut builder = std::fs::DirBuilder::new();
        #[cfg(unix)] {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder.create(&path)?;
        #[cfg(unix)] {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700))?;
        }
        Ok(Self { path })
    }
    pub(crate) fn write(&self, sequence: usize, record: &super::FixtureRecord)
        -> std::io::Result<()> {
        let path = self.path.join(format!("request-{sequence:04}.json"));
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)] {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(path)?;
        #[cfg(unix)] {
            use std::os::unix::fs::PermissionsExt;
            file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
        }
        serde_json::to_writer(&mut file, record).map_err(std::io::Error::other)?;
        std::io::Write::flush(&mut file)
    }
    pub(crate) fn path(&self) -> &std::path::Path { &self.path }
}
```

An I/O failure returns a separate persistence status; a partial file is not
reported as a completed artifact. Do not change the model terminal state.
Do not serialize raw `io::Error` strings that could include paths. If the
repo's Rust floor lacks `Error::other`, use `Error::new(ErrorKind::Other, error)`
without changing that floor.

- [x] Observe a compiled RED for session retention/byte-exact prefix separation
against preparatory no-op retention, then GREEN. Unit tests assert rendered
`"prompt\n"`, prefix `"prefix:"`, conditioned `"prompt\nprefix:"`, raw/delivered
`"body"`, separately collected parser result; failed/model-not-started captures;
two sequences never overwrite; capacity/lock-contention marked incomplete;
duplicate `write` fails with `AlreadyExists`; pre-existing symlink filename
fails; Unix modes are exact; injected write failure preserves model outcome;
route logging remains enabled without a session. All tests use temporary
synthetic files, not the project or user directories.

- [x] Run scoped server baseline/feature diagnostics tests, existing actor,
streaming and tool tests; Clippy/format; independent spec/quality review.
Socket-binding tests require normal escalation if sandbox-denied, not a false
pass. Explicit Task 3 commit paths exclude all user changes:

```bash
RAYON_NUM_THREADS=2 cargo test -p mivi-server --offline --lib --features fixture-diagnostics --jobs 1 fixture_diagnostics -- --test-threads=1
RAYON_NUM_THREADS=2 cargo test -p mivi-server --offline --lib --jobs 1 engine_actor::tests -- --test-threads=1
RAYON_NUM_THREADS=2 cargo test -p mivi-server --offline --lib --jobs 1 streaming -- --test-threads=1
RAYON_NUM_THREADS=2 cargo test -p mivi-server --offline --lib --jobs 1 tool -- --test-threads=1
RAYON_NUM_THREADS=2 cargo clippy -p mivi-server --offline --lib --tests --features fixture-diagnostics --jobs 1 -- -D warnings -A clippy::manual_div_ceil -A clippy::manual_is_multiple_of -A clippy::items_after_test_module -A clippy::field_reassign_with_default
cargo fmt -p mivi-server -p mivi-model -- --check
git diff --check
git add crates/mivi-server/Cargo.toml crates/mivi-server/src/lib.rs crates/mivi-server/src/fixture_diagnostics.rs crates/mivi-server/src/fixture_diagnostics/artifacts.rs crates/mivi-server/src/engine_actor.rs crates/mivi-server/src/routes/chat.rs
git commit -m "feat: capture isolated server fixture evidence privately"
```

## Task 4: Real-router fixture runner and reviewed release

**Files:** new `fixture_diagnostics/fixtures.rs`; collector/server regression
tests if fixture execution reveals defects; root metadata/changelog/spec/plan.

**Interfaces:** Consume `FixtureSession`, owned fixture engine and artifact
writer. Add ignored `fixture_diagnostics::fixtures::fixture_generation_capture`
and `fixture_observer_parity` tests. These are not public commands.

- [x] Use an exclusively created private temp workspace and `create_new` to
write `example.rs` with exact bytes:

```rust
const SOURCE: &str = "pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n";
```

Validate all finite limits before loading the explicit `MIVI_TEST_MODEL` path.
Missing path fails the explicitly invoked ignored test with a configuration
message, not a skipped-success report. Set context=4096, chunk tile=64,
request budget=120s, first-output budget=90s, threads=2, concurrency=1 for
these labelled fixtures; none become new production/model defaults.

- [x] Build `AppState::with_config("fixture-model", ToolBroker::new(),
engine.handle().ok_or("fixture engine shut down")?.clone(), None, config).with_workspace(workspace)` and call
`create_router(Arc::new(state))`. Use the actual route via Tower, no listener:

```rust
let request = axum::http::Request::builder()
    .method("POST").uri("/v1/chat/completions")
    .header("content-type", "application/json")
    .body(axum::body::Body::from(serde_json::to_vec(&payload)?))?;
use tower::ServiceExt;
let response = app.clone().oneshot(request).await?;
assert!(response.status().is_success());
```

- [x] Use this known controlled required-tool payload; aliases/function names
are fixture data, not runtime dispatch logic:

```rust
let payload = serde_json::json!({
    "model": "mivi", "stream": true, "temperature": 0, "seed": 7, "max_tokens": 48,
    "messages": [
        {"role":"system","content":"Use read_file to inspect the requested file before answering. Do not guess its contents."},
        {"role":"user","content":"Read example.rs using read_file. Then tell me the Rust function name and what it returns."}
    ],
    "tools": [{"type":"function","function":{
        "name":"read_file","description":"Read a UTF-8 file and return its contents.",
        "parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}
    }}], "tool_choice":"required"
});
```

- [x] Add a bounded streaming SSE reader using existing `Body::into_data_stream`
and `futures::StreamExt`. Start its monotonic clock before router dispatch;
carry partial UTF-8/event bytes between frames, parse `data:` events ending
in blank lines, ignore comments/role-only events, and require `[DONE]` or an
explicit incomplete-stream result. Cap buffered event bytes, total retained
SSE bytes, number of events and call fragments. Reject malformed JSON with a
diagnostic parser status while preserving bounded raw SSE; never dump bodies
in assertions. The whole drain is bounded:

```rust
let mut transcript = mivi_model::fixture_diagnostics::CapturedText::new(limits.model.text_bytes);
let observed_result = tokio::time::timeout(limits.stream_deadline,
    drain_fixture_sse(response.into_body(), limits.model.text_bytes,
        dispatch_started, &mut transcript))
    .await.unwrap_or(Err("fixture stream deadline"));
// Keep this Result until the physical snapshot has been obtained or the
// generation watchdog has expired; never return early before owned-worker cleanup.
```

`drain_fixture_sse(Body, usize, Instant, &mut CapturedText)
-> Result<ObservedStream, &'static str>`
updates the caller's bounded transcript and returns first visible content/tool delta, elapsed time,
finish reason, done flag, and reconstructed calls. `ObservedStream` retains
calls as index/ID/name/argument text with bounded builders and rejects duplicate
conflicting IDs while appending bounded name fragments. Test split `data:`/JSON/UTF-8/blank-line frames,
keep-alives, initial empty role, calls fragmented by index, error finish,
overflow, stream ending before DONE, and truncated events before live use.

Keep the transcript owned by the caller so an SSE parse failure does not lose
the bounded evidence. The reader accepts
`transcript: &mut CapturedText`; its error is stored as a diagnostic parser
status, with that partial transcript retained in the record. Use these types
and complete reader structure:

```rust
struct ObservedCall {
    index: usize, id: String, name: String, arguments: String,
}
struct ObservedStream {
    first_visible_delta: Option<std::time::Duration>,
    elapsed: std::time::Duration, finish: Option<String>,
    saw_done: bool, saw_error: bool, calls: Vec<ObservedCall>, content: String,
}
fn append_field(field: &mut String, chunk: &str, cap: usize) -> Result<(), &'static str> {
    let next = field.len().checked_add(chunk.len()).ok_or("fixture field overflow")?;
    if next > cap { return Err("fixture field exceeds bound"); }
    field.push_str(chunk); Ok(())
}
async fn drain_fixture_sse(
    body: axum::body::Body, cap: usize, started: std::time::Instant,
    transcript: &mut mivi_model::fixture_diagnostics::CapturedText,
) -> Result<ObservedStream, &'static str> {
    use futures::StreamExt;
    let mut stream = Box::pin(body.into_data_stream());
    let mut pending = Vec::<u8>::new();
    let mut decoder = mivi_tokenizer::Utf8StreamDecoder::new();
    let mut result = ObservedStream { first_visible_delta: None,
        elapsed: std::time::Duration::ZERO, finish: None, saw_done: false, saw_error: false,
        calls: Vec::new(), content: String::new() };
    let mut event_count = 0usize;
    while let Some(frame) = stream.next().await {
        let frame = frame.map_err(|_| "fixture body error")?;
        if frame.len() > cap {
            transcript.push(&decoder.feed(&frame[..cap]));
            transcript.truncated = true;
            return Err("fixture frame exceeds bound");
        }
        transcript.push(&decoder.feed(&frame));
        if transcript.truncated { return Err("fixture SSE retention exceeds bound"); }
        let next = pending.len().checked_add(frame.len()).ok_or("fixture event overflow")?;
        if next > cap { return Err("fixture event exceeds bound"); }
        pending.extend_from_slice(&frame);
        loop {
            let lf = pending.windows(2).position(|w| w == b"\n\n").map(|p| (p, 2));
            let crlf = pending.windows(4).position(|w| w == b"\r\n\r\n").map(|p| (p, 4));
            let boundary = match (lf, crlf) {
                (Some(a), Some(b)) => Some(if a.0 < b.0 { a } else { b }),
                (a, b) => a.or(b),
            };
            let Some((end, separator)) = boundary else { break; };
            event_count = event_count.checked_add(1).ok_or("fixture event count overflow")?;
            if event_count > 512 { return Err("fixture event count exceeds bound"); }
            let event = std::str::from_utf8(&pending[..end]).map_err(|_| "fixture event UTF-8")?;
            let data = event.lines().filter_map(|line| line.strip_prefix("data:"))
                .map(|line| line.strip_prefix(' ').unwrap_or(line)).collect::<Vec<_>>().join("\n");
            if !data.is_empty() {
                if result.saw_done { return Err("fixture data after DONE"); }
                if data == "[DONE]" {
                    result.saw_done = true;
                } else {
                    let value: serde_json::Value = serde_json::from_str(&data)
                        .map_err(|_| "fixture event JSON")?;
                    if value.get("error").is_some() {
                        result.saw_error = true;
                    } else {
                    let choices = value.get("choices").and_then(serde_json::Value::as_array)
                        .ok_or("fixture choices missing")?;
                    if choices.len() != 1 { return Err("fixture expects one choice"); }
                    let choice = &choices[0];
                    let delta = &choice["delta"];
                    let content = delta.get("content").and_then(serde_json::Value::as_str).unwrap_or("");
                    let tools = delta.get("tool_calls").and_then(serde_json::Value::as_array);
                    if (!content.is_empty() || tools.is_some_and(|calls| !calls.is_empty()))
                        && result.first_visible_delta.is_none() {
                        result.first_visible_delta = Some(started.elapsed());
                    }
                    append_field(&mut result.content, content, cap)?;
                    if let Some(tools) = tools {
                        if tools.len() > 16 { return Err("fixture tool-call count exceeds bound"); }
                        for fragment in tools {
                            let index = fragment.get("index").and_then(serde_json::Value::as_u64)
                                .and_then(|n| usize::try_from(n).ok()).ok_or("fixture tool index")?;
                            if index >= 16 { return Err("fixture tool index exceeds bound"); }
                            let position = result.calls.iter().position(|call| call.index == index);
                            let position = match position {
                                Some(p) => p,
                                None => {
                                    result.calls.push(ObservedCall { index, id: String::new(),
                                        name: String::new(), arguments: String::new() });
                                    result.calls.len() - 1
                                }
                            };
                            let call = &mut result.calls[position];
                            if let Some(id) = fragment.get("id").and_then(serde_json::Value::as_str) {
                                if !call.id.is_empty() && call.id != id { return Err("fixture conflicting call ID"); }
                                if call.id.is_empty() { append_field(&mut call.id, id, 256)?; }
                            }
                            let function = &fragment["function"];
                            append_field(&mut call.name, function.get("name").and_then(serde_json::Value::as_str).unwrap_or(""), 256)?;
                            append_field(&mut call.arguments, function.get("arguments").and_then(serde_json::Value::as_str).unwrap_or(""), cap)?;
                        }
                    }
                    if let Some(finish) = choice.get("finish_reason").and_then(serde_json::Value::as_str) {
                        if finish.len() > 64 { return Err("fixture finish reason exceeds bound"); }
                        if result.finish.as_deref().is_some_and(|prior| prior != finish) {
                            return Err("fixture conflicting finish reason");
                        }
                        result.finish = Some(finish.to_owned());
                    }
                    }
                }
            }
            pending.drain(..end + separator);
        }
    }
    transcript.push(&decoder.flush());
    if !result.saw_done { return Err("fixture stream ended before DONE"); }
    if !pending.is_empty() { return Err("fixture incomplete trailing event"); }
    result.elapsed = started.elapsed();
    result.calls.sort_by_key(|call| call.index);
    Ok(result)
}
```

The per-frame clocks measure consumer observations, not pure model compute.
An error record preserves the transcript/engine capture and marks parser or
retention failure separately; it never prints the failing event. Tool name
fragments append normally; repeated conflicting IDs are rejected, not guessed.
These parser limits are fixture infrastructure, not production protocol limits.
After `[DONE]`, continue polling the bounded body to EOF, permitting comments
but rejecting further data events. Only then take the final stream metrics.
Returning at the marker would drop the logged body before its EOF observation
and incorrectly count the harness's successful request as a client disconnect.
Test both body EOF and genuine early drop with the actual lifecycle wrapper.

- [x] Continue only when no error event occurred, `[DONE]` was observed, and
finish reason is `tool_calls`. If a valid call is emitted, require exactly `read_file` and string
`path="example.rs"`; return actual file bytes including trailing LF with the
same `tool_call_id`, using `tool_choice:"none"`. Never execute arbitrary model
expressions or read a model-selected path. Construct the continuation:

```rust
let mut followup = payload.clone();
followup["tool_choice"] = serde_json::json!("none");
let messages = followup["messages"].as_array_mut().ok_or("fixture message array")?;
messages.push(serde_json::json!({"role":"assistant","content":null,"tool_calls":[{
    "id":call.id.clone(),"type":"function","function":{"name":call.name.clone(),"arguments":call.arguments.clone()}
}]}));
messages.push(serde_json::json!({"role":"tool","tool_call_id":call.id,"content":SOURCE}));
```

Validate the actual file contents equal `SOURCE` before sending. Record final
answer-quality success only when it identifies `add` and `a + b`. If the call
is malformed, retain that failure, omit unsafe tool execution, and mark the
continuation unexercised. Capture infrastructure can pass while answer quality
fails, but no failed/unexercised tool round trip becomes a claimed success.

- [x] After each request, wait boundedly for its physical engine snapshot,
attach measured SSE/metrics result, and persist outside generation. API
timeout and engine-return observations remain distinct. Finish/join the
owned actor after dropping all router/state/handle clones. Report only caps,
statuses, timings, test results and artifact directory, never captured text.
Initialize `router_parse_error = None` and `answer_quality = NotAssessed`.
After `wait_finished`, move `transcript` into `record.router_stream`, then
match `observed_result`: on success copy its timing/finish/DONE observations;
on failure store its static error in `record.router_parse_error` and mark
`capture_incomplete = true`. Preserve partial evidence on a watchdog failure
with an honest unfinished engine state. Perform persistence and owned-worker
cleanup before returning any diagnostic failure. Persistence failure is a
separate harness result, never changed to a successful capture.

- [x] Run the scoped real-model fixture and parity tests sequentially:

The parity test uses two fresh, sequential model loads and an explicit
initial RNG state, so comparing post-generation RNG does not compare two
clock-seeded constructor states. Keep it inside the feature-only fixture
module and use this body:

```rust
#[test]
#[ignore = "requires explicit MIVI_TEST_MODEL; compares observer off/on"]
fn fixture_observer_parity() -> Result<(), Box<dyn std::error::Error>> {
    use mivi_model::fixture_diagnostics::{CaptureLimits, CapturedText};
    let path = std::env::var("MIVI_TEST_MODEL")?;
    let path = std::path::Path::new(&path);
    let prompt = "Return the word hello.";
    let mut baseline = mivi_model::Model::load_with_ctx(path, Some(512))?;
    baseline.sampler.config.temperature = 0.0;
    baseline.sampler.set_seed(7);
    let mut baseline_chunks = CapturedText::new(4096);
    let baseline_output = baseline.generate_streaming(prompt, 16, |_, text| {
        baseline_chunks.push(text); true
    })?;
    let baseline_rng = baseline.sampler.rng_state();
    assert!(!baseline_chunks.truncated);
    assert!(baseline.take_fixture_capture().is_none());
    drop(baseline);

    let mut observed = mivi_model::Model::load_with_ctx(path, Some(512))?;
    observed.sampler.config.temperature = 0.0;
    observed.sampler.set_seed(7);
    observed.start_fixture_capture(CaptureLimits { text_bytes: 4096, token_ids: 32 })?;
    let mut observed_chunks = CapturedText::new(4096);
    let observed_output = observed.generate_streaming(prompt, 16, |_, text| {
        observed_chunks.push(text); true
    })?;
    let snapshot = observed.take_fixture_capture().ok_or("missing parity capture")?;
    assert!(!observed_chunks.truncated && !snapshot.delivered.truncated);
    assert!(baseline_output == observed_output, "observer changed generation");
    assert!(baseline_chunks.text == observed_chunks.text, "observer changed delivery");
    assert!(snapshot.delivered.text == observed_chunks.text, "capture differs from delivery");
    assert_eq!(baseline_rng, observed.sampler.rng_state());
    Ok(())
}
```

Use a scoped GGUF-safe model descriptor, not the raw path, in summaries;
neither test prints prompt/generated text. Raw direct-model parity does not
claim correctness of a rendered chat/tool answer, which is measured by the
separate real-router fixture.

```bash
MIVI_TEST_MODEL="$PWD/models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf" MIVI_THREADS=2 RAYON_NUM_THREADS=2 cargo test -p mivi-server --offline --release --lib --features fixture-diagnostics --jobs 1 fixture_generation_capture -- --ignored --test-threads=1 --nocapture
MIVI_TEST_MODEL="$PWD/models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf" MIVI_THREADS=2 RAYON_NUM_THREADS=2 cargo test -p mivi-server --offline --release --lib --features fixture-diagnostics --jobs 1 fixture_observer_parity -- --ignored --test-threads=1 --nocapture
```

These commands select an existing test model only; the implementation must
work with any explicitly supplied supported model. No paired runtime or
broader numerical corpus rerun is included in this increment.
Run these commands from the repository root. The shell resolves the GGUF to
an absolute path before Cargo starts; Cargo's test process runs from its
package directory, so a repository-relative model path is not sufficient.
Label the real-router runs as release-profile diagnostics. Task 2's original
compiled RED/GREEN remains in its same debug profile; the observed 155.82s
debug RED motivated release-profile execution for additional live parity/
lifecycle fixtures. Neither profile is a controlled runtime speed comparison.

- [x] Independently review all source wiring and real evidence before the
release bump. Resolve Important findings with compiled regressions/re-review.
Then update root workspace version 0.2.61→0.2.62 once, changelog ideas/sources
from the spec plus exact tests/fixture status/caps/timings/privacy limitations,
and spec/plan execution status. Refresh only workspace lockfile versions.
Do not publish a speedup, model-quality fix, or successful agent test claim.

- [x] Run final scoped default and feature suites/checks sequentially; ignored
fixture tests above remain explicitly invoked, not silently counted here:

```bash
RAYON_NUM_THREADS=2 cargo test -p mivi-model --offline --lib --jobs 1 -- --test-threads=1
RAYON_NUM_THREADS=2 cargo test -p mivi-model --offline --lib --features fixture-diagnostics --jobs 1 -- --test-threads=1
RAYON_NUM_THREADS=2 cargo test -p mivi-server --offline --lib --jobs 1 -- --test-threads=1
RAYON_NUM_THREADS=2 cargo test -p mivi-server --offline --lib --features fixture-diagnostics --jobs 1 -- --test-threads=1
RAYON_NUM_THREADS=2 cargo clippy -p mivi-model -p mivi-server --offline --lib --tests --features fixture-diagnostics --jobs 1 -- -D warnings -A clippy::manual_div_ceil -A clippy::manual_is_multiple_of -A clippy::items_after_test_module -A clippy::field_reassign_with_default
cargo fmt -p mivi-model -p mivi-server -- --check
cargo tree -p mivi-server --offline -e normal --depth 1
RAYON_NUM_THREADS=2 cargo build -p mivi --offline --release --jobs 1
target/release/mivi --version
git diff --check
```

Root default executable must report v0.2.62 and expose no capture option;
normal dependency versions remain unchanged. No full-workspace command.

- [x] Obtain final whole-change review, stage only exact task source/docs/
metadata paths, commit, publish authorized main without force, and compare
`git rev-parse HEAD` with `git ls-remote origin refs/heads/main`. Record actual
publication after verification, preserving `.gitignore` and local artifacts.

## Plan self-review and execution status

- Feature-off/session-off behavior, bounded text/IDs, lifecycle/timing: Tasks 1–2.
- Real decoder before stop trimming, callback delivery, no inference drift: Task 2.
- Exact prompts/prefix separation, settings, record isolation, private exclusive
  files, failure states, fixture log suppression, worker ownership: Task 3.
- Actual router, bounded SSE, byte-exact tool result, measured metrics, cold/
  warm labelling, cancellation finalization, parity, release evidence: Task 4.
- Reference-runtime comparison and defect fixes remain separate follow-ups.

Execution status: subagent-driven implementation completed, independently
reviewed and published as v0.2.62 on `main`. Task 1 completed in `8e44231`, with
independent spec/quality review approved and no findings. Controller reran
12 focused collector tests and the feature-enabled model suite (53 passed,
4 ignored), formatting and whitespace checks. Original behavioral RED is
reported in the implementer evidence, not independently replayed.
Task 2 resumed after user interruption; hooks and actual-model parity/lifecycle
verification are implemented in `6b1cc89`, independently reviewed as spec
compliant and quality approved. Controller reran its feature-enabled suite
(57 passed, 6 ignored). Original debug RED/GREEN and release live verification
are implementer-reported, not independently replayed. The release parity/
lifecycle fixture passed in 9.29s, with two sequential model loads and observed
model reuse; this is correctness evidence, not a benchmark. Minor coverage
limitations remain explicit: actual-model nonempty decoder-byte flush, cache
hits, adapter fallback and mid-layer failures were not exercised.
Task 3 completed in `185da9c`, independently reviewed as spec compliant and
quality approved. The implementer reported 88 default and 103 feature-enabled
server tests passing; controller reran 13 focused diagnostics tests plus
formatting and whitespace checks. The private server contract explicitly
represents unobserved physical completion and retains bounded initial records
on watchdog failures. Minor review limitation: logging-helper coverage does
not exercise the actual streaming error branches; retain for final triage.
Task 4 source is implemented in `45034d6`. Controller verified 23 focused
diagnostics tests, scoped Clippy for library/test targets, and fresh live
router/parity runs. The first router run took 10.06s end-to-end and produced
a valid read-file continuation and the expected answer, with complete captures
and owned-worker join; observer parity passed with two sequential fresh loads.
The first/continuation captures used 110/96 conditioned prompt tokens, both
with zero prefix reuse. The continuation reused the loaded actor, not a
measured KV-cache warm hit. These are local correctness observations, not
a runtime speed comparison or general agent-quality claim.
Independent review found an Important metadata-completeness gap, fixed in
`f409922` with compiled behavioral RED/GREEN and 24 focused diagnostics tests
passing (2 live tests separately ignored). Full untruncated re-review approved
the original diff and fix, with no remaining Critical/Important findings.
Pre-bump scoped source suites passed
41/57 model tests (4/6 ignored), 88/113 server tests (0/2 ignored), after
permission was granted for the existing local socket-bind test.
Workspace bumped once to v0.2.62 after the task gates. Post-bump scoped suites
passed: model 41 default / 57 enabled (4/6 ignored), server 88 default / 114
enabled (0/2 ignored). Scoped Clippy/fmt/tree/diff checks passed, and the root
release build reports `mivi 0.2.62` with no capture option. Only14 workspace
lockfile versions changed; normal dependency versions did not. One controller
lint overlapped a preceding release command inadvertently; no concurrent model
loads, and subsequent checks were sequential. Live success predates the
predicate-only clipping fix and version bump; unclipped live settings are
unaffected, and the new clipping behavior has compiled regression coverage.
Whole-change reviewer read all 6005 packet lines to EOF and approved with no
Critical/Important findings; explicit Minor coverage limits remain follow-ups.
Main was fast-forwarded to reviewed release `c1e931d`, its 88 default server
tests passed again, and GitHub `main` was pushed without force. Remote/local
SHA equality was verified at `c1e931dcfd37b686952386719318b45c8db84fba` before
this documentation-only publication annotation. Source/version are unchanged
by the annotation; private artifacts and user `.gitignore` remain excluded.
User `.gitignore` preserved. Durable execution ledger:
`.superpowers/sdd/fixture-progress.md` (local, ignored).
