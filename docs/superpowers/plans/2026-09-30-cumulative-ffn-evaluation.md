# Cumulative FFN Evaluation Implementation Plan

> **For agentic workers:** Use test-driven development and inline execution for this tightly coupled harness extension; request an independent code review before publishing.

**Goal:** Measure cumulative packed-activation FFN logit sensitivity across both local models on coding/tool raw-text fixtures without changing production inference.

**Architecture:** Extend the existing test-only tile walker with down-only and full-FFN modes. Full FFN recomputes gate/up, SwiGLU, and down using complete matrices; unsupported formats use the existing F32-activation path. A recomputed, non-packed FFN control distinguishes harness effects from quantization effects. Apply residual deltas to every token row, retaining the existing rounding limitation.

**Tech Stack:** Rust, existing mivi-model public tile APIs, mivi-quant test-only packed Q4 kernels; no new dependencies.

## Global Constraints

- Cargo build/check/test jobs: 1; test threads: 1; Rayon/inference threads: 2.
- Scoped mivi-quant commands only, never a full-workspace build/check/test.
- No new unsafe, production hooks, model-name switches, or fixed token IDs.
- Preserve the user's unrelated .gitignore edit.
- Raw fixtures are not chat-template or agent success tests; no timing or quality acceptance claims.
- Update changelog with ideas/inspirations/sources, increment 0.2.56 to 0.2.57, and push verified changes under standing user authorization.

## Task 1: Test-only cumulative projection/FFN helper

**Files:** create `crates/mivi-quant/src/q4_k_m/packed_prefill/real_weights/captured_activations/cumulative.rs`; modify parent `captured_activations.rs`.

**Interfaces:** consume existing packed Q4/F32 batched matmul functions; produce a projection helper that reports packed versus fallback, plus full-FFN recomputation returning down outputs and coverage counts.

- [x] Add failing model-free tests: nonzero F32 fallback equals existing matmul; synthetic Q4 eligible packing executes; mixed gate/up/down recomputation matches the F32 control; invalid shapes/nonfinite inputs reject.
- [x] Run `RAYON_NUM_THREADS=2 cargo test -p mivi-quant --offline --lib --jobs 1 cumulative -- --test-threads=1`; observe assertion failure before implementation.
- [x] Implement type/shape-driven packed selection and full-row recomputation. Reuse `mivi_model::swiglu_rows` rather than copying activation math.
- [x] Rerun those tests and the existing quant library tests; expect zero failures.

## Task 2: Cumulative walker and multi-prompt evaluation

**Files:** modify `crates/mivi-quant/src/q4_k_m/packed_prefill/real_weights/captured_activations.rs`.

**Interfaces:** consume the helper outputs; extend the walker with replay, isolated down, cumulative down, full recomputed F32 control, and full packed FFN modes.

- [x] Add an ignored integration test whose initial implementation evaluates zero cases; assert at least three prompt/mode pairs and observe failure.
- [x] Factor metadata-BOS token preparation for reuse; clear prefix and recurrent/KV state between prompts.
- [x] Use three short raw fixtures: coding error handling, read_file request, and tool-result continuation. Reject fixture truncation instead of quietly evaluating incomplete fixtures.
- [x] For every fixture, require production baseline and non-packed FFN-recompute controls to agree within max absolute logit error 1e-3. Evaluate both modes using full matrices; require nonzero coverage, finite metrics, and report packed/fallback counts plus next-token IDs.
- [x] Run release ignored integration separately with each existing LFM GGUF and jobs 1; record actual values without imposing a fabricated quality threshold.

## Task 3: Verify, document, release

**Files:** `README.md`, `CHANGELOG.md`, `Cargo.toml`, `Cargo.lock`, this plan.

- [x] Independent read-only code review of the changed walker/helper, addressing important findings.
- [x] Record all prompt/mode measurements, control results, caveats, and source links in the changelog; document focused invocation and prompt/token controls.
- [x] Bump patch once. Run quant library tests, targeted model evaluations, scoped Clippy, fmt and git diff checks; confirm no added normal dependencies.

**Publication handoff:** Commit only task files, push main without force, and verify local/remote heads agree. Leave .gitignore unstaged. Git history is the publication record for this completed implementation.

## Execution Record

- Projection and corpus stubs failed the expected assertions before implementation.
- Review findings about direct production controls and derived finite metrics were reproduced with failing model-free regressions, fixed, and approved on re-review.
- v0.2.57 debug quant library: 31 passed, four opt-in tests ignored.
- Focused release `captured_activations --include-ignored`: eight passed per local model, including the existing isolated capture evaluation and the new six-case corpus.
- Both production and non-packed FFN controls matched exactly for all six model/fixture combinations. Twelve cumulative cases are recorded in CHANGELOG.md.
- Scoped Clippy, package formatting, and diff checks passed; no new normal dependencies. Production inference unchanged. Packed rollout rejected pending stronger quality evidence.
