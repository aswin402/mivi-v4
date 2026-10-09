# Four-row Isolation and Capture Implementation Plan

> **For agentic workers:** Use executing-plans and TDD for the approved diagnostics;
> GPT-6 Luna/high performs read-only review while main runs bounded diagnostics.

**Goal:** Distinguish accumulation cost from projection stages and test actual activations faithfully.

**Architecture:** New test-only children of `batch_scratch::four_row`; reuse established
SIMD/model tile operations, no production changes or new unsafe intrinsics.

**Tech Stack:** Rust, Rayon, existing GGUF/model public API and test fixtures.

## Constraints

Cargo jobs=1, Rayon threads=2; scoped offline commands and serial harness only.
Preserve unrelated `.gitignore`, no workspace-wide checks/builds/tests. Prompts/weights/
activations stay in memory; all timings and negative cases retained. Default dispatch unchanged.

## Tasks

- [x] Register test-only diagnostic module; add placeholder `validate_exact` and
      `isolation_shape` controls and observe RED on valid cases.
- [x] Implement strict length/finite/full-bit validation plus bounded shape controls.
- [x] Add accumulation-only diagnostic with four decoded rows, pretransposed inputs,
      baseline two-pair control, one warmup/six alternating pairs, 200 calls/member.
- [x] Implement one-tile baseline walker/capture using production tile functions;
      require full final-logit bit parity before any captured timing.
- [x] Add three-control captured replay, metadata selection, explicit privacy and
      work/cache declarations; three calls/member, one warmup/six permutations.
- [x] Run scoped controls then bounded ignored live diagnostics, Cargo1/Rayon2;
      isolate diagnostic command timeout 180s, captured command timeout 240s.
- [x] Review source/evidence with Luna/high; run final-version targeted regressions,
      formatting/diff checks; retain exact-gate failures if encountered, never relax them.
- [x] Record evidence/limits and ideas/sources; bump 0.2.77 to 0.2.78, commit/push scoped files.

Targeted commands (prefix every Cargo invocation with `CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2`):

```sh
cargo test -p mivi-quant --lib --offline --features batch-four-row-experiment faithful_diagnostic -- --test-threads=1
cargo test -p mivi-quant --lib --release --offline --features batch-four-row-experiment four_row_accumulation_isolation -- --ignored --test-threads=1 --nocapture
cargo test -p mivi-quant --lib --release --offline --features batch-four-row-experiment four_row_captured_activation -- --ignored --test-threads=1 --nocapture
```

Live capture requires `MIVI_TEST_MODEL` pointing to an existing local GGUF using an
absolute path (Cargo executes the test from its crate directory). No new
weights downloaded. No candidate integration or end-to-end latency claim is authorized.

Verification: v0.2.78 release batch filter 21 passed/3 ignored; feature-off release
batch filter 6 passed; debug diagnostic controls 3 passed/2 ignored. Both live
diagnostics passed before the publication bump, with unchanged timing/arithmetic
after review. Luna/high initial and follow-up reviews approved; 64KiB UTF-8 override
guard and clear operation counts address its two minor notes. Prompt-budget control
also observed RED before GREEN. Scoped rustfmt/git diff checks passed.

Publication: implementation/evidence/version commit `d9b9990` pushed to
`origin/feat/runtime-parity-profiling`; user `.gitignore` remains unstaged.
