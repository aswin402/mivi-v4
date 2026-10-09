# Batch-prefill scratch implementation plan

Approved stage 1 of the two-stage prefill experiment; existing feature branch,
in-place work, preserve unrelated `.gitignore` edits.

- [x] Add feature-gated API stub and focused correctness tests; observe RED.
- [x] Implement fallible bounded scratch ownership and pre-write validation.
- [x] Mirror established arithmetic using reusable disjoint worker buffers.
- [x] Verify bit-exact parity for six formats, batches 0/1/2/8/9/32/64/65,
      odd rows and one/two-thread partitioning; verify reuse and failure controls.
- [x] Run scoped quant batch regression tests, feature-off compilation through
      those tests, formatting and review. Cargo jobs=1, no workspace-wide commands.
- [x] Record evidence/limits, bump patch version, update sourced changelog.
- [ ] Commit and push scoped files, excluding unrelated `.gitignore`.

Stage 2 remains separate: implement an explicit four-row candidate, compare it
against both old API and scratch-only baseline with alternating fixed-work pairs.
Only a credible operator gain warrants costly model/agent and RSS validation.

## Final-version verification (v0.2.76)

- Release `cargo test -p mivi-quant --lib --release --offline --features batch-scratch-experiment batch -- --test-threads=1`: **12 passed**.
- Release `cargo test -p mivi-quant --lib --release --offline batch -- --test-threads=1`: **6 passed**, experiment disabled.
- Debug `cargo test -p mivi-quant --lib --offline --features batch-scratch-experiment batch_scratch::tests -- --test-threads=1`: **6 passed**.
- All commands used `CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2`.
- Scoped rustfmt check and `git diff --check` passed.
- GPT-6 Luna/high read-only review (Laplace) approved; no critical, important or minor findings. Reviewer ran no Cargo commands.
- Root manifest and all 14 local lockfile entries are 0.2.76; registry dependencies unchanged.
