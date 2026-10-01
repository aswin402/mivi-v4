# Group-32 Activation Packing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate independent 32-value activation scales against the existing 256-value codec without changing production inference.

**Architecture:** Add a focused scalar codec/dot/matmul child module under the test-only packed experiment. Thread a typed codec selection through the existing capture and cumulative walker; projection selection remains separate from codec selection. Reuse production tile/SwiGLU APIs and existing control checks.

**Tech Stack:** Rust, existing Q4_K decoding, Rayon, existing mivi-model test dependency; no new dependencies or unsafe code.

**Approved spec:** `docs/superpowers/specs/2026-10-01-group32-activation-packing-design.md`.

## Global Constraints

- Cargo jobs: one; test threads: one; Rayon/inference threads: two.
- Scope all checks/builds/tests to the necessary package/test filters. Never run a full-workspace check, build, or test, or rebuild the server unnecessarily.
- No new unsafe code or normal dependencies. Preserve the user's `.gitignore`.
- All new modules remain descendants of the existing test-only `packed_prefill` module. Do not add production exports or GGUF wire formats.
- Keep GGUF weight bytes unchanged. Group size follows Q4_K format layout, never model names, layer indices, dimensions, or token IDs.
- Group-256 remains the default. `MIVI_TEST_ACTIVATION_GROUP` accepts exactly `256` or `32`; malformed settings fail, rather than silently choosing a default.
- Retain non-packed projections, skipped SSM behavior, complete-row residual deltas, metadata BOS, finite metrics, ordered observations, final-state agreement, and the `1e-3` production-control tolerance.
- No packed-model quality threshold, production rollout, timing improvement, generated-agent success, or causal component-defect claim.
- Complete the experiment, review, and verify before incrementing 0.2.59 to 0.2.60 once. Changelog includes actual measurements, ideas, inspirations, and sources. Commit only task files and push without force under standing authorization.
- Use `apply_patch` for edits. Formatting and Cargo's lockfile refresh are mechanical exceptions. No destructive cleanup.

## File responsibilities and interfaces

| File | Responsibility |
|------|----------------|
| `crates/mivi-quant/src/q4_k_m/packed_prefill/group32.rs` (new) | Activation groups, scalar dot, checked batched matmul, model-free tests |
| `crates/mivi-quant/src/q4_k_m/packed_prefill.rs` | Register child module; private typed codec dispatch/error variants |
| `crates/mivi-quant/src/q4_k_m/packed_prefill/real_weights.rs` | Make the private selector visible to existing diagnostic descendants |
| `crates/mivi-quant/src/q4_k_m/packed_prefill/real_weights/captured_activations.rs` | Pass selector through replay/walker; retain existing behavior by default |
| `crates/mivi-quant/src/q4_k_m/packed_prefill/real_weights/captured_activations/cumulative.rs` | Selected-codec projection dispatch, while retaining independent FFN masks |
| `crates/mivi-quant/src/q4_k_m/packed_prefill/real_weights/captured_activations/layer_trace.rs` | Selected-codec activation statistics and labelled layer/logit comparisons |
| `README.md`, `CHANGELOG.md`, `Cargo.toml`, `Cargo.lock` | Invocation, results/caveats/sources, completed-feature patch increment |

The new module's interfaces are private to the test harness:

```rust
pub(super) const GROUP_WIDTH: usize = 32;
pub(super) const GROUPS: usize = Q4_K_BLOCK_SIZE / GROUP_WIDTH;
pub(super) struct Group32Activation {
    scales: [f32; GROUPS],
    values: [i8; Q4_K_BLOCK_SIZE],
    sums: [i16; GROUPS],
}
// Methods: pack(&[f32]) -> Result<Self, PackedError>;
// reconstructed(&self) -> [f32; Q4_K_BLOCK_SIZE];
// dot(&self, &PackedWeights<'_>) -> f32.
// Wrapper: packed_matmul(out: &mut [f32], weights: &[u8], inputs: &[f32],
//                       batch: usize, rows: usize, cols: usize)
//          -> Result<(), PackedError>.
```

No group-32 SIMD implementation is included. Group-256 dispatch continues to use `PackedKernel::detected_tiled()`.

---

## Task 1: Group codec and independent scalar dot

**Files:** Create `crates/mivi-quant/src/q4_k_m/packed_prefill/group32.rs`; modify `crates/mivi-quant/src/q4_k_m/packed_prefill.rs` to add `mod group32;`.

**Consumes:** Existing `PackedError`, `PackedActivation`, `PackedWeights`, Q4_K constants, `fixture_weights`, and `dequantize_q4_k_m` from the parent.

**Produces:** `Group32Activation::{pack,reconstructed,dot}` and format constants above.

- [x] Write the tests first, with zero-output codec/dot stubs so they compile and fail assertions. Include the following distinguishing fixture:

```rust
#[test]
fn group32_pack_limits_cross_group_outlier_error() {
    let mut input = [0.25f32; Q4_K_BLOCK_SIZE];
    input[0] = 12.0;
    let grouped = Group32Activation::pack(&input).unwrap().reconstructed();
    let old = PackedActivation::pack(&input).unwrap();
    let old_error: f64 = input.iter().zip(&old.values)
        .map(|(&x, &q)| (f64::from(x) - f64::from(f32::from(q) * old.scale)).powi(2))
        .sum();
    let new_error: f64 = input.iter().zip(grouped)
        .map(|(&x, y)| (f64::from(x) - f64::from(y)).powi(2)).sum();
    assert!(new_error < old_error);
    assert!(grouped[GROUP_WIDTH..].iter().all(|&x| x == 0.25));
}
```

- [x] Add codec tests for zeros, opposite signed maxima in separate groups, rounding ties, tiny finite inputs, malformed lengths, NaN/infinity, and scale underflow. Tie fixture: set each group maximum to `-127.0`, then values `0.5, 1.5, -0.5, -1.5`; expect integers `0, 2, 0, -2` with scale `1.0`. Underflow fixture: `[f32::from_bits(1); Q4_K_BLOCK_SIZE]` must return `ScaleUnderflow`. A block of all zeros must reconstruct zeros with all sums/scales zero.
- [x] Run `RAYON_NUM_THREADS=2 cargo test -p mivi-quant --offline --lib --jobs 1 group32 -- --test-threads=1`. Confirm assertion failures caused by the stubs, not import/type mistakes.
- [x] Implement the codec using per-group signed maximum and the original rounding convention:

```rust
pub(super) fn pack(input: &[f32]) -> Result<Self, PackedError> {
    if input.len() != Q4_K_BLOCK_SIZE {
        return Err(crate::QuantError::BufferTooSmall {
            expected: Q4_K_BLOCK_SIZE, actual: input.len(),
        }.into());
    }
    if input.iter().any(|x| !x.is_finite()) {
        return Err(PackedError::NonFinite);
    }
    let mut packed = Self {
        scales: [0.0; GROUPS], values: [0; Q4_K_BLOCK_SIZE], sums: [0; GROUPS],
    };
    for (g, group) in input.chunks_exact(GROUP_WIDTH).enumerate() {
        let maximum = group.iter().copied().fold(0.0f32, |m, x| {
            if x.abs() > m.abs() { x } else { m }
        });
        if maximum == 0.0 { continue; }
        packed.scales[g] = maximum / -127.0;
        if packed.scales[g] == 0.0 { return Err(PackedError::ScaleUnderflow); }
        for (i, &value) in group.iter().enumerate() {
            packed.values[g * GROUP_WIDTH + i] = ((value / maximum) * -127.0)
                .round_ties_even().clamp(-127.0, 127.0) as i8;
        }
        packed.sums[g] = packed.values[g * GROUP_WIDTH..(g + 1) * GROUP_WIDTH]
            .iter().map(|&x| i16::from(x)).sum();
    }
    Ok(packed)
}

pub(super) fn reconstructed(&self) -> [f32; Q4_K_BLOCK_SIZE] {
    std::array::from_fn(|i| f32::from(self.values[i]) * self.scales[i / GROUP_WIDTH])
}
```

Add a compile-time assertion that the Q4_K block is divisible by `GROUP_WIDTH` and `GROUPS == 8`. These constants describe Q4_K, not model dimensions.

- [x] Write a failing dot test before the dot body. Decode fixture weights independently with `dequantize_q4_k_m`; reconstruct activations with `reconstructed`; accumulate `sum(f64(weight) * f64(activation))`. Run seeds `0, 1, 47, 255` and signs/zeros/outlier groups. Reject a zero dot stub with a known nonzero fixture.
- [x] Implement the scalar dot, retaining each group's own activation scale and affine minimum correction:

```rust
pub(super) fn dot(&self, weight: &PackedWeights<'_>) -> f32 {
    let mut result = 0.0f32;
    for g in 0..GROUPS {
        let bytes = &weight.quants[(g / 2) * GROUP_WIDTH..(g / 2 + 1) * GROUP_WIDTH];
        let values = &self.values[g * GROUP_WIDTH..(g + 1) * GROUP_WIDTH];
        let integer_dot: i32 = bytes.iter().zip(values).map(|(&byte, &x)| {
            let q = if g % 2 == 0 { byte & 15 } else { byte >> 4 };
            i32::from(q) * i32::from(x)
        }).sum();
        let term = self.scales[g] * (
            weight.scale * weight.scales[g] as f32 * integer_dot as f32
            - weight.min_scale * weight.mins[g] as f32 * f32::from(self.sums[g])
        );
        result += term;
    }
    result
}
```

- [x] Test a rounding-aware bound rather than universal absolute tolerance. Compute the F64 sum of absolute affine components using unpacked nibbles, weight scale/min/submetadata, group scale, and signed activation magnitudes. Use `gamma = 64*eps/(1-64*eps)` for the documented conservative F32 operation budget; permit `gamma * component_energy + 64.0 * f64::from(f32::from_bits(1))` for arithmetic rounding and subnormal absolute error. Separately quantify original-activation packing error with per-group `0.501 * abs(scale)` plus F32 reconstruction allowance. Never turn this arithmetic bound into a model-quality threshold.
- [x] Rerun the scoped group tests; require zero failures. Obtain an independent read-only review of codec grouping, nibble mapping, sums, minimum correction, and reference/bound independence. Fix substantive findings with failing regressions.
- [x] Commit only the new module and registration with `git commit -m 'test: add group-32 activation codec and scalar dot'` after explicit staging. No patch release yet.

## Task 2: Checked batched matmul

**Files:** Modify `crates/mivi-quant/src/q4_k_m/packed_prefill/group32.rs` and test-only `PackedError` in `crates/mivi-quant/src/q4_k_m/packed_prefill.rs`.

**Consumes:** Task 1 interfaces, `crate::validate_matmul_args`, `PackedWeights::new`, existing Rayon pool/threshold.

**Produces:** `group32::packed_matmul` with the six-argument signature defined above, transactional output writes, unchanged output tails.

- [x] Write failing matmul tests against a zero-output stub. Use `(rows,batch,cols)` of `(3,2,512)`, `(1,1,256)`, `(257,3,256)`, and zero-work shapes `(0,3,256)`, `(3,0,256)`, `(3,2,0)`. Fill outputs with a sentinel plus one tail value. Independent expectations decode each full weight row and reconstructed activation block, and accumulate in F64. Test all-zero input produces zero outputs while the tail stays sentinel.
- [x] Write rejection-before-mutation tests using the concrete pattern:

```rust
#[test]
fn group32_matmul_rejects_before_output_write() {
    let weights = fixture_weights(3, 512, 47);
    let mut input = vec![0.25; 2 * 512];
    input[512 + 33] = f32::NAN;
    let mut out = [123.0; 7];
    assert!(packed_matmul(&mut out, &weights, &input, 2, 3, 512).is_err());
    assert_eq!(out, [123.0; 7]);
}
```

Add short output/weights/input, misaligned columns, checked-product overflow, late-block infinity/underflow, non-finite decoded weight scale/min, and non-finite derived output cases. Add private `PackedError::NonFiniteWeight` (message: `packed weights require finite scales`) and `PackedError::NonFiniteOutput` (message: `packed projection produced non-finite output`) variants; do not mislabel them as non-finite activations.

- [x] Run the exact Task 1 scoped test command and observe assertion failures for missing matmul behavior.
- [x] Implement the checked wrapper. Calculate `blocks`, checked `row_bytes`, then call `validate_matmul_args`. Return after validation for zero batch/rows. For nonzero work, validate only the used weight prefix's scale/min headers, pack only the used input prefix, and compute into a temporary row-major result. This guarantees invalid data never writes public output.
- [x] Use the following row kernel and existing Rayon scheduling pattern:

```rust
let compute = |row: usize, output: &mut [f32]| {
    for block in 0..blocks {
        let start = row * row_bytes + block * Q4_K_BYTES;
        let weight = PackedWeights::new(&weights[start..start + Q4_K_BYTES]);
        for (b, value) in output.iter_mut().enumerate() {
            *value += activations[b * blocks + block].dot(&weight);
        }
    }
};
if rows >= crate::RAYON_PARALLEL_THRESHOLD && rayon::current_num_threads() > 1 {
    row_major.par_chunks_mut(batch).enumerate()
        .for_each(|(row, output)| compute(row, output));
} else {
    row_major.chunks_mut(batch).enumerate()
        .for_each(|(row, output)| compute(row, output));
}
```

All sizes used in allocation/indexing must have passed checked products in the validator; retain explicit checked `rows * row_bytes` for the used prefix. With `cols == 0`, zero-filled temporary outputs are committed normally. Reject non-finite temporary results before this final conversion:

```rust
for row in 0..rows {
    for b in 0..batch {
        out[b * rows + row] = row_major[row * batch + b];
    }
}
```

- [x] Rerun group tests and `RAYON_NUM_THREADS=2 cargo test -p mivi-quant --offline --lib --jobs 1 -- --test-threads=1`. Request independent review of validation, checked offsets, transactional writes, prefixes/tails, and parallel result layout.
- [x] Commit only Task 2 files with `git commit -m 'test: add checked group-32 batched projection'`. No patch release yet.

## Task 3: Private codec dispatch and diagnostic integration

**Files:** Modify `crates/mivi-quant/src/q4_k_m/packed_prefill.rs`, `crates/mivi-quant/src/q4_k_m/packed_prefill/real_weights.rs`, `crates/mivi-quant/src/q4_k_m/packed_prefill/real_weights/captured_activations.rs`, and its `cumulative.rs`/`layer_trace.rs` children.

**Consumes:** Task 2 matmul and Task 1 reconstruction; current `WalkMode`, `FfnPacking`, `Projection`, coverage, and control helpers.

**Produces:** Private `ActivationCodec::{Group256,Group32}` with `parse`, `from_env`, `label`, `project`, and `reconstruct`; typed selector passed through diagnostic functions without reading environment variables inside numerical loops.

- [ ] Write failing parser/dispatcher tests before the selector implementation: (Original chronology was not met; user approved the disclosed retrospective-check deviation.)

```rust
#[test]
fn activation_codec_rejects_unsupported_settings() {
    assert_eq!(ActivationCodec::parse("256").unwrap(), ActivationCodec::Group256);
    assert_eq!(ActivationCodec::parse("32").unwrap(), ActivationCodec::Group32);
    for text in ["", "0", "16", "64", "128", "bad", " 32", "32 "] {
        assert!(ActivationCodec::parse(text).is_err());
    }
}
```

Use direct parsing, not environment mutation, in unit tests. Dispatch tests compare actual matrices against the corresponding existing/new wrapper; reconstruction tests compare actual arrays against each codec's packing result. Both codecs must preserve nonzero F32 fallback and one-packed/two-non-packed gate/up isolation outputs.

- [x] Implement the private selector with these exact interfaces:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ActivationCodec { Group256, Group32 }
impl ActivationCodec {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "256" => Ok(Self::Group256), "32" => Ok(Self::Group32),
            _ => Err(format!("unsupported MIVI_TEST_ACTIVATION_GROUP={value:?}; use 256 or 32")),
        }
    }
    fn from_env() -> Result<Self, Box<dyn std::error::Error>> {
        match std::env::var("MIVI_TEST_ACTIVATION_GROUP") {
            Ok(value) => Self::parse(&value).map_err(Into::into),
            Err(std::env::VarError::NotPresent) => Ok(Self::Group256),
            Err(error) => Err(error.into()),
        }
    }
    fn label(self) -> String {
        match self {
            Self::Group256 => format!("group=256, kernel={}", PackedKernel::detected_tiled().label()),
            Self::Group32 => "group=32, kernel=scalar".to_owned(),
        }
    }
    fn project(self, out: &mut [f32], weights: &[u8], input: &[f32],
               batch: usize, rows: usize, cols: usize) -> Result<(), PackedError> {
        match self {
            Self::Group256 => packed_matmul_with_kernel(out, weights, input, batch, rows, cols,
                                                       PackedKernel::detected_tiled()),
            Self::Group32 => group32::packed_matmul(out, weights, input, batch, rows, cols),
        }
    }
    fn reconstruct(self, input: &[f32]) -> Result<[f32; Q4_K_BLOCK_SIZE], PackedError> {
        match self {
            Self::Group256 => {
                let packed = PackedActivation::pack(input)?;
                Ok(std::array::from_fn(|i| f32::from(packed.values[i]) * packed.scale))
            }
            Self::Group32 => Ok(group32::Group32Activation::pack(input)?.reconstructed()),
        }
    }
}
```

- [x] Add `codec: ActivationCodec` to `walk`, `walk_observed`, `replay`, cumulative `project`, public-to-parent `recompute`, `FfnProjections::recompute`, and `activation_stats`. Read `ActivationCodec::from_env()` once at each ignored diagnostic entry point. Existing unit tests pass `Group256` explicitly unless looping over both codecs. Preserve separate `packed: bool` in `project` and `FfnPacking` masks; replace only its eligible branch with `codec.project(...)`.

```rust
if eligible {
    codec.project(out, projection.weights, input, batch, projection.rows, projection.cols)?;
} else {
    super::quantized_matmul_rows(out, projection.kind, projection.weights, input,
                                batch, projection.rows, projection.cols)?;
}
```

The existing input/output finiteness checks and `Ok(eligible)` stay outside this branch. Codec selection must not override the `eligible = packed && supported_shape(...)` mask.
- [x] In replay, keep the existing group-256 scalar/tiled equality test and error-bound check intact. When group-32 is selected, compute its sample alongside the existing F32-activation reference, print finite relative-L2/max-absolute errors with codec/kernel labels, and apply the group-specific reconstruction/rounding bound. Do not apply the old single-scale bound to new groups. Keep row caps explicit.
- [x] In layer statistics, replace the old single-scale reconstruction with `codec.reconstruct(block)?`; leave reference/error/RMS accumulation unchanged. Label all scalar baseline statistics and mode summaries with the codec. They still include fallback inputs as probes, not claims that those projections packed activations.

```rust
for block in input.chunks_exact(Q4_K_BLOCK_SIZE) {
    let reconstructed = codec.reconstruct(block)?;
    for (&original, reconstructed) in block.iter().zip(reconstructed) {
        let value = f64::from(original);
        let reconstructed = f64::from(reconstructed);
        reference_squared += value * value;
        error_squared += (reconstructed - value).powi(2);
        peak = peak.max(value.abs());
    }
}
```
- [x] Preserve `SingleDown` by routing its packed projection through the selector. Controls use no packed projection under either codec. Update corpus calls to accept the selector without expanding its default fixtures or forcing that corpus to run in this experiment.
- [x] Extend the existing two-row gate/up and mixed-format tests to loop over both codecs, with selected-codec expected outputs, unchanged coverage, fresh SwiGLU, invalid isolation rejection, and preserved down/control behavior. Run targeted selector/cumulative/stat tests, then the scoped quant library tests. Request read-only review of selector propagation and labels before model runs.
- [x] Commit Task 3 files with `git commit -m 'test: compare activation codecs in captured FFN diagnostics'`.

## Task 4: Bounded real-model evidence and completed release

**Files:** `README.md`, `CHANGELOG.md`, `Cargo.toml`, `Cargo.lock`, this plan's execution checkboxes. Source edits only if a failing verification requires a fix, with regression and re-review.

**Consumes:** Task 3 selector and existing ignored diagnostic names.

**Produces:** Recorded projection samples and complete-matrix short-prompt model comparisons; reviewed v0.2.60 publication, or an explicit incomplete status if verification fails.

- [x] Begin with one local GGUF and group-32 captured projections, one token tile, capped projection rows. Exact initial command from workspace root:

```bash
MIVI_TEST_MODEL=/home/aswin/programming/vscode/myProjects/ai_agent_tools/mivi_v4/models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf \
MIVI_TEST_ACTIVATION_GROUP=32 MIVI_TEST_CAPTURE_TOKENS=32 MIVI_TEST_MAX_ROWS=128 \
MIVI_THREADS=2 RAYON_NUM_THREADS=2 \
cargo test -p mivi-quant --offline --release --lib --jobs 1 \
  captured_prefill_projection_evaluation -- --ignored --test-threads=1 --nocapture
```

Expect successful finite samples and a production walker control below `1e-3`. The ignored capture test also runs an existing single-down sensitivity case on complete rows; label its result separately from capped projection samples.

- [x] Run the short default tool-request layer trace with each codec on the first GGUF:

```bash
MIVI_TEST_MODEL=/home/aswin/programming/vscode/myProjects/ai_agent_tools/mivi_v4/models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf \
MIVI_TEST_ACTIVATION_GROUP=256 MIVI_THREADS=2 RAYON_NUM_THREADS=2 \
cargo test -p mivi-quant --offline --release --lib --jobs 1 \
  layerwise_prefill_error_trace -- --ignored --test-threads=1 --nocapture

MIVI_TEST_MODEL=/home/aswin/programming/vscode/myProjects/ai_agent_tools/mivi_v4/models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf \
MIVI_TEST_ACTIVATION_GROUP=32 MIVI_THREADS=2 RAYON_NUM_THREADS=2 \
cargo test -p mivi-quant --offline --release --lib --jobs 1 \
  layerwise_prefill_error_trace -- --ignored --test-threads=1 --nocapture
```

- [x] Repeat the two commands sequentially with the path ending `models/LFM2.5-2.6B-Q4_K_M.gguf`. Never run two model loads or Cargo commands concurrently. Record eight codec/mode cases per model; repeated samples are not independent fixtures. Controls, coverage, observer ordering, finite metrics, and final state must pass. Lower/higher packed errors are measurements, not test acceptance thresholds.
- [x] Record original/new relative L2, maximum absolute logit error, greedy token changes, residual growth, controls, packed/non-packed coverage, raw token counts, row caps, scalar/actual-runtime kernel labels, and resource settings. Do not infer a speed improvement from these accuracy runs or promise generated-tool correctness.
- [ ] Independently review changes against the approved spec, fixing important findings via failing regression tests. Self-review documentation for placeholder measurements, contradictory codec/kernel labels, stale selector behavior, and unsupported quality claims.
- [x] After successful verification, change workspace version to `0.2.60`; let the next scoped Cargo command refresh workspace package versions in `Cargo.lock`. No other dependency changes. Add a changelog entry with current completion date, actual measurements, limitations, and the GGML/LLM.int8 sources from the spec. README documents `MIVI_TEST_ACTIVATION_GROUP`, scalar limitations, and focused commands. Do not overwrite prior measurements or claim the server binary was rebuilt.
- [x] Run final scoped verification (one Cargo command at a time):

```bash
RAYON_NUM_THREADS=2 cargo test -p mivi-quant --offline --lib --jobs 1 -- --test-threads=1
RAYON_NUM_THREADS=2 cargo clippy -p mivi-quant --offline --lib --tests --jobs 1 -- \
  -D warnings -A clippy::manual_div_ceil -A clippy::manual_is_multiple_of \
  -A clippy::items_after_test_module -A clippy::field_reassign_with_default
cargo fmt -p mivi-quant -- --check
cargo tree -p mivi-quant --offline -e normal --depth 1
git diff --check
```

Expect zero failed non-ignored library tests, successful scoped Clippy with the existing style allowances, clean formatting/whitespace, and unchanged normal dependencies. Model results from unchanged code remain valid after a metadata-only version bump; rerun model tests if numerical or selector code changes during review.

- [ ] Mark completed steps, commit task files explicitly excluding `.gitignore`, push the branch requested by the user without force, and verify local/pushed heads agree. Report measured outcome, unchanged production inference, commit/version, and remaining quality/performance work. If publishing from main, obtain explicit main-branch implementation approval or use an approved isolated branch/worktree.

## Plan self-review

- Codec grouping, signed rounding, rejection, reconstruction, scalar affine dot, arithmetic references: Task 1.
- Shapes/overflow, input/weight prefixes, transactional writes/tails, existing Rayon pool: Task 2.
- Private environment/type dispatch, every diagnostic path, masks/fallbacks, codec-labelled statistics, existing controls: Task 3.
- Capped samples, two-model one-prompt comparisons, sequential resources, independent review, release/docs/sources: Task 4.
- SIMD optimization, full corpus expansion, generated agent answers, and production deployment are intentionally outside this experiment.

Execution status: Tasks 1 and 2 implemented and independently reviewed; Task 3 functionally reviewed, with its disclosed original test-first chronology deviation approved by the user. Retrospective wrong-selector mutation checks failed as expected and restoration passed. Task 4 completed five sequential model runs and scoped verification; workspace version is 0.2.60. Final release reviews and GitHub publication are pending. Production inference remains unchanged.
