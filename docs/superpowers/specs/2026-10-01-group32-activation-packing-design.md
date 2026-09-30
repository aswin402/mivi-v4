# Group-32 Activation Packing Experiment

Status: written specification approved by the user; implementation pending.

## Goal and evidence

Determine whether a separate activation scale for each Q4_K weight sub-block
reduces the experimental packing error seen in v0.2.57–v0.2.59. This is an
accuracy-first, test-only experiment, not a production optimization or an
explanation of Minicode latency.

The existing codec shares one signed scale across 256 activation values.
Gate-only/up-only sensitivity depends on both model and prompt; neither
projection can be excluded universally based on the measured results.

## Approach and alternatives

Use eight independently scaled 32-value activation groups within each existing
256-value Q4_K block. The grouping follows Q4_K's eight weight sub-blocks, not
a model name, hidden dimension, layer number, or token ID. GGUF weights and
their quantization remain unchanged. This is a private in-memory codec, not a
new GGUF format or a bit-identical GGML Q8 serialization implementation.

Compared with the existing codec, this adds scale storage and floating-point
combination work. Smaller groups limit how many values share an extreme value's
scale; they do not guarantee lower final logit error or better agent answers.

An alternative is higher-precision correction for selected outlier dimensions.
That requires selection rules and an additional dot-product path. Defer it until
the simpler grouping experiment produces evidence. No clipping, calibration,
outlier threshold, automatic policy selection, or model-specific exclusions
are part of this design.

## Components and arithmetic

1. Add a focused child module under the existing test-only `packed_prefill`
   module. Keep the current 256-value codec and its SIMD/tiled kernels intact.
2. Represent a group-32 activation block with 256 signed 8-bit values, eight
   F32 scales, and eight signed integer group sums. Derive group counts from
   named Q4_K format constants and assert the expected divisibility.
3. For each 32-value group, choose the signed maximum-absolute input, scale by
   `maximum / -127`, and round normalized values to nearest-even in
   `[-127, 127]`, matching the existing experimental quantizer convention.
   An all-zero group has zero scale/values/sum. Reject non-finite inputs and
   nonzero groups whose scale underflows. This experiment does not introduce
   a silent fallback for those errors.
4. First implement a scalar Q4_K dot product. Decode the existing low/high
   nibble mapping and scale/min metadata through the existing checked path.
   For each weight sub-block, compute integer dot and activation sum, then its
   floating contribution:

   `activation_scale[g] * (weight_scale * weight_subscale[g] * integer_dot[g]
       - weight_min_scale * weight_submin[g] * activation_sum[g])`.

   Sum the eight floating contributions in a documented, deterministic order.
   Do not combine differently scaled groups into one integer accumulator.
   Include the affine minimum correction for every group.
5. Add a checked batched matmul wrapper that packs inputs once per invocation
   and reuses them across weight rows. Validate shapes, lengths, overflow,
   weights, and activation finiteness before output writes. Handle zero-work
   dimensions consistently with the existing wrapper and preserve output tails.
   Permit row parallelism through the existing Rayon pool; do not add a pool.
6. Add a private typed codec selector to the diagnostic projection dispatcher.
   The existing path remains the default. An opt-in diagnostic environment
   setting `MIVI_TEST_ACTIVATION_GROUP` selects only `256` or `32`; reject other
   values. Record the selected codec and actual kernel in result labels.
   Unsupported weight formats retain
   the existing F32-activation path, and deliberately unselected projections
   remain non-packed even when their weights are Q4_K.

No production API, inference configuration, or serialization changes are needed.
Runtime SIMD optimization of the new codec is deferred until numerical testing.
Do not report a scalar prototype as a production performance improvement.

## Verification and experiment boundaries

Use failing tests before implementation and independent decoded-weight
references, not only agreement between two implementations sharing the codec.

- Codec cases: zeros, signs, ties, extremes in different groups, tiny finite
  inputs, malformed lengths, NaN/infinity, and scale underflow. Reconstruction
  error is checked against each group's own rounding bound. Include a fixture
  where a large value in one group degrades the old codec's other groups;
  improvement on this fixture is not a general quality guarantee.
- Dot cases: decode Q4_K weights independently, reconstruct group-32 activation
  values, and accumulate an F64 reference. Use a documented rounding-aware
  arithmetic bound, including cancellation and affine minimum terms. Also
  compare against the original F32 activations to quantify packing error.
- Matmul cases: multiple weight rows, token rows, and Q4_K blocks; zero-work
  shapes, tail preservation, malformed buffers, and rejection before mutation.
- Selection cases: one packed/two non-packed projections for gate-only/up-only,
  mixed-format fallback, and preserved down/full/control behavior. Recompute
  both input projections and public `swiglu_rows` for gate/up isolation.
- Start real-weight validation with the existing bounded captured-projection
  harness. Label row-capped measurements as projection samples, not full-model
  or timing results.
- Then run the short raw tool-request fixture on the two local Q4_K_M GGUFs
  measured in v0.2.59, using their paths as test inputs, not code switches,
  comparing group-256 and group-32 packing for down, gate, up, and full FFN.
  Each codec uses complete matrices/all token rows for cumulative measurements.
  Keep the existing at-most-64-token limit and reject truncation/blank prompts.
  Load models sequentially, not concurrently.
- Require production/walker and non-packed recompute controls to agree within
  the existing max-absolute logit tolerance of `1e-3`. Require finite derived
  metrics, correct coverage, ordered observations, and final-state agreement.
  Report final-logit relative L2/max-absolute error, greedy next-token changes,
  and residual growth. This tolerance is a harness control, not packed quality
  acceptance. Existing residual-delta rounding limitations remain disclosed.

The initial experiment does not rerun every model/fixture combination by
default. Expansion to the three-fixture corpus, generated tool-call evaluation,
and Minicode testing are subsequent evidence stages, not success claims here.
No result automatically enables packing in production.

## Resource and publication constraints

- Cargo jobs: one; test threads: one; Rayon/inference threads: two.
- Scope all checks/builds/tests to the necessary package/test filters. Never
  run a full-workspace check, build, or test, or rebuild the server unnecessarily.
- No new unsafe code or normal dependencies. Preserve the user's `.gitignore`.
- Measure performance separately from correctness if a later optimized kernel
  is implemented; distinguish packing, kernel, and end-to-end inference costs.
- Once the experiment is implemented and verified, increment the patch version
  once, update README/changelog with actual results, caveats, ideas, inspirations,
  and sources, and push the reviewed task changes under standing authorization.
  This design-only commit does not constitute a completed feature release.

## Sources and inspiration

- Mivi v0.2.57–v0.2.59 cumulative and layer-wise diagnostics; existing
  `packed_prefill.rs`, checked quantized matmul, tile, and public SwiGLU APIs.
- [GGML block definitions](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h):
  Q4_K comprises eight 32-value weight groups.
- [GGML quantization reference](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c):
  quantized weight decoding and affine scale/min correction.
- [Dettmers et al., LLM.int8()](https://arxiv.org/abs/2208.07339): inspiration for
  investigating precision sensitivity. This design does not implement that
  paper's mixed-precision outlier decomposition or claim its quality results.
