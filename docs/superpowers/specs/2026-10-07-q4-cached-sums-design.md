# Faithful Q4 activation-sum experiment

Approved in conversation after the v0.2.73 decode attribution: evaluate one
bounded cached-sum optimization before changing ordinary inference dispatch.

The existing Q4 scalar and AVX2 kernels sum each 32-element activation segment
inside every weight row. Cache those sums once per projection in caller-owned
scratch. Preserve each route's accumulation and horizontal reduction order,
weight decoding, F32 activations, and final arithmetic expression. Scalar and
AVX2 are compared against their own baseline, not against one another.

Use the opt-in `q4-cached-sums-experiment` Cargo feature. Existing APIs and
ordinary builds remain unchanged. Allocate scratch once in RunState from
model dimensions; a per-state experiment switch defaults off. Route only Q4
FFN gate/up/down projections through the candidate. Other formats and LoRA
application retain their current path. No mutable globals, model-name rules,
new dependencies, activation quantization, or weight repacking.

Checked kernels validate all buffers and dimensions before modifying output
or scratch. Scratch is recomputed on every invocation, including successive
projections with changed inputs. Rayon workers read immutable prepared sums.

Gates: bit-exact scalar/AVX2 operator controls; scratch reuse and invalid-buffer
controls; same-model fixed-work full-logit and KV/SSM-state parity; alternating
baseline/candidate release timings with profiling disabled, one warmup pair,
three measured pairs, Cargo one job and inference two threads. Record negative
or inconclusive results. A small pilot is not enough to promote the default or
claim superiority over llama.cpp/Ollama. Broader agent-sized latency and RSS
gates in the existing CPU plan remain open.

Alternatives considered: fused gate/up work needs wider kernel changes;
packed activations change numerical behavior. Cached sums are the narrower
faithful candidate. Inspiration: Mivi's own Q4 implementation and the measured
[66.50% FFN share](../../DECODE_SUBSTAGE_EVIDENCE_2026-10-07.md). No external
inference code is copied.
