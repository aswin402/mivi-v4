# Four-row isolation and captured-activation evidence — 2026-10-10

## Outcome

Both bounded diagnostics passed. The candidate remains default-off, with no model
integration or runtime behavior change. Captured Q4_K batch 32 regressed **11.61%**
against scratch-only in the paired median, with all six measured ratios above one.
Captured Q4_K/Q6_K batch 64 were respectively **8.14%/15.92% lower time** against
scratch-only, with all six ratios below one. Q6_K batch 32 was mixed.

The accumulation-only pilot supports investigating shape-dependent costs, but does
not establish a cache/broadcast/spill root cause. Generated four-row groups are not
a full matrix working set. No end-to-end agent latency, model-state candidate
parity, general superiority over other engines or production promotion is claimed.

## Environment and origin gates

AMD Ryzen 7 7730U, AVX2/FMA host, rustc 1.99.0. Cargo jobs=1, Rayon threads=2,
serial test harness, release. No second Cargo invocation ran during live timing.
Diagnostics measured pre-publication v0.2.77 plus this test-only addition.
No core/model kernel or new unsafe code changed.

Existing local public LFM2.5-1.2B-Instruct-Q4_K_M.gguf:
SHA256 `b1b3de114215d9507409a662a501a631095a479a419584e8a2ded6304b19b4f5`.
Context 65, F32 KV, no active adapters; 32/64 effective prompt IDs after metadata
BOS policy, one chunked tile per case. Default public Rust-workspace prompt is
in the test source; caller prompt override stays in memory and is limited to 64KiB
of UTF-8 before tokenization. No prompts, token IDs,
weight contents or activation values are logged/persisted. Metadata/shapes are logged.

A reset baseline walker uses production attention/SSM tile functions, retains only
executed FFN gate/up norm inputs and post-SwiGLU down inputs, and computes the
production final norm/head. **All 65,536 final logits matched production chunked
prefill bits and were finite at both batch sizes before captured candidate timing.**
The candidates never feed back into inference. This proves origin fidelity for this
one-tile replay, not candidate-integrated intermediate/final model-state parity.
There is no intermediate-logit or KV/SSM state comparison claim.

First executed eligible Q4_K and Q6_K FFN projections are selected by wire format
and dimensions, not model name. Eligibility is nonzero columns <=16,384 aligned256,
nonzero rows <=8,192; accepted matrices are complete, not row-capped. Both are
layer0: gate Q4_K 8,192x2,048 and down Q6_K 2,048x8,192.

## Timing contracts

Isolation: four generated decoded F32 rows, generated column-major activations;
no allocation/decode/transpose in timers. Timers include output zeroing and 200
calls of either two established pair helpers or the four-row wrapper. One warmup
pair, six alternating measured pairs (three each order), six shapes. 16,800 timed
four-row-group iterations: 8,400 four-row wrapper calls and 16,800 pair-helper calls,
plus 42 complete finite-output bit comparisons. Passed
in 0.64s. Declared warm-small-group cache; no cache flush/pinning/hardware-counter
measurement. Inputs are prepared directly in transposed layout, not timed transpose.

Captured replay: ordinary=B, scratch-only=S, four-row=F. Each full projection timer
includes validation, transpose, weight decode, accumulation and output layout;
ordinary allocates internal temporaries per call. Scratch construction, capture,
reference and bit checks are outside timers. Four cases, one warmup plus six
measured triples, all six order permutations, three calls/member. **252 timed full
projections, 84 member-final complete output checks, four untimed references**.
All final outputs matched complete finite reference bits. Passed in 10.57s.
Only final outputs of three-call members are checked, not every repeated call.

Warm process/mapped model pages and resident per-case scratch; no OS page-cache drop,
no engine prefix reuse (no 256-token snapshot boundary is reached), no simulated
HTTP/client/agent work. Host background load, frequency and thermal conditions are
not controlled. Six pairs/triples are a pilot, not statistical confidence.

## Summary

Ratios are medians of six within-pair/triple ratios, **not** ratios of separate
medians. Round0 warmup excluded; lower than one means lower wall time. Isolation
compares four-row against two pair calls; capture retains the scratch-only control.

| Isolation cols | Batch | Four/pair median | Range |
|---:|---:|---:|---:|
| 2048 | 32 | 1.089730 | 0.773514–1.679686 |
| 2048 | 64 | 0.875060 | 0.836433–0.918262 |
| 2048 | 65 | 0.821980 | 0.814373–0.878191 |
| 8192 | 32 | 0.961949 | 0.786786–1.023038 |
| 8192 | 64 | 0.848151 | 0.816654–0.925395 |
| 8192 | 65 | 0.846551 | 0.766773–0.910639 |

| Captured format | Batch | F/B median | F/S median | F/S range |
|---|---:|---:|---:|---:|
| Q4_K | 32 | 1.002991 | 1.116057 | 1.054417–1.228232 |
| Q6_K | 32 | 0.947983 | 0.952664 | 0.889632–1.038460 |
| Q4_K | 64 | 0.943247 | 0.918635 | 0.865853–0.973754 |
| Q6_K | 64 | 0.832298 | 0.840792 | 0.760144–0.869051 |

The generated 2,048-column/batch32 accumulation case is noisy (range0.773514–1.679686);
it cannot alone explain the captured Q4 regression. All four larger-batch isolation
cases favor four-row in these six pairs. Q6 captured batch32 includes a regression
(1.038460); all results including negatives/warmups are retained below. Q4 batch64
differs from the prior generated-activation full-projection pilot, where it was
mixed. Workload/host variability is not attributed to a confirmed mechanism.

## Reproduction and verification

Use an **absolute model path**: Cargo test executes from the crate directory.
An initial relative-path invocation failed with GGUF Io/NotFound before capture;
the absolute-path retry passed. No exact parity gate failed or was relaxed.
The initial control stubs did fail two valid-case assertions (RED), then passed
after implementation. A compiler error using nonexistent ActiveAdapters::is_empty
was corrected to its public active-vector check; it is not a runtime defect.

```sh
CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 timeout 180s \
cargo test -p mivi-quant --lib --release --offline \
  --features batch-four-row-experiment four_row_accumulation_isolation \
  -- --ignored --test-threads=1 --nocapture

CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 \
MIVI_TEST_MODEL=/absolute/path/to/local.gguf timeout 240s \
cargo test -p mivi-quant --lib --release --offline \
  --features batch-four-row-experiment four_row_captured_activation \
  -- --ignored --test-threads=1 --nocapture
```

Final-version v0.2.78 release batch regressions passed 21 tests (three diagnostics
intentionally ignored); feature-off release batch controls passed six tests. The
debug exact-gate/shape/prompt-budget controls passed three tests (two diagnostics ignored).
GPT-6 Luna/high read-only review found no Critical/Important issues; minor operation
count wording and caller prompt tokenization budget were addressed. Prompt-budget
boundary test was observed RED before implementation and passed debug/release checks.
Luna/high approved the follow-up byte guard and count clarification; timing and
arithmetic were unchanged from the recorded live runs.
Scoped rustfmt and git diff checks pass; no workspace-wide Cargo check/build/test.

## Complete raw isolation timings

Original integer nanosecond totals for 200 calls/member; route0=two pair helpers,
route1=four-row. Round0 warmup, rounds1–6 measured. No timings removed.

```text
ISOLATION cols=2048 batch=32 round=0 order=[1, 0] times_ns=[1532322, 1674411]
ISOLATION cols=2048 batch=32 round=1 order=[0, 1] times_ns=[1571466, 1273292]
ISOLATION cols=2048 batch=32 round=2 order=[1, 0] times_ns=[1645336, 1272690]
ISOLATION cols=2048 batch=32 round=3 order=[0, 1] times_ns=[1250208, 1245028]
ISOLATION cols=2048 batch=32 round=4 order=[1, 0] times_ns=[1194232, 1413497]
ISOLATION cols=2048 batch=32 round=5 order=[0, 1] times_ns=[1359805, 2284045]
ISOLATION cols=2048 batch=32 round=6 order=[1, 0] times_ns=[1080697, 1616992]
ISOLATION cols=2048 batch=64 round=0 order=[1, 0] times_ns=[3066999, 2671149]
ISOLATION cols=2048 batch=64 round=1 order=[0, 1] times_ns=[3452408, 2887709]
ISOLATION cols=2048 batch=64 round=2 order=[1, 0] times_ns=[3209779, 2807197]
ISOLATION cols=2048 batch=64 round=3 order=[0, 1] times_ns=[3423965, 2910021]
ISOLATION cols=2048 batch=64 round=4 order=[1, 0] times_ns=[3254364, 2988359]
ISOLATION cols=2048 batch=64 round=5 order=[0, 1] times_ns=[3268770, 2861951]
ISOLATION cols=2048 batch=64 round=6 order=[1, 0] times_ns=[3470833, 3041421]
ISOLATION cols=2048 batch=65 round=0 order=[1, 0] times_ns=[4076400, 3481153]
ISOLATION cols=2048 batch=65 round=1 order=[0, 1] times_ns=[4203652, 3456827]
ISOLATION cols=2048 batch=65 round=2 order=[1, 0] times_ns=[3994044, 3360595]
ISOLATION cols=2048 batch=65 round=3 order=[0, 1] times_ns=[4146984, 3383148]
ISOLATION cols=2048 batch=65 round=4 order=[1, 0] times_ns=[4197359, 3686081]
ISOLATION cols=2048 batch=65 round=5 order=[0, 1] times_ns=[4134640, 3367138]
ISOLATION cols=2048 batch=65 round=6 order=[1, 0] times_ns=[4314812, 3545144]
ISOLATION cols=8192 batch=32 round=0 order=[1, 0] times_ns=[5938326, 5909172]
ISOLATION cols=8192 batch=32 round=1 order=[0, 1] times_ns=[5655041, 5413082]
ISOLATION cols=8192 batch=32 round=2 order=[1, 0] times_ns=[6244407, 5731655]
ISOLATION cols=8192 batch=32 round=3 order=[0, 1] times_ns=[6044918, 6184182]
ISOLATION cols=8192 batch=32 round=4 order=[1, 0] times_ns=[5910925, 5982290]
ISOLATION cols=8192 batch=32 round=5 order=[0, 1] times_ns=[6737641, 5301080]
ISOLATION cols=8192 batch=32 round=6 order=[1, 0] times_ns=[5735442, 5544360]
ISOLATION cols=8192 batch=64 round=0 order=[1, 0] times_ns=[15645532, 11677747]
ISOLATION cols=8192 batch=64 round=1 order=[0, 1] times_ns=[16224478, 13302384]
ISOLATION cols=8192 batch=64 round=2 order=[1, 0] times_ns=[15511478, 13438381]
ISOLATION cols=8192 batch=64 round=3 order=[0, 1] times_ns=[15776841, 13094019]
ISOLATION cols=8192 batch=64 round=4 order=[1, 0] times_ns=[17051535, 13925204]
ISOLATION cols=8192 batch=64 round=5 order=[0, 1] times_ns=[15549800, 14389714]
ISOLATION cols=8192 batch=64 round=6 order=[1, 0] times_ns=[15156265, 13835554]
ISOLATION cols=8192 batch=65 round=0 order=[1, 0] times_ns=[16609989, 14027628]
ISOLATION cols=8192 batch=65 round=1 order=[0, 1] times_ns=[16771554, 14193752]
ISOLATION cols=8192 batch=65 round=2 order=[1, 0] times_ns=[17472742, 14216917]
ISOLATION cols=8192 batch=65 round=3 order=[0, 1] times_ns=[16954170, 14356841]
ISOLATION cols=8192 batch=65 round=4 order=[1, 0] times_ns=[18395220, 14104954]
ISOLATION cols=8192 batch=65 round=5 order=[0, 1] times_ns=[18530235, 16874349]
ISOLATION cols=8192 batch=65 round=6 order=[1, 0] times_ns=[18260484, 15781550]
```

## Complete raw captured timings and origin gates

Integer nanosecond totals for three projections/member; route0=B,1=S,2=F.
Round0 warmup, rounds1–6 measured. No timings removed.

```text
CAPTURE_GATE batch=32 complete_logits=65536 bits=exact captured_formats=2 context=65 prompt_ids_activations=memory-only candidate_model_injection=false
CAPTURE layer=0 role=gate kind=Q4_K rows=8192 cols=2048 batch=32 round=0 order=[0, 1, 2] times_ns=[31667374, 27061360, 32248906]
CAPTURE layer=0 role=gate kind=Q4_K rows=8192 cols=2048 batch=32 round=1 order=[0, 1, 2] times_ns=[28539920, 25820159, 31713151]
CAPTURE layer=0 role=gate kind=Q4_K rows=8192 cols=2048 batch=32 round=2 order=[2, 1, 0] times_ns=[28613430, 26566442, 31080653]
CAPTURE layer=0 role=gate kind=Q4_K rows=8192 cols=2048 batch=32 round=3 order=[1, 2, 0] times_ns=[31764138, 26418822, 29593957]
CAPTURE layer=0 role=gate kind=Q4_K rows=8192 cols=2048 batch=32 round=4 order=[0, 2, 1] times_ns=[29692083, 28242016, 29778868]
CAPTURE layer=0 role=gate kind=Q4_K rows=8192 cols=2048 batch=32 round=5 order=[2, 0, 1] times_ns=[31849459, 28731073, 31946914]
CAPTURE layer=0 role=gate kind=Q4_K rows=8192 cols=2048 batch=32 round=6 order=[1, 0, 2] times_ns=[32348836, 29208137, 30823186]
CAPTURE layer=0 role=down kind=Q6_K rows=2048 cols=8192 batch=32 round=0 order=[0, 1, 2] times_ns=[41986211, 40172465, 40042980]
CAPTURE layer=0 role=down kind=Q6_K rows=2048 cols=8192 batch=32 round=1 order=[0, 1, 2] times_ns=[42158587, 44565836, 39647210]
CAPTURE layer=0 role=down kind=Q6_K rows=2048 cols=8192 batch=32 round=2 order=[2, 1, 0] times_ns=[44010805, 43923680, 41341570]
CAPTURE layer=0 role=down kind=Q6_K rows=2048 cols=8192 batch=32 round=3 order=[1, 2, 0] times_ns=[45394617, 42817755, 39526241]
CAPTURE layer=0 role=down kind=Q6_K rows=2048 cols=8192 batch=32 round=4 order=[0, 2, 1] times_ns=[42838525, 42457423, 40933767]
CAPTURE layer=0 role=down kind=Q6_K rows=2048 cols=8192 batch=32 round=5 order=[2, 0, 1] times_ns=[40746312, 40871770, 41169955]
CAPTURE layer=0 role=down kind=Q6_K rows=2048 cols=8192 batch=32 round=6 order=[1, 0, 2] times_ns=[43379330, 41389180, 42981025]
CAPTURE_GATE batch=64 complete_logits=65536 bits=exact captured_formats=2 context=65 prompt_ids_activations=memory-only candidate_model_injection=false
CAPTURE layer=0 role=gate kind=Q4_K rows=8192 cols=2048 batch=64 round=0 order=[0, 1, 2] times_ns=[71164244, 73918090, 70331566]
CAPTURE layer=0 role=gate kind=Q4_K rows=8192 cols=2048 batch=64 round=1 order=[0, 1, 2] times_ns=[73166937, 71733471, 69850776]
CAPTURE layer=0 role=gate kind=Q4_K rows=8192 cols=2048 batch=64 round=2 order=[2, 1, 0] times_ns=[70867131, 75541887, 68660772]
CAPTURE layer=0 role=gate kind=Q4_K rows=8192 cols=2048 batch=64 round=3 order=[1, 2, 0] times_ns=[72941381, 74688029, 66011984]
CAPTURE layer=0 role=gate kind=Q4_K rows=8192 cols=2048 batch=64 round=4 order=[0, 2, 1] times_ns=[76596474, 80644282, 74866868]
CAPTURE layer=0 role=gate kind=Q4_K rows=8192 cols=2048 batch=64 round=5 order=[2, 0, 1] times_ns=[74549688, 71678218, 69466658]
CAPTURE layer=0 role=gate kind=Q4_K rows=8192 cols=2048 batch=64 round=6 order=[1, 0, 2] times_ns=[77502471, 81564906, 70623201]
CAPTURE layer=0 role=down kind=Q6_K rows=2048 cols=8192 batch=64 round=0 order=[0, 1, 2] times_ns=[92038267, 97264737, 78745957]
CAPTURE layer=0 role=down kind=Q6_K rows=2048 cols=8192 batch=64 round=1 order=[0, 1, 2] times_ns=[93313733, 88521946, 72571420]
CAPTURE layer=0 role=down kind=Q6_K rows=2048 cols=8192 batch=64 round=2 order=[2, 1, 0] times_ns=[84346857, 85568351, 73288790]
CAPTURE layer=0 role=down kind=Q6_K rows=2048 cols=8192 batch=64 round=3 order=[1, 2, 0] times_ns=[84365373, 92475074, 70294409]
CAPTURE layer=0 role=down kind=Q6_K rows=2048 cols=8192 batch=64 round=4 order=[0, 2, 1] times_ns=[88818979, 84530385, 73461198]
CAPTURE layer=0 role=down kind=Q6_K rows=2048 cols=8192 batch=64 round=5 order=[2, 0, 1] times_ns=[84845162, 88586559, 73091928]
CAPTURE layer=0 role=down kind=Q6_K rows=2048 cols=8192 batch=64 round=6 order=[1, 0, 2] times_ns=[89446488, 86003917, 74364217]
```

## Ideas, inspirations and remaining gates

Reuse Mivi's checked [four-row API](../crates/mivi-quant/src/batch_scratch/four_row.rs)
and production tile walker pattern from the older tolerance-based capture test;
the old test remains untouched and its tolerance is not this exact gate.
[v0.2.77 operator evidence](FOUR_ROW_PREFILL_EVIDENCE_2026-10-10.md) motivates
separating accumulation and actual activations. Balanced order, cache declarations,
and negative evidence follow [Colibri benchmarking](https://github.com/JustVugg/colibri/blob/main/docs/benchmarking.md).
No external inference implementation was copied.

Next: investigate the Q4 batch32 regression with repeatable stage measurements
before changing kernels; separately approve model-workspace ownership and an
explicit default-off candidate integration. Promotion still requires complete
candidate-integrated model/KV/SSM parity, representative prompt/tool loops, RSS,
short/small-case regression and broader model/CPU evidence.
