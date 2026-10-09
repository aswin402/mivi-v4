# Four-row prefill operator experiment — 2026-10-10

## Outcome and decision

The bounded operator diagnostic passed complete finite-output bit comparisons for
all three controls. The four-row candidate is **not promoted**. Q6_K batch 64/65
and Q4_K batch 65 showed consistent lower paired time than scratch-only, but Q4_K
batch 32 and several synthetic/small-batch cases regressed. Q4_K batch 64 was mixed.
This is sufficient to retain a candidate for narrower investigation, not to enable
it in model/server dispatch or claim an agent TTFT improvement.

## Implementation

Feature `batch-four-row-experiment` in mivi-quant forwards to the isolated
mivi-core SIMD feature and depends on `batch-scratch-experiment`.
`BatchProjectionScratch::new_four_row` allocates four decoded rows per worker;
the existing constructor/control still allocates two. The explicit
`quantized_matmul_rows_four_row` API validates before writes, retains the baseline
worker partition boundaries, processes full groups of four, and delegates partial
groups and batches below 32 to the existing scratch-only helper.

The AVX2/FMA kernel uses sixteen batch lanes/eight accumulators, sharing each
activation load across four rows. It preserves ascending columns, panels of 128,
vector FMA and scalar-tail nonfused arithmetic. A checked public wrapper and runtime
CPU detection guard a private unsafe routine. The user explicitly approved this
bounded unsafe code; no external inference source or new dependency was introduced.
Fallback dispatch reuses the existing paired helper.

## Correctness gates

- Observed RED on the core no-op stub at batch=1/cols=1, and on the quant constructor
  stub with ArithmeticOverflow. The benchmark shape-validator stub also failed its
  valid-shape control before implementation.
- Core tests passed in debug and release: 77 batch/column cases including nonzero
  initial outputs, 128-column panel boundaries, full/vector/scalar tails and output
  sentinels; additional cancellation-sensitive cases and invalid-dimension panics
  before writes. Forced pair-helper fallback agrees on this host.
- Quant complete-bit matrix: **3,168 cases**, six formats, columns 0/256/512,
  rows 0/1/3/4/7/257/258/259, batches 0/1/2/8/9/31/32/33/63/64/65 and explicit
  one-/two-thread pools. One workspace per pool crosses all cases, starts poisoned
  with NaNs, and retains all buffer pointers/lengths/capacities. Output tails stay intact.
- Additional controls reject constructor count/byte overflow, zero workers,
  insufficient decoded workspace, short external buffers, unsupported format,
  misalignment, shape/worker capacities and overflow without changing output or
  workspace bits. Empty work retains external validation. Unaligned F32 batch-one
  delegation and nonblock-width F16/BF16/F32 projections match baseline.
- Portable dispatch is retained in source. No cross-ISA compilation/execution or
  scalar-only CPU parity is claimed from this AVX2 host.

## Workload and timing contract

Host: AMD Ryzen 7 7730U; AVX2/FMA detected; rustc 1.99.0.
Cargo jobs=1, Rayon threads=2, serial test harness. This evaluation started no other
Cargo invocation during measurement. Release source was based on v0.2.76 with these experimental
additions; publication patch bump follows measurement.

Inputs were deterministic generated finite F32 activations, **not captured model
activations**. Cases: synthetic Q8_0 257x256; full real Q4_K 8192x2048 and Q6_K
2048x8192 matrices from the local public LFM2.5-1.2B Instruct Q4_K_M GGUF.
Selection sorts metadata names and takes the first eligible matrix per wire format,
not a model-name dispatch. Only two-dimensional Q4_K/Q6_K with columns <=16,384
(block aligned), rows <=8,192 and nonzero dimensions are eligible. No rows are capped
or sampled in accepted real cases.

Each of the 15 format/batch cases has an untimed reference projection, then one
warmup triple and six measured triples. Each member executes three identical
projections; output/finiteness comparisons and logging occur **after** its timer.
All six control-order permutations balance first/middle/last position. The timed
region includes projection validation, input transpose, weight decode/accumulation
and output layout, plus ordinary API allocations; workspace construction is outside
timers. 945 timed projection calls executed, with 315 member-final complete-output
comparisons, plus 15 untimed reference projections. Diagnostic passed in 16.69s.

Cache state: warm process/model pages; no OS page-cache drop, engine prefix cache,
KV/SSM state or model inference. Ordinary temporary buffers allocate each call;
scratch workspaces remain resident throughout their case. User CPU scheduling,
frequency/thermal behavior and background load were not pinned or controlled.
Substantial within-case drift appears in Q4 batch 32 and Q6 batch 65; no cause is
attributed. Six triples are a pilot, not statistical confidence or general superiority.

```sh
CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 \
MIVI_TEST_MODEL=/absolute/path/to/local.gguf \
timeout --signal=TERM --kill-after=5s 180s \
cargo test -p mivi-quant --lib --release --offline \
  --features batch-four-row-experiment four_row_operator_measurement \
  -- --ignored --test-threads=1 --nocapture
```

## Summary

B=ordinary API, S=scratch-only, F=four-row. Times are separate medians in **ms per
projection** (member totals /3). Ratios are medians of the six **within-triple**
ratios, not ratios of separate medians; below one is lower time.
No warmup enters summaries.

| Source/format | Rows x cols | Batch | B ms | S ms | F ms | S/B | F/B | F/S | F/S range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| synthetic/Q8_0 | 257 x 256 | 8 | 0.100139 | 0.093200 | 0.107047 | 0.980595 | 1.095875 | 1.167307 | 0.957317–1.437754 |
| synthetic/Q8_0 | 257 x 256 | 9 | 0.280993 | 0.290845 | 0.338357 | 1.025019 | 1.201266 | 1.165724 | 0.894684–1.249567 |
| synthetic/Q8_0 | 257 x 256 | 32 | 0.150349 | 0.125436 | 0.142434 | 0.840190 | 0.938743 | 1.207075 | 1.032381–1.391034 |
| synthetic/Q8_0 | 257 x 256 | 64 | 0.277857 | 0.240886 | 0.254460 | 0.876144 | 0.939628 | 1.027514 | 0.952249–1.280581 |
| synthetic/Q8_0 | 257 x 256 | 65 | 0.333064 | 0.282641 | 0.273840 | 0.884084 | 0.816998 | 0.935012 | 0.845598–1.073541 |
| real/Q4_K | 8192 x 2048 | 8 | 17.172988 | 17.539604 | 17.786970 | 1.032004 | 1.029199 | 1.037525 | 0.990359–1.061403 |
| real/Q4_K | 8192 x 2048 | 9 | 62.394924 | 61.073487 | 60.594938 | 0.983324 | 0.987844 | 0.996668 | 0.980400–1.055564 |
| real/Q4_K | 8192 x 2048 | 32 | 14.466118 | 14.332530 | 14.679334 | 0.968558 | 1.046626 | 1.095763 | 1.002210–1.153217 |
| real/Q4_K | 8192 x 2048 | 64 | 23.714508 | 23.197837 | 22.617572 | 0.920356 | 0.966706 | 0.993372 | 0.905042–1.139621 |
| real/Q4_K | 8192 x 2048 | 65 | 28.643482 | 28.642834 | 24.536395 | 0.982916 | 0.844083 | 0.832634 | 0.816689–0.882345 |
| real/Q6_K | 2048 x 8192 | 8 | 10.933435 | 10.881860 | 11.096424 | 0.988998 | 1.014909 | 1.022969 | 1.012343–1.041255 |
| real/Q6_K | 2048 x 8192 | 9 | 31.422080 | 31.381842 | 31.746050 | 0.996276 | 1.007719 | 1.015773 | 0.993481–1.029802 |
| real/Q6_K | 2048 x 8192 | 32 | 13.085686 | 12.465991 | 12.658598 | 0.944562 | 0.947051 | 0.988335 | 0.936578–1.046309 |
| real/Q6_K | 2048 x 8192 | 64 | 26.359136 | 25.992954 | 22.074207 | 0.980513 | 0.838568 | 0.854297 | 0.828313–0.896369 |
| real/Q6_K | 2048 x 8192 | 65 | 33.503744 | 33.775028 | 27.995565 | 0.987250 | 0.820265 | 0.829216 | 0.811478–0.854438 |

Q6 batch 64 F/S median=0.854297 (14.57% lower time), all six ratios below one.
Q6 batch 65=0.829216 (17.08% lower), all six below one.
Q4 batch 65=0.832634 (16.74% lower), all six below one.
Retain Q4 batch 32=1.095763 (9.58% higher), **all six above one**.
Q4 batch 64 median=0.993372 but range 0.905042–1.139621; no reliable improvement claim.
Synthetic batch 32 F/B=0.938743 while F/S=1.207075: comparing only ordinary API would
hide the regression relative to reusable scratch. Batches 8/9 do not execute the
four-row SIMD branch; their differences are unchanged-kernel/wrapper/host variation,
not evidence of four-row accumulation gains.

Additional requested decoded storage versus scratch-only is 2 rows x columns x
2 workers x 4 bytes: Q8_0=4KiB, Q4_K=32KiB, Q6_K=128KiB. These are requested buffer
elements, **not measured RSS** or allocator overhead. No server memory accounting is
integrated here.

## Full raw timing evidence

Each row records original **three-call totals in integer nanoseconds**. Round 0 is
warmup; rounds 1–6 are measured. Orders map 0=B, 1=S, 2=F.
No pair/member or negative case was removed.

| Source/format | Batch | Round | Order | B ns | S ns | F ns |
|---|---:|---:|---|---:|---:|---:|
| synthetic/Q8_0 | 8 | 0 | 0,1,2 | 298608 | 299941 | 323856 |
| synthetic/Q8_0 | 8 | 1 | 0,1,2 | 313056 | 270545 | 320941 |
| synthetic/Q8_0 | 8 | 2 | 2,1,0 | 272529 | 279371 | 386987 |
| synthetic/Q8_0 | 8 | 3 | 1,2,0 | 294981 | 279832 | 321342 |
| synthetic/Q8_0 | 8 | 4 | 0,2,1 | 305852 | 309689 | 311994 |
| synthetic/Q8_0 | 8 | 5 | 2,0,1 | 337012 | 258401 | 371517 |
| synthetic/Q8_0 | 8 | 6 | 1,0,2 | 245478 | 307946 | 294802 |
| synthetic/Q8_0 | 9 | 0 | 0,1,2 | 796296 | 809000 | 890004 |
| synthetic/Q8_0 | 9 | 1 | 0,1,2 | 814741 | 817656 | 1012347 |
| synthetic/Q8_0 | 9 | 2 | 2,1,0 | 809170 | 821504 | 1026524 |
| synthetic/Q8_0 | 9 | 3 | 1,2,0 | 852342 | 1041713 | 932004 |
| synthetic/Q8_0 | 9 | 4 | 0,2,1 | 868703 | 894432 | 967732 |
| synthetic/Q8_0 | 9 | 5 | 2,0,1 | 861861 | 897298 | 1017798 |
| synthetic/Q8_0 | 9 | 6 | 1,0,2 | 833617 | 850639 | 1018348 |
| synthetic/Q8_0 | 32 | 0 | 0,1,2 | 541010 | 607967 | 517335 |
| synthetic/Q8_0 | 32 | 1 | 0,1,2 | 443113 | 404400 | 417495 |
| synthetic/Q8_0 | 32 | 2 | 2,1,0 | 441059 | 396705 | 551830 |
| synthetic/Q8_0 | 32 | 3 | 1,2,0 | 458983 | 409930 | 429287 |
| synthetic/Q8_0 | 32 | 4 | 0,2,1 | 489021 | 355908 | 405652 |
| synthetic/Q8_0 | 32 | 5 | 2,0,1 | 479562 | 333745 | 425319 |
| synthetic/Q8_0 | 32 | 6 | 1,0,2 | 424599 | 334267 | 441310 |
| synthetic/Q8_0 | 64 | 0 | 0,1,2 | 868423 | 939539 | 924960 |
| synthetic/Q8_0 | 64 | 1 | 0,1,2 | 857242 | 724049 | 743255 |
| synthetic/Q8_0 | 64 | 2 | 2,1,0 | 859887 | 757873 | 779474 |
| synthetic/Q8_0 | 64 | 3 | 1,2,0 | 836362 | 854387 | 813589 |
| synthetic/Q8_0 | 64 | 4 | 0,2,1 | 830781 | 721264 | 894773 |
| synthetic/Q8_0 | 64 | 5 | 2,0,1 | 668162 | 583550 | 747283 |
| synthetic/Q8_0 | 64 | 6 | 1,0,2 | 773132 | 679523 | 686627 |
| synthetic/Q8_0 | 65 | 0 | 0,1,2 | 799602 | 782419 | 641681 |
| synthetic/Q8_0 | 65 | 1 | 0,1,2 | 808299 | 749927 | 634137 |
| synthetic/Q8_0 | 65 | 2 | 2,1,0 | 1064606 | 806335 | 690084 |
| synthetic/Q8_0 | 65 | 3 | 1,2,0 | 1015984 | 884464 | 949508 |
| synthetic/Q8_0 | 65 | 4 | 0,2,1 | 1029780 | 924350 | 821564 |
| synthetic/Q8_0 | 65 | 5 | 2,0,1 | 982400 | 837194 | 821473 |
| synthetic/Q8_0 | 65 | 6 | 1,0,2 | 910293 | 858655 | 908490 |
| real/Q4_K | 8 | 0 | 0,1,2 | 52981403 | 53739646 | 55088244 |
| real/Q4_K | 8 | 1 | 0,1,2 | 50818677 | 53187195 | 56089590 |
| real/Q4_K | 8 | 2 | 2,1,0 | 51644789 | 53481094 | 56765015 |
| real/Q4_K | 8 | 3 | 1,2,0 | 51819852 | 50272718 | 53355456 |
| real/Q4_K | 8 | 4 | 0,2,1 | 55522831 | 52295327 | 53366366 |
| real/Q4_K | 8 | 5 | 2,0,1 | 50583619 | 52378366 | 51873395 |
| real/Q4_K | 8 | 6 | 1,0,2 | 51393140 | 52859261 | 52871395 |
| real/Q4_K | 9 | 0 | 0,1,2 | 189689096 | 181565673 | 180181608 |
| real/Q4_K | 9 | 1 | 0,1,2 | 181388406 | 185339549 | 181706893 |
| real/Q4_K | 9 | 2 | 2,1,0 | 187616482 | 183355412 | 193543395 |
| real/Q4_K | 9 | 3 | 1,2,0 | 186753061 | 179912799 | 179822918 |
| real/Q4_K | 9 | 4 | 0,2,1 | 201664166 | 181334065 | 181862732 |
| real/Q4_K | 9 | 5 | 2,0,1 | 181957935 | 183085511 | 180580022 |
| real/Q4_K | 9 | 6 | 1,0,2 | 188467065 | 186461858 | 185312351 |
| real/Q4_K | 32 | 0 | 0,1,2 | 62387310 | 56428337 | 62845213 |
| real/Q4_K | 32 | 1 | 0,1,2 | 54860893 | 52127650 | 59707730 |
| real/Q4_K | 32 | 2 | 2,1,0 | 46410968 | 45804735 | 52822803 |
| real/Q4_K | 32 | 3 | 1,2,0 | 43388906 | 45129188 | 45325582 |
| real/Q4_K | 32 | 4 | 0,2,1 | 43407801 | 40865993 | 42750420 |
| real/Q4_K | 32 | 5 | 2,0,1 | 41861849 | 36176135 | 41536429 |
| real/Q4_K | 32 | 6 | 1,0,2 | 36768343 | 38470844 | 38555865 |
| real/Q4_K | 64 | 0 | 0,1,2 | 85287825 | 75819060 | 81061850 |
| real/Q4_K | 64 | 1 | 0,1,2 | 80175814 | 73691349 | 68526688 |
| real/Q4_K | 64 | 2 | 2,1,0 | 66300290 | 74227250 | 67178742 |
| real/Q4_K | 64 | 3 | 1,2,0 | 71501171 | 61519129 | 60798918 |
| real/Q4_K | 64 | 4 | 0,2,1 | 70558166 | 65025674 | 64924973 |
| real/Q4_K | 64 | 5 | 2,0,1 | 70785879 | 69297265 | 78003950 |
| real/Q4_K | 64 | 6 | 1,0,2 | 77193367 | 69889754 | 79647819 |
| real/Q4_K | 65 | 0 | 0,1,2 | 88125808 | 89412227 | 73240422 |
| real/Q4_K | 65 | 1 | 0,1,2 | 94143364 | 90780303 | 74139234 |
| real/Q4_K | 65 | 2 | 2,1,0 | 87842229 | 87957529 | 73190127 |
| real/Q4_K | 65 | 3 | 1,2,0 | 85579191 | 89431785 | 74511112 |
| real/Q4_K | 65 | 4 | 0,2,1 | 86281699 | 83899474 | 74028242 |
| real/Q4_K | 65 | 5 | 2,0,1 | 80211554 | 78158906 | 68578168 |
| real/Q4_K | 65 | 6 | 1,0,2 | 82568229 | 81859972 | 67087561 |
| real/Q6_K | 8 | 0 | 0,1,2 | 33475976 | 33333285 | 33705683 |
| real/Q6_K | 8 | 1 | 0,1,2 | 32593326 | 32139913 | 32980172 |
| real/Q6_K | 8 | 2 | 2,1,0 | 32779891 | 32816641 | 33368241 |
| real/Q6_K | 8 | 3 | 1,2,0 | 33100481 | 32393696 | 33730090 |
| real/Q6_K | 8 | 4 | 0,2,1 | 33677069 | 32848903 | 33696796 |
| real/Q6_K | 8 | 5 | 2,0,1 | 32418533 | 32736048 | 33140117 |
| real/Q6_K | 8 | 6 | 1,0,2 | 32820719 | 32555113 | 33210300 |
| real/Q6_K | 9 | 0 | 0,1,2 | 95303985 | 95807093 | 95643701 |
| real/Q6_K | 9 | 1 | 0,1,2 | 97058986 | 94252113 | 96153081 |
| real/Q6_K | 9 | 2 | 2,1,0 | 93817976 | 94046642 | 94442616 |
| real/Q6_K | 9 | 3 | 1,2,0 | 94437567 | 93732484 | 95555285 |
| real/Q6_K | 9 | 4 | 0,2,1 | 93969616 | 94244409 | 97053036 |
| real/Q6_K | 9 | 5 | 2,0,1 | 94094914 | 93786237 | 94921017 |
| real/Q6_K | 9 | 6 | 1,0,2 | 95711283 | 95312392 | 94691059 |
| real/Q6_K | 32 | 0 | 0,1,2 | 35861508 | 36763886 | 36711967 |
| real/Q6_K | 32 | 1 | 0,1,2 | 39090185 | 35681004 | 35477908 |
| real/Q6_K | 32 | 2 | 2,1,0 | 39094643 | 36416435 | 35508998 |
| real/Q6_K | 32 | 3 | 1,2,0 | 39511697 | 40300629 | 37744694 |
| real/Q6_K | 32 | 4 | 0,2,1 | 41129607 | 38280052 | 38613387 |
| real/Q6_K | 32 | 5 | 2,0,1 | 38131529 | 36515895 | 38206893 |
| real/Q6_K | 32 | 6 | 1,0,2 | 39419472 | 39348216 | 38654154 |
| real/Q6_K | 64 | 0 | 0,1,2 | 79445347 | 79775687 | 64158557 |
| real/Q6_K | 64 | 1 | 0,1,2 | 78428172 | 76314648 | 65037891 |
| real/Q6_K | 64 | 2 | 2,1,0 | 80647496 | 77189794 | 65537361 |
| real/Q6_K | 64 | 3 | 1,2,0 | 75212751 | 80563746 | 66732025 |
| real/Q6_K | 64 | 4 | 0,2,1 | 80281009 | 76735410 | 65713217 |
| real/Q6_K | 64 | 5 | 2,0,1 | 74857435 | 79830902 | 71557923 |
| real/Q6_K | 64 | 6 | 1,0,2 | 79726644 | 78767929 | 67597714 |
| real/Q6_K | 65 | 0 | 0,1,2 | 93487490 | 89433953 | 80708353 |
| real/Q6_K | 65 | 1 | 0,1,2 | 91748330 | 89380893 | 75783528 |
| real/Q6_K | 65 | 2 | 2,1,0 | 95339867 | 91163938 | 73977530 |
| real/Q6_K | 65 | 3 | 1,2,0 | 96840855 | 97716130 | 81939818 |
| real/Q6_K | 65 | 4 | 0,2,1 | 104181610 | 104934041 | 86033571 |
| real/Q6_K | 65 | 5 | 2,0,1 | 107019824 | 107052235 | 87191877 |
| real/Q6_K | 65 | 6 | 1,0,2 | 112858819 | 106273302 | 90803983 |

## Verification and limits

Final v0.2.77 quant release batch filters passed 18 tests with four-row enabled
(1 diagnostic intentionally ignored), 12 scratch-only, and 6 with both experiments
disabled. Core release controls passed 3; quant debug four-row controls passed 6
(1 diagnostic ignored). Scoped formatting/diff checks passed. GPT-6 Luna/high approved
read-only arithmetic/API and subsequent diagnostic/evidence reviews with no findings.
Scoped Clippy passed with the pre-existing `chunks_exact_to_as_chunks` style lint
explicitly allowed; strict first-attempt findings and commands are recorded in the
[implementation plan](superpowers/plans/2026-10-09-four-row-prefill.md).
No full-workspace Cargo build/check/test is run or claimed. The diagnostic proves
same-host operator parity for generated activations; it does not prove model
state/logit parity, agent quality, HTTP/tool-loop success, TTFT, memory budget,
RSS, cross-engine superiority or usefulness on other architectures.

Next: investigate why batch 32/Q4 regresses, repeat targeted controls with captured
activations/other tensors and workload sizes, then design an explicit model
integration only if stable wins survive. Any integration must budget scratch from
tensor shapes/metadata and retain portable/default baselines; no model-name rules.

## Ideas, inspirations and sources

- Mivi's existing pair-panel AVX2/portable arithmetic supplies the numerical control.
- [Reusable scratch evidence](BATCH_SCRATCH_EVIDENCE_2026-10-09.md) provides the
  second control; [projection-cost evidence](PROJECTION_COST_EVIDENCE_2026-10-05.md)
  identified accumulation as the dominant projection cost.
- [Agent-sized decode evidence](Q4_AGENT_SIZED_EVIDENCE_2026-10-07.md) showed that
  decode-only gains do not optimize initial chunked-prefill wait.
- Separating variables, balancing order and retaining negative results follows
  [Colibri benchmarking](https://github.com/JustVugg/colibri/blob/main/docs/benchmarking.md).
  No external inference source was copied; this implementation adds no new research claim.
- [Design](superpowers/specs/2026-10-09-four-row-prefill-design.md) and
  [plan](superpowers/plans/2026-10-09-four-row-prefill.md) record scope and approval.

