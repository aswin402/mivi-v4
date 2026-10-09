# Q4 stage replay evidence — 2026-10-10

## Outcome and decision

The bounded live stage diagnostic passed exact finite outputs for all 144 timed
members and eight untimed replay gates over four cases. Both complete 65,536-logit
capture-origin gates passed. The separate existing three-control capture diagnostic
also passed after private helper extraction (252 timed projections, 84 member-final
comparisons, four references). No production kernel/default changed; candidate
outputs never feed into the model.

**No Q4 batch-32 fix or confirmed hardware root cause.** Its unchanged four-row /
scratch-only paired median was 1.003644 with mixed range 0.897858–1.163256. In the
profiled replay, Q4 batch32 accumulated worker elapsed work was higher
(PF/PS paired median 1.106910) while decode was similar/slightly lower
(0.980375). This narrows a hypothesis toward accumulation/group orchestration,
not a hardware explanation or measured wall percentage.

Batch-64 unchanged Q4/Q6 F/S paired medians 0.810169/0.857840 were lower in all eight
pairs. Retain the smaller-case negatives. Eight repetitions, one model/host and
warm process pages do not establish statistical confidence or general superiority.

## Implementation and fidelity

New safe test-only child of the existing captured diagnostic, gated by
batch-four-row-experiment plus cfg(test). Reuses unchanged dequantization and
pair/four-row SIMD wrappers. Original quant/model/server production functions
remain untouched. No dependencies, new unsafe code or model-name rules.

Replay mirrors validation, transpose, two worker partitions, decode/zero order,
accumulation and output layout. Pair zero occurs between its two row decodes;
four-row zero follows its four decodes. Worker reports are disjoint and no buffers
grow during calls. Checked duration conversion/addition, nonnegative wall residual,
pool/group/buffer/capacity validation and exact count checks protect interpretation.

The bounded stage replay accepts Q4_K/Q6_K, batch32/64, nonzero aligned columns
<=16,384 and rows between the existing parallel threshold and8,192. Both two-thread
partitions must be positive multiples4. Reject unhandled shapes rather than
trimming complete matrices; no partial-group/other-format claim.

The original model/prompt/BOS/origin gate moved to private shared helpers without
changing its original three-call timing loop. Context65, F32 KV, no adapters;
32/64 effective prompt IDs. UTF-8 prompt override <=64KiB before tokenization.
Actual norm/gate activation capture retains executed FFN inputs; no IDs, weights,
activations or output values printed/persisted. Logged metadata is intentional.

## Environment

AMD Ryzen7 7730U; AVX2/FMA, rustc1.99.0. Cargo jobs1, Rayon threads2, serial harness,
offline scoped commands. No concurrent Cargo command during live runs.
Measurements used v0.2.78 plus this addition before publication bump0.2.79.
Out-array allocation was subsequently made fallible and rejection tests expanded;
these changes are outside timers or test-only and do not change measured math.

Existing local public LFM2.5-1.2B-Instruct-Q4_K_M.gguf SHA256:
`b1b3de114215d9507409a662a501a631095a479a419584e8a2ded6304b19b4f5`.
Metadata/executed type selection chose complete layer0 Q4 gate8192x2048 and
Q6 down2048x8192; no rows sampled or capped.

Post-run host snapshot: load1.98/1.47/1.61; about7,505MiB memory available,
1,955MiB swap used of4,095MiB. This is a post-run snapshot, not a continuous monitor
or measured process RSS. No CPU pinning/thermal/frequency control/cache flush.
Warm mapped model pages and resident per-case scratch; no 256-token prefix snapshot
boundary reached and no prefix restore. Host variability remains uncontrolled.

## Timing and interpretation

Routes0=S unchanged scratch,1=F unchanged four-row,2=PS profiled pair replay,
3=PF profiled four-row replay. One warmup quadruple then eight measured quadruples
use balanced rotations/reversals: every route occurs twice in each position;
each relative pair ordering occurs four times. One full projection/member:
4cases x9rounds x4routes =144 timed projections and complete output comparisons.
Four ordinary references and eight untimed replay comparisons precede timings.
Stage diagnostic passed in7.30s; existing capture regression passed in9.46s.

Construction/model capture/reference/exact checks/logging are outside timers.
Outer wall includes replay metadata return; inner call wall ends before residual
calculation. Timed replay contains clocks/counters around each group and decode.
Instrumented replay is duplicated orchestration with identical output bits,
**not** the same compiled implementation as the unchanged control.

PS/S and PF/F quantify combined replay/instrumentation/host disturbance, not pure
clock overhead. Some replays run faster than unchanged controls; do not subtract
stage differences from control time or assume instrumentation only adds time.
Near-tied Q4 batch32 control times mean its earlier11.61%/3.26% regressions have
not become a stable per-host constant.

Worker decode/zero/accumulate elapsed sums overlap across workers; they are neither
CPU time nor elapsed wall shares. They must not be added to serial wall stages.
Independent stage medians do not sum to median total. Zero-duration tiny stages
are accepted; checked wall residual retains unclassified gaps.

## Summaries

All ratios below are medians of eight within-quadruple ratios, not ratios of
independent medians. Round0 excluded. Lower than1 means less elapsed time.

| Format | Batch | F/S | F/S range | PS/S | PF/F |
|---|---:|---:|---:|---:|---:|
| Q4_K | 32 | 1.003644 | 0.897858–1.163256 | 0.970656 | 1.025273 |
| Q6_K | 32 | 0.990703 | 0.947021–1.117128 | 0.968295 | 0.979142 |
| Q4_K | 64 | 0.810169 | 0.772166–0.903036 | 0.970264 | 0.979409 |
| Q6_K | 64 | 0.857840 | 0.813903–0.978653 | 0.982148 | 1.024005 |

Profile-only wall medians in milliseconds; compare PS/PF descriptively, not as a
substitute for unchanged-route paired timings.

| Format | Batch | Replay | Call | Validation | Transpose | Rows region | Layout | Residual |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Q4_K | 32 | PS | 9.111062 | 0.000365 | 0.056696 | 8.155715 | 0.908749 | 0.000206 |
| Q4_K | 32 | PF | 9.680186 | 0.000476 | 0.067752 | 8.703860 | 0.895776 | 0.000190 |
| Q6_K | 32 | PS | 11.562297 | 0.000596 | 0.552318 | 10.976431 | 0.203055 | 0.000196 |
| Q6_K | 32 | PF | 11.559757 | 0.000376 | 0.548225 | 10.852975 | 0.199208 | 0.000176 |
| Q4_K | 64 | PS | 24.710930 | 0.001002 | 0.192635 | 22.404395 | 2.106460 | 0.000230 |
| Q4_K | 64 | PF | 20.203685 | 0.000947 | 0.173505 | 17.933863 | 2.086969 | 0.000196 |
| Q6_K | 64 | PS | 26.511033 | 0.000701 | 0.930095 | 24.869744 | 0.554532 | 0.000260 |
| Q6_K | 64 | PF | 23.852406 | 0.000821 | 1.009418 | 22.242444 | 0.558054 | 0.000266 |

Separately summed two-worker elapsed work, medians in milliseconds, **not wall time**:

| Format | Batch | Replay | Decode work | Zero work | Accumulate work |
|---|---:|---|---:|---:|---:|
| Q4_K | 32 | PS | 3.259190 | 0.100386 | 11.975838 |
| Q4_K | 32 | PF | 3.235915 | 0.055459 | 13.468468 |
| Q6_K | 32 | PS | 6.972992 | 0.029454 | 14.242727 |
| Q6_K | 32 | PF | 6.855825 | 0.017224 | 14.339508 |
| Q4_K | 64 | PS | 4.245420 | 0.147071 | 39.342261 |
| Q4_K | 64 | PF | 3.811684 | 0.164605 | 30.847449 |
| Q6_K | 64 | PS | 8.422825 | 0.036646 | 40.316429 |
| Q6_K | 64 | PF | 8.266030 | 0.026596 | 35.110217 |

## Verification and reproduction

Observed RED on valid shape/accounting stub controls, then replay stub exact-output
fixture, then invalid measurement-order stub. Scoped debug/release tests passed
five stage controls (one live diagnostic intentionally ignored). Exact-bit fixtures
cover both formats/batches/group sizes with poisoned successive calls and stable
workspace allocation identities. Rejection controls preserve external output and
scratch bits for one-/two-thread pools, short buffers, invalid groups, short decoded/
transpose/output/worker capacity and short shape metadata. Count and wall accounting
tests include zero stages, overshoot and checked overflow.

Luna/high source review found no Critical/Important/Minor issues; its evidence
follow-up independently recomputed all ratio summaries/counts and found no
Critical/Important issues. Its pending-verification note is resolved by the
final-version results below.
v0.2.79 release feature-enabled batch filter: 26 passed, four diagnostics ignored.
Feature-off release batch filter: six passed. Debug diagnostic filter: eight
passed, three live diagnostics intentionally ignored. Scoped rustfmt passes;
staged whitespace checks are the final precommit gate. No Clippy-clean or
workspace-wide test claim is made.
No workspace-wide check/build/test; no hardware counters or assembly findings.

```sh
CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 \
MIVI_TEST_MODEL=/absolute/path/to/local.gguf timeout 240s \
cargo test -p mivi-quant --lib --release --offline \
  --features batch-four-row-experiment four_row_stage_replay_measurement \
  -- --ignored --test-threads=1 --nocapture
```

## Complete raw stage records

Original nanoseconds, all warmups/measured members retained. Round0 warmup;
rounds1–8 measured. Profile=Some only for replay; None for unchanged controls.
Work entries are each worker's separate elapsed measurements/counts. No records
filtered from the accepted144 members.

```text
STAGE_CONTRACT routes=S,F,PS,PF calls_per_member=1 profile=replay-not-production cache=warm-resident worker_time=overlapping-not-wall
CAPTURE_GATE batch=32 complete_logits=65536 bits=exact captured_formats=2 context=65 prompt_ids_activations=memory-only candidate_model_injection=false
STAGE_GATE kind=Q4_K batch=32 complete_output=262144 replay_bits=exact
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=32 round=0 order=[0, 1, 2, 3] route=0 wall_ns=9603914 profile=None
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=32 round=0 order=[0, 1, 2, 3] route=1 wall_ns=10011697 profile=None
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=32 round=0 order=[0, 1, 2, 3] route=2 wall_ns=9665699 profile=Some(Profile { validation_ns: 1353, transpose_ns: 92162, rows_wall_ns: 8692800, layout_ns: 878032, call_wall_ns: 9664477, residual_ns: 130, workers: [Work { rows: 4096, groups: 2048, decode_calls: 4096, helper_calls: 2048, decode_ns: 1577390, zero_ns: 48922, accumulate_ns: 6382878 }, Work { rows: 4096, groups: 2048, decode_calls: 4096, helper_calls: 2048, decode_ns: 1646626, zero_ns: 49842, accumulate_ns: 6707146 }] })
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=32 round=0 order=[0, 1, 2, 3] route=3 wall_ns=9483768 profile=Some(Profile { validation_ns: 390, transpose_ns: 67246, rows_wall_ns: 8522741, layout_ns: 892460, call_wall_ns: 9482997, residual_ns: 160, workers: [Work { rows: 4096, groups: 1024, decode_calls: 4096, helper_calls: 1024, decode_ns: 1550747, zero_ns: 27022, accumulate_ns: 6527407 }, Work { rows: 4096, groups: 1024, decode_calls: 4096, helper_calls: 1024, decode_ns: 1607733, zero_ns: 27807, accumulate_ns: 6668648 }] })
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=32 round=1 order=[0, 1, 2, 3] route=0 wall_ns=9191041 profile=None
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=32 round=1 order=[0, 1, 2, 3] route=1 wall_ns=9930354 profile=None
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=32 round=1 order=[0, 1, 2, 3] route=2 wall_ns=9908533 profile=Some(Profile { validation_ns: 751, transpose_ns: 56787, rows_wall_ns: 8825397, layout_ns: 1024767, call_wall_ns: 9907892, residual_ns: 190, workers: [Work { rows: 4096, groups: 2048, decode_calls: 4096, helper_calls: 2048, decode_ns: 1565827, zero_ns: 53906, accumulate_ns: 5882172 }, Work { rows: 4096, groups: 2048, decode_calls: 4096, helper_calls: 2048, decode_ns: 1696589, zero_ns: 50771, accumulate_ns: 6789569 }] })
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=32 round=1 order=[0, 1, 2, 3] route=3 wall_ns=9961592 profile=Some(Profile { validation_ns: 301, transpose_ns: 73808, rows_wall_ns: 8998451, layout_ns: 887971, call_wall_ns: 9960781, residual_ns: 250, workers: [Work { rows: 4096, groups: 1024, decode_calls: 4096, helper_calls: 1024, decode_ns: 1657308, zero_ns: 28244, accumulate_ns: 6987406 }, Work { rows: 4096, groups: 1024, decode_calls: 4096, helper_calls: 1024, decode_ns: 1735845, zero_ns: 28666, accumulate_ns: 7009619 }] })
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=32 round=2 order=[3, 2, 1, 0] route=3 wall_ns=9776757 profile=Some(Profile { validation_ns: 722, transpose_ns: 48871, rows_wall_ns: 8840876, layout_ns: 885467, call_wall_ns: 9776116, residual_ns: 180, workers: [Work { rows: 4096, groups: 1024, decode_calls: 4096, helper_calls: 1024, decode_ns: 1622415, zero_ns: 26734, accumulate_ns: 6766438 }, Work { rows: 4096, groups: 1024, decode_calls: 4096, helper_calls: 1024, decode_ns: 1702358, zero_ns: 27576, accumulate_ns: 6886789 }] })
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=32 round=2 order=[3, 2, 1, 0] route=2 wall_ns=8613060 profile=Some(Profile { validation_ns: 381, transpose_ns: 64590, rows_wall_ns: 7687969, layout_ns: 859047, call_wall_ns: 8612148, residual_ns: 161, workers: [Work { rows: 4096, groups: 2048, decode_calls: 4096, helper_calls: 2048, decode_ns: 1516030, zero_ns: 48101, accumulate_ns: 5759016 }, Work { rows: 4096, groups: 2048, decode_calls: 4096, helper_calls: 2048, decode_ns: 1587499, zero_ns: 48054, accumulate_ns: 5769686 }] })
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=32 round=2 order=[3, 2, 1, 0] route=1 wall_ns=9048024 profile=None
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=32 round=2 order=[3, 2, 1, 0] route=0 wall_ns=9068883 profile=None
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=32 round=3 order=[1, 2, 3, 0] route=1 wall_ns=9743424 profile=None
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=32 round=3 order=[1, 2, 3, 0] route=2 wall_ns=9115680 profile=Some(Profile { validation_ns: 892, transpose_ns: 56606, rows_wall_ns: 8139274, layout_ns: 918127, call_wall_ns: 9115070, residual_ns: 171, workers: [Work { rows: 4096, groups: 2048, decode_calls: 4096, helper_calls: 2048, decode_ns: 1579391, zero_ns: 49132, accumulate_ns: 5905511 }, Work { rows: 4096, groups: 2048, decode_calls: 4096, helper_calls: 2048, decode_ns: 1676573, zero_ns: 51044, accumulate_ns: 6108447 }] })
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=32 round=3 order=[1, 2, 3, 0] route=3 wall_ns=9584857 profile=Some(Profile { validation_ns: 391, transpose_ns: 86061, rows_wall_ns: 8587853, layout_ns: 909761, call_wall_ns: 9584256, residual_ns: 190, workers: [Work { rows: 4096, groups: 1024, decode_calls: 4096, helper_calls: 1024, decode_ns: 1564403, zero_ns: 27399, accumulate_ns: 6547054 }, Work { rows: 4096, groups: 1024, decode_calls: 4096, helper_calls: 1024, decode_ns: 1595241, zero_ns: 26982, accumulate_ns: 6750977 }] })
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=32 round=3 order=[1, 2, 3, 0] route=0 wall_ns=9792516 profile=None
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=32 round=4 order=[0, 3, 2, 1] route=0 wall_ns=8749355 profile=None
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=32 round=4 order=[0, 3, 2, 1] route=3 wall_ns=9411904 profile=Some(Profile { validation_ns: 531, transpose_ns: 68508, rows_wall_ns: 8454994, layout_ns: 887220, call_wall_ns: 9411403, residual_ns: 150, workers: [Work { rows: 4096, groups: 1024, decode_calls: 4096, helper_calls: 1024, decode_ns: 1580261, zero_ns: 29102, accumulate_ns: 6633432 }, Work { rows: 4096, groups: 1024, decode_calls: 4096, helper_calls: 1024, decode_ns: 1593495, zero_ns: 27259, accumulate_ns: 6610776 }] })
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=32 round=4 order=[0, 3, 2, 1] route=2 wall_ns=9330993 profile=Some(Profile { validation_ns: 341, transpose_ns: 57047, rows_wall_ns: 8199636, layout_ns: 1072175, call_wall_ns: 9329410, residual_ns: 211, workers: [Work { rows: 4096, groups: 2048, decode_calls: 4096, helper_calls: 2048, decode_ns: 1575991, zero_ns: 49001, accumulate_ns: 5818641 }, Work { rows: 4096, groups: 2048, decode_calls: 4096, helper_calls: 2048, decode_ns: 1708440, zero_ns: 51950, accumulate_ns: 6146083 }] })
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=32 round=4 order=[0, 3, 2, 1] route=1 wall_ns=10177737 profile=None
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=32 round=5 order=[2, 3, 0, 1] route=2 wall_ns=8986879 profile=Some(Profile { validation_ns: 411, transpose_ns: 54131, rows_wall_ns: 8057451, layout_ns: 874165, call_wall_ns: 8986369, residual_ns: 211, workers: [Work { rows: 4096, groups: 2048, decode_calls: 4096, helper_calls: 2048, decode_ns: 1547278, zero_ns: 48500, accumulate_ns: 5826447 }, Work { rows: 4096, groups: 2048, decode_calls: 4096, helper_calls: 2048, decode_ns: 1668220, zero_ns: 50427, accumulate_ns: 6049019 }] })
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=32 round=5 order=[2, 3, 0, 1] route=3 wall_ns=9274818 profile=Some(Profile { validation_ns: 621, transpose_ns: 66264, rows_wall_ns: 8305815, layout_ns: 901306, call_wall_ns: 9274186, residual_ns: 180, workers: [Work { rows: 4096, groups: 1024, decode_calls: 4096, helper_calls: 1024, decode_ns: 1551812, zero_ns: 27264, accumulate_ns: 6526207 }, Work { rows: 4096, groups: 1024, decode_calls: 4096, helper_calls: 1024, decode_ns: 1583674, zero_ns: 27125, accumulate_ns: 6485749 }] })
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=32 round=5 order=[2, 3, 0, 1] route=0 wall_ns=10174751 profile=None
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=32 round=5 order=[2, 3, 0, 1] route=1 wall_ns=9135477 profile=None
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=32 round=6 order=[1, 0, 3, 2] route=1 wall_ns=9272864 profile=None
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=32 round=6 order=[1, 0, 3, 2] route=0 wall_ns=9184800 profile=None
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=32 round=6 order=[1, 0, 3, 2] route=3 wall_ns=9831078 profile=Some(Profile { validation_ns: 1322, transpose_ns: 130033, rows_wall_ns: 8808545, layout_ns: 890246, call_wall_ns: 9830337, residual_ns: 191, workers: [Work { rows: 4096, groups: 1024, decode_calls: 4096, helper_calls: 1024, decode_ns: 1579046, zero_ns: 27921, accumulate_ns: 6852120 }, Work { rows: 4096, groups: 1024, decode_calls: 4096, helper_calls: 1024, decode_ns: 1652777, zero_ns: 27810, accumulate_ns: 6902922 }] })
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=32 round=6 order=[1, 0, 3, 2] route=2 wall_ns=9107415 profile=Some(Profile { validation_ns: 250, transpose_ns: 55755, rows_wall_ns: 8172155, layout_ns: 878694, call_wall_ns: 9107054, residual_ns: 200, workers: [Work { rows: 4096, groups: 2048, decode_calls: 4096, helper_calls: 2048, decode_ns: 1583385, zero_ns: 49163, accumulate_ns: 5853688 }, Work { rows: 4096, groups: 2048, decode_calls: 4096, helper_calls: 2048, decode_ns: 1695544, zero_ns: 51432, accumulate_ns: 6133263 }] })
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=32 round=7 order=[3, 0, 1, 2] route=3 wall_ns=9899386 profile=Some(Profile { validation_ns: 230, transpose_ns: 62307, rows_wall_ns: 8917680, layout_ns: 918568, call_wall_ns: 9898995, residual_ns: 210, workers: [Work { rows: 4096, groups: 1024, decode_calls: 4096, helper_calls: 1024, decode_ns: 1585671, zero_ns: 27281, accumulate_ns: 6596137 }, Work { rows: 4096, groups: 1024, decode_calls: 4096, helper_calls: 1024, decode_ns: 1668135, zero_ns: 31570, accumulate_ns: 6997826 }] })
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=32 round=7 order=[3, 0, 1, 2] route=0 wall_ns=9641714 profile=None
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=32 round=7 order=[3, 0, 1, 2] route=1 wall_ns=9279526 profile=None
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=32 round=7 order=[3, 0, 1, 2] route=2 wall_ns=8953426 profile=Some(Profile { validation_ns: 350, transpose_ns: 57568, rows_wall_ns: 7995515, layout_ns: 899372, call_wall_ns: 8953016, residual_ns: 211, workers: [Work { rows: 4096, groups: 2048, decode_calls: 4096, helper_calls: 2048, decode_ns: 1547878, zero_ns: 48882, accumulate_ns: 5749438 }, Work { rows: 4096, groups: 2048, decode_calls: 4096, helper_calls: 2048, decode_ns: 1659768, zero_ns: 50124, accumulate_ns: 5992630 }] })
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=32 round=8 order=[2, 1, 0, 3] route=2 wall_ns=9848631 profile=Some(Profile { validation_ns: 160, transpose_ns: 47419, rows_wall_ns: 8798246, layout_ns: 1002154, call_wall_ns: 9848220, residual_ns: 241, workers: [Work { rows: 4096, groups: 2048, decode_calls: 4096, helper_calls: 2048, decode_ns: 1598025, zero_ns: 49680, accumulate_ns: 6282437 }, Work { rows: 4096, groups: 2048, decode_calls: 4096, helper_calls: 2048, decode_ns: 1729804, zero_ns: 51164, accumulate_ns: 6704906 }] })
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=32 round=8 order=[2, 1, 0, 3] route=1 wall_ns=9246735 profile=None
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=32 round=8 order=[2, 1, 0, 3] route=0 wall_ns=8984314 profile=None
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=32 round=8 order=[2, 1, 0, 3] route=3 wall_ns=9573075 profile=Some(Profile { validation_ns: 421, transpose_ns: 66995, rows_wall_ns: 8599174, layout_ns: 905263, call_wall_ns: 9572334, residual_ns: 481, workers: [Work { rows: 4096, groups: 1024, decode_calls: 4096, helper_calls: 1024, decode_ns: 1570543, zero_ns: 27097, accumulate_ns: 6660051 }, Work { rows: 4096, groups: 1024, decode_calls: 4096, helper_calls: 1024, decode_ns: 1669465, zero_ns: 28089, accumulate_ns: 6682922 }] })
STAGE_GATE kind=Q6_K batch=32 complete_output=65536 replay_bits=exact
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=32 round=0 order=[0, 1, 2, 3] route=0 wall_ns=11620235 profile=None
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=32 round=0 order=[0, 1, 2, 3] route=1 wall_ns=11259740 profile=None
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=32 round=0 order=[0, 1, 2, 3] route=2 wall_ns=12272886 profile=Some(Profile { validation_ns: 481, transpose_ns: 506818, rows_wall_ns: 11553971, layout_ns: 210894, call_wall_ns: 12272384, residual_ns: 220, workers: [Work { rows: 1024, groups: 512, decode_calls: 1024, helper_calls: 512, decode_ns: 3338267, zero_ns: 14640, accumulate_ns: 6950423 }, Work { rows: 1024, groups: 512, decode_calls: 1024, helper_calls: 512, decode_ns: 3515422, zero_ns: 15648, accumulate_ns: 7913457 }] })
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=32 round=0 order=[0, 1, 2, 3] route=3 wall_ns=11044949 profile=Some(Profile { validation_ns: 301, transpose_ns: 550269, rows_wall_ns: 10303603, layout_ns: 190185, call_wall_ns: 11044559, residual_ns: 201, workers: [Work { rows: 1024, groups: 256, decode_calls: 1024, helper_calls: 256, decode_ns: 3342575, zero_ns: 7721, accumulate_ns: 6877567 }, Work { rows: 1024, groups: 256, decode_calls: 1024, helper_calls: 256, decode_ns: 3211914, zero_ns: 9094, accumulate_ns: 6653344 }] })
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=32 round=1 order=[0, 1, 2, 3] route=0 wall_ns=11230136 profile=None
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=32 round=1 order=[0, 1, 2, 3] route=1 wall_ns=12545496 profile=None
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=32 round=1 order=[0, 1, 2, 3] route=2 wall_ns=10785273 profile=Some(Profile { validation_ns: 511, transpose_ns: 563714, rows_wall_ns: 10016936, layout_ns: 203361, call_wall_ns: 10784712, residual_ns: 190, workers: [Work { rows: 1024, groups: 512, decode_calls: 1024, helper_calls: 512, decode_ns: 3276268, zero_ns: 14882, accumulate_ns: 6636067 }, Work { rows: 1024, groups: 512, decode_calls: 1024, helper_calls: 512, decode_ns: 3166086, zero_ns: 14598, accumulate_ns: 6609396 }] })
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=32 round=1 order=[0, 1, 2, 3] route=3 wall_ns=11446590 profile=Some(Profile { validation_ns: 281, transpose_ns: 547774, rows_wall_ns: 10703711, layout_ns: 194273, call_wall_ns: 11446189, residual_ns: 150, workers: [Work { rows: 1024, groups: 256, decode_calls: 1024, helper_calls: 256, decode_ns: 3420793, zero_ns: 8037, accumulate_ns: 7204564 }, Work { rows: 1024, groups: 256, decode_calls: 1024, helper_calls: 256, decode_ns: 3308584, zero_ns: 9564, accumulate_ns: 7083643 }] })
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=32 round=2 order=[3, 2, 1, 0] route=3 wall_ns=11342896 profile=Some(Profile { validation_ns: 180, transpose_ns: 302776, rows_wall_ns: 10837191, layout_ns: 202308, call_wall_ns: 11342595, residual_ns: 140, workers: [Work { rows: 1024, groups: 256, decode_calls: 1024, helper_calls: 256, decode_ns: 3447606, zero_ns: 7381, accumulate_ns: 7312587 }, Work { rows: 1024, groups: 256, decode_calls: 1024, helper_calls: 256, decode_ns: 3319953, zero_ns: 8788, accumulate_ns: 6974478 }] })
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=32 round=2 order=[3, 2, 1, 0] route=2 wall_ns=12035682 profile=Some(Profile { validation_ns: 711, transpose_ns: 540922, rows_wall_ns: 11286501, layout_ns: 207087, call_wall_ns: 12035371, residual_ns: 150, workers: [Work { rows: 1024, groups: 512, decode_calls: 1024, helper_calls: 512, decode_ns: 3714808, zero_ns: 16326, accumulate_ns: 7464041 }, Work { rows: 1024, groups: 512, decode_calls: 1024, helper_calls: 512, decode_ns: 3325330, zero_ns: 15755, accumulate_ns: 7302494 }] })
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=32 round=2 order=[3, 2, 1, 0] route=1 wall_ns=12046532 profile=None
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=32 round=2 order=[3, 2, 1, 0] route=0 wall_ns=11652776 profile=None
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=32 round=3 order=[1, 2, 3, 0] route=1 wall_ns=11946224 profile=None
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=32 round=3 order=[1, 2, 3, 0] route=2 wall_ns=11503186 profile=Some(Profile { validation_ns: 1042, transpose_ns: 637552, rows_wall_ns: 10665449, layout_ns: 198321, call_wall_ns: 11502565, residual_ns: 201, workers: [Work { rows: 1024, groups: 512, decode_calls: 1024, helper_calls: 512, decode_ns: 3547184, zero_ns: 14443, accumulate_ns: 7014887 }, Work { rows: 1024, groups: 512, decode_calls: 1024, helper_calls: 512, decode_ns: 3273598, zero_ns: 14985, accumulate_ns: 6822090 }] })
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=32 round=3 order=[1, 2, 3, 0] route=3 wall_ns=11450668 profile=Some(Profile { validation_ns: 190, transpose_ns: 525053, rows_wall_ns: 10726162, layout_ns: 198892, call_wall_ns: 11450467, residual_ns: 170, workers: [Work { rows: 1024, groups: 256, decode_calls: 1024, helper_calls: 256, decode_ns: 3464940, zero_ns: 8130, accumulate_ns: 7183660 }, Work { rows: 1024, groups: 256, decode_calls: 1024, helper_calls: 256, decode_ns: 3387367, zero_ns: 9122, accumulate_ns: 7056570 }] })
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=32 round=3 order=[1, 2, 3, 0] route=0 wall_ns=12353426 profile=None
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=32 round=4 order=[0, 3, 2, 1] route=0 wall_ns=11566735 profile=None
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=32 round=4 order=[0, 3, 2, 1] route=3 wall_ns=11475154 profile=Some(Profile { validation_ns: 471, transpose_ns: 548676, rows_wall_ns: 10726954, layout_ns: 198331, call_wall_ns: 11474632, residual_ns: 200, workers: [Work { rows: 1024, groups: 256, decode_calls: 1024, helper_calls: 256, decode_ns: 3455113, zero_ns: 7622, accumulate_ns: 7190270 }, Work { rows: 1024, groups: 256, decode_calls: 1024, helper_calls: 256, decode_ns: 3356152, zero_ns: 8627, accumulate_ns: 6968846 }] })
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=32 round=4 order=[0, 3, 2, 1] route=2 wall_ns=12183789 profile=Some(Profile { validation_ns: 631, transpose_ns: 598039, rows_wall_ns: 11390455, layout_ns: 193942, call_wall_ns: 12183258, residual_ns: 191, workers: [Work { rows: 1024, groups: 512, decode_calls: 1024, helper_calls: 512, decode_ns: 3741711, zero_ns: 14279, accumulate_ns: 7544115 }, Work { rows: 1024, groups: 512, decode_calls: 1024, helper_calls: 512, decode_ns: 3351176, zero_ns: 14923, accumulate_ns: 7648985 }] })
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=32 round=4 order=[0, 3, 2, 1] route=1 wall_ns=11423076 profile=None
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=32 round=5 order=[2, 3, 0, 1] route=2 wall_ns=11600618 profile=Some(Profile { validation_ns: 441, transpose_ns: 404416, rows_wall_ns: 10966292, layout_ns: 228838, call_wall_ns: 11600187, residual_ns: 200, workers: [Work { rows: 1024, groups: 512, decode_calls: 1024, helper_calls: 512, decode_ns: 3613091, zero_ns: 15453, accumulate_ns: 7241589 }, Work { rows: 1024, groups: 512, decode_calls: 1024, helper_calls: 512, decode_ns: 3292754, zero_ns: 15556, accumulate_ns: 6988290 }] })
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=32 round=5 order=[2, 3, 0, 1] route=3 wall_ns=12299275 profile=Some(Profile { validation_ns: 221, transpose_ns: 767455, rows_wall_ns: 11331204, layout_ns: 199764, call_wall_ns: 12298814, residual_ns: 170, workers: [Work { rows: 1024, groups: 256, decode_calls: 1024, helper_calls: 256, decode_ns: 3512182, zero_ns: 7918, accumulate_ns: 7516735 }, Work { rows: 1024, groups: 256, decode_calls: 1024, helper_calls: 256, decode_ns: 3479911, zero_ns: 9384, accumulate_ns: 7733122 }] })
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=32 round=5 order=[2, 3, 0, 1] route=0 wall_ns=11883398 profile=None
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=32 round=5 order=[2, 3, 0, 1] route=1 wall_ns=12601631 profile=None
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=32 round=6 order=[1, 0, 3, 2] route=1 wall_ns=11496905 profile=None
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=32 round=6 order=[1, 0, 3, 2] route=0 wall_ns=11832863 profile=None
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=32 round=6 order=[1, 0, 3, 2] route=3 wall_ns=11906611 profile=Some(Profile { validation_ns: 811, transpose_ns: 524942, rows_wall_ns: 11182937, layout_ns: 197179, call_wall_ns: 11906050, residual_ns: 181, workers: [Work { rows: 1024, groups: 256, decode_calls: 1024, helper_calls: 256, decode_ns: 3547045, zero_ns: 9106, accumulate_ns: 7543502 }, Work { rows: 1024, groups: 256, decode_calls: 1024, helper_calls: 256, decode_ns: 3522895, zero_ns: 9698, accumulate_ns: 7553004 }] })
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=32 round=6 order=[1, 0, 3, 2] route=2 wall_ns=12957356 profile=Some(Profile { validation_ns: 561, transpose_ns: 752117, rows_wall_ns: 12000888, layout_ns: 202749, call_wall_ns: 12956525, residual_ns: 210, workers: [Work { rows: 1024, groups: 512, decode_calls: 1024, helper_calls: 512, decode_ns: 3727077, zero_ns: 15558, accumulate_ns: 8157858 }, Work { rows: 1024, groups: 512, decode_calls: 1024, helper_calls: 512, decode_ns: 3758990, zero_ns: 15722, accumulate_ns: 7709968 }] })
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=32 round=7 order=[3, 0, 1, 2] route=3 wall_ns=11645312 profile=Some(Profile { validation_ns: 681, transpose_ns: 575677, rows_wall_ns: 10868760, layout_ns: 199524, call_wall_ns: 11644882, residual_ns: 240, workers: [Work { rows: 1024, groups: 256, decode_calls: 1024, helper_calls: 256, decode_ns: 3469157, zero_ns: 8212, accumulate_ns: 7316466 }, Work { rows: 1024, groups: 256, decode_calls: 1024, helper_calls: 256, decode_ns: 3400348, zero_ns: 8984, accumulate_ns: 7074343 }] })
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=32 round=7 order=[3, 0, 1, 2] route=0 wall_ns=12037206 profile=None
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=32 round=7 order=[3, 0, 1, 2] route=1 wall_ns=11399483 profile=None
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=32 round=7 order=[3, 0, 1, 2] route=2 wall_ns=11524907 profile=Some(Profile { validation_ns: 922, transpose_ns: 530903, rows_wall_ns: 10793459, layout_ns: 198902, call_wall_ns: 11524406, residual_ns: 220, workers: [Work { rows: 1024, groups: 512, decode_calls: 1024, helper_calls: 512, decode_ns: 3588985, zero_ns: 14322, accumulate_ns: 7103312 }, Work { rows: 1024, groups: 512, decode_calls: 1024, helper_calls: 512, decode_ns: 3265200, zero_ns: 14878, accumulate_ns: 7152263 }] })
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=32 round=8 order=[2, 1, 0, 3] route=2 wall_ns=11499549 profile=Some(Profile { validation_ns: 170, transpose_ns: 301354, rows_wall_ns: 10986570, layout_ns: 210424, call_wall_ns: 11498688, residual_ns: 170, workers: [Work { rows: 1024, groups: 512, decode_calls: 1024, helper_calls: 512, decode_ns: 3755223, zero_ns: 14284, accumulate_ns: 7121249 }, Work { rows: 1024, groups: 512, decode_calls: 1024, helper_calls: 512, decode_ns: 3350100, zero_ns: 14922, accumulate_ns: 6937545 }] })
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=32 round=8 order=[2, 1, 0, 3] route=1 wall_ns=11914365 profile=None
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=32 round=8 order=[2, 1, 0, 3] route=0 wall_ns=11988384 profile=None
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=32 round=8 order=[2, 1, 0, 3] route=3 wall_ns=11703210 profile=Some(Profile { validation_ns: 701, transpose_ns: 591126, rows_wall_ns: 10904346, layout_ns: 206285, call_wall_ns: 11702710, residual_ns: 252, workers: [Work { rows: 1024, groups: 256, decode_calls: 1024, helper_calls: 256, decode_ns: 3500760, zero_ns: 8484, accumulate_ns: 7202722 }, Work { rows: 1024, groups: 256, decode_calls: 1024, helper_calls: 256, decode_ns: 3358583, zero_ns: 8449, accumulate_ns: 7452374 }] })
CAPTURE_GATE batch=64 complete_logits=65536 bits=exact captured_formats=2 context=65 prompt_ids_activations=memory-only candidate_model_injection=false
STAGE_GATE kind=Q4_K batch=64 complete_output=524288 replay_bits=exact
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=64 round=0 order=[0, 1, 2, 3] route=0 wall_ns=25792047 profile=None
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=64 round=0 order=[0, 1, 2, 3] route=1 wall_ns=22088109 profile=None
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=64 round=0 order=[0, 1, 2, 3] route=2 wall_ns=24208896 profile=Some(Profile { validation_ns: 831, transpose_ns: 194334, rows_wall_ns: 21821690, layout_ns: 2190738, call_wall_ns: 24207834, residual_ns: 241, workers: [Work { rows: 4096, groups: 2048, decode_calls: 4096, helper_calls: 2048, decode_ns: 2134132, zero_ns: 68852, accumulate_ns: 19296250 }, Work { rows: 4096, groups: 2048, decode_calls: 4096, helper_calls: 2048, decode_ns: 2238751, zero_ns: 74093, accumulate_ns: 18848037 }] })
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=64 round=0 order=[0, 1, 2, 3] route=3 wall_ns=23181364 profile=Some(Profile { validation_ns: 521, transpose_ns: 171421, rows_wall_ns: 20766757, layout_ns: 2241904, call_wall_ns: 23180803, residual_ns: 200, workers: [Work { rows: 4096, groups: 1024, decode_calls: 4096, helper_calls: 1024, decode_ns: 2149872, zero_ns: 103170, accumulate_ns: 18182083 }, Work { rows: 4096, groups: 1024, decode_calls: 4096, helper_calls: 1024, decode_ns: 2030488, zero_ns: 81832, accumulate_ns: 16881187 }] })
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=64 round=1 order=[0, 1, 2, 3] route=0 wall_ns=29284741 profile=None
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=64 round=1 order=[0, 1, 2, 3] route=1 wall_ns=23005174 profile=None
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=64 round=1 order=[0, 1, 2, 3] route=2 wall_ns=25513756 profile=Some(Profile { validation_ns: 1182, transpose_ns: 199403, rows_wall_ns: 23082368, layout_ns: 2229371, call_wall_ns: 25512525, residual_ns: 201, workers: [Work { rows: 4096, groups: 2048, decode_calls: 4096, helper_calls: 2048, decode_ns: 2145111, zero_ns: 71220, accumulate_ns: 20191289 }, Work { rows: 4096, groups: 2048, decode_calls: 4096, helper_calls: 2048, decode_ns: 2235542, zero_ns: 82176, accumulate_ns: 20407702 }] })
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=64 round=1 order=[0, 1, 2, 3] route=3 wall_ns=20002837 profile=Some(Profile { validation_ns: 811, transpose_ns: 171642, rows_wall_ns: 17704167, layout_ns: 2125325, call_wall_ns: 20002136, residual_ns: 191, workers: [Work { rows: 4096, groups: 1024, decode_calls: 4096, helper_calls: 1024, decode_ns: 1892089, zero_ns: 81245, accumulate_ns: 15199169 }, Work { rows: 4096, groups: 1024, decode_calls: 4096, helper_calls: 1024, decode_ns: 1945883, zero_ns: 80670, accumulate_ns: 15373308 }] })
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=64 round=2 order=[3, 2, 1, 0] route=3 wall_ns=20190840 profile=Some(Profile { validation_ns: 871, transpose_ns: 135113, rows_wall_ns: 17946822, layout_ns: 2107162, call_wall_ns: 20190158, residual_ns: 190, workers: [Work { rows: 4096, groups: 1024, decode_calls: 4096, helper_calls: 1024, decode_ns: 1933642, zero_ns: 87554, accumulate_ns: 15297940 }, Work { rows: 4096, groups: 1024, decode_calls: 4096, helper_calls: 1024, decode_ns: 1934194, zero_ns: 74223, accumulate_ns: 15645427 }] })
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=64 round=2 order=[3, 2, 1, 0] route=2 wall_ns=24549563 profile=Some(Profile { validation_ns: 772, transpose_ns: 167062, rows_wall_ns: 22245543, layout_ns: 2134993, call_wall_ns: 24548601, residual_ns: 231, workers: [Work { rows: 4096, groups: 2048, decode_calls: 4096, helper_calls: 2048, decode_ns: 2141593, zero_ns: 68633, accumulate_ns: 19352195 }, Work { rows: 4096, groups: 2048, decode_calls: 4096, helper_calls: 2048, decode_ns: 2222030, zero_ns: 77605, accumulate_ns: 19601352 }] })
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=64 round=2 order=[3, 2, 1, 0] route=1 wall_ns=19739896 profile=None
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=64 round=2 order=[3, 2, 1, 0] route=0 wall_ns=24917411 profile=None
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=64 round=3 order=[1, 2, 3, 0] route=1 wall_ns=20682169 profile=None
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=64 round=3 order=[1, 2, 3, 0] route=2 wall_ns=23364897 profile=Some(Profile { validation_ns: 1313, transpose_ns: 189835, rows_wall_ns: 20943167, layout_ns: 2229010, call_wall_ns: 23363655, residual_ns: 330, workers: [Work { rows: 4096, groups: 2048, decode_calls: 4096, helper_calls: 2048, decode_ns: 1880545, zero_ns: 65363, accumulate_ns: 18529238 }, Work { rows: 4096, groups: 2048, decode_calls: 4096, helper_calls: 2048, decode_ns: 1931306, zero_ns: 74639, accumulate_ns: 18588447 }] })
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=64 round=3 order=[1, 2, 3, 0] route=3 wall_ns=20215385 profile=Some(Profile { validation_ns: 1403, transpose_ns: 224981, rows_wall_ns: 17920903, layout_ns: 2066777, call_wall_ns: 20214254, residual_ns: 190, workers: [Work { rows: 4096, groups: 1024, decode_calls: 4096, helper_calls: 1024, decode_ns: 1932429, zero_ns: 87248, accumulate_ns: 15597864 }, Work { rows: 4096, groups: 1024, decode_calls: 4096, helper_calls: 1024, decode_ns: 1947011, zero_ns: 80047, accumulate_ns: 15227897 }] })
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=64 round=3 order=[1, 2, 3, 0] route=0 wall_ns=24991490 profile=None
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=64 round=4 order=[0, 3, 2, 1] route=0 wall_ns=25081678 profile=None
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=64 round=4 order=[0, 3, 2, 1] route=3 wall_ns=20488356 profile=Some(Profile { validation_ns: 1082, transpose_ns: 201667, rows_wall_ns: 18280526, layout_ns: 2003378, call_wall_ns: 20486823, residual_ns: 170, workers: [Work { rows: 4096, groups: 1024, decode_calls: 4096, helper_calls: 1024, decode_ns: 1843918, zero_ns: 82635, accumulate_ns: 16064809 }, Work { rows: 4096, groups: 1024, decode_calls: 4096, helper_calls: 1024, decode_ns: 1797876, zero_ns: 74885, accumulate_ns: 15185986 }] })
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=64 round=4 order=[0, 3, 2, 1] route=2 wall_ns=24024962 profile=Some(Profile { validation_ns: 821, transpose_ns: 233046, rows_wall_ns: 21699091, layout_ns: 2091032, call_wall_ns: 24024210, residual_ns: 220, workers: [Work { rows: 4096, groups: 2048, decode_calls: 4096, helper_calls: 2048, decode_ns: 2110636, zero_ns: 67868, accumulate_ns: 19201295 }, Work { rows: 4096, groups: 2048, decode_calls: 4096, helper_calls: 2048, decode_ns: 2115837, zero_ns: 74982, accumulate_ns: 18711867 }] })
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=64 round=4 order=[0, 3, 2, 1] route=1 wall_ns=20853158 profile=None
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=64 round=5 order=[2, 3, 0, 1] route=2 wall_ns=25270512 profile=Some(Profile { validation_ns: 1142, transpose_ns: 175017, rows_wall_ns: 23016997, layout_ns: 2076284, call_wall_ns: 25269670, residual_ns: 230, workers: [Work { rows: 4096, groups: 2048, decode_calls: 4096, helper_calls: 2048, decode_ns: 2156153, zero_ns: 72177, accumulate_ns: 20450552 }, Work { rows: 4096, groups: 2048, decode_calls: 4096, helper_calls: 2048, decode_ns: 2108214, zero_ns: 76834, accumulate_ns: 19288655 }] })
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=64 round=5 order=[2, 3, 0, 1] route=3 wall_ns=19534653 profile=Some(Profile { validation_ns: 681, transpose_ns: 175368, rows_wall_ns: 17338835, layout_ns: 2018897, call_wall_ns: 19534021, residual_ns: 240, workers: [Work { rows: 4096, groups: 1024, decode_calls: 4096, helper_calls: 1024, decode_ns: 1886066, zero_ns: 82709, accumulate_ns: 14937233 }, Work { rows: 4096, groups: 1024, decode_calls: 4096, helper_calls: 1024, decode_ns: 1884402, zero_ns: 75383, accumulate_ns: 15094377 }] })
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=64 round=5 order=[2, 3, 0, 1] route=0 wall_ns=25143955 profile=None
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=64 round=5 order=[2, 3, 0, 1] route=1 wall_ns=19415318 profile=None
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=64 round=6 order=[1, 0, 3, 2] route=1 wall_ns=22108216 profile=None
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=64 round=6 order=[1, 0, 3, 2] route=0 wall_ns=27887306 profile=None
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=64 round=6 order=[1, 0, 3, 2] route=3 wall_ns=20475632 profile=Some(Profile { validation_ns: 1413, transpose_ns: 165810, rows_wall_ns: 18058160, layout_ns: 2249057, call_wall_ns: 20474640, residual_ns: 200, workers: [Work { rows: 4096, groups: 1024, decode_calls: 4096, helper_calls: 1024, decode_ns: 1968270, zero_ns: 103911, accumulate_ns: 15660017 }, Work { rows: 4096, groups: 1024, decode_calls: 4096, helper_calls: 1024, decode_ns: 1988883, zero_ns: 97761, accumulate_ns: 15511154 }] })
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=64 round=6 order=[1, 0, 3, 2] route=2 wall_ns=24874432 profile=Some(Profile { validation_ns: 861, transpose_ns: 195496, rows_wall_ns: 22563248, layout_ns: 2113374, call_wall_ns: 24873259, residual_ns: 280, workers: [Work { rows: 4096, groups: 2048, decode_calls: 4096, helper_calls: 2048, decode_ns: 2056574, zero_ns: 71053, accumulate_ns: 20104018 }, Work { rows: 4096, groups: 2048, decode_calls: 4096, helper_calls: 2048, decode_ns: 2058545, zero_ns: 79043, accumulate_ns: 19626958 }] })
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=64 round=7 order=[3, 0, 1, 2] route=3 wall_ns=20291880 profile=Some(Profile { validation_ns: 1022, transpose_ns: 171221, rows_wall_ns: 18086163, layout_ns: 2032533, call_wall_ns: 20291159, residual_ns: 220, workers: [Work { rows: 4096, groups: 1024, decode_calls: 4096, helper_calls: 1024, decode_ns: 1880348, zero_ns: 84983, accumulate_ns: 15056098 }, Work { rows: 4096, groups: 1024, decode_calls: 4096, helper_calls: 1024, decode_ns: 1905047, zero_ns: 84010, accumulate_ns: 15813040 }] })
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=64 round=7 order=[3, 0, 1, 2] route=0 wall_ns=25459767 profile=None
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=64 round=7 order=[3, 0, 1, 2] route=1 wall_ns=22991089 profile=None
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=64 round=7 order=[3, 0, 1, 2] route=2 wall_ns=25018271 profile=Some(Profile { validation_ns: 1423, transpose_ns: 195436, rows_wall_ns: 22720583, layout_ns: 2099547, call_wall_ns: 25017260, residual_ns: 271, workers: [Work { rows: 4096, groups: 2048, decode_calls: 4096, helper_calls: 2048, decode_ns: 2131599, zero_ns: 68781, accumulate_ns: 19729606 }, Work { rows: 4096, groups: 2048, decode_calls: 4096, helper_calls: 2048, decode_ns: 2214222, zero_ns: 79124, accumulate_ns: 20074339 }] })
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=64 round=8 order=[2, 1, 0, 3] route=2 wall_ns=23945153 profile=Some(Profile { validation_ns: 751, transpose_ns: 181630, rows_wall_ns: 21724339, layout_ns: 2037482, call_wall_ns: 23944432, residual_ns: 230, workers: [Work { rows: 4096, groups: 2048, decode_calls: 4096, helper_calls: 2048, decode_ns: 2087750, zero_ns: 67960, accumulate_ns: 19208313 }, Work { rows: 4096, groups: 2048, decode_calls: 4096, helper_calls: 2048, decode_ns: 2131703, zero_ns: 76222, accumulate_ns: 19178965 }] })
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=64 round=8 order=[2, 1, 0, 3] route=1 wall_ns=20577324 profile=None
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=64 round=8 order=[2, 1, 0, 3] route=0 wall_ns=24109390 profile=None
STAGE kind=Q4_K layer=0 role=gate rows=8192 cols=2048 batch=64 round=8 order=[2, 1, 0, 3] route=3 wall_ns=20194337 profile=Some(Profile { validation_ns: 872, transpose_ns: 188433, rows_wall_ns: 17865701, layout_ns: 2137829, call_wall_ns: 20193115, residual_ns: 280, workers: [Work { rows: 4096, groups: 1024, decode_calls: 4096, helper_calls: 1024, decode_ns: 1795407, zero_ns: 84819, accumulate_ns: 15673515 }, Work { rows: 4096, groups: 1024, decode_calls: 4096, helper_calls: 1024, decode_ns: 1821290, zero_ns: 90494, accumulate_ns: 14334379 }] })
STAGE_GATE kind=Q6_K batch=64 complete_output=131072 replay_bits=exact
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=64 round=0 order=[0, 1, 2, 3] route=0 wall_ns=28292256 profile=None
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=64 round=0 order=[0, 1, 2, 3] route=1 wall_ns=24740261 profile=None
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=64 round=0 order=[0, 1, 2, 3] route=2 wall_ns=28260125 profile=Some(Profile { validation_ns: 1032, transpose_ns: 1477754, rows_wall_ns: 26158935, layout_ns: 620591, call_wall_ns: 28258642, residual_ns: 330, workers: [Work { rows: 1024, groups: 512, decode_calls: 1024, helper_calls: 512, decode_ns: 4330425, zero_ns: 18373, accumulate_ns: 21689742 }, Work { rows: 1024, groups: 512, decode_calls: 1024, helper_calls: 512, decode_ns: 4398118, zero_ns: 18917, accumulate_ns: 20704515 }] })
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=64 round=0 order=[0, 1, 2, 3] route=3 wall_ns=23947718 profile=Some(Profile { validation_ns: 571, transpose_ns: 1087594, rows_wall_ns: 22141089, layout_ns: 717102, call_wall_ns: 23946646, residual_ns: 290, workers: [Work { rows: 1024, groups: 256, decode_calls: 1024, helper_calls: 256, decode_ns: 4045655, zero_ns: 11974, accumulate_ns: 17296376 }, Work { rows: 1024, groups: 256, decode_calls: 1024, helper_calls: 256, decode_ns: 4104881, zero_ns: 15818, accumulate_ns: 17881733 }] })
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=64 round=1 order=[0, 1, 2, 3] route=0 wall_ns=26084265 profile=None
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=64 round=1 order=[0, 1, 2, 3] route=1 wall_ns=23429348 profile=None
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=64 round=1 order=[0, 1, 2, 3] route=2 wall_ns=25442124 profile=Some(Profile { validation_ns: 852, transpose_ns: 818741, rows_wall_ns: 23924175, layout_ns: 697084, call_wall_ns: 25441122, residual_ns: 270, workers: [Work { rows: 1024, groups: 512, decode_calls: 1024, helper_calls: 512, decode_ns: 4153962, zero_ns: 18205, accumulate_ns: 19493926 }, Work { rows: 1024, groups: 512, decode_calls: 1024, helper_calls: 512, decode_ns: 4135239, zero_ns: 18181, accumulate_ns: 19643758 }] })
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=64 round=1 order=[0, 1, 2, 3] route=3 wall_ns=24715765 profile=Some(Profile { validation_ns: 701, transpose_ns: 1072817, rows_wall_ns: 22895260, layout_ns: 745765, call_wall_ns: 24714823, residual_ns: 280, workers: [Work { rows: 1024, groups: 256, decode_calls: 1024, helper_calls: 256, decode_ns: 4227299, zero_ns: 12709, accumulate_ns: 18549804 }, Work { rows: 1024, groups: 256, decode_calls: 1024, helper_calls: 256, decode_ns: 4259765, zero_ns: 17257, accumulate_ns: 17782250 }] })
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=64 round=2 order=[3, 2, 1, 0] route=3 wall_ns=24916431 profile=Some(Profile { validation_ns: 671, transpose_ns: 1052810, rows_wall_ns: 23187897, layout_ns: 673971, call_wall_ns: 24915669, residual_ns: 320, workers: [Work { rows: 1024, groups: 256, decode_calls: 1024, helper_calls: 256, decode_ns: 4317122, zero_ns: 13548, accumulate_ns: 18738226 }, Work { rows: 1024, groups: 256, decode_calls: 1024, helper_calls: 256, decode_ns: 4241431, zero_ns: 14805, accumulate_ns: 17841105 }] })
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=64 round=2 order=[3, 2, 1, 0] route=2 wall_ns=27104584 profile=Some(Profile { validation_ns: 621, transpose_ns: 946160, rows_wall_ns: 25601953, layout_ns: 554828, call_wall_ns: 27103812, residual_ns: 250, workers: [Work { rows: 1024, groups: 512, decode_calls: 1024, helper_calls: 512, decode_ns: 4299321, zero_ns: 18238, accumulate_ns: 21176555 }, Work { rows: 1024, groups: 512, decode_calls: 1024, helper_calls: 512, decode_ns: 4350725, zero_ns: 19203, accumulate_ns: 20602157 }] })
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=64 round=2 order=[3, 2, 1, 0] route=1 wall_ns=24645263 profile=None
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=64 round=2 order=[3, 2, 1, 0] route=0 wall_ns=28479295 profile=None
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=64 round=3 order=[1, 2, 3, 0] route=1 wall_ns=24551448 profile=None
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=64 round=3 order=[1, 2, 3, 0] route=2 wall_ns=26740112 profile=Some(Profile { validation_ns: 881, transpose_ns: 851984, rows_wall_ns: 25234817, layout_ns: 651328, call_wall_ns: 26739310, residual_ns: 300, workers: [Work { rows: 1024, groups: 512, decode_calls: 1024, helper_calls: 512, decode_ns: 4392494, zero_ns: 18023, accumulate_ns: 20714807 }, Work { rows: 1024, groups: 512, decode_calls: 1024, helper_calls: 512, decode_ns: 4237833, zero_ns: 18102, accumulate_ns: 19994815 }] })
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=64 round=3 order=[1, 2, 3, 0] route=3 wall_ns=23020955 profile=Some(Profile { validation_ns: 511, transpose_ns: 964965, rows_wall_ns: 21446629, layout_ns: 607757, call_wall_ns: 23020203, residual_ns: 341, workers: [Work { rows: 1024, groups: 256, decode_calls: 1024, helper_calls: 256, decode_ns: 4140233, zero_ns: 11596, accumulate_ns: 17184447 }, Work { rows: 1024, groups: 256, decode_calls: 1024, helper_calls: 256, decode_ns: 4013620, zero_ns: 13210, accumulate_ns: 16580596 }] })
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=64 round=3 order=[1, 2, 3, 0] route=0 wall_ns=25086979 profile=None
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=64 round=4 order=[0, 3, 2, 1] route=0 wall_ns=26359811 profile=None
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=64 round=4 order=[0, 3, 2, 1] route=3 wall_ns=24670831 profile=Some(Profile { validation_ns: 862, transpose_ns: 820625, rows_wall_ns: 23232791, layout_ns: 615552, call_wall_ns: 24670120, residual_ns: 290, workers: [Work { rows: 1024, groups: 256, decode_calls: 1024, helper_calls: 256, decode_ns: 4172216, zero_ns: 12053, accumulate_ns: 17604506 }, Work { rows: 1024, groups: 256, decode_calls: 1024, helper_calls: 256, decode_ns: 4205991, zero_ns: 14840, accumulate_ns: 18897749 }] })
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=64 round=4 order=[0, 3, 2, 1] route=2 wall_ns=25819349 profile=Some(Profile { validation_ns: 471, transpose_ns: 935230, rows_wall_ns: 24328140, layout_ns: 554237, call_wall_ns: 25818348, residual_ns: 270, workers: [Work { rows: 1024, groups: 512, decode_calls: 1024, helper_calls: 512, decode_ns: 4242531, zero_ns: 19045, accumulate_ns: 19923525 }, Work { rows: 1024, groups: 512, decode_calls: 1024, helper_calls: 512, decode_ns: 4189774, zero_ns: 19273, accumulate_ns: 19999712 }] })
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=64 round=4 order=[0, 3, 2, 1] route=1 wall_ns=21789861 profile=None
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=64 round=5 order=[2, 3, 0, 1] route=2 wall_ns=25806746 profile=Some(Profile { validation_ns: 921, transpose_ns: 924961, rows_wall_ns: 24274420, layout_ns: 605322, call_wall_ns: 25805895, residual_ns: 271, workers: [Work { rows: 1024, groups: 512, decode_calls: 1024, helper_calls: 512, decode_ns: 4082116, zero_ns: 16871, accumulate_ns: 19732771 }, Work { rows: 1024, groups: 512, decode_calls: 1024, helper_calls: 512, decode_ns: 4234262, zero_ns: 20137, accumulate_ns: 19882904 }] })
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=64 round=5 order=[2, 3, 0, 1] route=3 wall_ns=22632017 profile=Some(Profile { validation_ns: 781, transpose_ns: 1301755, rows_wall_ns: 20853410, layout_ns: 474758, call_wall_ns: 22630936, residual_ns: 232, workers: [Work { rows: 1024, groups: 256, decode_calls: 1024, helper_calls: 256, decode_ns: 3973023, zero_ns: 12046, accumulate_ns: 16649029 }, Work { rows: 1024, groups: 256, decode_calls: 1024, helper_calls: 256, decode_ns: 4028952, zero_ns: 14253, accumulate_ns: 16687847 }] })
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=64 round=5 order=[2, 3, 0, 1] route=0 wall_ns=26205071 profile=None
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=64 round=5 order=[2, 3, 0, 1] route=1 wall_ns=22115300 profile=None
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=64 round=6 order=[1, 0, 3, 2] route=1 wall_ns=21653026 profile=None
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=64 round=6 order=[1, 0, 3, 2] route=0 wall_ns=25464986 profile=None
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=64 round=6 order=[1, 0, 3, 2] route=3 wall_ns=22186675 profile=Some(Profile { validation_ns: 882, transpose_ns: 790819, rows_wall_ns: 20899537, layout_ns: 494264, call_wall_ns: 22185753, residual_ns: 251, workers: [Work { rows: 1024, groups: 256, decode_calls: 1024, helper_calls: 256, decode_ns: 3789483, zero_ns: 11634, accumulate_ns: 16995877 }, Work { rows: 1024, groups: 256, decode_calls: 1024, helper_calls: 256, decode_ns: 3752451, zero_ns: 12526, accumulate_ns: 16702691 }] })
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=64 round=6 order=[1, 0, 3, 2] route=2 wall_ns=27678627 profile=Some(Profile { validation_ns: 601, transpose_ns: 1912628, rows_wall_ns: 25236339, layout_ns: 528108, call_wall_ns: 27677906, residual_ns: 230, workers: [Work { rows: 1024, groups: 512, decode_calls: 1024, helper_calls: 512, decode_ns: 4235910, zero_ns: 17908, accumulate_ns: 20873390 }, Work { rows: 1024, groups: 512, decode_calls: 1024, helper_calls: 512, decode_ns: 4211460, zero_ns: 18450, accumulate_ns: 19956263 }] })
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=64 round=7 order=[3, 0, 1, 2] route=3 wall_ns=25519210 profile=Some(Profile { validation_ns: 1031, transpose_ns: 1308428, rows_wall_ns: 23700367, layout_ns: 508351, call_wall_ns: 25518429, residual_ns: 252, workers: [Work { rows: 1024, groups: 256, decode_calls: 1024, helper_calls: 256, decode_ns: 4544352, zero_ns: 14707, accumulate_ns: 19028516 }, Work { rows: 1024, groups: 256, decode_calls: 1024, helper_calls: 256, decode_ns: 4314455, zero_ns: 14616, accumulate_ns: 17755268 }] })
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=64 round=7 order=[3, 0, 1, 2] route=0 wall_ns=28032842 profile=None
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=64 round=7 order=[3, 0, 1, 2] route=1 wall_ns=22816023 profile=None
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=64 round=7 order=[3, 0, 1, 2] route=2 wall_ns=26283840 profile=Some(Profile { validation_ns: 781, transpose_ns: 1267301, rows_wall_ns: 24504671, layout_ns: 509824, call_wall_ns: 26282757, residual_ns: 180, workers: [Work { rows: 1024, groups: 512, decode_calls: 1024, helper_calls: 512, decode_ns: 3999380, zero_ns: 19369, accumulate_ns: 20380647 }, Work { rows: 1024, groups: 512, decode_calls: 1024, helper_calls: 512, decode_ns: 4030684, zero_ns: 17537, accumulate_ns: 19224618 }] })
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=64 round=8 order=[2, 1, 0, 3] route=2 wall_ns=26864205 profile=Some(Profile { validation_ns: 541, transpose_ns: 832027, rows_wall_ns: 25524479, layout_ns: 506287, call_wall_ns: 26863534, residual_ns: 200, workers: [Work { rows: 1024, groups: 512, decode_calls: 1024, helper_calls: 512, decode_ns: 4237218, zero_ns: 17803, accumulate_ns: 21167106 }, Work { rows: 1024, groups: 512, decode_calls: 1024, helper_calls: 512, decode_ns: 4176126, zero_ns: 17632, accumulate_ns: 19768804 }] })
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=64 round=8 order=[2, 1, 0, 3] route=1 wall_ns=23649532 profile=None
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=64 round=8 order=[2, 1, 0, 3] route=0 wall_ns=26615670 profile=None
STAGE kind=Q6_K layer=0 role=down rows=2048 cols=8192 batch=64 round=8 order=[2, 1, 0, 3] route=3 wall_ns=23035523 profile=Some(Profile { validation_ns: 1082, transpose_ns: 966027, rows_wall_ns: 21589628, layout_ns: 477764, call_wall_ns: 23034692, residual_ns: 191, workers: [Work { rows: 1024, groups: 256, decode_calls: 1024, helper_calls: 256, decode_ns: 3912273, zero_ns: 12693, accumulate_ns: 17562890 }, Work { rows: 1024, groups: 256, decode_calls: 1024, helper_calls: 256, decode_ns: 3745623, zero_ns: 11725, accumulate_ns: 16325490 }] })
```

## Existing capture diagnostic after helper extraction

Original three-call totals (routes0=ordinary,1=scratch,2=four-row), round0warmup
and rounds1–6 measured. These are a separate regression diagnostic, not pooled
with the one-call/four-route stage experiment.

```text
CAPTURE_GATE batch=32 complete_logits=65536 bits=exact captured_formats=2 context=65 prompt_ids_activations=memory-only candidate_model_injection=false
CAPTURE layer=0 role=gate kind=Q4_K rows=8192 cols=2048 batch=32 round=0 order=[0, 1, 2] times_ns=[26183609, 28948003, 34039995]
CAPTURE layer=0 role=gate kind=Q4_K rows=8192 cols=2048 batch=32 round=1 order=[0, 1, 2] times_ns=[26577195, 27407640, 29595836]
CAPTURE layer=0 role=gate kind=Q4_K rows=8192 cols=2048 batch=32 round=2 order=[2, 1, 0] times_ns=[27577358, 28334195, 29982329]
CAPTURE layer=0 role=gate kind=Q4_K rows=8192 cols=2048 batch=32 round=3 order=[1, 2, 0] times_ns=[27939625, 28373608, 29795549]
CAPTURE layer=0 role=gate kind=Q4_K rows=8192 cols=2048 batch=32 round=4 order=[0, 2, 1] times_ns=[26705345, 28272779, 29908291]
CAPTURE layer=0 role=gate kind=Q4_K rows=8192 cols=2048 batch=32 round=5 order=[2, 0, 1] times_ns=[26973587, 27975863, 29083967]
CAPTURE layer=0 role=gate kind=Q4_K rows=8192 cols=2048 batch=32 round=6 order=[1, 0, 2] times_ns=[27687553, 29584645, 32298564]
CAPTURE layer=0 role=down kind=Q6_K rows=2048 cols=8192 batch=32 round=0 order=[0, 1, 2] times_ns=[37411635, 39154118, 38678908]
CAPTURE layer=0 role=down kind=Q6_K rows=2048 cols=8192 batch=32 round=1 order=[0, 1, 2] times_ns=[39007303, 38618406, 38473404]
CAPTURE layer=0 role=down kind=Q6_K rows=2048 cols=8192 batch=32 round=2 order=[2, 1, 0] times_ns=[37512485, 38604068, 39065682]
CAPTURE layer=0 role=down kind=Q6_K rows=2048 cols=8192 batch=32 round=3 order=[1, 2, 0] times_ns=[37750641, 39014828, 39073298]
CAPTURE layer=0 role=down kind=Q6_K rows=2048 cols=8192 batch=32 round=4 order=[0, 2, 1] times_ns=[38057045, 36901602, 37872360]
CAPTURE layer=0 role=down kind=Q6_K rows=2048 cols=8192 batch=32 round=5 order=[2, 0, 1] times_ns=[39018375, 37201433, 38844058]
CAPTURE layer=0 role=down kind=Q6_K rows=2048 cols=8192 batch=32 round=6 order=[1, 0, 2] times_ns=[39211836, 37009925, 36636526]
CAPTURE_GATE batch=64 complete_logits=65536 bits=exact captured_formats=2 context=65 prompt_ids_activations=memory-only candidate_model_injection=false
CAPTURE layer=0 role=gate kind=Q4_K rows=8192 cols=2048 batch=64 round=0 order=[0, 1, 2] times_ns=[72767740, 74841712, 66731580]
CAPTURE layer=0 role=gate kind=Q4_K rows=8192 cols=2048 batch=64 round=1 order=[0, 1, 2] times_ns=[71754843, 74968730, 70050864]
CAPTURE layer=0 role=gate kind=Q4_K rows=8192 cols=2048 batch=64 round=2 order=[2, 1, 0] times_ns=[71798626, 73362212, 75050382]
CAPTURE layer=0 role=gate kind=Q4_K rows=8192 cols=2048 batch=64 round=3 order=[1, 2, 0] times_ns=[78656516, 77710532, 65936031]
CAPTURE layer=0 role=gate kind=Q4_K rows=8192 cols=2048 batch=64 round=4 order=[0, 2, 1] times_ns=[79492452, 68796408, 67267867]
CAPTURE layer=0 role=gate kind=Q4_K rows=8192 cols=2048 batch=64 round=5 order=[2, 0, 1] times_ns=[71267815, 71015603, 64389359]
CAPTURE layer=0 role=gate kind=Q4_K rows=8192 cols=2048 batch=64 round=6 order=[1, 0, 2] times_ns=[68466070, 71806443, 62855227]
CAPTURE layer=0 role=down kind=Q6_K rows=2048 cols=8192 batch=64 round=0 order=[0, 1, 2] times_ns=[81642958, 82415344, 68950197]
CAPTURE layer=0 role=down kind=Q6_K rows=2048 cols=8192 batch=64 round=1 order=[0, 1, 2] times_ns=[81621668, 82026109, 67540861]
CAPTURE layer=0 role=down kind=Q6_K rows=2048 cols=8192 batch=64 round=2 order=[2, 1, 0] times_ns=[79345862, 85730063, 68266280]
CAPTURE layer=0 role=down kind=Q6_K rows=2048 cols=8192 batch=64 round=3 order=[1, 2, 0] times_ns=[79618382, 77099175, 68144541]
CAPTURE layer=0 role=down kind=Q6_K rows=2048 cols=8192 batch=64 round=4 order=[0, 2, 1] times_ns=[79498798, 78615654, 67617475]
CAPTURE layer=0 role=down kind=Q6_K rows=2048 cols=8192 batch=64 round=5 order=[2, 0, 1] times_ns=[81580475, 82225375, 66525330]
CAPTURE layer=0 role=down kind=Q6_K rows=2048 cols=8192 batch=64 round=6 order=[1, 0, 2] times_ns=[83053567, 83815573, 66901208]
```

## Ideas, sources and next decision

[Approved design](superpowers/specs/2026-10-10-q4-stage-replay-design.md) and
[implementation plan](superpowers/plans/2026-10-10-q4-stage-replay.md).
Reuse Mivi's existing checked scratch/four-row APIs and production-helper capture.
[Projection-cost evidence](PROJECTION_COST_EVIDENCE_2026-10-05.md) supplies the
wall-versus-worker accounting distinction; [v0.2.78 evidence](FOUR_ROW_ISOLATION_CAPTURE_EVIDENCE_2026-10-10.md)
motivates regression investigation. Balanced comparisons, cache declarations and
retained negatives follow [Colibri benchmarking](https://github.com/JustVugg/colibri/blob/main/docs/benchmarking.md).
No external inference implementation copied.

Next narrow decision: obtain repeatable uninstrumented batch32 behavior or use
bounded hardware sampling/compiled-code inspection before changing register
blocking. No kernel change is justified as a proven fix by this pilot alone.
Separate model-workspace integration, candidate-integrated model/KV/SSM parity,
agent first-output/tool-loop latency, RSS and broader model/CPU coverage remain
open promotion gates. Candidate stays default-off.
