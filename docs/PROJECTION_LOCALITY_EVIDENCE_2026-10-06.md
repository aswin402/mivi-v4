# Projection locality experiment evidence — historical and fresh pilots

> Evidence status: the fresh schema-3 bounded pilot below is the sole evidence for the column-panel comparison. Earlier corrected-rerun figures in this document are historical; withdrawn token-width measurements remain historical context and are not evidence for this comparison. No private model, tensor, file-path, binary, activation, or raw-log identities are included.

## Historical corrected token-width pilot — 2026-10-07 (not column-panel evidence)

**Historical status only.** The earlier pilot was withdrawn after review found that its ordinary SIMD baseline had entered the experimental tile traversal. Those timings are excluded and were not reused. The corrected token-width run below was valid for that superseded token-width experiment, but it is not evidence for the redesigned column-panel variants. A separate historical pair rejected by the earlier timer-contract validation also remains excluded.

## Result

**No promotion.** Every explicit selector was slower than baseline in this corrected run. On the real-weight Q6_K workload, selector medians were 2.34–2.39× baseline; paired ratios stayed above 2.30× in all repetitions. The synthetic Q8_0 control was also slower for every selector. Keep ordinary production dispatch on the restored baseline and investigate the experimental path's slowdown before any broader or end-to-end performance evaluation.

## Scope and routing provenance

The build used source baseline `c42e588`, plus the corrected, uncommitted restoration of the ordinary single-row and paired AVX2 functions and wrappers. The opt-in release example was rebuilt after those source edits. Its binary identity was checked privately; executable and model hashes and local paths are retained only in the private session artifacts. No code, version, changelog, roadmap, commit, or push action was part of this rerun.

Source inspection confirmed the runtime routing: a null selector in both profiled and unprofiled measurements calls the ordinary checked projection API, whose SIMD wrappers dispatch to the restored ordinary AVX2 functions. Explicit selectors 32, 64, and 128 call the feature-gated tile-aware API. The rebuilt executable's feature fingerprint includes the locality experiment. All returned records retained the expected selector metadata, and the measured results showed the selector path's distinct timing behavior.

The opt-in release build command was:

```text
CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 cargo build -p mivi-model --example projection_measure --features projection-locality-experiment --release --offline
```

The paired driver commands were:

```text
python3 scripts/runtime_compare/projection_measure.py --manifest <PRIVATE_SESSION>/manifest.json --output-dir <PRIVATE_SESSION>/run --validate-only
python3 scripts/runtime_compare/projection_measure.py --manifest <PRIVATE_SESSION>/manifest.json --output-dir <PRIVATE_SESSION>/run
```

The descriptor-only preflight re-read the local GGUF metadata and confirmed Q6_K, 2,048 rows × 8,192 columns. The synthetic control was Q8_0, 257 rows × 256 columns. Both workloads used batch 64. The report intentionally omits model and tensor identity.

Each workload had one null-selector baseline and explicit 32/64/128 members. Every member used one warmup, one measured call per child, three paired repetitions, two threads, and both profiled and unprofiled modes. Workload-member order alternated between repetitions, as did mode order within each pair. The synthetic activations were deterministic; no captured activations or model forward pass were used.

## Bounds, host, and caveats

The schema-2 manifest passed `--validate-only`. Descriptor-aware preflight predicted 53,034,268 aggregate artifact bytes, below the 67,108,864-byte cap, before any child launched. Limits were 180 seconds per child, 2 GiB sampled process-tree RSS, and 900 seconds per session. Children ran sequentially. No system cache flush was performed.

The host was x86_64 Linux on an AMD Ryzen 7 7730U with 16 logical CPUs and 16 GiB RAM. A post-run snapshot showed load averages 0.62/1.32/1.26, 9.84 GiB available memory, and 2.26 GB of 4 GiB swap in use. An immediately pre-run load/swap snapshot was not captured, so the post-run swap level cannot be attributed to this measurement. The available `acpitz` sensor read 44°C after the run; that is not a direct CPU-package or throttling measurement. The supervisor sampled owned process-tree RSS every 100 ms; no RSS-limit event occurred, though brief peaks between samples may be missed. Driver-reported elapsed time was 16.67 seconds, below the session limit. Unknown cache residency, host activity, swap pressure, and thermal behavior limit timing precision.

## Attempts and artifact validation

All 48 child attempts completed: 24 profiled and 24 unprofiled. All 24 profiled/unprofiled pairs were accepted. The driver also accepted all 18 baseline-to-selector comparisons. Independent post-run inspection compared every full `u32` output vector: all 24 within-member profiled/unprofiled pairs and all 36 baseline-to-selector vectors (both modes for each of 18 comparisons) were bit-identical. Shape, format, batch, branch, thread count, and comparison metadata matched. Alternating order was verified. All 48 child cleanups succeeded and reaped their process groups; there were zero failed or rejected attempts in this fresh session.

The old withdrawn session was left intact and its timings were not reused. The fresh private tree, including its manifest, per-attempt inputs/results, private reports, and provenance, occupies 39,110,361 bytes including the manifest, below the 67,108,864-byte cap. Permission audit found all directories at 0700 and all files at 0600, with zero violations. No artifacts were removed.

## Unprofiled full-call timings

These outer full-call wall times are the primary measurements. Values are median (minimum–maximum) across three paired repetitions, in nanoseconds. The ratio column is computed per repetition; values above 1.0 mean the explicit selector was slower than the paired baseline.

| Workload | Selector | Median (range), ns | Paired selector / baseline ratio, median (range) |
| --- | ---: | ---: | ---: |
| Synthetic Q8_0, 257 × 256, batch 64 | baseline | 252,251 (236,170–270,696) | — |
|  | 32 | 472,109 (468,072–635,792) | 1.856 (1.744–2.692) |
|  | 64 | 390,424 (383,992–452,713) | 1.522 (1.442–1.917) |
|  | 128 | 423,237 (416,274–487,278) | 1.763 (1.678–1.800) |
| Q6_K, 2,048 × 8,192, batch 64 | baseline | 50,354,918 (50,347,233–51,561,477) | — |
|  | 32 | 118,664,049 (118,002,940–119,117,745) | 2.343 (2.310–2.357) |
|  | 64 | 120,267,105 (118,719,385–120,656,496) | 2.388 (2.302–2.396) |
|  | 128 | 119,267,039 (117,657,833–119,608,319) | 2.337 (2.313–2.376) |

## Profiled stage diagnostics

These are diagnostic inner-kernel wall stages from profiled calls, not the primary timing result. Values are median (minimum–maximum), in nanoseconds, across three calls. The stages must not be summed to estimate median call time. `Rows wall` is elapsed parallel-region wall time; worker-work durations are not elapsed wall time. The private report retains the remaining profile fields.

| Workload | Selector | Buffer initialization | Input transpose | Rows wall | Output layout |
| --- | ---: | ---: | ---: | ---: | ---: |
| Synthetic Q8_0 | baseline | 104,920 (4,789–155,456) | 35,377 (34,496–43,012) | 209,218 (189,672–341,201) | 28,284 (24,026–39,525) |
|  | 32 | 5,711 (4,529–193,198) | 26,921 (26,120–33,604) | 381,457 (374,814–417,485) | 20,029 (19,708–24,116) |
|  | 64 | 19,628 (8,556–26,080) | 32,462 (26,490–63,742) | 382,839 (321,443–432,975) | 40,016 (24,317–46,980) |
|  | 128 | 7,634 (4,508–10,440) | 32,471 (27,363–33,945) | 404,009 (384,392–422,716) | 24,066 (19,498–33,805) |
| Q6_K | baseline | 90,171 (81,265–125,739) | 3,218,484 (2,885,258–4,267,803) | 45,992,875 (45,714,595–50,101,986) | 902,369 (890,076–927,207) |
|  | 32 | 85,513 (79,321–132,833) | 3,040,924 (2,952,406–3,491,543) | 112,620,203 (112,566,590–115,553,201) | 1,051,534 (906,677–1,099,426) |
|  | 64 | 102,185 (83,850–110,301) | 3,052,757 (2,823,661–3,097,122) | 113,915,941 (93,984,059–114,773,426) | 1,060,130 (898,993–1,117,159) |
|  | 128 | 91,214 (76,044–91,735) | 2,964,920 (2,714,482–3,007,741) | 114,150,218 (113,602,815–114,769,328) | 948,187 (901,508–1,166,243) |

The large real-weight gap appears in the profiled `Rows wall` stage as well as the unprofiled outer calls. These diagnostic values locate the divergence but do not explain its cause; profiling can perturb timings and does not isolate every cost.

## Limitations and next decision

This is an operator pilot using deterministic synthetic activations, including for the real-weight case. It does not measure real-model inference, captured-activation behavior, model quality, or agent utility, and it does not establish an end-to-end performance result.

**Decision: do not promote the selector path.** Keep the restored ordinary baseline as production dispatch and investigate the explicit selector path's repeatable slowdown before considering another locality pilot or any P1-A end-to-end evaluation.

## Fresh schema-3 bounded pilot — 2026-10-07

This section uses only the fresh schema-3 session. The committed implementation revision matched the manifest revision. Ordinary projection calls continue through the default selector path; explicit selectors are available only through the experiment feature. The default and explicit 128 paths reach the same `pair_panel_avx2::<128>` const-generic column loop.

### Workloads and acceptance

| Workload | Shape | Batch | Selectors |
|---|---:|---:|---|
| Synthetic Q8_0 | 257x256 | 64 | baseline, 32, 64, 128 |
| Descriptor-verified Q6_K | 2048x8192 | 64 | baseline, 32, 64, 128 |

Each case used one warmup call, one measured call per child, two threads, profiled and unprofiled modes, and three repetitions. The driver completed all 48 sequential attempts. All 24 profiled/unprofiled pairs and all 36 baseline-to-variant comparisons had exact full-vector output parity and matching non-selector work metadata. All child processes were reaped successfully.

### Paired unprofiled full-call timings

Milliseconds across three repetitions; selector rows show median [min, max]. Paired ratios are variant/baseline median [min, max].

| Workload | Selector | Time (ms) | Paired ratio |
|---|---:|---:|---:|
| Q8_0 synthetic 257x256 | baseline | 0.161 [0.146, 0.184] | 1.000 |
| Q8_0 synthetic 257x256 | 32 | 0.145 [0.137, 0.207] | 0.937 [0.901, 1.123] |
| Q8_0 synthetic 257x256 | 64 | 0.176 [0.172, 0.177] | 1.097 [0.954, 1.173] |
| Q8_0 synthetic 257x256 | 128 | 0.153 [0.141, 0.236] | 0.964 [0.952, 1.280] |
| Q6_K verified 2048x8192 | baseline | 23.873 [23.336, 29.856] | 1.000 |
| Q6_K verified 2048x8192 | 32 | 21.869 [21.221, 23.734] | 0.909 [0.795, 0.916] |
| Q6_K verified 2048x8192 | 64 | 22.420 [21.812, 25.794] | 0.935 [0.751, 1.080] |
| Q6_K verified 2048x8192 | 128 | 22.405 [21.479, 23.273] | 0.920 [0.750, 0.975] |

### Profile diagnostics

Median per-call milliseconds. Profiled timings are diagnostic only and do not contribute to the unprofiled ratios.

| Workload | Selector | Buffer init | Input transpose | Rows wall | Output layout |
|---|---:|---:|---:|---:|---:|
| Q8_0 synthetic | baseline | 0.006 | 0.020 | 0.157 | 0.015 |
| Q8_0 synthetic | 32 | 0.005 | 0.024 | 0.123 | 0.017 |
| Q8_0 synthetic | 64 | 0.004 | 0.016 | 0.131 | 0.013 |
| Q8_0 synthetic | 128 | 0.007 | 0.024 | 0.133 | 0.017 |
| Q6_K verified | baseline | 0.060 | 1.872 | 21.619 | 0.448 |
| Q6_K verified | 32 | 0.039 | 1.272 | 19.024 | 0.449 |
| Q6_K verified | 64 | 0.041 | 1.348 | 20.513 | 0.503 |
| Q6_K verified | 128 | 0.046 | 1.309 | 21.911 | 0.518 |

### Control, resources, and decision

The baseline max/min timing spread was 25.9% for Q8_0 and 27.9% for Q6_K. Paired 128/baseline ratios ranged from 0.952 to 1.280 for Q8_0 and 0.750 to 0.975 for Q6_K. One Q8_0 repetition was a high outlier, about two percentage points beyond the baseline max/min spread; Q6_K had a broad baseline spread and its 128 ratios remained within that spread. Source control flow converges on the same 128-column const-generic loop. The release executable was stripped, so its inlined kernel had no symbol table for symbol-scoped disassembly. The timing observations do not show a material control-flow discrepancy, but the short pilot and broad variation make performance conclusions low confidence.

The plan was 48 attempts, 180 seconds per child, 2 GiB sampled RSS, 900 seconds total, and a 64 MiB artifact cap. Total elapsed time was 8.829 seconds. Predicted artifacts were 53,034,316 bytes; actual run artifacts were 39,107,542 bytes. Permission audit covered 100 files and 49 subdirectories: all data files were mode 0600, directories mode 0700, and the owner-only executable mode 0700. The driver recorded Linux owned-process-tree sampled-RSS scope for every child and enforced the cap; it does not retain peak RSS values.

Host caveats: one host and three repetitions, without cache flushing or overlapping inference/build work. Scheduling and thermal variation were not controlled. No selector is promoted; this run establishes parity and bounded execution, while its timing data remains a low-confidence pilot.
