# Projection cost pilot evidence — 2026-10-05

Status: Task 4 cost measurement complete with 15 accepted pairs across three retained attempts. The original 21 timer-contract rejections and six corrected-run validation failures remain explicit; a separate targeted retry accepted all six pairs for those two cases. The evidence selects exactly one next step: a faithful locality experiment as an exploratory measurement slice. No kernel optimization, promotion, model inference, or cross-engine parity conclusion follows.

## Scope and provenance

This is an opt-in operator pilot for `projection-diagnostics`, not a model-inference benchmark. Each measured call used deterministic synthetic F32 activations. Selected Q6_K and Q4_K weights came from a local GGUF, but no real activations or model statistics were used.

Attempt 1 used an earlier `projection-diagnostics` release executable. Attempts 2 and 3 used the same verified release executable, built from revision `d19ba8a` (workspace version 0.2.68); the public report omits executable and model hashes. The earlier build emitted a warning for a helper later restricted to test builds. The kernel arithmetic source was not changed between these attempts, but the first and later executable files are distinct; no binary-identity claim is made for Attempt 1. The recorded build used Rust/Cargo 1.94.0, offline, one Cargo job, and `RAYON_NUM_THREADS=2`.

The host was x86_64 Linux on an AMD Ryzen 7 7730U with AVX2, FMA, and F16C, 16 logical CPUs, and 16 GiB RAM. Before the corrected run, load averages were about 3.46/3.06/2.19 and only about 200 KiB of 4 GiB swap was free; no thermal sensor was available. No cache flush was performed. These conditions and the small repetition count limit timing interpretation.

## Attempt 1 — original timer-contract failure

The first run used seven explicitly selected cases, including synthetic F32 3×64 batch 2, synthetic Q8_0 257×256 batch 64, FFN down Q6_K 2,048×8,192 at batches 64 and 9, short-convolution output Q4_K 2,048×2,048 at batches 64 and 9, and FFN gate Q4_K 8,192×2,048 at batch 32. It used three paired repetitions, one warmup and one measured call per child, two threads, a 180-second child wall limit, a 900-second session limit, 2 GiB sampled child-tree RSS, 128 MiB heap allowance, and a 64 MiB artifact cap.

All 42 child attempts completed and cleaned up. The driver accepted 0 of 21 profiled/unprofiled pairs: 21 unprofiled attempts were complete and all 21 profiled results were rejected because the driver required the outer `invoke` call timer to equal the inner diagnostic kernel timer. Those boundaries are nested and distinct. The run took 9.16 seconds and retained 38,212,276 bytes.

Post-run inspection of all retained raw outputs found exact bit equality for all 21 attempted pairs and finite values throughout. That inspection does not change the driver's statuses or make those pairs accepted. For historical context, the unprofiled-only medians below are descriptive observations from this failed attempt; they are not matched comparisons and are excluded from accepted timing summaries.

| Unprofiled-only case | Median (ns) | Range (ns) |
| --- | ---: | ---: |
| Synthetic serial F32, batch 2 | 9,458 | 6,362–12,193 |
| Synthetic odd-row parallel Q8_0, batch 64 | 197,502 | 129,083–252,066 |
| FFN down Q6_K, batch 64 | 26,415,134 | 22,263,580–26,506,456 |
| FFN down Q6_K, batch 9 | 30,578,641 | 30,409,651–30,741,337 |
| Short-convolution output Q4_K, batch 64 | 5,925,220 | 5,589,888–6,023,826 |
| Short-convolution output Q4_K, batch 9 | 7,276,871 | 7,151,975–7,760,223 |
| FFN gate Q4_K, batch 32 | 9,189,530 | 9,061,178–10,283,104 |

## Attempt 2 — corrected timer boundary, partial acceptance

The corrected manifest explicitly selected five cases: synthetic serial F32 3×64 batch 2; synthetic odd-row Q8_0 257×256 batch 9; actual FFN down Q6_K 2,048×8,192 at batches 64 and 9; and actual short-convolution output Q4_K 2,048×2,048 at batch 64. The odd-row batch was reduced from 64 to 9 to fit the combined artifact ceiling. The gate and short-convolution batch-9 cases were omitted. All cases used three paired repetitions, one warmup and one measured call per child, two threads, 180 seconds per child, 900 seconds per session, 2 GiB sampled child-tree RSS, 128 MiB heap allowance, and a 28,000,000-byte new-session artifact cap.

Descriptor-only validation succeeded without creating results or starting a child. The driver then retained all 30 attempts and exited 2: 24 samples were accepted as complete and six profiled batch-9 samples were `result_error`. Nine of fifteen pairs were validated and accepted for comparison; the six rejected pairs remain failures. All 30 process cleanups succeeded and reaped their children. Raw-output inspection found exact bit equality for all 15 attempted pairs and 1,697,370 finite values out of 1,697,370. The 9 matched pairs below alone contribute accepted timings; raw output agreement does not retroactively accept the six rejected pairs.

The remaining integration error is in the Python driver's profile-stage availability rule. It rejects any non-null `input_transpose_ns` when batch is below 32. The measured kernel executes the across-batch FMA path starting at batch 9 and reports its transpose duration there; paired row decode starts at batch 32. Thus the two batch-9 cases produce valid raw results but fail this incorrect nullability rule. The underlying release executable and kernel were unchanged. No further child was launched after identifying this issue.

| Validated matched case | Profiled outer call median (range), ns | Unprofiled outer call median (range), ns | Per-pair profiled/unprofiled ratio median (range) |
| --- | ---: | ---: | ---: |
| Synthetic serial F32, batch 2 | 14,637 (13,546–19,116) | 14,267 (13,225–21,190) | 0.9495 (0.9021–1.1068) |
| FFN down Q6_K, batch 64 | 53,239,841 (50,038,377–57,814,479) | 54,386,770 (51,697,429–58,608,393) | 1.0298 (0.8538–1.0630) |
| Short-convolution output Q4_K, batch 64 | 13,126,020 (12,307,990–13,707,394) | 13,839,502 (12,628,924–13,909,304) | 0.9746 (0.9484–0.9855) |

The ratio ranges show material run-to-run variation. They describe paired outer-call observations, not isolated profiler overhead; faster profiled samples do not establish negative overhead. The outer timer includes wrapper and worker-pool boundaries, while the profile timer is an inner kernel timer. Stage values below are medians and ranges from that inner timer across the three accepted profiled repetitions.

| Case | Inner call wall | Buffer initialization | Input transpose | Rows wall | Output layout | Unclassified wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FFN down Q6_K, batch 64 | 53,141,576 (49,907,260–57,751,731) | 98,866 (88,957–118,964) | 3,234,917 (3,105,113–3,298,367) | 49,045,049 (45,638,928–53,271,841) | 913,268 (900,925–1,081,235) | 911 (672–1,031) |
| Short-convolution output Q4_K, batch 64 | 13,041,240 (12,069,722–13,457,594) | 141,075 (141,035–161,193) | 597,014 (555,075–770,550) | 11,179,156 (10,422,322–11,654,881) | 950,248 (928,587–1,063,492) | 671 (511–1,072) |

These stage timers do not quantify every allocation, initialization, decode, accumulation, or output-layout cost. Per-worker decode and accumulation durations are work measurements, not elapsed wall time: they are not added to serial wall stages or used as wall-time percentages. Separately labeled worker-work sums appear below. Timer instrumentation can perturb short calls, and the measured activation distribution is synthetic even where weights are real.

The preserved failed session and corrected session occupy 56,963,575 bytes combined, below the 67,108,864-byte combined cap. The corrected session is 18,751,299 bytes, including manifest, results, and private provenance. Its initial manifest mode violation was corrected to 0600; a follow-up audit found no file or directory mode violations. The original failed session remains unchanged.

## Superseded handoff before the targeted retry

At the time of this earlier handoff, the targeted retry was pending. The completion addendum below records its result without changing either earlier attempt's statuses.

## Task 4 completion addendum

### Three-attempt accounting and final decision

Attempts 2 and 3 used the same 767,248-byte release executable from source revision `d19ba8a` (workspace 0.2.68), with feature `projection-diagnostics`; its private hash matched the approved record before the retry. Attempt 1 used a distinct earlier build (777,352 bytes) that included the then-unrestricted helper and emitted a warning. The kernel arithmetic source was unchanged across these builds, but the artifacts are not identical. No rebuild or full-model inference was performed for the retry: the example memory-mapped the GGUF file and accessed the selected weight tensor. Exact binary/model hashes and local paths are retained only in the private report.

| Attempt | Child statuses | Accepted pairs | Retained bytes | Driver elapsed |
| --- | --- | ---: | ---: | ---: |
| Original timer-contract run | 21 `complete`, 21 `result_error` (profiled outer/inner timer equality check) | 0/21 | 38,212,276 | 9.157543196 s |
| Corrected timer run | 24 `complete`, 6 `result_error` (transpose incorrectly rejected below batch 32) | 9/15 | 18,751,299 | 9.998891673 s |
| Targeted two-case retry | 12 `complete`, 0 errors | 6/6 | 1,408,714 | 3.273508263 s |

Across all three attempts, 57 of 84 child records have status `complete` and 27 retain their original `result_error` status. All 84 cleanups succeeded and reaped their children. The original attempt's retained outputs were separately inspected after the live run and found finite and bit-identical across its 21 pairs; that post-run check does not change its zero accepted pairs. The corrected attempt accepted nine pairs and retained six rejected pairs. The retry validated all six remaining pairs and exact bit agreement. Thus final accepted evidence is 15 pairs (nine from attempt 2 and six from attempt 3), never 21 from attempt 1.

The retry selected only synthetic Q8_0 257×256, batch 9, and actual FFN-down Q6_K 2,048×8,192, batch 9. The latter descriptor was freshly read from the local GGUF and matched the recorded model hash. Each case used three paired repetitions, one warmup and one measured call per child, and two threads. Limits were 180 seconds per child, 900 seconds per session, sampled child-tree RSS 2 GiB, heap buffer 128 MiB, model-file size limit 798,004,032 bytes (actual model size plus 64 MiB), and an 8,000,000-byte artifact cap. Validate-only returned `manifest valid`, made no results directory, and launched no child.

The original and corrected artifact trees were preserved unchanged. Final retained sizes are 38,212,276 + 18,751,299 + 1,408,714 = 58,372,289 bytes, below the combined 67,108,864-byte ceiling with 8,736,575 bytes remaining. The new driver's manifest, report, and child artifacts account for 1,404,660 bytes; private provenance adds 4,054 bytes. Before launch, the prior trees totaled 56,963,575 bytes; the new session's full 8,000,000-byte cap therefore fit under the global ceiling with 2,145,289 bytes to spare. Final recursive audit found every directory mode 0700 and every file mode 0600, with zero violations across all three trees.

Each driver's elapsed clock starts before binary/model descriptor hashing and GGUF descriptor inspection. Total driver elapsed for all three attempts was 22.429943132 seconds, including those per-attempt preflights and all attempts, and below the 900-second per-session limit. This corrects the earlier private report wording that described the first two measured sessions as excluding preflight. Separate manifest preparation and validate-only time are outside those driver clocks. Child RSS was sampled at 100 ms intervals over owned process trees, so peaks between samples may be missed. No cache flush was used. The host is an x86_64 AMD Ryzen 7 7730U with AVX2/FMA/F16C, 16 logical CPUs, 16 GiB RAM, Linux kernel 7.0.0-38-generic. A post-retry snapshot showed about 639 KiB free swap of 4 GiB and load averages 1.61/1.60/2.06; no thermal sensor was available. No thermal control was applied. The short runs, low free swap, changing host load, and unknown cache/thermal state limit timing precision.

### Accepted paired call timings

The driver elapsed field is sampled before final JSON/Markdown serialization and persistence, which it excludes. GGUF diagnostic mapping uses Linux descriptor pinning; selected model and executable files must remain quiescent. Preflight hashes do not freeze their contents or support concurrent in-place mutation. Default server portability and inference behavior are unchanged.

Only driver-accepted matched pairs contribute. Times are outer call-wall nanoseconds; ratio triplets are profiled/unprofiled for repetitions 0, 1, and 2 in that order. Medians use those three pairs; ranges show observed minimum–maximum.

| Case | Profiled median (range) | Unprofiled median (range) | Ratio median (range); per-repetition ratios |
| --- | ---: | ---: | --- |
| Synthetic F32 3×64, batch 2 | 14,637 (13,546–19,116) | 14,267 (13,225–21,190) | 0.9495 (0.9021–1.1068); 0.9495, 1.1068, 0.9021 |
| Synthetic Q8_0 257×256, batch 9 | 401,489 (376,432–435,493) | 349,871 (311,128–360,431) | 1.2083 (1.0759–1.2904); 1.2904, 1.2083, 1.0759 |
| FFN-down Q6_K 2,048×8,192, batch 64 | 53,239,841 (50,038,377–57,814,479) | 54,386,770 (51,697,429–58,608,393) | 1.0298 (0.8538–1.0630); 1.0630, 0.8538, 1.0298 |
| FFN-down Q6_K 2,048×8,192, batch 9 | 65,281,397 (64,678,478–65,578,929) | 64,606,222 (64,386,798–65,031,845) | 1.0045 (1.0038–1.0151); 1.0045, 1.0038, 1.0151 |
| Short-convolution output Q4_K 2,048×2,048, batch 64 | 13,126,020 (12,307,990–13,707,394) | 13,839,502 (12,628,924–13,909,304) | 0.9746 (0.9484–0.9855); 0.9746, 0.9855, 0.9484 |

The profile outer-call ratio is not isolated instrumentation overhead. The odd-row synthetic case has a noticeably higher profiled call time, consistent with instrumentation disturbance at this short duration. The two accepted FFN-down Q6_K cases show ratio medians near one, with only three repetitions each. A profiled call occasionally ran faster; that is not evidence of negative overhead. Do not compare medians across the three separate sessions as though they were one randomized run.

### Diagnostic stages and worker work

Stage wall medians/ranges below are from accepted profiled calls. All values in the following two tables are nanoseconds. `buffer_init` measures the vector allocation/initialization block together; allocation cost is not independently isolated. `rows wall` is parallel region elapsed wall, while each worker's decode/accumulate values are separately timed work. Worker sums below are sums across both workers for each repetition, then summarized by median/range. They are not elapsed wall shares or CPU-time measurements. Independently computed stage medians must not be added to obtain the median call time.

| Case | Buffer init | Input transpose | Rows wall | Output layout |
| --- | ---: | ---: | ---: | ---: |
| Synthetic Q8_0, batch 9 | 18,004 (1,333–22,622) | 3,076 (2,865–3,156) | 343,218 (320,585–356,052) | 3,376 (3,337–4,669) |
| FFN-down Q6_K, batch 9 | 56,497 (42,621–70,293) | 311,899 (301,460–345,423) | 64,814,515 (64,200,795–65,044,218) | 40,056 (39,614–41,508) |
| FFN-down Q6_K, batch 64 | 98,866 (88,957–118,964) | 3,234,917 (3,105,113–3,298,367) | 49,045,049 (45,638,928–53,271,841) | 913,268 (900,925–1,081,235) |
| Short-convolution output Q4_K, batch 64 | 141,075 (141,035–161,193) | 597,014 (555,075–770,550) | 11,179,156 (10,422,322–11,654,881) | 950,248 (928,587–1,063,492) |

| Case | Decode worker-work sum | Accumulate worker-work sum | Scratch-init worker-work sum | Zero/copy worker-work sum |
| --- | ---: | ---: | ---: | ---: |
| Synthetic Q8_0, batch 9 | 34,463 (34,133–44,908) | 543,144 (485,665–543,397) | 1,173 (1,000–1,283) | 30,253 (27,212–30,977) |
| FFN-down Q6_K, batch 9 | 13,415,665 (13,329,364–13,504,777) | 114,102,650 (112,752,643–114,428,147) | 14,896 (14,137–19,707) | 233,513 (233,368–310,994) |
| FFN-down Q6_K, batch 64 | 14,541,836 (13,885,268–15,099,761) | 80,939,469 (75,101,974–87,454,085) | 18,345 (17,392–33,924) | 934,516 (751,191–1,068,856) |
| Short-convolution output Q4_K, batch 64 | 1,953,006 (1,701,975–1,977,498) | 18,865,907 (17,776,265–19,818,423) | 13,025 (12,713–15,128) | 897,892 (750,959–955,726) |

These numbers locate work inside instrumented calls; worker-work sums can exceed rows-wall time because workers overlap. The diagnostic timers add overhead, allocation and initialization remain combined, and synthetic inputs do not represent captured model activations. No Phase 0 percentages are reused.

### Decision

**Faithful locality experiment.** In accepted instrumented calls, the Q6_K batch-9 worker-work sums were 114.103 ms for accumulation, 13.416 ms for decode, and 0.0149 ms for scratch initialization; its buffer allocation/initialization block was 0.0565 ms against 64.8145 ms median rows wall. For Q4_K batch 64, accumulation work summed to 18.866 ms, decode to 1.953 ms, and scratch initialization to 0.0130 ms. These worker sums overlap across two workers and are not elapsed wall shares or CPU-time estimates. They show that scratch initialization is a small measured category and identify accumulation as the largest measured worker-work category in these cases. They do not show that memory locality, rather than arithmetic throughput or another effect, limits accumulation. The next slice should therefore be a faithful, measurement-only locality experiment around that hot accumulation path, preserving arithmetic and output bits; treat it as a hypothesis test with no expected speedup or promotion claim. The narrow shape set, synthetic activations, three repetitions, profiler disturbance, and host load/swap/thermal uncertainty limit how broadly these observations generalize.
