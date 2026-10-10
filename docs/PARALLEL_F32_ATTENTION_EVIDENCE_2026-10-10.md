# Parallel F32 attention: correctness and bounded evaluation

Date:2026-10-10. Experimental source based on v0.2.82, design commit7c800ee.
Status: all nine bounded model-pair cases and final scoped checks passed;
candidate remains default-off. Publication version0.2.83.

## Implementation and origin

The default-off `parallel-attention-experiment` feature adds an explicit RunState
selector and successful-dispatch counter. Compiling the feature alone retains
serial dispatch. A model diagnostic setter selects the candidate deliberately.
Single-worker/single-head execution and all non-F32 KV precision modes retain the
established serial path, including TurboQuant's shared `hb` scratch behavior.

Safe Rayon mutable chunks partition contiguous query heads into no more tasks than
the active pool's workers. Each head reads immutable Q/K/V and retains ascending
causal positions, the original GQA mapping, dot-product SIMD, online-softmax
branches and arithmetic, and output normalization. KV insertion and query-row
ordering remain sequential; all head work joins before downstream projection.
No new unsafe code, custom Send/Sync implementation, filename rule, precision
reduction, new dependency download or server/CLI default change is introduced.

The F32 candidate deliberately retains independent arithmetic from the existing
serial reference. It validates checked dimension products, divisibility, query
and output lengths, loaded KV width, layer mapping and stored/capacity positions
before writes, then uses checked KV getters throughout its scan. No pool is
created per token or layer. This candidate remains a diagnostic, not a generic
new-model capability claim.

## Correctness evidence

Observed RED: the release bit-parity test failed with
`ExecutionFailed("parallel attention not implemented")` on the candidate stub.
After implementation, four focused release tests passed:

- Complete output-bit comparisons in192 MHA/GQA cases, covering1/3/5/8 query
  heads, one or equally many KV heads, head dimensions4/7/16/64, positions0/1/16,
  and explicit one-/two-thread pools. Repeated calls retain immutable Q/KV and
  untouched shared FFN scratch. Every compared output is finite.
- Invalid dimensions/divisibility/overflow, short buffers, invalid/unstored
  positions, selective-layer rejection and actual KV-width mismatch fail without
  changing output bits. Valid calls retain extra output sentinels.
- F32, Q8_0, TurboQuant4 and TurboQuant2 dispatch controls preserve baseline bits.
  The counter records parallel work only when F32 and two workers are selected.
  Reset preserves the opt-in selector while clearing its counter.

The explicit tiny hybrid-fixture gate passed on the existing local94KiB GGUF.
It compares generated text/IDs, all final and three continuation-position logits,
complete KV exports and convolution/SSM bits. It covers token-major/chunked64
prefill,273 input IDs crossing multiple tiles/cache boundaries, empty-prefix and
restored-prefix cases, and reset. The fixture has2 heads,1 KV head and one block
of each kind. It is not proof of an independent large-model oracle or cross-ISA
equivalence. The two existing transformer tests also passed with the feature.

Commands used Cargo jobs1, Rayon threads2 and serial harnesses. Only mivi-model
lib targets/filters were compiled/tested, offline; no workspace-wide command.

Final v0.2.83 checks passed: four active candidate tests; the explicit tiny
hybrid-state gate; two existing transformer tests with the feature and two
without it. Scoped rustfmt checks and `git diff --check` also passed. The three
ignored diagnostics were selected explicitly when run; ordinary filtered test
success is not claimed as their execution. No final CLI binary rebuild was needed.

```sh
CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 cargo test -p mivi-model --lib \
  --release --offline --jobs 1 \
  --features parallel-attention-experiment,fixture-diagnostics \
  parallel_attention -- --test-threads=1

CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 \
MIVI_TEST_MODEL=/absolute/path/to/local-tiny-hybrid.gguf timeout 180s \
cargo test -p mivi-model --lib --release --offline --jobs 1 \
  --features parallel-attention-experiment,fixture-diagnostics \
  parallel_attention_model_state_parity -- --ignored --test-threads=1 --nocapture
```

## Operator pilot

One warmup pair and six measured alternating pairs per sequence length. Each
member executes16 identical complete32-head,8-KV-head,F32 attention queries with
head dimension64 and deterministic generated nonuniform inputs. Preparation,
allocation, logging and comparison are outside timers. Output-bit comparison is
after each16-call member, not after every repeated call. One-/two-thread correctness
is established separately; timed members use the same two-thread pool.
The diagnostic passed in1.21s. No model projections, real model activation capture,
HTTP, agent or end-to-end latency is measured by this pilot.

| Cached positions | Median candidate/serial pair ratio | Range |
|---|---:|---:|
| 32 | 0.881508 | 0.707899–1.260718 |
| 512 | 0.617140 | 0.479789–0.694916 |
| 2048 | 0.521264 | 0.484492–0.552944 |

Ratios are medians of six within-pair ratios; warmup excluded. Short-query
regression remains explicit. These observations cannot predict overall model
speedup: attention is only one measured part of prefill and scheduling adds costs.

All raw operator timings below are complete16-call totals in nanoseconds.
Pair0 is warmup; pairs1–6 are measured. Even pairs run serial first; odd pairs
run candidate first. No pair or warmup is discarded from this record.

| Positions | Pair | Serial ns | Candidate ns |
|---|---:|---:|---:|
| 32 | 0 | 765543 | 867283 |
| 32 | 1 | 733813 | 925131 |
| 32 | 2 | 766073 | 661117 |
| 32 | 3 | 738031 | 664243 |
| 32 | 4 | 832507 | 762757 |
| 32 | 5 | 869437 | 700280 |
| 32 | 6 | 951901 | 673850 |
| 512 | 0 | 16538684 | 10513260 |
| 512 | 1 | 15766880 | 9729413 |
| 512 | 2 | 17286924 | 10089778 |
| 512 | 3 | 17798401 | 11530404 |
| 512 | 4 | 18500995 | 11418814 |
| 512 | 5 | 19849008 | 9523338 |
| 512 | 6 | 17501064 | 12161765 |
| 2048 | 0 | 95834167 | 47008012 |
| 2048 | 1 | 81855962 | 45261745 |
| 2048 | 2 | 78875026 | 40655747 |
| 2048 | 3 | 87528298 | 45563911 |
| 2048 | 4 | 117897309 | 61538210 |
| 2048 | 5 | 87929219 | 47688136 |
| 2048 | 6 | 94809502 | 45934485 |

## Live measurement contract

The private sequential supervisor in `/tmp/mivi-parallel-attention.nXYatI`
enforces180s per child and1200s for the session. Its completed9 cases comprise three
alternating unprofiled cold pairs at512/2048 requested IDs, one separately profiled
cold pair per size, and one unprofiled exact-prefix pair at2048. Each child has an
additional cooperative160s generation budget including load/setup and both routes.
Timeout/error/incomplete output stops subsequent cases; raw failed logs remain.
All nine children exited0 with two accepted observations each; the session took
743.28s. No child/session budget was extended or retried. `results.json`, the
individual child logs and `run_pairs.py` remain in that private directory.

The same local LFM2.5-1.2B Instruct Q4_K_M GGUF is used for both routes. Its
SHA256 was reconfirmed after measurement as
`b1b3de114215d9507409a662a501a631095a479a419584e8a2ded6304b19b4f5`.
Tokenize once per child, same numeric IDs and metadata BOS policy for each member;
context4096, F32 KV, tile64, greedy sampling, repetition penalty1, seed7 and one
emitted-token cap. Nonempty emission and complete prefill are required for success.
Model loading and full result/state comparisons are outside inference timers.
Baseline and candidate must match generated output/ID and complete logits/KV/SSM
state at prefill and two teacher-forced continuation positions before accepting
the pair. Actual candidate-dispatch counts and computed/reused tokens are logged.

Profile controls are excluded from the unprofiled summaries. Causal-stage totals
include normalization/RoPE/KV insertion as before; nested times must not be summed
as independent costs. First-text measurement is the model callback, not HTTP or
client TTFT. Cold means cleared engine prefix state, not flushed OS pages.

## Completed cold-pair observations

All six unprofiled cold pairs passed complete output/logit/KV/SSM comparisons,
including two teacher-forced continuation positions per member. Each512-ID case
computed513 tokens and each2048-ID case computed2049 tokens after metadata BOS.
No cold member reused a prefix. Candidate dispatch counts were3078/12294 versus
zero in baseline; those are observed counts, not model-name dispatch conditions.

| Effective tokens | Serial median first-text s | Candidate median first-text s | Median paired candidate/serial | Pair ratio range |
|---|---:|---:|---:|---:|
| 513 | 15.327213 | 14.221024 | 0.966040 | 0.927828–0.985265 |
| 2049 | 64.513445 | 57.851393 | 0.890021 | 0.785314–0.935717 |

The3.40%/11.00% reductions come from medians of the three within-pair ratios,
not ratios of independent median times. All six pairs favored the candidate,
but the spread and three observations do not establish statistical confidence.
The roughly20% roadmap target is not achieved by the median long-case result.

| Effective tokens | Pair | First route | Serial first-text s | Candidate first-text s |
|---|---:|---|---:|---:|
| 513 | 0 | serial | 16.181671257 | 15.943229152 |
| 513 | 1 | candidate | 13.550404336 | 13.090238370 |
| 513 | 2 | serial | 15.327213472 | 14.221023560 |
| 2049 | 0 | serial | 75.585470618 | 59.358357123 |
| 2049 | 1 | candidate | 61.825733480 | 57.851392907 |
| 2049 | 2 | serial | 64.513445168 | 57.418344460 |

Separate profile controls also passed full output/state parity:

| Effective tokens | Serial causal region s | Candidate causal region s | Serial prefill s | Candidate prefill s |
|---|---:|---:|---:|---:|
| 513 | 0.794217 | 0.540909 | 13.560920 | 12.746773 |
| 2049 | 18.105278 | 8.647622 | 70.306680 | 60.544772 |

The long causal region was52.24% lower in this single profile control. It includes
unchanged RoPE/KV/norm work as well as head scans. Those observations support the
scheduling hypothesis, not a proven thermal/memory/hardware root cause or stable
speedup. Profile times are excluded from unprofiled medians; no assumed
instrumentation overhead is subtracted.

The separate exact-prefix pair also passed complete output/state comparisons.
Both routes restored2048 tokens and computed the final one of2049; first-text
times were0.109073168s serial and0.106727710s candidate. The candidate recorded
six actual attention calls, baseline zero. Each route prepared its own prefix
outside the observation timer; the child/session budgets still include that
preparation. One warm pair is a reuse smoke control, not a stable latency result.

For direct reproduction of one live cold pair:

```sh
CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 \
MIVI_TEST_MODEL=/absolute/path/to/local.gguf \
MIVI_ATTN_TOKENS=2048 MIVI_ATTN_PAIR=0 MIVI_ATTN_PROFILE=0 MIVI_ATTN_WARM=0 \
timeout --signal=TERM --kill-after=5s 180s \
cargo test -p mivi-model --lib --release --offline --jobs 1 \
  --features parallel-attention-experiment,fixture-diagnostics \
  parallel_attention_live_pair -- --ignored --test-threads=1 --nocapture
```

Use pair1/2 for the remaining alternating pairs, tokens512 for the short case,
profile1 for a separately instrumented pair, and warm1 for the prefix control.
Keep the outer session limit1200s and never run children concurrently.

Model timing used the v0.2.82 experimental test executable before publication.
After measurement, failure assertions were changed to avoid dumping complete
model-state vectors, single-head/short-query-buffer safety assertions were
expanded, and the workspace version was bumped. Those changes are outside
measured arithmetic/timers; no claim that the final executable is byte-identical
to the measured executable is made.

## Decision

Retain the explicit default-off candidate. It is bit-faithful on the tested host
and shows a useful exploratory head-scheduling gain, but the11.00% long-case
paired median first-text reduction is below the roughly20% roadmap target.
The15K-token Minicode timeout remains unresolved. Normal server/chat defaults
do not select this path, and no real agent success is inferred.

Next, evaluate the existing shape-gated projection/FFN candidates at the
model boundary with complete parity and repeated fixed-work timings. Preserve
negative small/tail cases and do not enable four-row dispatch globally. Stronger
long-context attention blocking would be a distinct numerical-design experiment.

## Review and limits

GPT-6 Luna/high read-only source review found no concrete correctness defect. It
identified external deadline/schedule evidence as unverified from the Rust source;
the supervisor and final raw results must supply that evidence before completion.
No subagent ran Cargo or inference alongside the sequential measurements.

AMD Ryzen7 7730U, AVX2/FMA host, rustc1.99.0. CPU affinity, thermal behavior,
frequency, OS cache state and unrelated user services are not controlled. No RSS,
cross-ISA execution, independent-engine numerical parity or real Minicode result
is claimed. Production promotion requires a separate review and stronger
end-to-end evidence even if this pilot is favorable.

## Ideas, inspirations and sources

- Existing Mivi [GQA implementation](../crates/mivi-model/src/transformer.rs)
  supplies head independence, arithmetic and fallback behavior; no external
  inference implementation was copied.
- [Checked KV access](../crates/mivi-kv/src/cache.rs) supplies safe shared reads;
  [RunState](../crates/mivi-core/src/arena.rs) supplies explicit diagnostic selector
  ownership and reset conventions.
- [CPU runtime evidence](CPU_RUNTIME_EVIDENCE_2026-10-05.md) and
  [captured projection evidence](FOUR_ROW_ISOLATION_CAPTURE_EVIDENCE_2026-10-10.md)
  motivated keeping operator, parity and actual model timing gates separate.
- The recent local prefill profile is retained in
  `/tmp/mivi-prefill-profile.49QxK4/REPORT.md`.
- [Approved design](superpowers/specs/2026-10-10-parallel-f32-attention-design.md)
  and [implementation plan](superpowers/plans/2026-10-10-parallel-f32-attention.md).

No new online research or superiority over llama.cpp/Ollama is claimed.
