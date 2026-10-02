# Private paired runtime comparison

This Python standard-library diagnostic driver supervises the private Mivi
replay example and an optional explicitly supplied local llama.cpp server.
It does not change Mivi's normal server, provider configuration or inference
backend. No model download, API key, shell invocation or automatic upload.

Build only the replay example if needed, with one Cargo job, as documented in
[runtime replay](RUNTIME_REPLAY_DIAGNOSTICS.md). Run scoped synthetic tests:

```sh
python3 -m unittest discover -s scripts/runtime_compare -p 'test_*.py'
```

Supply a private JSON manifest and a new result-directory path under a private
caller-owned directory. All paths should be absolute. Validation checks model
identity, executable paths, token metadata and reviewed diagnostic budgets
before launching children. Validation-only must not launch processes, load
model weights into the runtime or create the output directory:

```sh
python3 scripts/runtime_compare/compare.py --manifest /absolute/private/manifest.json --output-dir /absolute/private/new-results --validate-only
python3 scripts/runtime_compare/compare.py --manifest /absolute/private/manifest.json --output-dir /absolute/private/new-results
```

Manifest schema1 requires `schema`, `model_path`, `model_sha256`, `mivi_binary`,
`reference_binary`, `reference_revision`, `context`, `tile`, `max_tokens`,
`repetitions`, `wall_seconds`, `session_seconds`, `rss_bytes`, `artifact_bytes`
and `cases`. Unknown fields and duplicate case names are invalid. Each case has
`name` and `prompt_ids`. Use tokenizer-derived valid IDs; no prompt-text mutation.
Use the actual model SHA256, not a filename or file-size guess. Model hash
verification reads weights in bounded chunks; it is not an inference run.
Preflight inspects GGUF v2/v3 metadata without reading tensor weights into the
runtime. Its parser caps metadata span at64MiB, individual strings at1MiB and
item counts at1Mi; graph/tensor compatibility remains the model loader's job.
An absent reference is represented by null `reference_binary` and
`reference_revision`, and must remain unavailable rather than falsely matched.
The CLI exits0 only for a completed paired session; a native-only session
writes its report with status `partial` and exits2. Validation-only exits0.
The sample matrix must leave reserved artifact space for the final reports;
an insufficient `artifact_bytes` budget is rejected before children start.

The baseline measurement configuration is context4096, tile64, output48,
three repetitions over short/medium/approximately2636-ID workloads, two CPU
threads and one active engine. Lower budgets/smaller cases are useful smoke
checks, not a substitute for repeated evidence. The reviewed upper bounds are
context4096, tile128, output64, wall180s per run, session2700s, sampled child RSS
2GiB and artifacts64MiB. Inputs/results each remain bounded to4MiB and retained
text channels to64KiB. Larger budgets require a separately reviewed change.
Manifest wall/session budgets must be at least1s so cleanup has usable time.
RSS and artifact growth are sampled at100ms, not kernel-enforced quotas;
transient spikes can exceed a threshold between polls. The final artifact check
also runs after child exit and cleanup, so fast exits cannot become completed
samples merely by outrunning the watchdog. OS scheduling, blocking filesystem
calls and cleanup can delay physical return; recorded elapsed/cleanup outcomes
are authoritative, not a claim of a real-time deadline guarantee.

Keep manifests, results, prompts/IDs, decoded output, scores and logs private
and outside Git. The immediate output parent must be caller-owned and mode0700.
Unix destinations are owned directories0700 and exclusive files0600; reused
destinations, symlink ancestors, parent traversal and FIFO inputs fail closed.
Reads and writes use pinned directory descriptors. Retain failed samples, not
successful zero-latency replacements. Do not run this concurrently with Cargo
or other inference, and do not flush system-wide page caches.
If generated artifacts leave no report space after verified child cleanup,
the runner truncates only owned mode0600 single-link regular artifacts in the
new private output tree, preserves prefixes where space permits, and reports
discarded-byte/file counts with `artifact_limit` status. These discarded suffixes
cannot be recovered from the result. Models and manifests outside that tree are
never trimmed. Unsafe links/permissions or unverified cleanup refuse retention.

Paired results must disclose engine order, processed/normalized prompt counts,
observed exit and cleanup, failures/unavailable observations, and timing
boundaries. Model-callback first output is not network-visible TTFT; no direct
ratio between them is valid. Three samples support medians/ranges, not p95/p99.
Cold engine state does not imply cold OS weight pages.

Reference settings are checked against the supplied binary's help/version.
Unsupported required flags or a revision mismatch abort rather than silently
compare different configurations. The reference is CPU-only, loopback-only,
F32 K/V, parallel1, flash attention/fitting/warmup off, with default repacking.
No-key localhost access does not authorize any nonlocal security change.
The first completed native sample supplies its full effective stop-string list,
including runtime defaults plus metadata-derived EOS. Subsequent reference-first
repetitions reuse that verified policy. If no native completion has confirmed it,
reference execution is marked `not_run_unverified_stop_policy`; no model-family
stop literals are guessed or duplicated in this Python adapter.
The reference's `/props` confirms context, one slot and model path. Its native
completion response confirms serialized sampler settings and requires cold
`timings.cache_n == 0` with `prompt_n` equal to the normalized prompt length;
`tokens_cached` is current cache size, not prior reuse. Terminal EOS remains in
raw reference IDs but is separated from content IDs before token comparison.
Reference logprobs use the pinned `completion_probabilities` serializer and
pre-sampling probability mode, not raw logits. Reference TTFT stays unavailable
because this adapter uses nonstreaming native completions.

For divergent output, an identical-prefix next-token probe is supplemental
evidence: raw logits, probabilities and model state are different observations.
Missing equivalent scores/state remain unavailable, never equal. Terminal-policy
differences must stay explicit. Same-engine fixtures and synthetic HTTP tests
do not establish reference numerical equivalence or coding-agent usefulness.
Actual repeated measurements and independent oracle gates remain later tasks.

Ideas and sources: [Kimi's independent fixture discipline](https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/main/tests/fixtures/README.md),
[Colibri's controlled measurements](https://github.com/JustVugg/colibri/blob/main/docs/benchmarking.md),
the [pinned llama.cpp server contract](https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/tools/server/README.md),
its [response serializer](https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/tools/server/server-task.cpp)
and [cache/terminal implementation](https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/tools/server/server-context.cpp),
and the [GGUF format specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).
Implementation is native diagnostic orchestration, not a copied inference kernel.
