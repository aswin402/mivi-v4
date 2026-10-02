# Private runtime replay diagnostics

This feature-required example is a diagnostic tool, not a normal server option
or a provider API. It replays bounded numeric prompt IDs using an explicitly
supplied local GGUF. It does not execute agent tools or establish coding quality.

Build only this example, one Cargo job:

```sh
CARGO_BUILD_JOBS=1 cargo build -p mivi-model --example runtime_replay --features fixture-diagnostics --release
```

Invoke the resulting `target/release/examples/runtime_replay` with exactly
`--model PATH --input PATH --output PATH`. Paths are explicit; no model download,
filename-based architecture inference, API key, or hidden startup benchmark.

The input JSON requires every field:

```json
{
  "prompt_ids": [1, 2],
  "context": 128,
  "tile": 64,
  "max_tokens": 16,
  "profile": false,
  "split_prefill": false,
  "teacher_forced_ids": [],
  "logit_ids": []
}
```

Those IDs only illustrate the schema. Use valid IDs from the loaded tokenizer;
the runner also checks metadata-driven BOS normalization and vocabulary bounds.
Do not use arbitrary illustrative IDs as a matched-reference benchmark.

Use a caller-created private directory (Unix mode0700, owned by the running
user) for output. Results must not exist already. The tool creates exclusive
mode0600 files using component-by-component pinned directory descriptors, rejects symlink output
ancestors, and refuses parent traversal. Non-Unix output fails closed because
this implementation does not enforce private ACLs there. Input must be a regular
non-symlink file, at most4MiB. The writer is single-use, including failed writes.
Serialization is bounded to4MiB per result; stdout
and stderr do not print prompts, token IDs, logits, or decoded output.

The replay uses a scoped two-thread Rayon pool and F32 KV. Diagnostic caps:
context4096, tile128, output64, teacher positions16, selected logit IDs64, retained
text65536 bytes per channel. Teacher forcing requires zero sampled output.
These are diagnostic safety bounds, not model-specific serving constants.

Normal mode performs one generation call. Split mode prefills with zero output,
then generates from that position; compare them before trusting split timing.
Profile totals and substages overlap and must not be summed as independent work.
Model callback delivery is not network-visible TTFT. Process exit, RSS limits,
and hard startup/termination deadlines require an external supervisor; a result
written inside the process cannot prove that process exit occurred. The runner's
180s cooperative deadline is not a hard watchdog for loading/blocking operations.
The CLI entry clock includes argument parsing, input/output preflight, and model
loading; load and model-prefill durations are also reported separately. Direct
fixture calls supply their own entry clock and have no CLI preflight.

Keep all results private and outside Git. Preserve failures and unavailable
observations rather than converting them to successful latency samples. Hashing
the model and verifying reference settings belong to the paired comparison
driver; file size/config metadata alone do not establish weight identity.

Ideas and sources: reuse Mivi's existing fixture recorder and forward profiles;
follow [Kimi's independent fixture discipline](https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/main/tests/fixtures/README.md)
and [Colibri's benchmark methodology](https://github.com/JustVugg/colibri/blob/main/docs/benchmarking.md).
The [Phase0 plan](superpowers/plans/2026-10-02-runtime-parity-and-profiling.md)
records the remaining numerical and repeated-measurement gates.

Run the ignored short/long same-engine parity fixture with an absolute model
path: Cargo runs example tests from the package directory, not the workspace
root. Keep an external timeout and run only this target:

```sh
CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 MIVI_TEST_MODEL=/absolute/path/model.gguf timeout --signal=TERM --kill-after=5s 600s cargo test -p mivi-model --example runtime_replay --release --features fixture-diagnostics replay_split_prefill_matches_single_call --offline -- --ignored --test-threads=1
```

The fixture uses 110- and 2636-token synthetic prefixes sequentially. It rejects
insufficient model context and incomplete/truncated runs rather than silently
skipping the long case. Same-engine parity is not independent numerical
correctness or agent-quality evidence.
