# Private router profiling controls

These feature-gated fixtures distinguish model timing, useful router output,
request-handler return and owned actor shutdown. They do not add normal-server
capture settings or change the public API. Heartbeats, HTTP headers and role-only
stream envelopes are not useful model output. Missing observations stay absent.

Run only the affected model-independent tests, one Cargo job and one test thread:

```sh
CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 cargo test -p mivi-server --lib --features fixture-diagnostics fixture_profile --offline -- --test-threads=1
```

The ignored profiling-control fixture requires an explicit absolute supported
local GGUF path and a release test executable. Run profiled and unprofiled
controls sequentially, never alongside another build or model load:

```sh
CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 MIVI_TEST_MODEL=/absolute/private/model.gguf cargo test -p mivi-server --lib --release --features fixture-diagnostics fixture_profile_control_parity --offline -- --ignored --test-threads=1
```

When capture integration changes, the existing observer off/on control is also
relevant; do not run every model-required test:

```sh
CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 MIVI_TEST_MODEL=/absolute/private/model.gguf cargo test -p mivi-server --lib --release --features fixture-diagnostics fixture_observer_parity --offline -- --ignored --test-threads=1
```

Keep raw requests, IDs, text and stream captures in private exclusive files
outside Git. Use external process/RSS supervision for live tests; cooperative
model cancellation is not a hard termination guarantee. Record failure and
cleanup outcomes, not successful zero-duration replacements. Do not use an
HTTP200, a heartbeat, `[DONE]` or a sender closing as proof of physical worker
return. Verify the owned actor's teardown independently.

Profiling exports the existing prefill snapshot with checked integer microseconds.
Attention/SSM totals overlap their substages; do not add them together as separate
work. Profiling parity is a correctness control, not a speedup, real-agent-quality
result or reference-engine numerical oracle.

Ideas and sources: [Colibri's controlled benchmark methodology](https://github.com/JustVugg/colibri/blob/main/docs/benchmarking.md)
and [Kimi's independent fixture discipline](https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/main/tests/fixtures/README.md).
Implementation extends Mivi's existing collectors and actor lifecycle rather
than copying inference kernels or creating a duplicate timer hierarchy.
