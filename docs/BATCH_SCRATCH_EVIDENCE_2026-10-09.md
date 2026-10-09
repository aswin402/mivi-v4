# Reusable batch scratch: correctness evidence (stage 1)

## Change and limits

Feature `batch-scratch-experiment` in mivi-quant adds a caller-owned workspace
and an explicit scratch-taking batch API. Baseline kernels and default model/server
dispatch are unchanged. There is no CLI selector or model-specific rule. No new
dependency, unsafe block, lower-precision activation representation or accumulation
reordering is introduced.

The user-approved four-row experiment is **stage 2, not implemented here**.
There is no speed benchmark, real-model run, RSS measurement, HTTP/tool-loop result
or latency improvement claim in this release. Previous projection profiling found
accumulation dominant; scratch ownership alone is not expected to solve agent TTFT.

## Ownership and validation

Construction checks element/aggregate-byte arithmetic before fallible reservations.
The caller bounds maximum batch/rows/columns and worker partitions; requested memory
still needs to fit the caller's budget. This is not a server memory-limit integration.
Workers own separate decoded-row/output buffers. Projection calls never reserve or
resize workspace buffers. The existing batch-one matvec is delegated unchanged;
its allocations and Rayon's internal allocations are not a zero-allocation claim.

All supported format and external-buffer validation occurs before workspace writes.
For batches above one, capacities and actual baseline worker-partition count are
checked before transpose/compute. Moving to a larger pool can return a worker
capacity error. Empty/batch-one work needs no workspace shape capacity but must
still satisfy external-buffer checks. The error reports capacity units explicitly.

## Verification

Observed RED: `scratch_matches_established_batch_bits` failed on the placeholder
constructor with `ArithmeticOverflow`, exit 101. Implemented arithmetic then passed
that release-mode parity test. Six focused release tests subsequently passed:

- Complete bit-pattern parity against the untouched ordinary batch API for Q4_K,
  Q6_K, Q8_0, F16, BF16 and F32; columns 256/512; rows 0/1/3/257; batches
  0/1/2/8/9/32/64/65. Explicit Rayon pools use one and two threads (768 cases).
- The same workspace is reused across all shapes/formats in each pool, initially
  poisoned with NaNs. Persistent buffer pointers, lengths and capacities remain
  identical; output tails retain their sentinels. Inputs are immutable references.
- Invalid external lengths, unsupported format, misalignment, shape capacity and
  arithmetic overflow leave both external outputs and every workspace bit untouched.
- A two-thread pool requiring two partitions rejects one-worker scratch before writes.
- Constructor element/byte overflow and zero-worker capacity are rejected.
- Empty work still validates external buffers; aligned/unaligned F32 batch-one
  delegation agrees with baseline. Zero-column batches clear active output only.

Command (Cargo one job, inference two threads):

```sh
CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 cargo test -p mivi-quant --lib --release --offline --features batch-scratch-experiment batch_scratch::tests -- --test-threads=1
```

Final v0.2.76 verification: release batch filter passed 12 tests with the experiment
enabled and 6 disabled; debug scratch filter passed 6 tests. Scoped rustfmt and
`git diff --check` passed. GPT-6 Luna/high read-only source review found no issues.
Commands are recorded in the [implementation plan](superpowers/plans/2026-10-09-batch-prefill-scratch.md).
No workspace-wide Cargo build/check/test is required or claimed.
The parity matrix uses deterministic nonzero finite test weights/inputs. It establishes
same-host baseline agreement, not hardware-independent cross-ISA equality or real-model
quality; portable fallback code is retained, not separately cross-compiled here.

## Example use

```rust,ignore
use mivi_quant::batch_scratch::{BatchProjectionScratch, quantized_matmul_rows_with_scratch};
// Shape/worker capacities come from the caller's workload, not a model name.
let mut scratch = BatchProjectionScratch::new(max_batch, max_rows, max_cols, workers)?;
quantized_matmul_rows_with_scratch(
    out, tensor_type, weights, inputs, batch, rows, cols, &mut scratch,
)?;
```

## Ideas, inspirations and sources

- Mivi's existing checked batch kernel (`crates/mivi-quant/src/lib.rs`) supplies
  format decoding, operation order, row partitioning and safe SIMD dispatch.
- [Projection-cost evidence](PROJECTION_COST_EVIDENCE_2026-10-05.md) motivated
  separating reusable buffers from accumulation optimization.
- [Agent-sized Q4 evidence](Q4_AGENT_SIZED_EVIDENCE_2026-10-07.md) showed decode
  improvements do not optimize the initial chunked-prefill wait.
- [Colibri benchmarking](https://github.com/JustVugg/colibri/blob/main/docs/benchmarking.md)
  inspired separating changes and retaining negative outcomes. No external inference
  code was copied; no new online research is claimed for this implementation.

Next: independently gate four-row blocking on bit-exact parity and paired operator
timings against **both** ordinary and scratch-only APIs before expensive model tests.
