//! Matrix-vector kernel benchmark runner command.

use anyhow::Result;
use std::path::PathBuf;

pub const BENCH_DIM: usize = 1024;
pub const BENCH_N: usize = 1024;
pub const BENCH_ITERS: usize = 500;

fn benchmark_kernel<F>(name: &str, iters: usize, n: usize, dim: usize, mut f: F)
where
    F: FnMut(),
{
    let start = std::time::Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed();
    let per_op_ms = elapsed.as_secs_f64() * 1000.0 / (iters as f64);
    let gflops = (2.0 * (n as f64) * (dim as f64) / 1e9) / (per_op_ms / 1000.0);

    println!(
        "  {} Matvec [{}x{}]: {:.3} ms/op ({:.2} GFLOPS)",
        name, n, dim, per_op_ms, gflops
    );
}

pub fn run_bench(model: Option<PathBuf>) -> Result<()> {
    println!("=== Mivi-v4 CPU Kernel Benchmark ===");
    println!(
        "Target model: {:?}",
        model.as_deref().unwrap_or_else(|| std::path::Path::new(""))
    );
    println!(
        "Benchmarking matvec kernels (dim={}, n={})...",
        BENCH_DIM, BENCH_N
    );

    let dim = BENCH_DIM;
    let n = BENCH_N;
    let x = vec![1.0f32; dim];
    let mut out = vec![0.0f32; n];

    // 1. Q8_0 Matvec Benchmark
    let q8_bytes_per_row = (dim / mivi_quant::Q8_0_BLOCK_SIZE) * mivi_quant::Q8_0_BYTES;
    let q8_weights = vec![1u8; n * q8_bytes_per_row];
    benchmark_kernel("Q8_0", BENCH_ITERS, n, dim, || {
        mivi_quant::matvec_q8_0(&mut out, &q8_weights, &x, n, dim);
    });

    // 2. Q4_K_M Matvec Benchmark
    let q4_bytes_per_row = (dim / mivi_quant::Q4_K_BLOCK_SIZE) * mivi_quant::Q4_K_BYTES;
    let q4_weights = vec![1u8; n * q4_bytes_per_row];
    benchmark_kernel("Q4_K_M", BENCH_ITERS, n, dim, || {
        mivi_quant::matvec_q4_k_m(&mut out, &q4_weights, &x, n, dim);
    });

    // 3. Q6_K Matvec Benchmark
    let q6_bytes_per_row = (dim / mivi_quant::Q6_K_BLOCK_SIZE) * mivi_quant::Q6_K_BYTES;
    let q6_weights = vec![1u8; n * q6_bytes_per_row];
    benchmark_kernel("Q6_K", BENCH_ITERS, n, dim, || {
        mivi_quant::matvec_q6_k(&mut out, &q6_weights, &x, n, dim);
    });

    // 4. End-to-end model generation benchmark if model path is valid
    if let Some(ref model_path) = model {
        if model_path.exists() {
            println!("\n=== End-to-End Generation Benchmark ===");
            let mut model = mivi_model::Model::load(model_path)?;
            let prompt = "Explain briefly why the sky is blue.";
            let warmup_tokens = 8;
            let bench_tokens = 32;

            // Warmup
            let _ = model.generate(prompt, warmup_tokens)?;

            // Timed run
            let start = std::time::Instant::now();
            let mut gen_count = 0usize;
            let output = model.generate_streaming(prompt, bench_tokens, |_, _| {
                gen_count += 1;
                true
            })?;
            let elapsed = start.elapsed();
            let tok_per_sec = (gen_count as f64) / elapsed.as_secs_f64();
            println!("  Generated tokens : {}", gen_count);
            println!("  Elapsed time     : {:.2} s", elapsed.as_secs_f64());
            println!(
                "  Throughput       : {:.2} tokens/sec ({:.2} ms/token)",
                tok_per_sec,
                1000.0 / tok_per_sec
            );
            println!("  Output sample    : {:?}", output);
        }
    }

    println!("\nBenchmark complete.");
    Ok(())
}
