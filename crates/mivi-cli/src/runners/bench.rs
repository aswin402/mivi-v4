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

    // 4. End-to-end model generation & prefix cache benchmark if model path is valid
    if let Some(ref model_path) = model {
        if model_path.exists() {
            println!("\n=== End-to-End Generation & Prefix Cache Benchmark ===");
            let mut model = mivi_model::Model::load(model_path)?;
            model.sampler.config.temperature = 0.2;

            let system_prompt = "You are Mivi, an intelligent, concise, and helpful AI assistant designed for high-performance software engineering and systems programming in pure Rust. Always write clean code, follow standard formatting practices, verify all edge cases, and explain your technical reasoning clearly and concisely to the developer.";
            let prompt_cold = format!("<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\nWrite a one-sentence summary of Rust ownership.<|im_end|>\n<|im_start|>assistant\n", system_prompt);
            let prompt_warm = format!("<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n", system_prompt);

            let bench_tokens = 24;

            // Run 1: Cold Cache (Empty prefix cache)
            model.reset_context();
            let mut cold_ttft = std::time::Duration::ZERO;
            let mut cold_gen_count = 0usize;
            let cold_start = std::time::Instant::now();
            let mut first_token_seen = false;

            let output_cold = model.generate_streaming(&prompt_cold, bench_tokens, |_, _| {
                if !first_token_seen {
                    cold_ttft = cold_start.elapsed();
                    first_token_seen = true;
                }
                cold_gen_count += 1;
                true
            })?;
            let cold_total = cold_start.elapsed();
            let cached_chunks_after_cold = model.prefix_cache.len();

            println!("\n  🔹 Run 1: Cold Prefill (Cache Miss)");
            println!("  ─────────────────────────────────────────────────────────────────");
            println!("  Time To First Token (TTFT) : {:.2} ms", cold_ttft.as_secs_f64() * 1000.0);
            println!("  Total Generation Time      : {:.2} s ({} tokens)", cold_total.as_secs_f64(), cold_gen_count);
            println!("  Chunks in Prefix Cache     : {}", cached_chunks_after_cold);
            println!("  Output                     : {}", output_cold.trim());

            // Run 2: Warm Cache (Hits the pre-computed system prompt chunks!)
            model.reset_context();
            let mut warm_ttft = std::time::Duration::ZERO;
            let mut warm_gen_count = 0usize;
            let warm_start = std::time::Instant::now();
            first_token_seen = false;

            let output_warm = model.generate_streaming(&prompt_warm, bench_tokens, |_, _| {
                if !first_token_seen {
                    warm_ttft = warm_start.elapsed();
                    first_token_seen = true;
                }
                warm_gen_count += 1;
                true
            })?;
            let warm_total = warm_start.elapsed();

            let speedup = if warm_ttft.as_secs_f64() > 0.0 {
                cold_ttft.as_secs_f64() / warm_ttft.as_secs_f64()
            } else {
                1.0
            };

            println!("\n  ⚡ Run 2: Warm Prefill (LMCache Prefix Hit!)");
            println!("  ─────────────────────────────────────────────────────────────────");
            println!("  Time To First Token (TTFT) : {:.2} ms", warm_ttft.as_secs_f64() * 1000.0);
            println!("  Total Generation Time      : {:.2} s ({} tokens)", warm_total.as_secs_f64(), warm_gen_count);
            println!("  TTFT Speedup Factor        : {:.1}x FASTER", speedup);
            println!("  Output                     : {}", output_warm.trim());
        }
    }

    println!("\nBenchmark complete.");
    Ok(())
}
