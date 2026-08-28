use clap::Parser;
use mivi_cli::{Cli, Commands};
use mivi_model::Model;
use mivi_server::{create_router, AppState};
use mivi_tools::{extract_thinking, extract_tool_calls};
use std::io::{self, Write};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { port, host, model } => {
            let model_name = model
                .as_ref()
                .and_then(|p| p.file_name())
                .and_then(|n| n.to_str())
                .unwrap_or("mivi-v4-350m")
                .to_string();

            let broker = mivi_tools::ToolBroker::new();
            mivi_tools::register_builtin_tools(&broker, std::path::Path::new(".")).await;

            let loaded_model = if let Some(p) = model.as_ref() {
                if p.exists() {
                    match mivi_model::Model::load(p) {
                        Ok(m) => Some(Arc::new(Mutex::new(m))),
                        Err(e) => {
                            eprintln!("Warning: Failed to load model from {:?}: {}", p, e);
                            None
                        }
                    }
                } else {
                    None
                }
            } else {
                None
            };

            let state = Arc::new(Mutex::new(AppState {
                model_name: model_name.clone(),
                start_time: std::time::Instant::now(),
                broker,
                model: loaded_model,
            }));

            let app = create_router(state);
            let addr: SocketAddr = format!("{}:{}", host, port).parse()?;

            println!(
                r#"
  __  __ _____ _    _ _____          __   _  _   
 |  \/  |_   _| |  | |_   _|        / /  | || |  
 | \  / | | | | |  | | | |  ______ / /_  | || |_ 
 | |\/| | | | \ \  / / | | |______| '_ \ |__   _|
 | |  | |_| |_ \ \/ / _| |_        | (_) |  | |  
 |_|  |_|_____| \__/ |_____|        \___/   |_|  
                                                 
 Mivi-v4 Agent Engine
 Model: {}
 Listening on: http://{}
 OpenAI-compatible API ready.
"#,
                model_name, addr
            );

            let listener = tokio::net::TcpListener::bind(addr).await?;
            axum::serve(listener, app).await?;
        }
        Commands::Chat { model } => {
            println!("Loading Mivi model from {:?}...", model);
            let mut m = Model::load(&model)?;
            println!("Model loaded successfully! ({} parameters, 32K context)", m.config.name);
            println!("Mivi Interactive REPL. Type your prompt (or 'exit' to quit):\n");

            let mut conversation_history = Vec::new();

            loop {
                print!("user> ");
                io::stdout().flush()?;

                let mut input = String::new();
                io::stdin().read_line(&mut input)?;
                let trimmed = input.trim();

                if trimmed.is_empty() {
                    continue;
                }
                if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
                    println!("Goodbye!");
                    break;
                }

                conversation_history.push(mivi_tokenizer::ChatMessage {
                    role: mivi_tokenizer::Role::User,
                    content: Some(trimmed.to_string()),
                    name: None,
                });

                let prompt = mivi_tokenizer::format_chatml(&conversation_history, None, true);
                print!("assistant> ");
                io::stdout().flush()?;

                let output = m.generate(&prompt, 512)?;

                // Display thinking if present
                if let Some(think) = extract_thinking(&output) {
                    println!("\x1b[90m[think: {}]\x1b[0m", think);
                }

                // Display any tool calls
                let tool_calls = extract_tool_calls(&output);
                for tc in tool_calls {
                    println!("\x1b[33m[tool_call: {}({})]\x1b[0m", tc.name, tc.arguments);
                }

                println!("{}\n", output);

                conversation_history.push(mivi_tokenizer::ChatMessage {
                    role: mivi_tokenizer::Role::Assistant,
                    content: Some(output),
                    name: None,
                });
            }
        }
        Commands::Info { model } => {
            println!("=== Mivi GGUF Inspection ===");
            let gguf = mivi_model::GgufFile::open(&model)?;
            println!("File: {:?}", model);
            println!("GGUF Version: {}", gguf.version);
            println!("Metadata count: {}", gguf.metadata.len());
            println!("Tensor count: {}", gguf.tensors.len());

            println!("\n--- Key Model Hyperparameters ---");
            for (key, val) in &gguf.metadata {
                if key.starts_with("general.") || key.starts_with("lfm.") || key.starts_with("tokenizer.") {
                    match val {
                        mivi_model::GgufValue::String(s) => println!("  {}: \"{}\"", key, s),
                        mivi_model::GgufValue::U32(v) => println!("  {}: {}", key, v),
                        mivi_model::GgufValue::U64(v) => println!("  {}: {}", key, v),
                        mivi_model::GgufValue::F32(v) => println!("  {}: {}", key, v),
                        mivi_model::GgufValue::Array(a) => println!("  {}: [array of {} items]", key, a.len()),
                        _ => println!("  {}: {:?}", key, val),
                    }
                }
            }

            println!("\n--- Sample Quantized Tensors ---");
            let mut tensor_names: Vec<_> = gguf.tensors.keys().collect();
            tensor_names.sort();
            for name in tensor_names.iter().take(12) {
                let info = &gguf.tensors[*name];
                println!("  {:30} {:?} dims: {:?}", info.name, info.ggml_type, info.dims);
            }
            if gguf.tensors.len() > 12 {
                println!("  ... and {} more tensors", gguf.tensors.len() - 12);
            }
        }
        Commands::Doctor => {
            println!("=== Mivi-v4 System Diagnostics ===");
            println!("OS: {}", std::env::consts::OS);
            println!("Arch: {}", std::env::consts::ARCH);
            println!("CPUs: {}", num_cpus());
            #[cfg(target_arch = "x86_64")]
            {
                println!("AVX2 support: {}", is_x86_feature_detected!("avx2"));
                println!("FMA support:  {}", is_x86_feature_detected!("fma"));
            }
            println!("Status: OK");
        }
        Commands::Bench { model } => {
            println!("=== Mivi-v4 CPU Kernel Benchmark ===");
            println!("Target model: {:?}", model.unwrap_or_default());
            println!("Benchmarking matvec kernels (dim=1024, n=1024)...");

            let dim = 1024;
            let n = 1024;
            let x = vec![1.0f32; dim];
            let mut out = vec![0.0f32; n];

            // 1. Q8_0 Matvec Benchmark
            let q8_bytes_per_row = (dim / 32) * 34;
            let q8_weights = vec![1u8; n * q8_bytes_per_row];

            let start = std::time::Instant::now();
            let iters = 500;
            for _ in 0..iters {
                mivi_quant::matvec_q8_0(&mut out, &q8_weights, &x, n, dim);
            }
            let elapsed = start.elapsed();
            let per_op_ms = elapsed.as_secs_f64() * 1000.0 / (iters as f64);
            let gflops = (2.0 * (n as f64) * (dim as f64) / 1e9) / (per_op_ms / 1000.0);

            println!("  Q8_0 Matvec [{}x{}]: {:.3} ms/op ({:.2} GFLOPS)", n, dim, per_op_ms, gflops);

            // 2. Q4_K_M Matvec Benchmark
            let q4_bytes_per_row = (dim / 256) * 144;
            let q4_weights = vec![1u8; n * q4_bytes_per_row];

            let start = std::time::Instant::now();
            for _ in 0..iters {
                mivi_quant::matvec_q4_k_m(&mut out, &q4_weights, &x, n, dim);
            }
            let elapsed = start.elapsed();
            let per_op_ms = elapsed.as_secs_f64() * 1000.0 / (iters as f64);
            let gflops = (2.0 * (n as f64) * (dim as f64) / 1e9) / (per_op_ms / 1000.0);

            println!("  Q4_K_M Matvec [{}x{}]: {:.3} ms/op ({:.2} GFLOPS)", n, dim, per_op_ms, gflops);
            println!("\nBenchmark complete.");
        }
    }

    Ok(())
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}
