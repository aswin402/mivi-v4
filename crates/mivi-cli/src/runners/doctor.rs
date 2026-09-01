//! Diagnostic doctor runner command.

use anyhow::Result;
use std::path::Path;

pub fn run_doctor() -> Result<()> {
    println!("=== Mivi-v4 System Diagnostics ===");
    println!("Version:    v{}", env!("CARGO_PKG_VERSION"));
    println!("OS:         {}", std::env::consts::OS);
    println!("Arch:       {}", std::env::consts::ARCH);
    println!("CPUs:       {} physical/logical cores", num_cpus());

    #[cfg(target_arch = "x86_64")]
    {
        println!("AVX2:       {}", if is_x86_feature_detected!("avx2") { "✓ Detected" } else { "✗ Not detected" });
        println!("FMA:        {}", if is_x86_feature_detected!("fma") { "✓ Detected" } else { "✗ Not detected" });
        println!("AVX-512F:   {}", if is_x86_feature_detected!("avx512f") { "✓ Detected" } else { "✗ Not detected" });
    }

    #[cfg(target_arch = "aarch64")]
    {
        println!("NEON:       ✓ Detected (Native ARM64)");
    }

    // Check Rayon thread environment
    let threads_env = std::env::var("MIVI_THREADS")
        .or_else(|_| std::env::var("RAYON_NUM_THREADS"))
        .unwrap_or_else(|_| "auto".to_string());
    println!("Threads:    {}", threads_env);

    // Check .mivi workspace cache directory
    let cache_dir = Path::new(".mivi");
    let cache_status = if cache_dir.exists() {
        "✓ Found"
    } else {
        "• Not created yet (will initialize on demand)"
    };
    println!("Workspace:  {}", cache_status);

    // Check models directory
    let models_dir = Path::new("models");
    let models_status = if models_dir.is_dir() {
        let count = std::fs::read_dir(models_dir)
            .map(|rd| rd.flatten().filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("gguf")).count())
            .unwrap_or(0);
        format!("✓ Found ({} GGUF model files)", count)
    } else {
        "• Not found (place .gguf models in models/)".to_string()
    };
    println!("Models Dir: {}", models_status);

    println!("\nSystem Status: READY");
    Ok(())
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}
