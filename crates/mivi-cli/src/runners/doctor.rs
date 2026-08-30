//! Diagnostic doctor runner command.

use anyhow::Result;

pub fn run_doctor() -> Result<()> {
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
    Ok(())
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}
