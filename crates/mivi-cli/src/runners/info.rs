//! GGUF inspection runner command.

use anyhow::Result;
use std::path::PathBuf;

pub fn run_info(model: PathBuf) -> Result<()> {
    if !model.exists() {
        anyhow::bail!(
            "Model file not found at {:?}.\n\nTo inspect the test model fixture, run:\n  just info models/mivi-tiny-test.gguf\nOr generate the test fixture using:\n  python3 training/export/generate_fixture.py",
            model
        );
    }
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
                mivi_model::GgufValue::Array(a) => {
                    println!("  {}: [array of {} items]", key, a.len())
                }
                _ => println!("  {}: {:?}", key, val),
            }
        }
    }

    println!("\n--- Sample Quantized Tensors ---");
    let mut tensor_names: Vec<_> = gguf.tensors.keys().collect();
    tensor_names.sort();
    for name in tensor_names.iter().take(12) {
        let info = &gguf.tensors[*name];
        println!(
            "  {:30} {:?} dims: {:?}",
            info.name, info.ggml_type, info.dims
        );
    }
    if gguf.tensors.len() > 12 {
        println!("  ... and {} more tensors", gguf.tensors.len() - 12);
    }

    Ok(())
}
