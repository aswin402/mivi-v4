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
    let mut meta_keys: Vec<_> = gguf.metadata.keys().collect();
    meta_keys.sort();
    for key in meta_keys {
        let val = &gguf.metadata[key];
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

    println!("\n--- All Quantized Tensors ---");
    let mut tensor_names: Vec<_> = gguf.tensors.keys().collect();
    tensor_names.sort();
    for name in &tensor_names {
        let info = &gguf.tensors[*name];
        println!(
            "  {:35} {:?} dims: {:?}",
            info.name, info.ggml_type, info.dims
        );
    }

    Ok(())
}
