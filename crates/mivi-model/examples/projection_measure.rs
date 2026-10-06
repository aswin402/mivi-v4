#[path = "projection_measure/case.rs"]
mod case;
#[path = "runtime_replay/io.rs"]
#[allow(dead_code)]
mod private_io;

use case::{CaseInput, Source};
#[cfg(feature = "projection-locality-experiment")]
use mivi_core::simd::ProjectionColumnTile;
use mivi_model::gguf::GgufFile;
#[cfg(feature = "projection-locality-experiment")]
use mivi_quant::projection_diagnostics::quantized_matmul_rows_profiled_with_column_tile;
use mivi_quant::projection_diagnostics::{quantized_matmul_rows_profiled, ProjectionProfile};
use mivi_quant::{quantized_matmul_rows, GgmlType};
use std::error::Error;
use std::ffi::OsString;
use std::path::PathBuf;
use std::time::Instant;

type AnyError = Box<dyn Error + Send + Sync>;

struct Args {
    input: PathBuf,
    output: PathBuf,
}

fn parse_args(args: impl IntoIterator<Item = OsString>) -> Result<Args, &'static str> {
    let mut args = args.into_iter();
    let (mut input, mut output) = (None, None);
    while let Some(flag) = args.next() {
        let slot = match flag.to_str() {
            Some("--input") => &mut input,
            Some("--output") => &mut output,
            _ => return Err("unknown argument"),
        };
        if slot.is_some() {
            return Err("duplicate argument");
        }
        let value = args.next().ok_or("missing argument value")?;
        let path = PathBuf::from(value);
        if !path.is_absolute() {
            return Err("input and output paths must be absolute");
        }
        *slot = Some(path);
    }
    Ok(Args {
        input: input.ok_or("missing input")?,
        output: output.ok_or("missing output")?,
    })
}

#[cfg(not(test))]
fn main() -> std::process::ExitCode {
    let args = match parse_args(std::env::args_os().skip(1)) {
        Ok(args) => args,
        Err(reason) => {
            eprintln!("projection measure: {reason}; usage: --input ABS_JSON --output ABS_JSON");
            return std::process::ExitCode::FAILURE;
        }
    };
    match run_cli(args) {
        Ok(()) => {
            println!("projection measurement complete; private result written");
            std::process::ExitCode::SUCCESS
        }
        Err(_) => {
            eprintln!(
                "projection measurement failed; check the private case input and result location"
            );
            std::process::ExitCode::FAILURE
        }
    }
}

#[allow(dead_code)]
fn run_cli(args: Args) -> Result<(), AnyError> {
    let input: CaseInput = private_io::read_json(&args.input)?;
    input.validate().map_err(std::io::Error::other)?;
    let mut output = private_io::PrivateOutput::create(&args.output)?;
    let value = measure_case(input)?;
    output.write_json(&value)?;
    Ok(())
}

enum Weights {
    Synthetic(Vec<u8>),
    Gguf(GgufFile),
}

struct LoadedCase {
    weights: Weights,
    format: GgmlType,
    rows: usize,
    cols: usize,
    mapping_bytes: u64,
    source_kind: &'static str,
    model_path: Option<PathBuf>,
    tensor: Option<String>,
}

impl LoadedCase {
    fn weight_bytes(&self) -> Result<&[u8], AnyError> {
        match &self.weights {
            Weights::Synthetic(bytes) => Ok(bytes),
            Weights::Gguf(gguf) => Ok(gguf
                .get_tensor_data(self.tensor.as_deref().ok_or("missing tensor identity")?)?
                .1),
        }
    }
}

fn measure_case(input: CaseInput) -> Result<serde_json::Value, AnyError> {
    let setup_started = Instant::now();
    input.validate().map_err(std::io::Error::other)?;
    let loaded = load_case(&input)?;
    let output_count = input
        .batch
        .checked_mul(loaded.rows)
        .ok_or("output size overflow")?;
    let artifact_bound = output_count
        .checked_mul(11)
        .and_then(|n| n.checked_add(256 * 1024))
        .ok_or("output artifact bound overflow")?;
    if artifact_bound > private_io::MAX_FILE_BYTES {
        return Err(
            "output artifact bound exceeds private writer limit; select a smaller batch".into(),
        );
    }
    let required = input.required_heap_bytes(loaded.format, loaded.rows, loaded.cols)?;
    let inputs: Vec<f32> = (0..input
        .batch
        .checked_mul(loaded.cols)
        .ok_or("input size overflow")?)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.125)
        .collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(input.threads)
        .build()?;
    let mut out = vec![0.0f32; output_count];
    let weights = loaded.weight_bytes()?;
    let setup_ns = elapsed_ns(setup_started);

    if input.warmup_calls == 1 {
        pool.install(|| {
            invoke(
                &mut out,
                loaded.format,
                weights,
                &inputs,
                &input,
                loaded.rows,
                loaded.cols,
            )
        })?;
        ensure_finite(&out)?;
    }

    let mut reference_bits: Option<Vec<u32>> = None;
    let mut call_wall_ns = Vec::with_capacity(input.measured_calls);
    let mut profiles = Vec::with_capacity(if input.profile {
        input.measured_calls
    } else {
        0
    });
    for _ in 0..input.measured_calls {
        let call_started = Instant::now();
        let profile = pool.install(|| {
            let input_view = std::hint::black_box(inputs.as_slice());
            let profile = invoke(
                &mut out,
                loaded.format,
                weights,
                input_view,
                &input,
                loaded.rows,
                loaded.cols,
            )?;
            std::hint::black_box(out.as_slice());
            Ok::<_, AnyError>(profile)
        })?;
        call_wall_ns.push(elapsed_ns(call_started));
        ensure_finite(&out)?;
        if let Some(reference) = &reference_bits {
            if reference.len() != out.len()
                || reference
                    .iter()
                    .zip(&out)
                    .any(|(expected, actual)| *expected != actual.to_bits())
            {
                return Err("measured calls produced different output bits".into());
            }
        } else {
            reference_bits = Some(out.iter().map(|value| value.to_bits()).collect());
        }
        if let Some(profile) = profile {
            profiles.push(profile_json(&profile));
        }
    }

    let output_bits = reference_bits.ok_or("missing measured output")?;
    let branch = branch_for(input.batch);
    let (model_path, tensor) = match (&loaded.model_path, &loaded.tensor) {
        (Some(path), Some(name)) => (Some(path.display().to_string()), Some(name.as_str())),
        _ => (None, None),
    };
    Ok(serde_json::json!({
        "schema": 3,
        "comparison_group": input.comparison_group,
        "column_tile": input.column_tile,
        "status": "complete",
        "source_kind": loaded.source_kind,
        "model_path": model_path,
        "tensor_name": tensor,
        "mapping_bytes": loaded.mapping_bytes,
        "format": format_name(loaded.format),
        "ggml_type": loaded.format as u32,
        "rows": loaded.rows,
        "cols": loaded.cols,
        "batch": input.batch,
        "branch": branch,
        "threads": input.threads,
        "profile": input.profile,
        "activation_source": "synthetic_f32",
        "setup_ns": setup_ns,
        "call_wall_ns": call_wall_ns,
        "output_bits": output_bits,
        "all_calls_bit_identical": true,
        "profile_calls": profiles,
        "estimated_heap_bytes": required,
        "output_artifact_bound_bytes": artifact_bound
    }))
}

fn load_case(input: &CaseInput) -> Result<LoadedCase, AnyError> {
    match &input.source {
        Source::Synthetic {
            ggml_type,
            rows,
            cols,
        } => {
            let format = case::supported_type(*ggml_type).map_err(std::io::Error::other)?;
            input.required_heap_bytes(format, *rows, *cols)?;
            let bytes = synthetic_weights(format, *rows, *cols)?;
            Ok(LoadedCase {
                weights: Weights::Synthetic(bytes),
                format,
                rows: *rows,
                cols: *cols,
                mapping_bytes: 0,
                source_kind: "synthetic",
                model_path: None,
                tensor: None,
            })
        }
        Source::Gguf { model_path, tensor } => {
            let (gguf, mapping_bytes) = open_gguf_model(model_path, input.model_limit_bytes)?;
            let (info, data) = gguf.get_tensor_data(tensor)?;
            if info.n_dims != 2 || info.dims.len() != 2 {
                return Err("selected tensor rank must be exactly two".into());
            }
            let format =
                case::supported_type(info.ggml_type as u32).map_err(std::io::Error::other)?;
            let cols = info.dims[0];
            let rows = info.dims[1];
            if rows == 0 || cols == 0 {
                return Err("selected tensor dimensions must be positive".into());
            }
            case::validate_alignment(format, cols).map_err(std::io::Error::other)?;
            let required = input.required_heap_bytes(format, rows, cols)?;
            let row_bytes = (cols / format.block_size().ok_or("unsupported format")?)
                .checked_mul(format.type_size().ok_or("unsupported format")?)
                .ok_or("tensor row size overflow")?;
            if rows
                .checked_mul(row_bytes)
                .ok_or("tensor byte span overflow")?
                != data.len()
            {
                return Err(
                    "selected tensor byte span does not match its two-dimensional shape".into(),
                );
            }
            let output_values = input
                .batch
                .checked_mul(rows)
                .ok_or("output size overflow")?;
            if output_values
                .checked_mul(11)
                .and_then(|n| n.checked_add(256 * 1024))
                .ok_or("output artifact bound overflow")?
                > private_io::MAX_FILE_BYTES
            {
                return Err(
                    "output artifact bound exceeds private writer limit; select a smaller batch"
                        .into(),
                );
            }
            let _ = required;
            Ok(LoadedCase {
                weights: Weights::Gguf(gguf),
                format,
                rows,
                cols,
                mapping_bytes,
                source_kind: "gguf",
                model_path: Some(model_path.clone()),
                tensor: Some(tensor.clone()),
            })
        }
    }
}

fn open_gguf_model(
    model_path: &std::path::Path,
    model_limit_bytes: u64,
) -> Result<(GgufFile, u64), AnyError> {
    #[cfg(target_os = "linux")]
    {
        let verified = private_io::open_bounded_regular_file(model_path, model_limit_bytes)?;
        map_gguf_from_verified_file_with(&verified, model_limit_bytes, GgufFile::open)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (model_path, model_limit_bytes);
        Err("GGUF measurement requires Linux descriptor-backed mapping".into())
    }
}

#[cfg(target_os = "linux")]
fn map_gguf_from_verified_file_with<F>(
    verified: &std::fs::File,
    model_limit_bytes: u64,
    map: F,
) -> Result<(GgufFile, u64), AnyError>
where
    F: FnOnce(&std::path::Path) -> mivi_model::gguf::Result<GgufFile>,
{
    use std::os::fd::AsRawFd;

    let metadata = verified.metadata()?;
    if !metadata.is_file() || metadata.len() > model_limit_bytes {
        return Err("GGUF descriptor is not a regular file within model_limit_bytes".into());
    }
    let descriptor_path = PathBuf::from(format!("/proc/self/fd/{}", verified.as_raw_fd()));
    let mapped = map(&descriptor_path)?;
    let mapping_bytes = u64::try_from(mapped.mmap.len()).map_err(|_| "mapping size overflow")?;
    if mapping_bytes > model_limit_bytes {
        return Err("mapped GGUF exceeds model_limit_bytes".into());
    }
    Ok((mapped, mapping_bytes))
}

fn invoke(
    out: &mut [f32],
    format: GgmlType,
    weights: &[u8],
    inputs: &[f32],
    input: &CaseInput,
    rows: usize,
    cols: usize,
) -> Result<Option<ProjectionProfile>, AnyError> {
    let inputs = std::hint::black_box(inputs);
    let out = std::hint::black_box(out);
    if input.profile {
        let profile = match input.column_tile {
            None => quantized_matmul_rows_profiled(
                out,
                format,
                weights,
                inputs,
                input.batch,
                rows,
                cols,
            )?,
            Some(tile) => {
                #[cfg(feature = "projection-locality-experiment")]
                {
                    let selector = match tile {
                        32 => ProjectionColumnTile::Columns32,
                        64 => ProjectionColumnTile::Columns64,
                        128 => ProjectionColumnTile::Columns128,
                        _ => return Err("unsupported column tile selector".into()),
                    };
                    quantized_matmul_rows_profiled_with_column_tile(
                        out,
                        format,
                        weights,
                        inputs,
                        input.batch,
                        rows,
                        cols,
                        selector,
                    )?
                }
                #[cfg(not(feature = "projection-locality-experiment"))]
                {
                    let _ = tile;
                    return Err("column tile selector requires the experiment feature".into());
                }
            }
        };
        std::hint::black_box(&*out);
        Ok(Some(profile))
    } else {
        match input.column_tile {
            None => quantized_matmul_rows(out, format, weights, inputs, input.batch, rows, cols)?,
            Some(tile) => {
                #[cfg(feature = "projection-locality-experiment")]
                {
                    let selector = match tile {
                        32 => ProjectionColumnTile::Columns32,
                        64 => ProjectionColumnTile::Columns64,
                        128 => ProjectionColumnTile::Columns128,
                        _ => return Err("unsupported column tile selector".into()),
                    };
                    mivi_quant::quantized_matmul_rows_with_column_tile(
                        out,
                        format,
                        weights,
                        inputs,
                        input.batch,
                        rows,
                        cols,
                        selector,
                    )?;
                }
                #[cfg(not(feature = "projection-locality-experiment"))]
                {
                    let _ = tile;
                    return Err("column tile selector requires the experiment feature".into());
                }
            }
        }
        std::hint::black_box(&*out);
        Ok(None)
    }
}

fn synthetic_weights(format: GgmlType, rows: usize, cols: usize) -> Result<Vec<u8>, AnyError> {
    case::validate_alignment(format, cols).map_err(std::io::Error::other)?;
    let row_bytes = (cols / format.block_size().ok_or("unsupported format")?)
        .checked_mul(format.type_size().ok_or("unsupported format")?)
        .ok_or("weight size overflow")?;
    let total = rows.checked_mul(row_bytes).ok_or("weight size overflow")?;
    let mut weights = vec![0u8; total];
    match format {
        GgmlType::F32 => {
            for (i, chunk) in weights.chunks_exact_mut(4).enumerate() {
                chunk.copy_from_slice(&(((i % 13) as f32 - 6.0) * 0.125).to_le_bytes());
            }
        }
        GgmlType::F16 | GgmlType::BF16 => {
            for (i, chunk) in weights.chunks_exact_mut(2).enumerate() {
                let base = if format == GgmlType::F16 {
                    0x3800u16
                } else {
                    0x3f00u16
                };
                chunk.copy_from_slice(&(base + ((i % 5) as u16) * 0x100).to_le_bytes());
            }
        }
        GgmlType::Q8_0 | GgmlType::Q4_K | GgmlType::Q6_K => {
            for (block_index, block) in weights
                .chunks_exact_mut(format.type_size().ok_or("unsupported format")?)
                .enumerate()
            {
                for (j, byte) in block.iter_mut().enumerate() {
                    *byte = ((block_index + 7 * j) % 127 + 1) as u8;
                }
                match format {
                    GgmlType::Q8_0 => block[..2].copy_from_slice(&0x3000u16.to_le_bytes()),
                    GgmlType::Q4_K => {
                        block[..2].copy_from_slice(&0x3000u16.to_le_bytes());
                        block[2..4].copy_from_slice(&0x2800u16.to_le_bytes());
                    }
                    GgmlType::Q6_K => {
                        let scale_start = block.len() - 2;
                        block[scale_start..].copy_from_slice(&0x3000u16.to_le_bytes());
                    }
                    _ => unreachable!(),
                }
            }
        }
        _ => return Err("unsupported format".into()),
    }
    Ok(weights)
}

fn ensure_finite(values: &[f32]) -> Result<(), AnyError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err("kernel produced nonfinite output".into())
    }
}

fn elapsed_ns(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn branch_for(batch: usize) -> &'static str {
    match batch {
        0 => "empty",
        1 => "matvec",
        2..=8 => "per_input_dot",
        9..=31 => "across_batch",
        _ => "across_batch_pair",
    }
}

fn format_name(format: GgmlType) -> &'static str {
    match format {
        GgmlType::F32 => "F32",
        GgmlType::F16 => "F16",
        GgmlType::BF16 => "BF16",
        GgmlType::Q8_0 => "Q8_0",
        GgmlType::Q4_K => "Q4_K",
        GgmlType::Q6_K => "Q6_K",
        _ => "unsupported",
    }
}

fn profile_json(profile: &ProjectionProfile) -> serde_json::Value {
    serde_json::json!({
        "schema": profile.schema,
        "branch": profile.branch,
        "call_wall_ns": profile.call_wall_ns,
        "validation_ns": profile.validation_ns,
        "buffer_init_ns": profile.buffer_init_ns,
        "input_transpose_ns": profile.input_transpose_ns,
        "rows_wall_ns": profile.rows_wall_ns,
        "output_layout_ns": profile.output_layout_ns,
        "delegated_matvec_ns": profile.delegated_matvec_ns,
        "unclassified_wall_ns": profile.unclassified_wall_ns,
        "workers": profile.workers.iter().map(|worker| serde_json::json!({
            "scratch_init_ns": worker.scratch_init_ns,
            "decode_ns": worker.decode_ns,
            "accumulate_ns": worker.accumulate_ns,
            "zero_copy_ns": worker.zero_copy_ns,
            "rows": worker.rows
        })).collect::<Vec<_>>()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    #[test]
    fn cli_rejects_short_duplicate_and_relative_arguments() {
        let valid = parse_args([
            OsString::from("--input"),
            OsString::from("/tmp/in.json"),
            OsString::from("--output"),
            OsString::from("/tmp/out.json"),
        ])
        .unwrap();
        assert_eq!(valid.input, PathBuf::from("/tmp/in.json"));
        assert_eq!(valid.output, PathBuf::from("/tmp/out.json"));
        assert!(parse_args([OsString::from("--input")]).is_err());
        assert!(parse_args([
            OsString::from("--input"),
            OsString::from("/tmp/in.json"),
            OsString::from("--output"),
            OsString::from("/tmp/out.json"),
            OsString::from("--input"),
            OsString::from("/tmp/again.json"),
        ])
        .is_err());
        assert!(parse_args([
            OsString::from("--input"),
            OsString::from("relative.json"),
            OsString::from("--output"),
            OsString::from("/tmp/out.json"),
        ])
        .is_err());
    }

    #[test]
    fn synthetic_f32_measurement_is_finite_and_repeated_calls_match_bits() {
        let input = case::CaseInput::from_json(
            br#"{"schema":3,"comparison_group":"test-group","column_tile":null,"source":{"kind":"synthetic","ggml_type":0,"rows":3,"cols":4},"batch":2,"threads":1,"profile":true,"warmup_calls":0,"measured_calls":2,"buffer_limit_bytes":1048576,"model_limit_bytes":1048576}"#,
        )
        .unwrap();
        input.validate().unwrap();
        let result = measure_case(input).unwrap();
        assert_eq!(result["all_calls_bit_identical"], true);
        assert_eq!(result["output_bits"].as_array().unwrap().len(), 6);
        assert_eq!(result["call_wall_ns"].as_array().unwrap().len(), 2);
        let profile = &result["profile_calls"][0];
        for field in [
            "schema",
            "branch",
            "call_wall_ns",
            "validation_ns",
            "buffer_init_ns",
            "input_transpose_ns",
            "rows_wall_ns",
            "output_layout_ns",
            "delegated_matvec_ns",
            "unclassified_wall_ns",
            "workers",
        ] {
            assert!(
                profile.get(field).is_some(),
                "missing profile field {field}"
            );
        }
    }

    #[cfg(feature = "projection-locality-experiment")]
    #[test]
    fn profiled_synthetic_measurements_use_each_explicit_column_tile() {
        let baseline = case::CaseInput::from_json(
            br#"{"schema":3,"comparison_group":"test-group","column_tile":null,"source":{"kind":"synthetic","ggml_type":0,"rows":3,"cols":16},"batch":33,"threads":1,"profile":true,"warmup_calls":0,"measured_calls":1,"buffer_limit_bytes":1048576,"model_limit_bytes":1048576}"#,
        ).unwrap();
        let baseline_result = measure_case(baseline).unwrap();
        let unprofiled_baseline = case::CaseInput::from_json(
            br#"{"schema":3,"comparison_group":"test-group","column_tile":null,"source":{"kind":"synthetic","ggml_type":0,"rows":3,"cols":16},"batch":33,"threads":1,"profile":false,"warmup_calls":0,"measured_calls":1,"buffer_limit_bytes":1048576,"model_limit_bytes":1048576}"#,
        ).unwrap();
        let unprofiled_baseline_result = measure_case(unprofiled_baseline).unwrap();
        for tile in [32, 64, 128] {
            let input = case::CaseInput::from_json(
                format!(r#"{{"schema":3,"comparison_group":"test-group","column_tile":{tile},"source":{{"kind":"synthetic","ggml_type":0,"rows":3,"cols":16}},"batch":33,"threads":1,"profile":true,"warmup_calls":0,"measured_calls":1,"buffer_limit_bytes":1048576,"model_limit_bytes":1048576}}"#).as_bytes(),
            )
            .unwrap();
            let result = measure_case(input).unwrap();
            assert_eq!(result["schema"], 3);
            assert_eq!(result["column_tile"], tile);
            assert_eq!(result["output_bits"], baseline_result["output_bits"]);
            assert_eq!(result["branch"], "across_batch_pair");
            assert_eq!(result["profile_calls"].as_array().unwrap().len(), 1);
            assert_eq!(result["profile_calls"][0]["branch"], "across_batch_pair");
            assert!(result["profile_calls"][0]["workers"]
                .as_array()
                .is_some_and(|workers| !workers.is_empty()));
            assert_eq!(result["profile_calls"][0]["schema"], 1);

            let input = case::CaseInput::from_json(
                format!(r#"{{"schema":3,"comparison_group":"test-group","column_tile":{tile},"source":{{"kind":"synthetic","ggml_type":0,"rows":3,"cols":16}},"batch":33,"threads":1,"profile":false,"warmup_calls":0,"measured_calls":1,"buffer_limit_bytes":1048576,"model_limit_bytes":1048576}}"#).as_bytes(),
            ).unwrap();
            let result = measure_case(input).unwrap();
            assert_eq!(result["column_tile"], tile);
            assert_eq!(result["output_bits"], unprofiled_baseline_result["output_bits"]);
            assert!(result["profile_calls"].as_array().unwrap().is_empty());
        }
    }

    #[test]
    fn synthetic_q8_measurement_returns_finite_reference_bits() {
        let input = case::CaseInput::from_json(
            br#"{"schema":3,"comparison_group":"test-group","column_tile":null,"source":{"kind":"synthetic","ggml_type":8,"rows":3,"cols":32},"batch":2,"threads":1,"profile":false,"warmup_calls":0,"measured_calls":1,"buffer_limit_bytes":1048576,"model_limit_bytes":1048576}"#,
        )
        .unwrap();
        let result = measure_case(input).unwrap();
        let bits = result["output_bits"].as_array().unwrap();
        assert_eq!(bits.len(), 6);
        assert!(bits.iter().all(|value| {
            value
                .as_u64()
                .is_some_and(|bits| f32::from_bits(bits as u32).is_finite())
        }));
    }

    #[test]
    fn synthetic_supported_formats_produce_finite_outputs() {
        for (ggml_type, cols) in [(1, 32), (30, 32), (12, 256), (14, 256)] {
            let input = case::CaseInput::from_json(
                format!(r#"{{"schema":3,"comparison_group":"test-group","column_tile":null,"source":{{"kind":"synthetic","ggml_type":{ggml_type},"rows":3,"cols":{cols}}},"batch":2,"threads":1,"profile":false,"warmup_calls":0,"measured_calls":1,"buffer_limit_bytes":1048576,"model_limit_bytes":1048576}}"#).as_bytes(),
            ).unwrap();
            let result = measure_case(input).unwrap();
            assert!(
                result["output_bits"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|value| {
                        value
                            .as_u64()
                            .is_some_and(|bits| f32::from_bits(bits as u32).is_finite())
                    }),
                "nonfinite output for GGML type {ggml_type}"
            );
        }
    }

    fn tiny_gguf(tensor_name: &str, rank: usize) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0x46554747u32.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
        bytes.extend_from_slice(tensor_name.as_bytes());
        bytes.extend_from_slice(&(rank as u32).to_le_bytes());
        for _ in 0..rank {
            bytes.extend_from_slice(&4u64.to_le_bytes());
        }
        bytes.extend_from_slice(&(GgmlType::F32 as u32).to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        let data_offset = bytes.len().div_ceil(32) * 32;
        bytes.resize(data_offset + 64, 0);
        bytes
    }

    #[test]
    fn gguf_source_rejects_missing_tensor_and_non_matrix_rank() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("tiny.gguf");
        std::fs::write(&path, tiny_gguf("other", 2)).unwrap();
        let mut missing = case::CaseInput::from_json(
            format!(r#"{{"schema":3,"comparison_group":"test-group","column_tile":null,"source":{{"kind":"gguf","model_path":"{}","tensor":"missing"}},"batch":2,"threads":1,"profile":false,"warmup_calls":0,"measured_calls":1,"buffer_limit_bytes":1048576,"model_limit_bytes":1048576}}"#, path.display()).as_bytes(),
        ).unwrap();
        assert!(load_case(&missing).is_err());
        std::fs::write(&path, tiny_gguf("tensor", 1)).unwrap();
        if let Source::Gguf { tensor, .. } = &mut missing.source {
            *tensor = "tensor".into();
        }
        assert!(load_case(&missing).is_err());
    }

    #[test]
    fn gguf_measurement_uses_metadata_type_and_dimensions() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("metadata-derived.gguf");
        std::fs::write(&path, tiny_gguf("tensor", 2)).unwrap();
        let input = case::CaseInput::from_json(
            format!(r#"{{"schema":3,"comparison_group":"test-group","column_tile":null,"source":{{"kind":"gguf","model_path":"{}","tensor":"tensor"}},"batch":2,"threads":1,"profile":false,"warmup_calls":0,"measured_calls":1,"buffer_limit_bytes":1048576,"model_limit_bytes":1048576}}"#, path.display()).as_bytes(),
        ).unwrap();
        let result = measure_case(input).unwrap();
        assert_eq!(result["format"], "F32");
        assert_eq!(result["ggml_type"], 0);
        assert_eq!(result["rows"], 4);
        assert_eq!(result["cols"], 4);
        assert_eq!(result["output_bits"].as_array().unwrap().len(), 8);
        assert_eq!(result["mapping_bytes"].as_u64().unwrap(), 160);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn verified_gguf_descriptor_survives_original_path_swap_and_reports_map_length() {
        use std::os::fd::AsRawFd;
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.gguf");
        let replacement = directory.path().join("original-verified.gguf");
        let pinned_bytes = tiny_gguf("tensor", 2);
        let pinned_len = pinned_bytes.len() as u64;
        std::fs::write(&path, &pinned_bytes).unwrap();
        let verified = private_io::open_bounded_regular_file(&path, 1024).unwrap();
        let verified_fd = verified.as_raw_fd();

        let (mapped, mapping_bytes) =
            map_gguf_from_verified_file_with(&verified, 1024, |stable_path| {
                std::fs::rename(&path, &replacement).unwrap();
                std::fs::write(&path, b"replacement is not GGUF").unwrap();
                assert_eq!(
                    stable_path,
                    PathBuf::from(format!("/proc/self/fd/{verified_fd}"))
                );
                GgufFile::open(stable_path)
            })
            .unwrap();
        assert_eq!(mapped.get_tensor_data("tensor").unwrap().0.dims, [4, 4]);
        assert_eq!(mapping_bytes, pinned_len);
        assert_eq!(mapping_bytes, mapped.mmap.len() as u64);
        assert!(GgufFile::open(&path).is_err());

        let too_small_limit = map_gguf_from_verified_file_with(&verified, 128, GgufFile::open);
        assert!(too_small_limit.is_err());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn mapping_length_is_checked_after_descriptor_metadata_check() {
        use std::io::Write;
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("growing.gguf");
        let bytes = tiny_gguf("tensor", 2);
        let original_len = bytes.len() as u64;
        std::fs::write(&path, bytes).unwrap();
        let verified = private_io::open_bounded_regular_file(&path, original_len).unwrap();
        let result = map_gguf_from_verified_file_with(&verified, original_len, |stable_path| {
            let mut writer = std::fs::OpenOptions::new().append(true).open(&path)?;
            writer.write_all(b"x")?;
            GgufFile::open(stable_path)
        });
        assert!(result.is_err());
    }

    #[cfg(unix)]
    #[test]
    fn private_result_refuses_existing_output_without_overwriting() {
        use std::os::unix::fs::PermissionsExt;
        let directory = tempfile::tempdir().unwrap();
        std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        let path = directory.path().join("result.json");
        std::fs::write(&path, b"retain").unwrap();
        assert!(private_io::PrivateOutput::create(&path).is_err());
        assert_eq!(std::fs::read(path).unwrap(), b"retain");
    }
}
