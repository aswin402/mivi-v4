//! Opt-in real GGUF weight evaluation with generated, not captured, activations.
//! This is a projection benchmark, not a model/logit/agent quality evaluation.

use super::{packed_matmul, packed_matmul_with_kernel, PackedActivation, PackedKernel};
use crate::{quantized_matmul_rows, GgmlType};
use std::collections::{BTreeMap, BTreeSet};
use std::hint::black_box;
use std::time::Instant;

mod captured_activations;

#[derive(Debug, thiserror::Error)]
enum EvaluationError {
    #[error("projection outputs have different lengths")]
    LengthMismatch,
    #[error("projection output contains non-finite values")]
    NonFinite,
}

struct ProjectionError {
    relative_l2: f64,
    max_abs: f64,
}

impl ProjectionError {
    fn compare(expected: &[f32], actual: &[f32]) -> Result<Self, EvaluationError> {
        if expected.len() != actual.len() {
            return Err(EvaluationError::LengthMismatch);
        }
        let mut error_squared = 0.0;
        let mut reference_squared = 0.0;
        let mut max_abs = 0.0f64;
        for (&reference, &value) in expected.iter().zip(actual) {
            if !reference.is_finite() || !value.is_finite() {
                return Err(EvaluationError::NonFinite);
            }
            let error = f64::from(value) - f64::from(reference);
            error_squared += error * error;
            reference_squared += f64::from(reference).powi(2);
            max_abs = max_abs.max(error.abs());
        }
        Ok(Self {
            relative_l2: (error_squared / reference_squared.max(f64::MIN_POSITIVE)).sqrt(),
            max_abs,
        })
    }
}

type EvalResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[test]
#[ignore = "requires MIVI_TEST_MODEL; evaluates real weights with generated activations"]
fn real_weight_projection_benchmark() -> EvalResult<()> {
    let model_path =
        std::env::var_os("MIVI_TEST_MODEL").ok_or("set MIVI_TEST_MODEL to a GGUF path")?;
    let gguf = mivi_model::GgufFile::open(std::path::Path::new(&model_path))?;
    let tested = evaluate_real_projections(&gguf)?;
    assert!(tested > 0, "no real-weight projection cases were evaluated");
    Ok(())
}

fn positive_env(name: &str, default: usize) -> EvalResult<usize> {
    match std::env::var(name) {
        Ok(value) => parse_positive(name, &value),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error.into()),
    }
}

fn parse_positive(name: &str, value: &str) -> EvalResult<usize> {
    let parsed = value.parse::<usize>()?;
    if parsed == 0 {
        return Err(format!("{name} must be positive").into());
    }
    Ok(parsed)
}

fn supported_shape(ggml_type: u32, dims: &[usize]) -> bool {
    ggml_type == GgmlType::Q4_K as u32
        && dims.len() == 2
        && dims[0] > 0
        && dims[0] % super::Q4_K_BLOCK_SIZE == 0
        && dims[1] > 0
}

#[test]
fn real_projection_configuration_rejects_invalid_limits_and_shapes() {
    assert_eq!(parse_positive("rows", "1024").unwrap(), 1024);
    for value in ["0", "-1", "", "bad", "999999999999999999999999999999999999"] {
        assert!(parse_positive("rows", value).is_err());
    }
    assert!(supported_shape(GgmlType::Q4_K as u32, &[256, 3]));
    assert!(supported_shape(GgmlType::Q4_K as u32, &[512, 1024]));
    for dims in [
        &[][..],
        &[256][..],
        &[256, 3, 2][..],
        &[0, 3][..],
        &[256, 0][..],
        &[255, 3][..],
    ] {
        assert!(!supported_shape(GgmlType::Q4_K as u32, dims));
    }
    assert!(!supported_shape(GgmlType::Q6_K as u32, &[256, 3]));
}

fn evaluate_real_projections(gguf: &mivi_model::GgufFile) -> EvalResult<usize> {
    let row_cap = positive_env("MIVI_TEST_MAX_ROWS", 1024)?;
    let iterations = positive_env("MIVI_TEST_ITERS", 4)?;
    let tensor_limit = positive_env("MIVI_TEST_TENSOR_LIMIT", 2)?;
    let batches = match std::env::var("MIVI_TEST_BATCH") {
        Ok(_) => vec![positive_env("MIVI_TEST_BATCH", 32)?],
        Err(std::env::VarError::NotPresent) => vec![32, 64],
        Err(error) => return Err(error.into()),
    };
    let mut formats = BTreeMap::new();
    for tensor in gguf.tensors.values().filter(|t| t.dims.len() == 2) {
        *formats
            .entry(format!("{:?}", tensor.ggml_type))
            .or_insert(0usize) += 1;
    }
    println!("GGUF two-dimensional formats={formats:?}; Rayon threads={}; row cap={row_cap}; iterations={iterations}", rayon::current_num_threads());
    let supported = |tensor: &&mivi_model::TensorInfo| {
        // Cargo builds separate quant artifacts for this test and the model
        // dev-dependency. Compare the GGUF wire type, not cross-artifact types.
        supported_shape(tensor.ggml_type as u32, &tensor.dims)
    };
    let selected: Vec<_> = match std::env::var("MIVI_TEST_TENSORS") {
        Ok(names) => {
            let mut selected = Vec::new();
            for name in names.split(',').map(str::trim) {
                let tensor = gguf
                    .tensors
                    .get(name)
                    .ok_or_else(|| format!("tensor {name:?} not found"))?;
                if !supported(&tensor) {
                    return Err(format!("tensor {name:?} is not a supported two-dimensional Q4_K projection: {:?} {:?}", tensor.ggml_type, tensor.dims).into());
                }
                selected.push(tensor);
            }
            selected
        }
        Err(std::env::VarError::NotPresent) => {
            let mut candidates: Vec<_> = gguf.tensors.values().filter(supported).collect();
            candidates.sort_by(|a, b| {
                let work = |t: &mivi_model::TensorInfo| {
                    (t.dims[0] as u128) * (t.dims[1].min(row_cap) as u128)
                };
                work(b).cmp(&work(a)).then_with(|| a.name.cmp(&b.name))
            });
            let mut shapes = BTreeSet::new();
            candidates
                .into_iter()
                .filter(|t| shapes.insert((t.dims[0], t.dims[1].min(row_cap))))
                .take(tensor_limit)
                .collect()
        }
        Err(error) => return Err(error.into()),
    };
    if selected.is_empty() {
        return Err("GGUF has no eligible two-dimensional Q4_K matrices".into());
    }
    let mut tested = 0;
    for tensor in selected {
        let cols = tensor.dims[0];
        let rows = tensor.dims[1].min(row_cap);
        let (_, full_weights) = gguf.get_tensor_data(&tensor.name)?;
        let row_bytes = (cols / super::Q4_K_BLOCK_SIZE)
            .checked_mul(super::Q4_K_BYTES)
            .ok_or("row byte overflow")?;
        let weight_len = rows.checked_mul(row_bytes).ok_or("weight byte overflow")?;
        let weights = &full_weights[..weight_len];
        println!("tensor={:?}, full rows={}, evaluated rows={rows}, cols={cols}; first-row prefix only when capped", tensor.name, tensor.dims[1]);
        for &batch in &batches {
            for outliers in [false, true] {
                let len = batch.checked_mul(cols).ok_or("activation shape overflow")?;
                let inputs = generated_inputs(len, outliers);
                let mut reference =
                    vec![0.0; batch.checked_mul(rows).ok_or("output shape overflow")?];
                let mut scalar = reference.clone();
                let mut tiled = reference.clone();
                quantized_matmul_rows(
                    &mut reference,
                    GgmlType::Q4_K,
                    weights,
                    &inputs,
                    batch,
                    rows,
                    cols,
                )?;
                packed_matmul(&mut scalar, weights, &inputs, batch, rows, cols)?;
                let kernel = PackedKernel::detected_tiled();
                packed_matmul_with_kernel(&mut tiled, weights, &inputs, batch, rows, cols, kernel)?;
                assert_eq!(
                    scalar, tiled,
                    "real-weight tiled/scalar mismatch in {}",
                    tensor.name
                );
                let error = ProjectionError::compare(&reference, &tiled)?;
                validate_sampled_error_bound(
                    weights, &inputs, &reference, &tiled, batch, rows, cols,
                )?;
                println!(
                    "  batch={batch}, input={}, relative L2={:.6}, max abs error={:.6}",
                    if outliers {
                        "generated outliers"
                    } else {
                        "generated dense"
                    },
                    error.relative_l2,
                    error.max_abs
                );
                for packed in [false, true, true, false] {
                    let start = Instant::now();
                    for _ in 0..iterations {
                        if packed {
                            packed_matmul_with_kernel(
                                black_box(&mut tiled),
                                black_box(weights),
                                black_box(&inputs),
                                batch,
                                rows,
                                cols,
                                kernel,
                            )?;
                        } else {
                            quantized_matmul_rows(
                                black_box(&mut reference),
                                GgmlType::Q4_K,
                                black_box(weights),
                                black_box(&inputs),
                                batch,
                                rows,
                                cols,
                            )?;
                        }
                        black_box(if packed { &tiled } else { &reference });
                    }
                    println!(
                        "    {}: {:.3} ms/projection (packing/transposition included)",
                        if packed {
                            kernel.label()
                        } else {
                            "existing F32"
                        },
                        start.elapsed().as_secs_f64() * 1000.0 / iterations as f64
                    );
                }
                tested += 1;
            }
        }
    }
    Ok(tested)
}

fn generated_inputs(len: usize, outliers: bool) -> Vec<f32> {
    let mut state = 0x1234_5678u32;
    (0..len)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let value = (state >> 8) as f32 / 16_777_216.0 * 2.0 - 1.0;
            if outliers {
                if i % super::Q4_K_BLOCK_SIZE == 37 {
                    value.signum() * 12.0
                } else {
                    value * 0.25
                }
            } else {
                value
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn validate_sampled_error_bound(
    weights: &[u8],
    inputs: &[f32],
    reference: &[f32],
    tiled: &[f32],
    batch: usize,
    rows: usize,
    cols: usize,
) -> EvalResult<()> {
    let blocks = cols / super::Q4_K_BLOCK_SIZE;
    let row_bytes = blocks * super::Q4_K_BYTES;
    for row in [0, rows / 2, rows - 1] {
        let mut decoded = vec![0.0; cols];
        crate::q4_k_m::dequantize_q4_k_m_slice(
            &weights[row * row_bytes..(row + 1) * row_bytes],
            &mut decoded,
        );
        for b in [0, batch / 2, batch - 1] {
            let input = &inputs[b * cols..(b + 1) * cols];
            let mut expected = 0.0;
            let mut magnitude = 0.0;
            let mut bound = 0.0;
            for (w, x) in decoded
                .chunks_exact(super::Q4_K_BLOCK_SIZE)
                .zip(input.chunks_exact(super::Q4_K_BLOCK_SIZE))
            {
                let packed = PackedActivation::pack(x)?;
                bound += w.iter().map(|&v| f64::from(v).abs()).sum::<f64>()
                    * f64::from(packed.scale).abs()
                    * 0.501;
                for (&weight, &activation) in w.iter().zip(x) {
                    let product = f64::from(weight) * f64::from(activation);
                    expected += product;
                    magnitude += product.abs();
                }
            }
            assert!(
                (f64::from(reference[b * rows + row]) - expected).abs() <= magnitude * 1e-5 + 1e-6,
                "existing F32 disagrees with sampled f64 reference"
            );
            assert!(
                (f64::from(tiled[b * rows + row]) - expected).abs()
                    <= bound + magnitude * 1e-5 + 1e-6,
                "packed result exceeded sampled quantization error bound"
            );
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_group32_sampled_error_bound(
    weights: &[u8],
    inputs: &[f32],
    reference: &[f32],
    actual: &[f32],
    batch: usize,
    rows: usize,
    cols: usize,
) -> EvalResult<()> {
    let blocks = cols / super::Q4_K_BLOCK_SIZE;
    let row_bytes = blocks * super::Q4_K_BYTES;
    for row in [0, rows / 2, rows - 1] {
        let mut decoded = vec![0.0; cols];
        crate::q4_k_m::dequantize_q4_k_m_slice(
            &weights[row * row_bytes..(row + 1) * row_bytes],
            &mut decoded,
        );
        for b in [0, batch / 2, batch - 1] {
            let input = &inputs[b * cols..(b + 1) * cols];
            let mut expected = 0.0f64;
            let mut magnitude = 0.0f64;
            let mut reconstruction_bound = 0.0f64;
            let mut affine_components = 0.0f64;
            for (block_index, (w, x)) in decoded
                .chunks_exact(super::Q4_K_BLOCK_SIZE)
                .zip(input.chunks_exact(super::Q4_K_BLOCK_SIZE))
                .enumerate()
            {
                let activation = super::group32::Group32Activation::pack(x)?;
                let reconstructed = activation.reconstructed();
                let weight_start = row * row_bytes + block_index * super::Q4_K_BYTES;
                let packed_weight = super::PackedWeights::new(
                    &weights[weight_start..weight_start + super::Q4_K_BYTES],
                );
                for ((&weight, &original), reconstructed) in w.iter().zip(x).zip(reconstructed) {
                    let product = f64::from(weight) * f64::from(original);
                    expected += product;
                    magnitude += product.abs();
                    reconstruction_bound += f64::from(weight).abs()
                        * (f64::from(reconstructed) - f64::from(original)).abs();
                }

                for group in 0..super::group32::GROUPS {
                    let group_input = &x[group * super::group32::GROUP_WIDTH
                        ..(group + 1) * super::group32::GROUP_WIDTH];
                    let max_abs = group_input
                        .iter()
                        .fold(0.0f32, |max, value| max.max(value.abs()));
                    if max_abs == 0.0 {
                        continue;
                    }
                    let activation_scale = max_abs / 127.0;
                    let packed_values = &reconstructed[group * super::group32::GROUP_WIDTH
                        ..(group + 1) * super::group32::GROUP_WIDTH];
                    let activation_values: Vec<i32> = packed_values
                        .iter()
                        .map(|value| (value / -activation_scale).round() as i32)
                        .collect();
                    let start = (group / 2) * super::group32::GROUP_WIDTH;
                    let quant_bytes =
                        &packed_weight.quants[start..start + super::group32::GROUP_WIDTH];
                    let mut integer_magnitude = 0i64;
                    let mut activation_sum = 0i64;
                    for (&byte, &activation_value) in quant_bytes.iter().zip(&activation_values) {
                        let quant = if group % 2 == 0 { byte & 15 } else { byte >> 4 };
                        integer_magnitude += i64::from(quant) * i64::from(activation_value).abs();
                        activation_sum += i64::from(activation_value);
                    }
                    let weight_term =
                        f64::from(packed_weight.scale) * f64::from(packed_weight.scales[group]);
                    let minimum_term =
                        f64::from(packed_weight.min_scale) * f64::from(packed_weight.mins[group]);
                    affine_components += f64::from(activation_scale)
                        * (weight_term.abs() * integer_magnitude as f64
                            + minimum_term.abs() * activation_sum.abs() as f64);
                }
            }
            let reference_rounding = gamma(cols.saturating_add(8)) * magnitude
                + f64::from(f32::from_bits(1)) * cols as f64;
            assert!(
                (f64::from(reference[b * rows + row]) - expected).abs() <= reference_rounding,
                "existing F32 disagrees with the sampled f64 reference"
            );
            let affine_operations =
                (7 * super::group32::GROUPS + 1) * blocks + super::group32::GROUPS + 4;
            let codec_rounding = gamma(affine_operations) * affine_components
                + f64::from(f32::from_bits(1)) * (super::group32::GROUPS * blocks) as f64;
            let bound = reconstruction_bound + reference_rounding + codec_rounding;
            assert!(
                (f64::from(actual[b * rows + row]) - expected).abs() <= bound,
                "group-32 result exceeded its reconstruction and arithmetic bound"
            );
        }
    }
    Ok(())
}

fn gamma(operations: usize) -> f64 {
    // Standard forward-error factor for a sequence of rounded f32 operations.
    let product = operations as f64 * f64::from(f32::EPSILON);
    product / (1.0 - product)
}

#[test]
fn group32_sampled_bound_covers_affine_cancellation_and_multiple_blocks() {
    let (rows, batch, cols) = (3, 3, 512);
    let weights = crate::q4_k_m::packed_prefill::fixture_weights(rows, cols, 71);
    let inputs: Vec<_> = (0..batch * cols)
        .map(|i| {
            let within_block = i % super::Q4_K_BLOCK_SIZE;
            if within_block % 2 == 0 {
                0.375
            } else {
                -0.375
            }
        })
        .collect();
    let mut reference = vec![0.0; rows * batch];
    let mut actual = reference.clone();
    quantized_matmul_rows(
        &mut reference,
        GgmlType::Q4_K,
        &weights,
        &inputs,
        batch,
        rows,
        cols,
    )
    .unwrap();
    super::group32::packed_matmul(&mut actual, &weights, &inputs, batch, rows, cols).unwrap();
    validate_group32_sampled_error_bound(&weights, &inputs, &reference, &actual, batch, rows, cols)
        .unwrap();
}

#[test]
fn projection_error_measures_difference_and_rejects_invalid_outputs() {
    let metrics = ProjectionError::compare(&[1.0, 2.0], &[1.1, 1.9]).unwrap();
    assert!((metrics.relative_l2 - (0.02f64 / 5.0).sqrt()).abs() < 1e-7);
    assert!((metrics.max_abs - 0.1).abs() < 1e-7);
    assert!(matches!(
        ProjectionError::compare(&[1.0], &[]),
        Err(EvaluationError::LengthMismatch)
    ));
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(matches!(
            ProjectionError::compare(&[invalid], &[1.0]),
            Err(EvaluationError::NonFinite)
        ));
        assert!(matches!(
            ProjectionError::compare(&[1.0], &[invalid]),
            Err(EvaluationError::NonFinite)
        ));
    }
    let zero = ProjectionError::compare(&[0.0], &[0.0]).unwrap();
    assert_eq!(zero.relative_l2, 0.0);
    assert_eq!(zero.max_abs, 0.0);
}
