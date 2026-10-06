//! Opt-in diagnostics for the shared quantized row projection kernel.

use crate::{
    quantized_matmul_rows_impl, quantized_matmul_rows_with_tile_impl, GgmlType,
    ProjectionProfileInternal, Result, WorkerWorkInternal,
};

/// Timed work performed by one existing output-row worker chunk.
#[derive(Debug, Clone, Default)]
pub struct WorkerWork {
    pub scratch_init_ns: u64,
    pub decode_ns: u64,
    pub accumulate_ns: u64,
    pub zero_copy_ns: u64,
    pub rows: usize,
}

/// Stage timings for one shared-kernel row projection call.
#[derive(Debug, Clone)]
pub struct ProjectionProfile {
    pub schema: u32,
    pub branch: &'static str,
    pub call_wall_ns: u64,
    pub validation_ns: u64,
    pub buffer_init_ns: Option<u64>,
    pub input_transpose_ns: Option<u64>,
    pub rows_wall_ns: Option<u64>,
    pub output_layout_ns: Option<u64>,
    pub delegated_matvec_ns: Option<u64>,
    pub unclassified_wall_ns: u64,
    pub workers: Vec<WorkerWork>,
}

/// Run the same checked projection kernel while collecting opt-in stage timings.
pub fn quantized_matmul_rows_profiled(
    out: &mut [f32],
    ggml_type: GgmlType,
    weights: &[u8],
    inputs: &[f32],
    batch: usize,
    rows: usize,
    cols: usize,
) -> Result<ProjectionProfile> {
    let profile =
        quantized_matmul_rows_impl::<true>(out, ggml_type, weights, inputs, batch, rows, cols)?
            .expect("profiled kernel invocation returns diagnostics");
    Ok(from_internal(profile))
}

/// Run the checked projection kernel with an explicit column panel width while
/// collecting opt-in stage timings.
#[cfg(feature = "projection-locality-experiment")]
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_rows_profiled_with_column_tile(
    out: &mut [f32],
    ggml_type: GgmlType,
    weights: &[u8],
    inputs: &[f32],
    batch: usize,
    rows: usize,
    cols: usize,
    tile: mivi_core::simd::ProjectionColumnTile,
) -> Result<ProjectionProfile> {
    let profile = quantized_matmul_rows_with_tile_impl::<true>(
        out,
        ggml_type,
        weights,
        inputs,
        batch,
        rows,
        cols,
        Some(tile),
    )?
    .expect("profiled kernel invocation returns diagnostics");
    Ok(from_internal(profile))
}

fn from_internal(profile: ProjectionProfileInternal) -> ProjectionProfile {
    ProjectionProfile {
        schema: profile.schema,
        branch: profile.branch,
        call_wall_ns: profile.call_wall_ns,
        validation_ns: profile.validation_ns,
        buffer_init_ns: profile.buffer_init_ns,
        input_transpose_ns: profile.input_transpose_ns,
        rows_wall_ns: profile.rows_wall_ns,
        output_layout_ns: profile.output_layout_ns,
        delegated_matvec_ns: profile.delegated_matvec_ns,
        unclassified_wall_ns: profile.unclassified_wall_ns,
        workers: profile.workers.into_iter().map(from_worker).collect(),
    }
}

fn from_worker(worker: WorkerWorkInternal) -> WorkerWork {
    WorkerWork {
        scratch_init_ns: worker.scratch_init_ns,
        decode_ns: worker.decode_ns,
        accumulate_ns: worker.accumulate_ns,
        zero_copy_ns: worker.zero_copy_ns,
        rows: worker.rows,
    }
}
