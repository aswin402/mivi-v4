use mivi_quant::GgmlType;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaseInput {
    pub schema: u32,
    pub source: Source,
    pub batch: usize,
    pub threads: usize,
    pub profile: bool,
    pub warmup_calls: usize,
    pub measured_calls: usize,
    pub buffer_limit_bytes: usize,
    pub model_limit_bytes: u64,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Source {
    Synthetic {
        ggml_type: u32,
        rows: usize,
        cols: usize,
    },
    Gguf {
        model_path: PathBuf,
        tensor: String,
    },
}

impl CaseInput {
    pub fn from_json(bytes: &[u8]) -> Result<Self, String> {
        let input: Self = serde_json::from_slice(bytes).map_err(|_| "invalid case JSON")?;
        input.validate()?;
        Ok(input)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != 1 {
            return Err("unsupported case schema".into());
        }
        if !(1..=65).contains(&self.batch) || !(1..=2).contains(&self.threads) {
            return Err("batch or thread count is outside the allowed range".into());
        }
        if self.warmup_calls > 1 || !(1..=32).contains(&self.measured_calls) {
            return Err("warmup or measured call count is outside the allowed range".into());
        }
        if self.buffer_limit_bytes == 0 || self.model_limit_bytes == 0 {
            return Err("allocation limits must be positive".into());
        }
        match &self.source {
            Source::Synthetic {
                ggml_type,
                rows,
                cols,
            } => {
                supported_type(*ggml_type)?;
                if *rows == 0 || *cols == 0 {
                    return Err("synthetic rows and columns must be positive".into());
                }
                let format = GgmlType::from_u32(*ggml_type).map_err(|_| "unsupported format")?;
                validate_alignment(format, *cols)?;
                self.required_heap_bytes(format, *rows, *cols)?;
            }
            Source::Gguf { model_path, tensor } => {
                if !model_path.is_absolute()
                    || tensor.is_empty()
                    || tensor.len() > 16 * 1024
                    || model_path.to_string_lossy().len() > 16 * 1024
                {
                    return Err(
                        "GGUF source requires an absolute model path and tensor name".into(),
                    );
                }
            }
        }
        self.output_artifact_bound()?;
        Ok(())
    }

    pub fn required_heap_bytes(
        &self,
        ggml_type: GgmlType,
        rows: usize,
        cols: usize,
    ) -> Result<usize, String> {
        if rows == 0 || cols == 0 {
            return Err("rows and columns must be positive".into());
        }
        validate_alignment(ggml_type, cols)?;
        let elements = |a: usize, b: usize| {
            a.checked_mul(b)
                .ok_or_else(|| "buffer size overflow".to_owned())
        };
        let bytes = |count: usize, width: usize| elements(count, width);
        let input_elements = elements(self.batch, cols)?;
        let output_elements = elements(self.batch, rows)?;
        let mut total = bytes(input_elements, 4)?;
        total = total
            .checked_add(bytes(output_elements, 4)?)
            .ok_or("buffer size overflow")?;
        total = total
            .checked_add(bytes(output_elements, 4)?)
            .ok_or("buffer size overflow")?;
        if self.batch > 8 {
            total = total
                .checked_add(bytes(input_elements, 4)?)
                .ok_or("buffer size overflow")?;
        }
        let workers = rows.min(self.threads);
        let scratch_floats = if self.batch >= 32 {
            elements(2, cols)?
        } else {
            cols.checked_add(self.batch).ok_or("buffer size overflow")?
        };
        total = total
            .checked_add(bytes(elements(workers, scratch_floats)?, 4)?)
            .ok_or("buffer size overflow")?;
        if let Source::Synthetic { .. } = self.source {
            let weight_row_bytes = (cols / ggml_type.block_size().ok_or("unsupported format")?)
                .checked_mul(ggml_type.type_size().ok_or("unsupported format")?)
                .ok_or("buffer size overflow")?;
            total = total
                .checked_add(elements(rows, weight_row_bytes)?)
                .ok_or("buffer size overflow")?;
        }
        // Keep one f32 bit-pattern reference for the driver's paired comparison.
        total = total
            .checked_add(bytes(output_elements, 4)?)
            .ok_or("buffer size overflow")?;
        if total > self.buffer_limit_bytes {
            return Err("case exceeds buffer_limit_bytes".into());
        }
        Ok(total)
    }

    pub fn output_artifact_bound(&self) -> Result<usize, String> {
        let output_values = match &self.source {
            Source::Synthetic { rows, .. } => self.batch.checked_mul(*rows),
            Source::Gguf { .. } => None,
        };
        if let Some(values) = output_values {
            let bound = values
                .checked_mul(11)
                .and_then(|n| n.checked_add(256 * 1024))
                .ok_or("output artifact bound overflow")?;
            if bound > 4 * 1024 * 1024 {
                return Err("output artifact bound exceeds the private writer limit; select a smaller batch".into());
            }
            Ok(bound)
        } else {
            Ok(4 * 1024 * 1024)
        }
    }
}

pub fn supported_type(raw: u32) -> Result<GgmlType, String> {
    let format = GgmlType::from_u32(raw).map_err(|_| "unsupported format")?;
    if matches!(
        format,
        GgmlType::F32
            | GgmlType::F16
            | GgmlType::BF16
            | GgmlType::Q8_0
            | GgmlType::Q4_K
            | GgmlType::Q6_K
    ) {
        Ok(format)
    } else {
        Err("unsupported format".into())
    }
}

pub fn validate_alignment(format: GgmlType, cols: usize) -> Result<(), String> {
    let block = format.block_size().ok_or("unsupported format")?;
    if !cols.is_multiple_of(block) {
        return Err("column count is not aligned to the format block".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID: &[u8] = br#"{"schema":1,"source":{"kind":"synthetic","ggml_type":0,"rows":3,"cols":4},"batch":2,"threads":1,"profile":false,"warmup_calls":0,"measured_calls":1,"buffer_limit_bytes":1048576,"model_limit_bytes":1048576}"#;

    #[test]
    fn input_rejects_unknown_fields_and_boolean_counts() {
        let unknown = br#"{"schema":1,"source":{"kind":"synthetic","ggml_type":0,"rows":3,"cols":4},"batch":2,"threads":1,"profile":false,"warmup_calls":0,"measured_calls":1,"buffer_limit_bytes":1048576,"model_limit_bytes":1048576,"extra":0}"#;
        assert!(CaseInput::from_json(unknown).is_err());
        let boolean_count = std::str::from_utf8(VALID)
            .unwrap()
            .replace("\"batch\":2", "\"batch\":true");
        assert!(CaseInput::from_json(boolean_count.as_bytes()).is_err());
    }

    #[test]
    fn input_rejects_invalid_counts_formats_and_overflowing_budgets() {
        let unsupported = std::str::from_utf8(VALID)
            .unwrap()
            .replace("\"ggml_type\":0", "\"ggml_type\":999");
        assert!(CaseInput::from_json(unsupported.as_bytes()).is_err());
        let mut input = CaseInput::from_json(VALID).unwrap();
        input.batch = 66;
        assert!(input.validate().is_err());
        input.batch = 2;
        input.buffer_limit_bytes = usize::MAX;
        input.source = Source::Synthetic {
            ggml_type: 0,
            rows: usize::MAX,
            cols: 4,
        };
        assert!(input
            .required_heap_bytes(GgmlType::F32, usize::MAX, 4)
            .is_err());
    }
}
