//! GGUF v3 file parser with zero-copy memory mapping and security validation.

use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use mivi_quant::GgmlType;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read, Seek};
use std::path::Path;
use thiserror::Error;

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in LE
const MAX_STRING_LEN: usize = 1024 * 1024; // 1 MB limit for individual strings
const MAX_METADATA_COUNT: usize = 100_000;
const MAX_TENSOR_COUNT: usize = 50_000;
const MAX_ARRAY_LEN: usize = 10_000_000;

#[derive(Error, Debug)]
pub enum GgufError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid GGUF magic header: expected 0x46554747, got 0x{0:X}")]
    InvalidMagic(u32),
    #[error("Unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),
    #[error("Invalid UTF-8 string: {0}")]
    InvalidString(#[from] std::string::FromUtf8Error),
    #[error("Unsupported metadata value type: {0}")]
    UnsupportedValueType(u32),
    #[error("Tensor '{0}' not found in GGUF file")]
    TensorNotFound(String),
    #[error("Malformed GGUF file: {0}")]
    MalformedFile(String),
}

pub type Result<T> = std::result::Result<T, GgufError>;

#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub dims: Vec<usize>,
    pub ggml_type: GgmlType,
    pub offset: u64,
}

pub struct GgufFile {
    pub version: u32,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: HashMap<String, TensorInfo>,
    pub data_offset: usize,
    pub mmap: Mmap,
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < 16 {
            return Err(GgufError::MalformedFile("File too small to contain valid GGUF header".into()));
        }

        let mut cursor = Cursor::new(&mmap[..]);

        let magic = cursor.read_u32::<LittleEndian>()?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic(magic));
        }

        let version = cursor.read_u32::<LittleEndian>()?;
        if version != 2 && version != 3 {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let tensor_count = cursor.read_u64::<LittleEndian>()? as usize;
        if tensor_count > MAX_TENSOR_COUNT {
            return Err(GgufError::MalformedFile(format!(
                "Tensor count {} exceeds security limit {}",
                tensor_count, MAX_TENSOR_COUNT
            )));
        }

        let metadata_kv_count = cursor.read_u64::<LittleEndian>()? as usize;
        if metadata_kv_count > MAX_METADATA_COUNT {
            return Err(GgufError::MalformedFile(format!(
                "Metadata count {} exceeds security limit {}",
                metadata_kv_count, MAX_METADATA_COUNT
            )));
        }

        // Parse metadata KV pairs
        let mut metadata = HashMap::with_capacity(metadata_kv_count);
        for _ in 0..metadata_kv_count {
            let key = read_gguf_string(&mut cursor)?;
            let val = read_gguf_value(&mut cursor)?;
            metadata.insert(key, val);
        }

        // Parse tensor info
        let mut tensors = HashMap::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = read_gguf_string(&mut cursor)?;
            let n_dims = cursor.read_u32::<LittleEndian>()?;
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(cursor.read_u64::<LittleEndian>()? as usize);
            }
            let raw_type = cursor.read_u32::<LittleEndian>()?;
            let ggml_type = GgmlType::from_u32(raw_type)
                .map_err(|_| GgufError::UnsupportedValueType(raw_type))?;
            let offset = cursor.read_u64::<LittleEndian>()?;

            tensors.insert(
                name.clone(),
                TensorInfo {
                    name,
                    n_dims,
                    dims,
                    ggml_type,
                    offset,
                },
            );
        }

        // 32-byte alignment for tensor data offset
        let current_pos = cursor.position() as usize;
        let alignment = match metadata.get("general.alignment") {
            Some(GgufValue::U32(v)) => *v as usize,
            _ => 32,
        };
        let data_offset = (current_pos + alignment - 1) & !(alignment - 1);

        Ok(Self {
            version,
            metadata,
            tensors,
            data_offset,
            mmap,
        })
    }

    /// Read raw tensor bytes by name with strict overflow and bounds verification
    pub fn get_tensor_data(&self, name: &str) -> Result<(&TensorInfo, &[u8])> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_string()))?;

        let start = self
            .data_offset
            .checked_add(info.offset as usize)
            .ok_or_else(|| GgufError::MalformedFile("Integer overflow in tensor start offset".into()))?;

        let num_elements = info
            .dims
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| GgufError::MalformedFile("Integer overflow in tensor dimensions".into()))?;

        let type_size = info.ggml_type.type_size();
        let block_size = info.ggml_type.block_size();
        if block_size == 0 {
            return Err(GgufError::MalformedFile("Block size cannot be zero".into()));
        }

        let bytes_len = (num_elements.checked_mul(type_size))
            .ok_or_else(|| GgufError::MalformedFile("Integer overflow in tensor byte length".into()))?
            / block_size;

        let end = start
            .checked_add(bytes_len)
            .ok_or_else(|| GgufError::MalformedFile("Integer overflow in tensor range".into()))?;

        if end > self.mmap.len() {
            return Err(GgufError::MalformedFile(format!(
                "Tensor '{}' range [{}..{}] exceeds mapped file size {}",
                name,
                start,
                end,
                self.mmap.len()
            )));
        }

        Ok((info, &self.mmap[start..end]))
    }
}

fn read_gguf_string<R: Read>(r: &mut R) -> Result<String> {
    let len = r.read_u64::<LittleEndian>()? as usize;
    if len > MAX_STRING_LEN {
        return Err(GgufError::MalformedFile(format!(
            "String length {} exceeds limit {}",
            len, MAX_STRING_LEN
        )));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8(buf)?)
}

fn read_gguf_value<R: Read + Seek>(r: &mut R) -> Result<GgufValue> {
    let val_type = r.read_u32::<LittleEndian>()?;
    read_value_by_type(r, val_type)
}

fn read_value_by_type<R: Read + Seek>(r: &mut R, val_type: u32) -> Result<GgufValue> {
    match val_type {
        0 => Ok(GgufValue::U8(r.read_u8()?)),
        1 => Ok(GgufValue::I8(r.read_i8()?)),
        2 => Ok(GgufValue::U16(r.read_u16::<LittleEndian>()?)),
        3 => Ok(GgufValue::I16(r.read_i16::<LittleEndian>()?)),
        4 => Ok(GgufValue::U32(r.read_u32::<LittleEndian>()?)),
        5 => Ok(GgufValue::I32(r.read_i32::<LittleEndian>()?)),
        6 => Ok(GgufValue::F32(r.read_f32::<LittleEndian>()?)),
        7 => Ok(GgufValue::Bool(r.read_u8()? != 0)),
        8 => Ok(GgufValue::String(read_gguf_string(r)?)),
        9 => {
            let elem_type = r.read_u32::<LittleEndian>()?;
            let len = r.read_u64::<LittleEndian>()? as usize;
            if len > MAX_ARRAY_LEN {
                return Err(GgufError::MalformedFile(format!(
                    "Array length {} exceeds limit {}",
                    len, MAX_ARRAY_LEN
                )));
            }
            let mut arr = Vec::with_capacity(len);
            for _ in 0..len {
                arr.push(read_value_by_type(r, elem_type)?);
            }
            Ok(GgufValue::Array(arr))
        }
        10 => Ok(GgufValue::U64(r.read_u64::<LittleEndian>()?)),
        11 => Ok(GgufValue::I64(r.read_i64::<LittleEndian>()?)),
        12 => Ok(GgufValue::F64(r.read_f64::<LittleEndian>()?)),
        other => Err(GgufError::UnsupportedValueType(other)),
    }
}
