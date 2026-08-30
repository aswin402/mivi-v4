//! Persistent memory store in .mivi/ directory.

use crate::record::MemoryRecord;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MemoryError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("YAML error: {0}")]
    Yaml(#[from] serde_yaml::Error),
}

pub type Result<T> = std::result::Result<T, MemoryError>;

pub struct MemoryStore {
    root_dir: PathBuf,
}

pub const STORAGE_DIR_NAME: &str = ".mivi";
pub const MEMORY_SUBDIR_NAME: &str = "memory";
pub const FALLBACK_RECORD_TYPE: &str = "record";
pub const FALLBACK_RECORD_ID: &str = "unknown_id";

#[inline]
fn sanitize_identifier(s: &str, fallback: &str) -> String {
    let clean: String = s
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
        .collect();
    if clean.is_empty() {
        fallback.to_string()
    } else {
        clean
    }
}

impl MemoryStore {
    pub fn new(base_dir: &Path) -> Self {
        let root_dir = base_dir.join(STORAGE_DIR_NAME).join(MEMORY_SUBDIR_NAME);
        Self { root_dir }
    }

    pub fn init(&self) -> Result<()> {
        std::fs::create_dir_all(&self.root_dir)?;
        Ok(())
    }

    pub fn save_record(&self, record: &MemoryRecord) -> Result<PathBuf> {
        self.init()?;
        let clean_type = sanitize_identifier(&record.r#type.to_string(), FALLBACK_RECORD_TYPE);
        let clean_id = sanitize_identifier(&record.id.to_string(), FALLBACK_RECORD_ID);

        let filename = format!("{}_{}.md", clean_type, clean_id);
        let path = self.root_dir.join(filename);

        let yaml_frontmatter = serde_yaml::to_string(record)?;
        let full_content = format!("---\n{}\n---\n\n{}", yaml_frontmatter, record.content);
        std::fs::write(&path, full_content)?;

        Ok(path)
    }

    pub fn load_record(&self, path: &Path) -> Result<MemoryRecord> {
        let raw = std::fs::read_to_string(path)?;
        let normalized = raw.replace("\r\n", "\n");
        let trimmed = normalized.trim_start();
        if let Some(rest) = trimmed.strip_prefix("---") {
            let rest = rest.strip_prefix('\n').unwrap_or(rest);
            if let Some((yaml_str, body)) = rest.split_once("\n---") {
                let body = body.strip_prefix('\n').unwrap_or(body);
                let mut record: MemoryRecord = serde_yaml::from_str(yaml_str)?;
                record.content = body.trim_start().to_string();
                return Ok(record);
            }
        }
        let record: MemoryRecord = serde_yaml::from_str(&raw)?;
        Ok(record)
    }

    pub fn list_records(&self) -> Result<Vec<PathBuf>> {
        self.init()?;
        let mut paths = Vec::new();
        for entry in std::fs::read_dir(&self.root_dir)? {
            let entry = entry?;
            let p = entry.path();
            if p.extension().and_then(|e| e.to_str()) == Some("md") {
                paths.push(p);
            }
        }
        Ok(paths)
    }
}
