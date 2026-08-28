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

impl MemoryStore {
    pub fn new(base_dir: &Path) -> Self {
        let root_dir = base_dir.join(".mivi").join("memory");
        Self { root_dir }
    }

    pub fn init(&self) -> Result<()> {
        std::fs::create_dir_all(&self.root_dir)?;
        Ok(())
    }

    pub fn save_record(&self, record: &MemoryRecord) -> Result<PathBuf> {
        self.init()?;
        // Sanitize type and id: strip any slashes, backslashes, or dots
        let mut clean_type: String = record
            .r#type
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
            .collect();
        if clean_type.is_empty() {
            clean_type = "record".to_string();
        }

        let mut clean_id: String = record
            .id
            .to_string()
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
            .collect();
        if clean_id.is_empty() {
            clean_id = uuid::Uuid::new_v4().to_string();
        }

        let filename = format!("{}_{}.md", clean_type, clean_id);
        let path = self.root_dir.join(filename);

        let yaml_frontmatter = serde_yaml::to_string(record)?;
        let full_content = format!("---\n{}\n---\n\n{}", yaml_frontmatter, record.content);
        std::fs::write(&path, full_content)?;

        Ok(path)
    }
}
