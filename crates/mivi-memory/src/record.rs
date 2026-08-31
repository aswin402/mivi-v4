//! Open Knowledge Format (OKF) compliant memory records.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryType {
    Episodic,
    Semantic,
    Procedural,
    Entity,
    Custom(String),
}

impl std::fmt::Display for MemoryType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Episodic => write!(f, "episodic"),
            Self::Semantic => write!(f, "semantic"),
            Self::Procedural => write!(f, "procedural"),
            Self::Entity => write!(f, "entity"),
            Self::Custom(s) => write!(f, "{}", s),
        }
    }
}

impl std::str::FromStr for MemoryType {
    type Err = std::convert::Infallible;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        if s.eq_ignore_ascii_case("episodic") {
            Ok(Self::Episodic)
        } else if s.eq_ignore_ascii_case("semantic") {
            Ok(Self::Semantic)
        } else if s.eq_ignore_ascii_case("procedural") {
            Ok(Self::Procedural)
        } else if s.eq_ignore_ascii_case("entity") {
            Ok(Self::Entity)
        } else {
            Ok(Self::Custom(s.to_string()))
        }
    }
}

impl From<&str> for MemoryType {
    fn from(s: &str) -> Self {
        s.parse().unwrap_or(MemoryType::Custom(s.to_string()))
    }
}

impl From<String> for MemoryType {
    fn from(s: String) -> Self {
        s.as_str().into()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: Uuid,
    pub r#type: MemoryType,
    pub title: String,
    pub tags: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub importance: f32,
    #[serde(skip)]
    pub content: String,
}

impl MemoryRecord {
    pub fn new(
        r#type: impl Into<MemoryType>,
        title: &str,
        content: &str,
        tags: Vec<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            r#type: r#type.into(),
            title: title.to_string(),
            tags,
            created_at: Utc::now(),
            importance: 1.0,
            content: content.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_type_parsing_and_display() {
        let t: MemoryType = "episodic".into();
        assert_eq!(t, MemoryType::Episodic);
        assert_eq!(t.to_string(), "episodic");

        let custom: MemoryType = "conversation".into();
        assert_eq!(custom, MemoryType::Custom("conversation".to_string()));
        assert_eq!(custom.to_string(), "conversation");
    }

    #[test]
    fn test_memory_record_creation() {
        let record = MemoryRecord::new(
            MemoryType::Semantic,
            "Rust basics",
            "Rust uses ownership and borrowing.",
            vec!["rust".to_string(), "guide".to_string()],
        );
        assert_eq!(record.r#type, MemoryType::Semantic);
        assert_eq!(record.title, "Rust basics");
        assert_eq!(record.tags.len(), 2);
    }
}
