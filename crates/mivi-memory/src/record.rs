//! Open Knowledge Format (OKF) compliant memory records.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: Uuid,
    pub r#type: String,
    pub title: String,
    pub tags: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub importance: f32,
    pub content: String,
}

impl MemoryRecord {
    pub fn new(r#type: &str, title: &str, content: &str, tags: Vec<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            r#type: r#type.to_string(),
            title: title.to_string(),
            tags,
            created_at: Utc::now(),
            importance: 1.0,
            content: content.to_string(),
        }
    }
}
