//! Server-Sent Events (SSE) streaming types and chunk formatters.

use axum::response::sse::Event;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: ChunkDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
}

/// Create an SSE Event containing a content delta chunk.
pub fn create_content_chunk_event(
    id: &str,
    model: &str,
    content: &str,
) -> Event {
    let chunk = ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created: chrono::Utc::now().timestamp() as u64,
        model: model.to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: ChunkDelta {
                role: None,
                content: Some(content.to_string()),
                thinking: None,
                tool_calls: None,
            },
            finish_reason: None,
        }],
    };

    Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
}

/// Create an SSE Event containing a thinking delta chunk.
pub fn create_thinking_chunk_event(
    id: &str,
    model: &str,
    thinking: &str,
) -> Event {
    let chunk = ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created: chrono::Utc::now().timestamp() as u64,
        model: model.to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: ChunkDelta {
                role: None,
                content: None,
                thinking: Some(thinking.to_string()),
                tool_calls: None,
            },
            finish_reason: None,
        }],
    };

    Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
}

/// Create the final SSE Event with finish_reason.
pub fn create_done_chunk_event(
    id: &str,
    model: &str,
    finish_reason: &str,
) -> Event {
    let chunk = ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created: chrono::Utc::now().timestamp() as u64,
        model: model.to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: ChunkDelta::default(),
            finish_reason: Some(finish_reason.to_string()),
        }],
    };

    Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
}
