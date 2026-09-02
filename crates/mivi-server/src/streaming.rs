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

#[derive(Debug, Serialize)]
struct ChatCompletionChunkBorrow<'a> {
    id: &'a str,
    object: &'static str,
    created: u64,
    model: &'a str,
    choices: [ChunkChoiceBorrow<'a>; 1],
}

#[derive(Debug, Serialize)]
struct ChunkChoiceBorrow<'a> {
    index: usize,
    delta: ChunkDeltaBorrow<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<&'a str>,
}

#[derive(Debug, Default, Serialize)]
struct ChunkDeltaBorrow<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<&'a str>,
}

pub const OPENAI_CHUNK_OBJECT: &str = "chat.completion.chunk";
pub const OPENAI_COMPLETION_OBJECT: &str = "chat.completion";
pub const SSE_DONE_MARKER: &str = "[DONE]";
pub const ROLE_ASSISTANT: &str = "assistant";
pub const FINISH_REASON_STOP: &str = "stop";

/// Base helper to construct an SSE ChatCompletionChunk Event.
pub fn create_chunk_event(
    id: &str,
    model: &str,
    delta: ChunkDelta,
    finish_reason: Option<&str>,
) -> Event {
    let delta_borrow = ChunkDeltaBorrow {
        role: delta.role.as_deref(),
        content: delta.content.as_deref(),
        thinking: delta.thinking.as_deref(),
    };
    let chunk = ChatCompletionChunkBorrow {
        id,
        object: OPENAI_CHUNK_OBJECT,
        created: chrono::Utc::now().timestamp() as u64,
        model,
        choices: [ChunkChoiceBorrow {
            index: 0,
            delta: delta_borrow,
            finish_reason,
        }],
    };

    let data = match serde_json::to_string(&chunk) {
        Ok(json) => json,
        Err(e) => {
            tracing::error!("Failed to serialize SSE chunk: {}", e);
            "{}".to_string()
        }
    };

    Event::default().data(data)
}

/// Create the initial SSE Event establishing choices[0].delta.role = "assistant".
#[inline]
pub fn create_initial_chunk_event(id: &str, model: &str) -> Event {
    let chunk = ChatCompletionChunkBorrow {
        id,
        object: OPENAI_CHUNK_OBJECT,
        created: chrono::Utc::now().timestamp() as u64,
        model,
        choices: [ChunkChoiceBorrow {
            index: 0,
            delta: ChunkDeltaBorrow {
                role: Some(ROLE_ASSISTANT),
                content: Some(""),
                thinking: None,
            },
            finish_reason: None,
        }],
    };

    let data = match serde_json::to_string(&chunk) {
        Ok(json) => json,
        Err(e) => {
            tracing::error!("Failed to serialize SSE chunk: {}", e);
            "{}".to_string()
        }
    };

    Event::default().data(data)
}

/// Create an SSE error event adhering to standard OpenAI error format.
#[inline]
pub fn create_error_chunk_event(_id: &str, _model: &str, error: &str) -> Event {
    let payload = serde_json::json!({
        "error": {
            "message": error,
            "type": "api_error",
            "param": null,
            "code": "inference_error"
        }
    });
    Event::default().data(payload.to_string())
}

/// Create an SSE Event containing a content delta chunk without allocating extra strings.
#[inline]
pub fn create_content_chunk_event(id: &str, model: &str, content: &str) -> Event {
    let chunk = ChatCompletionChunkBorrow {
        id,
        object: OPENAI_CHUNK_OBJECT,
        created: chrono::Utc::now().timestamp() as u64,
        model,
        choices: [ChunkChoiceBorrow {
            index: 0,
            delta: ChunkDeltaBorrow {
                role: None,
                content: Some(content),
                thinking: None,
            },
            finish_reason: None,
        }],
    };

    let data = match serde_json::to_string(&chunk) {
        Ok(json) => json,
        Err(e) => {
            tracing::error!("Failed to serialize SSE chunk: {}", e);
            "{}".to_string()
        }
    };

    Event::default().data(data)
}

/// Create an SSE Event containing a thinking delta chunk.
#[inline]
pub fn create_thinking_chunk_event(id: &str, model: &str, thinking: &str) -> Event {
    let chunk = ChatCompletionChunkBorrow {
        id,
        object: OPENAI_CHUNK_OBJECT,
        created: chrono::Utc::now().timestamp() as u64,
        model,
        choices: [ChunkChoiceBorrow {
            index: 0,
            delta: ChunkDeltaBorrow {
                role: None,
                content: None,
                thinking: Some(thinking),
            },
            finish_reason: None,
        }],
    };

    let data = match serde_json::to_string(&chunk) {
        Ok(json) => json,
        Err(e) => {
            tracing::error!("Failed to serialize SSE chunk: {}", e);
            "{}".to_string()
        }
    };

    Event::default().data(data)
}

/// Create the final SSE Event with finish_reason.
#[inline]
pub fn create_done_chunk_event(id: &str, model: &str, finish_reason: &str) -> Event {
    let chunk = ChatCompletionChunkBorrow {
        id,
        object: OPENAI_CHUNK_OBJECT,
        created: chrono::Utc::now().timestamp() as u64,
        model,
        choices: [ChunkChoiceBorrow {
            index: 0,
            delta: ChunkDeltaBorrow::default(),
            finish_reason: Some(finish_reason),
        }],
    };

    let data = match serde_json::to_string(&chunk) {
        Ok(json) => json,
        Err(e) => {
            tracing::error!("Failed to serialize SSE chunk: {}", e);
            "{}".to_string()
        }
    };

    Event::default().data(data)
}

/// Create standard SSE [DONE] termination event.
#[inline]
pub fn create_done_event() -> Event {
    Event::default().data(SSE_DONE_MARKER)
}

/// Helper to send standard sequence: initial chunk -> thinking event -> content chunks -> stop chunk -> done event.
pub async fn send_sse_sequence<F, Fut>(
    tx: &tokio::sync::mpsc::Sender<std::result::Result<Event, std::convert::Infallible>>,
    id: &str,
    model: &str,
    thinking_msg: Option<&str>,
    body: F,
) where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    // Send initial assistant role chunk
    if tx.send(Ok(create_initial_chunk_event(id, model))).await.is_err() {
        return;
    }

    if let Some(msg) = thinking_msg {
        if tx
            .send(Ok(create_thinking_chunk_event(id, model, msg)))
            .await
            .is_err()
        {
            return;
        }
    }
    body().await;
    if tx
        .send(Ok(create_done_chunk_event(id, model, FINISH_REASON_STOP)))
        .await
        .is_err()
    {
        return;
    }
    let _ = tx.send(Ok(create_done_event())).await;
}
