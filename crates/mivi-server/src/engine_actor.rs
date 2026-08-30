//! Dedicated Engine Actor managing model inference on an isolated worker thread.

use tokio::sync::{mpsc, oneshot};

#[derive(Debug)]
pub enum EngineCommand {
    Generate {
        prompt: String,
        max_tokens: usize,
        responder: oneshot::Sender<Result<(String, usize, usize), String>>,
    },
    GenerateStream {
        prompt: String,
        max_tokens: usize,
        responder: mpsc::Sender<Result<String, String>>,
    },
    Encode {
        text: String,
        responder: oneshot::Sender<Vec<u32>>,
    },
}

use crate::config::ServerConfig;

#[derive(Clone)]
pub struct EngineHandle {
    tx: mpsc::Sender<EngineCommand>,
    has_model: bool,
    stream_buffer_capacity: usize,
}

pub const DEFAULT_STREAM_BUFFER_CAPACITY: usize = 64;
pub const ENGINE_ACTOR_THREAD_NAME: &str = "mivi-engine-actor";
pub const ENGINE_READY_MSG: &str = "Mivi-v4 inference engine ready.";
pub const MOCK_COMPLETION_TOKENS: usize = 6;
pub const MOCK_STREAM_CHUNKS: &[&str] = &["Mivi-v4 ", "inference ", "ready."];
pub const ERR_ENGINE_CHANNEL_DISCONNECTED: &str = "Engine actor channel disconnected";
pub const ERR_ENGINE_DROPPED_RESPONSE: &str = "Engine actor dropped response";

impl EngineHandle {
    pub fn new(tx: mpsc::Sender<EngineCommand>, has_model: bool) -> Self {
        Self {
            tx,
            has_model,
            stream_buffer_capacity: DEFAULT_STREAM_BUFFER_CAPACITY,
        }
    }

    pub fn with_capacity(
        tx: mpsc::Sender<EngineCommand>,
        has_model: bool,
        stream_buffer_capacity: usize,
    ) -> Self {
        Self {
            tx,
            has_model,
            stream_buffer_capacity,
        }
    }

    #[inline]
    pub fn has_model(&self) -> bool {
        self.has_model
    }

    #[inline]
    pub fn is_closed(&self) -> bool {
        self.tx.is_closed()
    }

    /// Submit a non-streaming completion job to the engine actor.
    pub async fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<(String, usize, usize), String> {
        let (responder, rx) = oneshot::channel();
        self.tx
            .send(EngineCommand::Generate {
                prompt: prompt.to_string(),
                max_tokens,
                responder,
            })
            .await
            .map_err(|_| ERR_ENGINE_CHANNEL_DISCONNECTED.to_string())?;

        rx.await
            .map_err(|_| ERR_ENGINE_DROPPED_RESPONSE.to_string())?
    }

    /// Submit a streaming generation job to the engine actor.
    pub async fn generate_stream(
        &self,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<mpsc::Receiver<Result<String, String>>, String> {
        let (responder, rx) = mpsc::channel(self.stream_buffer_capacity);
        self.tx
            .send(EngineCommand::GenerateStream {
                prompt: prompt.to_string(),
                max_tokens,
                responder,
            })
            .await
            .map_err(|_| ERR_ENGINE_CHANNEL_DISCONNECTED.to_string())?;

        Ok(rx)
    }

    /// Encode prompt tokens.
    pub async fn encode(&self, text: &str) -> Vec<u32> {
        let (responder, rx) = oneshot::channel();
        if self
            .tx
            .send(EngineCommand::Encode {
                text: text.to_string(),
                responder,
            })
            .await
            .is_ok()
        {
            rx.await.unwrap_or_else(|_| {
                tracing::warn!("Engine actor encode responder channel dropped");
                Vec::new()
            })
        } else {
            tracing::warn!("Failed to send encode command to engine actor");
            Vec::new()
        }
    }
}

pub struct EngineActor;

impl EngineActor {
    /// Spawn the engine actor on a dedicated OS compute thread with default config.
    #[track_caller]
    pub fn spawn(model: Option<mivi_model::Model>) -> EngineHandle {
        Self::spawn_with_config(model, &ServerConfig::default())
    }

    /// Try spawning the engine actor with a custom ServerConfig, returning an IO error on thread creation failure.
    pub fn try_spawn_with_config(
        mut model: Option<mivi_model::Model>,
        config: &ServerConfig,
    ) -> std::io::Result<EngineHandle> {
        let has_model = model.is_some();
        let stream_buffer_capacity = config.channel_capacity;
        let (tx, mut rx) = mpsc::channel::<EngineCommand>(config.channel_capacity);

        std::thread::Builder::new()
            .name(ENGINE_ACTOR_THREAD_NAME.to_string())
            .spawn(move || {
                let rt = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(e) => {
                        tracing::error!("Failed to create engine actor runtime: {}", e);
                        drop(rx);
                        return;
                    }
                };

                rt.block_on(async move {
                    while let Some(cmd) = rx.recv().await {
                        match cmd {
                            EngineCommand::Generate {
                                prompt,
                                max_tokens,
                                responder,
                            } => {
                                handle_generate(&mut model, prompt, max_tokens, responder);
                            }
                            EngineCommand::GenerateStream {
                                prompt,
                                max_tokens,
                                responder,
                            } => {
                                handle_generate_stream(&mut model, prompt, max_tokens, responder);
                            }
                            EngineCommand::Encode { text, responder } => {
                                handle_encode(&model, text, responder);
                            }
                        }
                    }
                });
            })?;

        Ok(EngineHandle::with_capacity(
            tx,
            has_model,
            stream_buffer_capacity,
        ))
    }

    /// Spawn the engine actor with a custom ServerConfig. Panics on OS thread spawn failure.
    #[track_caller]
    pub fn spawn_with_config(
        model: Option<mivi_model::Model>,
        config: &ServerConfig,
    ) -> EngineHandle {
        Self::try_spawn_with_config(model, config).expect("Failed to spawn engine actor")
    }
}

fn handle_generate(
    model: &mut Option<mivi_model::Model>,
    prompt: String,
    max_tokens: usize,
    responder: oneshot::Sender<std::result::Result<(String, usize, usize), String>>,
) {
    if let Some(ref mut m) = model {
        let p_tok = m.tokenizer.encode(&prompt).len();
        match m.generate(&prompt, max_tokens) {
            Ok(out) => {
                let c_tok = m.tokenizer.encode(&out).len();
                let _ = responder.send(Ok((out, p_tok, c_tok)));
            }
            Err(e) => {
                let _ = responder.send(Err(e.to_string()));
            }
        }
    } else {
        let p_tok = prompt.split_whitespace().count().max(1);
        let _ = responder.send(Ok((
            ENGINE_READY_MSG.to_string(),
            p_tok,
            MOCK_COMPLETION_TOKENS,
        )));
    }
}

fn handle_generate_stream(
    model: &mut Option<mivi_model::Model>,
    prompt: String,
    max_tokens: usize,
    responder: mpsc::Sender<std::result::Result<String, String>>,
) {
    if let Some(ref mut m) = model {
        let _ = m.generate_streaming(&prompt, max_tokens, |_, text| {
            responder.try_send(Ok(text.to_string())).is_ok()
        });
    } else {
        for &chunk in MOCK_STREAM_CHUNKS {
            if responder.try_send(Ok(chunk.to_string())).is_err() {
                break;
            }
        }
    }
}

fn handle_encode(
    model: &Option<mivi_model::Model>,
    text: String,
    responder: oneshot::Sender<Vec<u32>>,
) {
    if let Some(ref m) = model {
        let _ = responder.send(m.tokenizer.encode(&text));
    } else {
        let count = text.split_whitespace().count();
        let _ = responder.send(vec![0; count]);
    }
}
