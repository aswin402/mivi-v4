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

#[derive(Clone)]
pub struct EngineHandle {
    tx: mpsc::Sender<EngineCommand>,
    has_model: bool,
}

impl EngineHandle {
    pub fn new(tx: mpsc::Sender<EngineCommand>, has_model: bool) -> Self {
        Self { tx, has_model }
    }

    #[inline]
    pub fn has_model(&self) -> bool {
        self.has_model
    }

    /// Submit a non-streaming completion job to the engine actor.
    pub async fn generate(&self, prompt: &str, max_tokens: usize) -> Result<(String, usize, usize), String> {
        let (responder, rx) = oneshot::channel();
        self.tx
            .send(EngineCommand::Generate {
                prompt: prompt.to_string(),
                max_tokens,
                responder,
            })
            .await
            .map_err(|_| "Engine actor channel disconnected".to_string())?;

        rx.await.map_err(|_| "Engine actor dropped response".to_string())?
    }

    /// Submit a streaming generation job to the engine actor.
    pub async fn generate_stream(
        &self,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<mpsc::Receiver<Result<String, String>>, String> {
        let (responder, rx) = mpsc::channel(64);
        self.tx
            .send(EngineCommand::GenerateStream {
                prompt: prompt.to_string(),
                max_tokens,
                responder,
            })
            .await
            .map_err(|_| "Engine actor channel disconnected".to_string())?;

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
            rx.await.unwrap_or_default()
        } else {
            Vec::new()
        }
    }
}

pub struct EngineActor;

impl EngineActor {
    /// Spawn the engine actor on a dedicated OS compute thread.
    pub fn spawn(mut model: Option<mivi_model::Model>) -> EngineHandle {
        let has_model = model.is_some();
        let (tx, mut rx) = mpsc::channel::<EngineCommand>(64);

        std::thread::Builder::new()
            .name("mivi-engine-actor".to_string())
            .spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("Failed to create engine actor runtime");

                rt.block_on(async move {
                    while let Some(cmd) = rx.recv().await {
                        match cmd {
                            EngineCommand::Generate {
                                prompt,
                                max_tokens,
                                responder,
                            } => {
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
                                        "Mivi-v4 inference engine ready.".to_string(),
                                        p_tok,
                                        6,
                                    )));
                                }
                            }
                            EngineCommand::GenerateStream {
                                prompt,
                                max_tokens,
                                responder,
                            } => {
                                if let Some(ref mut m) = model {
                                    match m.generate(&prompt, max_tokens) {
                                        Ok(out) => {
                                            for word in out.split_inclusive(' ') {
                                                if responder.send(Ok(word.to_string())).await.is_err() {
                                                    break; // Client disconnected
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            let _ = responder.send(Err(e.to_string())).await;
                                        }
                                    }
                                } else {
                                    for chunk in &["Mivi-v4 ", "inference ", "ready."] {
                                        if responder.send(Ok(chunk.to_string())).await.is_err() {
                                            break;
                                        }
                                    }
                                }
                            }
                            EngineCommand::Encode { text, responder } => {
                                if let Some(ref m) = model {
                                    let _ = responder.send(m.tokenizer.encode(&text));
                                } else {
                                    let count = text.split_whitespace().count();
                                    let _ = responder.send(vec![0; count]);
                                }
                            }
                        }
                    }
                });
            })
            .expect("Failed to spawn engine actor thread");

        EngineHandle::new(tx, has_model)
    }
}
