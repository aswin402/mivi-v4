//! Dedicated Engine Actor managing model inference on an isolated worker thread.

use tokio::sync::{mpsc, oneshot};

use crate::generation::{validate_json_output, GenerationOptions, ResponseMode};

#[derive(Debug)]
pub enum EngineCommand {
    Generate {
        prompt: String,
        max_tokens: usize,
        options: GenerationOptions,
        responder: oneshot::Sender<Result<(String, usize, usize), String>>,
    },
    GenerateStream {
        prompt: String,
        max_tokens: usize,
        options: GenerationOptions,
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
pub const ERR_NO_MODEL: &str = "No model is loaded";

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

    /// Submit a non-streaming completion job to the engine actor with default sampling parameters.
    pub async fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<(String, usize, usize), String> {
        self.generate_with_params(prompt, max_tokens, None, None)
            .await
    }

    /// Submit a non-streaming completion job to the engine actor with custom sampling parameters.
    pub async fn generate_with_params(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: Option<f32>,
        top_p: Option<f32>,
    ) -> Result<(String, usize, usize), String> {
        let options = GenerationOptions {
            temperature,
            top_p,
            ..GenerationOptions::default()
        };
        self.generate_with_options(prompt, max_tokens, options).await
    }

    pub async fn generate_with_options(
        &self,
        prompt: &str,
        max_tokens: usize,
        options: GenerationOptions,
    ) -> Result<(String, usize, usize), String> {
        let (responder, rx) = oneshot::channel();
        self.tx
            .send(EngineCommand::Generate {
                prompt: prompt.to_string(),
                max_tokens,
                options,
                responder,
            })
            .await
            .map_err(|_| ERR_ENGINE_CHANNEL_DISCONNECTED.to_string())?;

        rx.await
            .map_err(|_| ERR_ENGINE_DROPPED_RESPONSE.to_string())?
    }

    /// Submit a streaming generation job to the engine actor with default sampling parameters.
    pub async fn generate_stream(
        &self,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<mpsc::Receiver<Result<String, String>>, String> {
        self.generate_stream_with_params(prompt, max_tokens, None, None)
            .await
    }

    /// Submit a streaming generation job to the engine actor with custom sampling parameters.
    pub async fn generate_stream_with_params(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: Option<f32>,
        top_p: Option<f32>,
    ) -> Result<mpsc::Receiver<Result<String, String>>, String> {
        let options = GenerationOptions {
            temperature,
            top_p,
            ..GenerationOptions::default()
        };
        self.generate_stream_with_options(prompt, max_tokens, options)
            .await
    }

    pub async fn generate_stream_with_options(
        &self,
        prompt: &str,
        max_tokens: usize,
        options: GenerationOptions,
    ) -> Result<mpsc::Receiver<Result<String, String>>, String> {
        let (responder, rx) = mpsc::channel(self.stream_buffer_capacity);
        self.tx
            .send(EngineCommand::GenerateStream {
                prompt: prompt.to_string(),
                max_tokens,
                options,
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
            rx.await.unwrap_or_default()
        } else {
            Vec::new()
        }
    }
}

pub struct EngineActor;

impl EngineActor {
    /// Spawn the engine actor on an isolated OS thread with default configuration.
    pub fn spawn(model: Option<mivi_model::Model>) -> EngineHandle {
        Self::spawn_with_config(model, &ServerConfig::default())
    }

    /// Spawn an explicit development mock engine without loading a model.
    pub fn spawn_mock() -> EngineHandle {
        Self::try_spawn_with_mode(None, &ServerConfig::default(), true)
            .expect("Failed to spawn mock engine actor")
    }

    /// Try spawning the engine actor with a custom ServerConfig. Returns Error on thread spawn failure.
    pub fn try_spawn_with_config(
        model: Option<mivi_model::Model>,
        config: &ServerConfig,
    ) -> std::io::Result<EngineHandle> {
        Self::try_spawn_with_mode(model, config, false)
    }

    fn try_spawn_with_mode(
        mut model: Option<mivi_model::Model>,
        config: &ServerConfig,
        mock_mode: bool,
    ) -> std::io::Result<EngineHandle> {
        let (tx, rx) = mpsc::channel(config.channel_capacity);
        let has_model = model.is_some() || mock_mode;
        let stream_buffer_capacity = config.channel_capacity;

        std::thread::Builder::new()
            .name(ENGINE_ACTOR_THREAD_NAME.to_string())
            .spawn(move || {
                let mut rx = rx;
                while let Some(cmd) = rx.blocking_recv() {
                    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        match cmd {
                            EngineCommand::Generate {
                                prompt,
                                max_tokens,
                                options,
                                responder,
                            } => {
                                handle_generate(&mut model, mock_mode, prompt, max_tokens, options, responder);
                            }
                            EngineCommand::GenerateStream {
                                prompt,
                                max_tokens,
                                options,
                                responder,
                            } => {
                                handle_generate_stream(&mut model, mock_mode, prompt, max_tokens, options, responder);
                            }
                            EngineCommand::Encode { text, responder } => {
                                handle_encode(&model, text, responder);
                            }
                        }
                    }));
                    if let Err(panic_err) = res {
                        tracing::error!(
                            "Engine actor caught panic during command execution: {:?}",
                            panic_err
                        );
                    }
                }
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
    mock_mode: bool,
    prompt: String,
    max_tokens: usize,
    options: GenerationOptions,
    responder: oneshot::Sender<std::result::Result<(String, usize, usize), String>>,
) {
    if let Some(ref mut m) = model {
        let checkpoint = SamplingCheckpoint::capture(m);
        apply_generation_options(m, &options);

        let p_tok = m.tokenizer.encode(&prompt).len();
        let res = match options.response_mode {
            ResponseMode::Text => m.generate(&prompt, max_tokens),
            ResponseMode::JsonObject => m.generate_with_json_grammar(&prompt, max_tokens),
        };
        let res = match res {
            Ok(out) => {
                if options.response_mode == ResponseMode::JsonObject {
                    if let Err(error) = validate_json_output(&out) {
                        checkpoint.restore(m);
                        let _ = responder.send(Err(error));
                        return;
                    }
                }
                let c_tok = m.tokenizer.encode(&out).len();
                Ok((out, p_tok, c_tok))
            }
            Err(e) => Err(e.to_string()),
        };

        checkpoint.restore(m);
        let _ = responder.send(res);
    } else if mock_mode {
        let p_tok = prompt.split_whitespace().count().max(1);
        let output = if options.response_mode == ResponseMode::JsonObject {
            "{}".to_string()
        } else {
            ENGINE_READY_MSG.to_string()
        };
        let _ = responder.send(Ok((
            output,
            p_tok,
            MOCK_COMPLETION_TOKENS,
        )));
    } else {
        let _ = responder.send(Err(ERR_NO_MODEL.to_string()));
    }
}

fn handle_generate_stream(
    model: &mut Option<mivi_model::Model>,
    mock_mode: bool,
    prompt: String,
    max_tokens: usize,
    options: GenerationOptions,
    responder: mpsc::Sender<std::result::Result<String, String>>,
) {
    if let Some(ref mut m) = model {
        let checkpoint = SamplingCheckpoint::capture(m);
        apply_generation_options(m, &options);

        let res = if options.response_mode == ResponseMode::JsonObject {
            Err("json_object response format is not supported for streaming".to_string())
        } else {
            m.generate_streaming(&prompt, max_tokens, |_, text| {
                responder.blocking_send(Ok(text.to_string())).is_ok()
            })
            .map(|_| ())
            .map_err(|e| e.to_string())
        };

        if let Err(e) = res {
            let _ = responder.blocking_send(Err(e));
        }

        checkpoint.restore(m);
    } else if mock_mode {
        for &chunk in MOCK_STREAM_CHUNKS {
            if responder.blocking_send(Ok(chunk.to_string())).is_err() {
                break;
            }
        }
    } else {
        let _ = responder.blocking_send(Err(ERR_NO_MODEL.to_string()));
    }
}

struct SamplingCheckpoint {
    config: mivi_model::GenerationConfig,
    rng_state: u64,
}

impl SamplingCheckpoint {
    fn capture(model: &mivi_model::Model) -> Self {
        Self {
            config: model.sampler.config.clone(),
            rng_state: model.sampler.rng_state(),
        }
    }

    fn restore(self, model: &mut mivi_model::Model) {
        model.sampler.config = self.config;
        model.sampler.restore_rng_state(self.rng_state);
    }
}

fn apply_generation_options(model: &mut mivi_model::Model, options: &GenerationOptions) {
    if let Some(value) = options.temperature {
        model.sampler.config.temperature = value;
    }
    if let Some(value) = options.top_p {
        model.sampler.config.top_p = value;
    }
    if let Some(value) = options.top_k {
        model.sampler.config.top_k = value;
    }
    if let Some(value) = options.min_p {
        model.sampler.config.min_p = value;
    }
    if let Some(value) = options.repetition_penalty {
        model.sampler.config.repetition_penalty = value;
    }
    if let Some(value) = options.presence_penalty {
        model.sampler.config.presence_penalty = value;
    }
    if let Some(value) = options.frequency_penalty {
        model.sampler.config.frequency_penalty = value;
    }
    if let Some(seed) = options.seed {
        model.sampler.set_seed(seed);
    }
    if let Some(stop_tokens) = &options.stop_tokens {
        for stop in stop_tokens {
            if !model.sampler.config.stop_tokens.contains(stop) {
                model.sampler.config.stop_tokens.push(stop.clone());
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_model_generation_returns_an_error() {
        let mut model = None;
        let (tx, rx) = oneshot::channel();
        handle_generate(
            &mut model,
            false,
            "hello".to_string(),
            8,
            GenerationOptions::default(),
            tx,
        );

        assert!(rx.blocking_recv().unwrap().is_err());
    }
}
