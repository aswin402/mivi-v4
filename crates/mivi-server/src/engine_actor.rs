//! Dedicated Engine Actor managing model inference on an isolated worker thread.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
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
        cancellation: GenerationCancellation,
    },
    Encode {
        text: String,
        responder: oneshot::Sender<Vec<u32>>,
    },
}

/// Cooperative cancellation handle for one streaming generation request.
#[derive(Debug, Clone)]
pub struct GenerationCancellation {
    cancelled: Arc<AtomicBool>,
}

impl GenerationCancellation {
    fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Request that the engine stop at its next safe checkpoint.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Return whether cancellation has been requested.
    #[inline]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

use crate::config::ServerConfig;

#[derive(Clone)]
pub struct EngineHandle {
    tx: mpsc::Sender<EngineCommand>,
    has_model: bool,
    stream_buffer_capacity: usize,
    model_metadata: Option<EngineModelMetadata>,
}

/// Metadata exposed by compatibility APIs for a loaded GGUF model.
#[derive(Debug, Clone, Default)]
pub struct EngineModelMetadata {
    pub size_bytes: u64,
    pub parameter_count: Option<u64>,
    pub family: Option<String>,
    pub quantization_level: Option<String>,
    pub chat_template: Option<String>,
    pub bos_token: Option<String>,
    pub tool_call_start_token: Option<String>,
    pub tool_call_end_token: Option<String>,
    pub context_length: Option<usize>,
}

impl EngineModelMetadata {
    fn from_model(model: &mivi_model::Model) -> Self {
        let parameter_count = model
            .gguf
            .tensors
            .values()
            .filter_map(|tensor| tensor_element_count(&tensor.dims))
            .try_fold(0_u64, |total, count| total.checked_add(count));

        let mut quantized_elements = Vec::<(String, u64)>::new();
        for tensor in model.gguf.tensors.values() {
            let Some(element_count) = tensor_element_count(&tensor.dims) else {
                continue;
            };
            let label = format!("{:?}", tensor.ggml_type);
            if let Some((_, total)) = quantized_elements
                .iter_mut()
                .find(|(existing, _)| existing == &label)
            {
                if let Some(updated) = total.checked_add(element_count) {
                    *total = updated;
                }
            } else {
                quantized_elements.push((label, element_count));
            }
        }

        let family = model
            .gguf
            .metadata
            .get("general.architecture")
            .and_then(mivi_model::GgufValue::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .or_else(|| {
                let has_ssm = model
                    .config
                    .block_types
                    .contains(&mivi_model::BlockType::SSM);
                let has_attention = model
                    .config
                    .block_types
                    .contains(&mivi_model::BlockType::Attention);
                match (has_ssm, has_attention) {
                    (true, true) => Some("hybrid".to_string()),
                    (true, false) => Some("ssm".to_string()),
                    (false, true) => Some("attention".to_string()),
                    (false, false) => None,
                }
            });

        let quantization_level = quantized_elements
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(label, _)| label);

        let chat_template = model
            .gguf
            .metadata
            .get("tokenizer.chat_template")
            .and_then(mivi_model::GgufValue::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_owned);

        let bos_token = model
            .gguf
            .metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|value| value.as_usize())
            .and_then(|id| u32::try_from(id).ok())
            .and_then(|id| model.tokenizer.vocab().get_token(id))
            .filter(|value| !value.is_empty())
            .map(str::to_owned);

        let tool_call_start_token = find_tool_delimiter_token(model.tokenizer.vocab(), false);
        let tool_call_end_token = find_tool_delimiter_token(model.tokenizer.vocab(), true);

        Self {
            size_bytes: model.gguf.mmap.len() as u64,
            parameter_count,
            family,
            quantization_level,
            chat_template,
            bos_token,
            tool_call_start_token,
            tool_call_end_token,
            context_length: Some(model.config.max_seq_len),
        }
    }
}

fn find_tool_delimiter_token(vocab: &mivi_tokenizer::Vocab, end: bool) -> Option<String> {
    let marker = if end { "end" } else { "start" };
    (0..vocab.len() as u32).find_map(|id| {
        let token = vocab.get_token(id)?;
        let normalized = token.to_ascii_lowercase();
        if token.starts_with("<|")
            && token.ends_with("|>")
            && normalized.contains("tool")
            && normalized.contains("call")
            && normalized.contains(marker)
        {
            Some(token.to_string())
        } else {
            None
        }
    })
}

fn tensor_element_count(dims: &[usize]) -> Option<u64> {
    dims.iter().try_fold(1_u64, |total, dimension| {
        total.checked_mul(u64::try_from(*dimension).ok()?)
    })
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
        Self::with_capacity_and_metadata(tx, has_model, DEFAULT_STREAM_BUFFER_CAPACITY, None)
    }

    pub fn with_capacity(
        tx: mpsc::Sender<EngineCommand>,
        has_model: bool,
        stream_buffer_capacity: usize,
    ) -> Self {
        Self::with_capacity_and_metadata(tx, has_model, stream_buffer_capacity, None)
    }

    /// Construct a handle for an external/custom engine with optional model metadata.
    pub fn with_capacity_and_metadata(
        tx: mpsc::Sender<EngineCommand>,
        has_model: bool,
        stream_buffer_capacity: usize,
        model_metadata: Option<EngineModelMetadata>,
    ) -> Self {
        Self {
            tx,
            has_model,
            stream_buffer_capacity: stream_buffer_capacity.max(1),
            model_metadata,
        }
    }

    #[inline]
    pub fn has_model(&self) -> bool {
        self.has_model
    }

    #[inline]
    pub fn model_metadata(&self) -> Option<&EngineModelMetadata> {
        self.model_metadata.as_ref()
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
        self.generate_with_options(prompt, max_tokens, options)
            .await
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
        self.generate_stream_with_options_cancelable(prompt, max_tokens, options)
            .await
            .map(|(rx, _cancellation)| rx)
    }

    /// Submit a streaming generation job and return a handle that can stop it cooperatively.
    pub async fn generate_stream_with_options_cancelable(
        &self,
        prompt: &str,
        max_tokens: usize,
        options: GenerationOptions,
    ) -> Result<
        (
            mpsc::Receiver<Result<String, String>>,
            GenerationCancellation,
        ),
        String,
    > {
        let (responder, rx) = mpsc::channel(self.stream_buffer_capacity);
        let cancellation = GenerationCancellation::new();
        self.tx
            .send(EngineCommand::GenerateStream {
                prompt: prompt.to_string(),
                max_tokens,
                options,
                responder,
                cancellation: cancellation.clone(),
            })
            .await
            .map_err(|_| ERR_ENGINE_CHANNEL_DISCONNECTED.to_string())?;

        Ok((rx, cancellation))
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
        config
            .prefill_strategy
            .validate()
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidInput, error))?;
        if let Some(model) = model.as_mut() {
            model
                .set_prefill_strategy(config.prefill_strategy)
                .map_err(|error| {
                    std::io::Error::new(std::io::ErrorKind::InvalidInput, error.to_string())
                })?;
        }
        let channel_capacity = config.channel_capacity.max(1);
        let (tx, rx) = mpsc::channel(channel_capacity);
        let has_model = model.is_some() || mock_mode;
        let stream_buffer_capacity = channel_capacity;
        let model_metadata = model.as_ref().map(EngineModelMetadata::from_model);

        std::thread::Builder::new()
            .name(ENGINE_ACTOR_THREAD_NAME.to_string())
            .spawn(move || {
                let mut rx = rx;
                while let Some(cmd) = rx.blocking_recv() {
                    let res =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match cmd {
                            EngineCommand::Generate {
                                prompt,
                                max_tokens,
                                options,
                                responder,
                            } => {
                                handle_generate(
                                    &mut model, mock_mode, prompt, max_tokens, options, responder,
                                );
                            }
                            EngineCommand::GenerateStream {
                                prompt,
                                max_tokens,
                                options,
                                responder,
                                cancellation,
                            } => {
                                handle_generate_stream(
                                    &mut model,
                                    mock_mode,
                                    prompt,
                                    max_tokens,
                                    options,
                                    responder,
                                    cancellation,
                                );
                            }
                            EngineCommand::Encode { text, responder } => {
                                handle_encode(&model, text, responder);
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

        Ok(EngineHandle::with_capacity_and_metadata(
            tx,
            has_model,
            stream_buffer_capacity,
            model_metadata,
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
        let checkpoint = SamplingCheckpoint::capture(m, options.seed);
        apply_generation_options(m, &options);

        let p_tok = m.tokenizer.encode(&prompt).len();
        let forced_prefix = options
            .forced_output_prefix
            .as_deref()
            .filter(|prefix| !prefix.is_empty());
        let model_prompt = prompt_with_forced_prefix(&prompt, forced_prefix);
        let res = match options.response_mode {
            ResponseMode::Text => m.generate(&model_prompt, max_tokens),
            ResponseMode::JsonObject => m.generate_with_json_grammar(&model_prompt, max_tokens),
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
                let out = prepend_forced_prefix(out, forced_prefix);
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
        let _ = responder.send(Ok((output, p_tok, MOCK_COMPLETION_TOKENS)));
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
    cancellation: GenerationCancellation,
) {
    if let Some(ref mut m) = model {
        let checkpoint = SamplingCheckpoint::capture(m, options.seed);
        apply_generation_options(m, &options);

        let forced_prefix = options
            .forced_output_prefix
            .as_deref()
            .filter(|prefix| !prefix.is_empty());
        let model_prompt = prompt_with_forced_prefix(&prompt, forced_prefix);
        let callback = match model_stream_callback(&responder, &cancellation, forced_prefix) {
            Ok(callback) => callback,
            Err(()) => {
                checkpoint.restore(m);
                return;
            }
        };
        let res = if options.response_mode == ResponseMode::JsonObject {
            Err("json_object response format is not supported for streaming".to_string())
        } else {
            m.generate_streaming_with_cancel(&model_prompt, max_tokens, callback, || {
                cancellation.is_cancelled()
            })
            .map(|_| ())
            .map_err(|e| e.to_string())
        };

        if let Err(e) = res {
            if !cancellation.is_cancelled() {
                let _ = responder.blocking_send(Err(e));
            }
        }

        checkpoint.restore(m);
    } else if mock_mode {
        for &chunk in MOCK_STREAM_CHUNKS {
            if cancellation.is_cancelled()
                || responder.blocking_send(Ok(chunk.to_string())).is_err()
            {
                break;
            }
        }
    } else if !cancellation.is_cancelled() {
        let _ = responder.blocking_send(Err(ERR_NO_MODEL.to_string()));
    }
}

fn model_stream_callback<'a>(
    responder: &'a mpsc::Sender<Result<String, String>>,
    cancellation: &'a GenerationCancellation,
    prefix: Option<&'a str>,
) -> Result<impl FnMut(u32, &str) -> bool + 'a, ()> {
    if cancellation.is_cancelled() || responder.is_closed() {
        return Err(());
    }
    let mut pending_prefix = prefix.filter(|prefix| !prefix.is_empty());
    Ok(move |_: u32, text: &str| {
        if cancellation.is_cancelled() || responder.is_closed() {
            return false;
        }
        if text.is_empty() {
            return true;
        }
        let output = match pending_prefix.take() {
            Some(prefix) => {
                let mut output = String::with_capacity(prefix.len() + text.len());
                output.push_str(prefix);
                output.push_str(text);
                output
            }
            None => text.to_string(),
        };
        !cancellation.is_cancelled()
            && responder.blocking_send(Ok(output)).is_ok()
            && !cancellation.is_cancelled()
    })
}

fn prompt_with_forced_prefix(prompt: &str, prefix: Option<&str>) -> String {
    match prefix {
        Some(prefix) => {
            let mut model_prompt = String::with_capacity(prompt.len() + prefix.len());
            model_prompt.push_str(prompt);
            model_prompt.push_str(prefix);
            model_prompt
        }
        None => prompt.to_string(),
    }
}

fn prepend_forced_prefix(mut output: String, prefix: Option<&str>) -> String {
    let Some(prefix) = prefix else {
        return output;
    };
    if output.starts_with(prefix) {
        return output;
    }
    let mut combined = String::with_capacity(prefix.len() + output.len());
    combined.push_str(prefix);
    combined.push_str(&output);
    output.clear();
    combined
}

struct SamplingCheckpoint {
    config: mivi_model::GenerationConfig,
    rng_state: u64,
    restore_rng: bool,
}

impl SamplingCheckpoint {
    fn capture(model: &mivi_model::Model, request_seed: Option<u64>) -> Self {
        Self {
            config: model.sampler.config.clone(),
            rng_state: model.sampler.rng_state(),
            restore_rng: Self::should_restore_rng(request_seed),
        }
    }

    fn should_restore_rng(request_seed: Option<u64>) -> bool {
        request_seed.is_some()
    }

    fn restore(self, model: &mut mivi_model::Model) {
        model.sampler.config = self.config;
        if self.restore_rng {
            model.sampler.restore_rng_state(self.rng_state);
        }
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
    use std::sync::{
        mpsc as std_mpsc,
        mpsc::{Receiver as StdReceiver, SyncSender as StdSyncSender},
    };
    use std::thread::JoinHandle;
    use std::time::Duration;

    const CALLBACK_WORKER_TIMEOUT: Duration = Duration::from_secs(2);

    struct CallbackWorker {
        receiver: mpsc::Receiver<Result<String, String>>,
        join_handle: Option<JoinHandle<()>>,
        finished: StdReceiver<()>,
        release: Option<StdSyncSender<()>>,
    }

    impl CallbackWorker {
        fn join(&mut self) {
            assert!(self.finished.recv_timeout(CALLBACK_WORKER_TIMEOUT).is_ok());
            assert!(self
                .join_handle
                .take()
                .expect("worker handle is present")
                .join()
                .is_ok());
        }
    }

    impl Drop for CallbackWorker {
        fn drop(&mut self) {
            if let Some(release) = self.release.take() {
                let _ = release.send(());
            }
            self.receiver.close();
            if let Some(join_handle) = self.join_handle.take() {
                if self.finished.recv_timeout(CALLBACK_WORKER_TIMEOUT).is_ok() {
                    let _ = join_handle.join();
                }
            }
        }
    }

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

    #[test]
    fn sampling_checkpoint_only_restores_rng_for_seeded_requests() {
        assert!(!SamplingCheckpoint::should_restore_rng(None));
        assert!(SamplingCheckpoint::should_restore_rng(Some(7)));
    }

    #[test]
    fn generation_cancellation_is_shared_across_handles() {
        let first = GenerationCancellation::new();
        let second = first.clone();

        assert!(!second.is_cancelled());
        first.cancel();
        assert!(second.is_cancelled());
    }

    #[test]
    fn deferred_prefix_waits_for_nonempty_model_output() {
        let (sender, mut receiver) = mpsc::channel(4);
        let cancellation = GenerationCancellation::new();
        let mut callback = model_stream_callback(&sender, &cancellation, Some("prefix:")).unwrap();

        assert!(matches!(
            receiver.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
        assert!(callback(0, ""));
        assert!(matches!(
            receiver.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
        assert!(callback(0, "first"));
        assert_eq!(receiver.try_recv().unwrap().unwrap(), "prefix:first");
        assert!(callback(0, "second"));
        assert_eq!(receiver.try_recv().unwrap().unwrap(), "second");
    }

    #[test]
    fn absent_and_empty_prefixes_preserve_nonempty_model_chunks() {
        for prefix in [None, Some("")] {
            let (sender, mut receiver) = mpsc::channel(4);
            let cancellation = GenerationCancellation::new();
            let mut callback = model_stream_callback(&sender, &cancellation, prefix).unwrap();

            assert!(callback(0, ""));
            assert!(matches!(
                receiver.try_recv(),
                Err(mpsc::error::TryRecvError::Empty)
            ));
            assert!(callback(0, "model"));
            assert_eq!(receiver.try_recv().unwrap().unwrap(), "model");
            drop(callback);
            drop(sender);
            assert_eq!(receiver.blocking_recv(), None);
        }
    }

    #[test]
    fn whitespace_model_output_receives_the_pending_prefix() {
        let (sender, mut receiver) = mpsc::channel(1);
        let cancellation = GenerationCancellation::new();
        let mut callback = model_stream_callback(&sender, &cancellation, Some("prefix:")).unwrap();

        assert!(callback(0, " "));
        assert_eq!(receiver.try_recv().unwrap().unwrap(), "prefix: ");
    }

    #[test]
    fn dropping_callback_before_output_closes_without_a_prefix() {
        let (sender, mut receiver) = mpsc::channel(1);
        let cancellation = GenerationCancellation::new();
        let callback = model_stream_callback(&sender, &cancellation, Some("prefix:")).unwrap();

        drop(callback);
        drop(sender);
        assert_eq!(receiver.blocking_recv(), None);
    }

    #[test]
    fn cancellation_and_closed_receiver_stop_callback_delivery() {
        let (sender, mut receiver) = mpsc::channel(2);
        let cancellation = GenerationCancellation::new();
        let mut callback = model_stream_callback(&sender, &cancellation, Some("prefix:")).unwrap();
        cancellation.cancel();
        assert!(!callback(0, ""));
        assert!(!callback(0, "first"));
        assert!(matches!(
            receiver.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));

        let (sender, receiver) = mpsc::channel(1);
        drop(receiver);
        assert!(
            model_stream_callback(&sender, &GenerationCancellation::new(), Some("prefix:"))
                .is_err()
        );

        let (sender, mut receiver) = mpsc::channel(1);
        let cancellation = GenerationCancellation::new();
        let mut callback = model_stream_callback(&sender, &cancellation, Some("prefix:")).unwrap();
        receiver.close();
        assert!(!callback(0, ""));
        assert!(!callback(0, "first"));
        assert_eq!(
            receiver.try_recv(),
            Err(mpsc::error::TryRecvError::Disconnected)
        );

        let (sender, _receiver) = mpsc::channel(1);
        let cancellation = GenerationCancellation::new();
        cancellation.cancel();
        assert!(model_stream_callback(&sender, &cancellation, Some("prefix:")).is_err());
    }

    #[test]
    fn generation_error_before_output_is_received_without_a_prefix() {
        let (sender, mut receiver) = mpsc::channel(2);
        let cancellation = GenerationCancellation::new();
        let callback = model_stream_callback(&sender, &cancellation, Some("prefix:")).unwrap();

        drop(callback);
        sender
            .blocking_send(Err("generation failed".to_string()))
            .unwrap();
        drop(sender);
        assert_eq!(
            receiver.blocking_recv(),
            Some(Err("generation failed".to_string()))
        );
        assert_eq!(receiver.blocking_recv(), None);
    }

    #[test]
    fn capacity_one_applies_backpressure_and_preserves_chunk_order() {
        let (sender, receiver) = mpsc::channel(1);
        let cancellation = GenerationCancellation::new();
        let (first_sent_tx, first_sent_rx) = std_mpsc::sync_channel(1);
        let (second_started_tx, second_started_rx) = std_mpsc::sync_channel(1);
        let (second_result_tx, second_result_rx) = std_mpsc::sync_channel(1);
        let (finished_tx, finished_rx) = std_mpsc::channel();
        let join_handle = std::thread::spawn(move || {
            let mut callback = model_stream_callback(&sender, &cancellation, Some("prefix:"))
                .expect("callback construction succeeds");
            let first_result = callback(0, "first");
            let _ = first_sent_tx.send(first_result);
            let _ = second_started_tx.send(());
            let second_result = callback(0, "second");
            let _ = second_result_tx.send(second_result);
            let _ = finished_tx.send(());
        });
        let mut worker = CallbackWorker {
            receiver,
            join_handle: Some(join_handle),
            finished: finished_rx,
            release: None,
        };

        assert_eq!(
            first_sent_rx.recv_timeout(CALLBACK_WORKER_TIMEOUT),
            Ok(true)
        );
        assert_eq!(
            second_started_rx.recv_timeout(CALLBACK_WORKER_TIMEOUT),
            Ok(())
        );
        assert_eq!(
            worker.receiver.blocking_recv(),
            Some(Ok("prefix:first".to_string()))
        );
        assert_eq!(
            second_result_rx.recv_timeout(CALLBACK_WORKER_TIMEOUT),
            Ok(true)
        );
        assert_eq!(
            worker.receiver.blocking_recv(),
            Some(Ok("second".to_string()))
        );
        worker.join();
    }

    #[test]
    fn cancellation_after_blocked_send_stops_following_callbacks() {
        let (sender, receiver) = mpsc::channel(1);
        let cancellation = GenerationCancellation::new();
        let (first_sent_tx, first_sent_rx) = std_mpsc::sync_channel(1);
        let (second_started_tx, second_started_rx) = std_mpsc::sync_channel(1);
        let (results_tx, results_rx) = std_mpsc::sync_channel(1);
        let (finished_tx, finished_rx) = std_mpsc::channel();
        let worker_cancellation = cancellation.clone();
        let join_handle = std::thread::spawn(move || {
            let mut callback =
                model_stream_callback(&sender, &worker_cancellation, Some("prefix:"))
                    .expect("callback construction succeeds");
            let first_result = callback(0, "first");
            let _ = first_sent_tx.send(first_result);
            let _ = second_started_tx.send(());
            let blocked_result = callback(0, "second");
            let after_cancel_result = callback(0, "third");
            let _ = results_tx.send((blocked_result, after_cancel_result));
            let _ = finished_tx.send(());
        });
        let mut worker = CallbackWorker {
            receiver,
            join_handle: Some(join_handle),
            finished: finished_rx,
            release: None,
        };

        assert_eq!(
            first_sent_rx.recv_timeout(CALLBACK_WORKER_TIMEOUT),
            Ok(true)
        );
        assert_eq!(
            second_started_rx.recv_timeout(CALLBACK_WORKER_TIMEOUT),
            Ok(())
        );
        cancellation.cancel();
        assert_eq!(
            worker.receiver.blocking_recv(),
            Some(Ok("prefix:first".to_string()))
        );
        assert_eq!(
            results_rx.recv_timeout(CALLBACK_WORKER_TIMEOUT),
            Ok((false, false))
        );
        assert_eq!(
            worker.receiver.blocking_recv(),
            Some(Ok("second".to_string()))
        );
        assert!(matches!(
            worker.receiver.try_recv(),
            Err(mpsc::error::TryRecvError::Disconnected)
        ));
        worker.join();
    }

    #[tokio::test]
    async fn idle_callback_deadline_expires_until_nonempty_output_is_released() {
        let (sender, receiver) = mpsc::channel(2);
        let cancellation = GenerationCancellation::new();
        let (ready_tx, ready_rx) = oneshot::channel();
        let (release_tx, release_rx) = std_mpsc::sync_channel(1);
        let (finished_tx, finished_rx) = std_mpsc::channel();
        let join_handle = std::thread::spawn(move || {
            let mut callback = model_stream_callback(&sender, &cancellation, Some("prefix:"))
                .expect("callback construction succeeds");
            let _ = ready_tx.send(());
            if release_rx.recv_timeout(CALLBACK_WORKER_TIMEOUT).is_ok() {
                let _ = callback(0, "first");
            }
            let _ = finished_tx.send(());
        });
        let mut worker = CallbackWorker {
            receiver,
            join_handle: Some(join_handle),
            finished: finished_rx,
            release: Some(release_tx),
        };

        tokio::time::timeout(CALLBACK_WORKER_TIMEOUT, ready_rx)
            .await
            .expect("callback worker becomes ready before watchdog")
            .expect("worker sends readiness");
        assert!(
            tokio::time::timeout(Duration::from_millis(50), worker.receiver.recv())
                .await
                .is_err()
        );

        worker.release.take().unwrap().send(()).unwrap();
        assert_eq!(
            tokio::time::timeout(CALLBACK_WORKER_TIMEOUT, worker.receiver.recv())
                .await
                .expect("model output arrives before watchdog"),
            Some(Ok("prefix:first".to_string()))
        );
        worker.join();
    }

    #[test]
    fn tensor_element_count_is_checked() {
        assert_eq!(tensor_element_count(&[]), Some(1));
        assert_eq!(tensor_element_count(&[2, 3, 4]), Some(24));
        assert_eq!(tensor_element_count(&[usize::MAX, 2]), None);
    }

    #[test]
    fn zero_channel_capacity_is_clamped_before_channel_creation() {
        let mut config = ServerConfig::default();
        config.channel_capacity = 0;

        let engine = EngineActor::spawn_with_config(None, &config);
        assert_eq!(engine.stream_buffer_capacity, 1);
    }

    #[test]
    fn invalid_prefill_strategy_is_rejected_before_spawn() {
        let mut config = ServerConfig::default();
        config.prefill_strategy = mivi_model::PrefillStrategy::Chunked { tile_tokens: 0 };

        assert!(EngineActor::try_spawn_with_config(None, &config).is_err());
    }
}
