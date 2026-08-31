//! Modern, minimal interactive chat REPL runner with incremental KV caching, thinking stream, telemetry & slash commands.

use crate::runners::chat_stream::{print_telemetry_footer, StreamFilter};
use anyhow::Result;
use mivi_model::Model;
use std::io::{self, Write};
use std::path::PathBuf;

const DEFAULT_CHAT_MAX_TOKENS: usize = 512;
const DIVIDER_LINE: &str =
    "  \x1b[2m─────────────────────────────────────────────────────────────────\x1b[0m";

#[derive(Debug, Clone)]
pub struct ChatArgs {
    pub model: PathBuf,
    pub temp: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub rep_penalty: f32,
    pub seed: Option<u64>,
    pub max_tokens: usize,
    pub ctx_size: Option<usize>,
    pub system: Option<String>,
    pub thinking: bool,
}

pub fn run_chat(args: ChatArgs) -> Result<()> {
    if !args.model.exists() {
        anyhow::bail!(
            "Model file not found at {:?}.\n\nTo test with the test model fixture, run:\n  just chat models/mivi-tiny-test.gguf\nOr generate the test fixture using:\n  python3 training/export/generate_fixture.py",
            args.model
        );
    }

    print!("  \x1b[2mLoading model from {:?}...\x1b[0m\r", args.model);
    let _ = io::stdout().flush();

    let mut m = Model::load_with_ctx(&args.model, args.ctx_size)?;
    m.sampler.config.temperature = args.temp;
    m.sampler.config.top_p = args.top_p;
    m.sampler.config.top_k = args.top_k;
    m.sampler.config.min_p = args.min_p;
    m.sampler.config.repetition_penalty = args.rep_penalty;
    m.sampler.config.seed = args.seed;
    if let Some(s) = args.seed {
        m.sampler.set_seed(s);
    }

    let max_tokens = if args.max_tokens > 0 {
        args.max_tokens
    } else {
        DEFAULT_CHAT_MAX_TOKENS
    };

    let mut thinking_enabled = args.thinking;
    let mut system_prompt = Some(
        args.system
            .unwrap_or_else(|| mivi_core::DEFAULT_SYSTEM_PROMPT.to_string()),
    );

    print_chat_header(
        &m.config.name,
        m.config.max_seq_len,
        mivi_core::estimate_process_memory_mb(),
    );

    let mut conversation_history: Vec<mivi_tokenizer::ChatMessage> = Vec::new();
    if let Some(ref sys_prompt) = system_prompt {
        conversation_history.push(mivi_tokenizer::ChatMessage {
            role: mivi_tokenizer::Role::System,
            content: Some(sys_prompt.clone()),
            name: None,
        });
    }

    // Tracks tokens that are currently in the model's KV cache
    let mut cached_tokens: Vec<u32> = Vec::new();

    loop {
        print!("  \x1b[1;36muser\x1b[0m \x1b[36m›\x1b[0m ");
        io::stdout().flush()?;

        let mut input = String::new();
        let bytes_read = io::stdin().read_line(&mut input)?;
        if bytes_read == 0 {
            println!("\n  \x1b[2mGoodbye!\x1b[0m\n");
            break;
        }
        let trimmed = input.trim();

        if trimmed.is_empty() {
            continue;
        }

        // Handle slash commands
        if trimmed.starts_with('/') {
            let handled = handle_slash_command(
                trimmed,
                &mut m,
                &mut conversation_history,
                &mut thinking_enabled,
                &mut system_prompt,
                &mut cached_tokens,
            );
            if handled {
                continue;
            }
            if trimmed.eq_ignore_ascii_case("/exit") || trimmed.eq_ignore_ascii_case("/quit") {
                println!("  \x1b[2mGoodbye!\x1b[0m\n");
                break;
            }
        }

        if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
            println!("  \x1b[2mGoodbye!\x1b[0m\n");
            break;
        }

        conversation_history.push(mivi_tokenizer::ChatMessage {
            role: mivi_tokenizer::Role::User,
            content: Some(trimmed.to_string()),
            name: None,
        });

        // 1. Extract active conversational context (System + recent turns) for model prompting
        const MAX_ACTIVE_HISTORY_TURNS: usize = 2;
        let active_context =
            get_active_chat_context(&conversation_history, MAX_ACTIVE_HISTORY_TURNS);

        // 2. Format ChatML prompt string
        let mut full_prompt =
            mivi_tokenizer::format_chatml(&active_context, None, thinking_enabled);
        let mut prompt_tokens = m.tokenizer.encode(&full_prompt);

        // 3. Sliding window context eviction if approaching max_seq_len
        let max_ctx = m.config.max_seq_len;
        if prompt_tokens.len() + max_tokens >= max_ctx {
            // Trim oldest non-system message pair (or message)
            let mut trimmed_any = false;
            while prompt_tokens.len() + max_tokens >= (max_ctx * 85) / 100 {
                let first_non_system = conversation_history
                    .iter()
                    .position(|msg| msg.role != mivi_tokenizer::Role::System);
                if let Some(idx) = first_non_system {
                    if idx < conversation_history.len().saturating_sub(1) {
                        conversation_history.remove(idx);
                        trimmed_any = true;
                        let active = get_active_chat_context(
                            &conversation_history,
                            MAX_ACTIVE_HISTORY_TURNS,
                        );
                        full_prompt =
                            mivi_tokenizer::format_chatml(&active, None, thinking_enabled);
                        prompt_tokens = m.tokenizer.encode(&full_prompt);
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            if trimmed_any {
                println!(
                    "  \x1b[33m⚠ Context window near capacity: trimmed older messages.\x1b[0m"
                );
                m.reset_context();
                cached_tokens.clear();
            }
        }

        // 4. Reset context and prefill active conversation prompt from position 0
        // This ensures the SSM recurrent shortconv state and Attention KV cache
        // are perfectly synchronized with the exact multi-turn ChatML prompt sequence.
        m.reset_context();
        let start_pos = 0;
        let new_prompt_tokens = &prompt_tokens[..];

        print!("  \x1b[1;32mmivi\x1b[0m \x1b[32m›\x1b[0m ");
        io::stdout().flush()?;

        let mut stream_filter = StreamFilter::new();

        let (output, generated_ids) = m.generate_tokens_incremental(
            new_prompt_tokens,
            start_pos,
            max_tokens,
            |_id, token_str| {
                stream_filter.on_token(token_str);
                true
            },
        )?;

        println!();
        let stats = stream_filter.finish();
        print_telemetry_footer(&stats);
        println!("{}", DIVIDER_LINE);

        // Update cached tokens
        cached_tokens.clear();
        cached_tokens.extend_from_slice(&prompt_tokens);
        cached_tokens.extend_from_slice(&generated_ids);

        conversation_history.push(mivi_tokenizer::ChatMessage {
            role: mivi_tokenizer::Role::Assistant,
            content: Some(output),
            name: None,
        });
    }

    Ok(())
}

fn print_chat_header(model_name: &str, max_ctx: usize, rss_mb: f32) {
    println!(
        "\n  \x1b[1;36m⚡ Mivi Chat\x1b[0m \x1b[2mv{}\x1b[0m \x1b[2m({} • {}K ctx • {:.1} MB RAM)\x1b[0m\n  \x1b[2mType your prompt, or /help for interactive commands, Ctrl+C to cancel.\x1b[0m\n{}",
        env!("CARGO_PKG_VERSION"),
        model_name,
        max_ctx / 1024,
        rss_mb,
        DIVIDER_LINE,
    );
}

fn handle_slash_command(
    cmd: &str,
    model: &mut Model,
    history: &mut Vec<mivi_tokenizer::ChatMessage>,
    thinking_enabled: &mut bool,
    system_prompt: &mut Option<String>,
    cached_tokens: &mut Vec<u32>,
) -> bool {
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    let base_cmd = parts[0].to_lowercase();

    match base_cmd.as_str() {
        "/help" | "/h" => {
            println!(
                "\n  \x1b[1mMivi Interactive Commands:\x1b[0m\n    \x1b[36m/stats\x1b[0m          Show model architecture, sampling hyperparameters & RAM\n    \x1b[36m/clear\x1b[0m          Clear conversation history and refresh screen\n    \x1b[36m/reset\x1b[0m          Reset conversation context without clearing screen\n    \x1b[36m/system <txt>\x1b[0m   View or update the active system persona prompt\n    \x1b[36m/temp <val>\x1b[0m     Adjust temperature on the fly (e.g. /temp 0.2)\n    \x1b[36m/min-p <val>\x1b[0m    Adjust Min-P threshold (e.g. /min-p 0.05)\n    \x1b[36m/top-p <val>\x1b[0m    Adjust Top-P threshold (e.g. /top-p 0.90)\n    \x1b[36m/top-k <val>\x1b[0m    Adjust Top-K sampling (e.g. /top-k 40)\n    \x1b[36m/rep <val>\x1b[0m      Adjust repetition penalty (e.g. /rep 1.10)\n    \x1b[36m/thinking\x1b[0m       Toggle thinking instructions on/off\n    \x1b[36m/help\x1b[0m           Show this command list\n    \x1b[36m/exit\x1b[0m, \x1b[36m/quit\x1b[0m   Exit chat session"
            );
            println!("{}", DIVIDER_LINE);
            true
        }
        "/stats" => {
            let turns = history
                .iter()
                .filter(|m| m.role == mivi_tokenizer::Role::User)
                .count();
            println!(
                "\n  \x1b[1mTelemetry & Runtime Stats:\x1b[0m\n    • \x1b[1mModel:\x1b[0m       {} ({} layers, {} dim)\n    • \x1b[1mContext:\x1b[0m     {} turns in history ({} tokens cached, max {})\n    • \x1b[1mSampling:\x1b[0m    temp={:.2} | top_p={:.2} | top_k={} | min_p={:.2} | rep_pen={:.2}\n    • \x1b[1mThinking:\x1b[0m    {}\n    • \x1b[1mMemory RSS:\x1b[0m  {:.1} MB",
                model.config.name,
                model.config.n_layers,
                model.config.dim,
                turns,
                cached_tokens.len(),
                model.config.max_seq_len,
                model.sampler.config.temperature,
                model.sampler.config.top_p,
                model.sampler.config.top_k,
                model.sampler.config.min_p,
                model.sampler.config.repetition_penalty,
                if *thinking_enabled {
                    "\x1b[32menabled\x1b[0m"
                } else {
                    "\x1b[2mdisabled\x1b[0m"
                },
                mivi_core::estimate_process_memory_mb()
            );
            println!("{}", DIVIDER_LINE);
            true
        }
        "/clear" => {
            print!("\x1b[2J\x1b[1;1H");
            let _ = io::stdout().flush();
            history.clear();
            cached_tokens.clear();
            model.reset_context();
            if let Some(ref sys_prompt) = system_prompt {
                history.push(mivi_tokenizer::ChatMessage {
                    role: mivi_tokenizer::Role::System,
                    content: Some(sys_prompt.clone()),
                    name: None,
                });
            }
            print_chat_header(
                &model.config.name,
                model.config.max_seq_len,
                mivi_core::estimate_process_memory_mb(),
            );
            true
        }
        "/reset" => {
            history.clear();
            cached_tokens.clear();
            model.reset_context();
            if let Some(ref sys_prompt) = system_prompt {
                history.push(mivi_tokenizer::ChatMessage {
                    role: mivi_tokenizer::Role::System,
                    content: Some(sys_prompt.clone()),
                    name: None,
                });
            }
            println!("  \x1b[2mContext history reset.\x1b[0m");
            println!("{}", DIVIDER_LINE);
            true
        }
        "/system" => {
            if parts.len() > 1 {
                let new_sys = parts[1..].join(" ");
                *system_prompt = Some(new_sys.clone());
                history.retain(|m| m.role != mivi_tokenizer::Role::System);
                history.insert(
                    0,
                    mivi_tokenizer::ChatMessage {
                        role: mivi_tokenizer::Role::System,
                        content: Some(new_sys.clone()),
                        name: None,
                    },
                );
                cached_tokens.clear();
                model.reset_context();
                println!("  \x1b[2mSystem prompt updated to:\x1b[0m \"{}\"", new_sys);
            } else if let Some(ref sys) = system_prompt {
                println!("  \x1b[2mActive system prompt:\x1b[0m \"{}\"", sys);
            } else {
                println!("  \x1b[2mNo system prompt active (default).\x1b[0m");
            }
            println!("{}", DIVIDER_LINE);
            true
        }
        "/temp" => {
            if parts.len() > 1 {
                if let Ok(val) = parts[1].parse::<f32>() {
                    model.sampler.config.temperature = val.max(0.0);
                    println!(
                        "  \x1b[2mTemperature set to:\x1b[0m {:.2}",
                        model.sampler.config.temperature
                    );
                } else {
                    println!("  \x1b[31mInvalid float value for /temp\x1b[0m");
                }
            } else {
                println!(
                    "  \x1b[2mCurrent temperature:\x1b[0m {:.2}",
                    model.sampler.config.temperature
                );
            }
            println!("{}", DIVIDER_LINE);
            true
        }
        "/min-p" => {
            if parts.len() > 1 {
                if let Ok(val) = parts[1].parse::<f32>() {
                    model.sampler.config.min_p = val.clamp(0.0, 1.0);
                    println!(
                        "  \x1b[2mMin-P threshold set to:\x1b[0m {:.2}",
                        model.sampler.config.min_p
                    );
                } else {
                    println!("  \x1b[31mInvalid float value for /min-p\x1b[0m");
                }
            } else {
                println!(
                    "  \x1b[2mCurrent Min-P:\x1b[0m {:.2}",
                    model.sampler.config.min_p
                );
            }
            println!("{}", DIVIDER_LINE);
            true
        }
        "/top-p" => {
            if parts.len() > 1 {
                if let Ok(val) = parts[1].parse::<f32>() {
                    model.sampler.config.top_p = val.clamp(0.0, 1.0);
                    println!(
                        "  \x1b[2mTop-P threshold set to:\x1b[0m {:.2}",
                        model.sampler.config.top_p
                    );
                } else {
                    println!("  \x1b[31mInvalid float value for /top-p\x1b[0m");
                }
            } else {
                println!(
                    "  \x1b[2mCurrent Top-P:\x1b[0m {:.2}",
                    model.sampler.config.top_p
                );
            }
            println!("{}", DIVIDER_LINE);
            true
        }
        "/top-k" => {
            if parts.len() > 1 {
                if let Ok(val) = parts[1].parse::<usize>() {
                    model.sampler.config.top_k = val;
                    println!(
                        "  \x1b[2mTop-K set to:\x1b[0m {}",
                        model.sampler.config.top_k
                    );
                } else {
                    println!("  \x1b[31mInvalid integer value for /top-k\x1b[0m");
                }
            } else {
                println!(
                    "  \x1b[2mCurrent Top-K:\x1b[0m {}",
                    model.sampler.config.top_k
                );
            }
            println!("{}", DIVIDER_LINE);
            true
        }
        "/rep" => {
            if parts.len() > 1 {
                if let Ok(val) = parts[1].parse::<f32>() {
                    model.sampler.config.repetition_penalty = val.max(0.0);
                    println!(
                        "  \x1b[2mRepetition penalty set to:\x1b[0m {:.2}",
                        model.sampler.config.repetition_penalty
                    );
                } else {
                    println!("  \x1b[31mInvalid float value for /rep\x1b[0m");
                }
            } else {
                println!(
                    "  \x1b[2mCurrent repetition penalty:\x1b[0m {:.2}",
                    model.sampler.config.repetition_penalty
                );
            }
            println!("{}", DIVIDER_LINE);
            true
        }
        "/thinking" => {
            *thinking_enabled = !*thinking_enabled;
            cached_tokens.clear();
            model.reset_context();
            println!(
                "  \x1b[2mThinking formatting:\x1b[0m {}",
                if *thinking_enabled {
                    "\x1b[32menabled\x1b[0m"
                } else {
                    "\x1b[2mdisabled\x1b[0m"
                }
            );
            println!("{}", DIVIDER_LINE);
            true
        }
        _ => false,
    }
}

/// Helper to extract the active conversational context (System + recent turns) for model prompting.
fn get_active_chat_context(
    history: &[mivi_tokenizer::ChatMessage],
    max_history_turns: usize,
) -> Vec<mivi_tokenizer::ChatMessage> {
    let mut out = Vec::new();
    // 1. Always retain system prompt at the top
    if let Some(sys) = history
        .iter()
        .find(|m| m.role == mivi_tokenizer::Role::System)
    {
        out.push(sys.clone());
    }
    // 2. Collect non-system messages
    let non_sys: Vec<_> = history
        .iter()
        .filter(|m| m.role != mivi_tokenizer::Role::System)
        .cloned()
        .collect();
    // 3. Keep last max_history_turns * 2 messages (e.g. 2 turns = 4 messages)
    let keep_count = (max_history_turns * 2).min(non_sys.len());
    let start_idx = non_sys.len().saturating_sub(keep_count);
    out.extend_from_slice(&non_sys[start_idx..]);
    out
}
