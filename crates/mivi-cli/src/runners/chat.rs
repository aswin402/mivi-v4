//! Interactive chat REPL runner command.

use anyhow::Result;
use mivi_model::Model;
use mivi_tools::extract_tool_calls;
use std::io::{self, Write};
use std::path::PathBuf;

const DEFAULT_CHAT_MAX_TOKENS: usize = 512;

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

    println!("Loading Mivi model from {:?}...", args.model);
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
    println!(
        "Model loaded successfully! ({}, {}K context)",
        m.config.name,
        m.config.max_seq_len / 1024
    );
    println!("Mivi Interactive REPL. Type your prompt (or 'exit' to quit):\n");

    let mut conversation_history = Vec::new();
    if let Some(sys_prompt) = args.system {
        conversation_history.push(mivi_tokenizer::ChatMessage {
            role: mivi_tokenizer::Role::System,
            content: Some(sys_prompt),
            name: None,
        });
    }

    loop {
        print!("user> ");
        io::stdout().flush()?;

        let mut input = String::new();
        let bytes_read = io::stdin().read_line(&mut input)?;
        if bytes_read == 0 {
            println!("\nGoodbye!");
            break;
        }
        let trimmed = input.trim();

        if trimmed.is_empty() {
            continue;
        }
        if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
            println!("Goodbye!");
            break;
        }

        conversation_history.push(mivi_tokenizer::ChatMessage {
            role: mivi_tokenizer::Role::User,
            content: Some(trimmed.to_string()),
            name: None,
        });

        let prompt = mivi_tokenizer::format_chatml(&conversation_history, None, args.thinking);
        print!("assistant> ");
        io::stdout().flush()?;

        let output = m.generate_streaming(&prompt, max_tokens, |_id, token_str| {
            print!("{}", token_str);
            let _ = io::stdout().flush();
            true
        })?;
        println!();

        // Display any tool calls
        let tool_calls = extract_tool_calls(&output);
        for tc in tool_calls {
            println!("\x1b[33m[tool_call: {}({})]\x1b[0m", tc.name, tc.arguments);
        }

        conversation_history.push(mivi_tokenizer::ChatMessage {
            role: mivi_tokenizer::Role::Assistant,
            content: Some(output),
            name: None,
        });
    }

    Ok(())
}
