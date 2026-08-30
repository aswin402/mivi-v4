//! Interactive chat REPL runner command.

use anyhow::Result;
use mivi_model::Model;
use mivi_tools::{extract_thinking, extract_tool_calls};
use std::io::{self, Write};
use std::path::PathBuf;

const DEFAULT_CHAT_MAX_TOKENS: usize = 512;

pub fn run_chat(model: PathBuf) -> Result<()> {
    println!("Loading Mivi model from {:?}...", model);
    let mut m = Model::load(&model)?;
    println!(
        "Model loaded successfully! ({} parameters, 32K context)",
        m.config.name
    );
    println!("Mivi Interactive REPL. Type your prompt (or 'exit' to quit):\n");

    let mut conversation_history = Vec::new();

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

        let prompt = mivi_tokenizer::format_chatml(&conversation_history, None, true);
        print!("assistant> ");
        io::stdout().flush()?;

        let output = m.generate(&prompt, DEFAULT_CHAT_MAX_TOKENS)?;

        // Display thinking if present
        if let Some(think) = extract_thinking(&output) {
            println!("\x1b[90m[think: {}]\x1b[0m", think);
        }

        // Display any tool calls
        let tool_calls = extract_tool_calls(&output);
        for tc in tool_calls {
            println!("\x1b[33m[tool_call: {}({})]\x1b[0m", tc.name, tc.arguments);
        }

        println!("{}\n", output);

        conversation_history.push(mivi_tokenizer::ChatMessage {
            role: mivi_tokenizer::Role::Assistant,
            content: Some(output),
            name: None,
        });
    }

    Ok(())
}
