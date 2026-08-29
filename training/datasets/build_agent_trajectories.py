#!/usr/bin/env python3
"""
Mivi-v4 Agentic Trajectory Dataset Generator.
Synthesizes and formats multi-turn agent interactions in ChatML format
with <tools>, <think>, <tool_call>, and <tool_result> blocks.
"""

import json
import os
import argparse
from typing import List, Dict, Any

DEFAULT_TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression and return the exact numeric result",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "The math expression, e.g., '15 * 4 + 10'"}
            },
            "required": ["expression"]
        }
    },
    {
        "name": "read_file",
        "description": "Read the text content of a file within the workspace",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write text content to a file, automatically creating parent directories",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path"},
                "content": {"type": "string", "description": "Text content to write"}
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "list_dir",
        "description": "List all files and folders inside a given workspace directory",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative directory path, e.g., '.'"}
            },
            "required": ["path"]
        }
    }
]

def format_system_prompt(tools: List[Dict[str, Any]]) -> str:
    tools_json = json.dumps(tools, indent=2)
    return (
        f"You are MIVI, an agentic AI assistant. You have access to tools.\n"
        f"<tools>\n{tools_json}\n</tools>\n"
        f"To use a tool, output: <tool_call>{{\"name\": \"...\", \"arguments\": {{...}}}}</tool_call>\n"
        f"Think inside <think>...</think> before acting."
    )

def generate_sample_trajectories() -> List[Dict[str, Any]]:
    system_prompt = format_system_prompt(DEFAULT_TOOLS)
    
    samples = [
        # Sample 1: Multi-step calculation with thinking
        {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "What is 48 multiplied by 16 plus 75?"},
                {
                    "role": "assistant",
                    "content": (
                        "<think>\n"
                        "The user wants to evaluate the arithmetic expression: 48 * 16 + 75.\n"
                        "I should call the calculator tool to compute this precisely without mental calculation errors.\n"
                        "</think>\n"
                        "<tool_call>{\"name\": \"calculator\", \"arguments\": {\"expression\": \"48 * 16 + 75\"}}</tool_call>"
                    )
                },
                {"role": "tool", "content": "<tool_result name=\"calculator\">843</tool_result>"},
                {
                    "role": "assistant",
                    "content": (
                        "<think>\n"
                        "The calculator returned 843. Let's verify: 48 * 16 = 768, and 768 + 75 = 843. The answer is verified.\n"
                        "</think>\n"
                        "48 multiplied by 16 plus 75 is **843**."
                    )
                }
            ]
        },
        # Sample 2: Inspecting and modifying workspace code
        {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Check what files are in the current project and read config.json"},
                {
                    "role": "assistant",
                    "content": (
                        "<think>\n"
                        "I first need to list the contents of the current directory to see if config.json exists.\n"
                        "</think>\n"
                        "<tool_call>{\"name\": \"list_dir\", \"arguments\": {\"path\": \".\"}}</tool_call>"
                    )
                },
                {"role": "tool", "content": "<tool_result name=\"list_dir\">- config.json (48 bytes)\n- src/ (directory)\n- Cargo.toml (320 bytes)</tool_result>"},
                {
                    "role": "assistant",
                    "content": (
                        "<think>\n"
                        "config.json is present. Now I will read its contents.\n"
                        "</think>\n"
                        "<tool_call>{\"name\": \"read_file\", \"arguments\": {\"path\": \"config.json\"}}</tool_call>"
                    )
                },
                {"role": "tool", "content": "<tool_result name=\"read_file\">{\"model\": \"mivi-v4\", \"max_seq_len\": 65536}</tool_result>"},
                {
                    "role": "assistant",
                    "content": (
                        "<think>\n"
                        "I have read the config file. It specifies model 'mivi-v4' with max_seq_len 65536.\n"
                        "</think>\n"
                        "The project contains `config.json`, `src/`, and `Cargo.toml`.\n\nThe `config.json` content is:\n```json\n{\n  \"model\": \"mivi-v4\",\n  \"max_seq_len\": 65536\n}\n```"
                    )
                }
            ]
        },
        # Sample 3: Writing output report
        {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Create a summary note in notes/summary.md stating that all engine tests have passed."},
                {
                    "role": "assistant",
                    "content": (
                        "<think>\n"
                        "I need to write a summary note to 'notes/summary.md'. The write_file tool will create parent directories automatically.\n"
                        "</think>\n"
                        "<tool_call>{\"name\": \"write_file\", \"arguments\": {\"path\": \"notes/summary.md\", \"content\": \"# Test Summary\\n\\nAll 26 engine and integration tests passed successfully.\\n\"}}</tool_call>"
                    )
                },
                {"role": "tool", "content": "<tool_result name=\"write_file\">Successfully wrote 82 bytes to 'notes/summary.md'</tool_result>"},
                {
                    "role": "assistant",
                    "content": (
                        "<think>\n"
                        "File was written successfully. I will confirm to the user.\n"
                        "</think>\n"
                        "I have created `notes/summary.md` with the test summary details."
                    )
                }
            ]
        }
    ]
    return samples

def main():
    parser = argparse.ArgumentParser(description="Build agentic trajectory training dataset for Mivi-v4")
    parser.add_argument("--output", type=str, default="training/datasets/agent_trajectories.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    trajectories = generate_sample_trajectories()

    with open(args.output, "w", encoding="utf-8") as f:
        for t in trajectories:
            f.write(json.dumps(t) + "\n")

    print(f"✅ Generated {len(trajectories)} agent trajectories into {args.output}")

if __name__ == "__main__":
    main()
