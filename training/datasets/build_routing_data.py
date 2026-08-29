#!/usr/bin/env python3
"""
Mivi-v4 Semantic Routing & Task Family Dataset Generator.
Synthesizes intent classification samples for routing prompts to LoRA specialists.
"""

import json
import os
import argparse
from typing import List, Dict, Any

ROUTING_SAMPLES = [
    # Debug
    {"prompt": "Fix the segmentation fault in my C++ pointer arithmetic", "family": "debug", "active_experts": ["code", "reasoning"]},
    {"prompt": "Why is my Python asyncio loop throwing RuntimeError: Event loop is closed?", "family": "debug", "active_experts": ["code", "reasoning"]},
    {"prompt": "Identify the race condition in this Rust Arc<Mutex> code", "family": "debug", "active_experts": ["code", "reasoning"]},
    
    # Code
    {"prompt": "Write a high-performance SwiGLU kernel in CUDA C++", "family": "code", "active_experts": ["code"]},
    {"prompt": "Implement a binary search tree insertion method in Python", "family": "code", "active_experts": ["code"]},
    {"prompt": "Create an Axum router with JWT bearer token verification", "family": "code", "active_experts": ["code"]},

    # Agent
    {"prompt": "Inspect the repository, run tests, and fix any failed assertions", "family": "agent", "active_experts": ["agent", "code", "reasoning"]},
    {"prompt": "Find all markdown files under docs/ and create an index.md table of contents", "family": "agent", "active_experts": ["agent", "code"]},
    {"prompt": "Execute the build pipeline and write deployment artifacts to dist/", "family": "agent", "active_experts": ["agent"]},

    # Research
    {"prompt": "Search the latest 2026 benchmarks for hybrid SSM and GQA architectures", "family": "research", "active_experts": ["agent", "chat"]},
    {"prompt": "Find papers explaining sub-quadratic KV cache scaling in Mamba2", "family": "research", "active_experts": ["agent", "reasoning"]},
    {"prompt": "Lookup documentation on Axum 0.7 middleware and error handling", "family": "research", "active_experts": ["agent", "code"]},

    # Chat
    {"prompt": "Hello! How are you doing today?", "family": "chat", "active_experts": ["chat"]},
    {"prompt": "Can you explain the difference between synchronous and asynchronous programming in simple terms?", "family": "chat", "active_experts": ["chat"]},
    {"prompt": "Summarize this paragraph in two bullet points", "family": "chat", "active_experts": ["chat"]}
]

def main():
    parser = argparse.ArgumentParser(description="Build semantic routing dataset for Mivi-v4")
    parser.add_argument("--output", type=str, default="training/datasets/routing_data.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for s in ROUTING_SAMPLES:
            f.write(json.dumps(s) + "\n")

    print(f"✅ Generated {len(ROUTING_SAMPLES)} routing samples into {args.output}")

if __name__ == "__main__":
    main()
