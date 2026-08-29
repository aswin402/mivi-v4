#!/usr/bin/env python3
"""
Mivi-v4 Reasoning & Thinking Dataset Generator.
Synthesizes chain-of-thought traces in ChatML <think>...</think> format.
"""

import json
import os
import argparse
from typing import List, Dict, Any

def generate_reasoning_samples() -> List[Dict[str, Any]]:
    system_prompt = (
        "You are MIVI, a precise reasoning AI assistant. "
        "Always reason carefully inside <think>...</think> before giving your final concise response."
    )

    samples = [
        {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "If 5 workers can build 5 tables in 5 days, how many days does it take 100 workers to build 100 tables?"},
                {
                    "role": "assistant",
                    "content": (
                        "<think>\n"
                        "Let's break down the rate of work:\n"
                        "1. 5 workers build 5 tables in 5 days.\n"
                        "2. Therefore, 1 worker builds 1 table in 5 days (5 tables / 5 workers = 1 table per worker in 5 days).\n"
                        "3. If we have 100 workers, each worker builds 1 table in 5 days.\n"
                        "4. In 5 days, 100 workers working in parallel will build 100 * 1 = 100 tables.\n"
                        "5. So the answer is 5 days, not 100 days.\n"
                        "</think>\n"
                        "It takes **5 days**.\n\n"
                        "Each worker takes 5 days to build 1 table. Thus, 100 workers working concurrently will complete 100 tables in 5 days."
                    )
                }
            ]
        },
        {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "In a sorted array [2, 5, 8, 12, 16, 23, 38, 56, 72, 91], which indices are checked during binary search for 23?"},
                {
                    "role": "assistant",
                    "content": (
                        "<think>\n"
                        "Array length = 10 (indices 0 to 9).\n"
                        "Target = 23.\n"
                        "Step 1: low = 0, high = 9. mid = (0 + 9) // 2 = 4. array[4] = 16. Since 23 > 16, search right half. low = mid + 1 = 5.\n"
                        "Step 2: low = 5, high = 9. mid = (5 + 9) // 2 = 7. array[7] = 56. Since 23 < 56, search left half. high = mid - 1 = 6.\n"
                        "Step 3: low = 5, high = 6. mid = (5 + 6) // 2 = 5. array[5] = 23. Target found!\n"
                        "The sequence of checked indices is: 4 (value 16), 7 (value 56), and 5 (value 23).\n"
                        "</think>\n"
                        "Binary search examines the following indices in order:\n"
                        "1. **Index 4** (value `16`) — target 23 is greater, so move right.\n"
                        "2. **Index 7** (value `56`) — target 23 is smaller, so move left.\n"
                        "3. **Index 5** (value `23`) — match found!"
                    )
                }
            ]
        }
    ]
    return samples

def main():
    parser = argparse.ArgumentParser(description="Build reasoning training dataset for Mivi-v4")
    parser.add_argument("--output", type=str, default="training/datasets/reasoning_data.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    samples = generate_reasoning_samples()

    with open(args.output, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"✅ Generated {len(samples)} reasoning samples into {args.output}")

if __name__ == "__main__":
    main()
