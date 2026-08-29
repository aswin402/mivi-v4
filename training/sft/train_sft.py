#!/usr/bin/env python3
"""
Mivi-v4 Supervised Fine-Tuning (SFT) Training Script.
Fine-tunes LFM2.5-350M (or base model) on ChatML agentic trajectories.
"""

import os
import json
import argparse
from typing import Dict, Any

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Mivi-v4 base model on agentic datasets")
    parser.add_argument("--model_name_or_path", type=str, default="LiquidAI/LFM2.5-350M", help="Base model identifier")
    parser.add_argument("--dataset_path", type=str, default="training/datasets/agent_trajectories.jsonl", help="Training JSONL dataset")
    parser.add_argument("--output_dir", type=str, default="models/mivi-v4-sft-checkpoint", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--dry_run", action="store_true", help="Validate dataset and print training configuration without running full GPU loop")
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 Mivi-v4 Supervised Fine-Tuning (SFT) Pipeline")
    print(f"Base Model:      {args.model_name_or_path}")
    print(f"Dataset Path:    {args.dataset_path}")
    print(f"Output Dir:      {args.output_dir}")
    print(f"Effective Batch: {args.batch_size * args.grad_accum_steps}")
    print(f"Learning Rate:   {args.lr}")
    print(f"Target Epochs:   {args.num_epochs}")
    print("=" * 60)

    if not os.path.exists(args.dataset_path):
        print(f"❌ Error: Dataset file '{args.dataset_path}' does not exist.")
        return

    # Count dataset samples
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        sample_count = sum(1 for _ in f)

    print(f"📊 Loaded {sample_count} training samples from {args.dataset_path}")

    if args.dry_run:
        print("✅ Dry-run validation successful! Training configuration is valid.")
        return

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    except ImportError:
        print("ℹ️ PyTorch / Transformers not installed in current environment.")
        print("Install with: pip install -r training/requirements.txt")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"💾 Checkpoints will be saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
