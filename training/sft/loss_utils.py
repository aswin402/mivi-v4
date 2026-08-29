"""
Loss masking utilities for ChatML agent training.
Masks prompt tokens (label = -100) so cross-entropy loss is only computed
on generated assistant thinking (<think>), tool calls (<tool_call>), and responses.
"""

from typing import List, Dict, Any
import torch

def mask_prompt_tokens(
    input_ids: List[int],
    tokenizer: Any,
    assistant_prefix: str = "<|im_start|>assistant",
    im_end_token: str = "<|im_end|>"
) -> List[int]:
    """
    Creates target labels where prompt tokens are replaced by -100 (ignored in CE loss).
    """
    labels = [-100] * len(input_ids)
    
    # Simple decode-based token boundary scan if tokenizer is available
    # Or token ID sequence scanning
    # In standard ChatML:
    # <|im_start|>assistant\n{CONTENT}<|im_end|>
    
    try:
        prefix_ids = tokenizer.encode(assistant_prefix, add_special_tokens=False)
        end_ids = tokenizer.encode(im_end_token, add_special_tokens=False)
    except Exception:
        # Fallback if tokenizer object isn't full HF tokenizer
        return list(input_ids)
        
    i = 0
    in_assistant = False
    prefix_len = len(prefix_ids)
    
    while i < len(input_ids):
        if not in_assistant:
            if input_ids[i:i + prefix_len] == prefix_ids:
                in_assistant = True
                i += prefix_len
                continue
            i += 1
        else:
            if end_ids and input_ids[i:i + len(end_ids)] == end_ids:
                labels[i] = input_ids[i]  # Include the end token in loss
                in_assistant = False
                i += len(end_ids)
                continue
            labels[i] = input_ids[i]
            i += 1
            
    return labels
