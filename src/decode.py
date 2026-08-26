"""
Stage 6: logit-masked, single-forward-pass constrained decoding
(Section 4.2 of medical_vqa_architecture.md).

Instead of autoregressive generate() + string parsing, we run exactly one
forward pass and take the argmax over the candidate first-token logits for
"A"/"B"/"C"/"D" (and common surface variants: " A", "A)", "(A"). This makes
the returned answer provably one of {A, B, C, D} by construction, removes
all parsing/retry logic, and is the single largest latency win in the
pipeline -- it avoids the KV-cache growth and multiple forward passes of
autoregressive generation entirely.
"""

import torch

from src import config
from src.model_loader import model, processor


def _get_choice_token_ids():
    """Pre-compute first-token ids for each choice letter's surface
    variants, once, at import time (never recomputed inside predict())."""
    ids = {}
    for letter, variants in config.CHOICE_TOKEN_VARIANTS.items():
        for variant in variants:
            tok_ids = processor.tokenizer.encode(variant, add_special_tokens=False)
            if len(tok_ids) >= 1:
                ids.setdefault(letter, []).append(tok_ids[0])
    return ids


# Global, computed once. Maps e.g. "A" -> [id(" A"), id("A"), id("A)"), id("(A")].
CHOICE_TOKEN_IDS = _get_choice_token_ids()


@torch.inference_mode()
def constrained_predict_letter(inputs) -> str:
    """
    Args:
        inputs: dict of tensors from processor(...), already on the
            model's device, containing the full chat-templated
            image + question + choices prompt with the generation
            prompt appended (i.e. ready for the model to emit the
            first answer token next).

    Returns:
        Single character: "A", "B", "C", or "D". Deterministic /
        greedy by construction (Section 5.5) -- no sampling involved.
    """
    outputs = model(**inputs)
    last_logits = outputs.logits[:, -1, :]  # next-token logits, shape [1, vocab]

    scores = {
        letter: max(last_logits[0, tok_id].item() for tok_id in tok_ids)
        for letter, tok_ids in CHOICE_TOKEN_IDS.items()
    }
    return max(scores, key=scores.get)
