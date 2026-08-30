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
def constrained_predict_with_scores(inputs, valid_letters=None) -> tuple:
    """
    Args:
        inputs: dict of tensors from processor(...), already on the
            model's device, containing the full chat-templated
            image + question + choices prompt with the generation
            prompt appended (i.e. ready for the model to emit the
            first answer token next).
        valid_letters: iterable of the letters actually offered for this
            query (e.g. ["A", "B"] for a 2-choice/Yes-No question). The
            argmax is restricted to exactly these -- a 2-choice question
            can never be answered "C" or "D" just because those tokens
            happened to score higher on unrelated logits. Defaults to all
            of CHOICE_TOKEN_IDS (i.e. "A"-"D") if omitted, preserving the
            original 4-choice-only behavior.

    Returns:
        (letter, scores) -- `letter` is a single character, one of
        `valid_letters`, deterministic/greedy by construction (Section
        5.5, no sampling involved). `scores` is the full {letter: logit}
        dict the argmax was taken over, exposed so callers that need more
        than the winning letter (e.g. confidence diagnostics -- see
        src/evaluate_omnimed.py's diagnostic sidecar log) don't need a
        second forward pass.
    """
    outputs = model(**inputs)
    last_logits = outputs.logits[:, -1, :]  # next-token logits, shape [1, vocab]

    letters = valid_letters if valid_letters is not None else CHOICE_TOKEN_IDS.keys()
    scores = {
        letter: max(last_logits[0, tok_id].item() for tok_id in CHOICE_TOKEN_IDS[letter])
        for letter in letters
    }
    return max(scores, key=scores.get), scores


def constrained_predict_letter(inputs, valid_letters=None) -> str:
    """Convenience wrapper over constrained_predict_with_scores() for
    callers that only need the winning letter (src/predict.py's predict())."""
    letter, _ = constrained_predict_with_scores(inputs, valid_letters=valid_letters)
    return letter
