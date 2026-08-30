"""
Replication of OmniMedVQA's own published "Prefix-based Score" metric --
the number in the paper's/leaderboard's tables actually comparable to a
model's MCQ performance (the paper's other metric, "Question-answering
Score", scores free-text generation matched to the nearest option by
embedding similarity; this module does NOT implement that one).

This is a DIFFERENT scoring mechanism from src/decode.py's Stage 6
(logit-masked single-token letter argmax, used by the competition
pipeline in src/predict.py's predict()). It exists solely so
src/evaluate_omnimed.py can report a number genuinely apples-to-apples
with OmniMedVQA's own leaderboard -- never imported by predict()/eval.py.

Reference methodology, read directly from the paper's own eval code
(OpenGVLab/Multi-Modality-Arena, MedicalEval/Prefix_based_Score/
medical_llava.py -- the pattern shared across every evaluated model
except RadFM/Med-flamingo/MedVInT, which use bespoke scripts but the same
underlying idea):

  1. Build a plain completion prompt with NO options listed at all:
         "Question: {question} The answer is"
     (prompt_idx=4 of 5 templates in their load_prompt() -- the default
     actually used to produce the reported results.)
  2. For each candidate option's full TEXT (not its letter), tokenize
     " {candidate_text}." appended after that prompt, run one forward
     pass (image + prompt + candidate), and compute the mean
     cross-entropy loss over ONLY the candidate+period tokens -- the
     image and prompt tokens are masked out of the loss via label -100
     (PyTorch cross_entropy's default ignore_index).
  3. The candidate with the LOWEST mean loss (highest average per-token
     likelihood) is the prediction. (Their code then applies
     softmax(1/loss) to log a "confidence" score, but since that's a
     strictly monotonic transform of the losses for loss > 0, argmax over
     it is provably identical to argmin over the raw losses -- this
     module just does the latter directly, with no behavior difference.)
  4. Ground truth in their harness is compared as candidate TEXT, not a
     letter -- consistent with src/evaluate_omnimed.py's existing
     _resolve_gt_letter(), which already handles gt_answer being either a
     letter or option text.

Adaptation for Qwen3-VL: the reference code drives older VLMs (LLaVA,
BLIP-2, MiniGPT-4, etc.) through a raw string-completion API with an
inline <image> token. Qwen3-VL is an instruction-tuned chat model with no
such raw completion path, so the same prompt text is instead placed as
the sole user turn's content (via the same chat template
src/predict.py's Stage 4 uses, add_generation_prompt=True), and the
candidate text is scored as if it were the assistant's completion. No
options are listed in the prompt (unlike Stage 4's normal user_prompt) --
that is the one deliberate divergence from Stage 4's prompt-building,
required to match the reference methodology, which never shows the model
an enumerated option list either.
"""

import torch
import torch.nn.functional as F

from src.model_loader import model, processor

PREFIX_SCORE_PROMPT_TEMPLATE = "Question: {question} The answer is"


def _build_prefix_text(question: str) -> str:
    """The plain completion prompt (image placeholder + "Question: ...
    The answer is"), chat-templated with add_generation_prompt=True --
    the shared prefix every candidate's full prompt text is built from."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PREFIX_SCORE_PROMPT_TEMPLATE.format(question=question)},
            ],
        },
    ]
    return processor.apply_chat_template(messages, add_generation_prompt=True)


def _tokenized_length(text: str, image) -> int:
    """Token count of `text` (with `image`) under the processor -- used to
    find where the shared prefix ends inside a longer (prefix+candidate)
    tokenization, so only the candidate's own tokens get scored."""
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    return inputs["input_ids"].shape[1]


@torch.inference_mode()
def _candidate_loss(image, prefix_text: str, prefix_token_len: int, candidate_text: str) -> float:
    """Mean cross-entropy loss (negative average log-likelihood per
    token) of " {candidate_text}." as a continuation of `prefix_text` --
    lower is more likely. Same masking/reduction as the reference
    implementation (independently tokenize the prefix alone vs.
    prefix+candidate together, and use the length difference as the
    candidate's token span -- exactly mirroring medical_llava.py's
    `lang_diff` computation, tolerant of any BPE merge-boundary shift at
    the seam between the two tokenizations).

    Deliberately reprocesses the full (prefix+candidate) text through the
    processor from scratch for every candidate, rather than manually
    splicing tensors from a cached prefix-only `inputs` dict: Qwen3-VL's
    M-RoPE position-id computation (get_rope_index) needs an internally
    derived image/text token-type map that only stays consistent with
    input_ids when the processor builds the whole sequence itself --
    splicing pre-tokenized suffix ids onto a shorter cached tensor left
    that map sized for the old (shorter) sequence and crashed with a
    shape-mismatched boolean index. Rerunning the processor is the same
    call pattern src/predict.py's own single-shot Stage 5 call already
    uses successfully, just repeated once per candidate. The image itself
    is cheap to reprocess relative to the LLM forward pass that follows."""
    full_text = f"{prefix_text} {candidate_text}."
    inputs = processor(text=[full_text], images=[image], return_tensors="pt").to(model.device)
    inputs.pop("token_type_ids", None)

    input_ids = inputs["input_ids"]
    targets = input_ids.clone()
    targets[:, :prefix_token_len] = -100  # mask everything but the candidate+period span

    # use_cache=False: this is a single from-scratch forward pass, never
    # incremental decoding -- no KV-cache/rope-delta state needs to
    # persist past this one call.
    logits = model(**inputs, use_cache=False).logits  # [1, seq_len, vocab]

    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = targets[:, 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1), reduction="mean")
    return float(loss.item())


def score_choices_by_prefix_loss(image, question: str, choices: dict) -> dict:
    """
    Args:
        image: an already-loaded/decoded PIL Image (NOT a path) -- callers
            handle Stage -1/corrupt-image loading and Stage 0-2
            preprocessing themselves, same division of responsibility as
            src/decode.py's constrained_predict_with_scores().
        question: natural-language question string. Options are
            deliberately NOT appended here -- see module docstring.
        choices: dict like {"A": "...", "B": "...", ...}.
    Returns:
        {letter: loss} for every letter in `choices`. The letter with the
        LOWEST loss is the Prefix-Score prediction
        (min(result, key=result.get)) -- see
        src/predict.py's predict_by_prefix_score().
    """
    prefix_text = _build_prefix_text(question)
    prefix_token_len = _tokenized_length(prefix_text, image)
    return {
        letter: _candidate_loss(image, prefix_text, prefix_token_len, text)
        for letter, text in choices.items()
    }
