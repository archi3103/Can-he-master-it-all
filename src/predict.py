"""
Top-level predict() entry point (Section 5.2 of medical_vqa_architecture.md),
wiring together every pipeline stage:

    Stage 0 (src/preprocessing.py)    -> universal_normalize() (RGBA/alpha
                                          stripping, 16-bit/float stretch)
    Stage 1 (src/router_modality.py)  -> route_modality(): coarse stream +
                                          subtype (12-modality scaling)
    Stage 2 (src/preprocessing.py)    -> subtype-specific preprocessing
    Stage 3 (src/router_intent.py)    -> query-intent / track detection
    Stage 4 (src/prompt_builder.py)   -> system prompt composition
    Stage 5 (src/model_loader.py)     -> shared VLM backbone (pre-loaded)
    Stage 6 (src/decode.py)           -> logit-masked constrained decoding

and layering in the Section 5.5 failure-mode safeguards: a hard timeout
around the forward pass, a corrupt/unreadable-image guard, and a
malformed-choices guard. predict() never raises -- every failure mode
degrades to config.FALLBACK_ANSWER_LETTER instead, so one bad query can't
crash the harness or blow up the average-inference-time term of the
scoring formula.
"""

import logging
import threading

from PIL import Image, UnidentifiedImageError

from src import config

# `model` is imported alongside `processor` (unused directly below) purely
# for its import-time side effect: importing src.model_loader is what
# triggers the global weight load + CUDA warm-up pass (Section 4.1).
from src.model_loader import model, processor  # noqa: F401
from src.router_modality import route_modality
from src.router_intent import detect_track
from src.preprocessing import preprocess_image, universal_normalize
from src.prompt_builder import build_system_prompt
from src.decode import constrained_predict_letter

logger = logging.getLogger(__name__)


def _run_with_timeout(fn, timeout_seconds: float):
    """Runs fn() on a daemon worker thread and returns its result, or None
    if it doesn't complete within timeout_seconds (Section 5.5 timeout
    guard). Thread-based rather than signal-based because signal.alarm is
    unavailable on Windows and unsafe outside the main thread. Note this
    cannot forcibly cancel a stuck CUDA call -- the worker thread keeps
    running in the background -- but it prevents one stalled query from
    blocking the harness's average-inference-time measurement."""
    result = {}

    def _target():
        try:
            result["value"] = fn()
        except Exception:  # noqa: BLE001 - logged, converted to a fallback by the caller
            logger.exception("Inference worker raised an exception")

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        return None
    return result.get("value")


def _validate_choices(choices) -> bool:
    if not isinstance(choices, dict):
        return False
    keys = set(choices.keys())
    return any(keys == set(option) for option in config.CHOICE_SET_OPTIONS)


def predict(image, query: str, choices: dict) -> str:
    """
    Args:
        image: PIL Image (raw medical image, arbitrary modality). A path
            or file-like object is also accepted defensively and decoded
            via PIL.
        query: natural language question string.
        choices: dict like {"A": "...", "B": "...", "C": "...", "D": "..."}
            (4-choice MCQ), or {"A": "...", "B": "..."} for a 2-choice/
            Yes-No question -- see config.CHOICE_SET_OPTIONS.
    Returns:
        Single character: "A", "B", "C", or "D". Never raises -- on any
        failure mode this degrades to config.FALLBACK_ANSWER_LETTER
        rather than crashing the harness (Section 5.5).
    """
    # --- Corrupt/unreadable image guard ---
    try:
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.warning("Unreadable image, returning fallback answer: %s", exc)
        return config.FALLBACK_ANSWER_LETTER

    # --- Empty/malformed choices guard ---
    if not _validate_choices(choices):
        logger.warning("Malformed choices dict %r, returning fallback answer", choices)
        return config.FALLBACK_ANSWER_LETTER

    try:
        # Stage 0: universal normalization (RGBA/alpha-stripping, 16-bit
        # TIFF/float grayscale stretch) -- runs before modality routing so
        # Stage 1's heuristics always see a clean 8-bit RGB image.
        image = universal_normalize(image)

        # Stage 1: hierarchical modality routing (cheap, <2ms) -- coarse
        # stream via the original, unchanged 4-bucket router, then a
        # stream-conditional subtype classifier (12-modality scaling; see
        # config.py's "Stage 1b" section and src/router_modality.py).
        modality, subtype = route_modality(image)

        # Stage 2: subtype-specific preprocessing
        image = preprocess_image(image, modality, subtype)

        # Stage 3: query-intent / track detection (cheap, <2ms)
        track = detect_track(query)

        # Stage 4: build final prompt
        system_prompt = build_system_prompt(modality, track)
        choices_str = "\n".join(f"{k}) {v}" for k, v in choices.items())
        user_prompt = f"{query}\n\n{choices_str}\n\nAnswer with only the letter."

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        text_input = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[text_input], images=[image], return_tensors="pt").to(config.DEVICE)
        # Qwen3-VL's own usage example pops this key before generate() --
        # Qwen3VLForConditionalGeneration.forward() doesn't accept it, and
        # some processor code paths emit it regardless of call style. No-op
        # if absent, so this is a safe default rather than an assumption.
        inputs.pop("token_type_ids", None)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pre-inference pipeline stage failed, returning fallback answer: %s", exc)
        return config.FALLBACK_ANSWER_LETTER

    # Stage 5+6: single forward pass + constrained decode, under a hard
    # timeout so one stalled query can't blow up the average inference
    # time in the scoring formula. valid_letters restricts the argmax to
    # exactly the choices this query actually offered (2-choice/Yes-No
    # rows never get answered "C" or "D").
    valid_letters = list(choices.keys())
    answer = _run_with_timeout(
        lambda: constrained_predict_letter(inputs, valid_letters=valid_letters),
        config.INFERENCE_TIMEOUT_SECONDS,
    )
    if answer is None:
        logger.warning("Inference timed out or failed, returning fallback answer")
        return config.FALLBACK_ANSWER_LETTER
    return answer
