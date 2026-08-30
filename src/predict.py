"""
Top-level predict() entry point (Section 5.2 of medical_vqa_architecture.md),
wiring together every pipeline stage:

    Stage -1 (src/volume_loader.py)   -> volumetric/DICOM ingest (NIfTI,
                                          single/multi-frame DICOM, DICOM
                                          series folders) into a 2D RGB image
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

Two public entry points share the Stage-wiring helpers below
(_load_input_image, _run_pipeline):
    predict(image, query, choices) -> str
        The competition path (used by eval.py). Unchanged behavior: any
        failure mode degrades immediately to config.FALLBACK_ANSWER_LETTER.
    predict_with_diagnostics(image, query, choices) -> dict
        Used by src/evaluate_omnimed.py. On an unreadable/unloadable image
        (or a Stage 0-4 crash), retries once on a neutral gray canvas
        instead of answering fully blind, and returns a rich diagnostics
        dict (modality, track, per-choice logits, fallback reason, etc.)
        instead of just the winning letter.
"""

import logging
import threading
import time
from pathlib import Path

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
from src.decode import constrained_predict_with_scores
from src.volume_loader import load_volume, VolumeLoadError

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


def _load_input_image(image) -> Image.Image:
    """Stage -1 (volumetric/DICOM ingest) + corrupt-image guard, factored
    out so predict() and predict_with_diagnostics() share one loading path.
    Returns a loaded PIL Image. Raises VolumeLoadError (from Stage -1) or
    UnidentifiedImageError/OSError/ValueError (from PIL) on failure --
    callers decide how to degrade; this function itself never degrades
    anything, it just loads or raises.
    """
    if isinstance(image, (str, Path)):
        volume_image = load_volume(Path(image))
        if volume_image is not None:
            image = volume_image
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image.load()
    return image


def _neutral_gray_canvas() -> Image.Image:
    """A blank, mid-gray canvas substituted for the real image when it
    can't be loaded at all (predict_with_diagnostics()'s blind-fallback
    path) -- lets the model still reason from the question text and
    choices alone, rather than skipping inference entirely."""
    size = config.BLIND_FALLBACK_IMAGE_SIZE
    value = config.BLIND_FALLBACK_GRAY_VALUE
    return Image.new("RGB", size, (value, value, value))


def _run_pipeline(image: Image.Image, query: str, choices: dict, valid_letters: list) -> dict:
    """Stages 0-6 given an already-loaded PIL image. Raises on any Stage
    0-4 failure (normalization/routing/preprocessing/prompt-building/
    tokenization) -- callers decide how to degrade. Stage 5-6's timeout is
    NOT raised; it's reported via the returned dict's "timed_out" key,
    since a stalled forward pass isn't retry-worthy the way a Stage 0-4
    crash is (see predict_with_diagnostics()'s docstring).

    Returns: {"answer", "modality", "subtype", "track", "scores",
    "timed_out"}. "answer"/"scores" are None when "timed_out" is True.
    """
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

    # Stage 5+6: single forward pass + constrained decode, under a hard
    # timeout so one stalled query can't blow up the average inference
    # time in the scoring formula. valid_letters restricts the argmax to
    # exactly the choices this query actually offered (2-choice/Yes-No
    # rows never get answered "C" or "D").
    result = _run_with_timeout(
        lambda: constrained_predict_with_scores(inputs, valid_letters=valid_letters),
        config.INFERENCE_TIMEOUT_SECONDS,
    )
    if result is None:
        return {"answer": None, "modality": modality, "subtype": subtype, "track": track, "scores": None, "timed_out": True}
    answer, scores = result
    return {"answer": answer, "modality": modality, "subtype": subtype, "track": track, "scores": scores, "timed_out": False}


def predict(image, query: str, choices: dict) -> str:
    """
    Args:
        image: PIL Image (raw medical image, arbitrary modality). A path
            or file-like object is also accepted defensively -- a path to
            a NIfTI volume, a single/multi-frame DICOM file, or a folder of
            DICOM slices is decoded by Stage -1 (src/volume_loader.py);
            anything else is decoded via PIL as before.
        query: natural language question string.
        choices: dict like {"A": "...", "B": "...", "C": "...", "D": "..."}
            (4-choice MCQ), or {"A": "...", "B": "..."} for a 2-choice/
            Yes-No question -- see config.CHOICE_SET_OPTIONS.
    Returns:
        Single character: "A", "B", "C", or "D". Never raises -- on any
        failure mode this degrades to config.FALLBACK_ANSWER_LETTER
        rather than crashing the harness (Section 5.5).
    """
    # --- Stage -1: volumetric/DICOM ingest ---
    # Must run before PIL ever touches the path: PIL cannot decode NIfTI or
    # DICOM at all, and would otherwise just raise straight into the
    # corrupt-image guard below, discarding the scan with no real analysis
    # (see eval.py's former KNOWN LIMITATION note). Only attempted for
    # path-like inputs -- an already-decoded PIL.Image is passed through
    # untouched, same as before.
    try:
        image = _load_input_image(image)
    except VolumeLoadError as exc:
        logger.warning("Unreadable volumetric/DICOM input, returning fallback answer: %s", exc)
        return config.FALLBACK_ANSWER_LETTER
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.warning("Unreadable image, returning fallback answer: %s", exc)
        return config.FALLBACK_ANSWER_LETTER
    except Exception as exc:  # noqa: BLE001 - defensive backstop, image loading must never crash the run
        logger.exception("Image loader raised unexpectedly, returning fallback answer: %s", exc)
        return config.FALLBACK_ANSWER_LETTER

    # --- Empty/malformed choices guard ---
    if not _validate_choices(choices):
        logger.warning("Malformed choices dict %r, returning fallback answer", choices)
        return config.FALLBACK_ANSWER_LETTER

    valid_letters = list(choices.keys())
    try:
        result = _run_pipeline(image, query, choices, valid_letters)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pre-inference pipeline stage failed, returning fallback answer: %s", exc)
        return config.FALLBACK_ANSWER_LETTER

    if result["timed_out"]:
        logger.warning("Inference timed out or failed, returning fallback answer")
        return config.FALLBACK_ANSWER_LETTER
    return result["answer"]


# ---------------------------------------------------------------------------
# Diagnostics variant -- used by src/evaluate_omnimed.py
# ---------------------------------------------------------------------------
# fallback_reason values (None when fallback_triggered is False):
_REASON_MALFORMED_CHOICES = "malformed_choices"
_REASON_IMAGE_LOAD_FAILED = "image_load_failed"
_REASON_PIPELINE_STAGE_FAILED = "pipeline_stage_failed"
_REASON_PIPELINE_FAILED_ON_GRAY_CANVAS = "pipeline_failed_on_gray_canvas"
_REASON_INFERENCE_TIMEOUT = "inference_timeout"


def _diagnostic_result(elapsed, answer=None, modality=None, subtype=None, track=None,
                        fallback_triggered=False, fallback_reason=None,
                        top1_logit=None, top2_logit=None, logit_margin=None) -> dict:
    return {
        "answer": answer if answer is not None else config.FALLBACK_ANSWER_LETTER,
        "modality": modality,
        "subtype": subtype,
        "track": track,
        "fallback_triggered": fallback_triggered,
        "fallback_reason": fallback_reason,
        "top1_logit": top1_logit,
        "top2_logit": top2_logit,
        "logit_margin": logit_margin,
        "inference_time": elapsed,
    }


def predict_with_diagnostics(image, query: str, choices: dict) -> dict:
    """
    Like predict(), but built for offline evaluation (src/evaluate_omnimed.py)
    rather than the competition harness:

      - An image that can't be loaded/decoded at all (or a Stage 0-4 crash
        on an image that DID load) is retried exactly once on a neutral
        gray canvas (_neutral_gray_canvas()) instead of short-circuiting to
        config.FALLBACK_ANSWER_LETTER -- the model still gets to reason
        from the question text and choices alone, rather than the answer
        being decided before it's ever called. Only one retry is ever
        attempted (no loops): if the gray canvas ALSO fails, that's a
        genuine bug rather than a bad image, and the absolute last-resort
        fallback letter is returned.
      - Stage 5-6 (the actual forward pass) is deliberately NOT retried on
        timeout -- a stalled/slow forward pass is a latency problem, not a
        content problem, and doubling the GPU work per query would only
        worsen timeout-budget pressure (see _run_with_timeout's docstring
        on abandoned work not actually being cancelled).
      - Malformed choices short-circuit before any image work at all
        (unlike predict()'s ordering) since there is nothing sensible to
        decode against regardless of the image.
      - Returns a rich diagnostics dict instead of just the winning letter.

    Returns:
        dict with keys: answer, modality, subtype, track,
        fallback_triggered (bool), fallback_reason (one of the
        _REASON_* constants above, or None), top1_logit, top2_logit,
        logit_margin (all None if the model was never actually called --
        malformed choices or a double pipeline failure), inference_time
        (seconds, wall-clock for this whole call). Never raises.
    """
    t0 = time.perf_counter()

    if not _validate_choices(choices):
        logger.warning("Malformed choices dict %r, returning fallback answer", choices)
        return _diagnostic_result(
            time.perf_counter() - t0,
            fallback_triggered=True, fallback_reason=_REASON_MALFORMED_CHOICES,
        )
    valid_letters = list(choices.keys())

    fallback_triggered = False
    fallback_reason = None
    try:
        loaded_image = _load_input_image(image)
    except Exception as exc:  # noqa: BLE001 - any load failure triggers the blind fallback below
        logger.warning(
            "Image load failed (%s: %s), falling back to a blind model call on a neutral gray canvas",
            type(exc).__name__, exc,
        )
        fallback_triggered = True
        fallback_reason = _REASON_IMAGE_LOAD_FAILED
        loaded_image = _neutral_gray_canvas()

    try:
        result = _run_pipeline(loaded_image, query, choices, valid_letters)
    except Exception as exc:  # noqa: BLE001
        if fallback_triggered:
            # Already on the gray canvas and it STILL crashed the pipeline
            # -- no further retry, go straight to the last-resort letter.
            logger.exception("Pipeline failed even on the gray-canvas fallback, returning fallback answer: %s", exc)
            return _diagnostic_result(
                time.perf_counter() - t0,
                fallback_triggered=True, fallback_reason=_REASON_PIPELINE_FAILED_ON_GRAY_CANVAS,
            )
        logger.warning(
            "Pipeline stage failed (%s), retrying with a blind model call on a neutral gray canvas", exc
        )
        fallback_triggered = True
        fallback_reason = _REASON_PIPELINE_STAGE_FAILED
        try:
            result = _run_pipeline(_neutral_gray_canvas(), query, choices, valid_letters)
        except Exception as retry_exc:  # noqa: BLE001
            logger.exception("Pipeline failed even on the gray-canvas retry, returning fallback answer: %s", retry_exc)
            return _diagnostic_result(
                time.perf_counter() - t0,
                fallback_triggered=True, fallback_reason=_REASON_PIPELINE_FAILED_ON_GRAY_CANVAS,
            )

    elapsed = time.perf_counter() - t0

    if result["timed_out"]:
        # If a gray-canvas retry was already in effect (fallback_triggered
        # True going in), a plain "inference_timeout" reason would silently
        # discard the fact that the ORIGINAL image had already failed to
        # load -- report both, since they're independent events that both
        # happened on this sample.
        reason = f"{fallback_reason}+{_REASON_INFERENCE_TIMEOUT}" if fallback_triggered else _REASON_INFERENCE_TIMEOUT
        logger.warning("Inference timed out, returning fallback answer")
        return _diagnostic_result(
            elapsed, modality=result["modality"], subtype=result["subtype"], track=result["track"],
            fallback_triggered=True, fallback_reason=reason,
        )

    scores = result["scores"]
    ranked = sorted(scores.values(), reverse=True)
    top1_logit = ranked[0]
    top2_logit = ranked[1] if len(ranked) > 1 else None
    logit_margin = (top1_logit - top2_logit) if top2_logit is not None else None

    return _diagnostic_result(
        elapsed, answer=result["answer"], modality=result["modality"], subtype=result["subtype"],
        track=result["track"], fallback_triggered=fallback_triggered, fallback_reason=fallback_reason,
        top1_logit=top1_logit, top2_logit=top2_logit, logit_margin=logit_margin,
    )
