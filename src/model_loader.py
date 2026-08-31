"""
Global, import-time loading of the VLM backbone, processor, and LoRA
adapter (Section 4.1 of medical_vqa_architecture.md).

Per the "global pre-loading" rule, every heavy object here -- model
weights, processor/tokenizer, and the LoRA adapter (if present) -- is
instantiated exactly once, at module import time, never inside predict().
Downstream modules (src/decode.py, src/predict.py, src/prefix_score.py)
simply `from src.model_loader import model, processor`; the act of
importing this module is what triggers the weight load and the CUDA
warm-up pass below. `model` is whatever object survives every stage
below -- the plain base model, or a peft.PeftModel wrapping it if a LoRA
adapter loaded successfully -- and is guaranteed to still support
`model(**inputs)` / `model.generate(...)` / `model.device` /
`model.eval()`, since every downstream call site (src/decode.py's
constrained_predict_with_scores, src/prefix_score.py's _candidate_loss,
the warm-up pass below) only ever uses that common surface, which both
object types provide identically.

Quantization (config.QUANTIZATION_MODE) is opt-in and OFF by default --
importing this module with the default config produces byte-identical
loading behavior to before quantization support existed.
"""

import logging
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from src import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Processor: bounds the vision-token count via MIN_PIXELS / MAX_PIXELS
# (Section 4.3) so prefill latency/VRAM stay bounded across wildly
# differing native image resolutions (X-ray vs. WSI microscopy crop).
# ---------------------------------------------------------------------------
processor = AutoProcessor.from_pretrained(
    config.MODEL_PATH,
    min_pixels=config.MIN_PIXELS,
    max_pixels=config.MAX_PIXELS,
)

# ---------------------------------------------------------------------------
# Quantization (opt-in, config.QUANTIZATION_MODE -- see src/config.py's
# note on why this is off by default). Built once, at module scope, and
# passed explicitly into every _load_model() call below -- deliberately
# NOT computed inside _load_model() or folded into its try/except, so a
# misconfigured/missing bitsandbytes install fails loudly and immediately
# with a clear, specific error, rather than being silently swallowed by
# the attn_implementation retry logic (which would then fail a second
# time for the SAME underlying reason, logged as a confusing "attention
# backend unavailable" message instead of the real "quantization
# unavailable" one).
# ---------------------------------------------------------------------------
def _build_quantization_config():
    if config.QUANTIZATION_MODE is None:
        return None

    # Validate the mode value itself BEFORE checking for bitsandbytes --
    # otherwise a typo'd/invalid QUANTIZATION_MODE would get masked by a
    # coincidental "install bitsandbytes" error even when installing it
    # wouldn't fix anything (the real problem is the bad config value).
    if config.QUANTIZATION_MODE not in ("4bit", "8bit"):
        raise ValueError(
            f"Unknown config.QUANTIZATION_MODE={config.QUANTIZATION_MODE!r} "
            "(expected None, '4bit', or '8bit')"
        )

    try:
        # transformers.BitsAndBytesConfig is just a plain dataclass and
        # constructs fine even without the bitsandbytes PACKAGE installed
        # -- the real dependency is only pulled in later, inside
        # from_pretrained()'s bnb quantizer dispatch. Importing
        # bitsandbytes directly here, up front, is what actually surfaces
        # a missing install as its own clear error at this point, instead
        # of deeper inside _load_model() where it would raise the same
        # ImportError type the attn_implementation retry logic below
        # catches -- conflating "quantization misconfigured" with
        # "flash-attn unavailable" and retrying with the wrong fix.
        import bitsandbytes  # noqa: F401
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError(
            f"config.QUANTIZATION_MODE={config.QUANTIZATION_MODE!r} but bitsandbytes "
            "isn't installed (see requirements.txt) -- install it, or set "
            "QUANTIZATION_MODE = None to load the native fp16 model instead."
        ) from exc

    if config.QUANTIZATION_MODE == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=config.QUANTIZATION_COMPUTE_DTYPE,
            bnb_4bit_quant_type=config.QUANTIZATION_4BIT_QUANT_TYPE,
            bnb_4bit_use_double_quant=config.QUANTIZATION_4BIT_USE_DOUBLE_QUANT,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


_quantization_config = _build_quantization_config()

# ---------------------------------------------------------------------------
# Backbone: Qwen3-VL-4B-Instruct. Native/unquantized fp16 by default
# (Section 3.3) -- migrated off AWQ-quantized Qwen2-VL-7B specifically to
# eliminate the autoawq/gptqmodel dependency chain (autoawq archived/
# deprecated 2025-05-11; its successor, gptqmodel, requires a C++
# toolchain to build from source on Windows). At ~8-9GB in fp16, the 4B
# model comfortably fits the 48GB RTX 6000 Ada budget without any
# quantization_config at all. (Deployed size was 8B initially; resized to
# 4B -- same class, same code path, config.MODEL_PATH-only change -- to
# fit the competition's 10GB submission file size limit; see Section
# 3.1.) config.QUANTIZATION_MODE opts into 4-bit/8-bit bitsandbytes
# loading instead, for constrained-VRAM deployments.
#
# device_map=config.DEVICE (a single fixed device string, e.g. "cuda:0")
# is kept as-is under quantization too, rather than switched to
# device_map="auto": "auto" enables accelerate's multi-GPU/CPU-offload
# sharding, which isn't wanted for this single-GPU deployment and isn't
# required for bitsandbytes quantization to work -- a single-device
# device_map already places 100% of the (quantized) weights on that one
# device, identically to the unquantized path.
# ---------------------------------------------------------------------------
def _load_model(attn_implementation: str):
    kwargs = dict(device_map=config.DEVICE, attn_implementation=attn_implementation)
    if _quantization_config is not None:
        kwargs["quantization_config"] = _quantization_config
        # torch_dtype here governs the model's non-quantized
        # parameters/buffers (embeddings, layernorms, vision tower, etc.)
        # -- matched to the same compute dtype bitsandbytes dequantizes
        # into, so there's a single consistent non-int8/int4 dtype
        # throughout the model rather than mixing it with config.TORCH_DTYPE.
        kwargs["torch_dtype"] = config.QUANTIZATION_COMPUTE_DTYPE
    else:
        kwargs["torch_dtype"] = config.TORCH_DTYPE
    return Qwen3VLForConditionalGeneration.from_pretrained(config.MODEL_PATH, **kwargs)


try:
    model = _load_model(config.ATTN_IMPLEMENTATION)
except (ImportError, ValueError) as exc:
    # flash-attn requires a matching CUDA toolkit and may not be buildable
    # without sudo in the offline eval environment -- fall back to the
    # scaled-dot-product-attention backend (Section 4.4 / 5.4 note).
    logger.warning(
        "attn_implementation=%r unavailable (%s); falling back to %r",
        config.ATTN_IMPLEMENTATION,
        exc,
        config.ATTN_IMPLEMENTATION_FALLBACK,
    )
    model = _load_model(config.ATTN_IMPLEMENTATION_FALLBACK)

# ---------------------------------------------------------------------------
# LoRA adapter (Section 1.3 / 3.2): attached via peft.PeftModel.from_
# pretrained(), which wraps `model` in a PeftModel rather than mutating it
# in place -- `model` is reassigned to that wrapper only on success (see
# below). Swapping to a different adapter later (e.g. a microscopy-
# specialized variant) is a cheap PeftModel.set_adapter() call, not a
# full checkpoint reload.
#
# Optional by design: no fine-tuning script exists in this repository yet
# (Section 3.2), so weights/lora-medvqa/ (config.LORA_PATH) won't exist
# for most setups, and an adapter trained against a different model size
# (Section 3.1's migration notes -- Qwen2-VL-7B / Qwen3-VL-8B adapters are
# NOT compatible with Qwen3-VL-4B) would fail to load even if the
# directory is present. Rather than hard-fail the whole import over a
# missing or incompatible adapter, fall back to the base model, loudly
# logged so it's never a silent, unnoticed downgrade.
#
# Zero-crash note: `model = PeftModel.from_pretrained(model, ...)` only
# rebinds the `model` name if the call succeeds -- if it raises, Python
# never reaches the assignment, so `model` is untouched (still the base
# model from the block above) and the except branch below runs against
# that same, still-valid, still-usable object. No partial/half-wrapped
# state is possible.
# ---------------------------------------------------------------------------
_lora_dir = Path(config.LORA_PATH)
_lora_available = _lora_dir.is_dir() and any(_lora_dir.iterdir())

if not _lora_available:
    logger.warning(
        "No LoRA adapter found at %s (missing or empty) -- running the "
        "base %s model unmodified. Train/place an adapter there to use one.",
        config.LORA_PATH,
        config.BASE_MODEL_NAME,
    )
else:
    try:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, config.LORA_PATH, adapter_name=config.LORA_ADAPTER_NAME)
        model.set_adapter(config.LORA_ADAPTER_NAME)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Found a LoRA adapter at %s but failed to load it (%s) -- "
            "running the base %s model unmodified instead.",
            config.LORA_PATH,
            exc,
            config.BASE_MODEL_NAME,
        )

model.eval()

# ---------------------------------------------------------------------------
# CUDA warm-up pass (Section 4.1): the first real inference call otherwise
# absorbs CUDA context initialization and kernel autotuning cost, which
# would badly skew the first recorded inference_time (and potentially the
# average). Running one throwaway forward pass here, at import time, keeps
# that cost out of every predict() call.
# ---------------------------------------------------------------------------
with torch.inference_mode():
    _warmup_inputs = processor(
        text="warmup", images=None, return_tensors="pt"
    ).to(config.DEVICE)
    _ = model.generate(**_warmup_inputs, max_new_tokens=1)
    torch.cuda.synchronize()
