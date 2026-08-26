"""
Global, import-time loading of the VLM backbone, processor, and LoRA
adapter (Section 4.1 of medical_vqa_architecture.md).

Per the "global pre-loading" rule, every heavy object here -- model
weights, processor/tokenizer, and the LoRA adapter (if present) -- is
instantiated exactly once, at module import time, never inside predict().
Downstream modules (src/decode.py, src/predict.py) simply `from
src.model_loader import model, processor`; the act of importing this
module is what triggers the weight load and the CUDA warm-up pass below.
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
# Backbone: Qwen3-VL-4B-Instruct, loaded natively/unquantized (Section 3.3).
# Migrated off AWQ-quantized Qwen2-VL-7B specifically to eliminate the
# autoawq/gptqmodel dependency chain (autoawq archived/deprecated
# 2025-05-11; its successor, gptqmodel, requires a C++ toolchain to build
# from source on Windows). No quantization_config is passed at all -- at
# ~8-9GB in fp16, the 4B model comfortably fits the 48GB RTX 6000 Ada
# budget without it. (Deployed size was 8B initially; resized to 4B --
# same class, same code path, config.MODEL_PATH-only change -- to fit the
# competition's 10GB submission file size limit; see Section 3.1.)
# ---------------------------------------------------------------------------
def _load_model(attn_implementation: str):
    return Qwen3VLForConditionalGeneration.from_pretrained(
        config.MODEL_PATH,
        torch_dtype=config.TORCH_DTYPE,
        device_map=config.DEVICE,
        attn_implementation=attn_implementation,
    )


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
# LoRA adapter hot-swap (Section 1.3 / 3.2): the fine-tuned adapter is
# loaded once and set active, if one is actually present. Swapping to a
# different adapter later (e.g. a microscopy-specialized variant) is a
# cheap set_adapter() call, not a full checkpoint reload.
#
# Optional by design: no fine-tuning script exists in this repository yet
# (Section 3.2), so weights/lora-medvqa/ won't exist for most setups, and
# an adapter trained against a different model size (Section 3.1's
# migration notes -- Qwen2-VL-7B / Qwen3-VL-8B adapters are NOT compatible
# with Qwen3-VL-4B) would fail to load even if the directory is present.
# Rather than hard-fail the whole import over a missing or incompatible
# adapter, fall back to the base model, loudly logged so it's never a
# silent, unnoticed downgrade.
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
        model.load_adapter(config.LORA_PATH, adapter_name=config.LORA_ADAPTER_NAME)
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
