"""
Global, import-time loading of the VLM backbone, processor, and LoRA
adapter (Section 4.1 of medical_vqa_architecture.md).

Per the "global pre-loading" rule, every heavy object here -- model
weights, processor/tokenizer, and the LoRA adapter -- is instantiated
exactly once, at module import time, never inside predict(). Downstream
modules (src/decode.py, src/predict.py) simply `from src.model_loader
import model, processor`; the act of importing this module is what
triggers the weight load and the CUDA warm-up pass below.
"""

import logging

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, AwqConfig

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
# Backbone: AWQ 4-bit quantized weights (Section 3.3) -- fused kernels are
# typically faster than bf16 on Ada tensor cores, not just smaller, which
# directly helps the k * time penalty term in the scoring formula.
# ---------------------------------------------------------------------------
_quantization_config = AwqConfig(**config.AWQ_CONFIG)


def _load_model(attn_implementation: str):
    return AutoModelForVision2Seq.from_pretrained(
        config.MODEL_PATH,
        torch_dtype=config.TORCH_DTYPE,
        device_map=config.DEVICE,
        quantization_config=_quantization_config,
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
# loaded once and set active. Swapping to a different adapter later (e.g.
# a microscopy-specialized variant) is a cheap set_adapter() call, not a
# full checkpoint reload.
# ---------------------------------------------------------------------------
model.load_adapter(config.LORA_PATH, adapter_name=config.LORA_ADAPTER_NAME)
model.set_adapter(config.LORA_ADAPTER_NAME)
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
