"""
LoRA fine-tuning of the Qwen3-VL-4B-Instruct backbone (Section 3.2 of
medical_vqa_architecture.md), against a local OmniMedVQA Open-access
mirror -- reuses src/evaluate_omnimed.py's exact local-loading
conventions (load_omnimed_samples, _resolve_gt_letter, the
--data-root/--dataset-names layout) so the same dataset serves both
evaluation and fine-tuning, and reuses Stages 0-4 of src/predict.py's
pipeline (universal_normalize, route_modality, preprocess_image,
detect_track, build_system_prompt) so a fine-tuned adapter is trained on
EXACTLY the same image preprocessing and prompt format predict() builds
at inference time -- not a hand-approximated variant that could drift.

Training objective -- deliberately narrow, matching Stage 6's decode
mechanism exactly: predict()'s constrained decoding only ever reads the
FIRST generated token's logits, masked to "A"/"B"/"C"/"D" (src/decode.py).
So the only thing worth training is "does the model's very first
generated token match the correct letter" -- next-token cross-entropy
loss, computed ONLY on the correct answer letter's token position
(everything else -- system prompt, question, image tokens -- masked to
-100 via `labels`), not a full-sequence language-modeling loss over some
free-text explanation the model was never asked to produce and Stage 6
would never read anyway.

Kaggle-friendly by default: loads the base model via bitsandbytes 4-bit
(QLoRA) unless --no-4bit is passed, since a free-tier Kaggle GPU (T4 x2 /
P100, 16GB) cannot comfortably hold gradients/activations/optimizer
state for a 4B model in fp16, even though the model alone fits fine
(config.py's backbone-model note: ~8-9GB at fp16, inference-only). Per-
device batch size is fixed at 1 (multi-image batching for a VLM needs
padding image_grid_thw/pixel_values consistently across a batch, which
adds real complexity for no benefit at fine-tuning-scale data volumes --
use --grad-accum-steps for a larger effective batch size instead).

Usage (run from the repository root; see the module docstring's sibling,
src/evaluate_omnimed.py, for --data-root's exact expected directory
layout):
    python src/train_lora.py --data-root /path/to/OmniMedVQA --dataset-names ACRIMA
    python src/train_lora.py --data-root /path/to/OmniMedVQA --dataset-names ACRIMA "Adam Challenge" \
        --max-samples 500 --epochs 3 --output-dir weights/lora-medvqa
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Must happen before `import torch` -- multi-GPU Kaggle sessions (e.g. "GPU
# T4 x2") otherwise leave 2 CUDA devices visible, and plain (non-torchrun/
# non-accelerate-launched) `transformers.Trainer` responds to that by
# silently wrapping the model in `torch.nn.DataParallel`. DataParallel's
# naive parameter-replication is incompatible with this script's fixed
# `device_map` model loading and (when --no-4bit isn't passed) bitsandbytes
# 4-bit quantized layers -- it manifests as `StopIteration` deep inside
# Qwen3-VL's `self.visual.dtype` property on the replicated copy, not as
# anything that looks like a GPU-count problem. A single T4 (16GB) already
# comfortably fits this 4B QLoRA run (~3GB VRAM at load), so there is no
# reason to want the second GPU here; respects an explicit override if one
# is already set in the environment (e.g. a real torchrun/accelerate launch).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Trainer, TrainingArguments  # noqa: E402

from src import config  # noqa: E402
from src.evaluate_omnimed import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_DATASET_NAMES,
    DEFAULT_SEED,
    _resolve_gt_letter,
    load_omnimed_samples,
)
from src.preprocessing import preprocess_image, universal_normalize  # noqa: E402
from src.prompt_builder import build_system_prompt  # noqa: E402
from src.router_intent import detect_track  # noqa: E402
from src.router_modality import route_modality  # noqa: E402
from src.volume_loader import VolumeLoadError, load_volume  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_GRAD_ACCUM_STEPS = 8
DEFAULT_CHECKPOINT_DIR = "lora_training_run"


# ---------------------------------------------------------------------------
# Image loading -- deliberately NOT `from src.predict import _load_input_image`:
# predict.py imports src.model_loader at its own top level, which would
# transitively trigger loading a SECOND full copy of the inference model
# just to reach this one small helper function. This is the same ~8-line
# body, kept independent so this script only ever loads the model once,
# through its own training-specific path below.
# ---------------------------------------------------------------------------
def _load_input_image(image):
    if isinstance(image, (str, Path)):
        volume_image = load_volume(Path(image))
        if volume_image is not None:
            image = volume_image
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image.load()
    return image


# ---------------------------------------------------------------------------
# Training example construction -- mirrors src/predict.py's _run_pipeline
# Stages 0-4 exactly (same normalize/route/preprocess/track/prompt calls),
# then appends the correct answer letter as the target completion.
# ---------------------------------------------------------------------------
def _tokenized_length(text: str, image, processor) -> int:
    """Token count of `text` (with `image`) under the processor. Used to
    find where the prompt ends inside the longer (prompt+answer-letter)
    tokenization, so only the letter's own token(s) get unmasked labels.
    Deliberately reprocesses from scratch rather than splicing token ids
    onto a cached prefix -- Qwen3-VL's M-RoPE position-id computation
    needs an internally-derived image/text token-type map that only
    stays consistent with input_ids when the processor builds the whole
    sequence itself (the same pitfall discovered and fixed in
    src/prefix_score.py -- see that module's _candidate_loss docstring)."""
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    return inputs["input_ids"].shape[1]


def _build_training_example(sample: dict, processor) -> dict:
    """
    Args:
        sample: dict with image_path, question, choices, gt_letter (the
            resolved answer letter -- callers attach this via
            _resolve_gt_letter() before calling this function).
    Returns:
        A dict of tensors (input_ids, attention_mask, pixel_values,
        image_grid_thw, labels, ...) ready for the model's forward(),
        batch dim already 1 courtesy of the processor call.
    Raises:
        Any exception from image loading/decoding/preprocessing propagates
        -- callers (_PrebuiltExampleDataset) catch it and skip the sample,
        logged, rather than poisoning training with a blind/wrong image
        the way predict_with_diagnostics()'s gray-canvas fallback would be
        appropriate for at INFERENCE time but not at TRAINING time (a
        wrong image paired with the "correct" answer would actively teach
        the model something false).
    """
    if sample["image_path"] is None:
        raise ValueError(f"question_id={sample.get('question_id')!r}: image_path did not resolve locally")

    image = _load_input_image(sample["image_path"])
    image = universal_normalize(image)
    modality, subtype = route_modality(image)
    image = preprocess_image(image, modality, subtype)
    track = detect_track(sample["question"])

    system_prompt = build_system_prompt(modality, track)
    choices_str = "\n".join(f"{k}) {v}" for k, v in sample["choices"].items())
    user_prompt = f"{sample['question']}\n\n{choices_str}\n\nAnswer with only the letter."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]},
    ]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    full_text = prompt_text + sample["gt_letter"]

    prompt_len = _tokenized_length(prompt_text, image, processor)

    inputs = processor(text=[full_text], images=[image], return_tensors="pt")
    inputs.pop("token_type_ids", None)

    labels = inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100  # only the answer-letter token(s) contribute to the loss
    inputs["labels"] = labels
    return inputs


class _PrebuiltExampleDataset(torch.utils.data.Dataset):
    """Eagerly builds every training example once at construction time
    (image decode + preprocessing + tokenization), rather than redoing
    that CPU-bound work in __getitem__ on every access across every
    epoch -- worthwhile at the dataset sizes LoRA fine-tuning actually
    uses (hundreds to low-thousands of examples, not millions). Samples
    that fail to build (unresolvable image, corrupt file, unresolvable
    ground truth already filtered out by the caller) are skipped and
    logged, never crash the run."""

    def __init__(self, samples: list, processor):
        self.examples = []
        skipped = 0
        for sample in samples:
            try:
                self.examples.append(_build_training_example(sample, processor))
            except (VolumeLoadError, UnidentifiedImageError, OSError, ValueError) as exc:
                skipped += 1
                logger.warning(
                    "Skipping question_id=%r while building its training example: %s",
                    sample.get("question_id"), exc,
                )
        if skipped:
            logger.info("%d/%d sample(s) skipped while building the training set.", skipped, len(samples))
        if not self.examples:
            raise RuntimeError(
                "No trainable examples were built -- check --data-root/--dataset-names, "
                "and the warnings above for why every sample was skipped."
            )
        logger.info("Built %d trainable example(s).", len(self.examples))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def _collate_single(batch):
    """per_device_train_batch_size is fixed at 1 (see module docstring),
    so each "batch" the Trainer hands the collator is a length-1 list of
    an already-batch-dim-1 tensor dict (straight from the processor call
    in _build_training_example) -- just unwrap it."""
    assert len(batch) == 1, "train_lora.py only supports per_device_train_batch_size=1"
    return batch[0]


# ---------------------------------------------------------------------------
# Model loading -- deliberately independent of src/model_loader.py: that
# module is inference-oriented (puts the model in eval() mode, runs a
# CUDA warm-up generate() call, and -- critically -- will auto-attach
# whatever adapter already exists at config.LORA_PATH, which would wrongly
# double-wrap a fresh training run with an existing adapter's weights
# already active underneath the new LoraConfig). This function is the
# training-time equivalent, sharing config.py's constants as the single
# source of truth for model identity/dtype/quantization tunables, but
# with its own from-scratch load and its own flash-attn->sdpa fallback
# (same logic as model_loader.py's, just not importing that module to
# avoid loading a second full copy of the model into memory).
# ---------------------------------------------------------------------------
def _load_base_model_for_training(use_4bit: bool):
    kwargs = dict(device_map=config.DEVICE, attn_implementation=config.ATTN_IMPLEMENTATION)
    if use_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=config.QUANTIZATION_COMPUTE_DTYPE,
            bnb_4bit_quant_type=config.QUANTIZATION_4BIT_QUANT_TYPE,
            bnb_4bit_use_double_quant=config.QUANTIZATION_4BIT_USE_DOUBLE_QUANT,
        )
        kwargs["torch_dtype"] = config.QUANTIZATION_COMPUTE_DTYPE
    else:
        kwargs["torch_dtype"] = config.TORCH_DTYPE

    try:
        return Qwen3VLForConditionalGeneration.from_pretrained(config.MODEL_PATH, **kwargs)
    except (ImportError, ValueError) as exc:
        logger.warning(
            "attn_implementation=%r unavailable (%s); falling back to %r",
            config.ATTN_IMPLEMENTATION, exc, config.ATTN_IMPLEMENTATION_FALLBACK,
        )
        kwargs["attn_implementation"] = config.ATTN_IMPLEMENTATION_FALLBACK
        return Qwen3VLForConditionalGeneration.from_pretrained(config.MODEL_PATH, **kwargs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tunes Qwen3-VL-4B-Instruct against a local OmniMedVQA "
                     "Open-access mirror (see src/evaluate_omnimed.py for the expected "
                     "--data-root layout)."
    )
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dataset-names", nargs="+", default=DEFAULT_DATASET_NAMES)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = no limit.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output-dir", type=str, default=config.LORA_PATH,
        help=f"Where the finished LoRA adapter is saved. Default: {config.LORA_PATH!r} "
             "(config.LORA_PATH -- src/model_loader.py picks it up from there automatically "
             "with no further configuration).",
    )
    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULT_CHECKPOINT_DIR,
                         help="Trainer's own output_dir (logs only -- save_strategy is "
                              "'no', only the final adapter in --output-dir matters).")
    parser.add_argument("--epochs", type=int, default=config.LORA_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--grad-accum-steps", type=int, default=DEFAULT_GRAD_ACCUM_STEPS,
                         help="Effective batch size = this x per_device_train_batch_size (fixed at 1).")
    parser.add_argument("--no-4bit", action="store_true",
                         help="Load the base model in native fp16 instead of 4-bit QLoRA. "
                              "Needs considerably more VRAM -- off (i.e. 4-bit ON) by "
                              "default specifically for free-tier Kaggle GPUs (T4/P100, 16GB).")
    return parser


def main():
    args = build_arg_parser().parse_args()
    use_4bit = not args.no_4bit

    processor = AutoProcessor.from_pretrained(
        config.MODEL_PATH, min_pixels=config.MIN_PIXELS, max_pixels=config.MAX_PIXELS,
    )

    logger.info("Loading base model (4bit=%s)...", use_4bit)
    model = _load_base_model_for_training(use_4bit)

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        trainer_gradient_checkpointing = False  # already enabled by the line above
    else:
        model.enable_input_require_grads()
        trainer_gradient_checkpointing = True

    lora_config = LoraConfig(
        r=config.LORA_RANK,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    max_samples = args.max_samples if args.max_samples > 0 else None
    samples = load_omnimed_samples(args.data_root, args.dataset_names, max_samples, args.seed)
    logger.info("Loaded %d raw sample(s); resolving ground truth...", len(samples))

    training_samples = []
    skipped_gt = 0
    for sample in samples:
        gt_letter = _resolve_gt_letter(sample["gt_answer"], sample["choices"])
        if gt_letter is None:
            skipped_gt += 1
            continue
        sample = dict(sample, gt_letter=gt_letter)
        training_samples.append(sample)
    if skipped_gt:
        logger.info("%d sample(s) skipped: unresolvable ground truth.", skipped_gt)
    if not training_samples:
        logger.error("No samples with resolvable ground truth -- nothing to train on.")
        sys.exit(1)

    dataset = _PrebuiltExampleDataset(training_samples, processor)

    training_args = TrainingArguments(
        output_dir=args.checkpoint_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="no",
        fp16=True,
        gradient_checkpointing=trainer_gradient_checkpointing,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=_collate_single,
    )
    trainer.train()

    model.save_pretrained(args.output_dir)
    logger.info("LoRA adapter saved to %s", Path(args.output_dir).resolve())


if __name__ == "__main__":
    main()
