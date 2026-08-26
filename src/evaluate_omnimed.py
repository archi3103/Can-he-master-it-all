"""
End-to-end accuracy evaluation of the full predict() pipeline (Stages 0-6,
see src/predict.py and medical_vqa_architecture.md) against a designated
OPEN-ACCESS subset of OmniMedVQA:
    https://huggingface.co/datasets/foreverbeliever/OmniMedVQA

Dataset structure (verified against the dataset's README):
    Images/<DatasetName>/...                             -- open-access images only
    QA_information/Open-access/<DatasetName>.json         -- QA items whose images
                                                               are actually available
    QA_information/Restricted-access/<DatasetName>.json   -- QA items only, no images
    Each QA item: {dataset, question_id, question_type, question, gt_answer,
                   image_path, option_A, option_B, option_C, option_D,
                   modality_type}

This script deliberately only ever reads QA_information/Open-access/ --
Restricted-access QA items reference images not distributed in this repo,
so they can't be run through predict() at all here. Pinning a caller-
specified, fixed list of open-access dataset names (--dataset-names) also
keeps this evaluation slice separate and reproducible from whatever gets
used for future fine-tuning (Section 3.2 of medical_vqa_architecture.md)
-- e.g. reserve some open-access dataset names for eval-only and
fine-tune on the rest.

Two fields in the raw JSON are documented only loosely by the dataset
README (no example row was independently verified), so this script
handles both plausible interpretations defensively rather than guessing:
  - `image_path`: not pinned down as relative-to-repo-root vs.
    relative-to-the-dataset's-own-Images-subfolder -- see
    _resolve_image_path().
  - `gt_answer`: not pinned down as the letter (A/B/C/D) vs. the answer
    text (matching one of option_A..D) -- see _resolve_gt_letter().
A sample that can't be resolved under either interpretation is skipped
and logged, per requirement #4, rather than silently mis-scored.

Usage (run from the repository root):
    python src/evaluate_omnimed.py
    python src/evaluate_omnimed.py --dataset-names ACRIMA "Adam Challenge" --max-samples 50
    python src/evaluate_omnimed.py --list-datasets
    python src/evaluate_omnimed.py --data-root /path/to/local/OmniMedVQA --max-samples 100

NOTE on dataset names: this script does not hardcode a catalog of
OmniMedVQA's ~73 source datasets -- only "ACRIMA" (a small, confirmed
open-access glaucoma-fundus dataset) is used as the default, specifically
so the default max_samples=20 smoke test is fast and doesn't rely on
guessed dataset names. Run with --list-datasets to fetch the real,
current list from the Hub before choosing others.
"""

import argparse
import csv
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

# `python src/evaluate_omnimed.py` (run from the repo root, as intended --
# see the module docstring) only puts this file's own directory (src/) on
# sys.path, not the repo root -- so `from src...` imports below would
# otherwise fail with ModuleNotFoundError. Fix that before any such import.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from PIL import Image, UnidentifiedImageError  # noqa: E402

from src import config  # noqa: E402 - cheap import, no model weights loaded

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ID = "foreverbeliever/OmniMedVQA"
REPO_TYPE = "dataset"
DEFAULT_DATASET_NAMES = ["ACRIMA"]
DEFAULT_MAX_SAMPLES = 20
DEFAULT_OUTPUT_CSV = "predictions.csv"
DEFAULT_SEED = 42

_OPTION_FIELD_BY_LETTER = {letter: f"option_{letter}" for letter in config.CHOICE_LETTERS}


# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------
def _ensure_local_data(data_root, dataset_names, cache_dir):
    """Ensures QA_information/Open-access/<name>.json and Images/<name>/ are
    present locally for each requested dataset name, downloading only that
    minimal slice from the Hub via huggingface_hub.snapshot_download if
    data_root wasn't supplied. Returns the local root Path to read from."""
    if data_root is not None:
        root = Path(data_root)
        if not root.exists():
            raise FileNotFoundError(f"--data-root {root} does not exist")
        return root

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download OmniMedVQA from the Hub "
            "(see requirements.txt); install it, or pass --data-root to point "
            "at an already-downloaded local copy instead."
        ) from exc

    allow_patterns = [f"QA_information/Open-access/{name}.json" for name in dataset_names]
    allow_patterns += [f"Images/{name}/**" for name in dataset_names]

    logger.info(
        "Fetching %d dataset(s) from %s (only the requested slice; cached after the first run)...",
        len(dataset_names), REPO_ID,
    )
    local_dir = snapshot_download(
        repo_id=REPO_ID, repo_type=REPO_TYPE, allow_patterns=allow_patterns, cache_dir=cache_dir,
    )
    return Path(local_dir)


def list_available_open_access_datasets():
    """Fetches the real, current list of Open-access QA JSON files from the
    Hub -- use this instead of guessing dataset names."""
    from huggingface_hub import list_repo_files

    files = list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    prefix = "QA_information/Open-access/"
    return sorted(
        f[len(prefix):-len(".json")] for f in files if f.startswith(prefix) and f.endswith(".json")
    )


def _resolve_image_path(root: Path, dataset_name: str, image_path: str):
    """`image_path`'s exact prefix convention isn't pinned down by the
    dataset README, so this tries a few plausible resolutions in order and
    returns the first that actually exists locally, or None."""
    if not image_path:
        return None
    candidates = [
        root / image_path,
        root / "Images" / image_path,
        root / "Images" / dataset_name / image_path,
        root / "Images" / dataset_name / Path(image_path).name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _resolve_gt_letter(gt_answer, choices: dict):
    """`gt_answer`'s format isn't pinned down by the dataset README either
    -- handles both plausible cases: gt_answer is itself a letter (A/B/C/D),
    or gt_answer is the answer *text*, matched (case/whitespace-normalized)
    against the option texts to find which letter it corresponds to.
    Returns None if neither resolves."""
    if gt_answer is None:
        return None
    normalized = str(gt_answer).strip()
    if normalized.upper() in config.CHOICE_LETTERS:
        return normalized.upper()

    normalized_lower = normalized.lower()
    for letter, text in choices.items():
        if text is not None and str(text).strip().lower() == normalized_lower:
            return letter
    return None


def load_eval_samples(data_root, dataset_names, max_samples, seed, cache_dir):
    """
    Loads QA items from QA_information/Open-access/<name>.json for each
    requested dataset name via the Hugging Face `datasets` library,
    shuffles deterministically (--seed) across the combined pool, and
    truncates to max_samples (None = no limit). Samples whose image file
    can't be resolved locally are dropped here (logged), not counted as
    evaluated.

    Returns a list of dicts: {image_path, question, choices, gt_answer,
    modality_type, dataset, question_id}.
    """
    import datasets as hf_datasets

    root = _ensure_local_data(data_root, dataset_names, cache_dir)

    json_paths = []
    for name in dataset_names:
        json_path = root / "QA_information" / "Open-access" / f"{name}.json"
        if not json_path.is_file():
            logger.warning("No Open-access QA file for dataset %r at %s -- skipping it.", name, json_path)
            continue
        json_paths.append((name, json_path))

    if not json_paths:
        raise FileNotFoundError(
            f"None of the requested dataset names {dataset_names!r} have an Open-access "
            f"QA_information/*.json file under {root}. Try --list-datasets to see what's "
            "actually available."
        )

    records = []
    for name, json_path in json_paths:
        ds = hf_datasets.load_dataset("json", data_files=str(json_path), split="train")
        records.extend((name, item) for item in ds)

    rng = random.Random(seed)
    rng.shuffle(records)
    if max_samples:
        records = records[:max_samples]

    samples = []
    skipped_missing_image = 0
    for name, item in records:
        image_path = _resolve_image_path(root, name, item.get("image_path", ""))
        if image_path is None:
            skipped_missing_image += 1
            logger.warning(
                "Skipping question_id=%r (dataset=%r): image_path %r not found locally.",
                item.get("question_id"), name, item.get("image_path"),
            )
            continue

        choices = {letter: item.get(field) for letter, field in _OPTION_FIELD_BY_LETTER.items()}
        samples.append({
            "image_path": image_path,
            "question": item.get("question", ""),
            "choices": choices,
            "gt_answer": item.get("gt_answer"),
            "modality_type": item.get("modality_type", "unknown"),
            "dataset": name,
            "question_id": item.get("question_id"),
        })

    if skipped_missing_image:
        logger.info("Skipped %d sample(s) with unresolvable image paths before evaluation.", skipped_missing_image)

    return samples


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def run_evaluation(samples):
    """Calls src.predict.predict(image, query, choices) once per sample,
    end-to-end (Stages 0-6). Corrupt/missing images, unparseable ground
    truth, and any exception from predict() itself are all caught here and
    skipped rather than crashing the run (requirement #4) -- predict() is
    documented to never raise, but this script doesn't rely on that blindly.

    Returns (results, skipped_count)."""
    from src.predict import predict  # deferred: this import triggers the full
    # model/weights load (Section 4.1) -- keep --list-datasets/--help cheap.

    results = []
    skipped = 0
    for sample in samples:
        gt_letter = _resolve_gt_letter(sample["gt_answer"], sample["choices"])
        if gt_letter is None:
            logger.warning(
                "Skipping question_id=%r: gt_answer %r doesn't match a letter or any option text.",
                sample["question_id"], sample["gt_answer"],
            )
            skipped += 1
            continue

        try:
            image = Image.open(sample["image_path"])
            image.load()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            logger.warning(
                "Skipping question_id=%r: corrupt/unreadable image %s (%s)",
                sample["question_id"], sample["image_path"], exc,
            )
            skipped += 1
            continue

        try:
            t0 = time.perf_counter()
            predicted = predict(image, sample["question"], sample["choices"])
            elapsed = time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping question_id=%r: predict() raised %s", sample["question_id"], exc)
            skipped += 1
            continue

        results.append({
            "question_id": sample["question_id"],
            "dataset": sample["dataset"],
            "modality_type": sample["modality_type"],
            "predicted": predicted,
            "gt_letter": gt_letter,
            "correct": predicted == gt_letter,
            "inference_time": elapsed,
        })

    return results, skipped


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_report(results, skipped, total_requested):
    if not results:
        print("\nNo samples were successfully evaluated.")
        return

    total = len(results)
    correct = sum(r["correct"] for r in results)
    accuracy = 100.0 * correct / total
    avg_time = sum(r["inference_time"] for r in results) / total

    print("\n" + "=" * 60)
    print("OmniMedVQA End-to-End Evaluation")
    print("=" * 60)
    print(f"Requested samples:                      {total_requested}")
    print(f"Evaluated:                               {total}")
    print(f"Skipped (corrupt/missing/unresolvable):  {skipped}")
    print(f"Overall accuracy:                        {accuracy:.2f}% ({correct}/{total})")
    print(f"Avg inference time:                      {avg_time:.4f}s")

    per_modality = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        bucket = per_modality[r["modality_type"]]
        bucket["total"] += 1
        bucket["correct"] += int(r["correct"])

    print("\nPer-Modality Performance")
    print("-" * 60)
    print(f"{'Modality':<35}{'Correct/Total':<15}{'Accuracy':>10}")
    print("-" * 60)
    for modality in sorted(per_modality, key=lambda m: -per_modality[m]["total"]):
        stats = per_modality[modality]
        acc = 100.0 * stats["correct"] / stats["total"] if stats["total"] else 0.0
        ratio = f"{stats['correct']}/{stats['total']}"
        print(f"{modality:<35}{ratio:<15}{acc:>9.2f}%")
    print("=" * 60)


def _write_csv(results, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["question_id", "dataset", "modality_type", "predicted", "gt_letter", "correct", "inference_time"]
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="End-to-end accuracy evaluation of src.predict.predict() against the "
                     "Open-access subset of OmniMedVQA (foreverbeliever/OmniMedVQA on the "
                     "Hugging Face Hub)."
    )
    parser.add_argument(
        "--dataset-names", nargs="+", default=DEFAULT_DATASET_NAMES,
        help=f"One or more OmniMedVQA Open-access dataset names (e.g. ACRIMA), matching "
             f"QA_information/Open-access/<name>.json in the repo. Default: {DEFAULT_DATASET_NAMES}. "
             "Run with --list-datasets to see the real available names.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=DEFAULT_MAX_SAMPLES,
        help=f"Max samples to evaluate, 0 = no limit. Default: {DEFAULT_MAX_SAMPLES} (quick smoke test).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Shuffle seed for sample selection.")
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Local path to an already-downloaded OmniMedVQA repo copy (containing Images/ "
             "and QA_information/) -- skips the Hub download entirely.",
    )
    parser.add_argument("--cache-dir", type=str, default=None, help="huggingface_hub cache directory override.")
    parser.add_argument(
        "--output-csv", type=str, default=DEFAULT_OUTPUT_CSV,
        help=f"Path to write a per-sample results CSV, relative to the current working "
             f"directory. Default: {DEFAULT_OUTPUT_CSV!r} (always written, matching "
             "run_predictions.py's convention). Pass an empty string to skip writing one.",
    )
    parser.add_argument(
        "--list-datasets", action="store_true",
        help="List available Open-access dataset names from the Hub and exit (no evaluation, no model load).",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()

    if args.list_datasets:
        names = list_available_open_access_datasets()
        print(f"{len(names)} Open-access OmniMedVQA dataset(s) available:")
        for name in names:
            print(f"  - {name}")
        return

    max_samples = args.max_samples if args.max_samples > 0 else None
    samples = load_eval_samples(args.data_root, args.dataset_names, max_samples, args.seed, args.cache_dir)
    if not samples:
        logger.error("No evaluable samples found -- nothing to run.")
        sys.exit(1)

    logger.info("Loaded %d sample(s); loading the predict() pipeline (this triggers the full model load)...", len(samples))
    results, skipped = run_evaluation(samples)
    print_report(results, skipped, total_requested=len(samples))

    if args.output_csv:
        _write_csv(results, args.output_csv)
        logger.info("Per-sample results written to %s", Path(args.output_csv).resolve())


if __name__ == "__main__":
    main()
