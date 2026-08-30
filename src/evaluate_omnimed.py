"""
End-to-end accuracy evaluation of the full predict_with_diagnostics()
pipeline (Stages -1 through 6, see src/predict.py and
medical_vqa_architecture.md) against a designated OPEN-ACCESS subset of
OmniMedVQA, read directly from a local directory mirror (e.g. a Google
Drive folder mounted in Colab) -- NOT downloaded from the Hugging Face
Hub. Point --data-root at a local copy laid out like:
    {data_root}/QA_information/Open-access/{dataset_name}.json -- QA items
        whose images are actually available (see below)
    {data_root}/Images/{dataset_name}/{image_file_name}         -- the
        images those QA items reference
    {data_root}/QA_information/Restricted-access/{dataset_name}.json --
        QA items only, no images (never read by this script)
    Each QA item: {dataset, question_id, question_type, question, gt_answer,
                   image_path, option_A, option_B, option_C, option_D,
                   modality_type}

This script deliberately only ever reads QA_information/Open-access/ --
Restricted-access QA items reference images that typically aren't
distributed, so they can't be run through the pipeline at all here.
Pinning a caller-specified, fixed list of open-access dataset names
(--dataset-names) also keeps this evaluation slice separate and
reproducible from whatever gets used for future fine-tuning (Section 3.2
of medical_vqa_architecture.md) -- e.g. reserve some open-access dataset
names for eval-only and fine-tune on the rest.

Two fields in the raw JSON aren't universally pinned down across
OmniMedVQA dataset dumps, so this script handles both plausible
interpretations defensively rather than guessing:
  - `image_path`: not pinned down as relative-to-data-root vs. relative-
    to-the-dataset's-own-Images-subfolder vs. a bare filename -- see
    _resolve_image_path().
  - `gt_answer`: not pinned down as the letter (A/B/C/D) vs. the answer
    text (matching one of option_A..D) -- see _resolve_gt_letter().
A sample whose ground truth can't be resolved under either interpretation
is skipped and logged (a dataset-annotation issue, not something the
pipeline can be blindly scored against) -- this is the ONLY reason a
sample is ever skipped. A sample whose IMAGE fails to load (corrupt file,
unsupported volumetric format, unresolvable path, etc.) is NOT skipped:
predict_with_diagnostics() (src/predict.py) degrades it to a blind model
call on a neutral gray canvas instead, so it still gets scored (almost
certainly wrong, but that's an honest data point, not a silently dropped
one).

Robustness note: predict_with_diagnostics() runs the same Stage -1
volumetric/DICOM ingest (src/volume_loader.py) as the competition path
(src/predict.py's predict(), used by eval.py) -- a NIfTI volume, a DICOM
file/series, or any other unreadable image is decoded or gracefully
degraded, never crashes this script.

Outputs (both always written, matching eval.py's competition contract for
the first one):
    predictions.csv         -- query_id, answer, inference_time (the
                                exact 3-column format eval.py's harness
                                writes, so this script's output is
                                directly comparable to a real submission).
    eval_diagnostic_log.csv -- per-sample diagnostic sidecar: query_id,
                                answer, gold, correct, inference_time,
                                predicted_modality, predicted_intent,
                                n_choices, fallback_triggered,
                                fallback_reason, top1_logit, top2_logit,
                                logit_margin.

Usage (run from the repository root):
    python src/evaluate_omnimed.py
    python src/evaluate_omnimed.py --dataset-names ACRIMA "Adam Challenge" --max-samples 50
    python src/evaluate_omnimed.py --list-datasets
    python src/evaluate_omnimed.py --data-root /content/drive/MyDrive/OmniMedVQA --max-samples 100

NOTE on dataset names: this script does not hardcode a catalog of
OmniMedVQA's ~73 source datasets -- only "ACRIMA" (a small, confirmed
open-access glaucoma-fundus dataset) is used as the default, specifically
so the default max_samples=20 smoke test is fast and doesn't rely on
guessed dataset names. Run with --list-datasets to see which
QA_information/Open-access/<name>.json files actually exist under your
--data-root before choosing others.
"""

import argparse
import csv
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

# `python src/evaluate_omnimed.py` (run from the repo root, as intended --
# see the module docstring) only puts this file's own directory (src/) on
# sys.path, not the repo root -- so `from src...` imports below would
# otherwise fail with ModuleNotFoundError. Fix that before any such import.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402 - cheap import, no model weights loaded

from src import config  # noqa: E402 - cheap import, no model weights loaded

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = "/content/drive/MyDrive/OmniMedVQA"
DEFAULT_DATASET_NAMES = ["ACRIMA"]
DEFAULT_MAX_SAMPLES = 20
DEFAULT_OUTPUT_CSV = "predictions.csv"
DEFAULT_DIAGNOSTIC_CSV = "eval_diagnostic_log.csv"
DEFAULT_SEED = 42
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_CI = 0.95

_OPTION_FIELD_BY_LETTER = {letter: f"option_{letter}" for letter in config.CHOICE_LETTERS}


# ---------------------------------------------------------------------------
# Data acquisition -- purely local filesystem, no Hugging Face Hub access.
# ---------------------------------------------------------------------------
def _require_local_data_root(data_root) -> Path:
    """Validates --data-root points at an existing local directory. No
    download/fetch step of any kind -- the caller (e.g. a Colab notebook)
    is responsible for making the OmniMedVQA mirror available locally
    first (a mounted Google Drive folder, an already-extracted archive,
    etc.)."""
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"--data-root {root} does not exist or is not a directory. Point it at a local "
            "OmniMedVQA mirror containing QA_information/Open-access/<name>.json and "
            "Images/<name>/<image_file_name> (e.g. a mounted Google Drive folder in Colab)."
        )
    return root


def list_available_open_access_datasets(data_root):
    """Lists the Open-access QA JSON files actually present under
    {data_root}/QA_information/Open-access/ -- use this instead of
    guessing dataset names."""
    root = _require_local_data_root(data_root)
    qa_dir = root / "QA_information" / "Open-access"
    if not qa_dir.is_dir():
        raise FileNotFoundError(f"{qa_dir} does not exist under --data-root {root}")
    return sorted(p.stem for p in qa_dir.glob("*.json"))


def _load_qa_json(json_path: Path):
    """Reads one QA_information/Open-access/<name>.json file as a plain
    list of QA item dicts -- no Hugging Face `datasets` library or Hub
    access involved. Tolerates both a single JSON array (the common case)
    and JSON Lines (one JSON object per line), since OmniMedVQA dumps in
    the wild use either convention."""
    text = json_path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _resolve_image_path(root: Path, dataset_name: str, image_path: str):
    """Resolves a QA item's image_path to a local file under the fixed
    local mirror layout: {root}/Images/{dataset_name}/{image_file_name}
    -- the primary, authoritative resolution, using only the basename of
    image_path since some dataset dumps store it with a nested prefix
    (e.g. "<dataset_name>/xyz.png") that doesn't match this flat
    per-dataset Images/<name>/ directory. Two looser fallbacks are tried
    after that in case image_path is already root-relative or
    root/Images-relative, for robustness across dataset dump variations.
    Uses .exists() rather than .is_file() since a volumetric sample may be
    a directory of slices (see src/volume_loader.py), not a single file."""
    if not image_path:
        return None
    candidates = [
        root / "Images" / dataset_name / Path(image_path).name,
        root / image_path,
        root / "Images" / image_path,
    ]
    for candidate in candidates:
        if candidate.exists():
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


def load_omnimed_samples(data_root, dataset_names, max_samples, seed):
    """
    Loads QA items directly from the local OmniMedVQA mirror at
    --data-root (QA_information/Open-access/<name>.json for each
    requested dataset name -- see module docstring for the expected
    layout), read with plain json.load()/_load_qa_json() -- no Hugging
    Face `datasets` library or Hub access involved. Shuffles
    deterministically (--seed) across the combined pool, and truncates to
    max_samples (None = no limit). Samples whose image file can't be
    resolved locally are still kept (predict_with_diagnostics() handles
    that as a blind-fallback case, not a skip -- see module docstring);
    only unresolvable ground truth is dropped, in run_evaluation() below.

    Returns a list of dicts: {image_path, question, choices, gt_answer,
    modality_type, dataset, question_id}. `image_path` is None if it
    couldn't be resolved locally.
    """
    root = _require_local_data_root(data_root)

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
        records.extend((name, item) for item in _load_qa_json(json_path))

    rng = random.Random(seed)
    rng.shuffle(records)
    if max_samples:
        records = records[:max_samples]

    samples = []
    unresolved_image_count = 0
    for name, item in records:
        # Only non-empty option fields become choices -- mirrors eval.py's
        # load_queries() convention for its CSV choice_A..D columns, and
        # matters here specifically so n_choices (2 vs. 4) reflects what
        # was actually offered, not just which JSON keys happen to exist.
        choices = {
            letter: item.get(field)
            for letter, field in _OPTION_FIELD_BY_LETTER.items()
            if item.get(field) not in (None, "")
        }
        image_path = _resolve_image_path(root, name, item.get("image_path", ""))
        if image_path is None:
            unresolved_image_count += 1
            logger.warning(
                "question_id=%r (dataset=%r): image_path %r not found locally -- "
                "will be scored via predict_with_diagnostics()'s blind gray-canvas fallback.",
                item.get("question_id"), name, item.get("image_path"),
            )
        samples.append({
            "image_path": image_path,
            "question": item.get("question", ""),
            "choices": choices,
            "gt_answer": item.get("gt_answer"),
            "modality_type": item.get("modality_type", "unknown"),
            "dataset": name,
            "question_id": item.get("question_id"),
        })

    if unresolved_image_count:
        logger.info(
            "%d sample(s) have no locally resolvable image_path -- they will still be "
            "evaluated via the blind-fallback path, not skipped.", unresolved_image_count,
        )

    return samples


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def run_evaluation(samples):
    """Calls src.predict.predict_with_diagnostics(image, query, choices)
    once per sample, end-to-end (Stages -1 through 6). The ONLY reason a
    sample is skipped here is an unresolvable ground-truth letter (a
    dataset-annotation problem -- there is nothing to score against).
    Every other failure mode -- missing/corrupt/volumetric image, a Stage
    0-4 crash, an inference timeout -- is handled inside
    predict_with_diagnostics() itself (never raises, by design; the
    try/except below is a defensive backstop, not load-bearing) and still
    produces a scored row, with fallback_triggered/fallback_reason
    recording what happened.

    Returns (results, skipped_count)."""
    from src.predict import predict_with_diagnostics  # deferred: this import triggers the full
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

        # sample["image_path"] may be None (unresolvable locally) --
        # predict_with_diagnostics() treats that exactly like any other
        # unreadable image (Image.open(None) raises TypeError, caught by
        # its broad except around _load_input_image) and degrades to the
        # blind gray-canvas path.
        try:
            diag = predict_with_diagnostics(sample["image_path"], sample["question"], sample["choices"])
        except Exception as exc:  # noqa: BLE001 - predict_with_diagnostics() shouldn't raise, but this boundary must never crash the run
            logger.exception(
                "predict_with_diagnostics() raised unexpectedly for question_id=%r: %s",
                sample["question_id"], exc,
            )
            skipped += 1
            continue

        results.append({
            "query_id": sample["question_id"],
            "dataset": sample["dataset"],
            "modality_type": sample["modality_type"],
            "n_choices": len(sample["choices"]),
            "answer": diag["answer"],
            "gold": gt_letter,
            "correct": diag["answer"] == gt_letter,
            "inference_time": diag["inference_time"],
            "predicted_modality": diag["modality"],
            "predicted_intent": diag["track"],
            "fallback_triggered": diag["fallback_triggered"],
            "fallback_reason": diag["fallback_reason"] or "",
            "top1_logit": diag["top1_logit"],
            "top2_logit": diag["top2_logit"],
            "logit_margin": diag["logit_margin"],
        })

    return results, skipped


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _bootstrap_ci(correct_flags, n_resamples=BOOTSTRAP_RESAMPLES, ci=BOOTSTRAP_CI, seed=DEFAULT_SEED):
    """Percentile bootstrap CI on exact-match accuracy: resamples the
    per-item correct/incorrect vector with replacement n_resamples times
    and takes the (1-ci)/2 / (1+ci)/2 percentiles of the resampled means.
    Returns (lo_pct, hi_pct). Deterministic given `seed`, independent of
    the sample-selection --seed."""
    arr = np.asarray(correct_flags, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    resampled_means = arr[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(resampled_means, alpha * 100.0))
    hi = float(np.percentile(resampled_means, (1.0 - alpha) * 100.0))
    return lo * 100.0, hi * 100.0


def _stratify(results, key_fn):
    buckets = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        bucket = buckets[key_fn(r)]
        bucket["total"] += 1
        bucket["correct"] += int(r["correct"])
    return buckets


def _print_stratification(title, buckets):
    print(f"\n{title}")
    print("-" * 60)
    print(f"{'Group':<30}{'Correct/Total':<15}{'Accuracy':>10}")
    print("-" * 60)
    for key in sorted(buckets, key=lambda k: -buckets[k]["total"]):
        stats = buckets[key]
        acc = 100.0 * stats["correct"] / stats["total"] if stats["total"] else 0.0
        ratio = f"{stats['correct']}/{stats['total']}"
        print(f"{str(key):<30}{ratio:<15}{acc:>9.2f}%")


def print_report(results, skipped, total_requested):
    if not results:
        print("\nNo samples were successfully evaluated.")
        return

    total = len(results)
    correct_flags = [r["correct"] for r in results]
    correct = sum(correct_flags)
    accuracy = 100.0 * correct / total
    ci_lo, ci_hi = _bootstrap_ci(correct_flags)
    avg_time = sum(r["inference_time"] for r in results) / total
    fallback_count = sum(r["fallback_triggered"] for r in results)
    fallback_rate = 100.0 * fallback_count / total

    print("\n" + "=" * 60)
    print("OmniMedVQA End-to-End Evaluation")
    print("=" * 60)
    print(f"Requested samples:                        {total_requested}")
    print(f"Evaluated:                                 {total}")
    print(f"Skipped (unresolvable ground truth):       {skipped}")
    print(f"Overall Exact-Match accuracy:               {accuracy:.2f}% ({correct}/{total})")
    print(f"  {int(BOOTSTRAP_CI * 100)}% bootstrap CI ({BOOTSTRAP_RESAMPLES:,} resamples):  [{ci_lo:.2f}%, {ci_hi:.2f}%]")
    print(f"Fallback rate (blind/degraded answers):     {fallback_rate:.2f}% ({fallback_count}/{total})")
    print(f"Avg inference time:                         {avg_time:.4f}s")

    _print_stratification(
        "Accuracy by Modality (ground truth modality_type)",
        _stratify(results, lambda r: r["modality_type"]),
    )
    _print_stratification(
        "Accuracy by Number of Choices",
        _stratify(results, lambda r: r["n_choices"]),
    )
    print("=" * 60)


def _write_submission_csv(results, path):
    """query_id, answer, inference_time -- the exact 3-column format
    eval.py's competition harness writes, so this file is directly
    comparable to (or droppable in as) a real submission."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "answer", "inference_time"])
        for r in results:
            writer.writerow([r["query_id"], r["answer"], f"{r['inference_time']:.4f}"])


_DIAGNOSTIC_FIELDNAMES = [
    "query_id", "answer", "gold", "correct", "inference_time",
    "predicted_modality", "predicted_intent", "n_choices",
    "fallback_triggered", "fallback_reason",
    "top1_logit", "top2_logit", "logit_margin",
]


def _write_diagnostic_csv(results, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_DIAGNOSTIC_FIELDNAMES)
        writer.writeheader()
        for r in results:
            writer.writerow({name: r.get(name, "") for name in _DIAGNOSTIC_FIELDNAMES})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="End-to-end accuracy evaluation of src.predict.predict_with_diagnostics() "
                     "against a local OmniMedVQA Open-access mirror (no Hugging Face Hub access; "
                     "see --data-root)."
    )
    parser.add_argument(
        "--dataset-names", nargs="+", default=DEFAULT_DATASET_NAMES,
        help=f"One or more OmniMedVQA Open-access dataset names (e.g. ACRIMA), matching "
             f"{{data_root}}/QA_information/Open-access/<name>.json. Default: {DEFAULT_DATASET_NAMES}. "
             "Run with --list-datasets to see the names actually present under --data-root.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=DEFAULT_MAX_SAMPLES,
        help=f"Max samples to evaluate, 0 = no limit. Default: {DEFAULT_MAX_SAMPLES} (quick smoke test).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Shuffle seed for sample selection.")
    parser.add_argument(
        "--data-root", type=str, default=DEFAULT_DATA_ROOT,
        help=f"Local path to an OmniMedVQA mirror, containing QA_information/Open-access/ "
             f"and Images/ (e.g. a Google Drive folder mounted in Colab at "
             f"/content/drive/MyDrive/...). No download step -- this directory must already "
             f"exist. Default: {DEFAULT_DATA_ROOT!r}.",
    )
    parser.add_argument(
        "--output-csv", type=str, default=DEFAULT_OUTPUT_CSV,
        help=f"Path to write the standard submission-format CSV (query_id, answer, "
             f"inference_time), relative to the current working directory. Default: "
             f"{DEFAULT_OUTPUT_CSV!r}. Pass an empty string to skip writing it.",
    )
    parser.add_argument(
        "--diagnostic-csv", type=str, default=DEFAULT_DIAGNOSTIC_CSV,
        help=f"Path to write the rich per-sample diagnostic sidecar CSV. Default: "
             f"{DEFAULT_DIAGNOSTIC_CSV!r}. Pass an empty string to skip writing it.",
    )
    parser.add_argument(
        "--list-datasets", action="store_true",
        help="List Open-access dataset names present under --data-root and exit "
             "(no evaluation, no model load).",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()

    if args.list_datasets:
        names = list_available_open_access_datasets(args.data_root)
        print(f"{len(names)} Open-access OmniMedVQA dataset(s) available under {args.data_root}:")
        for name in names:
            print(f"  - {name}")
        return

    max_samples = args.max_samples if args.max_samples > 0 else None
    samples = load_omnimed_samples(args.data_root, args.dataset_names, max_samples, args.seed)
    if not samples:
        logger.error("No evaluable samples found -- nothing to run.")
        sys.exit(1)

    logger.info(
        "Loaded %d sample(s); loading the predict pipeline (this triggers the full model load)...",
        len(samples),
    )
    results, skipped = run_evaluation(samples)
    print_report(results, skipped, total_requested=len(samples))

    if args.output_csv:
        _write_submission_csv(results, args.output_csv)
        logger.info("Submission-format predictions written to %s", Path(args.output_csv).resolve())
    if args.diagnostic_csv:
        _write_diagnostic_csv(results, args.diagnostic_csv)
        logger.info("Diagnostic sidecar log written to %s", Path(args.diagnostic_csv).resolve())


if __name__ == "__main__":
    main()
