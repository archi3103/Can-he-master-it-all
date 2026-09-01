"""
Converts an external (non-OmniMedVQA) volumetric/2D dataset into the exact
local-mirror shape src/evaluate_omnimed.py and src/train_lora.py already
read (see that module's docstring for the layout):
    {output_root}/QA_information/Open-access/{dataset_name}.json
    {output_root}/Images/{dataset_name}/{image_file_name}

This is deliberately the ONLY new piece of code involved in adding an
external dataset -- load_omnimed_samples()/_resolve_gt_letter() (already
tested, already used by both train_lora.py and evaluate_omnimed.py) work
unchanged on whatever this script produces, since they only care about
the directory shape and JSON schema, not where the data actually came
from. Every adapter below writes `image_path` as an ABSOLUTE path to the
original raw file rather than copying multi-GB volumes into a new
location -- _resolve_image_path() in evaluate_omnimed.py resolves an
absolute image_path correctly (root / absolute_path == absolute_path,
per pathlib), so this costs zero extra disk I/O or wait time.

Each adapter produces exactly one QA item per image/volume it emits, with
a deterministic train/eval split by case (not by image -- so, e.g., all
slices/sequences from the same patient stay on the same side of the
split) via --eval-fraction/--seed, written as two separate dataset names
("{name}_train" / "{name}_eval") so --dataset-names can select either
side without any filtering logic downstream.

Supported --dataset values (see each adapter's docstring for the exact
raw directory layout it expects):
    mosmed   -- MosMedData chest CT severity (CT-0..CT-4), NIfTI
    brainmri -- Medical Segmentation Decathlon Task01_BrainTumour, NIfTI
    busi     -- BUSI breast ultrasound (benign/malignant/normal), PNG

Usage (run once per dataset, from the repository root):
    python src/prepare_external_dataset.py --dataset mosmed \
        --raw-root /path/to/MosMedData --output-root /path/to/OmniMedVQA
    python src/prepare_external_dataset.py --dataset brainmri \
        --raw-root /path/to/Task01_BrainTumour --output-root /path/to/OmniMedVQA
    python src/prepare_external_dataset.py --dataset busi \
        --raw-root "/path/to/Dataset_BUSI_with_GT" --output-root /path/to/OmniMedVQA

After running all three, --dataset-names for train_lora.py/evaluate_omnimed.py
gains six new names: MosMedData_train, MosMedData_eval, BrainMRI_train,
BrainMRI_eval, BUSI_train, BUSI_eval (verify with --list-datasets, same as
any other dataset in this local mirror).
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_EVAL_FRACTION = 0.2
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Shared split + write helpers
# ---------------------------------------------------------------------------
def _split_cases(case_ids: list, eval_fraction: float, seed: int) -> tuple:
    """Deterministic, case-level (not image-level) train/eval split --
    every QA item derived from the same case lands on the same side, so
    a model can never see a case at train time and its held-out twin at
    eval time. Returns (train_case_ids, eval_case_ids)."""
    ids = sorted(case_ids)  # sort first so shuffle is reproducible regardless of filesystem iteration order
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_eval = max(1, int(round(len(ids) * eval_fraction))) if ids else 0
    return ids[n_eval:], ids[:n_eval]


def _write_split(output_root: Path, dataset_name: str, train_items: list, eval_items: list):
    qa_dir = output_root / "QA_information" / "Open-access"
    qa_dir.mkdir(parents=True, exist_ok=True)
    for suffix, items in (("_train", train_items), ("_eval", eval_items)):
        path = qa_dir / f"{dataset_name}{suffix}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)
        logger.info("Wrote %d item(s) to %s", len(items), path)


def _qa_item(question_id, question, options: dict, gt_answer_letter, modality_type, image_path, dataset):
    """options: dict like {"A": "...", "B": "...", ...}. gt_answer_letter
    is stored as the option TEXT (matching OmniMedVQA's own gt_answer
    convention, confirmed against its README example) -- both letter and
    text forms are handled by _resolve_gt_letter() regardless, so this
    just mirrors the real dataset's own schema as closely as possible."""
    item = {
        "dataset": dataset,
        "question_id": question_id,
        "question_type": "External-Prepared",
        "question": question,
        "gt_answer": options[gt_answer_letter],
        "image_path": str(image_path),
        "modality_type": modality_type,
    }
    for letter, text in options.items():
        item[f"option_{letter}"] = text
    return item


# ---------------------------------------------------------------------------
# Adapter: MosMedData (chest CT, 5-class severity, NIfTI)
# ---------------------------------------------------------------------------
_MOSMED_SEVERITY_OPTIONS = {
    "A": "No significant involvement (normal lung)",
    "B": "Mild involvement (ground-glass opacity, under 25% of lung parenchyma)",
    "C": "Moderate involvement (ground-glass opacity, 25-50% of lung parenchyma)",
    "D": "Severe involvement (opacity and consolidation, over 50% of lung parenchyma)",
}
_MOSMED_CATEGORY_TO_LETTER = {"CT-0": "A", "CT-1": "B", "CT-2": "C", "CT-3": "D", "CT-4": "D"}
_MOSMED_QUESTION = (
    "What is the severity of lung involvement (ground-glass opacification/consolidation) "
    "shown in this chest CT scan?"
)


def _prepare_mosmed(raw_root: Path, output_root: Path, eval_fraction: float, seed: int):
    """Expects the common MosMedData layout: subfolders (anywhere under
    raw_root, matched by name, not position) named CT-0 .. CT-4, each
    containing that severity class's .nii/.nii.gz studies -- e.g.
    raw_root/studies/CT-1/study_0001.nii.gz. Tolerates either the
    original nested layout or a flattened one, since only the immediate
    parent directory name is used to infer severity."""
    cases = []
    for path in raw_root.rglob("*"):
        if not path.is_file():
            continue
        name_lower = path.name.lower()
        if not (name_lower.endswith(".nii") or name_lower.endswith(".nii.gz")):
            continue
        category = path.parent.name.upper().replace("_", "-")
        if category not in _MOSMED_CATEGORY_TO_LETTER:
            continue
        cases.append((path, _MOSMED_CATEGORY_TO_LETTER[category]))

    if not cases:
        raise FileNotFoundError(
            f"No CT-0..CT-4-categorized .nii/.nii.gz files found under {raw_root} -- "
            "check --raw-root points at the extracted MosMedData folder."
        )
    logger.info("Found %d MosMedData case(s).", len(cases))

    case_ids = [str(p) for p, _ in cases]
    train_ids, eval_ids = _split_cases(case_ids, eval_fraction, seed)
    train_set, eval_set = set(train_ids), set(eval_ids)

    train_items, eval_items = [], []
    for i, (path, letter) in enumerate(cases):
        item = _qa_item(
            question_id=f"MosMedData_{i:05d}",
            question=_MOSMED_QUESTION,
            options=_MOSMED_SEVERITY_OPTIONS,
            gt_answer_letter=letter,
            modality_type="CT(Computed Tomography)",
            image_path=path.resolve(),
            dataset="MosMedData",
        )
        (train_items if str(path) in train_set else eval_items).append(item)

    _write_split(output_root, "MosMedData", train_items, eval_items)


# ---------------------------------------------------------------------------
# Adapter: Medical Segmentation Decathlon Task01_BrainTumour (brain MRI, NIfTI)
# ---------------------------------------------------------------------------
_BRAINMRI_OPTIONS = {"A": "Brain", "B": "Chest/thorax", "C": "Abdomen", "D": "Pelvis"}
_BRAINMRI_QUESTION = "Which anatomical region is primarily depicted in this MRI scan?"


def _prepare_brainmri(raw_root: Path, output_root: Path, eval_fraction: float, seed: int):
    """Expects EITHER of two layouts, tried in order:

    Layout A -- official Medical Segmentation Decathlon Task01_BrainTumour:
    raw_root/imagesTr/*.nii.gz, one 4D file per case (channel 0 = FLAIR --
    src/volume_loader.py's _load_nifti() already reduces any 4D NIfTI to
    its first channel automatically, so no preprocessing/splitting of the
    4 sequences is needed here). Excludes imagesTs/ (no ground truth
    available for the test split).

    Layout B -- per-patient BraTS-style mirrors (e.g. several Kaggle
    "BraTS20" repackagings): one file per MRI sequence per patient,
    typically PLAIN .nii (not gzipped), nested in an arbitrarily deep
    per-patient subfolder. Searched recursively; exactly one file is kept
    per patient folder (prefers the file with "flair" in its name to
    match Layout A's channel-0 convention, otherwise the first remaining
    file) so the case-level split stays meaningful -- taking all 4
    sequences per patient would inflate the case count without adding
    distinct patients. Segmentation mask files (name contains "seg") are
    always excluded -- they're a label map, not a scan.

    In both layouts, every case in this dataset is a brain scan, so the
    anatomy-ID answer is constant -- a real, correctly-labeled question,
    just not a within-dataset-diverse one."""
    images_dir = raw_root / "imagesTr"
    if not images_dir.is_dir():
        images_dir = raw_root  # tolerate raw_root already pointing at imagesTr/
    cases = sorted(
        p for p in images_dir.glob("*.nii.gz")
        if not p.name.startswith(".") and not p.name.startswith("_")  # skip macOS/hidden junk files
    )

    if not cases:
        by_case_dir = {}
        for p in raw_root.rglob("*"):
            if not p.is_file() or p.name.startswith("."):
                continue
            name_lower = p.name.lower()
            if not (name_lower.endswith(".nii") or name_lower.endswith(".nii.gz")):
                continue
            if "seg" in name_lower:
                continue
            by_case_dir.setdefault(p.parent, []).append(p)
        for case_dir, files in sorted(by_case_dir.items()):
            flair = [f for f in files if "flair" in f.name.lower()]
            cases.append(sorted(flair or files)[0])
        cases = sorted(cases)

    if not cases:
        raise FileNotFoundError(
            f"No .nii/.nii.gz brain MRI files found under {raw_root} -- check --raw-root "
            "points at the extracted Task01_BrainTumour folder (or its imagesTr/ subfolder), "
            "or a per-patient BraTS-style mirror."
        )
    logger.info("Found %d BrainMRI case(s).", len(cases))

    case_ids = [str(p) for p in cases]
    train_ids, eval_ids = _split_cases(case_ids, eval_fraction, seed)
    train_set = set(train_ids)

    train_items, eval_items = [], []
    for i, path in enumerate(cases):
        item = _qa_item(
            question_id=f"BrainMRI_{i:05d}",
            question=_BRAINMRI_QUESTION,
            options=_BRAINMRI_OPTIONS,
            gt_answer_letter="A",
            modality_type="MRI(Magnetic Resonance Imaging)",
            image_path=path.resolve(),
            dataset="BrainMRI",
        )
        (train_items if str(path) in train_set else eval_items).append(item)

    _write_split(output_root, "BrainMRI", train_items, eval_items)


# ---------------------------------------------------------------------------
# Adapter: BUSI (breast ultrasound, benign/malignant/normal, PNG)
# ---------------------------------------------------------------------------
_BUSI_OPTIONS = {"A": "Benign lesion", "B": "Malignant lesion", "C": "Normal tissue, no lesion"}
_BUSI_CATEGORY_TO_LETTER = {"benign": "A", "malignant": "B", "normal": "C"}
_BUSI_QUESTION = "What does this breast ultrasound image most likely show?"


def _prepare_busi(raw_root: Path, output_root: Path, eval_fraction: float, seed: int):
    """Expects the standard BUSI layout: raw_root/{benign,malignant,normal}/*.png
    (case-insensitive folder names), each image paired with a same-stem
    "..._mask.png" segmentation file -- mask files are excluded by name."""
    cases = []
    for category, letter in _BUSI_CATEGORY_TO_LETTER.items():
        category_dir = None
        for candidate in raw_root.iterdir():
            if candidate.is_dir() and candidate.name.lower() == category:
                category_dir = candidate
                break
        if category_dir is None:
            logger.warning("No %r subfolder found under %s -- skipping that category.", category, raw_root)
            continue
        for path in sorted(category_dir.glob("*.png")):
            if "mask" in path.stem.lower():
                continue
            cases.append((path, letter))

    if not cases:
        raise FileNotFoundError(
            f"No benign/malignant/normal PNG files found under {raw_root} -- check "
            "--raw-root points at the extracted 'Dataset_BUSI_with_GT' folder."
        )
    logger.info("Found %d BUSI case(s).", len(cases))

    case_ids = [str(p) for p, _ in cases]
    train_ids, eval_ids = _split_cases(case_ids, eval_fraction, seed)
    train_set = set(train_ids)

    train_items, eval_items = [], []
    for i, (path, letter) in enumerate(cases):
        item = _qa_item(
            question_id=f"BUSI_{i:05d}",
            question=_BUSI_QUESTION,
            options=_BUSI_OPTIONS,
            gt_answer_letter=letter,
            modality_type="US(Ultrasound)",
            image_path=path.resolve(),
            dataset="BUSI",
        )
        (train_items if str(path) in train_set else eval_items).append(item)

    _write_split(output_root, "BUSI", train_items, eval_items)


_ADAPTERS = {"mosmed": _prepare_mosmed, "brainmri": _prepare_brainmri, "busi": _prepare_busi}


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(_ADAPTERS.keys()))
    parser.add_argument("--raw-root", required=True, type=str, help="Path to the extracted raw dataset.")
    parser.add_argument(
        "--output-root", required=True, type=str,
        help="Local OmniMedVQA mirror root (the same --data-root you pass to "
             "evaluate_omnimed.py/train_lora.py) -- QA_information/Open-access/*.json "
             "gets written here. Image files are referenced by absolute path, not copied.",
    )
    parser.add_argument("--eval-fraction", type=float, default=DEFAULT_EVAL_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main():
    args = build_arg_parser().parse_args()
    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    if not raw_root.is_dir():
        logger.error("--raw-root %s does not exist or is not a directory.", raw_root)
        sys.exit(1)
    output_root.mkdir(parents=True, exist_ok=True)

    _ADAPTERS[args.dataset](raw_root, output_root, args.eval_fraction, args.seed)


if __name__ == "__main__":
    main()
