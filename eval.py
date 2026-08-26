"""
Competition evaluation entrypoint (Section 5 of medical_vqa_architecture.md).

Fixed invocation contract (per the competition harness):
    python eval.py input_dir <path_to_queries>

`<path_to_queries>` is expected to contain:
    dev_metadata.csv   -- columns: query_id, image, question,
                           choice_A, choice_B, choice_C, choice_D
                           (choice_C/choice_D empty for 2-choice/Yes-No rows)
    images/            -- referenced image files (.png/.jpg/.nii/.dcm) or
                           folders of volumetric/sliced data

Writes predictions.csv to the current working directory (the repository
root, when invoked as `python eval.py ...` from there) with columns:
    query_id, answer, inference_time
where `answer` is strictly a single character: "A", "B", "C", or "D".
Every row in dev_metadata.csv gets exactly one output row -- unresolvable
images or any predict() failure degrade to config.FALLBACK_ANSWER_LETTER
rather than skipping the row or crashing the run (Section 5.5), since a
missing row would presumably be scored as wrong by the harness anyway.

KNOWN LIMITATION: image resolution below only locates the file/folder on
disk; it does not decode .nii (NIfTI) or .dcm (DICOM) volumes, nor select
a slice from a folder of sliced data. predict()'s existing corrupt-image
guard (PIL can't open these formats) already prevents a crash on them --
such rows fall back to config.FALLBACK_ANSWER_LETTER, answered "blind"
with no real image analysis. That's a real accuracy gap, not a bug, and
building real volumetric/DICOM support is a separate, deliberately
undecided piece of work (see the chat discussion this file came out of).
"""

import csv
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

METADATA_FILENAME = "dev_metadata.csv"
IMAGES_SUBDIR = "Images"
OUTPUT_CSV_PATH = "predictions.csv"
CHOICE_COLUMNS = ["choice_A", "choice_B", "choice_C", "choice_D"]


def _resolve_image_path(input_dir: Path, image_value: str):
    """Resolves dev_metadata.csv's `image` cell to a local path. Its exact
    prefix convention (relative to input_dir root vs. the images/
    subfolder) isn't specified, so this tries a few plausible resolutions
    -- same defensive approach as src/evaluate_omnimed.py's
    _resolve_image_path(). Uses .exists() rather than .is_file() since a
    volumetric sample may be a directory of slices, not a single file."""
    if not image_value:
        return None
    candidates = [
        Path(image_value) if Path(image_value).is_absolute() else None,
        input_dir / image_value,
        input_dir / IMAGES_SUBDIR / image_value,
        input_dir / IMAGES_SUBDIR / Path(image_value).name,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return None


def load_queries(input_dir: Path):
    """Reads dev_metadata.csv and yields (query_id, image_path, question,
    choices) tuples. `choices` only contains the non-empty choice_*
    columns for that row, so 2-choice (Yes/No) and 4-choice (A-D) rows
    are handled uniformly -- whatever's actually populated in the CSV."""
    metadata_path = input_dir / METADATA_FILENAME
    if not metadata_path.is_file():
        raise FileNotFoundError(f"{METADATA_FILENAME} not found under {input_dir}")

    with open(metadata_path, newline="", encoding="latin1") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_id = row.get("query_id")
            image_value = (row.get("image") or "").strip()
            question = row.get("question", "")

            choices = {}
            for column in CHOICE_COLUMNS:
                letter = column.rsplit("_", 1)[-1]
                value = (row.get(column) or "").strip()
                if value:
                    choices[letter] = value

            image_path = _resolve_image_path(input_dir, image_value)
            yield query_id, image_path, question, choices


def main():
    if len(sys.argv) < 3:
        print("Usage: python eval.py input_dir <path_to_queries>", file=sys.stderr)
        sys.exit(1)
    # argv[1] is the harness's fixed literal label ("input_dir"); argv[2]
    # is the actual path. Not validated against the literal string --
    # only its position is relied on, so this still works if the harness
    # invokes it slightly differently.
    input_dir = Path(sys.argv[2])
    if not input_dir.is_dir():
        logger.error("input_dir %s does not exist or is not a directory", input_dir)
        sys.exit(1)

    # Deferred: importing src.predict triggers the full model load
    # (Section 4.1) -- keep argument validation above cheap and fast.
    from src import config
    from src.predict import predict

    queries = list(load_queries(input_dir))
    logger.info("Loaded %d quer%s from %s", len(queries), "y" if len(queries) == 1 else "ies", input_dir)

    with open(OUTPUT_CSV_PATH, "w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["query_id", "answer", "inference_time"])

        for query_id, image_path, question, choices in queries:
            t0 = time.perf_counter()
            if image_path is None:
                logger.warning("query_id=%r: image not found under %s -- using fallback answer", query_id, input_dir)
                answer = config.FALLBACK_ANSWER_LETTER
            else:
                try:
                    answer = predict(image_path, question, choices)
                except Exception as exc:  # noqa: BLE001 - predict() shouldn't raise, but this boundary must never crash the run
                    logger.warning("query_id=%r: predict() raised %s -- using fallback answer", query_id, exc)
                    answer = config.FALLBACK_ANSWER_LETTER
            elapsed = time.perf_counter() - t0
            writer.writerow([query_id, answer, f"{elapsed:.4f}"])

    logger.info("Wrote %d prediction(s) to %s", len(queries), Path(OUTPUT_CSV_PATH).resolve())


if __name__ == "__main__":
    main()
