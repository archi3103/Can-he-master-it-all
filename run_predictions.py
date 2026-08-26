"""
Harness entrypoint / CSV writer (Section 5.3 of medical_vqa_architecture.md
-- Stage 7 in the architecture diagram).

Loops the evaluation dataset, calls predict() once per (image, query,
choices) triple, times each call, and writes query_id, answer, and
inference_time to predictions.csv.

`load_eval_dataset` is intentionally not implemented here -- per the doc,
it is "user-provided or harness-provided": the competition harness (or a
local `data_loader.py`) supplies the actual dataset iterator. This file
only depends on it yielding (query_id, image, query, choices) tuples.
"""

import csv
import logging
import time

from src.predict import predict
from data_loader import load_eval_dataset  # user-provided or harness-provided

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

OUTPUT_CSV_PATH = "predictions.csv"


def main():
    dataset = load_eval_dataset()  # yields (query_id, image, query, choices)

    with open(OUTPUT_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "answer", "inference_time"])

        for query_id, image, query, choices in dataset:
            t0 = time.perf_counter()
            answer = predict(image, query, choices)
            elapsed = time.perf_counter() - t0
            writer.writerow([query_id, answer, f"{elapsed:.4f}"])


if __name__ == "__main__":
    main()
