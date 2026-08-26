"""
Stage 3: Query-Intent / Track Classifier (Section 2.1 of
medical_vqa_architecture.md).

Classifies each incoming question into one of five clinical reasoning
tracks -- diagnostic, spatial, severity, modality_id, differential -- via:

    1. A fast regex/keyword pass (covers ~70-80% of queries, <1ms).
    2. A fallback: cosine similarity between a sentence-embedding of the
       query (a small frozen CPU encoder, all-MiniLM-L6-v2, ~22M params)
       and precomputed per-track centroid embeddings -- used only when
       the regex pass finds no match, so the CPU-bound embedding call
       (and its CPU<->GPU sync stall risk, Section 4.4) is skipped on
       the large majority of queries the regex pass already resolves.

Per the "global pre-loading" rule (Section 4.1: "model weights,
processor/tokenizer, LoRA adapters, router model, sentence-embedding
encoder -- must be instantiated once at module import time"), both the
sentence-embedding encoder and the track-centroid embeddings it produces
are computed once here, at import time, never inside detect_track().
"""

import re

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - surfaced at import time
    raise ImportError(
        "router_intent.py requires sentence-transformers (see requirements.txt)"
    ) from exc

from src import config

# ---------------------------------------------------------------------------
# Stage 3a: regex/keyword pass (Section 2.1 table)
# ---------------------------------------------------------------------------
TRACK_TRIGGER_PHRASES = {
    config.TRACK_DIAGNOSTIC: [
        "what is the diagnosis",
        "which condition",
        "most likely disease",
        "most likely diagnosis",
        "what is the most likely",
    ],
    config.TRACK_SPATIAL: [
        "which lobe",
        "located in",
        "left or right",
        "which quadrant",
        "which side",
        "anatomical location",
    ],
    config.TRACK_SEVERITY: [
        "grade",
        "stage",
        "severity",
        "bi-rads",
        "birads",
        "how advanced",
        "gleason",
        "tnm",
        "fitzpatrick",
        "isup",
    ],
    config.TRACK_MODALITY_ID: [
        "what imaging modality",
        "which sequence",
        "contrast used",
        "imaging technique",
        "which modality",
        "what type of scan",
    ],
    config.TRACK_DIFFERENTIAL: [
        "which of the following is not",
        "best explains",
        "most consistent with",
        "rule out",
        "differential diagnosis",
    ],
}


def _compile_track_patterns() -> dict:
    return {
        track: [re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE) for phrase in phrases]
        for track, phrases in TRACK_TRIGGER_PHRASES.items()
    }


# Global, compiled once.
TRACK_PATTERNS = _compile_track_patterns()


def _regex_scores(query: str) -> dict:
    return {
        track: sum(1 for pattern in patterns if pattern.search(query))
        for track, patterns in TRACK_PATTERNS.items()
    }


def _detect_track_regex(query: str):
    """Returns the highest-scoring track by keyword-match count, or None if
    no track's patterns matched at all (triggers the embedding fallback)."""
    scores = _regex_scores(query)
    best_track = max(config.TRACKS, key=lambda t: scores[t])
    return best_track if scores[best_track] > 0 else None


# ---------------------------------------------------------------------------
# Stage 3b: embedding fallback (Section 2.1, item 2)
# ---------------------------------------------------------------------------
# Global, loaded once, pinned to CPU (Section 4.4: "Pin router + embedding
# models to CPU, VLM to GPU, to avoid contention and unnecessary PCIe
# transfer of small intermediate tensors").
_embedding_model = SentenceTransformer(config.INTENT_EMBEDDING_MODEL, device=config.INTENT_EMBEDDING_DEVICE)


def _build_track_centroids() -> dict:
    """Embeds each track's trigger phrases and averages them into a single
    centroid vector per track, once, at import time."""
    centroids = {}
    for track, phrases in TRACK_TRIGGER_PHRASES.items():
        embeddings = _embedding_model.encode(phrases, convert_to_numpy=True, normalize_embeddings=True)
        centroids[track] = embeddings.mean(axis=0)
    return centroids


# Global, computed once.
TRACK_CENTROID_EMBEDDINGS = _build_track_centroids()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _detect_track_embedding(query: str) -> str:
    query_embedding = _embedding_model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    similarities = {
        track: _cosine_similarity(query_embedding, centroid)
        for track, centroid in TRACK_CENTROID_EMBEDDINGS.items()
    }
    return max(config.TRACKS, key=lambda t: similarities[t])


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def detect_track(query: str) -> str:
    """
    Args:
        query: natural-language question string.
    Returns:
        One of config.TRACKS: "diagnostic", "spatial", "severity",
        "modality_id", or "differential".
    """
    track = _detect_track_regex(query)
    if track is not None:
        return track
    return _detect_track_embedding(query)
