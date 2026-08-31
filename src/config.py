"""
Central configuration for the Unified Medical Multi-Modal VQA System.

All paths, constants, and tunable thresholds referenced across
src/model_loader.py, src/router_modality.py, src/router_intent.py,
src/preprocessing.py, src/prompt_builder.py, src/decode.py, and
src/predict.py live here so there is a single source of truth,
loaded once at import time (see Section 4.1 of
medical_vqa_architecture.md).
"""

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Directory layout (see Section 5.1 of medical_vqa_architecture.md)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"

MODEL_PATH = str(WEIGHTS_DIR / "qwen3-vl-4b-instruct")
LORA_PATH = str(WEIGHTS_DIR / "lora-medvqa")

# ---------------------------------------------------------------------------
# Backbone model (Section 3.1 / 4.1)
# ---------------------------------------------------------------------------
# Migrated from AWQ-quantized Qwen2-VL-7B-Instruct to native (unquantized)
# Qwen3-VL (Section 3.3): eliminates the autoawq/gptqmodel dependency chain
# entirely -- no C++ kernel compilation required. Deployed size within the
# Qwen3-VL line was later resized from 8B down to 4B: the 8B weights
# (~16-17.5GB) fit the 48GB RTX 6000 Ada VRAM budget comfortably but
# exceed the competition's strict 10GB submission file size limit; 4B
# (~8-9GB in fp16) fits inside it, in the same model family/generation.
# NOTE: any LoRA adapter previously trained against Qwen2-VL-7B OR
# Qwen3-VL-8B's architecture is NOT compatible with Qwen3-VL-4B (different
# hidden sizes/attention config per model size) and must be retrained.
BASE_MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"
TORCH_DTYPE = torch.float16
DEVICE = "cuda:0"
LORA_ADAPTER_NAME = "medvqa"

# Attention backend: prefer Flash-Attention 2, fall back to sdpa if the
# wheel cannot be built locally (Section 4.4 / 5.4 note).
ATTN_IMPLEMENTATION = "flash_attention_2"
ATTN_IMPLEMENTATION_FALLBACK = "sdpa"

# ---------------------------------------------------------------------------
# LoRA fine-tuning hyperparameters (Section 3.2)
# ---------------------------------------------------------------------------
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
LORA_EPOCHS = 3

# ---------------------------------------------------------------------------
# Image resolution control (Section 4.3)
# ---------------------------------------------------------------------------
MIN_PIXELS = 256 * 256
MAX_PIXELS = 1024 * 1024

# ---------------------------------------------------------------------------
# Stage -1: Volumetric/DICOM ingest (src/volume_loader.py)
# ---------------------------------------------------------------------------
# Runs before Stage 0 (universal_normalize) and before PIL ever touches the
# file. Decodes 3D NIfTI volumes, single DICOM files, multi-frame DICOM
# files, and DICOM series folders into a single 2D RGB image; flat 2D
# images (.png/.jpg/etc.) pass straight through untouched.
#
# Slice-selection policy: three slices at these relative depths are
# extracted and tiled into one 1x3 composite, rather than picking one
# arbitrary slice (risks landing on an uninformative edge slice) or
# decoding the whole volume (the VLM only accepts a single 2D image, so
# most of that work would be wasted). Cheap heuristic, not a learned
# key-slice selector -- consistent with this codebase's latency-first
# design elsewhere (see the feature-flag section below).
VOLUME_TRISLICE_DEPTH_FRACTIONS = (0.35, 0.50, 0.65)

# Vote-MI-inspired content-based slice selection (src/volume_loader.py's
# _select_top_k_slice_indices) -- the primary slice-selection policy;
# VOLUME_TRISLICE_DEPTH_FRACTIONS above is now only the tier-2 fallback
# used when content scoring itself errors.
VOLUME_SLICE_SELECTION_COUNT = 3  # how many representative slices to composite
# Minimum gap between selected slice indices, as a fraction of total
# depth -- keeps the selection spread across the volume instead of
# collapsing onto a cluster of near-duplicate adjacent slices.
VOLUME_SLICE_MIN_SEPARATION_FRACTION = 0.10
# Relative weight of edge-density vs. intensity variance in the composite
# informativeness score -- both are min-max normalized to [0, 1] first,
# so 1.0 means equal weight by default.
VOLUME_EDGE_DENSITY_WEIGHT = 1.0
# Above this many slices, content scoring strides through a coarse sample
# instead of every slice, to bound worst-case latency on very deep
# volumes (e.g. a 1000+-slice thin-cut CT series). Sobel + variance is
# cheap per-slice, but not free at that scale.
VOLUME_SCORING_MAX_SLICES = 128

# Channel-split microscopy folders (src/volume_loader.py's
# _tile_color_channels_into_grid): each channel is resized to this fixed
# (square) size before tiling into the 2x2 grid. Chosen so the finished
# grid (2 cells + 3 borders per axis) lands comfortably under MAX_PIXELS
# on its own -- a full-resolution per-channel grid was measured landing
# right at the MAX_PIXELS cap (Stage 0 then downscaling it to 1024x1024
# regardless), pushing vision-token count -- and single-query inference
# time -- well above flat 2D images, right at/over
# INFERENCE_TIMEOUT_SECONDS even with no other load on the GPU. This
# trades some per-channel resolution for meaningfully fewer vision
# tokens.
VOLUME_CHANNEL_GRID_CELL_SIZE = 448

# DICOM series folders (src/volume_loader.py's _load_dicom_series): number
# of files read concurrently via a thread pool. Reading is I/O-bound
# (pydicom's file read + numpy pixel-array decode release the GIL), and
# each file open carries real round-trip latency on a network/FUSE-backed
# filesystem (e.g. a Colab Google Drive mount) -- overlapping reads cuts
# wall-clock time roughly by this factor there. Harmless on local disk,
# where I/O is fast enough that this barely matters either way.
VOLUME_DICOM_READ_WORKERS = 16

# ---------------------------------------------------------------------------
# Stage 2: Modality-specific preprocessing (Section 1.2)
# ---------------------------------------------------------------------------
# Bumped whenever the preprocessing heuristics/tunables below change in a
# way that could shift model outputs -- useful for correlating a
# predictions.csv run against the exact preprocessing logic that produced
# it, independent of the git commit hash.
PREPROCESS_VERSION = "1.1.0"

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
# Per-subtype override dicts (CLAHE_PARAMS, MACROSCOPIC_PARAMS,
# MICROSCOPY_PARAMS, ULTRASOUND_PARAMS) are grouped together in the
# "Stage 1b: Hierarchical Subtype Routing" section below, alongside
# STREAM_SUBTYPE_DEFAULTS -- they all key off the same subtype strings
# route_modality() (src/router_modality.py) produces.

# Percentile-based intensity windowing (soft-tissue/bone heuristic stand-in
# for true Hounsfield-unit windowing, which requires raw DICOM pixel data
# with rescale slope/intercept that a generic PIL image does not carry).
RADIOLOGY_WINDOW_LOW_PERCENTILE = 0.5
RADIOLOGY_WINDOW_HIGH_PERCENTILE = 99.5
RADIOLOGY_TARGET_SIZE = (896, 896)  # aspect-preserved + padded

MACROSCOPIC_FUNDUS_CROP_MARGIN = 0.02  # fraction trimmed off each side after centering

MICROSCOPY_TILE_SIZE = 512
# Above this pixel count, treat the image as WSI-like and select the
# single most-informative tile rather than downsampling the whole image.
MICROSCOPY_WSI_TRIGGER_PIXELS = 1536 * 1536

# Macenko normalization is O(pixels) (SVD + lstsq over every stained
# pixel); skip it above this size even after tile selection, to bound
# worst-case microscopy latency.
STAIN_NORM_MAX_PIXELS = 512 * 512

# Standard Macenko (2009) H&E reference stain vectors -- the published,
# non-site-specific values (i.e. "neutral": not tuned to any particular
# scanner/stain batch). Used as the fallback whenever no calibrated
# reference has been placed on disk.
_STAIN_REFERENCE_MATRIX_FALLBACK = np.array(
    [[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]]
)
STAIN_REFERENCE_MATRIX_PATH = WEIGHTS_DIR / "stain_reference_matrix.npy"


def _load_stain_reference_matrix() -> np.ndarray:
    """Loads a site-calibrated Macenko reference stain matrix from
    weights/stain_reference_matrix.npy if present and well-formed;
    otherwise falls back to the standard Macenko reference values so
    microscopy preprocessing never breaks on a missing or corrupt
    weights file."""
    try:
        matrix = np.load(STAIN_REFERENCE_MATRIX_PATH)
    except (FileNotFoundError, OSError, ValueError) as exc:
        logger.info(
            "No stain reference matrix at %s (%s); using the standard "
            "Macenko reference values.",
            STAIN_REFERENCE_MATRIX_PATH,
            exc,
        )
        return _STAIN_REFERENCE_MATRIX_FALLBACK

    if matrix.shape != _STAIN_REFERENCE_MATRIX_FALLBACK.shape:
        logger.warning(
            "Stain reference matrix at %s has shape %s, expected %s; "
            "falling back to the standard Macenko reference values.",
            STAIN_REFERENCE_MATRIX_PATH,
            matrix.shape,
            _STAIN_REFERENCE_MATRIX_FALLBACK.shape,
        )
        return _STAIN_REFERENCE_MATRIX_FALLBACK

    return matrix


STAIN_REFERENCE_MATRIX = _load_stain_reference_matrix()

# ---------------------------------------------------------------------------
# Feature flags / ablation gates -- latency-budget protections. Each of
# these disables a measurable-but-costly preprocessing step by default;
# flip to True only after validation shows the accuracy gain outweighs the
# added wall-clock time (Section 3.3: "latency is the binding constraint").
# ---------------------------------------------------------------------------
# Blackhat-morphology hair/artifact inpainting in preprocess_macroscopic().
ENABLE_HAIR_INPAINT = False

# HSV-based specular-highlight suppression (suppress_specular_highlights()
# in src/preprocessing.py) for glossy/wet macroscopic subtypes (endoscopy,
# colposcopy). Same latency-budget rationale as ENABLE_HAIR_INPAINT above.
ENABLE_SPECULAR_SUPPRESSION = False

# Speckle-reducing anisotropic diffusion is latency-expensive; skip by
# default per Section 3.3 ("latency is the binding constraint ... tilt
# every design decision toward minimizing wall-clock time").
ULTRASOUND_SPECKLE_FILTER_ENABLED = False
ULTRASOUND_DIFFUSION_ITERATIONS = 8

# ---------------------------------------------------------------------------
# Stage 1: Modality-Stream Router (Section 1.2)
# ---------------------------------------------------------------------------
MODALITY_RADIOLOGY = "radiology"
MODALITY_MACROSCOPIC = "macroscopic"
MODALITY_MICROSCOPY = "microscopy"
MODALITY_ULTRASOUND = "ultrasound"

MODALITIES = [
    MODALITY_RADIOLOGY,
    MODALITY_MACROSCOPIC,
    MODALITY_MICROSCOPY,
    MODALITY_ULTRASOUND,
]

# Use the heuristic (zero-cost, zero-VRAM) router as primary; the learned
# router (MobileNetV3-Small / EfficientNet-B0) is an optional fallback /
# confidence booster only if spare latency budget allows it.
USE_LEARNED_ROUTER = False
MODALITY_CLASSIFIER_PATH = str(WEIGHTS_DIR / "modality-router")
MODALITY_ROUTER_DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Stage 1b: Hierarchical Subtype Routing (12-modality scaling)
#
# route_modality() (src/router_modality.py) returns (coarse_stream, subtype):
#   coarse_stream = detect_modality(image) -- the original, UNCHANGED Stage 1
#                   4-bucket heuristic router above.
#   subtype       = a second, stream-conditional heuristic classifier that
#                   only runs once the coarse stream is known, picking among
#                   the finer-grained modalities each stream absorbs.
#
# This keeps the well-tuned coarse router untouched while covering 12
# OmniMedVQA-style modalities: radiology{xray,mri,ct}, macroscopic
# {dermoscopy,fundus,endoscopy,colposcopy,gross_pathology,iri}, microscopy
# {histopathology}, ultrasound{ultrasound,oct}.
# ---------------------------------------------------------------------------
RADIOLOGY_SUBTYPES = ["xray", "mri", "ct"]
MACROSCOPIC_SUBTYPES = ["gross_pathology", "dermoscopy", "fundus", "endoscopy", "colposcopy", "iri"]
MICROSCOPY_SUBTYPES = ["histopathology"]
ULTRASOUND_SUBTYPES = ["ultrasound", "oct"]

STREAM_SUBTYPES = {
    MODALITY_RADIOLOGY: RADIOLOGY_SUBTYPES,
    MODALITY_MACROSCOPIC: MACROSCOPIC_SUBTYPES,
    MODALITY_MICROSCOPY: MICROSCOPY_SUBTYPES,
    MODALITY_ULTRASOUND: ULTRASOUND_SUBTYPES,
}

# Safe fallback subtype per stream when subtype-classifier confidence is
# low, the classifier errors, or it returns something unrecognized -- each
# is that stream's most common/general-purpose member.
STREAM_SUBTYPE_DEFAULTS = {
    MODALITY_RADIOLOGY: "xray",
    MODALITY_MACROSCOPIC: "gross_pathology",
    MODALITY_MICROSCOPY: "histopathology",
    MODALITY_ULTRASOUND: "ultrasound",
}

# Per-subtype radiology CLAHE overrides, consumed by
# src/preprocessing.py's preprocess_radiology(image, subtype).
CLAHE_PARAMS = {
    "xray": {"clip_limit": 2.5, "tile_grid_size": (8, 8)},
    "mri": {"clip_limit": 1.5, "tile_grid_size": (8, 8)},
    "ct": {"clip_limit": 2.0, "tile_grid_size": (16, 16)},
}

# Per-subtype macroscopic overrides, consumed by
# src/preprocessing.py's preprocess_macroscopic(image, subtype). Schema:
#   enable_hair_inpaint: bool -- also gated globally by ENABLE_HAIR_INPAINT
#   suppress_specular:   bool -- also gated globally by ENABLE_SPECULAR_SUPPRESSION
#   letterbox:           bool -- force letterbox_resize() regardless of the
#                                generic vignette-detection fallback check
#   target_size:         int  -- letterbox_resize() target (square)
MACROSCOPIC_PARAMS = {
    "gross_pathology": {"enable_hair_inpaint": False, "suppress_specular": False, "letterbox": False, "target_size": 896},
    "dermoscopy": {"enable_hair_inpaint": True, "suppress_specular": False, "letterbox": False, "target_size": 896},
    "fundus": {"enable_hair_inpaint": False, "suppress_specular": False, "letterbox": True, "target_size": 896},
    "endoscopy": {"enable_hair_inpaint": False, "suppress_specular": True, "letterbox": False, "target_size": 896},
    "colposcopy": {"enable_hair_inpaint": False, "suppress_specular": True, "letterbox": False, "target_size": 896},
    # IRI (Infrared Reflectance Imaging): near-grayscale, wide-field,
    # camera/scanner-acquired (often alongside OCT + color fundus on the
    # same ophthalmic device) -- placed under macroscopic rather than
    # microscopy because its acquisition geometry and field of view match
    # fundus photography, not cellular-resolution tissue-section imaging.
    # See the KNOWN LIMITATIONS note below for its coarse-routing risk.
    "iri": {"enable_hair_inpaint": False, "suppress_specular": False, "letterbox": True, "target_size": 896},
}

# Per-subtype microscopy overrides, consumed by
# src/preprocessing.py's preprocess_microscopy(image, subtype). Currently a
# single "histopathology" key -- cytology is intentionally folded into it
# rather than given its own key this pass; see the KNOWN LIMITATIONS note.
# Values default to the existing flat MICROSCOPY_*/STAIN_NORM_MAX_PIXELS
# constants so there is one source of truth for the numbers themselves.
MICROSCOPY_PARAMS = {
    "histopathology": {
        "tile_size": MICROSCOPY_TILE_SIZE,
        "wsi_trigger_pixels": MICROSCOPY_WSI_TRIGGER_PIXELS,
        "stain_norm_max_pixels": STAIN_NORM_MAX_PIXELS,
        "enable_stain_norm": True,
    },
}

# Per-subtype ultrasound-stream overrides, consumed by
# src/preprocessing.py's preprocess_ultrasound(image, subtype).
ULTRASOUND_PARAMS = {
    "ultrasound": {"crop_roi": True, "enable_speckle_filter": True},
    # OCT (Optical Coherence Tomography): light-based cross-sectional
    # B-scans with horizontal layer-banding and (typically) no black
    # fan-corner masking -- structurally different from ultrasound, so
    # neither the fan-shape ROI crop nor the speckle-diffusion filter
    # (tuned/validated for ultrasound speckle statistics only) apply.
    "oct": {"crop_roi": False, "enable_speckle_filter": False},
}

# ---------------------------------------------------------------------------
# KNOWN LIMITATIONS -- subtype-taxonomy review (flagged before implementing
# the classifiers in src/router_modality.py; no existing "Step 5 validation
# slice" was found in this codebase to anchor these to, so they live here,
# next to the routing tables they concern, and are echoed in the relevant
# _classify_*_subtype() docstrings).
#
# 1. Coarse-router misrouting risk for near-grayscale macroscopic subtypes.
#    detect_modality() (Stage 1, UNCHANGED per design) leans heavily on
#    saturation/grayscale-deviation to separate "radiology" from
#    "macroscopic". Both IRI (near-grayscale, single-wavelength reflectance)
#    and OCT (near-grayscale cross-sectional B-scan) are low-saturation
#    modalities that risk being misrouted into "radiology" by the coarse
#    router before any subtype classifier below ever runs. This is a
#    pre-existing constraint of the coarse router (which this pass
#    deliberately does not modify), not a bug in the subtype classifiers
#    themselves -- flagged as accepted, bounded risk to revisit if
#    validation shows it materially hurts accuracy.
#
# 2. Cytology/histopathology split is deferred, not implemented. The
#    MICROSCOPY_PARAMS/MICROSCOPY_SUBTYPES above intentionally have only
#    one key ("histopathology"); cytology specimens are preprocessed
#    identically for now. A cheap future heuristic already has groundwork
#    in src/preprocessing.py's _select_informative_tile() tissue-fraction
#    metric (`< 235` pixel threshold): cytology's sparse, scattered-cell-
#    cluster pattern should show a measurably lower tissue fraction than
#    histopathology's dense, continuous tissue architecture. Not
#    implemented this pass -- documented here as the natural next step.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Stage 3: Query-Intent / Track Classifier (Section 2.1)
# ---------------------------------------------------------------------------
TRACK_DIAGNOSTIC = "diagnostic"
TRACK_SPATIAL = "spatial"
TRACK_SEVERITY = "severity"
TRACK_MODALITY_ID = "modality_id"
TRACK_DIFFERENTIAL = "differential"

TRACKS = [
    TRACK_DIAGNOSTIC,
    TRACK_SPATIAL,
    TRACK_SEVERITY,
    TRACK_MODALITY_ID,
    TRACK_DIFFERENTIAL,
]

# Frozen sentence-embedding encoder used as the fallback classifier when the
# regex/keyword pass does not confidently match a track.
INTENT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
INTENT_EMBEDDING_DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Stage 6: Constrained decoding (Section 4.2)
# ---------------------------------------------------------------------------
CHOICE_LETTERS = ["A", "B", "C", "D"]
CHOICE_TOKEN_VARIANTS = {
    letter: [letter, f" {letter}", f"{letter})", f"({letter}"]
    for letter in CHOICE_LETTERS
}

# Valid *sets* of offered choice keys a query may present -- 4-choice
# (A-D) MCQ as the default, plus 2-choice (A/B, e.g. Yes/No) rows like
# those in dev_metadata.csv. predict.py's _validate_choices() checks
# against these; decode.py's constrained_predict_letter() is told which
# subset to actually consider per-query so a 2-choice question can never
# be answered "C" or "D".
CHOICE_SET_OPTIONS = [["A", "B"], CHOICE_LETTERS]

# ---------------------------------------------------------------------------
# Scoring-formula latency awareness: Score = Accuracy - (k * Avg Inference
# Time). k is not known ahead of time, so this is a conservative, tunable
# placeholder rather than a value derived from a known k.
# ---------------------------------------------------------------------------
LATENCY_PENALTY_K = None  # set once k is known/estimated via validation

# ---------------------------------------------------------------------------
# Section 5.5: Failure-mode safeguards
# ---------------------------------------------------------------------------
INFERENCE_TIMEOUT_SECONDS = 5.0
FALLBACK_ANSWER_LETTER = "A"

# ---------------------------------------------------------------------------
# Blind-fallback support (src/predict.py's predict_with_diagnostics(), used
# by src/evaluate_omnimed.py). When the real image can't be loaded/decoded
# at all, a neutral mid-gray canvas of this size is substituted and run
# through the full pipeline instead of skipping straight to
# FALLBACK_ANSWER_LETTER -- lets the model still reason from the question
# text and choices alone rather than answering fully blind. Not used by
# predict()/eval.py's competition path, which keeps its original
# immediate-fallback behavior unchanged.
# ---------------------------------------------------------------------------
BLIND_FALLBACK_IMAGE_SIZE = (512, 512)
BLIND_FALLBACK_GRAY_VALUE = 128
