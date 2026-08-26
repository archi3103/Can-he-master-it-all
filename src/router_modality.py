"""
Stage 1: Modality-Stream Router (Section 1.2 of medical_vqa_architecture.md).

Classifies each incoming image into one of four modality streams --
radiology, macroscopic, microscopy, ultrasound -- using cheap, zero-VRAM
image-statistics heuristics (grayscale/RGB ratio, saturation, hue
clustering, border darkness, texture roughness, DICOM-tag presence)
rather than a learned classifier, per the doc's recommendation:

    "Use the heuristic router as primary, with the learned router as a
    fallback/confidence booster if you have spare latency budget."

`config.USE_LEARNED_ROUTER` gates an optional learned-router hook
(MobileNetV3-Small / EfficientNet-B0 class) that is intentionally left
unimplemented here -- no trained checkpoint ships with this repo, and the
doc treats it as optional. It is never invoked while the flag is False
(the shipped default), so it adds no import-time or per-query cost.
"""

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError as exc:  # pragma: no cover - surfaced at import time
    raise ImportError(
        "router_modality.py requires opencv-python-headless "
        "(see requirements.txt)"
    ) from exc

from src import config

# ---------------------------------------------------------------------------
# Hue ranges (degrees, 0-360) used for color-cluster heuristics.
# ---------------------------------------------------------------------------
PURPLE_PINK_HUE_RANGES_DEG = [(260.0, 335.0)]  # H&E stain (purple/magenta/pink)
BLUE_IHC_HUE_RANGES_DEG = [(190.0, 250.0)]  # blue IHC counterstain
SKIN_RED_HUE_RANGES_DEG = [(0.0, 40.0), (340.0, 360.0)]  # skin tone / erythema / red

NATIVE_GRAYSCALE_MODES = {"L", "I", "I;16", "I;16B", "I;16L", "F", "1"}
DICOM_METADATA_KEYS = {"PatientID", "Modality", "StudyInstanceUID", "SeriesInstanceUID", "dicom"}

_MIN_SATURATION_FOR_HUE = 0.15


# ---------------------------------------------------------------------------
# Low-level feature extraction
# ---------------------------------------------------------------------------
def _to_rgb_array(image: Image.Image) -> np.ndarray:
    """Convert a PIL image (any mode, including high-bit-depth DICOM-derived
    'I'/'I;16'/'F' modes) into an 8-bit RGB numpy array."""
    if image.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
        arr = np.asarray(image, dtype=np.float32)
        arr = arr - arr.min()
        max_val = arr.max()
        if max_val > 0:
            arr = arr / max_val
        gray = (arr * 255.0).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _has_dicom_signature(image: Image.Image) -> bool:
    fmt = (getattr(image, "format", "") or "").upper()
    if fmt == "DICOM":
        return True
    info = getattr(image, "info", {}) or {}
    return any(key in info for key in DICOM_METADATA_KEYS)


def _grayscale_deviation(rgb: np.ndarray) -> float:
    """Mean inter-channel absolute difference, normalized to [0, 1].
    ~0 => effectively grayscale content regardless of file color mode."""
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    diff = (np.abs(r - g) + np.abs(g - b) + np.abs(r - b)) / 3.0
    return float(diff.mean() / 255.0)


def _border_darkness(gray: np.ndarray) -> float:
    """1.0 => border frame is fully black (typical of radiographs and
    ultrasound sector scans with UI letterboxing); 0.0 => bright border."""
    h, w = gray.shape
    band = max(1, int(round(0.03 * min(h, w))))
    border_pixels = np.concatenate(
        [
            gray[:band, :].ravel(),
            gray[-band:, :].ravel(),
            gray[:, :band].ravel(),
            gray[:, -band:].ravel(),
        ]
    )
    return float(1.0 - border_pixels.mean() / 255.0)


def _texture_roughness(gray: np.ndarray) -> float:
    """Laplacian-variance-based texture/speckle indicator, squashed into
    [0, 1). High => granular speckle texture (ultrasound); low => smooth
    gradients (radiographs) or large flat color regions."""
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_32F, ksize=3).var())
    return laplacian_var / (laplacian_var + 500.0)


def _hue_entropy(hsv: np.ndarray) -> float:
    """Shannon entropy (normalized to [0, 1]) of the hue histogram, computed
    only over pixels with meaningful saturation. High => many distinct hues
    present (typical of tiled histology/cytology fields)."""
    hue = hsv[..., 0].astype(np.int32)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    mask = sat > _MIN_SATURATION_FOR_HUE
    if not np.any(mask):
        return 0.0
    hist, _ = np.histogram(hue[mask], bins=32, range=(0, 180))
    probs = hist.astype(np.float64)
    probs = probs[probs > 0] / probs.sum()
    entropy = float(-(probs * np.log2(probs)).sum())
    return entropy / np.log2(32)


def _hue_fraction(hsv: np.ndarray, hue_ranges_deg, min_saturation: float = _MIN_SATURATION_FOR_HUE) -> float:
    """Fraction of sufficiently-saturated pixels whose hue falls in any of
    `hue_ranges_deg` (each a (lo, hi) tuple in degrees, 0-360)."""
    hue_deg = hsv[..., 0].astype(np.float32) * 2.0  # OpenCV H in [0, 179] -> degrees
    sat = hsv[..., 1].astype(np.float32) / 255.0
    sat_mask = sat >= min_saturation
    total = int(sat_mask.sum())
    if total == 0:
        return 0.0
    hue_mask = np.zeros_like(sat_mask)
    for lo, hi in hue_ranges_deg:
        hue_mask |= (hue_deg >= lo) & (hue_deg <= hi)
    return float(np.logical_and(sat_mask, hue_mask).sum() / total)


def _has_circular_vignette(gray: np.ndarray) -> bool:
    """Detects the dark-corner / bright-center pattern typical of fundus
    photography, on a roughly square frame."""
    h, w = gray.shape
    patch = max(1, int(round(0.06 * min(h, w))))
    corners = [
        gray[:patch, :patch],
        gray[:patch, -patch:],
        gray[-patch:, :patch],
        gray[-patch:, -patch:],
    ]
    corner_mean = float(np.mean([c.mean() for c in corners]) / 255.0)
    cy0, cy1 = int(0.4 * h), int(0.6 * h)
    cx0, cx1 = int(0.4 * w), int(0.6 * w)
    center_mean = float(gray[cy0:cy1, cx0:cx1].mean() / 255.0)
    aspect = w / h if h else 1.0
    return corner_mean < 0.12 and center_mean > 0.25 and 0.8 <= aspect <= 1.25


def _extract_features(image: Image.Image) -> dict:
    rgb = _to_rgb_array(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    return {
        "saturation_mean": float(hsv[..., 1].mean() / 255.0),
        "saturation_std": float(hsv[..., 1].std() / 255.0),
        "grayscale_deviation": _grayscale_deviation(rgb),
        "border_darkness": _border_darkness(gray),
        "texture_roughness": _texture_roughness(gray),
        "hue_entropy": _hue_entropy(hsv),
        "purple_pink_fraction": _hue_fraction(hsv, PURPLE_PINK_HUE_RANGES_DEG),
        "blue_ihc_fraction": _hue_fraction(hsv, BLUE_IHC_HUE_RANGES_DEG),
        "skin_red_fraction": _hue_fraction(hsv, SKIN_RED_HUE_RANGES_DEG),
        "has_vignette": _has_circular_vignette(gray),
        "is_native_grayscale_mode": image.mode in NATIVE_GRAYSCALE_MODES,
        "has_dicom_signature": _has_dicom_signature(image),
    }


# ---------------------------------------------------------------------------
# Feature -> modality scoring (see the preprocessing table in Section 1.2)
# ---------------------------------------------------------------------------
def _score_modalities(features: dict) -> dict:
    grayscale_strength = (
        0.6 * (1.0 - features["saturation_mean"])
        + 0.4 * (1.0 - features["grayscale_deviation"])
    )
    dicom_bonus = 1.0 if (features["is_native_grayscale_mode"] or features["has_dicom_signature"]) else 0.0
    vignette_bonus = 1.0 if features["has_vignette"] else 0.0
    stain_fraction = max(features["purple_pink_fraction"], features["blue_ihc_fraction"])

    scores = {
        # Low saturation, high grayscale entropy, dark background (Sec 1.2 table).
        config.MODALITY_RADIOLOGY: (
            2.0 * grayscale_strength
            + 1.0 * features["border_darkness"]
            + 1.5 * dicom_bonus
            - 1.5 * features["texture_roughness"]  # speckle -> ultrasound, not radiology
        ),
        # High saturation, skin-tone/red-hue clusters, circular vignette (fundus).
        config.MODALITY_MACROSCOPIC: (
            2.0 * features["saturation_mean"]
            + 1.5 * features["skin_red_fraction"]
            + 1.5 * vignette_bonus
            - 1.0 * stain_fraction
        ),
        # High color variance, tiled/repeating texture, purple-pink (H&E) or blue (IHC).
        config.MODALITY_MICROSCOPY: (
            2.0 * stain_fraction
            + 1.2 * features["hue_entropy"]
            + 1.0 * features["saturation_std"]
            - 1.0 * vignette_bonus
        ),
        # Fan/sector ROI, black borders, speckle texture, low color.
        config.MODALITY_ULTRASOUND: (
            1.8 * grayscale_strength
            + 2.0 * features["texture_roughness"]
            + 1.2 * features["border_darkness"]
            - 1.0 * dicom_bonus  # true DICOM tags more often indicate CT/MR/X-ray archives
        ),
    }
    return scores


# ---------------------------------------------------------------------------
# Optional learned-router hook (Section 1.2: "fallback/confidence booster").
# Inert unless config.USE_LEARNED_ROUTER is explicitly enabled AND a
# checkpoint has been trained/placed at config.MODALITY_CLASSIFIER_PATH.
# ---------------------------------------------------------------------------
def _learned_router_boost(image: Image.Image, scores: dict) -> dict:
    raise NotImplementedError(
        "Learned modality router is disabled by default (heuristic-only is "
        "the doc's recommendation absent validated accuracy gains). To use "
        "it, train a MobileNetV3-Small/EfficientNet-B0 classifier, save it "
        f"to {config.MODALITY_CLASSIFIER_PATH}, load it here on "
        f"{config.MODALITY_ROUTER_DEVICE}, and blend its softmax output "
        "into `scores` before enabling config.USE_LEARNED_ROUTER."
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def detect_modality(image: Image.Image) -> str:
    """
    Args:
        image: PIL Image (raw medical image, arbitrary modality).
    Returns:
        One of config.MODALITIES: "radiology", "macroscopic",
        "microscopy", or "ultrasound".
    """
    features = _extract_features(image)
    scores = _score_modalities(features)
    if config.USE_LEARNED_ROUTER:
        scores = _learned_router_boost(image, scores)
    return max(config.MODALITIES, key=lambda modality: scores[modality])


# ---------------------------------------------------------------------------
# Stage 1b: hierarchical subtype classifiers (12-modality scaling).
#
# Each of these is a cheap, zero-VRAM, UNVALIDATED heuristic starting
# point -- tune against real labeled samples before relying on them beyond
# routing to a slightly-better preprocessing preset. A wrong subtype only
# ever selects a suboptimal-but-harmless preprocessing param set (see
# config.py's MACROSCOPIC_PARAMS/MICROSCOPY_PARAMS/ULTRASOUND_PARAMS
# schemas); it never changes the coarse stream, which stays governed
# entirely by the untouched detect_modality() above.
# ---------------------------------------------------------------------------
def _classify_radiology_subtype(gray: np.ndarray) -> str:
    """Splits the radiology stream into xray/mri/ct.

    Signals:
      - aspect_squareness: CT/MRI slices are typically near-square;
        radiographs (chest, limb, etc.) vary more widely.
      - circular_fov: the classic CT gantry field-of-view is a circular/
        oval cutoff with black corners outside it; MRI/X-ray don't
        typically show this.
      - black_background_fraction: radiographs commonly have substantial
        black film background outside the body silhouette; CT/MRI
        reconstructions are usually cropped tighter to the FOV.
      - edge_sharpness: bone/soft-tissue edges are sharp in X-ray/CT; MRI
        soft-tissue contrast tends to be smoother.
    """
    h, w = gray.shape
    aspect_ratio = w / h if h else 1.0
    aspect_squareness = 1.0 - min(abs(aspect_ratio - 1.0), 1.0)

    circular_fov = 1.0 if _has_circular_vignette(gray) else 0.0
    black_background_fraction = float((gray < 10).mean())
    edge_sharpness = _texture_roughness(gray)

    scores = {
        "xray": (
            (1.0 - aspect_squareness) * 1.5
            + black_background_fraction * 2.0
            + edge_sharpness * 1.0
            - circular_fov * 1.0
        ),
        "ct": (
            aspect_squareness * 1.0
            + circular_fov * 2.0
            + edge_sharpness * 1.0
            - black_background_fraction * 1.0
        ),
        "mri": (
            aspect_squareness * 1.0
            + (1.0 - edge_sharpness) * 1.5
            - circular_fov * 0.5
            - black_background_fraction * 1.0
        ),
    }
    return max(config.RADIOLOGY_SUBTYPES, key=lambda subtype: scores[subtype])


def _classify_macroscopic_subtype(rgb: np.ndarray, gray: np.ndarray, hsv: np.ndarray) -> str:
    """Splits the macroscopic stream into gross_pathology/dermoscopy/
    fundus/endoscopy/colposcopy/iri.

    endoscopy vs. colposcopy is the LOWEST-CONFIDENCE pair here: both are
    glossy, saturated, pink/red mucosal photography, distinguished only by
    a coarse vignette proxy (endoscopy video is more often circularly
    masked; colposcopy photos more often are not) -- a weak signal, called
    out explicitly rather than presented as reliable.

    iri detection (near-grayscale + vignette) is the direct mitigation for
    config.py's KNOWN LIMITATIONS note #1: *if* an IRI image survives
    coarse routing into "macroscopic" at all, this is a clean, confident
    signal (nothing else in this stream is both low-saturation and
    vignetted) -- the risk is entirely at the coarse-routing step, not here.
    """
    vignette = 1.0 if _has_circular_vignette(gray) else 0.0
    saturation_mean = float(hsv[..., 1].mean() / 255.0)
    reddish_fraction = _hue_fraction(hsv, SKIN_RED_HUE_RANGES_DEG)

    v = hsv[..., 2].astype(np.float32) / 255.0
    s = hsv[..., 1].astype(np.float32) / 255.0
    specular_fraction = float(((v > 0.92) & (s < 0.15)).mean())

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    hairiness_score = float((blackhat > 10).mean())

    scores = {
        "gross_pathology": (
            0.5
            + (1.0 - reddish_fraction) * 0.5
            - specular_fraction * 0.5
            - hairiness_score * 0.5
            - vignette * 0.5
        ),
        "dermoscopy": hairiness_score * 2.0 + vignette * 0.5 + saturation_mean * 0.5,
        "fundus": vignette * 1.5 + reddish_fraction * 1.5 + saturation_mean * 0.5 - hairiness_score * 1.0,
        "endoscopy": specular_fraction * 2.0 + reddish_fraction * 1.0 + vignette * 0.5,
        "colposcopy": specular_fraction * 2.0 + reddish_fraction * 1.0 - vignette * 0.5,
        "iri": (1.0 - saturation_mean) * 2.0 + vignette * 1.5,
    }
    return max(config.MACROSCOPIC_SUBTYPES, key=lambda subtype: scores[subtype])


def _classify_microscopy_subtype(rgb: np.ndarray, gray: np.ndarray) -> str:
    """Stub: config.MICROSCOPY_SUBTYPES currently has a single member
    ("histopathology") -- cytology is intentionally folded into it this
    pass (see config.py's KNOWN LIMITATIONS note #2), so there is nothing
    to discriminate yet. Kept as a real function, not a bare constant, so
    route_modality() and the eventual cytology split both have a stable
    call site."""
    return config.STREAM_SUBTYPE_DEFAULTS[config.MODALITY_MICROSCOPY]


def _classify_ultrasound_subtype(gray: np.ndarray) -> str:
    """Splits the ultrasound stream into ultrasound/oct.

    Signals:
      - corner_darkness: ultrasound's fan/sector ROI is typically
        black-corner-masked; OCT B-scans typically fill the frame.
      - band_score: row-wise mean-intensity variability relative to
        overall image contrast -- high for OCT's horizontal layer-banding,
        low for ultrasound's more uniformly-distributed speckle.
      - wide_aspect_bonus: OCT B-scans are often noticeably wider than
        tall; a weak secondary signal.
    """
    h, w = gray.shape
    patch = max(1, int(round(0.06 * min(h, w))))
    corners = [gray[:patch, :patch], gray[:patch, -patch:], gray[-patch:, :patch], gray[-patch:, -patch:]]
    corner_darkness = 1.0 - float(np.mean([c.mean() for c in corners]) / 255.0)

    row_means = gray.mean(axis=1)
    overall_std = float(gray.std()) + 1e-6
    band_score = float(row_means.std()) / overall_std

    aspect_ratio = w / h if h else 1.0
    wide_aspect_bonus = 1.0 if aspect_ratio > 1.6 else 0.0

    scores = {
        "ultrasound": corner_darkness * 2.0 + (1.0 - band_score) * 1.0,
        "oct": band_score * 2.0 + (1.0 - corner_darkness) * 1.5 + wide_aspect_bonus * 0.5,
    }
    return max(config.ULTRASOUND_SUBTYPES, key=lambda subtype: scores[subtype])


def route_modality(image: Image.Image):
    """
    Two-level hierarchical routing (12-modality scaling):
      1. coarse_stream = detect_modality(image) -- the original, UNCHANGED
         Stage 1 heuristic router above.
      2. subtype = a second, stream-conditional heuristic classifier that
         only runs once coarse_stream is known.

    Args:
        image: PIL Image (raw medical image, arbitrary modality).
    Returns:
        (coarse_stream, subtype) -- coarse_stream is one of
        config.MODALITIES; subtype is one of
        config.STREAM_SUBTYPES[coarse_stream].
    """
    coarse_stream = detect_modality(image)

    rgb = _to_rgb_array(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    if coarse_stream == config.MODALITY_RADIOLOGY:
        subtype = _classify_radiology_subtype(gray)
    elif coarse_stream == config.MODALITY_MACROSCOPIC:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        subtype = _classify_macroscopic_subtype(rgb, gray, hsv)
    elif coarse_stream == config.MODALITY_MICROSCOPY:
        subtype = _classify_microscopy_subtype(rgb, gray)
    elif coarse_stream == config.MODALITY_ULTRASOUND:
        subtype = _classify_ultrasound_subtype(gray)
    else:
        subtype = None

    valid_subtypes = config.STREAM_SUBTYPES.get(coarse_stream, [])
    if subtype not in valid_subtypes:
        subtype = config.STREAM_SUBTYPE_DEFAULTS.get(coarse_stream, config.STREAM_SUBTYPE_DEFAULTS[config.MODALITY_RADIOLOGY])

    return coarse_stream, subtype


# Alias matching the blueprint's expected name for the (untouched) coarse
# router. detect_modality is the original entry point and is what
# src/predict.py imports directly -- this alias exists purely so
# route_modality()'s description of calling "detect_coarse_stream()" has a
# literal referent, without changing detect_modality's signature or import
# path.
detect_coarse_stream = detect_modality
