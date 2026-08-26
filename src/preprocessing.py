"""
Stage 2: Modality-specific image preprocessing (Section 1.2 of
medical_vqa_architecture.md), now subtype-aware for 12-modality scaling
(see config.py's "Stage 1b: Hierarchical Subtype Routing" section), plus:

  - universal_normalize(): a front-of-pipeline pass that flattens any
    source format (RGBA, 16-bit/float TIFF grayscale, palette-with-
    transparency, etc.) down to a clean 8-bit RGB image, so every
    downstream stage -- including Stage 1 modality routing -- can assume
    a uniform input. Callers should invoke it before src.router_modality.
    route_modality(); preprocess_image() also calls it defensively
    (idempotent on an already-normalized image) so this module is
    self-sufficient if used standalone.
  - A resolution-capping safety net (Section 4.3) applied before any
    modality-specific work so CLAHE, stain normalization, hair
    inpainting, etc. never run on an unnecessarily huge raw image.

Each coarse stream now has a public, subtype-aware entry point that looks
up its parameter dict from src/config.py:

    preprocess_radiology(image, subtype)    -> config.CLAHE_PARAMS
        CLAHE contrast enhancement (per-subtype clip/tile settings),
        intensity windowing, resize with aspect preserved + padding.
    preprocess_macroscopic(image, subtype)  -> config.MACROSCOPIC_PARAMS
        color constancy normalization, optional hair/artifact inpainting
        (dermoscopy) or specular-highlight suppression (endoscopy/
        colposcopy), letterbox resize for vignetted subtypes (fundus, iri).
    preprocess_microscopy(image, subtype)   -> config.MICROSCOPY_PARAMS
        most-informative-tile selection for WSI-like inputs, Macenko
        stain normalization (config.STAIN_REFERENCE_MATRIX, gated by
        config.STAIN_NORM_MAX_PIXELS).
    preprocess_ultrasound(image, subtype)   -> config.ULTRASOUND_PARAMS
        ROI crop (removes UI overlays / black borders) for true
        ultrasound; skipped for OCT, which isn't fan/sector-masked.
"""

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError as exc:  # pragma: no cover - surfaced at import time
    raise ImportError(
        "preprocessing.py requires opencv-python-headless (see requirements.txt)"
    ) from exc

from src import config

# Macenko target stain concentrations. config.STAIN_REFERENCE_MATRIX (the
# actual stain color vectors) is loaded once at import time in
# src/config.py, with a robust fallback -- see _macenko_normalize() below.
_REFERENCE_MAX_CONCENTRATION = np.array([1.9705, 1.0308])


# ---------------------------------------------------------------------------
# Universal normalization -- runs first, before modality routing.
# ---------------------------------------------------------------------------
_ALPHA_MODES = {"RGBA", "LA"}
_HIGH_BITDEPTH_GRAYSCALE_MODES = {"I", "I;16", "I;16B", "I;16L", "F"}

# Kept separate from config.RADIOLOGY_WINDOW_*_PERCENTILE even though the
# values happen to match: this stretch is a generic "make it viewable"
# step for any 16-bit/float source, not the radiology-specific windowing
# applied later in preprocess_radiology().
_UNIVERSAL_STRETCH_LOW_PERCENTILE = 0.5
_UNIVERSAL_STRETCH_HIGH_PERCENTILE = 99.5


def _percentile_stretch_to_uint8(arr: np.ndarray, low_percentile: float, high_percentile: float) -> np.ndarray:
    lo = float(np.percentile(arr, low_percentile))
    hi = float(np.percentile(arr, high_percentile))
    if hi <= lo:
        arr_min, arr_max = float(arr.min()), float(arr.max())
        span = (arr_max - arr_min) or 1.0
        return ((arr - arr_min) / span * 255.0).astype(np.uint8)
    stretched = np.clip(arr, lo, hi)
    stretched = (stretched - lo) / (hi - lo) * 255.0
    return stretched.astype(np.uint8)


def universal_normalize(image: Image.Image) -> Image.Image:
    """
    Flattens any source image format to a clean 8-bit RGB PIL Image:

      - RGBA / LA / palette-with-transparency -> alpha-composited onto a
        white background, then flattened to RGB. Naively dropping the
        alpha channel would leave unpremultiplied/garbage color data
        behind fully-transparent regions.
      - 16-bit or floating-point grayscale (PIL modes "I", "I;16",
        "I;16B", "I;16L", "F" -- common for exported DICOM/TIFF slides
        and radiographs) -> robust percentile stretch to 8-bit, rather
        than a naive min-max, so a handful of hot/dead outlier pixels
        can't crush the visible dynamic range.
      - Anything else (8-bit grayscale, CMYK, palette without
        transparency, etc.) -> a plain RGB conversion.

    Idempotent: calling this again on an already-RGB image is a no-op.
    """
    mode = image.mode

    if mode in _ALPHA_MODES or (mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        return background

    if mode in _HIGH_BITDEPTH_GRAYSCALE_MODES:
        arr = np.asarray(image, dtype=np.float32)
        gray = _percentile_stretch_to_uint8(arr, _UNIVERSAL_STRETCH_LOW_PERCENTILE, _UNIVERSAL_STRETCH_HIGH_PERCENTILE)
        return Image.fromarray(gray, mode="L").convert("RGB")

    if mode != "RGB":
        return image.convert("RGB")

    return image


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _cap_resolution(image: Image.Image) -> Image.Image:
    """Bounds total pixel count to [MIN_PIXELS, MAX_PIXELS]. Complements the
    HF processor's own min/max_pixels resize (src/model_loader.py) by
    keeping the *raw* image cheap to run modality-specific ops on."""
    w, h = image.size
    pixel_count = w * h
    if pixel_count == 0:
        return image
    if pixel_count > config.MAX_PIXELS:
        scale = (config.MAX_PIXELS / pixel_count) ** 0.5
    elif pixel_count < config.MIN_PIXELS:
        scale = (config.MIN_PIXELS / pixel_count) ** 0.5
    else:
        return image
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return image.resize(new_size, Image.LANCZOS)


def _resize_with_padding(arr: np.ndarray, target_size: tuple) -> np.ndarray:
    """Aspect-preserving resize onto a target_w x target_h canvas, padded
    (black) to fill the remainder. Used by both radiology (via
    preprocess_radiology) and macroscopic (via letterbox_resize)."""
    target_w, target_h = target_size
    h, w = arr.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas_shape = (target_h, target_w) if arr.ndim == 2 else (target_h, target_w, arr.shape[2])
    canvas = np.zeros(canvas_shape, dtype=arr.dtype)
    y0, x0 = (target_h - new_h) // 2, (target_w - new_w) // 2
    if arr.ndim == 2:
        canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    else:
        canvas[y0 : y0 + new_h, x0 : x0 + new_w, :] = resized
    return canvas


def letterbox_resize(image: Image.Image, target: int = 896) -> Image.Image:
    """Aspect-preserving resize onto a square `target` x `target` canvas,
    padded (not cropped) to fill the remainder. Used in place of a
    center-crop for vignetted macroscopic subtypes (fundus, iri): a
    center-crop assumes peripheral content is unimportant, which can
    discard genuine peripheral pathology; a letterbox keeps the entire
    original field of view."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    padded = _resize_with_padding(rgb, (target, target))
    return Image.fromarray(padded)


def suppress_specular_highlights(rgb: np.ndarray) -> np.ndarray:
    """Detects and inpaints small, very bright/desaturated specular
    highlight blobs (glare from a light source reflecting off wet
    mucosa) -- common in endoscopy/colposcopy. Cheap HSV-based mask
    (high V, low S) + cv2.inpaint, no external model. Gated by
    config.ENABLE_SPECULAR_SUPPRESSION and the calling subtype's
    `suppress_specular` param (see preprocess_macroscopic)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    v = hsv[..., 2].astype(np.float32) / 255.0
    s = hsv[..., 1].astype(np.float32) / 255.0
    mask = ((v > 0.92) & (s < 0.15)).astype(np.uint8) * 255
    if not np.any(mask):
        return rgb
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=1)
    return cv2.inpaint(rgb, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


# ---------------------------------------------------------------------------
# Radiology: CLAHE + intensity windowing + aspect-preserved padded resize
# ---------------------------------------------------------------------------
def _percentile_window(gray: np.ndarray) -> np.ndarray:
    """Stand-in for soft-tissue/bone HU windowing: clip to a robust
    intensity percentile range, then rescale to the full 0-255 band. True
    Hounsfield-unit windowing requires raw DICOM pixel data (rescale
    slope/intercept) that a generic PIL image does not carry."""
    lo = np.percentile(gray, config.RADIOLOGY_WINDOW_LOW_PERCENTILE)
    hi = np.percentile(gray, config.RADIOLOGY_WINDOW_HIGH_PERCENTILE)
    if hi <= lo:
        return gray
    windowed = np.clip(gray, lo, hi)
    windowed = (windowed - lo) / (hi - lo) * 255.0
    return windowed.astype(np.uint8)


def _apply_clahe(
    gray: np.ndarray,
    clip_limit: float = config.CLAHE_CLIP_LIMIT,
    tile_grid_size: tuple = config.CLAHE_TILE_GRID_SIZE,
) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def preprocess_radiology(image: Image.Image, subtype: str = None) -> Image.Image:
    """
    Args:
        image: RGB PIL Image (already universal_normalize()d).
        subtype: one of config.RADIOLOGY_SUBTYPES ("xray"/"mri"/"ct");
            defaults to config.STREAM_SUBTYPE_DEFAULTS[radiology] if
            omitted or unrecognized.
    """
    default_subtype = config.STREAM_SUBTYPE_DEFAULTS[config.MODALITY_RADIOLOGY]
    subtype = subtype or default_subtype
    params = config.CLAHE_PARAMS.get(subtype, config.CLAHE_PARAMS[default_subtype])

    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    gray = _percentile_window(gray)
    gray = _apply_clahe(gray, params["clip_limit"], params["tile_grid_size"])
    gray = _resize_with_padding(gray, config.RADIOLOGY_TARGET_SIZE)
    return Image.fromarray(gray).convert("RGB")


# ---------------------------------------------------------------------------
# Macroscopic: color constancy, optional hair inpainting / specular
# suppression, letterbox resize for vignetted subtypes
# ---------------------------------------------------------------------------
def _gray_world_white_balance(rgb: np.ndarray) -> np.ndarray:
    result = rgb.astype(np.float32)
    channel_means = result.reshape(-1, 3).mean(axis=0)
    overall_mean = channel_means.mean()
    for c in range(3):
        if channel_means[c] > 1e-6:
            result[..., c] *= overall_mean / channel_means[c]
    return np.clip(result, 0, 255).astype(np.uint8)


def _remove_hair_artifacts(rgb: np.ndarray) -> np.ndarray:
    """Standard dermoscopy hair-removal recipe: blackhat morphology to
    isolate thin dark hair strands, threshold to a mask, then inpaint."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    if not np.any(hair_mask):
        return rgb
    return cv2.inpaint(rgb, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


def _looks_like_fundus(rgb: np.ndarray) -> bool:
    """Dark-corner / bright-center vignette on a roughly square frame.
    Kept as a generic fallback safety net in preprocess_macroscopic:
    regardless of what the subtype classifier decided, an image that is
    physically vignetted should always be letterboxed rather than
    center-cropped."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    patch = max(1, int(round(0.06 * min(h, w))))
    corners = [gray[:patch, :patch], gray[:patch, -patch:], gray[-patch:, :patch], gray[-patch:, -patch:]]
    corner_mean = float(np.mean([c.mean() for c in corners]) / 255.0)
    aspect = w / h if h else 1.0
    return corner_mean < 0.12 and 0.8 <= aspect <= 1.25


def preprocess_macroscopic(image: Image.Image, subtype: str = None) -> Image.Image:
    """
    Args:
        image: RGB PIL Image (already universal_normalize()d).
        subtype: one of config.MACROSCOPIC_SUBTYPES; defaults to
            config.STREAM_SUBTYPE_DEFAULTS[macroscopic] if omitted or
            unrecognized.
    """
    default_subtype = config.STREAM_SUBTYPE_DEFAULTS[config.MODALITY_MACROSCOPIC]
    subtype = subtype or default_subtype
    params = config.MACROSCOPIC_PARAMS.get(subtype, config.MACROSCOPIC_PARAMS[default_subtype])

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    rgb = _gray_world_white_balance(rgb)

    if params.get("suppress_specular") and config.ENABLE_SPECULAR_SUPPRESSION:
        rgb = suppress_specular_highlights(rgb)

    if params.get("enable_hair_inpaint") and config.ENABLE_HAIR_INPAINT:
        rgb = _remove_hair_artifacts(rgb)

    result = Image.fromarray(rgb)
    if params.get("letterbox") or _looks_like_fundus(rgb):
        result = letterbox_resize(result, target=params.get("target_size", 896))
    return result


# ---------------------------------------------------------------------------
# Microscopy: most-informative-tile selection + Macenko stain normalization
# ---------------------------------------------------------------------------
def _select_informative_tile(rgb: np.ndarray, tile_size: int) -> np.ndarray:
    """For WSI-like inputs, scans a coarse grid of candidate tiles and keeps
    the one with the highest non-background ("tissue") pixel fraction,
    rather than downsampling the entire slide."""
    h, w = rgb.shape[:2]
    ty, tx = min(tile_size, h), min(tile_size, w)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    best_score, best_box = -1.0, (0, 0, tx, ty)
    step_y, step_x = max(1, ty // 2), max(1, tx // 2)
    for y in range(0, max(1, h - ty + 1), step_y):
        for x in range(0, max(1, w - tx + 1), step_x):
            tile = gray[y : y + ty, x : x + tx]
            tissue_fraction = float((tile < 235).mean())  # H&E background is near-white
            if tissue_fraction > best_score:
                best_score, best_box = tissue_fraction, (x, y, x + tx, y + ty)

    x0, y0, x1, y1 = best_box
    return rgb[y0:y1, x0:x1]


def _macenko_normalize(rgb: np.ndarray, beta: float = 0.15, alpha_percentile: float = 1.0) -> np.ndarray:
    """Compact Macenko (2009) H&E stain normalization against the preloaded
    config.STAIN_REFERENCE_MATRIX (site-calibrated if
    weights/stain_reference_matrix.npy exists, else the standard Macenko
    reference values -- see src/config.py's _load_stain_reference_matrix).
    Falls back to the untouched tile if the optical-density decomposition
    degenerates (e.g. near-blank tile with little stained tissue). Color
    deconvolution beyond this normalization step is left unapplied, per
    the doc marking it "optional"."""
    h, w = rgb.shape[:2]
    pixels = rgb.reshape(-1, 3).astype(np.float64)

    optical_density = -np.log10((pixels + 1.0) / 255.0)
    stained = optical_density[np.all(optical_density > beta, axis=1)]
    if stained.shape[0] < 10:
        return rgb

    _, eigvecs = np.linalg.eigh(np.cov(stained.T))
    top_eigvecs = eigvecs[:, -2:]  # two largest-variance directions span the stain plane
    if top_eigvecs[0, 0] < 0:
        top_eigvecs[:, 0] *= -1
    if top_eigvecs[0, 1] < 0:
        top_eigvecs[:, 1] *= -1

    projection = stained @ top_eigvecs
    angles = np.arctan2(projection[:, 1], projection[:, 0])
    min_angle = np.percentile(angles, alpha_percentile)
    max_angle = np.percentile(angles, 100 - alpha_percentile)

    v_min = top_eigvecs @ np.array([np.cos(min_angle), np.sin(min_angle)])
    v_max = top_eigvecs @ np.array([np.cos(max_angle), np.sin(max_angle)])
    stain_matrix = np.stack([v_min, v_max] if v_min[0] > v_max[0] else [v_max, v_min], axis=1)

    od_all = optical_density.T
    concentrations, _, _, _ = np.linalg.lstsq(stain_matrix, od_all, rcond=None)

    max_conc = np.percentile(concentrations, 99, axis=1)
    max_conc[max_conc == 0] = 1e-6
    concentrations = concentrations / max_conc[:, None] * _REFERENCE_MAX_CONCENTRATION[:, None]

    od_normalized = config.STAIN_REFERENCE_MATRIX @ concentrations
    normalized = 255.0 * np.exp(-od_normalized * np.log(10.0))
    normalized = np.clip(normalized, 0, 255).T.reshape(h, w, 3)
    return normalized.astype(np.uint8)


def preprocess_microscopy(image: Image.Image, subtype: str = None) -> Image.Image:
    """
    Args:
        image: RGB PIL Image (already universal_normalize()d).
        subtype: one of config.MICROSCOPY_SUBTYPES (currently just
            "histopathology" -- cytology is folded into it, see
            config.py's KNOWN LIMITATIONS note #2); defaults to
            config.STREAM_SUBTYPE_DEFAULTS[microscopy] if omitted or
            unrecognized.
    """
    default_subtype = config.STREAM_SUBTYPE_DEFAULTS[config.MODALITY_MICROSCOPY]
    subtype = subtype or default_subtype
    params = config.MICROSCOPY_PARAMS.get(subtype, config.MICROSCOPY_PARAMS[default_subtype])

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]

    if h * w > params["wsi_trigger_pixels"]:
        rgb = _select_informative_tile(rgb, params["tile_size"])
        h, w = rgb.shape[:2]

    # Macenko normalization is O(pixels) (SVD + lstsq over every stained
    # pixel); skip it above stain_norm_max_pixels even after tile
    # selection, to bound worst-case microscopy latency (Section 3.3).
    if params.get("enable_stain_norm", True) and h * w <= params["stain_norm_max_pixels"]:
        try:
            rgb = _macenko_normalize(rgb)
        except np.linalg.LinAlgError:
            pass  # keep the un-normalized tile rather than fail the query

    return Image.fromarray(rgb)


# ---------------------------------------------------------------------------
# Ultrasound: ROI crop (true ultrasound only) + optional speckle-reducing
# anisotropic diffusion
# ---------------------------------------------------------------------------
def _crop_ultrasound_roi(rgb: np.ndarray) -> np.ndarray:
    """Crops to the bounding box of non-black pixels, removing the black
    borders / UI overlays surrounding the fan-shaped scan region."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = gray > 10
    if not np.any(mask):
        return rgb
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    if (y1 - y0) < 0.3 * gray.shape[0] or (x1 - x0) < 0.3 * gray.shape[1]:
        return rgb  # degenerate crop -- keep the original instead
    return rgb[y0:y1, x0:x1]


def _anisotropic_diffusion(gray: np.ndarray, iterations: int, kappa: float = 30.0, gamma: float = 0.1) -> np.ndarray:
    """Perona-Malik anisotropic diffusion: speckle-reducing, edge-preserving
    filter. Tuned/validated for ultrasound speckle statistics only -- not
    applied to OCT (see config.ULTRASOUND_PARAMS)."""
    img = gray.astype(np.float32)
    for _ in range(iterations):
        north = np.roll(img, -1, axis=0) - img
        south = np.roll(img, 1, axis=0) - img
        east = np.roll(img, -1, axis=1) - img
        west = np.roll(img, 1, axis=1) - img

        c_n = np.exp(-((north / kappa) ** 2))
        c_s = np.exp(-((south / kappa) ** 2))
        c_e = np.exp(-((east / kappa) ** 2))
        c_w = np.exp(-((west / kappa) ** 2))

        img += gamma * (c_n * north + c_s * south + c_e * east + c_w * west)
    return np.clip(img, 0, 255).astype(np.uint8)


def preprocess_ultrasound(image: Image.Image, subtype: str = None) -> Image.Image:
    """
    Args:
        image: RGB PIL Image (already universal_normalize()d).
        subtype: one of config.ULTRASOUND_SUBTYPES ("ultrasound"/"oct");
            defaults to config.STREAM_SUBTYPE_DEFAULTS[ultrasound] if
            omitted or unrecognized.
    """
    default_subtype = config.STREAM_SUBTYPE_DEFAULTS[config.MODALITY_ULTRASOUND]
    subtype = subtype or default_subtype
    params = config.ULTRASOUND_PARAMS.get(subtype, config.ULTRASOUND_PARAMS[default_subtype])

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)

    if params.get("crop_roi", True):
        rgb = _crop_ultrasound_roi(rgb)

    if params.get("enable_speckle_filter") and config.ULTRASOUND_SPECKLE_FILTER_ENABLED:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = _anisotropic_diffusion(gray, config.ULTRASOUND_DIFFUSION_ITERATIONS)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    return Image.fromarray(rgb)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
_STREAM_PREPROCESSORS = {
    config.MODALITY_RADIOLOGY: preprocess_radiology,
    config.MODALITY_MACROSCOPIC: preprocess_macroscopic,
    config.MODALITY_MICROSCOPY: preprocess_microscopy,
    config.MODALITY_ULTRASOUND: preprocess_ultrasound,
}


def preprocess_image(image: Image.Image, coarse_stream: str, subtype: str = None, **kwargs) -> Image.Image:
    """
    Args:
        image: raw PIL Image as received by predict().
        coarse_stream: one of config.MODALITIES, from
            src.router_modality.route_modality() (or detect_modality()).
        subtype: one of config.STREAM_SUBTYPES[coarse_stream]; defaults to
            config.STREAM_SUBTYPE_DEFAULTS[coarse_stream] inside each
            stream's preprocessor if omitted or unrecognized.
        **kwargs: reserved for future per-call overrides; currently
            unused, accepted so new keys can be added without breaking
            existing call sites.
    Returns:
        A subtype-appropriately preprocessed RGB PIL Image, resolution-
        capped to the VLM processor's [MIN_PIXELS, MAX_PIXELS] budget. If
        coarse_stream is unrecognized, only universal normalization +
        resolution capping are applied (no modality-specific step runs).
    """
    image = universal_normalize(image)
    image = _cap_resolution(image)
    preprocessor = _STREAM_PREPROCESSORS.get(coarse_stream)
    if preprocessor is None:
        return image
    return preprocessor(image, subtype)
