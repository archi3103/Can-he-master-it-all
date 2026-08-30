"""
Stage -1: Volumetric/DICOM ingest -- runs before Stage 0 (universal_normalize)
and before PIL ever touches the path.

PIL cannot decode 3D NIfTI volumes, raw DICOM files, or folders of DICOM
slices at all -- previously any such input hit predict.py's corrupt-image
guard (PIL raising UnidentifiedImageError/OSError) and fell back to
config.FALLBACK_ANSWER_LETTER with no real image analysis. This module
decodes those formats into a single 2D RGB PIL Image so they flow into the
existing Stage 0-6 pipeline exactly like any flat 2D image.

Dispatch (see load_volume()):
    flat 2D image (.png/.jpg/...)        -> returns None; caller uses PIL as before
    .nii / .nii.gz                       -> nibabel, reoriented via as_closest_canonical
    .dcm / .dicom, or DICOM magic bytes  -> pydicom, single file
    directory                            -> pydicom, DICOM series (all slices)

Pixel-value handling (single source of correctness for HU-based windowing --
see config.RADIOLOGY_WINDOW_LOW_PERCENTILE/HIGH_PERCENTILE, whose docstring
in src/preprocessing.py notes true HU windowing needs raw DICOM data a
generic PIL image doesn't carry; this module is that raw data path):
    - RescaleSlope/RescaleIntercept applied before any windowing, so pixel
      values are real Hounsfield units (or the modality's native scale)
      rather than raw stored integers.
    - MONOCHROME1 (inverted grayscale -- 0 = white) is flipped so all
      output is MONOCHROME2-equivalent (0 = black), matching what every
      downstream heuristic in router_modality.py/preprocessing.py assumes.

Slice-selection policy for true 3D volumes (NIfTI, multi-frame DICOM, and
DICOM series folders): rather than a single arbitrary slice (which could
land on an uninformative edge slice) or decoding/downsampling the whole
volume (the VLM only accepts one 2D image, so most of that would be wasted
work), three slices at fixed relative depths (35%/50%/65%) are extracted
and tiled side-by-side into one 1x3 composite. This is a cheap, zero-model
heuristic -- not a learned key-slice selector -- deliberately, per this
codebase's latency-first design elsewhere (see config.py's feature-flag
section).
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

try:
    import nibabel as nib
except ImportError as exc:  # pragma: no cover - surfaced at import time
    raise ImportError(
        "volume_loader.py requires nibabel (see requirements.txt)"
    ) from exc

try:
    import pydicom
except ImportError as exc:  # pragma: no cover - surfaced at import time
    raise ImportError(
        "volume_loader.py requires pydicom (see requirements.txt)"
    ) from exc

from src import config

logger = logging.getLogger(__name__)


class VolumeLoadError(Exception):
    """Raised when a path was confidently identified as a volumetric/DICOM
    input (by extension or magic bytes) but could not actually be decoded
    (corrupt file, unsupported internal encoding, empty series folder,
    etc.). Callers (src/predict.py) should catch this and degrade to
    config.FALLBACK_ANSWER_LETTER, exactly like any other unreadable-image
    failure mode."""


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
_DICOM_SUFFIXES = {".dcm", ".dicom"}
# Extensions PIL already handles -- skip magic-byte sniffing for these so
# the overwhelmingly common flat-2D-image case pays zero extra I/O cost.
_KNOWN_RASTER_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp", ".ppm", ".pgm",
}


def _sniff_dicom_magic(path: Path) -> bool:
    """DICOM Part 10 files carry a 128-byte preamble followed by the magic
    string "DICM". Files exported without a preamble (implicit VR, common
    in some research dumps) won't match this and simply fall through to
    PIL, same as before this module existed -- no regression, just no
    extra detection for that rarer case."""
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except OSError:
        return False


def load_volume(path: Path) -> Optional[Image.Image]:
    """
    Args:
        path: filesystem path resolved by eval.py's _resolve_image_path()
            (or any other caller) -- may be a single file or a directory
            of DICOM slices.
    Returns:
        A 2D RGB PIL Image ready for Stage 0 (universal_normalize), if
        `path` was recognized as a volumetric/DICOM input this module
        handles. None if `path` is not one of those formats -- the caller
        should then fall back to opening it with PIL directly, unchanged.
    Raises:
        VolumeLoadError if `path` was confidently identified as
        volumetric/DICOM but could not be decoded (corrupt/unsupported).
        Never raises any other exception type.
    """
    try:
        is_dir = path.is_dir()
    except OSError as exc:
        raise VolumeLoadError(f"Cannot stat {path}: {exc}") from exc

    if is_dir:
        return _load_dicom_series(path)

    name_lower = path.name.lower()
    if name_lower.endswith(".nii") or name_lower.endswith(".nii.gz"):
        return _load_nifti(path)

    suffix = path.suffix.lower()
    if suffix in _DICOM_SUFFIXES:
        return _load_single_dicom(path)

    if suffix in _KNOWN_RASTER_SUFFIXES:
        return None

    if _sniff_dicom_magic(path):
        return _load_single_dicom(path)

    return None


# ---------------------------------------------------------------------------
# Shared pixel-value helpers
# ---------------------------------------------------------------------------
def _apply_rescale(arr: np.ndarray, ds) -> np.ndarray:
    """Converts stored pixel values to real-world units (Hounsfield units
    for CT) via DICOM's RescaleSlope/RescaleIntercept. Defaults (1, 0) are
    a no-op for modalities/files that don't carry these tags."""
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    if slope == 1.0 and intercept == 0.0:
        return arr
    return arr * slope + intercept


def _correct_monochrome1(arr: np.ndarray, ds) -> np.ndarray:
    """MONOCHROME1 stores 0 = white / max = black (inverted vs. the
    MONOCHROME2 convention every downstream grayscale heuristic in this
    codebase assumes) -- flip it in-range so 0 = black consistently."""
    if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
        return (arr.max() + arr.min()) - arr
    return arr


def _percentile_bounds(arr: np.ndarray) -> tuple:
    lo = float(np.percentile(arr, config.RADIOLOGY_WINDOW_LOW_PERCENTILE))
    hi = float(np.percentile(arr, config.RADIOLOGY_WINDOW_HIGH_PERCENTILE))
    return lo, hi


def _apply_window_to_uint8(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi <= lo:
        arr_min, arr_max = float(arr.min()), float(arr.max())
        span = (arr_max - arr_min) or 1.0
        return np.clip((arr - arr_min) / span * 255.0, 0, 255).astype(np.uint8)
    stretched = np.clip(arr, lo, hi)
    stretched = (stretched - lo) / (hi - lo) * 255.0
    return np.clip(stretched, 0, 255).astype(np.uint8)


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _volume_to_trislice_image(volume: np.ndarray) -> Image.Image:
    """volume: (H, W, depth) array, any numeric dtype, already
    rescaled/orientation-corrected. Extracts slices at
    config.VOLUME_TRISLICE_DEPTH_FRACTIONS, windows them all against one
    shared percentile range (computed across the extracted slices only --
    cheap, and keeps the three tiles visually consistent with each other),
    and tiles them horizontally into one 1x3 RGB composite."""
    depth = volume.shape[-1]
    indices = [
        _clamp(int(round(frac * (depth - 1))), 0, depth - 1)
        for frac in config.VOLUME_TRISLICE_DEPTH_FRACTIONS
    ]
    slices = [volume[..., i] for i in indices]

    sample = np.concatenate([s.ravel() for s in slices])
    lo, hi = _percentile_bounds(sample)

    tiles = [_apply_window_to_uint8(s, lo, hi) for s in slices]
    tiled = np.concatenate(tiles, axis=1)
    return Image.fromarray(tiled, mode="L").convert("RGB")


# ---------------------------------------------------------------------------
# NIfTI
# ---------------------------------------------------------------------------
def _load_nifti(path: Path) -> Image.Image:
    try:
        img = nib.load(str(path))
        # Reorients to the closest canonical (RAS+) axis ordering, so axis
        # 2 of the returned array is consistently the superior-inferior
        # (axial slice) axis regardless of how the file was acquired/stored.
        img = nib.as_closest_canonical(img)
        data = np.asarray(img.get_fdata(dtype=np.float32))
    except Exception as exc:  # noqa: BLE001 - any nibabel/numpy failure means "corrupt"
        raise VolumeLoadError(f"Cannot decode NIfTI file {path}: {exc}") from exc

    if data.ndim == 4:
        # 4D (e.g. fMRI/DTI time series or multi-volume acquisitions) --
        # a single representative volume is enough for a single VQA image.
        data = data[..., 0]
    if data.ndim == 2:
        data = data[:, :, np.newaxis]
    if data.ndim != 3:
        raise VolumeLoadError(f"Unsupported NIfTI array shape {data.shape} in {path}")
    if data.shape[-1] == 0:
        raise VolumeLoadError(f"NIfTI volume {path} has zero depth")

    return _volume_to_trislice_image(data)


# ---------------------------------------------------------------------------
# DICOM
# ---------------------------------------------------------------------------
def _is_color_dicom(ds) -> bool:
    photometric = str(getattr(ds, "PhotometricInterpretation", ""))
    return getattr(ds, "SamplesPerPixel", 1) >= 3 or photometric.startswith(("RGB", "YBR"))


def _load_single_dicom(path: Path) -> Image.Image:
    try:
        ds = pydicom.dcmread(str(path), force=True)
        arr = ds.pixel_array
    except Exception as exc:  # noqa: BLE001 - any pydicom/codec failure means "corrupt"
        raise VolumeLoadError(f"Cannot decode DICOM file {path}: {exc}") from exc

    if _is_color_dicom(ds):
        # Color DICOM (e.g. color Doppler ultrasound) is comparatively rare
        # in this pipeline's target modalities. Full YBR->RGB colorspace
        # conversion is intentionally not implemented here -- pixel data is
        # cast directly to uint8 RGB, which is correct for the common
        # already-RGB case and merely imperfect (not crash-inducing) for
        # YBR-encoded frames.
        if arr.ndim == 4:  # multi-frame color cine loop -- take the middle frame
            arr = arr[arr.shape[0] // 2]
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")

    arr = _apply_rescale(arr.astype(np.float32), ds)
    arr = _correct_monochrome1(arr, ds)

    if arr.ndim == 3:
        # Multi-frame grayscale packed into one file (frames, rows, cols)
        # -- treat as its own volume and reuse the tri-slice policy.
        volume = np.moveaxis(arr, 0, -1)
        return _volume_to_trislice_image(volume)

    lo, hi = _percentile_bounds(arr)
    gray = _apply_window_to_uint8(arr, lo, hi)
    return Image.fromarray(gray, mode="L").convert("RGB")


def _dicom_instance_number(ds):
    try:
        return float(ds.InstanceNumber)
    except Exception:
        return None


def _dicom_z_position(ds):
    try:
        return float(ds.ImagePositionPatient[2])
    except Exception:
        return None


def _sort_dicom_slices(items: list) -> list:
    """items: list of (ds, arr) tuples, initially in filesystem/name order
    (a reasonable fallback if neither tag below is usable). Prefers a
    single consistent ordering key across the whole series rather than
    mixing keys per-slice, which could silently produce a scrambled
    ordering worse than just leaving filesystem order alone."""
    if all(_dicom_instance_number(ds) is not None for ds, _ in items):
        return sorted(items, key=lambda item: _dicom_instance_number(item[0]))
    if all(_dicom_z_position(ds) is not None for ds, _ in items):
        return sorted(items, key=lambda item: _dicom_z_position(item[0]))
    return items


def _load_dicom_series(folder: Path) -> Image.Image:
    try:
        candidate_files = sorted(p for p in folder.rglob("*") if p.is_file())
    except OSError as exc:
        raise VolumeLoadError(f"Cannot list DICOM series folder {folder}: {exc}") from exc

    datasets = []
    for f in candidate_files:
        try:
            ds = pydicom.dcmread(str(f), force=True)
            arr = ds.pixel_array
        except Exception:  # noqa: BLE001 - skip non-DICOM/unreadable files in the folder
            continue
        if _is_color_dicom(ds) or arr.ndim != 2:
            continue  # color/multi-frame members mixed into a series folder are unsupported
        datasets.append((ds, arr))

    if not datasets:
        raise VolumeLoadError(f"No readable single-frame grayscale DICOM slices found in {folder}")

    datasets = _sort_dicom_slices(datasets)

    processed = []
    for ds, arr in datasets:
        arr = _apply_rescale(arr.astype(np.float32), ds)
        arr = _correct_monochrome1(arr, ds)
        processed.append(arr)

    # A series folder should be homogeneous, but tolerate a stray
    # differently-shaped file (e.g. a localizer/scout image mixed in) by
    # keeping only the majority shape rather than failing the whole series.
    shape_counts = Counter(a.shape for a in processed)
    target_shape = shape_counts.most_common(1)[0][0]
    processed = [a for a in processed if a.shape == target_shape]

    volume = np.stack(processed, axis=-1)  # (H, W, depth)
    return _volume_to_trislice_image(volume)
