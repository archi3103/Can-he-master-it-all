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
    directory, channel-split filenames   -> PIL, 2x2 channel grid (see below)
    directory, otherwise                 -> pydicom, DICOM series (all slices)

Channel-split folders: some microscopy/multi-spectral sources (e.g. Human
Protein Atlas-style fluorescence exports) distribute a single sample as
one grayscale PNG per channel -- "<id>_red.png", "_green.png",
"_blue.png", and optionally "_yellow.png" -- rather than as DICOM. These
are parallel spectral VIEWS of the same imaging plane, not spatial depth
slices, so treating them as a DICOM Z-stack is wrong regardless of
whether pydicom could even parse them (it can't -- they're plain PNGs).
Folders matching this naming convention are detected before the
DICOM-series path is even attempted (any 1-4 of the four recognized
channel files is enough to confidently route here -- see
_find_channel_split_images) and composited into one fixed 2x2 grid (blue/
green top row, red/yellow bottom row) via _tile_color_channels_into_grid,
so the VLM inspects every available channel at full resolution, unmixed,
in a single forward pass -- see that function's docstring for the
graceful-degradation behavior when fewer than 4 channels are present.

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
work), the top few most "content-rich" slices are selected and tiled
side-by-side into one composite -- see "Vote-MI-inspired slice selection"
below. This is a cheap, zero-model heuristic -- not a learned key-slice
selector -- deliberately, per this codebase's latency-first design
elsewhere (see config.py's feature-flag section).

Vote-MI-inspired slice selection (_select_top_k_slice_indices): rather
than blindly taking the middle slice (or fixed 35%/50%/65% depths, this
module's original policy -- still kept as the tier-2 fallback below),
each slice is scored by two unsupervised signals, computed on a shared,
volume-wide percentile-stretched 0-255 scale so scores are comparable
across slices regardless of the scan's native intensity range (HU units,
16-bit DICOM, etc.):
    - intensity variance: a mostly-uniform air/background slice scores
      near zero; a slice cutting through actual anatomy has real
      contrast.
    - edge density: mean Sobel gradient magnitude -- a proxy for how much
      structural/anatomical detail (organ boundaries, lesion margins,
      bone edges) is visible in that slice.
Both signals are min-max normalized across the candidate slice pool, then
summed (config.VOLUME_EDGE_DENSITY_WEIGHT controls edge density's
relative weight) into one composite "informativeness" score per slice.
The top config.VOLUME_SLICE_SELECTION_COUNT slices are picked greedily by
that score, subject to a minimum index separation
(config.VOLUME_SLICE_MIN_SEPARATION_FRACTION x depth) so the selection
doesn't collapse onto a cluster of near-duplicate adjacent slices --
"representative" coverage of the volume, not just its single busiest
region. This is a heuristic voting signal inspired by representative-
slice-selection principles in multi-instance medical volume analysis, not
a literal reproduction of any specific published algorithm -- tune
against real labeled volumes before relying on it beyond "probably more
informative than a fixed depth fraction."

Graded, zero-crash fallback chain (_volume_to_representative_image): if
content-based scoring itself errors, slice selection falls back to the
original fixed-percentage (35%/50%/65%) heuristic; if compositing/tiling
then also fails, it falls back further to a single slice with a plain
percentile stretch (no tiling at all). Only if that single-slice path
also fails does an exception propagate out of this module -- callers
(src/predict.py) then substitute a blind gray canvas + text-only model
call as the final, outermost safety net (see predict_with_diagnostics()/
predict_by_prefix_score()'s module docstring in src/predict.py).
"""

import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError as exc:  # pragma: no cover - surfaced at import time
    raise ImportError(
        "volume_loader.py requires opencv-python-headless (see requirements.txt)"
    ) from exc

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
        # Channel-split detection runs unconditionally, before any DICOM
        # series parsing is even attempted: a real DICOM series folder
        # would essentially never contain a file matching
        # "..._<red|green|blue|yellow>.<ext>", so finding even one such
        # file is confident, cheap (filename-only) evidence that this
        # folder holds parallel spectral channels of one plane, not a
        # Z-stack -- graceful degradation to 1-3 available channels
        # included (see _tile_color_channels_into_grid).
        channels = _find_channel_split_images(path)
        if channels:
            return _tile_color_channels_into_grid(channels)
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


def _fixed_percentage_slice_indices(volume: np.ndarray) -> list:
    """The original, content-blind slice-selection policy (fixed relative
    depths) -- kept as the tier-2 fallback for when content-based scoring
    itself errors (see _volume_to_representative_image)."""
    depth = volume.shape[-1]
    return [
        _clamp(int(round(frac * (depth - 1))), 0, depth - 1)
        for frac in config.VOLUME_TRISLICE_DEPTH_FRACTIONS
    ]


def _slice_content_scores(volume: np.ndarray, indices) -> np.ndarray:
    """Per-slice (variance, edge-density) pair for each index in
    `indices`, computed on a shared volume-wide percentile-stretched 0-255
    scale so the two signals -- and every slice's scores -- are directly
    comparable. Returns an (len(indices), 2) array: column 0 = variance,
    column 1 = mean Sobel gradient magnitude."""
    sample = np.concatenate([volume[..., i].ravel() for i in indices])
    lo, hi = _percentile_bounds(sample)

    scores = np.empty((len(indices), 2), dtype=np.float64)
    for row, i in enumerate(indices):
        gray = _apply_window_to_uint8(volume[..., i], lo, hi).astype(np.float32)
        scores[row, 0] = gray.var()
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        scores[row, 1] = np.sqrt(gx**2 + gy**2).mean()
    return scores


def _min_max_normalize(arr: np.ndarray) -> np.ndarray:
    span = float(arr.max() - arr.min())
    if span <= 1e-6:
        return np.zeros_like(arr)
    return (arr - arr.min()) / span


def _select_top_k_slice_indices(volume: np.ndarray, k: int) -> list:
    """Vote-MI-inspired content-based slice selection -- see module
    docstring. Scores every slice along the depth axis (a coarse stride
    is used above config.VOLUME_SCORING_MAX_SLICES to bound worst-case
    latency on very deep volumes -- Sobel is cheap per-slice, but a
    1000+-slice CT series would otherwise scan the whole thing), then
    greedily picks the top `k` by composite score subject to a minimum
    index gap so selections don't cluster onto near-duplicate adjacent
    slices. Returns indices in ascending (anatomical) order."""
    depth = volume.shape[-1]
    if depth <= k:
        return list(range(depth))

    if depth <= config.VOLUME_SCORING_MAX_SLICES:
        candidate_indices = list(range(depth))
    else:
        stride = max(1, depth // config.VOLUME_SCORING_MAX_SLICES)
        candidate_indices = list(range(0, depth, stride))

    raw_scores = _slice_content_scores(volume, candidate_indices)
    variance_norm = _min_max_normalize(raw_scores[:, 0])
    edge_norm = _min_max_normalize(raw_scores[:, 1])
    composite = variance_norm + config.VOLUME_EDGE_DENSITY_WEIGHT * edge_norm

    ranked_positions = np.argsort(composite)[::-1]  # descending, best first
    min_gap = max(1, int(round(config.VOLUME_SLICE_MIN_SEPARATION_FRACTION * depth)))

    selected = []
    for pos in ranked_positions:
        idx = candidate_indices[pos]
        if all(abs(idx - s) >= min_gap for s in selected):
            selected.append(idx)
        if len(selected) == k:
            break
    if len(selected) < k:
        # The min-separation constraint left too few candidates for this
        # depth (e.g. a short, dense volume) -- fill the rest by score
        # alone, ignoring the gap, rather than returning fewer than k.
        for pos in ranked_positions:
            idx = candidate_indices[pos]
            if idx not in selected:
                selected.append(idx)
            if len(selected) == k:
                break
    return sorted(selected)


def _tile_slices(volume: np.ndarray, indices: list) -> Image.Image:
    """volume: (H, W, depth) array, any numeric dtype, already
    rescaled/orientation-corrected. Windows the slices at `indices`
    against one shared percentile range (computed across just those
    slices -- cheap, and keeps the tiles visually consistent with each
    other), and tiles them horizontally into one RGB composite. Works
    identically for a single index (no visible "tiling", just that one
    slice) or several -- the multi-slice and single-slice tiers of
    _volume_to_representative_image's fallback chain both go through
    this same function."""
    slices = [volume[..., i] for i in indices]

    sample = np.concatenate([s.ravel() for s in slices])
    lo, hi = _percentile_bounds(sample)

    tiles = [_apply_window_to_uint8(s, lo, hi) for s in slices]
    tiled = np.concatenate(tiles, axis=1)
    return Image.fromarray(tiled, mode="L").convert("RGB")


def _volume_to_representative_image(volume: np.ndarray) -> Image.Image:
    """Vote-MI-inspired representative-slice selection wrapped in a
    graded, zero-crash fallback chain -- see module docstring for the
    full policy. Only propagates an exception (to src/predict.py's
    blind-gray-canvas-and-text-only last resort) if even the innermost
    single-slice/plain-percentile-stretch tier fails."""
    depth = volume.shape[-1]
    k = min(config.VOLUME_SLICE_SELECTION_COUNT, depth)

    try:
        indices = _select_top_k_slice_indices(volume, k)
    except Exception as exc:  # noqa: BLE001 - content scoring failed; fall back to the blind heuristic
        logger.info(
            "Vote-MI-inspired content scoring failed (%s); falling back to fixed-percentage slice indices.", exc
        )
        indices = _fixed_percentage_slice_indices(volume)

    try:
        return _tile_slices(volume, indices)
    except Exception as exc:  # noqa: BLE001 - tiling/compositing failed; fall back to a single slice
        logger.info(
            "Multi-slice tiling failed (%s); falling back to a single slice with a plain percentile stretch.", exc
        )

    try:
        single_index = indices[len(indices) // 2] if indices else depth // 2
        return _tile_slices(volume, [single_index])
    except Exception as exc:  # noqa: BLE001 - even the single-slice path failed; last resort before raising
        logger.info("Single-slice fallback failed (%s); using the raw middle slice directly.", exc)
        mid = depth // 2
        gray = _apply_window_to_uint8(volume[..., mid], *_percentile_bounds(volume[..., mid]))
        return Image.fromarray(gray, mode="L").convert("RGB")


# ---------------------------------------------------------------------------
# NIfTI
# ---------------------------------------------------------------------------
def _load_nifti(path: Path) -> Image.Image:
    try:
        # mmap=False: nibabel memory-maps uncompressed .nii files by
        # default (nib.load(path) alone), which is fine on a local disk
        # but pathological on a network/FUSE-backed filesystem -- e.g. a
        # Colab Google Drive mount. get_fdata()'s page-fault-driven reads
        # then turn into many small synchronous round-trips to Drive
        # instead of one sequential read, and their latency is highly
        # inconsistent file-to-file depending on Drive's cache/API state
        # at that moment -- this is what turns "read a 40MB file" into
        # anywhere from ~0.2s to several minutes for no reason visible
        # from file size alone. mmap=False forces a plain, fully-buffered
        # read instead, which FUSE filesystems handle far more
        # predictably. Immaterial on a local disk either way.
        img = nib.load(str(path), mmap=False)
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

    return _volume_to_representative_image(data)


# ---------------------------------------------------------------------------
# Channel-split microscopy folders (e.g. Human Protein Atlas-style
# multi-spectral/fluorescence exports: one grayscale PNG per channel,
# named "<sample_id>_<channel>.png", rather than a DICOM Z-stack).
# ---------------------------------------------------------------------------
_CHANNEL_SUFFIX_PATTERN = re.compile(r"_(red|green|blue|yellow)(?: \(\d+\))?\.\w+$", re.IGNORECASE)

# Fixed 2x2 layout: (row, col) per channel. Deliberately fixed regardless
# of which channels are actually present for a given sample (see
# _tile_color_channels_into_grid) -- a missing channel leaves its
# quadrant blank rather than reflowing the grid, so "top-right is always
# green" stays a stable convention across every sample.
_CHANNEL_GRID_LAYOUT = {
    "blue": (0, 0),    # top-left
    "green": (0, 1),   # top-right
    "red": (1, 0),     # bottom-left
    "yellow": (1, 1),  # bottom-right
}
_CHANNEL_GRID_BORDER_PX = 4
_CHANNEL_GRID_BORDER_COLOR = (40, 40, 40)  # dark gray separator, also the empty-quadrant fill


def _find_channel_split_images(folder: Path) -> dict:
    """Filename-only scan (no file reads) for '<id>_<channel>.<ext>'
    members -- cheap enough to run unconditionally before deciding whether
    a directory is a DICOM series at all. Tolerates the same duplicate-
    download ' (<n>)' suffix eval.py's _resolve_image_path() has to
    handle. Returns e.g. {"red": Path(...), "green": Path(...), ...} --
    1 to 4 entries; missing channels are simply absent from the dict."""
    channels = {}
    try:
        entries = list(folder.iterdir())
    except OSError:
        return channels
    for f in entries:
        if not f.is_file():
            continue
        match = _CHANNEL_SUFFIX_PATTERN.search(f.name)
        if match:
            channels.setdefault(match.group(1).lower(), f)
    return channels


def _tile_color_channels_into_grid(channels: dict) -> Image.Image:
    """Arranges up to 4 single-channel microscopy PNGs (red/green/blue/
    yellow -- parallel spectral views of the SAME imaging plane, not
    spatial depth slices, so no volumetric/Z-stack handling applies here)
    into one fixed 2x2 grid composite: blue top-left, green top-right,
    red bottom-left, yellow bottom-right -- so the VLM sees every
    available spectral channel at full resolution, unmixed, in a single
    forward pass, rather than only a blended pseudo-color guess at which
    channel contributed what.

    Graceful degradation: works with any 1-4 of the channels present
    (e.g. a sample missing its yellow/ER channel) -- a missing quadrant
    is filled with a flat border-colored placeholder rather than
    reflowing the grid to 1x2/1x3, so the layout (and therefore what each
    quadrant means) stays identical regardless of which channels a given
    sample happens to ship.

    Every channel is resized to a fixed config.VOLUME_CHANNEL_GRID_
    CELL_SIZE square before tiling -- both to normalize any (unexpected,
    for a real same-plane export) size mismatch across channels, and to
    keep the finished grid comfortably under config.MAX_PIXELS on its
    own: a full-native-resolution grid was measured landing right at the
    MAX_PIXELS cap (896x896 wasn't much smaller than fundus/microscopy
    channel exports commonly ship), which pushed per-query vision-token
    count -- and inference time -- to right at/over
    config.INFERENCE_TIMEOUT_SECONDS even with no other load on the GPU.
    This trades some per-channel resolution for meaningfully fewer
    vision tokens.

    Args:
        channels: dict like {"red": Path(...), "green": Path(...), ...}
            (from _find_channel_split_images) -- 1 to 4 entries.
    Returns:
        A single RGB PIL Image: the bordered 2x2 grid.
    Raises:
        VolumeLoadError if any present channel's file can't be decoded.
    """
    def _read_one_channel(item):
        name, path = item
        return name, Image.open(path).convert("RGB")

    try:
        # Same rationale as _load_dicom_series's parallel reads: on a
        # network/FUSE-backed filesystem each file open carries real
        # round-trip latency, so overlapping the (at most 4) reads is
        # worth it even at this small scale.
        with ThreadPoolExecutor(max_workers=len(channels)) as pool:
            images = dict(pool.map(_read_one_channel, channels.items()))
    except Exception as exc:  # noqa: BLE001 - any PIL failure means "corrupt"
        raise VolumeLoadError(f"Cannot decode channel image(s) in {channels}: {exc}") from exc

    cell_size = config.VOLUME_CHANNEL_GRID_CELL_SIZE
    cell_w = cell_h = cell_size

    def _cell_for(name: str) -> Image.Image:
        img = images.get(name)
        if img is None:
            return Image.new("RGB", (cell_w, cell_h), _CHANNEL_GRID_BORDER_COLOR)
        if img.size != (cell_w, cell_h):
            img = img.resize((cell_w, cell_h), Image.LANCZOS)
        return img

    border = _CHANNEL_GRID_BORDER_PX
    grid_w = cell_w * 2 + border * 3
    grid_h = cell_h * 2 + border * 3
    grid = Image.new("RGB", (grid_w, grid_h), _CHANNEL_GRID_BORDER_COLOR)

    for name, (row, col) in _CHANNEL_GRID_LAYOUT.items():
        x = border + col * (cell_w + border)
        y = border + row * (cell_h + border)
        grid.paste(_cell_for(name), (x, y))

    return grid


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
        return _volume_to_representative_image(volume)

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


def _read_one_dicom_slice(path: Path):
    """Reads and validates a single candidate DICOM slice file. Returns
    (ds, arr) on success, or None on any failure (non-DICOM file,
    corrupt, color/multi-frame member) -- never raises, so this can be
    safely mapped over many files concurrently without one bad file
    aborting the whole series."""
    try:
        ds = pydicom.dcmread(str(path), force=True)
        arr = ds.pixel_array
    except Exception:  # noqa: BLE001 - skip non-DICOM/unreadable files in the folder
        return None
    if _is_color_dicom(ds) or arr.ndim != 2:
        return None  # color/multi-frame members mixed into a series folder are unsupported
    return ds, arr


def _load_dicom_series(folder: Path) -> Image.Image:
    try:
        candidate_files = sorted(p for p in folder.rglob("*") if p.is_file())
    except OSError as exc:
        raise VolumeLoadError(f"Cannot list DICOM series folder {folder}: {exc}") from exc

    # Reading is I/O-bound (each file is a separate open+read) and
    # embarrassingly parallel -- on a local disk this barely matters, but
    # on a network/FUSE-backed filesystem (e.g. a Colab Google Drive
    # mount) each file open carries real round-trip latency, and a
    # 275-slice series read one file at a time was measured taking ~20s
    # there. A thread pool overlaps those round-trips instead of
    # serializing them; pydicom's file read + numpy pixel-array decode
    # both release the GIL for their I/O-bound portions, so this
    # genuinely parallelizes, not just in appearance. ThreadPoolExecutor.
    # map() preserves input order in its results regardless of which
    # worker finishes first, so downstream ordering (before
    # _sort_dicom_slices takes over) is unaffected.
    with ThreadPoolExecutor(max_workers=config.VOLUME_DICOM_READ_WORKERS) as pool:
        results = list(pool.map(_read_one_dicom_slice, candidate_files))
    datasets = [r for r in results if r is not None]

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
    return _volume_to_representative_image(volume)
