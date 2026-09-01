# Unified Medical Multi-Modal VQA System
## Technical Architecture & Production Design Document
### "Can He Master It All?" — Cross-Modality Medical Visual Question Answering

---

## 0. Problem Framing & Design Philosophy

The core difficulty of this challenge is **not** building a strong VQA model — it's building a system that is *modality-agnostic at inference time* while still being *modality-aware internally*. A CT slice, a dermoscopy image, and an H&E-stained histology tile share almost nothing in low-level visual statistics (contrast, color distribution, texture frequency, spatial scale), yet the same `predict()` function must handle all of them with a single, low-latency code path.

Given the scoring formula:

```
Score = Accuracy - (k * Avg Inference Time)
```

the design must jointly optimize **two competing axes**:

1. **Accuracy** → favors bigger models, ensembling, multi-pass reasoning, test-time augmentation.
2. **Latency** → favors a single forward pass, small/quantized models, short generation length.

The architecture below resolves this tension via **implicit routing** (cheap, deterministic, non-learned classification of modality and query intent) feeding into **one shared VLM backbone** rather than an ensemble of modality-specific models — because on a single RTX 6000 Ada (48GB VRAM), loading multiple large VLMs simultaneously is VRAM-feasible but latency-hostile if you invoke more than one per query. We use routing to select *prompts and preprocessing*, not to select *different heavyweight models*, except for one lightweight, currently-unimplemented, optional CNN classifier used purely for routing (see Section 1.2).

---

## 1. System Architecture & High-Level Design

### 1.1 Architectural Diagram (Conceptual Flow)

```
                         ┌─────────────────────────────┐
                         │   Raw Input: Image/Path + Query │
                         │   + Choices [A, B, C, D]     │
                         └──────────────┬───────────────┘
                                        │
                     ┌──────────────────▼───────────────────┐
                     │  STAGE -1: Volumetric/DICOM Ingest      │
                     │  Format dispatch (magic bytes/path):    │
                     │  .nii/.nii.gz (nibabel, RAS+ canonical),│
                     │  DICOM single/series (pydicom, HU       │
                     │  rescale + MONOCHROME1, parallel reads),│
                     │  RGBY channel-split folders -> 2x2 grid │
                     │  Vote-MI content-scored slice selection │
                     │  (variance + edge density) with a       │
                     │  graded fallback chain; not a volumetric│
                     │  format -> returns None, PIL handles it │
                     │  as before (Section 1.5)                │
                     └──────────────────┬───────────────────┘
                                        │
                     ┌──────────────────▼───────────────────┐
                     │  STAGE 0: Universal Normalization       │
                     │  RGBA/alpha stripping, 16-bit/float     │
                     │  percentile stretch -> clean 8-bit RGB  │
                     └──────────────────┬───────────────────┘
                                        │
                     ┌──────────────────▼───────────────────┐
                     │  STAGE 1: Hierarchical Modality Router │
                     │  (pure heuristics -- cv2/numpy; no      │
                     │   learned model loaded today)           │
                     │  1a. Coarse stream (unchanged):         │
                     │      Radiology / Macroscopic /          │
                     │      Microscopy / Ultrasound            │
                     │  1b. Stream-conditional subtype:         │
                     │      12 OmniMedVQA-style modalities      │
                     │      (see Section 1.4)                   │
                     └──────────────────┬───────────────────┘
                                        │
                     ┌──────────────────▼───────────────────┐
                     │  STAGE 2: Subtype-Specific Preproc     │
                     │  - Radiology: windowing, per-subtype   │
                     │    CLAHE (xray/mri/ct)                 │
                     │  - Macroscopic: color norm, optional   │
                     │    hair inpaint / specular suppression,│
                     │    LETTERBOX (not crop) for fundus/iri │
                     │  - Microscopy: stain norm, tiling       │
                     │  - Ultrasound: ROI crop (ultrasound     │
                     │    only, skipped for OCT), optional     │
                     │    speckle denoise                      │
                     └──────────────────┬───────────────────┘
                                        │
                     ┌──────────────────▼───────────────────┐
                     │  STAGE 3: Query-Intent Classifier      │
                     │  Tracks: Diagnostic / Spatial /        │
                     │          Severity / Modality-ID /      │
                     │          Differential                  │
                     │  (regex + sentence-embedding fallback) │
                     └──────────────────┬───────────────────┘
                                        │
                     ┌──────────────────▼───────────────────┐
                     │  STAGE 4: Prompt Constructor             │
                     │  System prompt = f(coarse_stream, track) │
                     │  (subtype not yet threaded into the       │
                     │   prompt -- see Section 1.4 note)          │
                     └──────────────────┬───────────────────┘
                                        │
                     ┌──────────────────▼───────────────────┐
                     │  STAGE 5: Frozen/Fine-Tuned VLM Core    │
                     │  Qwen3-VL-4B-Instruct (native fp16 by     │
                     │    default, fits 10GB submission size     │
                     │    limit; opt-in 4-bit/8-bit bitsandbytes │
                     │    quantization via config.QUANTIZATION_  │
                     │    MODE for constrained-VRAM deployments) │
                     │  + one general-purpose LoRA adapter      │
                     │    ("medvqa"), attached via PeftModel.    │
                     │    from_pretrained() when present         │
                     └──────────────────┬───────────────────┘
                                        │
                     ┌──────────────────▼───────────────────┐
                     │  STAGE 6: Constrained Decoding           │
                     │  Logit mask → {A, B, C, D} only          │
                     │  (argmax over each letter's best-scoring  │
                     │   surface-variant token id)                │
                     └──────────────────┬───────────────────┘
                                        │
                     ┌──────────────────▼───────────────────┐
                     │  STAGE 7: CSV Writer                     │
                     │  query_id, answer, inference_time        │
                     └───────────────────────────────────────┘
```

### 1.2 Modality-Stream Preprocessing Router

Rather than relying on the VLM to infer modality implicitly from raw pixels (which wastes capacity and adds variance), we insert a **cheap, dedicated router** upfront (`src/router_modality.py`). As implemented, this is a two-level, purely heuristic router — **no learned model is loaded today**:

- **Coarse stream (`detect_modality()`, unchanged since first implementation):** a weighted-feature scorer over cheap cv2/numpy image statistics — HSV saturation mean/std, an inter-channel grayscale-deviation score, border darkness (mean intensity of the outer ~3% frame), a Laplacian-variance texture/speckle indicator, hue-histogram Shannon entropy, hue-range fractions for H&E purple/pink, IHC blue, and skin-tone/red, a dark-corner/bright-center circular-vignette detector, and PIL-mode / DICOM-tag signature checks. These feed a linear weighted score per coarse bucket (Radiology, Macroscopic, Microscopy, Ultrasound); the highest-scoring bucket wins.

- **Stream-conditional subtype classifier (added for 12-modality scaling — see Section 1.4):** once the coarse stream is known, a second, stream-specific heuristic function further classifies into one of that stream's finer-grained subtypes (e.g. radiology → xray/mri/ct). These reuse aspect-ratio, circular-FOV, edge-sharpness, specular-highlight, and "hairiness" signals not used at the coarse level. `route_modality(image)` runs both levels and returns `(coarse_stream, subtype)`.

- **Learned router (designed, not implemented):** `config.USE_LEARNED_ROUTER` gates an optional MobileNetV3-Small/EfficientNet-B0-class booster. As shipped, this is a deliberate stub — `_learned_router_boost()` raises `NotImplementedError` if ever invoked — no checkpoint ships with the repo, and the flag defaults to `False`, so it adds zero import-time or per-query cost. This differs from earlier drafts of this document, which described the learned router as if evaluated on every query (`<5ms on GPU`); as implemented, it simply never runs.

**Recommendation (unchanged):** heuristic-only is the shipped default; only enable the learned-router path after validation shows it's worth the added latency and after an actual checkpoint is trained and placed at `config.MODALITY_CLASSIFIER_PATH`.

Router output feeds two downstream systems **asymmetrically**: **(a)** the preprocessing pipeline receives both the coarse stream *and* the subtype; **(b)** the prompt constructor (Section 2.3) currently receives only the coarse stream — subtype is not yet threaded into system-prompt selection. See Section 1.4 for the full subtype taxonomy and this gap.

**Preprocessing pipeline (as implemented, `src/preprocessing.py`):**

Every image first passes through two universal steps, run before any modality-specific logic:

1. `universal_normalize()` — flattens any source format to a clean 8-bit RGB image: RGBA/LA/palette-with-transparency is alpha-composited onto a white background (never a naive channel-drop, which would leave unpremultiplied color data behind transparent regions); 16-bit or floating-point grayscale (PIL modes `I`, `I;16`, `I;16B`, `I;16L`, `F` — common for exported DICOM/TIFF) gets a robust 0.5–99.5 percentile stretch to 8-bit, not a naive min-max.
2. `_cap_resolution()` — bounds total pixel count into `[MIN_PIXELS, MAX_PIXELS]` (Section 4.3) *before* any modality-specific op runs, so CLAHE, stain normalization, hair inpainting, etc. never operate on an unbounded raw image. This is in addition to, not a replacement for, the HF processor's own `min_pixels`/`max_pixels` resize.

**Modality-specific preprocessing table** (subtype-parameterized via `config.CLAHE_PARAMS` / `MACROSCOPIC_PARAMS` / `MICROSCOPY_PARAMS` / `ULTRASOUND_PARAMS`):

| Coarse Stream | Subtypes | Detection Signal | Preprocessing |
|---|---|---|---|
| Radiology | xray, mri, ct | Low saturation, high grayscale entropy, dark background (coarse); aspect-squareness, circular gantry FOV, background darkness, edge sharpness (subtype) | Percentile-based intensity windowing (soft-tissue/bone heuristic stand-in — see note below), CLAHE with **per-subtype** clip-limit/tile-grid from `CLAHE_PARAMS`, aspect-preserved padded resize to `RADIOLOGY_TARGET_SIZE` |
| Macroscopic | gross_pathology, dermoscopy, fundus, endoscopy, colposcopy, iri | High saturation, skin-tone/red-hue clusters, circular vignette (coarse); specular-highlight fraction, "hairiness" score (subtype) | Gray-world color-constancy normalization; **optional** hair/artifact inpainting (dermoscopy only, gated by `ENABLE_HAIR_INPAINT`, default off); **optional** HSV-based specular-highlight suppression (endoscopy/colposcopy, gated by `ENABLE_SPECULAR_SUPPRESSION`, default off); **letterbox resize, not center-crop**, for vignetted subtypes (fundus, iri) — deliberately chosen over cropping so peripheral pathology is never discarded |
| Microscopy | histopathology (cytology folded in — see Section 1.4) | High color variance, tiled/repeating texture, purple-pink (H&E) or blue (IHC) staining | Most-informative-tile selection above `MICROSCOPY_WSI_TRIGGER_PIXELS`; Macenko stain normalization against `config.STAIN_REFERENCE_MATRIX` (loaded from `weights/stain_reference_matrix.npy` if present, else the standard published Macenko reference vectors), **skipped above `STAIN_NORM_MAX_PIXELS`** to bound worst-case latency; color deconvolution beyond normalization remains unimplemented ("optional") |
| Ultrasound | ultrasound, oct | Fan/sector-shaped ROI + black borders (ultrasound); horizontal layer-banding, no corner masking (oct) | ROI crop to remove UI overlays/black borders — **ultrasound only**, skipped for OCT (structurally different, not fan-masked); speckle-reducing Perona-Malik anisotropic diffusion, **optional**, gated by `ULTRASOUND_SPECKLE_FILTER_ENABLED` (default off) — applies to ultrasound only, never to OCT (the filter is tuned/validated for ultrasound speckle statistics) |

**Note on radiology windowing:** the intensity windowing above is a percentile-clip-and-rescale stand-in for true Hounsfield-unit windowing; a generic PIL image doesn't carry the raw DICOM pixel data (rescale slope/intercept) that true HU windowing requires.

### 1.3 Why a Single Shared Backbone, Not an Ensemble

A naive "MoE of VLMs" (one full VLM per modality) would require either (a) loading 4 large models into VRAM simultaneously (feasible on 48GB — even unquantized, ~8-9GB each in fp16 for the deployed 4B model — but adds complexity in swapping CUDA contexts) or (b) dynamically loading/unloading weights per query (catastrophic for the latency term in the scoring formula). Instead:

- **One shared VLM backbone** stays resident in VRAM at all times (pre-loaded globally).
- **Modality/intent-specific behavior is injected via prompting and lightweight LoRA adapter switching**, not full model swapping. LoRA adapters (a few hundred MB each) can be hot-swapped in <50ms via PEFT's `set_adapter()`, which is far cheaper than reloading a full checkpoint.

**As implemented today, exactly one general-purpose adapter is loaded and active, and it's optional:** `config.LORA_ADAPTER_NAME = "medvqa"`, attached in `src/model_loader.py` via `peft.PeftModel.from_pretrained(model, config.LORA_PATH, adapter_name=...)` **only if `config.LORA_PATH` exists and is non-empty**. This wraps `model` in a `PeftModel` rather than mutating it in place — `model` is only reassigned to that wrapper on success (Python never reaches the assignment if the call raises), so a failed load leaves the original base model bound and untouched, no partial/half-wrapped state possible. `src/train_lora.py` (Section 3.2) is the script that produces an adapter at that path; until it's been run, that directory won't exist for most setups, and rather than hard-fail the whole import, the loader logs a warning and continues on the base model unmodified — the same fallback applies if a directory is present but the adapter fails to load (e.g. one trained against a different model size, Section 3.1's migration notes). `PeftModel` preserves the same `model(**inputs)` / `.generate()` / `.device` / `.eval()` surface every downstream call site (`decode.py`, `prefix_score.py`, the warm-up pass) relies on, so nothing downstream needed to change for this. The multi-adapter hot-swap capability described above is architecturally supported (`PeftModel.set_adapter()` is the mechanism, and it's already called once per load) but not yet exercised for *switching* — no second (e.g. microscopy-specialized) adapter ships or is switched at runtime.

### 1.4 Hierarchical Subtype Routing (12-Modality Scaling)

To extend coverage from the original 4 coarse streams toward the ~12-modality breadth of OmniMedVQA-style datasets without touching the validated coarse router, `src/router_modality.py` implements a second routing level:

```python
def route_modality(image: Image.Image):
    coarse_stream = detect_modality(image)   # unchanged Stage 1 router
    ...                                        # stream-conditional subtype classifier
    return coarse_stream, subtype
```

`detect_coarse_stream` is a plain alias for `detect_modality` — both names point at the same, unmodified function; the alias exists only so code (and this document) can refer to "the coarse router" by that name without a second implementation.

**Subtype taxonomy** (`config.STREAM_SUBTYPES`):

| Coarse stream | Subtypes | Classifier | Key signals |
|---|---|---|---|
| radiology | xray, mri, ct | `_classify_radiology_subtype()` | aspect-squareness, circular gantry-FOV detection, black-background fraction, edge sharpness |
| macroscopic | gross_pathology, dermoscopy, fundus, endoscopy, colposcopy, iri | `_classify_macroscopic_subtype()` | vignette, saturation, reddish-hue fraction, specular-highlight fraction, blackhat "hairiness" score |
| microscopy | histopathology | `_classify_microscopy_subtype()` | stub — single valid subtype (cytology folded in, see limitation #2 below) |
| ultrasound | ultrasound, oct | `_classify_ultrasound_subtype()` | black-corner darkness, row-wise band-intensity structure, aspect ratio |

Every classifier is a cheap, zero-VRAM, **unvalidated heuristic starting point** — none have been checked against labeled samples. `route_modality()` validates each classifier's output against `config.STREAM_SUBTYPES[coarse_stream]` and falls back to `config.STREAM_SUBTYPE_DEFAULTS` on anything unrecognized, so a bad subtype call only ever selects a suboptimal-but-harmless preprocessing preset (Section 1.2's table) — it never changes the coarse stream or crashes the pipeline.

**Known limitations** (documented in `config.py`, next to the routing tables they concern, and echoed in the relevant `_classify_*_subtype()` docstrings):

1. **Coarse-router misrouting risk for near-grayscale macroscopic subtypes.** `detect_modality()` leans heavily on saturation/grayscale-deviation to separate "radiology" from "macroscopic." Both IRI (near-grayscale, single-wavelength infrared reflectance imaging) and OCT (near-grayscale cross-sectional B-scan) are low-saturation modalities that risk being misrouted into "radiology" before any subtype classifier ever runs — a pre-existing constraint of the coarse router (deliberately left unmodified), accepted as bounded risk pending validation.
2. **Cytology is folded into `histopathology`, not a separate subtype.** Preprocessing treatment (Macenko normalization + tissue-fraction tile selection) is effectively identical for both; a future split has cheap groundwork already in `_select_informative_tile()`'s tissue-fraction metric (cytology's sparse cell-cluster pattern should show a measurably lower tissue fraction than dense histopathology).
3. **Subtype is not threaded into prompt composition.** `build_system_prompt()` (Section 2.3) is keyed only by `config.MODALITIES` (the 4 coarse streams); an OCT B-scan and an ultrasound image currently receive the *same* "This is an ultrasound image..." modality-context sentence. Extending `MODALITY_PROMPTS` to be subtype-aware is a natural next step, not yet done.

### 1.5 Stage -1: Volumetric/DICOM Ingest

PIL cannot decode 3D NIfTI volumes, raw DICOM files, or folders of DICOM slices at all. Originally, any such input hit `predict()`'s corrupt-image guard (PIL raising `UnidentifiedImageError`/`OSError`) and silently fell back to `config.FALLBACK_ANSWER_LETTER`, with no real image analysis. `src/volume_loader.py` implements a dedicated ingest stage, **Stage -1**, that runs before PIL ever touches the path — before Stage 0 — and decodes these formats into a single 2D RGB PIL Image that flows into the existing Stage 0-6 pipeline exactly like any flat 2D image. It is invoked from `predict.py`'s `_load_input_image()` helper, shared by every entry point (`predict()`, `predict_with_diagnostics()`, `predict_by_prefix_score()` — Section 5.2).

**Format dispatch (`load_volume(path)`):**

| Input | Handling |
|---|---|
| Flat 2D image (`.png`/`.jpg`/`.bmp`/`.tif`/etc.) | Returns `None` immediately — zero extra I/O cost; caller opens it with PIL as before |
| `.nii` / `.nii.gz` | `nibabel.load(path, mmap=False)`, reoriented via `nib.as_closest_canonical()` so axis 2 of the returned array is consistently the superior-inferior (axial) axis regardless of acquisition orientation |
| `.dcm` / `.dicom`, or a file with unrecognized extension whose first 128+4 bytes match the DICOM Part 10 magic (`DICM`) | `pydicom.dcmread(path, force=True)`, single file (including multi-frame single-file DICOM, treated as its own volume) |
| A directory containing `<id>_red.png` / `_green.png` / `_blue.png` / `_yellow.png`-style filenames (any 1-4 of the four) | **Not** treated as a DICOM series at all — these are parallel spectral/fluorescence *views of the same imaging plane* (e.g. Human Protein Atlas exports), not spatial depth slices. Composited via `_tile_color_channels_into_grid()` (below) |
| Any other directory | `pydicom.dcmread()` on every contained file (recursively), keeping only single-frame grayscale slices, geometrically sorted, treated as a DICOM series |

`mmap=False` is not a minor detail: nibabel memory-maps uncompressed `.nii` files by default, which is fine on local disk but pathological on a network/FUSE-backed filesystem (a Colab Google Drive mount, in particular) — `get_fdata()`'s page-fault-driven reads turn into many small synchronous round-trips instead of one sequential read, and were observed taking up to several minutes for a ~40MB file with no correlation to file size. `mmap=False` forces a plain, fully-buffered read instead.

**Pixel-value correctness (the "raw DICOM data" true-HU-windowing gap noted in Section 1.2's preprocessing table is closed here, not there):**
- `RescaleSlope`/`RescaleIntercept` are applied before any windowing, so pixel values become real Hounsfield units (or the modality's native scale) rather than raw stored integers.
- `MONOCHROME1` (DICOM's inverted-grayscale convention, 0 = white) is flipped to the `MONOCHROME2` convention (0 = black) that every downstream heuristic in `router_modality.py`/`preprocessing.py` assumes.
- Color DICOM (e.g. color Doppler) is detected (`SamplesPerPixel >= 3` or a `RGB`/`YBR` `PhotometricInterpretation`) and passed through as-is (no YBR colorspace conversion — rare enough in this pipeline's target modalities not to warrant it, and this only affects color fidelity, never crashes).

**Vote-MI-inspired representative-slice selection.** Rather than a single arbitrary slice (risks landing on an uninformative edge slice) or a fixed relative depth, every candidate slice (strided above `config.VOLUME_SCORING_MAX_SLICES` = 128 to bound worst-case latency on very deep series) is scored by two unsupervised signals computed on a shared, volume-wide percentile-stretched 0-255 scale:
- **Intensity variance** — a mostly-uniform air/background slice scores near zero.
- **Edge density** — mean Sobel gradient magnitude, a proxy for visible anatomical structure/boundaries.

Both signals are min-max normalized across the candidate pool and summed (`config.VOLUME_EDGE_DENSITY_WEIGHT`, default equal weight) into one composite score per slice. The top `config.VOLUME_SLICE_SELECTION_COUNT` (default 3) slices are picked greedily by that score, subject to a minimum index separation (`config.VOLUME_SLICE_MIN_SEPARATION_FRACTION` × depth, default 10%) so the selection can't collapse onto a cluster of near-duplicate adjacent slices — representative coverage of the volume, not just its single busiest region. Verified against real dev-data volumes: a 275-slice CT series correctly picked three genuinely distinct, anatomically rich cross-sections (neck, upper chest, lung) versus the fixed-percentage policy's three arbitrary, less-informative picks; a brain MRI's picks spanned skull base → orbits → cortex with visible gyri. This is a heuristic voting signal inspired by representative-slice-selection principles in multi-instance medical volume analysis, not a literal reproduction of any specific published algorithm — unvalidated against labeled data, same caveat as Section 1.4's subtype classifiers.

The selected slices are tiled horizontally into one composite via `_tile_slices()`, windowed against one shared percentile range so the tiles stay visually consistent with each other.

**Graded, zero-crash fallback chain**, entirely inside `volume_loader.py`, layered *underneath* the outer blind-gray-canvas fallback described in Section 5.2/5.5:
1. Content-based (Vote-MI) multi-slice selection + tiling (primary path).
2. If content scoring itself errors: fall back to the original fixed-percentage (35%/50%/65% depth) slice-index heuristic, still tiled via the same function.
3. If tiling/compositing fails at either of the above: fall back further to a single slice (the same policy's middle pick) with a plain percentile stretch, no tiling.
4. Only if that also fails does a `VolumeLoadError` propagate out of this module — the outer layer (`predict.py`) then substitutes a neutral gray canvas and still attempts a real (blind) model call, rather than a hardcoded fallback letter (Section 5.2).

**Multi-channel grid compositing (`_tile_color_channels_into_grid()`).** Channel-split microscopy/fluorescence folders are arranged into a fixed 2×2 grid — blue top-left, green top-right, red bottom-left, yellow bottom-right, with a thin border between quadrants — so the VLM inspects every available channel at full (capped) resolution, unmixed, in one forward pass, rather than a blended pseudo-color guess at which channel contributed what. Each channel is resized to `config.VOLUME_CHANNEL_GRID_CELL_SIZE` (448px square) before tiling — both to normalize any size mismatch across channels and to keep the finished grid comfortably under the processor's `MAX_PIXELS` cap on its own; an earlier full-native-resolution version of this grid was measured landing right at that cap, pushing single-query vision-token count and inference latency to right at/over `config.INFERENCE_TIMEOUT_SECONDS` even with no other system load. Missing channels (a sample with only 2-3 of the 4) get a flat placeholder quadrant rather than a reflowed grid, so "top-right is always green" stays a stable convention regardless of which channels a given sample ships.

**I/O parallelism.** Both the DICOM-series file reads and the (at most 4) channel-grid file reads happen concurrently via `ThreadPoolExecutor` (`config.VOLUME_DICOM_READ_WORKERS`, default 16 workers for the series case) rather than one file at a time — pydicom's read and PIL's decode both release the GIL for their I/O-bound portions, so this genuinely overlaps. Immaterial on local disk; meaningful on a network/FUSE-backed filesystem where each file open carries real round-trip latency (a 275-slice series read sequentially was measured at ~20s on a Colab Google Drive mount).

---

## 2. VQA Routing & Prompt-Routing Tracks

### 2.1 Query-Intent Classification

We classify each incoming question into one of several **clinical reasoning tracks**, each of which benefits from a different system prompt and reasoning style. This is done via a fast rule-based + embedding-similarity classifier (no LLM call needed — this must be near-instant).

| Track | Description | Example Trigger Phrases |
|---|---|---|
| **Diagnostic** | Asks for a disease/condition/finding identification | "what is the diagnosis", "which condition", "most likely disease" |
| **Spatial** | Asks about location, laterality, anatomical structure | "which lobe", "located in", "left or right", "which quadrant" |
| **Severity/Grading** | Asks about staging, grading, severity scoring | "grade", "stage", "severity", "BI-RADS", "how advanced" |
| **Modality-ID / Technical** | Asks about the imaging technique itself | "what imaging modality", "which sequence", "contrast used" |
| **Comparative/Differential** | Asks to rule in/out between options directly | "which of the following is NOT", "best explains", "most consistent with" |

**Classification implementation (matches `src/router_intent.py` exactly):**
1. Fast regex/keyword pass (covers ~70-80% of queries, <1ms). `router_intent.py` scores all 5 tracks by trigger-phrase match count and returns the highest-scoring track, or `None` if nothing matched.
2. Fallback: cosine similarity between a sentence-embedding of the query (using a small frozen encoder, `all-MiniLM-L6-v2`, ~22M params, runs on CPU in <5ms) and a set of precomputed track-centroid embeddings — invoked only when step 1 returns `None`.

### 2.2 System Prompt Templates per Track

Each track gets a **specialized clinical system prompt** injected before the VLM sees the image/question. These are designed to bias the model's internal reasoning toward the correct clinical framework without lengthening generation (since we constrain output to a single token anyway — the prompt shapes the *hidden reasoning*, not visible chain-of-thought, to keep latency low).

```text
[DIAGNOSTIC TRACK]
You are a board-certified diagnostic radiologist and pathologist with expertise
across all imaging modalities (radiography, CT, MRI, ultrasound, dermoscopy,
histopathology, fundoscopy). Examine the image carefully for the modality-
appropriate diagnostic features (opacity/density patterns for radiographs,
morphology/border irregularity for dermoscopy, cellular architecture for
histology). Select the single most likely diagnosis from the choices given.
Respond with ONLY the letter of the correct answer.

[SPATIAL TRACK]
You are an expert in medical imaging anatomy. Identify the anatomical
structure, location, or laterality referenced in the image, accounting for
standard radiological convention (patient left = image right for AP/PA
views unless stated otherwise). Respond with ONLY the letter of the
correct answer.

[SEVERITY TRACK]
You are a specialist in clinical grading and staging systems (e.g., BI-RADS,
Gleason, TNM, Fitzpatrick, ISUP). Assess the visual severity indicators
present in the image and match them to the most appropriate established
grading criteria among the choices. Respond with ONLY the letter of the
correct answer.

[MODALITY-ID TRACK]
You are an imaging physicist and radiologic technologist. Identify technical
imaging characteristics such as modality type, sequence weighting, contrast
phase, or acquisition parameters visible in the image. Respond with ONLY
the letter of the correct answer.

[DIFFERENTIAL TRACK]
You are a senior attending physician conducting differential diagnosis.
Systematically evaluate each choice against the visual evidence and
eliminate options that are inconsistent with the image. Respond with ONLY
the letter of the correct answer that best fits or is best excluded, as
asked.
```

**Design rationale:** Because we use constrained decoding (Section 4.2) to force a single-letter output, the model never actually emits free-form chain-of-thought text at inference. The system prompt's role is entirely to condition the model's internal hidden states toward the right "expert persona" and feature-attention pattern before the first (and only) token is generated. This is a well-documented technique for improving zero-shot MCQ accuracy in VLMs without adding generation-length latency.

These templates are reproduced verbatim as `TRACK_PROMPTS` in `src/prompt_builder.py`.

### 2.3 Modality × Track Prompt Composition

The final system prompt is a **composition** of the modality context (from Stage 1) and the track context (from Stage 3):

```python
def build_system_prompt(modality: str, track: str) -> str:
    modality_context = MODALITY_PROMPTS[modality]   # e.g. "This is a histopathology image..."
    track_context = TRACK_PROMPTS[track]              # e.g. "You are a diagnostic specialist..."
    return f"{track_context}\n\nImaging context: {modality_context}"
```

This matches `src/prompt_builder.py`'s `build_system_prompt()` exactly. `src/prompt_builder.py` additionally pre-composes all `len(MODALITIES) x len(TRACKS)` combinations into a `PROMPT_CACHE` dict at import time (`get_system_prompt(modality, track)` is the cached-lookup equivalent), so the string concatenation isn't repeated on every `predict()` call — see Section 4.4.

This keeps the prompt library small and combinatorial (4 modalities × 5 tracks = 20 combinations from just 9 template strings) rather than requiring 20 hand-written prompts.

**Note:** this composition is keyed on the **coarse stream only** — the 12-subtype taxonomy added in Section 1.4 is not yet reflected in prompt selection. An OCT B-scan and a true ultrasound image currently receive the identical "This is an ultrasound image..." modality-context sentence; see Section 1.4's known-limitation #3.

---

## 3. Model Selection & Fine-Tuning Strategy

### 3.1 Backbone Candidates

| Model | Params | Medical Pretraining | VRAM (fp16) | Notes |
|---|---|---|---|---|
| **Qwen3-VL-4B-Instruct** | 4B | General, strong OCR/grounding; enhanced MRope + DeepStack multi-level ViT features | ~8-9GB (unquantized) | **Recommended primary (current).** Deployed natively, no quantization; fits the competition's 10GB submission file size limit — see the migration notes below |
| **Qwen3-VL-8B-Instruct** | 8B | Same architecture, larger LLM decoder | ~16-17.5GB (unquantized) | Rejected as the deployed size: comfortably fits the 48GB VRAM budget, but the weights alone exceed the competition's 10GB submission file size limit — see the second migration note below |
| **LLaVA-Med-7B (v1.5)** | 7B | Biomedical (PMC articles) | ~15GB | Strong domain prior but weaker general visual grounding; older base architecture (LLaVA v1.5); also exceeds the 10GB submission limit |
| **Qwen3-VL-2B-Instruct** | 2B | General | ~4GB | Lower-latency fallback within the same generation, unquantized; worth A/B testing against the scoring formula's `k` if it's known or estimable |
| **CheXagent / BiomedCLIP+LLM hybrids** | Varies | Modality-specific (CXR only) | — | Rejected: single-modality specialists don't generalize to the required breadth |
| **MedGemma (Google, 4B)** | 4B | Broad biomedical multimodal | ~9GB | Strong candidate if available locally; good accuracy/latency trade-off; comparable size to the current primary |

**Recommendation: Qwen3-VL-4B-Instruct as the base**, fine-tuned with **LoRA** on medical VQA corpora. Rationale:
- Native support for arbitrary image resolutions/aspect ratios (critical since radiology, dermoscopy, and histology have wildly different native resolutions and aspect ratios) via its dynamic-resolution ViT encoder, now with DeepStack multi-level feature integration — a family-level feature shared across the 2B/4B/8B Qwen3-VL sizes, not lost by going smaller.
- Strong instruction-following baseline makes constrained MCQ decoding more reliable out-of-the-box.
- LoRA fine-tuning via PEFT still applies unchanged (Section 1.3).
- Fits comfortably inside the competition's 10GB submission file size limit (~8-9GB unquantized) — the deciding factor over the 8B variant, which otherwise would have been kept for its larger reasoning ceiling (see the second migration note below).
- **Qwen3-VL-2B-Instruct** remains a lower-latency fallback within the same family if the scoring formula's `k` ends up favoring it.

**Migration note (AWQ → native, Qwen2-VL → Qwen3-VL):** this project originally targeted AWQ-quantized Qwen2-VL-7B-Instruct (see Section 3.3's earlier rationale for why AWQ was chosen). It was migrated to a native, unquantized Qwen3-VL model after repeated operational friction from the AWQ dependency chain in practice: `autoawq` was archived/deprecated (2025-05-11), its successor `gptqmodel` requires a C++ toolchain to build from source on Windows, and `transformers`' AWQ quantizer eventually made `gptqmodel` a hard, unconditional runtime requirement with no config-level opt-out. Rather than keep patching around an increasingly fragile dependency chain, the backbone moved to the unquantized Qwen3-VL line entirely. See Section 3.3 for the full trade-off discussion, and the note directly below for why the deployed size within that line changed again shortly after.

**Migration note (8B → 4B, submission size limit):** the Qwen3-VL migration above was first deployed at the 8B size (~16-17.5GB unquantized) — VRAM-wise this fits the 48GB RTX 6000 Ada budget with room to spare. However, the competition imposes a strict 10GB submission file size limit, which the 8B weights alone exceed. Qwen3-VL-4B-Instruct (~8-9GB unquantized) fits comfortably inside that limit while staying in the same model family/generation (same architecture, MRope/DeepStack features, identical `Qwen3VLForConditionalGeneration`/`Qwen3VLProcessor` classes — Section 5, item 3 verification), so it was chosen over quantizing the 8B model back down. Re-introducing quantization purely to hit a size target would reopen exactly the autoawq/gptqmodel dependency problems the prior migration eliminated.

`config.BASE_MODEL_NAME` currently holds `"Qwen/Qwen3-VL-4B-Instruct"`; `config.MODEL_PATH` points at the local unquantized weights directory the running system actually loads from (Section 5.1).

### 3.2 Fine-Tuning Data Strategy

Use publicly available, license-compliant medical VQA datasets **offline, pre-competition** (this is fine-tuning "on your side," not "training on the evaluator's side," which is prohibited only at evaluation time):

- **OmniMedVQA**: ~120K QA pairs across 12 modalities (X-ray, CT, MRI, fundus, dermoscopy, histology, ultrasound, etc.) — closest match to this challenge's exact heterogeneity profile. Use this as the **primary fine-tuning set**.
- **PMC-VQA**: ~227K QA pairs derived from PubMed Central figures — good for diagnostic/differential reasoning diversity.
- **VQA-RAD / SLAKE**: Smaller, radiology-focused, useful for spatial-track reinforcement.
- **PathVQA**: Histopathology-specific QA, strengthens the microscopy stream.

**Fine-tuning blueprint (LoRA):**

```
Base model:        Qwen3-VL-4B-Instruct (frozen backbone, unquantized)
Adapter method:     LoRA (rank=16-32, alpha=32, dropout=0.05)
Target modules:     q_proj, k_proj, v_proj, o_proj (language model attention)
                     + optionally vision-tower cross-attn projections
Training format:    Reformat all QA pairs into strict MCQ format matching
                     eval-time format exactly: image + question + "A) ... B) ...
                     C) ... D) ..." + single-letter target
Loss:               Standard next-token CE loss, masked to only the answer-letter token
Epochs:             2-3 (medical VQA sets are noisy; more risks overfitting to
                     dataset-specific phrasing quirks)
Hardware:           Single RTX 6000 Ada is sufficient for LoRA fine-tuning of
                     a 4B VLM in fp16 + LoRA (no quantization needed on 48GB)
Adapter variants:   Optionally train 2-3 separate LoRA adapters:
                     (a) general-purpose (all modalities)
                     (b) microscopy-specialized (histology/cytology have the
                         most distinct visual statistics)
                     Hot-swap via PEFT set_adapter() based on router output
```

**As implemented**, `src/config.py` holds these exact hyperparameters as named constants (`LORA_RANK = 16`, `LORA_ALPHA = 32`, `LORA_DROPOUT = 0.05`, `LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]`, `LORA_EPOCHS = 3`), and **`src/train_lora.py` now consumes them** — a complete, working LoRA fine-tuning script, run offline/pre-competition, entirely separate from the runtime `predict()` path (Section 5), which only ever *loads* an already-trained adapter (Section 3.1, Section 1.3).

**As implemented (`src/train_lora.py`), differing from the blueprint above in a few deliberate ways:**

- **Data loading is shared with the evaluation tooling**, not a separate loader: `train_lora.py` calls `src.evaluate_omnimed.load_omnimed_samples()` / `_resolve_gt_letter()` directly, so the same local OmniMedVQA mirror (`--data-root`, Section 5.6) serves both fine-tuning and evaluation with byte-identical sample resolution. Only OmniMedVQA is wired up this way today — PMC-VQA/VQA-RAD/SLAKE/PathVQA remain unimplemented, offline-strategy-only, as in the original blueprint.
- **Training examples are built from the exact same Stage 0-4 pipeline `predict()` uses at inference** (`universal_normalize`, `route_modality`, `preprocess_image`, `detect_track`, `build_system_prompt`, then the same chat template + `"{query}\n\n{choices_str}\n\nAnswer with only the letter."` user-prompt format from Section 5.2) — not a hand-approximated variant that could drift from eval-time formatting. This directly satisfies the "match the fine-tuning answer format exactly to the eval format" requirement below by construction, rather than by manual discipline.
- **Loss is masked to only the answer-letter token**, exactly as specified below, computed by tokenizing the prompt alone and the prompt-plus-letter together (independently, not by splicing token tensors — Qwen3-VL's M-RoPE position-id computation needs an internally-derived image/text token-type map that only stays consistent when the processor builds the whole sequence itself; a discovered-and-fixed pitfall, see Section 5.6's `prefix_score.py` note) and masking every position before where the letter starts.
- **QLoRA (4-bit bitsandbytes) is the default, not full-precision LoRA on a 48GB card**, specifically because this script is written to also run on a free-tier Kaggle GPU (T4/P100, 16GB) — see Section 3.3. `--no-4bit` opts back into the original "single RTX 6000 Ada, fp16 + LoRA, no quantization needed" path.
- **Per-device batch size is fixed at 1**, with `--grad-accum-steps` (default 8) for a larger effective batch size, rather than a padded/collated multi-example batch — multi-image batching for a VLM needs `pixel_values`/`image_grid_thw` padded consistently across differently-sized images, which adds real complexity for no benefit at the data volumes LoRA fine-tuning actually uses.
- **Model loading is independent of `src/model_loader.py`**, deliberately: that module is inference-oriented (`eval()` mode, a CUDA warm-up `generate()` call, and — critically — it would auto-attach whatever adapter already exists at `config.LORA_PATH`, wrongly double-wrapping a fresh training run with a stale adapter's weights already active underneath the new `LoraConfig`). `train_lora.py` has its own from-scratch model load, sharing only `config.py`'s constants as the source of truth.
- **The adapter variants described in the blueprint (general-purpose + microscopy-specialized) remain unimplemented** — `train_lora.py` produces exactly one adapter per run, matching Section 1.3's "not yet exercised" note on adapter hot-swapping.

Verified end-to-end on real hardware (not just code-reviewed): a real training run against a synthetic local OmniMedVQA mirror completed successfully in 4-bit on a 6GB GPU; all 144 `lora_B` weight matrices were confirmed to have moved off their zero-initialization (proof gradients genuinely flowed, not just that the script exited 0); and the resulting adapter was loaded back through the real `src/model_loader.py` `PeftModel.from_pretrained()` path and used in a live `predict()` call successfully — full round-trip.

**Critical: match the fine-tuning answer format exactly to the eval format.** If training examples present choices as "A) ... B) ... C) ... D) ..." and expect a bare letter as the target, the eval-time prompt must be byte-identical in structure. Format mismatch between fine-tuning and inference is the single most common cause of degraded constrained-decoding accuracy. `src/predict.py`'s actual prompt construction (`f"{query}\n\n{choices_str}\n\nAnswer with only the letter."` with `choices_str` built as `"A) ...\nB) ...\n"`) is the eval-time format this must match.

### 3.3 Quantization & Trade-offs

**Current decision: no quantization by default, but opt-in bitsandbytes 4-bit/8-bit support now exists.** `src/model_loader.py` loads Qwen3-VL-4B-Instruct natively (fp16) unless `config.QUANTIZATION_MODE` (`None` | `"4bit"` | `"8bit"`, default `None`) is explicitly set — with the default, loading behavior is byte-identical to before this option existed. This isn't a reversal of the "no quantization" decision for the competition deployment (the 4B backbone still comfortably fits in fp16 on the target 48GB RTX 6000 Ada, Section 3.1); it exists for constrained-VRAM environments, most concretely `src/train_lora.py`'s default QLoRA fine-tuning path on a free-tier Kaggle GPU (Section 3.2), where fp16 gradients/activations/optimizer state genuinely don't fit even though inference-only fp16 weights would.

**Implementation notes, since this is a real, tested code path, not just a design intent:**
- `_build_quantization_config()` (`model_loader.py`) constructs a `transformers.BitsAndBytesConfig` — `load_in_4bit=True` with `bnb_4bit_compute_dtype`/`bnb_4bit_quant_type` (`"nf4"`)/`bnb_4bit_use_double_quant` from `config.py`, or plain `load_in_8bit=True`.
- It is built once, at module scope, **before** the flash-attn → sdpa retry `try/except` (Section 4.1) — not folded into it. `transformers.BitsAndBytesConfig` constructs fine even without the `bitsandbytes` package actually installed (the real dependency is only pulled in later, inside `from_pretrained()`'s quantizer dispatch); an explicit `import bitsandbytes` presence check surfaces a missing install as its own clear, immediate `ImportError`, rather than letting it surface deeper inside the model load and get misattributed to "flash-attn unavailable" by the attention-fallback retry logic.
- `device_map` stays a single fixed device string (`config.DEVICE`, e.g. `"cuda:0"`) under quantization too, not switched to `device_map="auto"` — `"auto"` enables accelerate's multi-GPU/CPU-offload sharding, not wanted for this single-GPU deployment and not required for bitsandbytes to work.
- Tested (with `bitsandbytes` installed, both mocked-construction and a real end-to-end 4-bit training run, Section 3.2): `_build_quantization_config()`'s three branches (`None`, `"4bit"`, `"8bit"`), its invalid-mode `ValueError`, and its missing-dependency `ImportError` all behave correctly and independently — including a real ordering bug caught during testing, where an invalid mode string was originally being masked by the bitsandbytes-presence check before mode validation ever ran.

This section originally recommended AWQ 4-bit quantization for Qwen2-VL-7B; that recommendation and the trade-off table behind it are kept below for the historical comparison, followed by why the project moved off it.

| Config | VRAM | Relative Latency | Accuracy Impact |
|---|---|---|---|
| bf16/fp16 full precision (**current**) | ~8-9GB (4B model) | 1.0x (baseline) | Best |
| 8-bit (bitsandbytes) | ~5GB | ~1.1-1.3x slower (dequant overhead) | Negligible loss (<1%) |
| 4-bit NF4 (QLoRA-style, bitsandbytes) | ~3GB | ~1.0-1.2x (kernel-dependent) | Small loss (~1-2%) |
| AWQ 4-bit (pre-quantized, purpose-built kernels) | ~3GB | Faster than bf16 in practice (optimized kernels) | Small loss (~1-2%), often better than bnb 4-bit |

**Why the original AWQ recommendation was dropped, in practice, not just in theory:** AWQ's fused kernels were genuinely attractive on paper (lower VRAM, competitive-or-better latency than bf16). But operationally, the dependency chain proved fragile: `autoawq` (the library providing those kernels) was archived and deprecated on 2025-05-11 and is now unmaintained; its designated successor, `gptqmodel`, requires a C++ toolchain to build from source on Windows (a real blocker encountered running this project); and `transformers`' AWQ quantizer eventually made `gptqmodel` a hard, unconditional runtime dependency for AWQ loading with no config-level way to opt out (confirmed by reading `transformers.quantizers.quantizer_awq` directly — `AwqConfig` forcibly coerces any legacy pure-`autoawq` backend selection back to the `gptqmodel`-backed path). Rather than continue patching around an increasingly unmaintained dependency, the backbone moved to the **unquantized** Qwen3-VL line — first at 8B, then resized to 4B specifically to fit the competition's 10GB submission file size limit (Section 3.1's second migration note), not for any VRAM or latency reason.

**Since the RTX 6000 Ada has 48GB VRAM, VRAM was never the binding constraint here — latency is** (unchanged from the original design philosophy, Section 0). An unquantized 8B model at ~16GB fp16 still leaves enormous VRAM headroom, so the quantization trade-off table above is no longer load-bearing for this deployment; it's kept for context on the historical decision. This should still tilt every *other* design decision (image resolution, LoRA adapter count, prompt length) toward minimizing wall-clock time per query rather than minimizing memory footprint.

---

## 4. Inference Optimization & Latency Control

### 4.1 Global Pre-Loading (Mandatory per Rules)

All heavy objects — model weights, processor/tokenizer, and the LoRA adapter (when present) — are instantiated **once at module import time**, outside `predict()`. (The modality router itself loads no model at all today — it's pure heuristics; see Section 1.2's note on the learned-router stub. The sentence-embedding encoder used for query-intent fallback is likewise loaded once at import time, in `src/router_intent.py`, not here.)

As implemented (`src/model_loader.py`), paths and hyperparameters are read from `src/config.py` rather than hardcoded, and the attention backend has a runtime fallback:

```python
# === src/model_loader.py -- GLOBAL, LOADED ONCE ===
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from src import config

processor = AutoProcessor.from_pretrained(
    config.MODEL_PATH, min_pixels=config.MIN_PIXELS, max_pixels=config.MAX_PIXELS,
)

def _build_quantization_config():
    """None unless config.QUANTIZATION_MODE is explicitly set (Section 3.3)
    -- built once, before the attn-implementation retry below, so a
    misconfigured/missing bitsandbytes install fails loudly and
    immediately rather than being conflated with a flash-attn failure."""
    if config.QUANTIZATION_MODE is None:
        return None
    import bitsandbytes  # presence check; raises its own clear ImportError
    from transformers import BitsAndBytesConfig
    if config.QUANTIZATION_MODE == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=config.QUANTIZATION_COMPUTE_DTYPE,
            bnb_4bit_quant_type=config.QUANTIZATION_4BIT_QUANT_TYPE,
            bnb_4bit_use_double_quant=config.QUANTIZATION_4BIT_USE_DOUBLE_QUANT,
        )
    return BitsAndBytesConfig(load_in_8bit=True)  # "8bit"

_quantization_config = _build_quantization_config()

def _load_model(attn_implementation):
    kwargs = dict(device_map=config.DEVICE, attn_implementation=attn_implementation)
    if _quantization_config is not None:
        kwargs["quantization_config"] = _quantization_config
        kwargs["torch_dtype"] = config.QUANTIZATION_COMPUTE_DTYPE
    else:
        kwargs["torch_dtype"] = config.TORCH_DTYPE
    return Qwen3VLForConditionalGeneration.from_pretrained(config.MODEL_PATH, **kwargs)

try:
    model = _load_model(config.ATTN_IMPLEMENTATION)           # "flash_attention_2"
except (ImportError, ValueError):
    model = _load_model(config.ATTN_IMPLEMENTATION_FALLBACK)  # "sdpa"

# LoRA is optional: only attempted if config.LORA_PATH exists and is
# non-empty, and degrades to the base model (logged) if loading fails --
# `model` is only reassigned on success, so a failed attach leaves the
# original base model bound and untouched (Section 1.3).
if Path(config.LORA_PATH).is_dir() and any(Path(config.LORA_PATH).iterdir()):
    try:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, config.LORA_PATH, adapter_name=config.LORA_ADAPTER_NAME)
        model.set_adapter(config.LORA_ADAPTER_NAME)
    except Exception:
        pass  # logged, falls back to the base model -- see src/model_loader.py
model.eval()

# Warm-up pass to trigger CUDA kernel compilation / cudnn autotune
with torch.inference_mode():
    _warmup_inputs = processor(text="warmup", images=None, return_tensors="pt").to(config.DEVICE)
    _ = model.generate(**_warmup_inputs, max_new_tokens=1)
    torch.cuda.synchronize()
```

**Model class history, briefly:** this loader has gone through `AutoModelForVision2Seq` → `Qwen2VLForConditionalGeneration` (some installed `transformers` builds didn't export the generic Auto class) → the current `Qwen3VLForConditionalGeneration`, following the Section 3.3 migration off AWQ-quantized Qwen2-VL entirely. No `quantization_config` is passed by default — see Section 3.3 for the opt-in bitsandbytes path this loader now also supports. `Qwen3VLForConditionalGeneration` is size-agnostic — the same class loads the 2B/4B/8B Instruct variants alike, so the later 8B → 4B resize (Section 3.1) changed only `config.MODEL_PATH`/`config.BASE_MODEL_NAME`, with zero code changes here.

**`model` may be a `Qwen3VLForConditionalGeneration` or a `peft.PeftModel` wrapping one**, depending on whether a LoRA adapter attached successfully (Section 1.3). Every downstream call site (`decode.py`, `prefix_score.py`, the warm-up pass above) only ever uses the common surface both object types provide identically (`model(**inputs)`, `.generate()`, `.device`, `.eval()`), so nothing downstream branches on which one it actually got.

**Current default is the `sdpa` fallback path, not Flash-Attention 2:** `requirements.txt` (Section 5.4) currently ships with `flash-attn` commented out, so unless it's installed separately, `_load_model(config.ATTN_IMPLEMENTATION)` raises and every deployment falls through to `attn_implementation="sdpa"`. This is a correct, functioning code path (see Section 4.4), just worth knowing it's the realistic default right now rather than a rare edge case.

The warm-up call is important: the first real inference call otherwise absorbs CUDA context initialization and kernel autotuning cost, which would badly skew the *first* recorded `inference_time` and potentially the average.

### 4.2 Forcing Output to {A, B, C, D} via Logit Masking

Rather than letting the model generate free text and post-hoc parsing it (slow, fragile, adds retry logic), we **directly constrain the vocabulary at the logits level** for a single-token generation:

```python
import torch

def get_choice_token_ids(processor):
    """Pre-compute token ids for 'A','B','C','D' (and common variants) once, globally."""
    ids = {}
    for letter in ["A", "B", "C", "D"]:
        for variant in [letter, f" {letter}", f"{letter})", f"({letter}"]:
            tok_ids = processor.tokenizer.encode(variant, add_special_tokens=False)
            if len(tok_ids) >= 1:
                ids.setdefault(letter, []).append(tok_ids[0])
    return ids

CHOICE_TOKEN_IDS = get_choice_token_ids(processor)  # global, computed once

@torch.inference_mode()
def constrained_predict_letter(inputs):
    outputs = model(**inputs)
    last_logits = outputs.logits[:, -1, :]  # logits for next token
    scores = {}
    for letter, tok_ids in CHOICE_TOKEN_IDS.items():
        scores[letter] = max(last_logits[0, t].item() for t in tok_ids)
    return max(scores, key=scores.get)
```

**As implemented** (`src/decode.py`), this matches almost verbatim: the per-letter variant list `[letter, f" {letter}", f"{letter})", f"({letter}"]` is sourced from `config.CHOICE_TOKEN_VARIANTS` rather than inlined, and `model`/`processor` are imported from `src.model_loader` rather than assumed as bare globals. Functionally identical, with one addition: the actual function is `constrained_predict_with_scores(inputs, valid_letters=None) -> (letter, scores)`, returning the full `{letter: logit}` dict alongside the winning letter — `constrained_predict_letter()` is now a thin wrapper (`letter, _ = constrained_predict_with_scores(...)`) kept for the competition path's call sites. The scores dict exists so callers that need more than the winning letter (`predict_with_diagnostics()`'s `top1_logit`/`top2_logit`/`logit_margin` diagnostic fields, Section 5.2) don't need a second forward pass. `valid_letters` (defaulting to all of `CHOICE_TOKEN_IDS`) restricts the argmax to exactly the letters a given query actually offers, so a 2-choice/Yes-No row can never be answered "C" or "D" just because those tokens happened to score higher on unrelated logits.

This approach:
- **Eliminates generation loops entirely** — one forward pass, not autoregressive decoding, since we only need the argmax over the candidate first-token logits (up to 4 surface-variant token ids per letter, then the best of those per letter, then the best letter overall).
- **Removes all parsing/retry logic** — no risk of the model producing "The answer is C." and needing regex extraction.
- **Is provably robust** — the answer is always exactly one of A/B/C/D by construction, satisfying the strict output contract even under adversarial or malformed model outputs.

This single change (single forward pass + logit masking, vs. `generate()` with `max_new_tokens=5-10` and string parsing) is typically the **single largest latency win** available, often cutting per-query inference time by 60-80% since it avoids the KV-cache growth and multiple forward passes of autoregressive generation.

### 4.3 Image Resolution Control

Qwen3-VL's dynamic resolution ViT (like its Qwen2-VL predecessor's, which its image processor is still built on — see Section 5.4's note on `Qwen2VLImageProcessor`) is powerful but can be latency-expensive on very large images (a 4000×3000 histology tile, uncapped, produces a large number of vision tokens, ballooning both compute and context length). We cap resolution explicitly:

```python
MAX_PIXELS = 1024 * 1024  # tune based on latency budget
MIN_PIXELS = 256 * 256

processor = AutoProcessor.from_pretrained(
    MODEL_PATH, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
)
```

This bounds the number of vision tokens fed into the LLM, directly bounding both prefill latency and VRAM use for attention — the single biggest lever for controlling latency variance across wildly differing native image resolutions (a chest X-ray vs. a whole-slide microscopy crop).

**As implemented, this cap is applied twice, independently:** once here, at the HF processor level (`src/model_loader.py`), and again in `src/preprocessing.py`'s `_cap_resolution()`, run as the very first step of `preprocess_image()` — before any modality-specific op (CLAHE, stain normalization, hair inpainting, etc.) touches the raw image. The second cap exists because those modality-specific ops are themselves O(pixels) and shouldn't run on an unbounded raw image just because the eventual HF-processor resize would shrink it anyway.

### 4.4 Additional Latency Levers

- **`torch.inference_mode()`** everywhere (no autograd graph tracking) — used in `decode.py`'s `constrained_predict_with_scores()` (Section 4.2) and the warm-up pass in `model_loader.py`.
- **Flash-Attention 2** backend for the vision and language attention blocks when available, with a runtime fallback to `sdpa` (`src/model_loader.py`'s `_load_model()` try/except) if the wheel isn't installed or fails to import. **As currently configured, `requirements.txt` ships `flash-attn` commented out, so `sdpa` is the practical default** (Section 4.1/5.4) — re-enable that line and reinstall once a compatible wheel/CUDA toolkit is confirmed available to get the Flash-Attention 2 speedup.
- **Batching (if evaluation allows batched calls):** Not applicable if `predict()` is called one query at a time per spec; not implemented.
- **Skip the CPU-based sentence-embedding fallback classifier when the regex router already matches** — implemented exactly as described: `router_intent.py`'s `detect_track()` only calls the embedding-based `_detect_track_embedding()` when the regex pass (`_detect_track_regex()`) returns no match.
- **Cache the composed system-prompt strings** so they aren't re-concatenated on every call: `prompt_builder.py`'s `PROMPT_CACHE` precomputes all `len(MODALITIES) x len(TRACKS)` combinations at import time. **Note:** this caches the composed *string*, not pre-tokenized token-ID tensors — actual tokenization (`processor.apply_chat_template` + `processor(...)`) still runs once per `predict()` call in `predict.py`. True pre-tokenization (skipping tokenizer overhead per call entirely) remains a possible future optimization, not yet implemented.
- **Pin router + embedding models to CPU**, VLM to GPU, to avoid contention and unnecessary PCIe transfer of small intermediate tensors. The sentence-embedding encoder is pinned via `config.INTENT_EMBEDDING_DEVICE = "cpu"` (`router_intent.py`); the modality router has no learned model to pin today (`config.MODALITY_ROUTER_DEVICE = "cpu"` exists for the not-yet-implemented learned-router path — see Section 1.2).

---

## 5. Production Code Skeleton & Submission Structure

### 5.1 Directory Layout

**This repository's actual layout differs from the originally-planned `submission/`-wrapped skeleton** — there is no `submission/` directory; everything lives at the repo root, there is no `README.md` yet, and `weights/` (model weights, LoRA adapter, and the optional `stain_reference_matrix.npy` / `modality-router` checkpoint) is a deployment-time directory referenced by `src/config.py`'s path constants — it is **not** checked into this repository:

```
<repo root>/
├── requirements.txt
├── eval.py                       # THE competition entrypoint (fixed invocation
│                                  #   contract: `python eval.py input_dir <path>`,
│                                  #   parses dev_metadata.csv -- Section 5.3)
├── run_predictions.py            # generic scaffold entrypoint: loops a harness-
│                                  #   provided data_loader.py, calls predict(), writes CSV
├── kaggle_finetune_lora.ipynb    # copy-paste-ready Kaggle notebook driving
│                                  #   src/train_lora.py end-to-end (Section 5.7)
├── medical_vqa_architecture.md
├── datasets.md                   # which dataset is used for evaluation vs.
│                                  #   fine-tuning, per PS-named modality (Section 5.7)
├── MedVQA_Pipeline_Technical_Audit.md  # external review + status/TODO tracking
├── src/
│   ├── __init__.py
│   ├── config.py                 # paths, constants, MIN/MAX_PIXELS, subtype taxonomy
│   │                              #   (Section 1.4), volumetric ingest / quantization /
│   │                              #   blind-fallback tunables, feature flags
│   ├── model_loader.py           # global model/processor/adapter loading (import-time),
│   │                              #   opt-in quantization (Section 3.3)
│   ├── volume_loader.py          # Stage -1: volumetric/DICOM ingest (Section 1.5)
│   ├── router_modality.py        # Stage 1: hierarchical modality routing (coarse
│   │                              #   4-bucket router, unchanged + Section 1.4 subtypes)
│   ├── router_intent.py          # Stage 3: query-intent / track classification
│   ├── preprocessing.py          # Stage 0 + Stage 2: universal normalization +
│   │                              #   subtype-specific image preprocessing
│   ├── prompt_builder.py         # Stage 4: system prompt composition
│   ├── decode.py                  # Stage 6: logit-masked constrained decoding
│   ├── prefix_score.py            # OmniMedVQA-paper-comparable scoring (Section 5.6)
│   ├── predict.py                 # predict() (competition path), plus
│   │                              #   predict_with_diagnostics() /
│   │                              #   predict_by_prefix_score() (Section 5.2)
│   ├── evaluate_omnimed.py        # local-mirror OmniMedVQA evaluation harness,
│   │                              #   metrics/bootstrap-CI/diagnostic sidecar (Section 5.6)
│   ├── train_lora.py              # LoRA fine-tuning script (Section 3.2)
│   └── prepare_external_dataset.py  # converts external volumetric/2D datasets
│                                  #   (MosMedData, Task01_BrainTumour, BUSI) into the
│                                  #   OmniMedVQA local-mirror shape (Section 5.7, datasets.md)
└── weights/                      # NOT present in this repo -- expected at deployment
    ├── qwen3-vl-4b-instruct/      # unquantized base weights (local, no internet)
    ├── lora-medvqa/               # fine-tuned LoRA adapter -- produced by
    │                              #   src/train_lora.py (Section 3.2), or trained
    │                              #   elsewhere; NOTE: must be trained against
    │                              #   Qwen3-VL-4B specifically -- an adapter trained
    │                              #   against Qwen2-VL-7B OR Qwen3-VL-8B is NOT
    │                              #   architecture-compatible (different hidden
    │                              #   sizes/attention config per model size)
    ├── modality-router/            # optional, only used if USE_LEARNED_ROUTER is enabled
    └── stain_reference_matrix.npy  # optional; falls back to the standard Macenko
                                     #   reference vectors if absent (src/config.py)
```

### 5.2 `predict()` Reference Implementation

**As implemented** (`src/predict.py`) — this reflects the actual current pipeline, including the Section 5.5 failure-mode safeguards inline (they are not a separate layer wrapped around a simpler core):

```python
# src/predict.py
import logging, threading, time
from pathlib import Path
from PIL import Image, UnidentifiedImageError

from src import config
from src.model_loader import model, processor         # import triggers global load (Section 4.1)
from src.volume_loader import load_volume, VolumeLoadError  # Stage -1 (Section 1.5)
from src.router_modality import route_modality          # Stage 1 (Section 1.2 / 1.4)
from src.router_intent import detect_track                # Stage 3 (Section 2.1)
from src.preprocessing import preprocess_image, universal_normalize  # Stage 0 + 2
from src.prompt_builder import build_system_prompt          # Stage 4 (Section 2.3)
from src.decode import constrained_predict_with_scores        # Stage 6 (Section 4.2)

logger = logging.getLogger(__name__)


def _run_with_timeout(fn, timeout_seconds):
    """Thread-based hard timeout (Section 5.5) -- not signal-based, since
    signal.alarm is unavailable on Windows / outside the main thread."""
    result = {}
    def _target():
        try:
            result["value"] = fn()
        except Exception:
            logger.exception("Inference worker raised an exception")
    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    return None if thread.is_alive() else result.get("value")


def _load_input_image(image) -> Image.Image:
    """Stage -1 + corrupt-image guard, shared by all three entry points below."""
    if isinstance(image, (str, Path)):
        volume_image = load_volume(Path(image))  # Section 1.5 -- None if not volumetric
        if volume_image is not None:
            image = volume_image
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image.load()
    return image


def predict(image, query: str, choices: dict) -> str:
    # --- Stage -1 + corrupt/unreadable image guard (Section 5.5) ---
    try:
        image = _load_input_image(image)
    except (VolumeLoadError, UnidentifiedImageError, OSError, ValueError):
        return config.FALLBACK_ANSWER_LETTER

    # --- Malformed-choices guard: {"A","B"} or {"A","B","C","D"} (Section 5.5) ---
    if not (isinstance(choices, dict) and set(choices) in ({"A", "B"}, set(config.CHOICE_LETTERS))):
        return config.FALLBACK_ANSWER_LETTER

    try:
        # Stage 0: universal normalization -- before modality routing
        image = universal_normalize(image)

        # Stage 1: hierarchical routing -- coarse stream + subtype (Section 1.4)
        modality, subtype = route_modality(image)

        # Stage 2: subtype-specific preprocessing
        image = preprocess_image(image, modality, subtype)

        # Stage 3: query-intent / track detection
        track = detect_track(query)

        # Stage 4: build final prompt (coarse stream + track -- Section 2.3)
        system_prompt = build_system_prompt(modality, track)
        choices_str = "\n".join(f"{k}) {v}" for k, v in choices.items())
        user_prompt = f"{query}\n\n{choices_str}\n\nAnswer with only the letter."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": user_prompt},
            ]},
        ]
        text_input = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[text_input], images=[image], return_tensors="pt").to(config.DEVICE)
    except Exception:
        return config.FALLBACK_ANSWER_LETTER

    # Stage 5+6: single forward pass + constrained decode, under a hard timeout.
    # valid_letters restricts the argmax to exactly this query's offered choices.
    result = _run_with_timeout(
        lambda: constrained_predict_with_scores(inputs, valid_letters=list(choices.keys())),
        config.INFERENCE_TIMEOUT_SECONDS,
    )
    return result[0] if result is not None else config.FALLBACK_ANSWER_LETTER
```

Differences from an earlier draft of this section, made explicit: `Image.open()` is now preceded by Stage -1's `load_volume()` dispatch (Section 1.5); the coarse-modality-only `detect_modality()` import is now `route_modality()` (returns `(modality, subtype)`); `preprocess_image()` takes both `modality` and `subtype`; the malformed-choices guard now accepts either the 4-choice key set or the 2-choice `{"A","B"}` set (`config.CHOICE_SET_OPTIONS`), not only 4; `.to("cuda")` is `.to(config.DEVICE)`; and the corrupt-image/malformed-choices/timeout guards (Section 5.5) are inline in the real function, not layered on separately as this section previously implied.

**Two additional entry points share `_load_input_image()` and the Stage 0-2 preprocessing above, but diverge from Stage 3 onward — neither is used by the competition path, both live in `src/predict.py` for the evaluation tooling in Section 5.6:**

- **`predict_with_diagnostics(image, query, choices) -> dict`** — used by `evaluate_omnimed.py`'s default `--scoring-method pipeline`. Runs the exact same Stage 0-6 as `predict()` above, with two differences: an image that can't be loaded/decoded at all (or a Stage 0-4 crash on one that did load) is retried **once** on a neutral gray canvas (`config.BLIND_FALLBACK_IMAGE_SIZE`/`BLIND_FALLBACK_GRAY_VALUE`) instead of short-circuiting straight to `config.FALLBACK_ANSWER_LETTER` — the model still gets to reason from the question text and choices alone, rather than the answer being decided before it's ever called — and it returns a rich diagnostics dict (`answer`, `modality`, `subtype`, `track`, `fallback_triggered`, `fallback_reason`, `top1_logit`/`top2_logit`/`logit_margin` from `constrained_predict_with_scores`' returned scores dict, `inference_time`) instead of just the winning letter. Stage 5-6's timeout is deliberately *not* retried on the gray canvas — a stalled forward pass is a latency problem, not a content problem, and doubling GPU work per query would only worsen timeout-budget pressure.
- **`predict_by_prefix_score(image, query, choices) -> dict`** — used by `evaluate_omnimed.py`'s `--scoring-method prefix_score`. Same Stage -1/0/1/2 handling and blind-fallback behavior as above, but Stages 3-6 are replaced entirely with a replication of OmniMedVQA's own published "Prefix-based Score" methodology (`src/prefix_score.py`, Section 5.6) — no track/system-prompt, one forward pass per candidate option instead of one total.

### 5.3 Harness / CSV Writer

**Two harness entrypoints exist at the repository root, not one — `eval.py` and `run_predictions.py` — and `eval.py`, not `run_predictions.py`, is the actual competition entrypoint.** `run_predictions.py` (documented below) is a generic scaffold that depends on an undefined `data_loader.py` ("user-provided or harness-provided"); `eval.py` implements the competition's actual fixed invocation contract directly and needs nothing else supplied.

**`eval.py` — the competition entrypoint (`python eval.py input_dir <path_to_queries>`):**

- Reads `dev_metadata.csv` from the given directory (columns: `query_id, image, question, choice_A, choice_B, choice_C, choice_D`, with `choice_C`/`choice_D` empty for 2-choice/Yes-No rows) and resolves each row's `image` cell to a local path under `<path>/Images/` (or a few other plausible locations — the exact prefix convention isn't specified by the harness).
- **`_resolve_image_path()` tries an exact-name match first, then falls back to a regex scan for browser-downloaded-duplicate filenames** (`"0002.png"` referenced in the CSV, `"0002 (1).png"` actually on disk) — confirmed against a real competition dev-data sample, where this pattern affected roughly 70% of rows before the fallback was added; without it, those rows all silently degraded to the blind fallback letter with no image ever loaded.
- **Reads the CSV as `cp1252`, not `latin1`.** The dev metadata file (Excel/Windows-authored) contains Windows-1252 curly-quote bytes (e.g. `0x92` = right single quotation mark) that `latin1` decodes without error but into the *wrong* character (a stray C1 control code, not the intended apostrophe) — corrupting answer-choice text that flows straight into the model prompt. `cp1252` decodes the same file correctly with zero errors.
- Calls `predict()` (not `predict_with_diagnostics()`/`predict_by_prefix_score()` — this is the strict competition path, Section 5.2) once per row, under the same zero-crash contract as `predict()` itself: an unresolvable image path skips straight to `config.FALLBACK_ANSWER_LETTER` without calling `predict()` at all, and any exception `predict()` itself doesn't already catch is caught here too, as a final backstop — every row in `dev_metadata.csv` gets exactly one output row, never zero.
- Writes `predictions.csv` (`query_id, answer, inference_time`) to the current working directory.

**`run_predictions.py` — generic scaffold** (documented as originally written; unchanged this pass):

```python
# run_predictions.py
import csv, logging, time
from src.predict import predict
from data_loader import load_eval_dataset   # user-provided or harness-provided

logging.basicConfig(level=logging.WARNING)
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
```

`load_eval_dataset` remains intentionally unimplemented in this repository — it is user-provided or harness-provided; `run_predictions.py` only depends on it yielding `(query_id, image, query, choices)` tuples.

### 5.4 `requirements.txt`

**Current actual content** (`flash-attn` and `bitsandbytes` are both commented out — see the notes below and Section 4.4):

```
torch>=2.5.0
torchvision>=0.20.0
transformers>=4.57.0
accelerate>=0.33.0
peft>=0.12.0
pillow>=10.0.0
numpy>=1.26.0
opencv-python-headless>=4.10.0
sentence-transformers>=3.0.0
scikit-image>=0.24.0
datasets>=2.19.0
huggingface_hub>=0.23.0
nibabel>=5.2.0
pydicom>=2.4.0
# flash-attn>=2.6.0
# bitsandbytes>=0.43.0
```

**No `autoawq` or `gptqmodel`, deliberately:** the backbone migrated off AWQ-quantized Qwen2-VL-7B specifically to eliminate this dependency chain (Section 3.3) — `autoawq` is archived/deprecated (2025-05-11, unmaintained since), and its successor `gptqmodel` requires a C++ toolchain to build from source on Windows, which `transformers`' AWQ quantizer eventually made a hard, unconditional runtime requirement with no config-level opt-out. Neither package is needed at all for native, unquantized Qwen3-VL loading.

`transformers>=4.57.0` (bumped from `4.45.0`): 4.57.0 is the minimum release with Qwen3-VL support (`Qwen3VLForConditionalGeneration`, `Qwen3VLProcessor`).

`torch>=2.5.0` / `torchvision>=0.20.0` (bumped from `2.3.0`): this floor predates the Qwen3-VL migration — it was originally driven by `Qwen2VLVideoProcessor` requiring PyTorch >= 2.5 (disabling itself with an `ImportError` otherwise). `Qwen3VLProcessor` wraps `Qwen2VLImageProcessor` for images and `Qwen3VLVideoProcessor` for video (per `transformers`' own Qwen3-VL docs), so the same torch-version sensitivity carries forward; the floor stays at `2.5.0` regardless.

`nibabel`/`pydicom` (new, required, not optional): Stage -1's volumetric/DICOM ingest (Section 1.5) depends on both directly — unlike `flash-attn`/`bitsandbytes` below, there's no runtime fallback if these are missing; `src/volume_loader.py` raises `ImportError` at import time with a message pointing back here.

`bitsandbytes` (new, **commented out**, opt-in): only needed if `config.QUANTIZATION_MODE` is set (Section 3.3) or `src/train_lora.py`'s default 4-bit QLoRA path is used (Section 3.2) — CUDA-only, uncommented/installed on demand rather than unconditionally, the same treatment as `flash-attn` below.

`datasets`/`huggingface_hub`: **no longer used by `src/evaluate_omnimed.py`**, despite an earlier version of this document's claim — that module was refactored this pass to read a local OmniMedVQA mirror directly (`json.load()`, no `datasets` library, no Hub access at all; Section 5.6). `huggingface_hub` remains listed because `snapshot_download()` is still the documented way to fetch the *base model weights themselves* (Section 5.1's `weights/qwen3-vl-4b-instruct/`) into a fresh environment (e.g. a Colab/Kaggle notebook, Sections 5.6/5.7); `datasets` is no longer required by anything in this repository and is a candidate for removal, not yet done this pass.

*(Pin exact versions post-validation against the offline environment. `flash-attn` is commented out because it requires a matching CUDA toolkit and is often not buildable without sudo in the offline eval environment; `src/model_loader.py`'s `_load_model()` already falls back to the `sdpa` attention backend automatically when it's unavailable (Section 4.1/4.4), so leaving it commented out is a safe default, not a broken one. `bitsandbytes` is commented out for the same reason plus the fact that it's genuinely optional — `config.QUANTIZATION_MODE` defaults to `None` and the competition deployment doesn't need it (Section 3.3). Uncomment either and reinstall once actually needed/confirmed available in the target environment.)*

### 5.5 Failure-Mode Safeguards

**As implemented** (`src/predict.py`, `src/volume_loader.py`):

- **Timeout guard:** `_run_with_timeout()` wraps the Stage 5+6 constrained-decode call (`constrained_predict_with_scores()`, which itself calls `model(**inputs)`) on a **daemon worker thread**, with a `config.INFERENCE_TIMEOUT_SECONDS` (default `5.0`) join timeout — thread-based rather than signal-based, since `signal.alarm` is unavailable on Windows and unsafe outside the main thread. This can't forcibly cancel a stuck CUDA call — the worker thread keeps running in the background — but it does prevent one stalled query from blocking the harness's average-inference-time measurement. On timeout or any exception, `predict()` returns `config.FALLBACK_ANSWER_LETTER` (currently `"A"`, a fixed constant, not a distribution-derived value). **`predict_by_prefix_score()` (Section 5.2/5.6) deliberately has no equivalent guard** — it runs 2-4 untimed forward passes per query (one per candidate option), so a single slow/stuck call there is not bounded the way the competition path is; this is a known, accepted trade-off of that evaluation-only tool, not a gap in the competition path itself.
- **Volumetric/DICOM ingest guard (Stage -1, new — Section 1.5):** `load_volume()` raises `VolumeLoadError` for anything confidently identified as volumetric/DICOM but undecodable (corrupt file, empty series folder, unsupported internal encoding); `_load_input_image()` catches it in the same guard as the corrupt-image case below, degrading to the fallback letter identically. Internally, `volume_loader.py` layers its own graded fallback chain *before* ever reaching that point (Section 1.5) — a `VolumeLoadError` is the last resort, not the first response to any hiccup.
- **Corrupt/unreadable image guard:** `try/except` around Stage -1 + `PIL.Image.open()` / `.load()`, catching `VolumeLoadError`, `UnidentifiedImageError`, `OSError`, and `ValueError`; on failure, returns `config.FALLBACK_ANSWER_LETTER` immediately, before any other stage runs.
- **Empty/malformed choices guard:** validates that `choices` is a `dict` whose key set is **exactly** one of `config.CHOICE_SET_OPTIONS` — `{"A", "B"}` (2-choice/Yes-No) or `{"A", "B", "C", "D"}` — not merely "some keys" of any kind; degrades to the fallback letter otherwise.
- **Deterministic decoding:** no `generate()`/sampling call exists in the hot path at all — `constrained_predict_with_scores()` runs a single forward pass and an `argmax` over precomputed candidate logits (Section 4.2), so there is no `do_sample` flag to get wrong; behavior is deterministic and reproducible by construction.
- **Blind-fallback-not-hardcoded-letter (evaluation tooling only, Section 5.2/5.6):** `predict_with_diagnostics()`/`predict_by_prefix_score()` retry an unreadable image once on a neutral gray canvas rather than immediately returning the fallback letter — the competition path (`predict()`) does not do this and keeps its original immediate-fallback behavior unchanged, since this is a deliberate scope boundary, not an oversight (see Section 5.2).

### 5.6 Evaluation Tooling: SOTA Comparison Against OmniMedVQA

Beyond the competition harness (Section 5.3), two modules exist purely to measure this pipeline's accuracy against OmniMedVQA — the same benchmark the fine-tuning strategy (Section 3.2) draws from — in a way genuinely comparable to published leaderboard numbers, not just for internal tracking. Neither is imported by `eval.py`/`predict()`; both are `src/evaluate_omnimed.py`/`src/prefix_score.py`, run standalone.

**`src/evaluate_omnimed.py` — local-mirror OmniMedVQA harness.** Reads directly from a local directory mirror (`--data-root`, default `/content/drive/MyDrive/OmniMedVQA` — written for a Google Drive mount in Colab, but any local path works) laid out as `{data_root}/QA_information/Open-access/{dataset_name}.json` + `{data_root}/Images/{dataset_name}/{image_file_name}`, with **no Hugging Face Hub access at all** — a deliberate change from an earlier version of this tooling that downloaded via `huggingface_hub.snapshot_download()` and parsed JSON via the `datasets` library; both were replaced with plain `json.load()` (tolerating both a JSON array and JSON Lines, since real dataset dumps use either) and a `_require_local_data_root()` existence check with no fallback download. Only `QA_information/Open-access/` is ever read (Restricted-access items reference images that typically aren't distributed, so they can't be scored here anyway). `_resolve_image_path()` tries `{data_root}/Images/{dataset_name}/{basename of image_path}` first (the authoritative convention for this local layout) with two looser fallbacks for dump variations; `_resolve_gt_letter()` handles `gt_answer` being either a bare letter or the answer's full text (not pinned down by the dataset's own documentation).

Every sample with resolvable ground truth gets scored, whether or not its image resolves locally — an unresolvable/corrupt/volumetric image is **not** skipped, it's handled by the chosen scorer's blind-fallback path (Section 5.2) and still produces an honest (almost certainly wrong) data point rather than a silently dropped row. `--scoring-method {pipeline, prefix_score}` selects which of `predict.py`'s two diagnostics-returning entry points runs the actual scoring (default `pipeline`, i.e. `predict_with_diagnostics()`).

Two output files, always both written: `predictions.csv` (`query_id, answer, inference_time` — the exact 3-column format `eval.py` writes, directly comparable to a real submission) and `eval_diagnostic_log.csv` (`query_id, answer, gold, correct, inference_time, predicted_modality, predicted_intent, n_choices, fallback_triggered, fallback_reason, top1_logit, top2_logit, logit_margin`). Console output reports overall exact-match accuracy with a 95% bootstrap confidence interval (10,000 resamples, `numpy.random.default_rng`), accuracy stratified by ground-truth `modality_type` and by `n_choices`, and the overall fallback rate.

**`src/prefix_score.py` — OmniMedVQA's own published "Prefix-based Score" metric, replicated.** OmniMedVQA's paper reports two metrics per model, neither of which is this pipeline's Stage 6 letter-argmax: "Question-answering Score" (the model generates free text, embedded and matched by similarity to the nearest candidate option — **not implemented here**) and "Prefix-based Score" (the log-likelihood of each full candidate option's *text*, not just its letter, as a continuation of a plain completion prompt — **this is what `prefix_score.py` replicates**), read directly from the paper's own reference implementation (`OpenGVLab/Multi-Modality-Arena`, `MedicalEval/Prefix_based_Score/`):

1. A plain completion prompt with **no options listed at all**: `"Question: {question} The answer is"`.
2. For each candidate option's full text, tokenize `" {candidate_text}."` appended after that prompt, run one forward pass, and compute the mean cross-entropy loss over *only* the candidate+period tokens (everything before is masked to `-100`, PyTorch's default `ignore_index`).
3. The candidate with the **lowest** mean loss is the prediction.

Adapted for Qwen3-VL (the reference code drives older VLMs like LLaVA/BLIP-2/MiniGPT-4 through a raw string-completion API with an inline `<image>` token, which Qwen3-VL — an instruction-tuned chat model — has no equivalent of): the same prompt text becomes the sole user turn's content via the normal chat template (`add_generation_prompt=True`), and the candidate text is scored as if it were the assistant's completion. Each candidate reprocesses the full (prompt+candidate) text through the processor from scratch, rather than manually splicing pre-tokenized token ids onto a cached prefix — a deliberate choice after discovering Qwen3-VL's M-RoPE position-id computation (`get_rope_index`) needs an internally-derived image/text token-type map that only stays consistent with `input_ids` when the processor builds the whole sequence itself; splicing left that map sized for a shorter, stale sequence and crashed with a shape-mismatched boolean index. `use_cache=False` is also required, not just an optimization — Qwen3-VL caches M-RoPE deltas on the model instance across forward calls for incremental generation, and each candidate here is an independent, differently-shaped sequence, not a continuation of the previous one.

Sanity-checked (not just unit-tested for crashes) against the live model: given one fluent-English candidate and one gibberish candidate for the same image, the fluent one scored a substantially lower loss (large margin), confirming the mechanism measures something real. `predict_by_prefix_score()` (Section 5.2) wraps this with the same Stage -1/0/1/2 handling and blind-fallback behavior as `predict_with_diagnostics()`, costing roughly `n_choices`× the forward passes per query (2-4, one per candidate, versus 1) in exchange for a genuinely paper-comparable number.

### 5.7 Fine-Tuning Tooling

`src/train_lora.py` is documented in full in Section 3.2 (fine-tuning data strategy, implementation notes, and the real end-to-end validation performed). It is listed here only for the directory-layout cross-reference (Section 5.1) and because, like Section 5.6's evaluation tooling, it is a standalone script never imported by the competition path.

**`src/prepare_external_dataset.py`** — converts an external (non-OmniMedVQA) dataset into the exact local-mirror shape `load_omnimed_samples()`/`evaluate_omnimed.py`/`train_lora.py` already read, so none of that code needed to change to support new data sources. Written specifically because OmniMedVQA's own CT/MRI entries are pre-sliced 2D images cut from 3D volumes (confirmed in the dataset's own README), not raw NIfTI/DICOM — meaning OmniMedVQA alone can never exercise or validate Stage -1's volumetric ingest code (Section 1.5), regardless of which CT/MRI sub-dataset is selected. Three adapters ship: `mosmed` (MosMedData chest CT, real NIfTI, 5-class severity → 4-choice MCQ), `brainmri` (Medical Segmentation Decathlon Task01_BrainTumour, real 4D NIfTI → anatomy-ID MCQ, relying on `_load_nifti()`'s existing 4D→first-channel reduction rather than needing new sequence-splitting code), and `busi` (breast ultrasound, PNG, benign/malignant/normal → 3-choice MCQ). See `datasets.md` for the full per-dataset table (what's used for evaluation vs. fine-tuning, format, size, question design) — this document intentionally does not duplicate that table.

Two design choices worth calling out:
- **Image files are referenced by absolute path, not copied** into the mirror — `_resolve_image_path()`'s `root / image_path` candidate resolves correctly for an already-absolute `image_path` (pathlib drops `root` when joining an absolute path), so multi-GB volumes never need to be moved.
- **Splitting is case-level, not image-level**, via a deterministic seeded shuffle (`--eval-fraction`/`--seed`) — every QA item derived from the same case lands on the same side of the train/eval split, so a model can never see a case at train time and its held-out twin at eval time. Verified this session: disjoint `question_id` sets between the `_train`/`_eval` outputs of the same source dataset, plus a full round-trip through the real `predict_with_diagnostics()` on converted items with zero fallback triggered.

**`kaggle_finetune_lora.ipynb`** (repo root) — the actual runnable notebook for `src/train_lora.py`, written for Kaggle's free-tier GPU (T4/P100) rather than assuming Colab's Drive-mount convenience: gets the OmniMedVQA data via a Kaggle Dataset attachment or `gdown` from a shared Drive link (Kaggle does not mount Drive natively), gets the base weights via `huggingface_hub.snapshot_download()`, runs the external-dataset conversion (previous paragraph) if the raw archives are available, runs the fine-tuning itself, and includes a separate, independently-runnable section for producing a pre-quantized, smaller base-model checkpoint for the competition submission (Section 3.3) — with an explicit on-disk size check after saving, since `save_pretrained()` on a bitsandbytes-quantized model is documented as supported by Hugging Face but was flagged by at least one other source as sometimes writing the original (non-compact) weights back out instead; this could not be fully verified end-to-end in this development environment (a local disk/memory constraint interrupted the save step, independent of the quantization mechanism itself, which loading-side diagnostics did confirm works correctly) and needs to be confirmed on Kaggle's healthier environment before being relied on for the actual submission.

---

## 6. Summary of Key Design Decisions

| Decision | Rationale |
|---|---|
| Single shared VLM backbone (Qwen3-VL-4B) + one general-purpose LoRA adapter, not model ensemble | Avoids VRAM/latency cost of multiple large models resident or swapped per query; 4B size chosen to fit the competition's 10GB submission file size limit |
| Heuristic-only modality router (learned router designed, not implemented) | Near-zero latency cost; the coarse 4-bucket router is unchanged since first implementation |
| Two-level hierarchical routing: coarse stream (unchanged) + stream-conditional subtype classifier | Extends coverage toward 12 OmniMedVQA-style modalities (Section 1.4) without touching the validated coarse router; an unrecognized subtype degrades to a safe per-stream default rather than failing |
| Letterbox resize (not center-crop) for vignetted macroscopic subtypes | Avoids discarding genuine peripheral pathology (e.g. peripheral retinal findings) that a fixed-margin crop would assume is unimportant |
| Logit-masked single-forward-pass decoding | Eliminates autoregressive generation loop entirely — largest single latency win |
| Native, unquantized backbone (Section 3.3 migration off AWQ), with an automatic `sdpa` attention fallback | Drops the autoawq/gptqmodel dependency chain entirely (deprecated/unmaintained, Windows C++ build friction) while still fitting the 48GB VRAM budget comfortably; the `sdpa` fallback keeps the system working even when `flash-attn` isn't installed (the current shipped default) |
| Prompt-based track/modality conditioning instead of long chain-of-thought | Shapes hidden reasoning without adding generation-length latency |
| Global pre-loading + warm-up pass | Satisfies hard rule; prevents CUDA cold-start from skewing first-query latency |
| Fine-tune on OmniMedVQA + 3 external volumetric/2D datasets (offline, `src/train_lora.py` + `src/prepare_external_dataset.py`, Section 3.2/5.7, full breakdown in `datasets.md`) | Compliant with "no training at evaluation time"; matches heterogeneity profile of the challenge and covers all 8 PS-named modalities on both the evaluation and fine-tuning sides, including genuine volumetric CT/MRI (OmniMedVQA alone cannot, since its CT/MRI are pre-sliced 2D); PMC-VQA/PathVQA/VQA-RAD/SLAKE remain offline-strategy-only, not wired into the training script |
| Stage -1 volumetric/DICOM ingest, ahead of Stage 0 (Section 1.5) | Closes a real blind spot: `.nii`/DICOM inputs previously always fell back to a blind guess; a graded internal fallback chain (content-based → fixed-percentage → single-slice) degrades gracefully rather than jumping straight to "give up" |
| Vote-MI-inspired content-based slice selection over fixed relative depths | A blind depth-fraction pick can land on an uninformative/background slice; scoring by variance + edge density picks genuinely diagnostic cross-sections, verified visually against real CT/MRI data |
| Multi-channel (RGBY) folders composited as a 2×2 grid, not blended into one pseudo-color image | Preserves each fluorescence channel at full, unmixed resolution for the VLM to inspect directly, rather than an ambiguous blend that discards which channel contributed what |
| Blind gray-canvas fallback (evaluation tooling only) instead of an immediate hardcoded letter | An unreadable image still gets a real (if blind) model call reasoning from question text and choices alone, rather than the answer being decided before the model is ever invoked — deliberately scoped to `predict_with_diagnostics()`/`predict_by_prefix_score()`, not the competition path, which keeps its original immediate-fallback behavior |
| Opt-in bitsandbytes 4-bit/8-bit quantization (`config.QUANTIZATION_MODE`, default off) | The competition deployment doesn't need it (4B fits fp16 on 48GB comfortably), but `src/train_lora.py`'s default QLoRA path needs it to fit a free-tier Kaggle GPU (16GB) — zero behavior change for the existing inference path when left at the default |
| LoRA attached via `peft.PeftModel.from_pretrained()`, not `model.load_adapter()` | `model` is only reassigned on success, so a failed/incompatible adapter leaves the original base model bound untouched, no partial-wrap state possible; `PeftModel` preserves the same call surface every downstream site already relies on |
| Separate `evaluate_omnimed.py`/`prefix_score.py` scoring tooling, reading a local dataset mirror with no Hub access | Reports a number genuinely comparable to OmniMedVQA's own published leaderboard (Prefix-Score), not just this pipeline's own fast decode mechanism; local-only loading matches how the dataset is actually supplied in practice (a mounted Drive folder) rather than assuming live Hub access |
