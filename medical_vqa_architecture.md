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
                         │   Raw Input: Image + Query   │
                         │   + Choices [A, B, C, D]     │
                         └──────────────┬───────────────┘
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
                     │  Qwen3-VL-4B-Instruct (native, no        │
                     │    quantization; fits 10GB submission     │
                     │    size limit)                             │
                     │  + one general-purpose LoRA adapter      │
                     │    ("medvqa")                             │
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

**As implemented today, exactly one general-purpose adapter is loaded and active, and it's optional:** `config.LORA_ADAPTER_NAME = "medvqa"`, loaded in `src/model_loader.py` via `model.load_adapter(...)` + `model.set_adapter(...)` **only if `config.LORA_PATH` exists and is non-empty**. Since no fine-tuning script exists in this repository (Section 3.2), that directory won't exist for most setups; rather than hard-fail the whole import, the loader logs a warning and continues on the base model unmodified — and the same fallback applies if a directory is present but the adapter fails to load (e.g. one trained against a different model size, Section 3.1's migration notes). The multi-adapter hot-swap capability described above is architecturally supported (PEFT's `set_adapter()` is the mechanism, and it's already in the code) but not yet exercised — no second (e.g. microscopy-specialized) adapter ships or is switched at runtime.

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

**As implemented**, `src/config.py` holds these exact hyperparameters as named constants (`LORA_RANK = 16`, `LORA_ALPHA = 32`, `LORA_DROPOUT = 0.05`, `LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]`, `LORA_EPOCHS = 3`), ready for a fine-tuning script to consume — but **no fine-tuning/training script exists in this repository**. Section 3 remains an offline, pre-competition strategy; the runtime `predict()` path (Section 5) only ever *loads* an already-trained adapter (Section 3.1's single "medvqa" adapter, Section 1.3), it never trains one.

**Critical: match the fine-tuning answer format exactly to the eval format.** If training examples present choices as "A) ... B) ... C) ... D) ..." and expect a bare letter as the target, the eval-time prompt must be byte-identical in structure. Format mismatch between fine-tuning and inference is the single most common cause of degraded constrained-decoding accuracy. `src/predict.py`'s actual prompt construction (`f"{query}\n\n{choices_str}\n\nAnswer with only the letter."` with `choices_str` built as `"A) ...\nB) ...\n"`) is the eval-time format this must match.

### 3.3 Quantization & Trade-offs

**Current decision: no quantization.** `src/model_loader.py` loads Qwen3-VL-4B-Instruct natively (fp16), with no `quantization_config` at all. This section originally recommended AWQ 4-bit quantization for Qwen2-VL-7B; that recommendation and the trade-off table behind it are kept below for the historical comparison, followed by why the project moved off it.

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

def _load_model(attn_implementation):
    return Qwen3VLForConditionalGeneration.from_pretrained(
        config.MODEL_PATH, torch_dtype=config.TORCH_DTYPE, device_map=config.DEVICE,
        attn_implementation=attn_implementation,
    )

try:
    model = _load_model(config.ATTN_IMPLEMENTATION)           # "flash_attention_2"
except (ImportError, ValueError):
    model = _load_model(config.ATTN_IMPLEMENTATION_FALLBACK)  # "sdpa"

# LoRA is optional: only attempted if config.LORA_PATH exists and is
# non-empty, and degrades to the base model (logged) if loading fails.
if Path(config.LORA_PATH).is_dir() and any(Path(config.LORA_PATH).iterdir()):
    try:
        model.load_adapter(config.LORA_PATH, adapter_name=config.LORA_ADAPTER_NAME)
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

**Model class history, briefly:** this loader has gone through `AutoModelForVision2Seq` → `Qwen2VLForConditionalGeneration` (some installed `transformers` builds didn't export the generic Auto class) → the current `Qwen3VLForConditionalGeneration`, following the Section 3.3 migration off AWQ-quantized Qwen2-VL entirely. No `quantization_config` is passed at all now — see Section 3.3. `Qwen3VLForConditionalGeneration` is size-agnostic — the same class loads the 2B/4B/8B Instruct variants alike, so the later 8B → 4B resize (Section 3.1) changed only `config.MODEL_PATH`/`config.BASE_MODEL_NAME`, with zero code changes here.

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

**As implemented** (`src/decode.py`), this matches almost verbatim: the per-letter variant list `[letter, f" {letter}", f"{letter})", f"({letter}"]` is sourced from `config.CHOICE_TOKEN_VARIANTS` rather than inlined, and `model`/`processor` are imported from `src.model_loader` rather than assumed as bare globals. Functionally identical.

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

- **`torch.inference_mode()`** everywhere (no autograd graph tracking) — used in `decode.py`'s `constrained_predict_letter()` and the warm-up pass in `model_loader.py`.
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
├── run_predictions.py           # entrypoint: loops dataset, calls predict(), writes CSV
├── medical_vqa_architecture.md
├── src/
│   ├── __init__.py
│   ├── config.py                 # paths, constants, MIN/MAX_PIXELS, subtype taxonomy
│   │                              #   (Section 1.4), feature flags, k-aware thresholds
│   ├── model_loader.py           # global model/processor/adapter loading (import-time)
│   ├── router_modality.py        # Stage 1: hierarchical modality routing (coarse
│   │                              #   4-bucket router, unchanged + Section 1.4 subtypes)
│   ├── router_intent.py          # Stage 3: query-intent / track classification
│   ├── preprocessing.py          # Stage 0 + Stage 2: universal normalization +
│   │                              #   subtype-specific image preprocessing
│   ├── prompt_builder.py         # Stage 4: system prompt composition
│   ├── decode.py                  # Stage 6: logit-masked constrained decoding
│   └── predict.py                 # top-level predict(image, query, choices) function
└── weights/                      # NOT present in this repo -- expected at deployment
    ├── qwen3-vl-4b-instruct/      # unquantized base weights (local, no internet)
    ├── lora-medvqa/               # fine-tuned LoRA adapter -- NOTE: must be retrained
    │                              #   against Qwen3-VL-4B specifically; an adapter
    │                              #   trained against Qwen2-VL-7B OR Qwen3-VL-8B is
    │                              #   NOT architecture-compatible (different hidden
    │                              #   sizes/attention config per model size)
    ├── modality-router/            # optional, only used if USE_LEARNED_ROUTER is enabled
    └── stain_reference_matrix.npy  # optional; falls back to the standard Macenko
                                     #   reference vectors if absent (src/config.py)
```

### 5.2 `predict()` Reference Implementation

**As implemented** (`src/predict.py`) — this reflects the actual current pipeline, including the Section 5.5 failure-mode safeguards inline (they are not a separate layer wrapped around a simpler core):

```python
# src/predict.py
import logging, threading
from PIL import Image, UnidentifiedImageError

from src import config
from src.model_loader import model, processor         # import triggers global load (Section 4.1)
from src.router_modality import route_modality          # Stage 1 (Section 1.2 / 1.4)
from src.router_intent import detect_track                # Stage 3 (Section 2.1)
from src.preprocessing import preprocess_image, universal_normalize  # Stage 0 + 2
from src.prompt_builder import build_system_prompt          # Stage 4 (Section 2.3)
from src.decode import constrained_predict_letter             # Stage 6 (Section 4.2)

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


def predict(image, query: str, choices: dict) -> str:
    # --- Corrupt/unreadable image guard (Section 5.5) ---
    try:
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image.load()
    except (UnidentifiedImageError, OSError, ValueError):
        return config.FALLBACK_ANSWER_LETTER

    # --- Malformed-choices guard: exactly {"A","B","C","D"} keys (Section 5.5) ---
    if not (isinstance(choices, dict) and set(choices) == set(config.CHOICE_LETTERS)):
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

    # Stage 5+6: single forward pass + constrained decode, under a hard timeout
    answer = _run_with_timeout(lambda: constrained_predict_letter(inputs), config.INFERENCE_TIMEOUT_SECONDS)
    return answer if answer is not None else config.FALLBACK_ANSWER_LETTER
```

Differences from an earlier draft of this section, made explicit: the coarse-modality-only `detect_modality()` import is now `route_modality()` (returns `(modality, subtype)`); `preprocess_image()` takes both `modality` and `subtype`; `.to("cuda")` is `.to(config.DEVICE)`; and the corrupt-image/malformed-choices/timeout guards (Section 5.5) are inline in the real function, not layered on separately as this section previously implied.

### 5.3 Harness / CSV Writer

**As implemented** (`run_predictions.py`, at the repository root — not inside a `submission/` wrapper; see Section 5.1):

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

**Current actual content** (`flash-attn` is commented out — see the note below and Section 4.4):

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
# flash-attn>=2.6.0
```

**No `autoawq` or `gptqmodel`, deliberately:** the backbone migrated off AWQ-quantized Qwen2-VL-7B specifically to eliminate this dependency chain (Section 3.3) — `autoawq` is archived/deprecated (2025-05-11, unmaintained since), and its successor `gptqmodel` requires a C++ toolchain to build from source on Windows, which `transformers`' AWQ quantizer eventually made a hard, unconditional runtime requirement with no config-level opt-out. Neither package is needed at all for native, unquantized Qwen3-VL loading.

`transformers>=4.57.0` (bumped from `4.45.0`): 4.57.0 is the minimum release with Qwen3-VL support (`Qwen3VLForConditionalGeneration`, `Qwen3VLProcessor`).

`torch>=2.5.0` / `torchvision>=0.20.0` (bumped from `2.3.0`): this floor predates the Qwen3-VL migration — it was originally driven by `Qwen2VLVideoProcessor` requiring PyTorch >= 2.5 (disabling itself with an `ImportError` otherwise). `Qwen3VLProcessor` wraps `Qwen2VLImageProcessor` for images and `Qwen3VLVideoProcessor` for video (per `transformers`' own Qwen3-VL docs), so the same torch-version sensitivity carries forward; the floor stays at `2.5.0` regardless. `datasets` / `huggingface_hub` support `src/evaluate_omnimed.py`'s OmniMedVQA loading (Section 5, evaluation tooling) and aren't needed by the `predict()` runtime path itself.

*(Pin exact versions post-validation against the offline environment. `flash-attn` is commented out because it requires a matching CUDA toolkit and is often not buildable without sudo in the offline eval environment; `src/model_loader.py`'s `_load_model()` already falls back to the `sdpa` attention backend automatically when it's unavailable (Section 4.1/4.4), so leaving it commented out is a safe default, not a broken one. Uncomment and reinstall only once a compatible wheel/toolkit is confirmed available in the target environment.)*

### 5.5 Failure-Mode Safeguards

**As implemented** (`src/predict.py`):

- **Timeout guard:** `_run_with_timeout()` wraps the Stage 5+6 constrained-decode call (`constrained_predict_letter()`, which itself calls `model(**inputs)`) on a **daemon worker thread**, with a `config.INFERENCE_TIMEOUT_SECONDS` (default `5.0`) join timeout — thread-based rather than signal-based, since `signal.alarm` is unavailable on Windows and unsafe outside the main thread. This can't forcibly cancel a stuck CUDA call — the worker thread keeps running in the background — but it does prevent one stalled query from blocking the harness's average-inference-time measurement. On timeout or any exception, returns `config.FALLBACK_ANSWER_LETTER` (currently `"A"`, a fixed constant, not a distribution-derived value).
- **Corrupt/unreadable image guard:** `try/except` around `PIL.Image.open()` / `.load()`, catching `UnidentifiedImageError`, `OSError`, and `ValueError`; on failure, returns `config.FALLBACK_ANSWER_LETTER` immediately, before any other stage runs.
- **Empty/malformed choices guard:** validates that `choices` is a `dict` whose key set is **exactly** `{"A", "B", "C", "D"}` (`set(choices) == set(config.CHOICE_LETTERS)`) — not merely "4 keys" of any kind — before any prompt construction; degrades to the fallback letter otherwise.
- **Deterministic decoding:** no `generate()`/sampling call exists in the hot path at all — `constrained_predict_letter()` runs a single forward pass and an `argmax` over precomputed candidate logits (Section 4.2), so there is no `do_sample` flag to get wrong; behavior is deterministic and reproducible by construction.

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
| Fine-tune on OmniMedVQA/PMC-VQA/PathVQA/VQA-RAD offline | Compliant with "no training at evaluation time"; matches heterogeneity profile of the challenge (strategy documented in Section 3; no training script ships in this repo) |
