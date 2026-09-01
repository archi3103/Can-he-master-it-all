# Medical VQA Pipeline — Technical Audit and Experimentation Roadmap

**Scope:** Deep architectural review of a 7-stage (Stages 0–6) production medical Visual Question Answering pipeline built on `Qwen3-VL-4B-Instruct`, with prioritized recommendations across nine experimentation areas.

**Document status:** Working engineering reference. All performance figures are drawn from published literature and are directional; your own held-out development set is the only authoritative measure for decision-making.

---

## Table of Contents

0. [Status Update — Implementation Progress](#status-update--implementation-progress)
1. [Executive Summary](#executive-summary)
2. [Area 1 — Modality Classification (Stage 1 Router)](#area-1--modality-classification-stage-1-router)
3. [Area 2 — Query-Intent Routing (Stage 3)](#area-2--query-intent-routing-stage-3)
4. [Area 3 — VLM Backbone Alternatives](#area-3--vlm-backbone-alternatives)
5. [Area 4 — Fine-Tuning Datasets and Mixing Strategy](#area-4--fine-tuning-datasets-and-mixing-strategy)
6. [Area 5 — Volumetric File Support (.nii and DICOM)](#area-5--volumetric-file-support-nii-and-dicom)
7. [Area 6 — Benchmarking Framework and Dataset Mapping](#area-6--benchmarking-framework-and-dataset-mapping)
8. [Area 7 — Prompt Construction and Clinical Persona](#area-7--prompt-construction-and-clinical-persona)
9. [Area 8 — Evaluation Metrics and State-of-the-Art Baselines](#area-8--evaluation-metrics-and-state-of-the-art-baselines)
10. [Area 9 — Data Augmentation in Medical VQA](#area-9--data-augmentation-in-medical-vqa)
11. [Implementation Roadmap](#implementation-roadmap)
12. [Closing Assessment](#closing-assessment)

---

## Status Update — Implementation Progress

**As of 2026-09-01. Submission deadline: 2026-09-02, 1159 hours — roughly a day and a half remaining.** This section tracks which of this document's recommendations have been acted on since the audit was written, which were deliberately deferred, and which are still open. Cross-reference `medical_vqa_architecture.md` (updated architecture) and `datasets.md` (full dataset breakdown) for implementation detail — this section is status only, not a re-derivation.

| Item | Status | Where |
|---|---|---|
| **P0** Volumetric ingest layer (Stage −1) | ✅ **DONE** | `src/volume_loader.py` — NIfTI (`nibabel`, `as_closest_canonical`, `mmap=False`) + DICOM single/series (`pydicom`, geometric sort, RescaleSlope/Intercept, MONOCHROME1) + RGBY channel-grid tiling; Vote-MI-inspired content-based slice selection (variance + edge density) with graded fallback |
| **P0** Fallback degraded gracefully (tiled → single slice → blind model, not straight to constant) | ⚠️ **PARTIAL — scoped differently than recommended** | `predict_with_diagnostics()` does the full graded chain down to a blind gray-canvas model call. The actual competition entrypoint `predict()` deliberately keeps the immediate constant-letter fallback (`config.FALLBACK_ANSWER_LETTER`) for safety/determinism under the harness's timeout — this was a conscious choice, not an oversight, but it means the audit's "never jump directly to the constant" recommendation is realized in the diagnostics path, not the submitted path. **Worth a final decision before submission**: is `predict()` still expected to see the same volumetric inputs the old blind-`"A"` bug was written against, or should the blind-model retry be pulled into `predict()` itself now that Stage −1 makes it cheap? |
| **P0** Ablation harness (`PipelineConfig`, sweep runner, bootstrap CI, McNemar) | ⚠️ **PARTIAL** | `evaluate_omnimed.py` has a diagnostic sidecar CSV, per-item stratification (modality/intent/n_choices/fallback), and bootstrap CIs (10,000 resamples). No `PipelineConfig` dataclass, no automated sweep runner, no McNemar pairwise testing, and the five mandatory baselines (random / always-"A" / blind model / naked backbone / full pipeline) were not run as a battery — not attempted this session, given time spent on the volumetric gap and fine-tuning instead |
| **P0** Option-order debiasing / null-prompt calibration | ❌ **NOT DONE** | Not attempted. Still the cheapest documented lever (Area 7) that hasn't been picked up — worth doing only if time remains after the submission package is otherwise safe |
| **P1** Backbone comparison (MedGemma-4B, HuatuoGPT-Vision-7B, Qwen2.5-VL-7B) | ❌ **NOT DONE — explicitly out of scope** | Deadline-driven decision: swapping backbones this late would invalidate the LoRA adapter and quantization verification work already done. Qwen3-VL-4B-Instruct remains the backbone |
| **P1** LoRA fine-tuning on MCQ-formatted medical data | ✅ **Tooling DONE; training run itself is the user's remaining execution step** | `src/train_lora.py` (QLoRA 4-bit default, answer-letter-only loss masking, verified end-to-end with a real training run — all 144 `lora_B` matrices confirmed non-zero, round-tripped through `model_loader.py`), `kaggle_finetune_lora.ipynb` for the actual Kaggle run. Training-mix composition differs from the audit's original recipe (PMC-VQA/PathVQA/SLAKE/VQA-RAD) — see below |
| — Dataset selection for fine-tuning/eval, covering all 8 PS-named modalities on both sides, including genuine volumetric CT/MRI | ✅ **DONE** (not in original audit — added this session because OmniMedVQA's CT/MRI are pre-sliced 2D and cannot exercise Stage −1) | `src/prepare_external_dataset.py` (MosMedData, BrainMRI/Task01_BrainTumour, BUSI → OmniMedVQA-shaped local mirror, deterministic case-level 80/20 split), full breakdown in `datasets.md` |
| — Image-hash deduplication between train/eval (Area 4 contamination guard) | ⚠️ **PARTIAL** | Splits are deterministic and case-level (no case appears in both train and eval within a dataset), but there is no image-hash dedup *across* the OmniMedVQA sets used for eval vs. fine-tuning — relied on manually choosing disjoint dataset names instead (see `datasets.md` §1/§2). Lower risk than the audit's original concern since nothing here trains on OmniMedVQA sets used for scoring, but not independently verified by hash |
| — Vision-tower freeze / general-domain replay / per-epoch option shuffling (Area 4 forgetting mitigations) | ❌ **NOT DONE** | `train_lora.py` does LoRA + answer-token loss masking only; none of the other mitigations in the audit's ranked list were implemented. Acceptable risk for a short, low-epoch fine-tune under deadline pressure, but a real one — the LoRA rank/target-module regularization is the only forgetting guard in place |
| — `visual_reliance` tracking (blind-vs-real accuracy per checkpoint) | ❌ **NOT DONE** | Not built. The blind gray-canvas capability exists (`predict_with_diagnostics()`, `_neutral_gray_canvas()`) but isn't wired into a training-checkpoint comparison loop |
| **P2** Improved modality router / intent classifier / two-pass CoT | ❌ **NOT DONE — deprioritized** | Correctly identified in the original audit as lower-priority than the P0 items; no session time was spent here, which matches the audit's own ordering |
| **New — not in original audit:** Quantization for 10GB submission-size compliance | ✅ **Load-time DONE and verified**; ⚠️ **save/reload round-trip UNVERIFIED locally** | `src/model_loader.py` (`BitsAndBytesConfig`, NF4 4-bit + double-quant, opt-in via `config.QUANTIZATION_MODE`) — confirmed working at load time (2.91GB VRAM, `Linear4bit` layers). Whether `model.save_pretrained()` on the quantized model produces a genuinely compact on-disk checkpoint could **not** be confirmed on the local dev machine (Windows paging-file/disk-space errors blocked the test). `kaggle_finetune_lora.ipynb` Section 9 has an explicit verify-then-fallback cell (VRAM assert, disk-size check, printed warning) so this gets checked early on Kaggle rather than discovered at submission time. **This is the single largest open risk for the submission.** |
| **New — not in original audit:** SOTA-comparable OmniMedVQA scoring (Prefix-based Score) | ✅ **DONE** | `src/prefix_score.py` implements OmniMedVQA's own published Prefix-based Score methodology (reprocesses full prefix+candidate text per candidate due to an M-RoPE splicing bug discovered along the way); `evaluate_omnimed.py --scoring-method prefix_score` vs. the pipeline's own fast letter-argmax decoding (`--scoring-method pipeline`) |
| **New — not in original audit:** Google Drive FUSE-mount I/O latency fixes | ✅ **DONE** | `nib.load(path, mmap=False)` fixed a 444s→2.5s NIfTI pathology over Drive; `ThreadPoolExecutor` parallel reads fixed ~20s DICOM-series/channel-grid latency |
| `eval.py` / `dev_metadata.csv` parsing correctness | ✅ **DONE** | Duplicate-download-suffix filename fallback, `cp1252` CSV encoding fix (confirmed via raw byte inspection, not `latin1`) |

**Net read**: the audit's Finding 1 (volumetric-blind) is closed. Finding 2 (unmeasured pipeline) is only partially addressed — there's now per-item diagnostic logging but no formal ablation/baseline battery, so the marginal value of Stages 1–4 (router, intent classifier, persona prompts) is still unmeasured, same as when the audit was written. Finding 3 (underexploited decoding — calibration, debiasing) is untouched. Given the deadline, the recommended remaining order is: (1) get the Kaggle fine-tuning run + quantization verification done and the submission zip under 10GB, (2) only if time remains, pick up null-prompt calibration (Area 7a) since it's a few lines and one extra forward pass total, (3) everything else in this table stays backlog for after submission.

---

## Executive Summary

The pipeline is well-engineered and the zero-crash design philosophy is sound. However, the audit identifies two structural problems that currently dominate all other considerations.

### Finding 1 — Volumetric inputs are answered blind

`.nii` files and DICOM directories fail at the PIL layer and fall through to a hardcoded `"A"`. These items are answered without ever consulting the model. If volumetric content represents *V*% of the evaluation set, up to *V*% of the achievable score is forfeited before inference begins. No amount of prompt engineering, router tuning, or backbone scaling can recover this. **This is the single highest-value fix in the document.**

### Finding 2 — Six stages of preprocessing logic with no measured contribution

The pipeline contains a modality router, subtype-specific preprocessing, an intent classifier, and 20 cached prompt permutations — and no ablation harness to determine whether any of it improves accuracy. The contribution could plausibly be +4 points or −2 points; the current instrumentation cannot distinguish these cases. Building the ablation harness produces zero direct accuracy gain but is the precondition for every other decision in this document.

### Finding 3 — Underexploited decoding architecture

Stage 6 performs a single-pass argmax over pre-computed choice token IDs. This design makes two well-established, low-cost accuracy improvements — null-prompt calibration and option-order debiasing — nearly free to implement. Most pipelines cannot do this cheaply. Yours can.

### Prioritized backlog (ordered by impact per unit of effort)

| Priority | Item | Effort | Expected gain | Rationale | Status |
|---|---|---|---|---|---|
| **P0** | Volumetric ingest layer (`pydicom` + `nibabel`) | Medium | **+2 to +15 pts** (scales with volumetric share) | Converts blind guesses into informed predictions | ✅ DONE |
| **P0** | Ablation harness and stratified evaluation logging | Low | 0 pts directly; unlocks all other work | Currently operating without measurement | ⚠️ PARTIAL (sidecar+strata done, no sweep runner/McNemar) |
| **P0** | Option-order debiasing and null-prompt calibration | **Low** | **+1 to +5 pts** | Logit machinery already exists | ❌ NOT DONE |
| **P1** | Backbone comparison (MedGemma-4B, HuatuoGPT-Vision-7B, Qwen2.5-VL-7B) | Medium | **+3 to +10 pts** | Backbone dominates all downstream engineering | ❌ NOT DONE — out of scope for deadline |
| **P1** | LoRA fine-tuning on MCQ-formatted medical data | Medium–High | **+5 to +15 pts** | Largest lever after backbone selection | ✅ Tooling DONE (`train_lora.py`, verified run); actual Kaggle run is the user's next step |
| **P2** | Improved modality router (BiomedCLIP gate or MobileNetV3) | Medium | +0 to +2 pts | Only material if Stage 2 preprocessing demonstrably helps | ❌ NOT DONE — deprioritized per this doc's own ordering |
| **P2** | Improved intent classifier (linear probe / SetFit) | Low | +0 to +1 pts | Ablate first; the track routing may contribute nothing | ❌ NOT DONE — deprioritized |
| **P2** | Two-pass chain-of-thought decoding | Medium | +2 to +6 pts, at **10–20× latency** | Requires raising the 5.0s timeout | ❌ NOT DONE — deprioritized |
| **P3** | Training-time image augmentation | Medium | +1 to +3 pts | Only relevant once fine-tuning begins; see Area 9 risks | ❌ NOT DONE |

---

## Area 1 — Modality Classification (Stage 1 Router)

### Current implementation

Twelve hand-engineered image statistics (Laplacian variance, saturation, border darkness, hue entropy, and others) feed a two-level decision process: four coarse streams, then fine subtypes. Zero GPU cost, fully deterministic, and straightforward to debug. These are real advantages that should not be discarded lightly.

### Weaknesses

The core issue is that these statistics measure **acquisition and encoding properties**, not clinical content.

- **Border darkness** is unreliable. It collapses when a scan has already been cropped, when a DICOM export includes padding, or when an ultrasound arrives as a full-screen capture with UI chrome. Identical anatomy can produce opposite votes.
- **Saturation** separates grayscale from color reliably, but cannot distinguish dermoscopy from endoscopy from a clinical photograph of a specimen container. All three land in "Macroscopic."
- **Laplacian variance is resolution-coupled.** A 512×512 CT and a 2048×2048 CT downsampled by the `max_pixels` cap produce different texture energy. This means the router's decision boundary shifts silently whenever the memory configuration changes — a hidden coupling between two unrelated settings.
- **Hue entropy** for H&E-stained histology and for soft-tissue clinical photography occupy overlapping ranges.
- **No confidence estimate.** The router emits a hard label, and Stage 2 immediately applies a destructive transform based on it.

### The critical insight: the risk lives in Stage 2, not Stage 1

A misclassification is harmless if all downstream branches preprocess similarly. Your branches are aggressive and irreversible:

- Macenko stain normalization applied to a chest radiograph imposes meaningless color statistics on a grayscale image.
- CT bone windowing applied to a fundus photograph crushes dynamic range and eliminates retinal vasculature.
- Whole-slide tissue-tile selection applied to an ultrasound crops out the region of interest and retains speckle.

The quantity that matters is therefore **router error rate × Stage-2 destructiveness**, not router error rate alone. This reframing changes the recommended fix.

### Recommended approaches

#### Option A — Soft routing with a conservative Stage 2 (implement first, approximately one day)

```python
# Replace: stream = router(img); img = STAGE2[stream](img)
scores = router_scores(img)                 # dict: stream -> [0,1], normalized
top, second = top_two(scores)
margin = scores[top] - scores[second]

if margin < CONF_TAU:                       # tune on dev; start around 0.25
    img = generic_safe_preprocess(img)      # letterbox + mild CLAHE only
else:
    img = STAGE2[top](img)
```

In parallel, partition Stage-2 operations into two tiers:

- **Safe:** letterboxing, mild CLAHE, gray-world white balance. Applicable to almost anything without harm.
- **Destructive:** Macenko normalization, hard windowing, tissue-tile cropping. Gate these behind a high confidence threshold.

This captures most of the robustness benefit at zero latency cost and without new dependencies.

#### Option B — BiomedCLIP zero-shot gate on the uncertain tail

`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` is trained on PubMed Central figure–caption pairs, which covers essentially every medical imaging modality. Zero-shot classification with prompts such as `"a chest radiograph"`, `"an H&E stained histopathology slide"`, `"a dermoscopic image of skin"` works well without any labeled data.

- **Cost:** ~86M image-tower parameters; roughly 5–10 ms per image on GPU at batch size 1, or 60–120 ms on CPU.
- **Deployment pattern:** invoke only when the heuristic margin is low. If heuristics are confident on 80% of inputs, average added latency is approximately 0.2× the CLIP cost.
- **Advantages:** no labeling effort, robust to acquisition style, produces reasonably calibrated scores.
- **Disadvantages:** a second model in memory (~350 MB in FP16) competing for VRAM with an unquantized FP16 4B backbone; an additional dependency.

#### Option C — Small fine-tuned CNN (highest accuracy ceiling)

MobileNetV3-Small (~2.5M parameters) or EfficientNet-B0 trained on modality labels. Labels can be assembled at no annotation cost: OmniMedVQA ships modality metadata across 12 modalities, and datasets such as VQA-RAD, PathVQA, and dermoscopy collections are single-modality by construction. A few thousand images per class typically yields >97% modality accuracy.

- **Latency:** 2–4 ms on GPU, 10–20 ms on CPU at 224² resolution.
- **Advantages:** highest accuracy, minimal footprint, ONNX-exportable, no heavy dependency.
- **Disadvantages:** requires a training run and a labeled corpus; will not generalize to unlabeled modalities. Mitigate with an explicit `unknown` class trained on out-of-distribution imagery.

### Recommendation

Implement **Option A immediately**. Add **Option C** once a training loop exists for Area 4 work. Retain **Option B** as the zero-labeling stopgap. In all cases, log the router decision as a debug column in the prediction sidecar so accuracy can later be sliced by predicted modality.

---

## Area 2 — Query-Intent Routing (Stage 3)

### Current implementation

A regex/keyword fast path, falling back to cosine similarity against per-track centroid embeddings from a frozen, CPU-bound `all-MiniLM-L6-v2`.

### Critique

**The centroid representation is the primary weakness, not the regex.** A class centroid is the mean of a handful of seed phrases, which imposes a spherical, equal-radius decision region on every track. Real query distributions are neither spherical nor equally sized: "Differential" questions are linguistically diverse, while "Modality-ID" questions consist of a few repeated templates. Centroid cosine similarity systematically over-assigns the broad class and under-assigns the narrow one.

Additional issues:

- **Regex-first means regex is never overruled, only consulted first.** A question containing "left" is routed to Spatial even when the question is *"which of the following is the most likely diagnosis of the left upper lobe mass?"* — a Diagnostic question with incidental spatial vocabulary.
- **Single-label routing against multi-label reality.** Most clinical MCQ items are simultaneously Diagnostic and Severity, or Spatial and Differential.
- **No abstention path.** Every question receives a track, including cases where top-1 cosine is 0.31 against a runner-up of 0.30.
- **Domain mismatch.** MiniLM is trained on general web text. Terms such as "enhancement," "attenuation," "grade," and "stage" carry radiology-specific senses that general embeddings represent poorly.

### Required first step: ablate before optimizing

Run three configurations on the development set:

1. Full pipeline with all 5 tracks.
2. A single generic prompt for all questions.
3. Random track assignment.

**If (1) ≈ (2), the intent router contributes nothing** and further investment is wasted. This outcome is common — strong instruction-tuned VLMs largely ignore persona framing under constrained MCQ decoding. If (2) ≈ (3) as well, prompt variety is pure noise, and Stages 3 and 4 can be collapsed entirely, reclaiming the CPU model and a substantial amount of complexity.

Proceed with the improvements below only if (1) exceeds (2) by a margin whose bootstrap confidence interval excludes zero.

### Improved methods, ordered by cost

#### 1. Linear probe on frozen MiniLM embeddings (approximately 30 minutes of work)

Retain the identical embedding model; replace `argmax cosine(e, centroid_k)` with logistic regression trained on roughly 50–200 labeled questions per track. This learns a genuine decision boundary with per-class scaling and yields calibrated probabilities, which enables abstention.

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
clf.fit(embeddings_train, track_labels)          # trains in ~1 second

probs = clf.predict_proba(emb)[0]
track = TRACKS[probs.argmax()] if probs.max() >= 0.55 else "GENERIC"
```

This strictly dominates centroid cosine at identical latency and with no new dependencies. It is worth implementing even if nothing else in this area is pursued.

#### 2. SetFit (best few-shot option)

SetFit contrastively fine-tunes the sentence transformer on 8–32 examples per class, then fits a classification head. It typically outperforms a fine-tuned DistilBERT in the sub-100-examples-per-class regime, trains in minutes on CPU, and the deployed artifact remains a MiniLM encoder plus a lightweight head.

#### 3. Fine-tuned DistilBERT / MiniLM sequence classifier

Requires roughly 500+ labeled examples per track to outperform SetFit. Approximately 66M parameters, 5–15 ms on CPU. Justified only with substantial labeled volume.

#### 4. Zero-shot NLI (`bart-large-mnli`) — not recommended here

Approximately 400M parameters, 200 ms+ per question on CPU, and it must run once per candidate label. The latency profile is inappropriate for a routing decision that precedes the main model call.

#### 5. Label bootstrapping

If no labeled questions exist: sample 2,000 training questions, have a strong LLM assign them to the 5 tracks in batches, manually verify 200, then train method 1 or 2 on the result. This takes roughly half a day and produces a labeled set that doubles as evaluation slice keys.

### Recommendation

Ablate first, then implement the linear probe with an abstention threshold, then consider SetFit if the ablation demonstrated real lift. Extend the router to emit a primary and secondary track, allowing Stage 4 to append a secondary clause. Log the predicted track to the prediction sidecar for stratified analysis.

---

## Area 3 — VLM Backbone Alternatives

The backbone is where the largest share of achievable accuracy resides — considerably more than in prompt templates or preprocessing.

### Assessment of the current backbone

`Qwen3-VL-4B-Instruct` is a stronger choice for this task than is often appreciated:

- Native dynamic resolution and strong OCR capability. Medical figures frequently contain burned-in text, scale bars, and laterality markers ("L"/"R"); a model that reads these gains accuracy at no additional cost.
- Excellent instruction-following, which matters substantially for constrained multiple-choice tasks.
- 4B in FP16 occupies roughly 8–9 GB, leaving headroom for vision prefill at the configured `max_pixels` cap.

Its principal weakness is **medical visual priors**: it has seen far fewer histopathology tiles and fundus photographs than general web imagery.

### Candidate comparison

| Model | Params | Medical pre-training | FP16 weights | Assessment for constrained MCQ |
|---|---|---|---|---|
| **Qwen3-VL-4B-Instruct** (current) | 4B | No | ~8 GB | Strong general baseline, excellent instruction-following |
| **MedGemma-4B-it** | 4B | Yes | ~8 GB | **Strongest same-size candidate.** Direct size-for-size swap |
| **HuatuoGPT-Vision-7B** | 7B | Yes | ~15 GB | Strong multi-modality medical MCQ (PubMedVision training data) |
| **Lingshu-7B** | 7B | Yes | ~15 GB | Recent, competitive on medical benchmarks |
| **Qwen2.5-VL-7B-Instruct** | 7B | No | ~15 GB | Useful control arm for isolating parameter scaling |
| **InternVL3 / 3.5-8B** | 8B | No | ~16 GB | Strong general performance, good high-resolution tiling |
| **LLaVA-Med-7B** | 7B | Yes | ~15 GB | Dated; generative-first, weak instruction-following |
| **Med-Flamingo** | 8.3B | Yes | ~17 GB | Dated; few-shot era architecture, poor on modern MCQ |
| **LLaVA-1.5-7B** | 7B | No | ~15 GB | Superseded; weak OCR, fixed 336px input |

### Published performance context

**OmniMedVQA** (multiple-choice, ~118k images / ~128k QA pairs across 12 modalities):

- Older medical models perform poorly: Med-Flamingo ~34.9% average, LLaVA-Med ~41.3% average.
- HuatuoGPT-Vision-7B reaches ~50.0% average.
- Notably, general-purpose BLIP-2 outperformed several medical-domain LVLMs in the original benchmark paper. **Medical pre-training is not automatically beneficial** — the quality and recency of the adaptation matters more than the label "medical."
- Modern general models: Qwen2.5-VL-7B ~60.8%, LLaVA-v1.6-34B ~58.7%, Qwen2-VL-72B ~68.1% on modality subsets. Strong specialist and reasoning-oriented systems reach ~73–78%.

**VQA-RAD and SLAKE:** MedGemma-4B shows roughly +16 points token-F1 over its own Gemma-3-4B base on VQA-RAD (49.9 vs 33.6) and SLAKE (72.3 vs 40.2). This is the cleanest available evidence that **medical pre-training at equal parameter count delivers substantial gains**.

**Supervised fine-tuned specialists** on closed-form subsets reach VQA-RAD ~85%, SLAKE ~92%, PathVQA ~95%. These are in-domain SFT ceilings and are **not appropriate zero-shot targets**.

### Scaling economics for a single-GPU deployment

- **4B → 7B/8B:** typically **+3 to +8 points** on medical MCQ, at roughly 2× weights and 1.6–2× combined prefill and decode time.
- **7B → 32B/72B:** a further **+5 to +10 points**, but out of reach for a single consumer card in FP16.
- **General 4B → medical 4B:** **+5 to +15 points** on radiology and pathology at **zero additional cost**.

**Conclusion: at this size class, medical pre-training delivers more accuracy per unit of VRAM than parameter scaling.**

### Two implications of the constrained-decoding design

1. **Instruction-following is largely bypassed.** Because Stage 6 reads logits rather than parsing generated text, the "LLaVA-Med cannot follow instructions" weakness is partially neutralized in this specific setup. Older generative medical models score better under prefix/likelihood scoring than under free-generation scoring. It is therefore worth re-testing them under your actual harness rather than relying on their published headline numbers.
2. **Reasoning-tuned models cannot express their advantage.** Any model whose gains derive from chain-of-thought (Med-R1 and similar reasoning-distilled variants) produces no benefit in a single forward pass. Do not pay for these unless two-pass decoding is also adopted (see Area 7).

### Recommended experiment

Hold everything else constant and run the identical harness on the identical development set across four backbones:

- `Qwen3-VL-4B-Instruct` (control)
- `MedGemma-4B-it` (medical, same size)
- `HuatuoGPT-Vision-7B` (medical, larger)
- `Qwen2.5-VL-7B-Instruct` (general, larger — isolates scaling from medical pre-training)

Report accuracy, peak VRAM, and p50/p95 latency. Select from the accuracy-per-second Pareto front. If MedGemma-4B wins, the gain is free. If the 7B wins by less than 2 points, remain at 4B and reinvest the headroom in volumetric decoding and multi-slice tiling.

### VRAM planning for a 7B move

A 7B model in FP16 requires ~15 GB for weights plus vision prefill overhead. On a 24 GB card this is comfortable at moderate `max_pixels`. On 16 GB, 8-bit or AWQ/GPTQ 4-bit quantization becomes necessary.

**Quantization caveat specific to this pipeline:** quantization perturbs the exact logits being compared during argmax. Weight-only INT8 typically costs under 1 point; 4-bit can shift choice-token calibration by 1–3 points and interacts poorly with option-order bias. If quantizing, re-tune calibration (Area 7) **after** quantization, never before.

---

## Area 4 — Fine-Tuning Datasets and Mixing Strategy

### Available datasets

| Dataset | Size | Modality | Format | Primary use |
|---|---|---|---|---|
| **VQA-RAD** | ~3.5k QA / 315 images | Radiology (CXR, CT, MRI) | Open + closed | Radiology reasoning; very small |
| **SLAKE** | ~14k QA / 642 images | Radiology, bilingual EN/ZH | Open + closed; includes KG and segmentation masks | Spatial/anatomy track |
| **PathVQA** | ~32k QA / ~4,998 images | Histopathology | Open + yes/no | Microscopy stream |
| **PMC-VQA** | ~227k QA / ~149k images | Mixed (PMC figures) | **Native multiple-choice A–D** | **Best format match to your task** |
| **PubMedVision** | ~1.3M | Mixed | Instruction | Large-scale alignment stage |
| **OmniMedVQA** | ~128k QA / 118k images, 12 modalities | All | Multiple-choice | **Evaluation only** |
| **Quilt-VQA / Quilt-1M** | ~1M pairs | Histopathology | Caption / VQA | Pathology depth |
| **VQA-Med 2019/2021** | ~15k / ~5k | Radiology | Short answer | Modality-ID track |
| **GMAI-MMBench** | Very large, 38 modalities | All | MCQ | **Evaluation only** |
| **ROCO / MedICaT** | ~80k / ~200k | Mixed radiology | Captions | Vision-encoder alignment warmup |

### Contamination risks

These are the most common sources of invalid results in this domain.

- **Never train on OmniMedVQA or GMAI-MMBench if you intend to report on them.** Both are assembled from other public datasets, which means training on VQA-RAD or PathVQA can leak into OmniMedVQA indirectly. **Deduplicate by image hash, not by dataset name.**
- **The original VQA-RAD split contains duplicated images across train and test** (identical images paired with different questions). The MedGemma authors published clean splits specifically to eliminate this. Use clean splits, or your reported numbers will be inflated and non-comparable.
- **PMC-VQA distractors are LLM-generated and noisy.** A meaningful fraction of items are answerable from the question text alone.

### The text-shortcut risk

Recent grounding research shows models scoring approximately 63% on VQA-RAD while retaining **around 81% of that performance with blank images**. Text-only reinforcement learning has produced *negative* visual reliance on PathVQA (i.e., the model performed better with mismatched images). The practical implication: **a fine-tuned model with higher accuracy may have learned to read the question better, not the image better.**

Guard against this with a permanent evaluation mode:

```python
# Blind evaluation: identical questions, image replaced by a neutral gray canvas
acc_real  = evaluate(pipeline, dev, image_mode="real")
acc_blind = evaluate(pipeline, dev, image_mode="blank")

visual_reliance = (acc_real - acc_blind) / max(acc_real - random_chance, 1e-6)
```

Track `visual_reliance` on every fine-tuning checkpoint. If accuracy rises while reliance falls, the training run has produced a language-prior model rather than a vision model, and the checkpoint should be rejected.

### Recommended mixing recipe

**Stage A — Format alignment (short, 1 epoch).** Convert all sources into the exact inference format: image + question + lettered options + "Respond with ONLY the letter." For open-ended sources (PathVQA, VQA-RAD open subset), synthesize distractors by sampling answers from the same modality and question-type bucket. For MCQ tasks, format match matters more than raw data volume.

**Stage B — Balanced domain mix.** Suggested per-step sampling weights:

```
30%  PMC-VQA            (format-native MCQ, broad coverage)
20%  PathVQA            (microscopy — currently the weakest stream)
15%  SLAKE              (spatial / anatomy)
10%  VQA-RAD (clean)    (radiology diagnostic)
10%  Derm / fundus / US  (ISIC, ODIR, BUSI converted to MCQ)
15%  General replay     (LLaVA-Instruct or similar — the forgetting brake)
```

### Catastrophic forgetting mitigations, ordered by effectiveness

1. **Use LoRA rather than full fine-tuning.** Rank 8–32 on attention projections (`q`, `k`, `v`, `o`), optionally extending to MLP layers. Low rank is itself a strong regularizer — base weights cannot drift.
2. **Freeze the vision tower for the first epoch.** Most medical VQA gains originate in the language and adapter layers. Unfreezing the ViT early is the fastest route to destroying general visual competence.
3. **Retain 15–20% general-domain replay.** Inexpensive and consistently effective.
4. **Low learning rate, short schedule.** LoRA LR of 1e-4 to 2e-4, 1–2 epochs. Medical VQA sets are small; three or more epochs memorizes.
5. **Loss masking.** For MCQ items, compute loss only on the answer-letter token. This aligns the training objective exactly with argmax decoding. However, it also makes "always predict B" trivially learnable, which requires the next item.
6. **Shuffle option order every epoch.** Randomly permute A–D per sample per epoch. This is the single most important augmentation in this document and costs nothing.
7. **Hold out a per-dataset development slice plus a general-VQA canary set.** Early-stop on the canary set, not on the medical sets.

### Hardware requirements

LoRA on a 4B VLM with a frozen vision tower, bf16 precision, gradient checkpointing, batch size 4 with accumulation 8, fits in approximately 16–20 GB. A 7B requires QLoRA (4-bit base) at roughly 12–16 GB. Both are feasible on a single consumer GPU.

---

## Area 5 — Volumetric File Support (.nii and DICOM)

This is the highest-priority item in the audit.

### Root cause analysis

**PIL is a plugin-dispatch library.** `Image.open()` reads the leading bytes of a file and queries each registered plugin in turn. It ships plugins for PNG, JPEG, TIFF, BMP, GIF, WebP, and related formats. It has **no plugin for NIfTI and no plugin for DICOM Part 10**. When every plugin declines, it raises `UnidentifiedImageError`. This is correct behavior, not a defect — PIL is accurately reporting that the format lies outside its supported set.

#### NIfTI (`.nii` / `.nii.gz`)

- A 348-byte header (NIfTI-1) or 540-byte header (NIfTI-2), with magic bytes `n+1\0` at offset 344, followed by raw voxel data.
- `.nii.gz` is that structure gzip-wrapped, so the leading bytes are `1f 8b` and the header is not directly visible.
- The payload is a **3D or 4D array** (X, Y, Z[, T]), commonly `float32` or `int16`, accompanied by an **affine matrix** encoding voxel spacing and world orientation. There is no single 2D image inside to return, and PIL's API has no representation for a volume even if it could parse the container.

#### DICOM (`.dcm`)

- A 128-byte preamble followed by the four ASCII bytes `DICM` at offset 128, then a tag-length-value stream. PIL's format sniffers examine byte 0 and never reach offset 128.
- Pixel data resides in tag `(7FE0,0010)` and may be stored raw, RLE-compressed, or encapsulated as JPEG-Lossless, JPEG-LS, or JPEG-2000. Decoding compressed transfer syntaxes requires `pylibjpeg` or `gdcm`.
- Stored values are **not display values**. Correct rendering requires applying `RescaleSlope` and `RescaleIntercept` to obtain Hounsfield Units, then a VOI LUT or explicit window, and honoring `PhotometricInterpretation == MONOCHROME1` (which is inverted).

#### DICOM directory → `IsADirectoryError`

A CT or MR study is stored as one file **per slice**. Calling `Image.open("/path/CT_STUDY/")` invokes `open()` on a directory, and the operating system raises `IsADirectoryError` (errno 21). The existing guard catches this and emits the fallback answer.

**Summary: three distinct failure modes producing one shared symptom — a blind `"A"`.**

### Architectural blueprint: introduce Stage −1, "Volume Ingest"

Do not patch this inside Stage 0. Add a new stage **before** Stage 0 whose sole responsibility is converting any input path into a list of RGB PIL images. Stages 0 through 6 then remain untouched, and the zero-crash guarantee is preserved.

```
path ──► STAGE -1: VOLUME INGEST ──► List[PIL.Image (RGB)] ──► STAGE 0 (normalize) ──► ...
             │
             ├── dispatch by: is_dir? / suffix / magic bytes
             ├── .png .jpg .tif   → [Image.open(p).convert("RGB")]
             ├── .nii .nii.gz     → nibabel → reorient → window → slice policy
             ├── .dcm (single)    → pydicom → VOI LUT → [1 image]
             └── directory/series → pydicom → sort → stack → window → slice policy
```

#### Dispatcher (magic-byte based — do not trust file extensions)

```python
import gzip
from pathlib import Path

def detect_kind(path: str) -> str:
    p = Path(path)
    if p.is_dir():
        return "dicom_series" if any(
            f.suffix.lower() in {".dcm", ""} for f in p.iterdir()
        ) else "unknown"

    head = open(p, "rb").read(400)
    if head[:2] == b"\x1f\x8b":                       # gzip container
        head = gzip.open(p, "rb").read(400)

    if len(head) > 132 and head[128:132] == b"DICM":
        return "dicom_single"
    if head[344:348] in (b"n+1\x00", b"ni1\x00"):     # NIfTI-1
        return "nifti"
    if head[4:8] in (b"n+2\x00", b"ni2\x00"):         # NIfTI-2
        return "nifti"
    return "flat2d"
```

#### NIfTI reader

```python
import nibabel as nib
import numpy as np

def load_nifti(path):
    img = nib.load(path)                       # lazy: header only, voxels not yet read
    img = nib.as_closest_canonical(img)        # reorient to RAS+  — CRITICAL
    vol = np.asanyarray(img.dataobj)           # (X, Y, Z) or (X, Y, Z, T)
    if vol.ndim == 4:
        vol = vol[..., 0]                      # first volume / first echo
    zooms = img.header.get_zooms()[:3]         # voxel spacing in mm, for aspect correction
    return vol, zooms
```

**`as_closest_canonical` is not optional.** Without it, one scanner's volume is axial along Z while another's is sagittal along Z, and the "middle slice" becomes a coronal cut through the ear. This single line eliminates an entire class of silent errors.

#### DICOM series reader

```python
import pydicom
import numpy as np
from pathlib import Path
from pydicom.pixel_data_handlers.util import apply_voi_lut

def load_dicom_series(folder):
    dss = []
    for f in Path(folder).rglob("*"):
        if f.is_file():
            try:
                dss.append(pydicom.dcmread(str(f), force=True))
            except Exception:
                continue
    dss = [d for d in dss if hasattr(d, "PixelData")]
    if not dss:
        raise ValueError("no pixel data found")

    # 1) Split by series — a folder often contains a scout plus multiple sequences
    series = {}
    for d in dss:
        series.setdefault(getattr(d, "SeriesInstanceUID", "0"), []).append(d)
    chosen = max(series.values(), key=len)      # heuristic: longest series

    # 2) Sort geometrically, not by InstanceNumber (which is unreliable)
    def zpos(d):
        ipp = getattr(d, "ImagePositionPatient", None)
        iop = getattr(d, "ImageOrientationPatient", None)
        if ipp is not None and iop is not None:
            n = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
            return float(np.dot(np.array(ipp, float), n))   # project onto slice normal
        return float(getattr(d, "InstanceNumber", 0))
    chosen.sort(key=zpos)

    # 3) Stack with rescale to physical units (HU for CT)
    slices = []
    for d in chosen:
        a = d.pixel_array.astype(np.float32)
        a = a * float(getattr(d, "RescaleSlope", 1.0)) \
              + float(getattr(d, "RescaleIntercept", 0.0))
        if getattr(d, "PhotometricInterpretation", "") == "MONOCHROME1":
            a = a.max() - a                     # MONOCHROME1 is inverted
        slices.append(a)

    return np.stack(slices, axis=-1), chosen[0]
```

This implementation handles several conditions that naive code does not: multiple series in a single folder, unreliable or absent `InstanceNumber`, missing rescale tags, MONOCHROME1 inversion, and files without a `.dcm` extension (DICOMDIR exports frequently have no extension). Install `pylibjpeg`, `pylibjpeg-libjpeg`, and `pylibjpeg-openjpeg` — or `python-gdcm` — or compressed studies will continue to fail.

### Intensity windowing policy

For single DICOM images, prefer the file's own metadata: `apply_voi_lut(ds.pixel_array, ds)` uses `WindowCenter` and `WindowWidth` as set by the originating radiologist. When these are absent, or for NIfTI (which carries no window tags), fall back to modality presets:

| Target tissue | Window Center | Window Width | Applicable to |
|---|---|---|---|
| Brain | 40 | 80 | Head CT |
| Lung | −600 | 1500 | Chest CT, nodule assessment |
| Mediastinum / soft tissue | 40 | 400 | Chest and abdomen CT |
| Abdomen | 60 | 400 | Liver, bowel |
| Bone | 400 | 2000 | Fracture, lytic lesion |
| MRI / arbitrary float | — | — | Use 0.5–99.5 percentile stretch (**already implemented in Stage 0**) |

**Question-conditioned windowing is a substantial, low-cost improvement.** Stage 3 already classifies question intent; extend it with an anatomy keyword pass to select the window:

```python
WINDOW_KEYWORDS = {
    ("lung", "pulmonary", "nodule", "pneumothorax", "emphysema"): (-600, 1500),
    ("brain", "cerebral", "hemorrhage", "stroke", "ventricle"):   (40, 80),
    ("bone", "fracture", "vertebra", "cortical", "lytic"):        (400, 2000),
    ("liver", "kidney", "spleen", "abdomen", "bowel"):            (60, 400),
}
```

A lung nodule question rendered on a soft-tissue window is close to unanswerable. Identical voxel data, entirely different visibility.

Additionally, for non-CT volumes, **window per volume rather than per slice**. Per-slice normalization gives slice *N* and slice *N+1* different brightness semantics, which destroys cross-slice comparability in multi-slice tiling.

### Slice-selection policies

| Policy | Implementation | Advantages | Disadvantages | Recommended for |
|---|---|---|---|---|
| **Middle slice** | `vol[:, :, D // 2]` | One image, fastest, trivial | Misses off-center findings; incorrect when the volume is not centered on the anatomy | Baseline / fallback |
| **Maximum foreground** | Select the slice maximizing non-background area or entropy | Robust to padding and air slices; essentially free | May select the widest slice rather than the diagnostic one | Better default than middle |
| **Maximum intensity projection (MIP)** | `vol.max(axis=2)` over a slab | Excellent for vasculature, contrast studies, bright nodules | Destroys soft-tissue contrast; inappropriate for brain MRI | Angiography, lung nodule search |
| **Multi-slice tiling** | 3–9 evenly spaced slices in a grid, or as separate images | **Highest accuracy**; restores 3D context | 3–9× prefill cost | **Recommended default** |
| **Question-guided slab** | Center the slab on the anatomy implied by the question | Highest accuracy achievable | Requires anatomy detection | Later refinement |

**Recommended default: three slices at 35%, 50%, and 65% of depth**, either composited into a single 1×3 letterboxed image (one prefill, cheaper) or passed as three separate images if the processor supports multi-image input (better accuracy, ~3× prefill).

**Important interaction:** the `max_pixels` cap applies to the *composited* image. Tiling three slices into one image that is then downsampled by the cap leaves each slice at one-third resolution, which can make results worse than a single full-resolution slice. Either raise `max_pixels` for tiled inputs or pass slices as separate images.

### Aspect ratio handling

CT and MR voxels are typically anisotropic (for example, 0.7 × 0.7 × 5.0 mm). Coronal or sagittal reformats must be rescaled by the `zooms` ratio or the anatomy will be geometrically distorted. Axial slices are usually isotropic in plane and safe. Prefer axial unless the question specifically requires another orientation.

### Preserving the zero-crash guarantee

Wrap Stage −1 in the same defensive philosophy, but make degradation **graded rather than binary**:

| Condition | Response |
|---|---|
| Volumetric read succeeds | Tiled multi-slice input (best) |
| Read succeeds, windowing fails | Percentile stretch (good) |
| Series sorting fails | Middle file, single slice (acceptable) |
| Total failure | Gray canvas + question-only prompt (**still preferable to a constant "A"**) |

The final row deserves emphasis. Even on complete image failure, submitting the question with a blank canvas and allowing the model to apply language priors outperforms a hardcoded letter. Published work shows models retain a large share of accuracy from text alone; on 4-choice items a language-prior guess is worth roughly 35–45% versus 25% for a fixed letter, and on 2-choice yes/no items it comfortably beats always-"A". **Change the fallback from a constant to a blind model call.** This is inexpensive and can be implemented immediately.

### Expected impact

If volumetric items constitute *V*% of the evaluation set and currently score at or below chance, moving them to the pipeline's average accuracy yields approximately `V × (avg_acc − current_fallback_acc)`. For *V* = 20%, average accuracy 62%, and fallback accuracy 25%, this is **+7.4 points**. No other single item in this document offers comparable return for the engineering effort.

---

## Area 6 — Benchmarking Framework and Dataset Mapping

### Capability-to-benchmark mapping

| Capability under test | Primary dataset | Secondary | Notes |
|---|---|---|---|
| Radiology, closed-form | **VQA-RAD** (clean splits) | SLAKE closed | Small (n=315 images) — bootstrap CIs mandatory |
| Radiology, spatial / anatomy | **SLAKE** | — | Includes knowledge graph and segmentation masks; ideal for the Spatial track |
| Pathology / microscopy | **PathVQA** | Quilt-VQA | Yes/no heavy — report open and closed subsets separately |
| Cross-modality breadth | **OmniMedVQA** | GMAI-MMBench | 12 modalities, MCQ format — matches your output format |
| Dermatology / fundus / OCT / ultrasound | **OmniMedVQA subsets** | ISIC, ODIR, BUSI (converted to MCQ) | Covers the Macroscopic and Ultrasound streams |
| Chest X-ray reasoning depth | **ReXVQA** | MIMIC-CXR-VQA | Large; five distinct reasoning skill categories |
| Volumetric / 3D | **M3D-VQA** | RadImageNet-VQA | Use to validate the new Stage −1 |
| Hard clinical reasoning | MedXpertQA-MM, MMMU Health & Medicine | MedFrameQA | Expect low scores — GPT-4o reaches only ~46% on MedFrameQA |

### Restructuring the evaluation script

The current `eval.py` performs its job but is a scoring script rather than an experiment harness. Refactor around three principles.

#### 1. Every behavior is configuration, not a constant

```python
from dataclasses import dataclass

@dataclass
class PipelineConfig:
    backbone: str = "Qwen3-VL-4B-Instruct"
    use_modality_router: bool = True
    use_stage2_preprocessing: bool = True
    use_intent_router: bool = True
    prompt_style: str = "persona"          # persona | generic | cot | fewshot
    volumetric_policy: str = "tri_slice"   # middle | mip | tri_slice | none
    calibrate_options: bool = True
    option_permutations: int = 1           # 1 = off, 4 = cyclic debiasing
    fallback_mode: str = "blind_model"     # constant_A | blind_model
    seed: int = 0
```

Every claim in this document then becomes a single line in a sweep file, and every result is reproducible.

#### 2. Log rich per-item records

Keep `predictions.csv` unchanged for submission compatibility and write a separate diagnostic sidecar:

```csv
query_id,answer,gold,correct,inference_time,
predicted_modality,predicted_intent,n_choices,
image_kind,slice_policy,fallback_triggered,fallback_reason,
top1_logit,top2_logit,logit_margin,stage_times_json
```

`fallback_triggered` and `logit_margin` are the two most valuable fields currently missing.

#### 3. Stratify every reported number

Report accuracy sliced by modality × intent × number of choices × image kind × fallback status. Aggregate accuracy conceals which subsystem is failing. Learning that the pipeline scores 71% on 2-choice items and 38% on 4-choice items identifies a calibration problem, not a vision problem — and those require entirely different fixes.

### Required statistical treatment

Development sets in this domain are small, which makes naive comparison unreliable.

- **Bootstrap confidence intervals** (10,000 resamples) on every headline number. At n = 500, ±4 points is noise. Many reported improvements at this scale are not real.
- **McNemar's test** for paired pipeline-versus-baseline comparisons. Because both systems see identical items, paired testing is substantially more powerful than comparing independent accuracies.
- **Seed variance.** Even with argmax decoding, cuDNN kernel selection and batching introduce nondeterminism. Run three seeds and report mean ± standard deviation.

### Mandatory baselines

1. **Random** (25% / 50%) — the floor.
2. **Always-"A"** — quantifies the current fallback's value and exposes gold-answer position bias in the dataset.
3. **Blind model** (question only, blank image) — **the most important currently-missing baseline.** Everything the pipeline gains above this figure represents the value of actually processing the image.
4. **Naked backbone** (raw image, generic prompt, Stages 1–4 disabled) — quantifies the value of the entire preprocessing and routing architecture.
5. **Published state of the art** — for context, not as a competitive target.

If baseline 4 matches the full pipeline, Stages 1–4 should be removed in favor of a simpler and faster system. That is a legitimate and valuable result, not a failure.

---

## Area 7 — Prompt Construction and Clinical Persona

### Realistic assessment of persona framing

Persona prompting is the weakest lever discussed in this document. The published evidence is unflattering: role framing such as "You are an expert radiologist" produces **small and inconsistent** effects on accuracy for modern instruction-tuned models, and effects observed on one model frequently vanish or reverse on another. The 20 cached persona × modality permutations represent elegant engineering built on a weakly supported premise.

Elements that **do** reliably help:

1. **Modality grounding stated as fact.** "This is an axial contrast-enhanced CT of the abdomen." This is not persona framing — it is information, and it removes an inference the model would otherwise have to make.
2. **Task format clarity.** Exactly one correct option; output format explicitly specified. The current implementation already does this well.
3. **Removing permission to hedge.** For forced-choice MCQ, do **not** include phrasing such as "if unsure, say so" — it shifts probability mass toward hedging tokens and away from the choice letters.
4. **Presence of option text.** Because Stage 6 argmaxes over letter tokens, the option *text* must be present in context. `A. Pneumothorax` scores very differently from a bare `A.`

Likely noise: "Attending Physician" versus "Radiologist" versus "Board-certified specialist." Test rather than assume. The relevant ablation configuration is `prompt_style ∈ {persona, generic, information_only}`.

### The high-value opportunity: calibration and debiasing

Because the pipeline computes explicit logits over candidate letters, two well-established improvements become almost free — an option unavailable to pipelines that parse generated text.

#### (a) Null-prompt (content-free) calibration

Models carry a strong intrinsic prior over the tokens "A", "B", "C", and "D" that is unrelated to the image. Measure it once and divide it out.

```python
# Once, at startup: model preference with no real content
null_logits = model_choice_logits(
    image=GRAY_CANVAS, question="N/A", options=["A", "B", "C", "D"]
)
null_prior = softmax(null_logits)          # e.g. [0.41, 0.22, 0.21, 0.16] — note the "A" bias

# At inference:
p = softmax(real_logits)
p_calibrated = p / null_prior
answer = LETTERS[p_calibrated.argmax()]
```

This is the standard "calibrate before use" correction. On multiple-choice tasks it commonly yields **+1 to +4 points** and costs one additional forward pass **in total**, not per item. There is no reason not to implement it.

#### (b) Cyclic option-order ensembling

Models are measurably sensitive to the position of the correct answer. Evaluate each item multiple times with options cyclically rotated, map each result back to the original option identity, and aggregate.

```python
scores = np.zeros(n)
for shift in range(n_perm):                 # n_perm = 4
    opts = rotate(options, shift)
    p = calibrated_probs(image, question, opts)
    scores += unrotate(p, shift)            # map back to original indices
answer = LETTERS[scores.argmax()]
```

**Cost:** *n* forward passes, which conflicts with the 5.0-second timeout guard. Two mitigations: the image prefill dominates total cost, so either cache the image KV state and re-run only the short text tail, or execute the four permutations as a single batch of four. Batched, wall-clock cost is typically **1.3–1.8×** rather than 4×.

**Expected gain: +1 to +3 points**, with materially improved robustness.

**Recommendation:** implement (a) unconditionally. Implement (b) if the latency budget permits, gated on `logit_margin` so the additional cost is incurred only for the roughly 30% of items where the model is genuinely uncertain.

### Chain-of-thought: effective but architecturally disruptive

Chain-of-thought reasoning genuinely helps on multi-step clinical questions, and reasoning-supervised medical models score meaningfully higher on difficult sets. However, **Stage 6 is a single forward pass over candidate tokens, which provides no place for reasoning to occur.**

Adopting CoT requires a two-pass design:

```
Pass 1: generate ≤96 reasoning tokens (greedy, temperature 0)
Pass 2: append reasoning + "Therefore, the answer is", then argmax over letter tokens
```

**Cost:** approximately 96 sequential decode steps that do not currently exist. At 25–40 tokens/second for a 4B model on a single card, this adds **2.5–4 seconds per item**. The 5.0-second daemon timeout would fire frequently. Either raise the timeout to approximately 15 seconds or bound reasoning to roughly 48 tokens.

**Benefit:** typically **+2 to +6 points** on Differential and Severity tracks; often **neutral or slightly negative** on Modality-ID and simple Spatial items, where forced reasoning can cause the model to talk itself out of a correct initial judgment.

**Therefore, route CoT by intent track.** Apply CoT to Differential, Severity, and Diagnostic questions; retain single-pass decoding for Modality-ID and Spatial. This is the strongest justification for retaining the Stage 3 intent classifier — it converts a decorative routing decision into a compute-allocation decision with measurable value.

### Few-shot in-context examples

- **Image-based few-shot is expensive.** Each exemplar image adds a full vision prefill; three exemplars cost roughly 4× the baseline prefill. Rarely justified for a strong instruction-tuned model.
- **Text-only few-shot is inexpensive but marginal** — two or three worked question-to-letter examples without images, purely to lock output format. Since decoding is already constrained, format is already enforced, so the gain is minimal.
- **Recommendation:** skip few-shot and allocate the equivalent compute to option-order ensembling, which delivers better accuracy per FLOP.

### Recommended prompt skeleton

```
[MODALITY FACT]   This is a {subtype} image ({stream}).
[TASK]            Answer the multiple-choice question about this image.
[QUESTION]        {question}
[OPTIONS]         A. {a}
                  B. {b}
                  C. {c}
                  D. {d}
[CONSTRAINT]      Exactly one option is correct.
                  Respond with ONLY the letter of the correct answer.
```

Ship this as the `generic` arm of the ablation. If the 20-permutation persona system cannot beat it in a paired McNemar test, remove the templates and accept the simpler codebase.

---

## Area 8 — Evaluation Metrics and State-of-the-Art Baselines

### Applicable metrics

The task is **closed-set multiple choice**, which narrows the metric question considerably.

- **Exact-Match Accuracy is the metric.**
- **BLEU, ROUGE, and CIDEr are not applicable.** These exist for open-ended generative medical VQA and are widely criticized even there, since they reward string overlap — "no pneumothorax" and "pneumothorax" score nearly identically despite opposite clinical meaning.
- **Token-level F1** becomes relevant only if open-ended output is added later. This is the metric MedGemma reports for SLAKE and VQA-RAD.

### Supplementary diagnostic metrics

| Metric | Definition | Purpose |
|---|---|---|
| Overall EM accuracy | correct / total | Headline figure |
| **Per-stratum accuracy** | Sliced by modality, intent, choice count | Localizes the failing subsystem |
| **Fallback rate** | fallbacks / total | Currently the largest hidden source of loss |
| **Blind accuracy** | Same evaluation with blank images | Detects language shortcutting |
| **Visual Reliance Score** | (acc_real − acc_blind) / (acc_real − chance) | Confirms the model is using the image |
| **Choice-position bias** | Distribution of predicted letters versus gold | Detects collapse toward a single letter |
| **Expected Calibration Error** | ECE over logit softmax | Required for abstention or ensembling |
| **p50 / p95 latency** | Per item and per stage | Timeout budgeting |
| **Timeout rate** | Items hitting the 5.0s daemon guard | Silent accuracy loss |

### Published performance context

Use these figures to set realistic targets.

**OmniMedVQA (multiple-choice, 12 modalities):**

| Model | Approximate accuracy |
|---|---|
| Med-Flamingo (8.3B) | ~35% |
| LLaVA-Med (7B) | ~41% |
| HuatuoGPT-Vision (7B) | ~50% |
| LLaVA-v1.6-34B | ~59% |
| Qwen2.5-VL-7B | ~61% |
| Qwen2-VL-72B (zero-shot) | ~68% |
| HealthGPT | ~68% |
| Strong reasoning / agentic systems | ~73–78% |
| BiomedCLIP + LLaMA-3-8B (efficient design) | ~73% open-ended / ~77% yes-no |

**VQA-RAD / SLAKE / PathVQA:**

- MedGemma-4B shows large gains over its own base model (SLAKE token-F1 ~72 versus ~40; VQA-RAD ~50 versus ~34) — the cleanest available same-size evidence for medical pre-training.
- Supervised fine-tuned specialists on closed subsets: VQA-RAD ~85%, SLAKE ~92%, PathVQA ~95%. These are in-domain SFT ceilings, not zero-shot targets.
- GPT-4o on VQA-RAD: approximately 70%, as reported in comparison studies.
- On harder multi-image reasoning (MedFrameQA), **all evaluated models score below 55%**; GPT-4o reaches ~46% and Qwen2.5-VL-72B ~43%.

### Realistic targets for this pipeline

Assuming a mixed-modality, predominantly 4-choice development set containing some volumetric items:

| Milestone | Target range | Enabling work |
|---|---|---|
| Random floor | 25% | — |
| Current, with blind fallbacks | **Measure this first** | Instrumentation |
| Naked Qwen3-VL-4B, generic prompt | ~55–62% | Likely near current state |
| + volumetric ingest + blind-model fallback | ~60–68% | Area 5 |
| + calibration + option debiasing | ~62–70% | Area 7 |
| + MedGemma-4B or HuatuoGPT-Vision-7B backbone | ~66–74% | Area 3 |
| + LoRA fine-tuning on mixed medical MCQ | ~70–78% | Area 4 |
| Practical ceiling without substantially larger model | ~78–82% | Diminishing returns |

**Important caveat:** every figure above depends on split, prompt, and scoring protocol. Published medical VQA results are notoriously non-comparable across papers due to contaminated splits, prefix-scoring versus generation-scoring differences, and inconsistent subset selection. Treat these numbers as directional only. Your own held-out development set should govern all decisions.

---

## Area 9 — Data Augmentation in Medical VQA

### Governing principle

**Augmentation applies to training, not inference.** If fine-tuning has not begun, image augmentation is not yet relevant. The only inference-time augmentation worth adopting is **option-order permutation** (Area 7), which is text-side, clinically meaningless, and therefore entirely safe.

### Clinical validity by transform and modality

| Transform | Radiology (CXR/CT/MR) | Histopathology | Dermoscopy | Fundus | Ultrasound |
|---|---|---|---|---|---|
| Rotation ±10–15° | Safe | **Safe at any angle** | Safe at any angle | Safe | Mild only |
| Rotation 90° / 180° | **Unsafe** — breaks anatomy | Safe | Safe | **Unsafe** — breaks disc/macula geometry | **Unsafe** — breaks probe geometry |
| **Horizontal flip** | **High risk — see below** | Safe | Safe | **High risk** | Risky |
| Vertical flip | Unsafe | Safe | Safe | Unsafe | Unsafe |
| Brightness / contrast jitter | Mild only | Safe | Safe | Safe | Safe |
| HU or window jitter | Window jitter acceptable; HU shift **unsafe** | n/a | n/a | n/a | n/a |
| Stain augmentation (HED / Macenko jitter) | n/a | **Strongly recommended** | Not applicable | Not applicable | n/a |
| Gaussian noise | Mild, safe | Safe | Safe | Safe | Safe (speckle-like) |
| Elastic deformation | Mild only | Safe | Safe | Risky | Risky |
| Random crop | May remove the finding | Safe (tiles) | Risky | Unsafe — may remove optic disc | Risky |
| CutMix / MixUp | **Unsafe** — fabricates anatomy | Marginal | Marginal | Unsafe | Unsafe |

### Why horizontal flip is the canonical failure

1. **Laterality is diagnostic.** For a question such as "Which lung shows the effusion?", flipping the image invalidates the gold label. The result is a mislabeled training example that teaches the model left and right are interchangeable, corrupting the entire Spatial track.
2. **Human anatomy is not left-right symmetric.** The cardiac silhouette sits left, the aortic knob is left-sided, the gastric bubble sits under the left hemidiaphragm, and the liver is right-sided. A flipped chest radiograph depicts **situs inversus** — a genuine but rare condition. Flipping teaches the model that approximately 1-in-10,000 anatomy is normal.
3. **Burned-in markers.** Radiographs carry "L" and "R" lead markers and text overlays. Flipping produces mirror-writing, which a strong-OCR backbone such as Qwen3-VL will read and be misled by.
4. **Fundus laterality.** In a right eye (OD), the optic disc sits nasally. Flipping converts an OD image into something resembling an OS image while the label continues to assert OD.
5. **Histopathology is the exception.** Tissue on a slide has no canonical orientation, so flips, arbitrary rotations, and transposes are all clinically valid. This is why pathology models routinely use full dihedral-8 augmentation.

**Safe-flip policy:** gate augmentation on modality. Note the failure mode, however — if the Stage 1 router is wrong, a chest radiograph may be flipped under the belief that it is a histology tile. **Use ground-truth modality labels for training augmentation, never predicted ones.**

### Highest-value augmentations, ranked

1. **Option-order shuffling.** Free, carries no clinical risk, and directly addresses the primary failure mode of MCQ fine-tuning. Apply every epoch.
2. **Question paraphrasing.** Generate two to three LLM paraphrases per question. Prevents memorization of template surface forms and improves intent-routing robustness. No image-side risk.
3. **Stain augmentation for histology.** HED color jitter or Macenko parameter perturbation. Well established, with large gains in cross-scanner generalization. Note that this partially substitutes for the Stage 2 Macenko normalization — you can normalize or augment, and augmenting generally generalizes better.
4. **Window jitter for CT.** Perturb window center and width by ±10% rather than modifying HU values. Teaches window robustness without misrepresenting tissue density.
5. **Mild geometric transforms** (±10° rotation, ±5% scale and translation) for all modalities except ultrasound and fundus.
6. **Distractor resampling.** For MCQ items, regenerate distractors from a modality-matched answer pool each epoch to prevent memorization of specific option sets.

### The one absolute constraint

**Never apply an augmentation that changes the correct answer.** This is frequently violated in practice. Before adding any transform, ask whether any question in the training set has a gold answer the transform would invalidate. For flips, the required filter is "exclude every question containing left, right, lateral, medial, ipsilateral, or contralateral" — and once that filter is applied, so much data is lost that the flip was not worthwhile.

---

## Implementation Roadmap

### Phase 1 — Instrumentation and immediate wins

- [x] Add `fallback_triggered`, `fallback_reason`, `image_kind`, and `logit_margin` to a diagnostic sidecar log. Establish the true fallback rate. — **DONE**, `evaluate_omnimed.py` diagnostic sidecar CSV
- [x] Replace the constant `"A"` fallback with a **blind model call** (gray canvas + question). Immediate, low-effort gain. — **DONE in `predict_with_diagnostics()`**; the competition `predict()` entrypoint intentionally kept the constant fallback (see Status Update above) — revisit before submission if time allows
- [ ] Build the ablation harness: `PipelineConfig` dataclass, sweep runner, bootstrap confidence intervals, McNemar testing. — **PARTIAL**: bootstrap CIs and stratified logging exist; no `PipelineConfig`/sweep runner/McNemar
- [ ] Run all five baselines: random, always-"A", blind model, naked backbone, full pipeline. — **NOT DONE**
- [ ] Implement null-prompt calibration (one additional forward pass, total). — **NOT DONE**

### Phase 2 — Volumetric support

- [x] Implement Stage −1 volume ingest: magic-byte dispatch, `nibabel` with `as_closest_canonical`, `pydicom` with series splitting, geometric sorting, rescale application, and MONOCHROME1 handling. — **DONE**, `src/volume_loader.py`
- [ ] Install `pylibjpeg` / `gdcm` for compressed transfer syntaxes. — **NOT DONE**; not added to `requirements.txt` this session, worth a quick check against the real dev-data DICOM series before submission in case any use a compressed transfer syntax
- [ ] Implement and ablate all four slice policies: `middle`, `max_foreground`, `mip`, `tri_slice`. — **PARTIAL**: shipped a single Vote-MI-inspired content-based policy (variance + edge density) with graded fallback to fixed-percentage then single-slice, not a formal ablation across all four named policies
- [ ] Add question-conditioned CT windowing driven by anatomy keywords. — **NOT DONE**
- [ ] Raise the 5.0-second timeout to accommodate tiled prefill; re-measure p95 latency. — **STATUS UNCLEAR THIS SESSION**: observed real inference times of ~20–23s on volumetric/DICOM inputs (per user report) — well above a nominal 5.0s guard. Worth confirming the actual configured timeout before submission; if it's still 5.0s, volumetric items may be timing out silently

### Phase 3 — Backbone selection and architecture pruning

- [ ] Run the backbone bake-off: Qwen3-VL-4B (control), MedGemma-4B-it, HuatuoGPT-Vision-7B, Qwen2.5-VL-7B. — **NOT DONE — explicitly out of scope given the deadline**
- [ ] Report accuracy, VRAM, and p95 latency; select from the Pareto front. — **NOT DONE**
- [ ] Ablate Stages 1–4 against the generic prompt. Remove any component that does not demonstrate measurable value. — **NOT DONE**
- [ ] Add margin-gated option-order ensembling for low-confidence items. — **NOT DONE**

### Phase 4 — Domain adaptation

- [x] Assemble the MCQ-formatted training mix, deduplicated against every evaluation set. — **DONE, different composition than originally recommended**: OmniMedVQA subsets (11 names/groups) + 3 external volumetric/near-volumetric datasets (MosMedData, BrainMRI, BUSI) converted via `src/prepare_external_dataset.py`, chosen to cover all 8 PS-named modalities on both eval and fine-tuning sides — not the PMC-VQA/PathVQA/SLAKE/VQA-RAD union originally proposed (those remain unintegrated, see Area 4). Splits are case-level deterministic; **not** independently verified by image hash against the eval sets. Full table in `datasets.md`
- [x] Train LoRA at rank 16 [...], loss computed on the answer token only. — **PARTIAL**: `src/train_lora.py` does QLoRA (4-bit) + answer-letter-only loss masking, verified with a real training run. Vision-tower freeze, 15–20% general-domain replay, and per-epoch option shuffling — **not implemented**; the LoRA rank/target-module regularization is currently the only catastrophic-forgetting guard in place
- [ ] Track `visual_reliance` alongside accuracy at every checkpoint. Reject any checkpoint where accuracy increases while reliance decreases. — **NOT DONE**
- [ ] Enable intent-routed chain-of-thought for the Differential and Severity tracks only. — **NOT DONE**

---

## Closing Assessment

Three conclusions summarize the audit.

**1. The pipeline is unmeasured.** Six stages of preprocessing and routing logic exist without a single ablation. Building the evaluation harness produces no direct accuracy gain, but it converts every other recommendation in this document from a hypothesis into a measurement. It is the highest-leverage engineering work available, precisely because it is unglamorous.

**2. Silent failure is more costly than loud failure.** The zero-crash design is correct engineering, but a graceful `"A"` is indistinguishable from a genuine prediction on a scoreboard. Every fallback must be counted, categorized, and degraded incrementally — tiled slices, then single slice, then blind model, and only then a constant. The pipeline should never jump directly to the constant.

**3. Effort should follow the derivative.** Backbone selection and volumetric ingest dominate the achievable gains. Persona wording and heuristic router tuning are second-order effects. Resolve the volumetric gap and establish measurement before investing further in the sophisticated components.

**Priority order: close the volumetric gap, establish measurement, then optimize.**
