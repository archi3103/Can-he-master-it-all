# Dataset Selection: Evaluation & Fine-Tuning

**Status**: decisions finalized this session; fine-tuning run itself not yet executed (see `medical_vqa_architecture.md` Section 3.2 / 5.7 for the tooling, `kaggle_finetune_lora.ipynb` for the run). This document is the single source of truth for *which* datasets are used *where* — cross-reference it instead of re-deriving the list from chat history.

**Guiding constraint**: the problem statement (`Can He Master It All.pdf`) names exactly 8 imaging modalities the evaluation grid may hand the system: cross-sectional CT, MRI, projection radiographs (X-ray), retinal photographs (fundus), dermatological surface images (dermoscopy), microscopic cellular/tissue images (histopathology), cross-sectional optical scans (OCT), and acoustic/sonographic images (ultrasound). Every selection below is driven by covering all 8, on both the evaluation and fine-tuning sides, without evaluation/fine-tuning image overlap.

---

## 1. Evaluation set (held out, never trained on)

| Modality | Dataset | Source | Format | Volumetric? | Ground truth | Notes |
|---|---|---|---|---|---|---|
| Fundus | Diabetic Retinopathy | OmniMedVQA (Open-access) | 2D PNG | No | MCQ, native | |
| X-ray | Chest X-Ray PA | OmniMedVQA (Open-access) | 2D PNG | No | MCQ, native | |
| Dermoscopy | ISIC2019 | OmniMedVQA (Open-access) | 2D PNG | No | MCQ, native | |
| CT | Chest CT Scan | OmniMedVQA (Open-access) | 2D PNG | **No** | MCQ, native | Despite the modality, OmniMedVQA's CT/MRI entries are pre-sliced 2D images cut from 3D volumes (confirmed in the dataset's own README) — this does NOT exercise Stage -1's volumetric ingest code at all |
| Histopathology/microscopy | CRC100k | OmniMedVQA (Open-access) | 2D PNG | No | MCQ, native | Colorectal tissue classification — chosen over BreakHis/others for eval specifically because it's tissue-architecture-focused, matching the PS's "cellular and tissue images" wording most literally |
| OCT | Retinal OCT-C8 | OmniMedVQA (Open-access) | 2D PNG | No | MCQ, native | Chosen over "OCT & X-Ray 2017" for eval to keep the eval set unambiguously single-modality per entry |
| CT (volumetric) | **MosMedData_eval** | External (converted) | NIfTI (`.nii`/`.nii.gz`) | **Yes** | 4-choice MCQ, derived from 5-class severity label | ~20% case-level split of 1110 real chest CT studies; see Section 3 |
| MRI (volumetric) | **BrainMRI_eval** | External (converted) | NIfTI, 4D (4 sequences/case) | **Yes** | 4-choice MCQ, anatomy-ID | ~20% case-level split of Medical Segmentation Decathlon Task01_BrainTumour (~484 cases); see Section 3 |
| Ultrasound | **BUSI_eval** | External (converted) | 2D PNG | No | 3-choice MCQ, native (benign/malignant/normal) | ~20% case-level split of 780 real breast ultrasound images; see Section 3 |

**Not part of the scored evaluation set** — the competition-provided development data (`development data/`, samples 0004-0007, 0010): real organizer-sourced CT DICOM series, MRI NIfTI, and an ultrasound image, but `dev_metadata.csv` carries **no ground-truth column at all** (confirmed by inspecting the file directly). Useful only as a final "does the pipeline run without crashing, what's the inference_time" check on the exact format the real harness will hand you — not usable for an accuracy number.

## 2. Fine-tuning set (disjoint from the evaluation set above)

| Modality | Dataset(s) | Source | Format | Volumetric? | Ground truth | Why this one |
|---|---|---|---|---|---|---|
| Fundus | ACRIMA, PALM2019 | OmniMedVQA (Open-access) | 2D PNG | No | MCQ, native | Glaucoma + pathologic myopia — disease diversity away from the eval set's diabetic-retinopathy focus |
| X-ray | COVIDx CXR-4, Mura | OmniMedVQA (Open-access) | 2D PNG | No | MCQ, native | Chest pathology + musculoskeletal — different body regions, not just more chest data |
| Dermoscopy | ISIC2018, Fitzpatrick 17k | OmniMedVQA (Open-access) | 2D PNG | No | MCQ, native | Classic lesion labels + skin-tone/condition diversity beyond lesions |
| CT | Covid CT | OmniMedVQA (Open-access) | 2D PNG | No | MCQ, native | The other 3 COVID-CT-named OmniMedVQA sets (tianchi/heywhale/SARS-CoV-2) are near-duplicate sources of the same content — one is enough |
| Histopathology/microscopy | BreakHis, ALL Challenge, NLM- Malaria Data | OmniMedVQA (Open-access) | 2D PNG | No | MCQ, native | Tissue histopathology + leukemia cytology + parasitology — three genuinely distinct visual subdomains; likely the weakest/most novel stream, given the most training diversity |
| OCT | OCT & X-Ray 2017 | OmniMedVQA (Open-access) | 2D PNG | No | MCQ, native | Dual-purpose: reinforces both OCT and X-ray training data from one source |
| CT (volumetric) | **MosMedData_train** | External (converted) | NIfTI | **Yes** | 4-choice MCQ, derived severity | ~80% case-level split, same source as the eval entry, disjoint cases |
| MRI (volumetric) | **BrainMRI_train** | External (converted) | NIfTI, 4D | **Yes** | 4-choice MCQ, anatomy-ID | ~80% case-level split, same source as the eval entry, disjoint cases |
| Ultrasound | **BUSI_train** | External (converted) | 2D PNG | No | 3-choice MCQ, native | ~80% case-level split, same source as the eval entry, disjoint cases |

**PS modality coverage — both sides now complete:**

| PS-named modality | Evaluation | Fine-tuning |
|---|---|---|
| CT | Chest CT Scan (2D) + MosMedData_eval (volumetric) | Covid CT (2D) + MosMedData_train (volumetric) |
| MRI | BrainMRI_eval (volumetric) | BrainMRI_train (volumetric) |
| X-ray | Chest X-Ray PA | COVIDx CXR-4, Mura |
| Fundus | Diabetic Retinopathy | ACRIMA, PALM2019 |
| Dermoscopy | ISIC2019 | ISIC2018, Fitzpatrick 17k |
| Histopathology/microscopy | CRC100k | BreakHis, ALL Challenge, NLM-Malaria Data |
| OCT | Retinal OCT-C8 | OCT & X-Ray 2017 |
| Ultrasound | BUSI_eval | BUSI_train |

## 3. External volumetric datasets — detail

Added specifically because OmniMedVQA's CT/MRI entries are pre-sliced 2D and cannot exercise or validate Stage -1 (`src/volume_loader.py`) at all. Converted into the OmniMedVQA local-mirror JSON schema by `src/prepare_external_dataset.py` (verified end-to-end this session, including a real `predict_with_diagnostics()` call on converted items with zero fallback triggered) — `train_lora.py`/`evaluate_omnimed.py` need no code changes to read them.

| Dataset | Raw format & size | Question design | Split |
|---|---|---|---|
| **MosMedData** | Real NIfTI chest CT, 1110 studies, organized into `CT-0`..`CT-4` severity folders (Moscow municipal hospitals, COVID-era) | "What is the severity of lung involvement shown in this chest CT scan?" — 4 choices (none/mild/moderate/severe), ground truth mapped from the CT-0..CT-4 folder name (CT-3/CT-4 collapsed into "severe" since CT-4 has only 2 cases in the source data) | Deterministic case-level 80/20 train/eval, seeded |
| **BrainMRI** (Medical Segmentation Decathlon Task01_BrainTumour) | Real 4D NIfTI brain MRI, ~484 cases, 4 sequences per case (FLAIR/T1w/T1gd/T2w) in one file — `src/volume_loader.py`'s `_load_nifti()` already reduces any 4D NIfTI to its first channel (FLAIR) automatically, so no manual sequence-splitting was needed | "Which anatomical region is primarily depicted in this MRI scan?" — 4 choices (brain/chest/abdomen/pelvis), ground truth always "brain" (every case in this dataset is a brain scan) | Deterministic case-level 80/20 train/eval, seeded |
| **BUSI** | Real breast ultrasound PNGs, 780 images, `benign`/`malignant`/`normal` folders, paired segmentation masks (excluded by filename match on "mask") | "What does this breast ultrasound image most likely show?" — 3 choices (benign/malignant/normal), ground truth from the folder name | Deterministic case-level 80/20 train/eval, seeded |

**Known limitation, accepted given the deadline**: BrainMRI's question is anatomy-ID only (constant answer within the dataset) since Task01_BrainTumour ships segmentation masks, not diagnosis labels — deriving a genuine tumor-presence/grading question would need mask-centroid logic that wasn't built this pass. Still real, correctly-labeled MRI signal; just narrower in question variety than the other two external sets.

**To reproduce the conversion** (run once per dataset, after downloading each raw archive):
```bash
python src/prepare_external_dataset.py --dataset mosmed    --raw-root <extracted MosMedData>          --output-root <OmniMedVQA data-root>
python src/prepare_external_dataset.py --dataset brainmri  --raw-root <extracted Task01_BrainTumour>   --output-root <OmniMedVQA data-root>
python src/prepare_external_dataset.py --dataset busi      --raw-root <extracted Dataset_BUSI_with_GT> --output-root <OmniMedVQA data-root>
python src/evaluate_omnimed.py --list-datasets --data-root <OmniMedVQA data-root>  # should now show all 6 new names
```

## 4. Investigated and rejected

| Dataset | Why rejected |
|---|---|
| **M3D-VQA / M3D-Cap** | Purpose-built 3D medical VQA (real MCQ schema), which is exactly what this task needs on paper — but verified via the actual HF dataset pages: the image data (M3D-Cap) is ~978GB, CT-only, stored as pre-normalized `.npy` arrays (not NIfTI/DICOM, so it would bypass Stage -1's real ingest code entirely even if downloaded), and is currently access-disabled by a DMCA takedown notice regardless of the other issues. Don't re-investigate this one. |
| VQA-RAD / SLAKE / PathVQA / PMC-VQA | Not integrated this pass — `train_lora.py`/`evaluate_omnimed.py` only read the OmniMedVQA-shaped local mirror; pulling these in would need a new data loader written and tested under deadline pressure, for uncertain payoff given the OmniMedVQA-based plan already reaches full 8-modality coverage on both eval and fine-tuning sides. Legitimate future work, not required for this submission. |
| RadImageNet (mixed CT/MRI/ultrasound) | Considered as a way to get more MRI/ultrasound OmniMedVQA-native coverage by filtering its `modality_type` field, but `load_omnimed_samples()` only filters by dataset name today, not by per-item modality — would need new filtering code. Superseded by the external volumetric datasets in Section 3, which give real volumetric MRI/ultrasound coverage anyway (RadImageNet's images are 2D). |

## 5. Sizing / sanity numbers

- MosMedData: 1110 cases total → ~888 train / ~222 eval.
- BrainMRI (Task01_BrainTumour): ~484 cases total → ~387 train / ~97 eval.
- BUSI: 780 images total → ~624 train / ~156 eval.
- All three splits are large enough for `evaluate_omnimed.py`'s bootstrap CI (10,000 resamples) to be meaningful, unlike the 4-sample dev-data volumetric set would have been.
