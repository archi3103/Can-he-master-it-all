"""
Stage 4: System prompt composition (Section 2.2 / 2.3 of
medical_vqa_architecture.md).

The final system prompt is a composition of the track context (Stage 3's
clinical-reasoning track) and the modality context (Stage 1's detected
modality). This keeps the prompt library small and combinatorial:
len(config.MODALITIES) x len(config.TRACKS) = 4 x 5 = 20 combinations,
built from just 5 + 4 = 9 template strings, rather than requiring 20
hand-written prompts.
"""

from src import config

# ---------------------------------------------------------------------------
# Track prompt templates (Section 2.2, verbatim)
# ---------------------------------------------------------------------------
TRACK_PROMPTS = {
    config.TRACK_DIAGNOSTIC: (
        "You are a board-certified diagnostic radiologist and pathologist with expertise "
        "across all imaging modalities (radiography, CT, MRI, ultrasound, dermoscopy, "
        "histopathology, fundoscopy). Examine the image carefully for the modality-"
        "appropriate diagnostic features (opacity/density patterns for radiographs, "
        "morphology/border irregularity for dermoscopy, cellular architecture for "
        "histology). Select the single most likely diagnosis from the choices given. "
        "Respond with ONLY the letter of the correct answer."
    ),
    config.TRACK_SPATIAL: (
        "You are an expert in medical imaging anatomy. Identify the anatomical "
        "structure, location, or laterality referenced in the image, accounting for "
        "standard radiological convention (patient left = image right for AP/PA "
        "views unless stated otherwise). Respond with ONLY the letter of the "
        "correct answer."
    ),
    config.TRACK_SEVERITY: (
        "You are a specialist in clinical grading and staging systems (e.g., BI-RADS, "
        "Gleason, TNM, Fitzpatrick, ISUP). Assess the visual severity indicators "
        "present in the image and match them to the most appropriate established "
        "grading criteria among the choices. Respond with ONLY the letter of the "
        "correct answer."
    ),
    config.TRACK_MODALITY_ID: (
        "You are an imaging physicist and radiologic technologist. Identify technical "
        "imaging characteristics such as modality type, sequence weighting, contrast "
        "phase, or acquisition parameters visible in the image. Respond with ONLY "
        "the letter of the correct answer."
    ),
    config.TRACK_DIFFERENTIAL: (
        "You are a senior attending physician conducting differential diagnosis. "
        "Systematically evaluate each choice against the visual evidence and "
        "eliminate options that are inconsistent with the image. Respond with ONLY "
        "the letter of the correct answer that best fits or is best excluded, as "
        "asked."
    ),
}

# ---------------------------------------------------------------------------
# Modality context templates. Section 2.3's build_system_prompt() shows
# `MODALITY_PROMPTS[modality]  # e.g. "This is a histopathology image..."`
# without pinning exact wording -- these mirror the modality-detection-
# signal language from the Section 1.2 preprocessing table so the prompt
# stays consistent with what Stage 1/2 actually detected and normalized.
# ---------------------------------------------------------------------------
MODALITY_PROMPTS = {
    config.MODALITY_RADIOLOGY: (
        "This is a radiology image (X-ray, CT, or MRI). Attend to opacity/density "
        "patterns, tissue contrast, and standard radiological viewing conventions; "
        "grayscale intensity carries the diagnostic signal."
    ),
    config.MODALITY_MACROSCOPIC: (
        "This is a macroscopic clinical photograph (dermatology, fundus, or gross "
        "pathology). Attend to lesion morphology, color, border irregularity, and "
        "surface texture."
    ),
    config.MODALITY_MICROSCOPY: (
        "This is a microscopy image (histopathology or cytology), typically H&E- or "
        "IHC-stained. Attend to cellular architecture, tissue organization, and "
        "staining patterns rather than gross anatomy."
    ),
    config.MODALITY_ULTRASOUND: (
        "This is an ultrasound image. Attend to echogenicity and tissue texture "
        "within the fan/sector-shaped field of view, accounting for speckle noise "
        "inherent to the modality."
    ),
}


# ---------------------------------------------------------------------------
# Stage 4: composition (Section 2.3, verbatim signature/body)
# ---------------------------------------------------------------------------
def build_system_prompt(modality: str, track: str) -> str:
    modality_context = MODALITY_PROMPTS[modality]  # e.g. "This is a histopathology image..."
    track_context = TRACK_PROMPTS[track]  # e.g. "You are a diagnostic specialist..."
    return f"{track_context}\n\nImaging context: {modality_context}"


# ---------------------------------------------------------------------------
# Pre-composed at import time (Section 4.4: "Avoid re-tokenizing static
# system prompt text per call -- pre-tokenize the 9 template strings
# (Section 2.3) once at load time"). Tokenization itself happens downstream
# in the processor's chat template, but the string-concatenation work for
# all 4 x 5 combinations is cheap to do once here rather than repeating it
# on every predict() call.
# ---------------------------------------------------------------------------
PROMPT_CACHE = {
    (modality, track): build_system_prompt(modality, track)
    for modality in config.MODALITIES
    for track in config.TRACKS
}


def get_system_prompt(modality: str, track: str) -> str:
    """Cached equivalent of build_system_prompt(modality, track): returns
    the precomputed string instead of re-concatenating it on every call."""
    return PROMPT_CACHE[(modality, track)]
