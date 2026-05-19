"""WorldJen — 16 dimensions, their text definitions, their sampling modes.

Ported verbatim from
  worldjen_local/system_prompts/phase_b_vqa_generator.txt
  worldjen_local/config_local.py
to keep VLM behavior bit-for-bit consistent.
"""

from __future__ import annotations


# 16 dims in 4 macro-categories. Order matches the PHAS weight vector.
WORLDJEN_DIMENSIONS = [
    # motion_stability
    "subject_consistency",
    "scene_consistency",
    "motion_smoothness",
    "temporal_flickering",
    "inertial_consistency",
    # logic_physics
    "physical_mechanics",
    "object_permanence",
    "human_fidelity",
    "dynamic_degree",
    # instruction_adherence
    "semantic_adherence",
    "spatial_relationship",
    "semantic_drift",
    # aesthetic_quality
    "composition_framing",
    "lighting_volumetric",
    "color_harmony",
    "structural_gestalt",
]


WORLDJEN_CATEGORIES: dict[str, list[str]] = {
    "motion_stability": [
        "subject_consistency", "scene_consistency", "motion_smoothness",
        "temporal_flickering", "inertial_consistency",
    ],
    "logic_physics": [
        "physical_mechanics", "object_permanence", "human_fidelity", "dynamic_degree",
    ],
    "instruction_adherence": [
        "semantic_adherence", "spatial_relationship", "semantic_drift",
    ],
    "aesthetic_quality": [
        "composition_framing", "lighting_volumetric", "color_harmony", "structural_gestalt",
    ],
}


# One-line natural-language definition for each dim — fed to the VQA generator.
WORLDJEN_DEFINITIONS: dict[str, str] = {
    "subject_consistency":   "Does the main character/object change shape, color, or identity during the video?",
    "scene_consistency":     "Does the environment (trees, buildings, background) stay stable or 'warp'/'melt' as the camera moves?",
    "motion_smoothness":     "Does the video have 'stuttering,' 'jitter,' or frames that look like they're skipping?",
    "temporal_flickering":   "Are there flashes of light or sudden brightness changes (unwanted flickering/artifacts)?",
    "inertial_consistency":  "Do objects follow the laws of momentum — speeding up and slowing down naturally?",
    "physical_mechanics":    "Do gravity, friction, and collisions look realistic?",
    "object_permanence":     "If an object goes out of view or behind a wall, does it look exactly the same when it reappears?",
    "human_fidelity":        "Are humans rendered without 'alien' artifacts like extra fingers, distorted faces, etc.?",
    "dynamic_degree":        "Is there actual movement, or is it just a still image with zoom?",
    "semantic_adherence":    "Does the video contain exactly what was asked for in the prompt?",
    "spatial_relationship":  "Are objects in the right place relative to each other?",
    "semantic_drift":        "Does the AI start following the prompt but 'forget' it and change the scene halfway through?",
    "composition_framing":   "Is the shot well-balanced, or does it feel like a random crop?",
    "lighting_volumetric":   "Is the lighting realistic with depth, or does it look flat and 'CGI-like'?",
    "color_harmony":         "Are the colors pleasing and consistent, or is there 'digital bleeding'?",
    "structural_gestalt":    "Do the elements look like they belong in the same world, or like stickers pasted on?",
}


# Per-dim frame sampling mode (controls n_frames + temporal density).
# NOTE: upstream `worldjen_local/config_local.py` had an inconsistency — it listed
# `aesthetic_quality` (umbrella name) instead of `spatial_relationship`. We fix
# that here: spatial_relationship runs in holistic mode (content-level dim).
WORLDJEN_DIMENSION_MODES: dict[str, str] = {
    "semantic_adherence":   "holistic",
    "spatial_relationship": "holistic",
    "composition_framing":  "holistic",
    "lighting_volumetric":  "holistic",
    "color_harmony":        "holistic",
    "structural_gestalt":   "holistic",
    "scene_consistency":    "sampled",
    "object_permanence":    "sampled",
    "subject_consistency":  "sampled",
    "motion_smoothness":    "micro",
    "temporal_flickering":  "micro",
    "inertial_consistency": "micro",
    "physical_mechanics":   "micro",
    "human_fidelity":       "micro",
    "dynamic_degree":       "holistic",
    "semantic_drift":       "holistic",
}


# Upstream parallel_vlm_evaluator.py:52-61:
#   sampled  → 16 frames, evenly spaced (i * total/16)
#   micro    → up to 12 frames, range(0, min(total,60), 5) — dense prefix
#   holistic → up to 32 frames, evenly spaced
WORLDJEN_FRAMES_PER_MODE: dict[str, int] = {
    "holistic": 32,
    "sampled":  16,
    "micro":    12,
}
