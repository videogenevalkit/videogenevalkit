"""Artifact taxonomy for the ``artifact-diagnostic`` metric.

Ported from Artifact-Bench (arXiv 2605.18984, FrankYang-17/Artifact-Bench),
which defines a three-level hierarchy of AI-generated-video realism artifacts:
3 top categories -> failure families -> fine-grained artifact types.

The 3 top categories are taken directly from the paper (temporal
inconsistencies / structural distortions / semantic incoherence). The family
and fine-grained leaf names below are the v0.2 working port grounded in that
structure and the AIGC-video artifact literature.

NOTE (v0.2): the paper publishes the 11-family / 30-type leaves as a figure,
not machine-readable text. Treat the leaf strings here as the operational
taxonomy and reconcile them against arXiv 2605.18984 §taxonomy before any
paper-comparable reporting. The full Artifact-Bench judge-eval benchmark
lands in v0.3 — see docs/design and the integration record. This taxonomy is
the single source of truth: correct it here and the metric follows.

License: Artifact-Bench is academic-research-only (commercial use prohibited).
"""

from __future__ import annotations

# category -> family -> [fine-grained artifact types]
ARTIFACT_TAXONOMY: dict[str, dict[str, list[str]]] = {
    "temporal_inconsistencies": {
        "flicker_jitter": [
            "flickering",
            "frame_jitter",
            "temporal_aliasing",
        ],
        "motion_anomaly": [
            "motion_discontinuity",
            "unnatural_motion_speed",
            "object_teleportation",
        ],
        "continuity_break": [
            "object_popping",
            "identity_drift",
            "texture_sliding",
            "background_instability",
        ],
    },
    "structural_distortions": {
        "anatomy_distortion": [
            "human_anatomy_distortion",
            "face_distortion",
            "hand_finger_distortion",
        ],
        "geometry_distortion": [
            "object_shape_distortion",
            "perspective_geometry_error",
            "rigid_object_deformation",
        ],
        "rendering_degradation": [
            "blur_softness",
            "noise_grain",
            "compression_blocking",
        ],
        "overlay_artifact": [
            "text_rendering_error",
            "watermark_logo_artifact",
        ],
    },
    "semantic_incoherence": {
        "physics_violation": [
            "rigid_body_physics_violation",
            "implausible_fluid_dynamics",
        ],
        "lighting_optics": [
            "lighting_shadow_inconsistency",
            "reflection_error",
        ],
        "object_logic": [
            "object_count_error",
            "object_relation_error",
            "nonsensical_object",
        ],
        "scene_action_logic": [
            "scene_layout_incoherence",
            "action_semantics_error",
        ],
    },
}


def _build_flat() -> tuple[list[str], dict[str, str], dict[str, str]]:
    types: list[str] = []
    type_to_family: dict[str, str] = {}
    type_to_category: dict[str, str] = {}
    for category, families in ARTIFACT_TAXONOMY.items():
        for family, leaves in families.items():
            for t in leaves:
                types.append(t)
                type_to_family[t] = family
                type_to_category[t] = category
    return types, type_to_family, type_to_category


ARTIFACT_TYPES, TYPE_TO_FAMILY, TYPE_TO_CATEGORY = _build_flat()
ARTIFACT_FAMILIES = [f for fams in ARTIFACT_TAXONOMY.values() for f in fams]
ARTIFACT_CATEGORIES = list(ARTIFACT_TAXONOMY)

N_ARTIFACT_TYPES = len(ARTIFACT_TYPES)  # 30
