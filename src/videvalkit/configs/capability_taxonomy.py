"""Capability tags — controlled vocab for cross-bench ability evaluation.

Per docs/CAPABILITY_TAGS_DESIGN.md §3 (user 2026-05-20 confirmed):
  * 10 top-level tags  +  34 sub-tags  =  44 total
  * v1 controlled vocab — no free-form tags allowed
  * Schema version bumps required for vocab changes

Each metric / bench dim is annotated with 1-2 tags from this taxonomy via
the ``tags=[...]`` field on its registry entry. The capability resolver
(``capability.resolve_capability``) builds reverse indices.
"""

from __future__ import annotations

TAG_SCHEMA_VERSION = 1


# Top-level tags (10). Order matters for display.
TOP_LEVEL_TAGS: list[str] = [
    "motion",
    "visual_quality",
    "text_alignment",
    "object_fidelity",
    "subject_consistency",
    "physical_plausibility",
    "temporal_coherence",
    "realism",
    "compositional",
    "style",
]


# Sub-tags grouped by top-level (34 total). Each value is a list of sub-tags
# under that parent. Sub-tag canonical form is "<prefix>.<leaf>".
SUB_TAGS_BY_TOP: dict[str, list[str]] = {
    "motion": [
        "motion.smoothness",
        "motion.magnitude",
        "motion.accuracy",
        "motion.naturalness",
    ],
    "visual_quality": [
        "vq.aesthetic",
        "vq.imaging",
        "vq.artifact_free",
        "vq.sharpness",
    ],
    "text_alignment": [
        "align.text2video",
        "align.prompt_following",
        "align.action_verb",
    ],
    "object_fidelity": [
        "obj.presence",
        "obj.count",
        "obj.attribute",
        "obj.binding",
    ],
    "subject_consistency": [
        "subj.identity",
        "subj.appearance",
        "subj.character",
    ],
    "physical_plausibility": [
        "phys.gravity",
        "phys.causality",
        "phys.anatomy",
        "phys.kinematics",
    ],
    "temporal_coherence": [
        "temp.flickering",
        "temp.continuity",
        "temp.scene_consistency",
    ],
    "realism": [
        "real.distribution",
        "real.detection",
        "real.artifact_rate",
    ],
    "compositional": [
        "comp.multi_object",
        "comp.spatial",
        "comp.numeracy",
    ],
    "style": [
        "style.aesthetic",
        "style.cg_anime",
        "style.consistency",
    ],
}


# Flat set of all valid tags (top + sub) — for fast membership checks.
ALL_TAGS: frozenset[str] = frozenset(
    list(TOP_LEVEL_TAGS) + [t for subs in SUB_TAGS_BY_TOP.values() for t in subs]
)

# Sub-tag prefix → parent top-level. Used by `capabilities show <sub>`
# to climb up to its parent for display.
SUB_TAG_TO_TOP: dict[str, str] = {
    sub: top for top, subs in SUB_TAGS_BY_TOP.items() for sub in subs
}


# Human-readable descriptions for `videvalkit capabilities show <tag>`.
TAG_DESCRIPTIONS: dict[str, str] = {
    "motion":                "How things move — speed, smoothness, accuracy, naturalness",
    "motion.smoothness":     "Frame-to-frame motion smoothness (no judder/stutter)",
    "motion.magnitude":      "Magnitude of motion across the clip (static vs dynamic)",
    "motion.accuracy":       "Whether the motion shown matches what the prompt asked for",
    "motion.naturalness":    "Whether motion looks natural vs artificial",

    "visual_quality":        "Frame-level visual quality — aesthetic / imaging / artifacts",
    "vq.aesthetic":          "Aesthetic quality (LAION-style)",
    "vq.imaging":             "Imaging quality (MUSIQ / IQA)",
    "vq.artifact_free":      "Absence of visible artifacts (flickering, noise, glitches)",
    "vq.sharpness":          "Clarity / sharpness / focus",

    "text_alignment":        "How well the video follows the text prompt",
    "align.text2video":      "Per-frame or per-clip text-video alignment (CLIP-style)",
    "align.prompt_following":"Overall prompt-following fidelity",
    "align.action_verb":     "Whether the verb / action in the prompt is shown",

    "object_fidelity":       "Object presence, count, attributes, binding",
    "obj.presence":          "The objects mentioned in the prompt actually appear",
    "obj.count":             "Number of objects matches the prompt",
    "obj.attribute":         "Object attributes (color, material, ...) match the prompt",
    "obj.binding":           "Correct object-attribute binding (red ball vs blue ball)",

    "subject_consistency":   "Same subject preserved across frames",
    "subj.identity":         "Identity / who-is-this preservation",
    "subj.appearance":       "Appearance continuity (clothes, hair, pose)",
    "subj.character":        "Character consistency across scenes",

    "physical_plausibility": "Physics, anatomy, causality realism",
    "phys.gravity":          "Gravity / falling / settling behave naturally",
    "phys.causality":        "Cause-effect relationships are coherent",
    "phys.anatomy":          "Anatomy (hands, faces, bodies) is plausible",
    "phys.kinematics":       "Motion mechanics (joints, friction) are plausible",

    "temporal_coherence":    "Frame-to-frame coherence",
    "temp.flickering":       "Absence of frame-level flicker",
    "temp.continuity":       "Temporal continuity of content",
    "temp.scene_consistency":"Scene composition stays consistent",

    "realism":               "Overall realism / real-vs-fake",
    "real.distribution":     "Distribution-level realism (FVD-family)",
    "real.detection":        "Whether AI-generated is detectable as such",
    "real.artifact_rate":    "Frequency of detectable artifacts",

    "compositional":         "Multi-object scenes, spatial layout, numeracy",
    "comp.multi_object":     "Multi-object composition",
    "comp.spatial":          "Spatial relationships (above/below/left/right/inside)",
    "comp.numeracy":         "Numeric counts of objects",

    "style":                 "Aesthetic / artistic style",
    "style.aesthetic":       "Aesthetic style match",
    "style.cg_anime":        "CG / animation style fidelity",
    "style.consistency":     "Style consistency across the clip",
}


def is_valid_tag(tag: str) -> bool:
    """True if ``tag`` is in the v1 controlled vocab (top-level OR sub)."""
    return tag in ALL_TAGS


def parent_of(sub_tag: str) -> str | None:
    """Return the top-level parent of a sub-tag, or None for top-level tags."""
    return SUB_TAG_TO_TOP.get(sub_tag)


def expand_capability(name: str) -> list[str]:
    """Expand a capability name to all tags it covers.

    - If ``name`` is a top-level tag, return [top, *all_subs_under_top].
    - If ``name`` is a sub-tag, return [name] only.
    - Otherwise raise ValueError.
    """
    if name in SUB_TAGS_BY_TOP:
        return [name] + SUB_TAGS_BY_TOP[name]
    if name in SUB_TAG_TO_TOP:
        return [name]
    raise ValueError(
        f"unknown capability tag {name!r}; valid tags: "
        f"top-level={TOP_LEVEL_TAGS}; "
        f"see docs/CAPABILITY_TAGS_DESIGN.md §3"
    )
