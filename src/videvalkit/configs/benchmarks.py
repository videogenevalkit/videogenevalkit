"""Benchmark registry."""

from videvalkit.benchmarks.physics_iq import PhysicsIQBenchmark
from videvalkit.benchmarks.semantics_axis import SemanticsAxisBenchmark
from videvalkit.benchmarks.t2vcompbench import T2VCompBenchBenchmark
from videvalkit.benchmarks.v_reasonbench import VReasonBenchBenchmark
from videvalkit.benchmarks.vbench import VBenchBenchmark
from videvalkit.benchmarks.vbench2 import VBench2Benchmark
from videvalkit.benchmarks.vbench_pp import VBenchPPBenchmark
from videvalkit.benchmarks.videobench import VideoBenchBenchmark
from videvalkit.benchmarks.worldjen import WorldJenBenchmark
from videvalkit.benchmarks.worldscore import WorldScoreBenchmark


# All four benchmarks share one conda env (`videvalkit`). Upstream's
# version pins are overrideable in practice; see envs/videvalkit.yaml.
_SHARED_ENV = "videvalkit"

SUPPORTED_BENCHMARKS = {
    "vbench": dict(
        cls=VBenchBenchmark,
        env=_SHARED_ENV,
        needs_gpu=True,
        needs_judge=False,
        default_aggregator="vbench_weighted",
    ),
    "vbench2": dict(
        cls=VBench2Benchmark,
        env=_SHARED_ENV,
        needs_gpu=True,
        needs_judge=True,
        default_judge="local-llava-video-7b",
        paper_judge="local-llava-video-7b",
        default_aggregator="vbench2_category",
    ),
    "videobench": dict(
        cls=VideoBenchBenchmark,
        env=_SHARED_ENV,
        needs_gpu=False,
        needs_judge=True,
        default_judge="gpt-4o",
        paper_judge="gpt-4o",
        default_aggregator="weighted_sum",
    ),
    "worldjen": dict(
        cls=WorldJenBenchmark,
        env=_SHARED_ENV,
        needs_gpu=False,
        needs_judge=True,
        default_judge="gemma-4-31b-local",
        paper_judge="gemma-4-31b-local",
        default_aggregator="phas",
    ),
    "t2vcompbench": dict(
        cls=T2VCompBenchBenchmark,
        env=_SHARED_ENV,
        needs_gpu=True,                # GroundingDINO / Depth / SAM
        needs_judge=True,              # LLaVA-Video for 4 of 7 dims
        default_judge="local-llava-video-7b",
        paper_judge="paper-llava-1.6-34b",
        default_aggregator="weighted_sum",
    ),
    "physics_iq": dict(
        cls=PhysicsIQBenchmark,
        env=_SHARED_ENV,
        needs_gpu=False,               # pixel-level CV; no model weights
        needs_judge=False,             # no VLM judge in the headline metric
        default_aggregator="weighted_sum",
    ),
    "vbench_pp": dict(
        cls=VBenchPPBenchmark,
        env=_SHARED_ENV,               # shares VBench v1 env
        needs_gpu=True,
        needs_judge=True,              # I2V + Trustworthiness use a VLM
        default_judge="local-llava-video-7b",
        paper_judge="local-llava-video-7b",
        default_aggregator="vbench_weighted",
    ),
    "v_reasonbench": dict(
        cls=VReasonBenchBenchmark,
        env=_SHARED_ENV,
        needs_gpu=False,               # deterministic per-task verifiers
        needs_judge=False,
        default_aggregator="weighted_sum",
    ),
    "worldscore": dict(
        cls=WorldScoreBenchmark,
        env=_SHARED_ENV,
        needs_gpu=True,                # VGG16 + pyiqa CLIP run on CUDA when available
        needs_judge=False,             # pure CV / pyiqa pipeline
        default_aggregator="weighted_sum",
    ),
    "semantics_axis": dict(
        cls=SemanticsAxisBenchmark,
        env=_SHARED_ENV,
        needs_gpu=False,               # VLM-judge only, no local CV checkpoints
        needs_judge=True,
        default_judge="gemma-4-31b-local",
        paper_judge="gemma-4-31b-local",
        default_aggregator="weighted_sum",
    ),
}


# ---------------------------------------------------------------------------
# Per-dimension capability tags (CAPABILITY_TAGS_DESIGN.md §5).
# Mapping: bench_name → {dim_name: [tags...]}
# Tags must come from the 44-entry controlled vocab in
# `configs.capability_taxonomy.ALL_TAGS`. Test
# `tests/test_capability_taxonomy_consistency.py` enforces this.
#
# v0.2 ships full mapping for vbench (16 dims, complete). Other benches will be
# annotated incrementally as their dim implementations land. Missing entries
# are permitted but flagged by the coverage report.
# ---------------------------------------------------------------------------
# Convention: a dim that is VLM/judge-scored AND checks whether the prompt's
# content appears in the video carries ``align.text2video`` IN ADDITION to its
# aspect tag(s). Intrinsic-quality dims (aesthetics / physics / smoothness)
# stay aspect-only. Pure prompt-following axes also get ``align.prompt_following``.
DIM_TAGS_BY_BENCH: dict[str, dict[str, list[str]]] = {
    "vbench": {
        # intrinsic / CV (no prompt-checking) — aspect only
        "subject_consistency":      ["subj.identity"],
        "background_consistency":   ["subj.appearance", "temp.continuity"],
        "temporal_flickering":      ["temp.flickering"],
        "motion_smoothness":        ["motion.smoothness"],
        "dynamic_degree":           ["motion.magnitude"],
        "aesthetic_quality":        ["vq.aesthetic"],
        "imaging_quality":          ["vq.imaging"],
        # prompt-checking (CV/VLM verifies prompt content in video) — add align
        "object_class":             ["obj.presence", "align.text2video"],
        "multiple_objects":         ["comp.multi_object", "obj.count", "align.text2video"],
        "human_action":             ["align.action_verb", "align.text2video"],
        "color":                    ["obj.attribute", "align.text2video"],
        "spatial_relationship":     ["comp.spatial", "align.text2video"],
        "scene":                    ["comp.multi_object", "align.text2video"],
        "appearance_style":         ["style.consistency", "align.text2video"],
        "temporal_style":           ["style.consistency", "temp.scene_consistency", "align.text2video"],
        "overall_consistency":      ["align.prompt_following", "align.text2video"],
    },
    "vbench2": {
        # Creativity / Commonsense — intrinsic
        "Composition":              ["style.consistency", "vq.aesthetic"],
        "Diversity":                ["style.consistency"],
        "Instance_Preservation":    ["subj.identity", "temp.continuity"],
        "Motion_Rationality":       ["phys.kinematics", "phys.causality"],
        # Controllability — prompt-following → align
        "Camera_Motion":            ["motion.magnitude", "align.text2video"],
        "Complex_Landscape":        ["comp.multi_object", "align.text2video"],
        "Complex_Plot":             ["comp.multi_object", "align.text2video"],
        "Dynamic_Attribute":        ["obj.attribute", "align.text2video"],
        "Dynamic_Spatial_Relationship": ["comp.spatial", "align.text2video"],
        "Human_Interaction":        ["comp.multi_object", "align.text2video"],
        "Motion_Order_Understanding": ["motion.accuracy", "align.text2video"],
        # Human Fidelity — intrinsic (anatomical correctness)
        "Human_Anatomy":            ["phys.anatomy"],
        "Human_Clothes":            ["subj.appearance", "phys.anatomy"],
        "Human_Identity":           ["subj.identity", "subj.character"],
        # Physics — intrinsic
        "Material":                 ["phys.kinematics"],
        "Mechanics":                ["phys.gravity", "phys.kinematics", "phys.causality"],
        "Multi-View_Consistency":   ["temp.scene_consistency"],
        "Thermotics":               ["phys.kinematics", "phys.causality"],
    },
    "videobench": {
        # static / dynamic — intrinsic
        "imaging_quality":          ["vq.imaging"],
        "aesthetic_quality":        ["vq.aesthetic"],
        "temporal_consistency":     ["temp.continuity", "temp.flickering"],
        "motion_effects":           ["motion.smoothness", "motion.naturalness"],
        # alignment category — all VLM prompt-checking
        "video_text_consistency":   ["align.text2video", "align.prompt_following"],
        "object_class_consistency": ["obj.presence", "align.text2video"],
        "color_consistency":        ["obj.attribute", "align.text2video"],
        "action_consistency":       ["align.action_verb", "align.text2video"],
        "scene_consistency":        ["comp.multi_object", "align.text2video"],
    },
    "worldjen": {
        # motion_stability — intrinsic
        "subject_consistency":      ["subj.identity"],
        "scene_consistency":        ["temp.scene_consistency", "subj.appearance"],
        "motion_smoothness":        ["motion.smoothness"],
        "temporal_flickering":      ["temp.flickering"],
        "inertial_consistency":     ["phys.kinematics", "motion.naturalness"],
        # logic_physics — intrinsic
        "physical_mechanics":       ["phys.gravity", "phys.kinematics", "phys.causality"],
        "object_permanence":        ["obj.presence", "temp.continuity"],
        "human_fidelity":           ["phys.anatomy", "subj.character"],
        "dynamic_degree":           ["motion.magnitude"],
        # instruction_adherence — pure alignment
        "semantic_adherence":       ["align.text2video", "align.prompt_following"],
        "spatial_relationship":     ["comp.spatial", "align.text2video"],
        "semantic_drift":           ["align.text2video", "temp.scene_consistency"],
        # aesthetic_quality — intrinsic
        "composition_framing":      ["vq.aesthetic", "style.consistency"],
        "lighting_volumetric":      ["vq.imaging"],
        "color_harmony":            ["vq.aesthetic"],
        "structural_gestalt":       ["vq.aesthetic", "style.consistency"],
    },
    "worldscore": {
        # 10 official dims (7 static + 3 dynamic). CV-heavy (DROID-SLAM /
        # RAFT / SAM / IQA). Prompt-conditioned dims gain align.text2video.
        # static
        "camera_control":           ["motion.magnitude", "align.text2video"],
        "object_control":           ["obj.presence", "align.text2video"],
        "content_alignment":        ["align.text2video", "align.prompt_following"],
        "3d_consistency":           ["temp.scene_consistency", "phys.kinematics"],
        "photometric_consistency":  ["temp.continuity", "vq.imaging"],
        "style_consistency":        ["style.consistency"],
        "subjective_quality":       ["vq.aesthetic", "vq.imaging"],
        # dynamic
        "motion_accuracy":          ["motion.accuracy", "align.action_verb", "align.text2video"],
        "motion_magnitude":         ["motion.magnitude"],
        "motion_smoothness":        ["motion.smoothness"],
    },
    "t2vcompbench": {
        # all 7 dims check prompt-following (4 MLLM-judged + 3 CV) → all get align
        "consistent_attribute":     ["obj.attribute", "align.text2video", "align.prompt_following"],
        "dynamic_attribute":        ["obj.attribute", "align.text2video", "align.prompt_following"],
        "action_binding":           ["align.action_verb", "align.text2video"],
        "object_interactions":      ["comp.multi_object", "align.text2video"],
        "spatial_relationships":    ["comp.spatial", "align.text2video"],
        "generative_numeracy":      ["comp.numeracy", "obj.count", "align.text2video"],
        "motion_binding":           ["motion.accuracy", "align.action_verb", "align.text2video"],
    },
    "physics_iq": {
        "physics_iq":               ["phys.gravity", "phys.kinematics", "phys.causality"],
    },
    "vbench_pp": {
        # VBench++ = VBench v1 (16, tagged under 'vbench' above) + 4 I2V extras
        # + 5 Trustworthiness dims. Only the 9 NEW dims are listed here;
        # capability lookups under --bench vbench_pp need to consult both keys.
        # I2V extras
        "i2v_subject":              ["subj.identity", "align.text2video"],
        "i2v_background":           ["subj.appearance", "temp.continuity", "align.text2video"],
        "camera_motion":            ["motion.magnitude", "align.text2video"],
        "video_image_consistency":  ["subj.appearance", "align.text2video"],
        # Trustworthiness — content-safety / bias / refusal; no canonical
        # capability fit beyond intrinsic content. Tag with vq.artifact_free
        # as a stand-in for 'unsafe outputs detected'; trust is a separate
        # axis the 44-tag taxonomy doesn't model and shouldn't pretend to.
        "culture_fairness":         ["vq.artifact_free"],
        "gender_bias":              ["vq.artifact_free"],
        "skin_tone_bias":           ["vq.artifact_free"],
        "content_safety":           ["vq.artifact_free"],
        "refusal_rate":             ["vq.artifact_free"],
    },
    "v_reasonbench": {
        # 13 deterministic reasoning verifiers in 4 categories. None map
        # cleanly to the visual capability vocab; tag by closest aspect.
        # structured_problem_solving (combinatorial / numeric reasoning)
        "tic_tac_toe":              ["comp.numeracy", "comp.spatial"],
        "sudoku":                   ["comp.numeracy"],
        "n_queens":                 ["comp.numeracy", "comp.spatial"],
        "tower_of_hanoi":           ["comp.numeracy", "phys.causality"],
        # spatial_cognition
        "color_connection":         ["comp.spatial", "obj.attribute"],
        "maze_navigation":          ["comp.spatial", "motion.accuracy"],
        "shape_matching":           ["comp.spatial", "obj.attribute"],
        # pattern_inference
        "rule_following":           ["align.prompt_following"],
        "sequence_continuation":    ["align.prompt_following", "temp.continuity"],
        # physical_dynamics
        "block_sliding":            ["phys.kinematics", "phys.causality"],
        "domino_chain":             ["phys.causality", "phys.kinematics"],
        "pendulum":                 ["phys.gravity", "phys.kinematics"],
        "projectile":               ["phys.gravity", "phys.kinematics"],
    },
    "semantics_axis": {
        # ALL 21 axes are VLM prompt-following; every axis gets the pair
        # (align.text2video, align.prompt_following) + aspect.
        "object_class":                  ["obj.presence", "align.text2video", "align.prompt_following"],
        "multiple_objects":              ["comp.multi_object", "obj.count", "align.text2video", "align.prompt_following"],
        "color":                         ["obj.attribute", "align.text2video", "align.prompt_following"],
        "material":                      ["obj.attribute", "align.text2video", "align.prompt_following"],
        "scene":                         ["comp.multi_object", "align.text2video", "align.prompt_following"],
        "style":                         ["style.consistency", "align.text2video", "align.prompt_following"],
        "pose":                          ["phys.anatomy", "align.text2video", "align.prompt_following"],
        "emotion":                       ["subj.character", "align.text2video", "align.prompt_following"],
        "text_ocr":                      ["obj.attribute", "align.text2video", "align.prompt_following"],
        "spatial_relationship":          ["comp.spatial", "align.text2video", "align.prompt_following"],
        "action":                        ["align.action_verb", "align.text2video", "align.prompt_following"],
        "motion_order":                  ["motion.accuracy", "align.text2video", "align.prompt_following"],
        "dynamic_attribute":             ["obj.attribute", "align.text2video", "align.prompt_following"],
        "dynamic_spatial_relationship":  ["comp.spatial", "align.text2video", "align.prompt_following"],
        "human_interaction":             ["comp.multi_object", "align.text2video", "align.prompt_following"],
        "complex_plot":                  ["comp.multi_object", "align.text2video", "align.prompt_following"],
        "complex_landscape":             ["comp.multi_object", "align.text2video", "align.prompt_following"],
        "camera_motion":                 ["motion.magnitude", "align.text2video", "align.prompt_following"],
        "shot_composition":              ["style.consistency", "align.text2video", "align.prompt_following"],
        "temporal_modifier":             ["temp.continuity", "align.text2video", "align.prompt_following"],
        "overall":                       ["align.prompt_following", "align.text2video"],
    },
}

# Attach dim_tags into each bench entry for ergonomic lookup.
for _bench_name, _dim_tags in DIM_TAGS_BY_BENCH.items():
    if _bench_name in SUPPORTED_BENCHMARKS:
        SUPPORTED_BENCHMARKS[_bench_name]["dim_tags"] = _dim_tags


# Public superset relationships, surfaced by `videvalkit list benchmarks`.
# Use when one benchmark's dimension set is a strict superset of another's
# and shares the v1 implementation.
BENCHMARK_RELATIONS: dict[str, dict[str, str]] = {
    "vbench_pp": {"superset_of": "vbench"},
}
