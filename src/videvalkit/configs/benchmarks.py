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
DIM_TAGS_BY_BENCH: dict[str, dict[str, list[str]]] = {
    "vbench": {
        "subject_consistency":      ["subj.identity"],
        "background_consistency":   ["subj.appearance", "temp.continuity"],
        "temporal_flickering":      ["temp.flickering"],
        "motion_smoothness":        ["motion.smoothness"],
        "dynamic_degree":           ["motion.magnitude"],
        "aesthetic_quality":        ["vq.aesthetic"],
        "imaging_quality":          ["vq.imaging"],
        "object_class":             ["obj.presence"],
        "multiple_objects":         ["comp.multi_object", "obj.count"],
        "human_action":             ["align.action_verb"],
        "color":                    ["obj.attribute"],
        "spatial_relationship":     ["comp.spatial"],
        "scene":                    ["comp.multi_object"],
        "appearance_style":         ["style.consistency"],
        "temporal_style":           ["style.consistency", "temp.scene_consistency"],
        "overall_consistency":      ["align.prompt_following"],
    },
    "vbench2": {
        # 18 dims across 5 categories — full mapping deferred to v0.2 M3 work;
        # cover the most common ones for capability resolver demonstration:
        "Camera_Motion":            ["motion.magnitude"],
        "Human_Anatomy":            ["phys.anatomy"],
        "Motion_Order":             ["motion.accuracy", "phys.causality"],
        "Multi-View_Consistency":   ["temp.scene_consistency"],
        "Diversity":                ["style.consistency"],
    },
    "videobench": {
        "video_text_consistency":   ["align.text2video", "align.prompt_following"],
        "action_consistency":       ["align.action_verb"],
        "object_consistency":       ["obj.presence"],
        "color_consistency":        ["obj.attribute"],
    },
    "worldjen": {
        # 16 dims in 4 categories — partial mapping for v0.2.
        # Full mapping in M3 lift-out work.
    },
    "worldscore": {
        "motion_magnitude":         ["motion.magnitude"],
        "motion_accuracy":          ["motion.accuracy", "align.action_verb"],
        "camera_control":           ["motion.magnitude"],
        "3d_consistency":           ["temp.scene_consistency"],
        "subject_consistency":      ["subj.identity"],
        "style":                    ["style.consistency"],
    },
    "t2vcompbench": {
        "object_binding":           ["obj.binding", "obj.presence"],
        "spatial_relationship":     ["comp.spatial"],
        "numeracy":                 ["comp.numeracy", "obj.count"],
        "consistent_attribute":     ["obj.attribute"],
        "motion_binding":           ["motion.accuracy", "align.action_verb"],
    },
    "physics_iq": {
        "physics_iq":               ["phys.gravity", "phys.kinematics", "phys.causality"],
    },
    "vbench_pp": {},
    "v_reasonbench": {},
    "semantics_axis": {},
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
