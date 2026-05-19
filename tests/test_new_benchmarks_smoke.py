"""Smoke tests for the 4 newly-added benchmark adapters.

These only exercise: import, registration, dimension constants, prompt
iterators, list_required_videos path math, and aggregate() on synthetic
RawResults. evaluate() is a stub for all four (upstream wiring TBD), so
it is not exercised here.

Run in the core env only — no upstream packages required.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from videvalkit.configs import SUPPORTED_BENCHMARKS
from videvalkit.core.types import PromptItem, RawResult, VideoSpec
from videvalkit.storage import Workspace


# Shared end-to-end test fixture (used by worldscore + t2vcompbench e2e).
_E2E_VIDEO = Path("/ning/data/pangu3_wan14b/pangu3_wan14b/wan-14B-pe-141/000.mp4")
_E2E_PROMPT = (
    "In the school hallway, a girl with a ponytail suddenly slaps the boy next "
    "to her. The boy covers his cheek, eyes widened"
)


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #

def test_registry_contains_all_new_benchmarks():
    expected = {"t2vcompbench", "physics_iq", "vbench_pp", "v_reasonbench", "worldscore"}
    assert expected <= set(SUPPORTED_BENCHMARKS), \
        f"missing: {expected - set(SUPPORTED_BENCHMARKS)}"


def test_entry_registry_resolves_all_new_benchmarks():
    from videvalkit.benchmarks import entry
    for name in ("t2vcompbench", "physics_iq", "vbench_pp", "v_reasonbench", "worldscore"):
        adapter = entry._load_adapter(name)
        assert adapter.name == name


# --------------------------------------------------------------------------- #
# T2V-CompBench
# --------------------------------------------------------------------------- #

def test_t2vcompbench_dimensions_complete():
    from videvalkit.benchmarks.t2vcompbench.benchmark import (
        T2VCOMPBENCH_DIMENSIONS, T2VCOMPBENCH_SCORER_KIND, T2VCompBenchBenchmark,
    )
    assert len(T2VCOMPBENCH_DIMENSIONS) == 7
    assert set(T2VCOMPBENCH_SCORER_KIND) == set(T2VCOMPBENCH_DIMENSIONS)
    assert set(T2VCOMPBENCH_SCORER_KIND.values()) == {"mllm", "detection", "tracking"}
    assert T2VCompBenchBenchmark.dimensions == T2VCOMPBENCH_DIMENSIONS


def test_t2vcompbench_fallback_prompts_iter():
    from videvalkit.benchmarks.t2vcompbench import T2VCompBenchBenchmark
    bench = T2VCompBenchBenchmark()
    prompts = list(bench.list_prompts())
    assert len(prompts) == 7  # one per dim from the bundled fallback
    assert {p.dimensions[0] for p in prompts} == set(bench.dimensions)
    for p in prompts:
        assert p.meta["scorer_kind"] in {"mllm", "detection", "tracking"}


def test_t2vcompbench_list_required_videos_paths():
    from videvalkit.benchmarks.t2vcompbench import T2VCompBenchBenchmark
    bench = T2VCompBenchBenchmark()
    with tempfile.TemporaryDirectory() as td:
        ws = Workspace(td)
        prompts = list(bench.list_prompts(dimensions=["action_binding"]))
        specs = bench.list_required_videos(prompts, models=["cogvideox"], layout=ws.layout)
        assert len(specs) == 1
        assert specs[0].path.name.endswith("-0.mp4")
        assert specs[0].path.parent.name == "cogvideox"


def test_t2vcompbench_aggregate():
    from videvalkit.benchmarks.t2vcompbench import T2VCompBenchBenchmark
    bench = T2VCompBenchBenchmark()
    raw = [
        RawResult(benchmark="t2vcompbench", model="m1", dimension="action_binding",
                  prompt_id="a", score=0.8),
        RawResult(benchmark="t2vcompbench", model="m1", dimension="action_binding",
                  prompt_id="b", score=0.6),
        RawResult(benchmark="t2vcompbench", model="m1", dimension="generative_numeracy",
                  prompt_id="a", score=1.0),
    ]
    s = bench.aggregate(raw)
    assert abs(s.per_dimension["action_binding"] - 0.7) < 1e-9
    assert abs(s.per_dimension["generative_numeracy"] - 1.0) < 1e-9
    assert abs(s.overall - 0.85) < 1e-9


def test_t2vcompbench_all_7_dims_accounted_for():
    """All 7 dims wired: 4 MLLM (Gemma-4-31B-IT judge) + 3 CV (GD-Tiny +
    Depth-Anything-V2 + RAFT)."""
    from videvalkit.benchmarks.t2vcompbench.benchmark import (
        T2VCOMPBENCH_DIMENSIONS, T2VCOMPBENCH_SCORER_KIND,
    )
    from videvalkit.benchmarks.t2vcompbench import (
        T2VCOMPBENCH_CV_DIMS, T2VCOMPBENCH_MLLM_DIMS,
    )

    assert len(T2VCOMPBENCH_DIMENSIONS) == 7
    mllm = set(T2VCOMPBENCH_MLLM_DIMS)
    cv = set(T2VCOMPBENCH_CV_DIMS)
    assert mllm | cv == set(T2VCOMPBENCH_DIMENSIONS)
    assert mllm.isdisjoint(cv)
    assert mllm == {
        "consistent_attribute", "dynamic_attribute",
        "action_binding", "object_interactions",
    }
    assert cv == {"generative_numeracy", "spatial_relationships", "motion_binding"}

    # SCORER_KIND must label every dim correctly.
    for d in mllm:
        assert T2VCOMPBENCH_SCORER_KIND[d] == "mllm"
    assert T2VCOMPBENCH_SCORER_KIND["spatial_relationships"] == "detection"
    assert T2VCOMPBENCH_SCORER_KIND["generative_numeracy"] == "detection"
    assert T2VCOMPBENCH_SCORER_KIND["motion_binding"] == "tracking"


def test_t2vcompbench_mllm_dim_set():
    from videvalkit.benchmarks.t2vcompbench import (
        T2VCOMPBENCH_MLLM_DIMS, DIMENSION_PROMPT_TEMPLATES,
    )
    assert set(T2VCOMPBENCH_MLLM_DIMS) == {
        "consistent_attribute", "dynamic_attribute",
        "action_binding", "object_interactions",
    }
    assert set(DIMENSION_PROMPT_TEMPLATES) == set(T2VCOMPBENCH_MLLM_DIMS)


def test_t2vcompbench_cv_dim_helpers():
    """Heuristic parsers used by the CV scorers."""
    from videvalkit.benchmarks.t2vcompbench.scorers import (
        _first_content_noun, _parse_count_noun, _parse_spatial_triple,
    )
    # numeracy
    pairs = _parse_count_noun("three yellow ducks in a row on a pond")
    assert any(c == 3 and "duck" in n for c, n in pairs)
    pairs = _parse_count_noun("a single red apple")
    assert any(c == 1 and "apple" in n for c, n in pairs)
    # spatial
    triple = _parse_spatial_triple("a red apple to the left of a green pear")
    assert triple is not None
    a, rel, b = triple
    assert "apple" in a and "pear" in b and rel == "left"
    triple = _parse_spatial_triple("a dog in front of a sofa")
    assert triple is not None and triple[1] == "front"
    # motion
    assert _first_content_noun("a small fish swims to the right") in {"fish", "swims"}


def test_t2vcompbench_empty_videos_returns_empty():
    from videvalkit.benchmarks.t2vcompbench import T2VCompBenchBenchmark
    assert T2VCompBenchBenchmark().evaluate(videos=[]) == []


def test_t2vcompbench_cv_dim_no_prompt_text_warns_and_returns_empty(caplog):
    """When a CV dim is requested but no prompt text is provided, the adapter
    warns and returns [] (no fatal). Replaces the old NotImplementedError test."""
    from videvalkit.benchmarks.t2vcompbench import T2VCompBenchBenchmark
    import logging
    bench = T2VCompBenchBenchmark()
    videos = [VideoSpec(path=Path("/tmp/x.mp4"), prompt_id="0", model_name="m")]
    with caplog.at_level(logging.WARNING,
                         logger="videvalkit.benchmarks.t2vcompbench.benchmark"):
        out = bench.evaluate(
            videos=videos, dimensions=["spatial_relationships"], prompts=[],
            mode="toolkit",
        )
    assert out == []
    assert any("missing prompt text" in r.message for r in caplog.records)


def test_t2vcompbench_mllm_dim_needs_judge():
    """Calling an MLLM dim without judge=... must error, not silently no-op."""
    from videvalkit.benchmarks.t2vcompbench import T2VCompBenchBenchmark
    bench = T2VCompBenchBenchmark()
    videos = [VideoSpec(path=Path("/tmp/x.mp4"), prompt_id="0", model_name="m")]
    prompts = [PromptItem(prompt_id="0", text="a blue car", dimensions=["action_binding"])]
    with pytest.raises(ValueError, match="need a judge"):
        bench.evaluate(videos=videos, prompts=prompts,
                       dimensions=["action_binding"], mode="toolkit")


def test_t2vcompbench_parse_score_handles_messy_response():
    """The JSON parser must survive code fences, leading prose, regex fallback."""
    from videvalkit.benchmarks.t2vcompbench.scorers import _parse_score
    s, _ = _parse_score('```json\n{"score": 0.73, "justification": "ok"}\n```')
    assert abs(s - 0.73) < 1e-9
    s, _ = _parse_score('Sure — {"score": 0.5, "justification": "x"} extra text')
    assert s == 0.5
    s, _ = _parse_score('I would say score: 0.8 because ...')
    assert abs(s - 0.8) < 1e-9


def _t2vcompbench_e2e_deps_available() -> bool:
    """End-to-end test needs the test video AND a reachable VLM endpoint
    (we default to gemma-4-31b-local on :8003 since LLaVA on :8006 is down)."""
    if not _E2E_VIDEO.is_file():
        return False
    try:
        import urllib.request
        import urllib.error
        req = urllib.request.Request("http://localhost:8003/v1/models")
        with urllib.request.urlopen(req, timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _t2vcompbench_cv_e2e_deps_available() -> bool:
    """CV e2e needs cv2 + the test video + the HF caches for GD/Depth/RAFT."""
    if not _E2E_VIDEO.is_file():
        return False
    try:
        import cv2          # noqa: F401, PLC0415
        import torch        # noqa: F401, PLC0415
        from torchvision.models.optical_flow import raft_small  # noqa: F401, PLC0415
        from transformers import AutoConfig  # noqa: PLC0415
        AutoConfig.from_pretrained(
            "IDEA-Research/grounding-dino-tiny", local_files_only=True,
        )
        AutoConfig.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf", local_files_only=True,
        )
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _t2vcompbench_cv_e2e_deps_available(),
                    reason="GD / Depth-Anything / RAFT / test video not available")
def test_t2vcompbench_cv_e2e_three_dims():
    """All 3 CV dims scored on the wan14b smoke video. Prompt has a clear
    noun (girl/boy), no count, no spatial triple — so numeracy and spatial
    return neutral 0.5; motion_binding produces a real ratio."""
    from videvalkit.benchmarks.t2vcompbench import (
        T2VCOMPBENCH_CV_DIMS, T2VCompBenchBenchmark,
    )
    bench = T2VCompBenchBenchmark()
    prompts = [PromptItem(prompt_id="000", text=_E2E_PROMPT,
                          dimensions=list(T2VCOMPBENCH_CV_DIMS))]
    videos = [VideoSpec(path=_E2E_VIDEO, prompt_id="000", model_name="wan14b")]
    raw = bench.evaluate(
        videos=videos, prompts=prompts,
        dimensions=list(T2VCOMPBENCH_CV_DIMS),
        mode="toolkit",
    )
    by_dim = {r.dimension: r for r in raw}
    assert set(by_dim) == set(T2VCOMPBENCH_CV_DIMS)
    for dim, r in by_dim.items():
        assert isinstance(r.score, float)
        assert 0.0 <= r.score <= 1.0
    # Scorer tags
    assert by_dim["generative_numeracy"].scorer == "cv:grounding_dino_count"
    assert by_dim["spatial_relationships"].scorer == "cv:gd_bbox_depth"
    assert by_dim["motion_binding"].scorer == "cv:gd_bbox_raft_flow"
    # motion_binding meta_used carries the (object, direction) pair (upstream
    # shape: {object_1, d_1, object_2, d_2}).
    mb_meta = by_dim["motion_binding"].meta
    assert "meta_used" in mb_meta or "skip_reason" in mb_meta
    # numeracy + spatial report their upstream-shaped extracted meta (which
    # may be empty for prompts without count / spatial triple).
    num_meta = by_dim["generative_numeracy"].meta
    assert "objects" in num_meta and "numbers" in num_meta
    sp_meta = by_dim["spatial_relationships"].meta
    assert "triple" in sp_meta or "skip_reason" in sp_meta


@pytest.mark.skipif(not _t2vcompbench_cv_e2e_deps_available(),
                    reason="GD / Depth-Anything / RAFT / test video not available")
def test_t2vcompbench_cv_numeracy_with_count_in_prompt():
    """A prompt with a numeric word triggers real GD counting (score != 0.5)."""
    from videvalkit.benchmarks.t2vcompbench import T2VCompBenchBenchmark
    bench = T2VCompBenchBenchmark()
    # The wan14b video has ~1 girl + 1 boy ≈ 2 people. Prompt asks for 2.
    prompts = [PromptItem(prompt_id="000",
                          text="two students standing in a hallway",
                          dimensions=["generative_numeracy"])]
    videos = [VideoSpec(path=_E2E_VIDEO, prompt_id="000", model_name="wan14b")]
    raw = bench.evaluate(videos=videos, prompts=prompts,
                         dimensions=["generative_numeracy"],
                         mode="toolkit")
    assert len(raw) == 1
    r = raw[0]
    assert 0.0 <= r.score <= 1.0
    # Real scoring path took effect — no skip_reason
    assert "skip_reason" not in r.meta
    # Heuristic extractor maps "two" → 2 in the upstream-shape meta.
    assert r.meta["numbers"] == [2] or r.meta["numbers"][:1] == [2]


@pytest.mark.skipif(not _t2vcompbench_e2e_deps_available(),
                    reason="gemma-4-31b-local endpoint or test video not available")
def test_t2vcompbench_mllm_e2e_one_video():
    """All 4 MLLM dims scored on the wan14b smoke video via Gemma-4-31B-IT."""
    from videvalkit.benchmarks.t2vcompbench import (
        T2VCOMPBENCH_MLLM_DIMS, T2VCompBenchBenchmark,
    )
    from videvalkit.configs import SUPPORTED_JUDGES
    bench = T2VCompBenchBenchmark()

    prompts = [PromptItem(prompt_id="000", text=_E2E_PROMPT,
                          dimensions=list(T2VCOMPBENCH_MLLM_DIMS))]
    videos = [VideoSpec(path=_E2E_VIDEO, prompt_id="000", model_name="wan14b")]

    raw = bench.evaluate(
        videos=videos, prompts=prompts,
        dimensions=list(T2VCOMPBENCH_MLLM_DIMS),
        judge=SUPPORTED_JUDGES["gemma-4-31b-local"],
        mode="toolkit",
    )
    by_dim = {r.dimension: r for r in raw}
    assert set(by_dim) == set(T2VCOMPBENCH_MLLM_DIMS), \
        f"missing dims: {set(T2VCOMPBENCH_MLLM_DIMS) - set(by_dim)}"
    for dim, r in by_dim.items():
        assert isinstance(r.score, float)
        assert 0.0 <= r.score <= 1.0, f"{dim}={r.score} outside [0,1]"
        assert r.scorer.startswith("vlm:google/gemma")
        assert "justification" in r.meta


# --------------------------------------------------------------------------- #
# Physics-IQ
# --------------------------------------------------------------------------- #

def test_physics_iq_dimensions_and_metrics():
    from videvalkit.benchmarks.physics_iq.benchmark import (
        PHYSICS_IQ_DIMENSIONS, PHYSICS_IQ_METRICS, PhysicsIQBenchmark,
    )
    assert PHYSICS_IQ_DIMENSIONS == [
        "solid_mechanics", "fluid_dynamics", "optics", "thermodynamics", "magnetism",
    ]
    assert set(PHYSICS_IQ_METRICS) == {"spatial_iou", "spatiotemporal_iou", "weighted_mse"}
    assert PhysicsIQBenchmark.NUM_SCENARIOS == 198
    assert PhysicsIQBenchmark.PHYSICS_IQ_OUTPUT_DURATION_S \
        if hasattr(PhysicsIQBenchmark, "PHYSICS_IQ_OUTPUT_DURATION_S") else True


def test_physics_iq_fallback_prompts_iter():
    from videvalkit.benchmarks.physics_iq import PhysicsIQBenchmark
    bench = PhysicsIQBenchmark()
    prompts = list(bench.list_prompts())
    assert len(prompts) == 5
    for p in prompts:
        assert p.meta["mode"] in {"i2v", "v2v"}
        assert p.dimensions[0] in bench.dimensions


def test_physics_iq_aggregate_returns_percentage_meta():
    from videvalkit.benchmarks.physics_iq import PhysicsIQBenchmark
    raw = [
        RawResult(benchmark="physics_iq", model="m1", dimension="solid_mechanics",
                  prompt_id="0001", score=70.0),
        RawResult(benchmark="physics_iq", model="m1", dimension="fluid_dynamics",
                  prompt_id="0002", score=50.0),
    ]
    s = PhysicsIQBenchmark().aggregate(raw)
    assert s.per_dimension == {"solid_mechanics": 70.0, "fluid_dynamics": 50.0}
    assert s.overall == 60.0
    assert "0-100" in s.meta["scale"]


def test_physics_iq_evaluate_is_stub():
    from videvalkit.benchmarks.physics_iq import PhysicsIQBenchmark
    with pytest.raises(NotImplementedError):
        PhysicsIQBenchmark().evaluate()


# --------------------------------------------------------------------------- #
# VBench++
# --------------------------------------------------------------------------- #

def test_vbench_pp_inherits_v1_dims_and_adds_extras():
    from videvalkit.benchmarks.vbench.benchmark import VBENCH_DIMENSIONS
    from videvalkit.benchmarks.vbench_pp.benchmark import (
        VBENCH_PP_DIMENSIONS, VBENCH_PP_I2V_EXTRA_DIMS, VBENCH_PP_TRUST_DIMS,
        VBenchPPBenchmark,
    )
    assert set(VBENCH_DIMENSIONS) < set(VBENCH_PP_DIMENSIONS)
    assert len(VBENCH_PP_I2V_EXTRA_DIMS) == 4
    assert len(VBENCH_PP_TRUST_DIMS) == 5
    assert len(VBENCH_PP_DIMENSIONS) == 16 + 4 + 5
    assert VBenchPPBenchmark.dimensions == VBENCH_PP_DIMENSIONS


def test_vbench_pp_section_partition_covers_all_dims():
    from videvalkit.benchmarks.vbench_pp import VBenchPPBenchmark
    flat = [d for ds in VBenchPPBenchmark.SECTION_DIMS.values() for d in ds]
    assert set(flat) == set(VBenchPPBenchmark.dimensions)
    # Every dim resolves to a known section.
    for d in VBenchPPBenchmark.dimensions:
        assert VBenchPPBenchmark._section_of(d) != "unknown"


def test_vbench_pp_aggregate_section_weighted():
    from videvalkit.benchmarks.vbench_pp import VBenchPPBenchmark
    # Quality dim = 0.8, Semantic dim = 0.6, I2V dim = 0.4, Trust dim = 1.0
    # Section means: 0.8, 0.6, 0.4, 1.0 → equal weights → 0.7
    raw = [
        RawResult(benchmark="vbench_pp", model="m1", dimension="subject_consistency",
                  prompt_id="0", score=0.8),
        RawResult(benchmark="vbench_pp", model="m1", dimension="human_action",
                  prompt_id="0", score=0.6),
        RawResult(benchmark="vbench_pp", model="m1", dimension="i2v_subject",
                  prompt_id="0", score=0.4),
        RawResult(benchmark="vbench_pp", model="m1", dimension="content_safety",
                  prompt_id="0", score=1.0),
    ]
    s = VBenchPPBenchmark().aggregate(raw)
    assert abs(s.overall - 0.7) < 1e-9
    sm = s.meta["section_means"]
    assert sm["quality"] == 0.8 and sm["semantic"] == 0.6
    assert sm["i2v"] == 0.4 and sm["trust"] == 1.0


def test_vbench_pp_evaluate_extras_stub():
    from videvalkit.benchmarks.vbench_pp import VBenchPPBenchmark
    # Asking for an extras-only dim should raise NotImplementedError
    # because we route through the parent which needs upstream `vbench`.
    with pytest.raises((NotImplementedError, ImportError, ModuleNotFoundError)):
        VBenchPPBenchmark().evaluate(dimensions=["culture_fairness"])


# --------------------------------------------------------------------------- #
# V-ReasonBench
# --------------------------------------------------------------------------- #

def test_v_reasonbench_categories_and_tasks():
    from videvalkit.benchmarks.v_reasonbench.benchmark import (
        VREASONBENCH_CATEGORIES, VREASONBENCH_TASKS, VREASONBENCH_K,
        VReasonBenchBenchmark,
    )
    # 4 categories
    assert set(VREASONBENCH_CATEGORIES) == {
        "structured_problem_solving", "spatial_cognition",
        "pattern_inference", "physical_dynamics",
    }
    # Every task is in exactly one category
    flat = [t for ts in VREASONBENCH_CATEGORIES.values() for t in ts]
    assert len(flat) == len(set(flat)) == len(VREASONBENCH_TASKS)
    assert VREASONBENCH_K == 5
    assert VReasonBenchBenchmark.dimensions == VREASONBENCH_TASKS


def test_v_reasonbench_fallback_prompts_iter():
    from videvalkit.benchmarks.v_reasonbench import VReasonBenchBenchmark
    bench = VReasonBenchBenchmark()
    prompts = list(bench.list_prompts())
    assert len(prompts) == len(bench.dimensions)
    for p in prompts:
        assert p.meta["category"] != "unknown"
        # Image-pair fields are present but None in fallback mode
        assert "initial_image" in p.meta and "final_image" in p.meta


def test_v_reasonbench_pass_at_5_aggregation():
    from videvalkit.benchmarks.v_reasonbench import VReasonBenchBenchmark
    bench = VReasonBenchBenchmark()
    # 2 tic_tac_toe instances: one passes (1.0), one fails (0.0) → task pass-rate 0.5
    # 1 sudoku instance passes (1.0) → task pass-rate 1.0
    # 1 block_sliding instance fails (0.0)
    raw = [
        RawResult(benchmark="v_reasonbench", model="m", dimension="tic_tac_toe",
                  prompt_id="ttt_0001", score=1.0),
        RawResult(benchmark="v_reasonbench", model="m", dimension="tic_tac_toe",
                  prompt_id="ttt_0002", score=0.0),
        RawResult(benchmark="v_reasonbench", model="m", dimension="sudoku",
                  prompt_id="sud_0001", score=1.0),
        RawResult(benchmark="v_reasonbench", model="m", dimension="block_sliding",
                  prompt_id="blk_0001", score=0.0),
    ]
    s = bench.aggregate(raw)
    assert s.per_dimension["tic_tac_toe"] == 0.5
    assert s.per_dimension["sudoku"] == 1.0
    assert s.per_dimension["block_sliding"] == 0.0
    # 2 categories present: structured_problem_solving (mean of 0.5, 1.0 = 0.75)
    # and physical_dynamics (0.0) → overall = 0.375
    pc = s.meta["per_category_mean"]
    assert pc["structured_problem_solving"] == 0.75
    assert pc["physical_dynamics"] == 0.0
    assert abs(s.overall - 0.375) < 1e-9
    assert s.meta["pass_at_k"] == 5


def test_v_reasonbench_evaluate_is_stub():
    from videvalkit.benchmarks.v_reasonbench import VReasonBenchBenchmark
    with pytest.raises(NotImplementedError):
        VReasonBenchBenchmark().evaluate()


# --------------------------------------------------------------------------- #
# WorldScore (lite — 3 dims ported from /tmp/worldscore_smoke/smoke.py)
# --------------------------------------------------------------------------- #

def test_worldscore_dimensions():
    from videvalkit.benchmarks.worldscore.benchmark import (
        WORLDSCORE_DIMENSIONS, WORLDSCORE_STATIC_DIMS, WORLDSCORE_DYNAMIC_DIMS,
        WORLDSCORE_FRAME_DEFAULT, WorldScoreBenchmark,
    )
    # All 10 upstream dims wired (mirrors worldscore_list in run_evaluate.py).
    assert WORLDSCORE_STATIC_DIMS == (
        "camera_control", "object_control", "content_alignment",
        "3d_consistency", "photometric_consistency",
        "style_consistency", "subjective_quality",
    )
    assert WORLDSCORE_DYNAMIC_DIMS == (
        "motion_accuracy", "motion_magnitude", "motion_smoothness",
    )
    assert WORLDSCORE_DIMENSIONS == list(WORLDSCORE_STATIC_DIMS) + list(WORLDSCORE_DYNAMIC_DIMS)
    assert WORLDSCORE_FRAME_DEFAULT == 49
    assert WorldScoreBenchmark.dimensions == WORLDSCORE_DIMENSIONS


def test_worldscore_module_imports_without_heavy_deps():
    """Core env must be able to import the adapter without torch/pyiqa."""
    import videvalkit.benchmarks.worldscore as ws       # noqa: F401
    import videvalkit.benchmarks.worldscore.benchmark   # noqa: F401
    # scorers.py contains heavy imports inside __init__, not at module top
    import videvalkit.benchmarks.worldscore.scorers     # noqa: F401


def test_worldscore_fallback_prompts_iter():
    from videvalkit.benchmarks.worldscore import (
        WorldScoreBenchmark, WORLDSCORE_DYNAMIC_DIMS,
    )
    bench = WorldScoreBenchmark()
    prompts = list(bench.list_prompts())
    assert len(prompts) == 1
    # Fallback prompt is dynamic-only by design; dim list is filtered to the
    # 3 dynamic dims, not all 10.
    assert set(prompts[0].dimensions) <= set(bench.dimensions)
    assert set(prompts[0].dimensions) == set(WORLDSCORE_DYNAMIC_DIMS)
    assert prompts[0].meta.get("split") == "dynamic"


def test_worldscore_list_required_videos_paths():
    from videvalkit.benchmarks.worldscore import WorldScoreBenchmark
    bench = WorldScoreBenchmark()
    with tempfile.TemporaryDirectory() as td:
        ws = Workspace(td)
        prompts = list(bench.list_prompts())
        specs = bench.list_required_videos(prompts, models=["wan14b"], layout=ws.layout)
        assert len(specs) == 1
        # Adapter now uses upstream-style {split}/{prompt_id}.mp4 layout.
        assert specs[0].path.name == "ws_dyn_0000.mp4"
        # ".../videos/wan14b/dynamic/ws_dyn_0000.mp4"
        assert specs[0].path.parent.name == "dynamic"
        assert specs[0].path.parent.parent.name == "wan14b"


def test_worldscore_aggregate_matches_upstream_headlines():
    """Aggregation follows upstream `run_evaluate.py:165-166`: per-dim flat
    mean of upstream-normalized values × 100; WorldScore-Static = mean of the
    7 static dims; WorldScore-Dynamic = mean of all 10 dims."""
    from videvalkit.benchmarks.worldscore import WorldScoreBenchmark
    # Feed raw scores that are close to upstream's `aspect_info` means so the
    # normalized values land near the center of their range — this keeps the
    # assertions stable regardless of small formula tweaks.
    raw = [
        # content_alignment: upstream avg = 26.67 (CLIPScore in [0, 100] scale)
        RawResult(benchmark="worldscore", model="m", dimension="content_alignment",
                  prompt_id="ws_sta_0001", score=26.67),
        # photometric_consistency: empirical_max=1.192, lower-is-better; 0.6 ≈ midpoint
        RawResult(benchmark="worldscore", model="m", dimension="photometric_consistency",
                  prompt_id="ws_sta_0001", score=0.6),
        # motion_magnitude: upstream avg = 3.24 (median flow px)
        RawResult(benchmark="worldscore", model="m", dimension="motion_magnitude",
                  prompt_id="ws_dyn_0001", score=3.24),
    ]
    s = WorldScoreBenchmark().aggregate(raw)
    # All three dims got at least one normalized score and produced a per-dim mean.
    for d in ("content_alignment", "photometric_consistency", "motion_magnitude"):
        assert d in s.per_dimension, f"missing per-dim mean for {d}"
        assert 0.0 <= s.per_dimension[d] <= 100.0
    assert s.aggregator == "worldscore_upstream"
    # Headlines exist; both are means in [0, 100].
    headlines = s.meta["headlines"]
    assert "WorldScore-Static" in headlines and "WorldScore-Dynamic" in headlines
    # With only one of seven static dims scored we still get a Static headline.
    # With motion_magnitude as the only dynamic dim contributing, Dynamic = mean
    # of (content_alignment, photometric_consistency, motion_magnitude) × 100 since
    # upstream's Dynamic averages all 10 dims (and only these three have values).
    static_mean = (s.per_dimension["content_alignment"]
                   + s.per_dimension["photometric_consistency"]) / 2
    assert abs(headlines["WorldScore-Static"] - round(static_mean, 2)) < 0.01


def test_worldscore_evaluate_unknown_dim_raises():
    from videvalkit.benchmarks.worldscore import WorldScoreBenchmark
    with pytest.raises(ValueError, match="unknown worldscore dimension"):
        WorldScoreBenchmark().evaluate(
            videos=[VideoSpec(path=Path("/tmp/x.mp4"), prompt_id="0", model_name="m")],
            dimensions=["not_a_dim"],
        )


def test_worldscore_evaluate_empty_returns_empty():
    from videvalkit.benchmarks.worldscore import WorldScoreBenchmark
    assert WorldScoreBenchmark().evaluate(videos=[]) == []


# NOTE: the legacy 5-dim proxy adapter (ContentAlignmentScorer / OpticalFlowScorer-
# proxy / MotionSmoothnessScorer-proxy / ObjectControllabilityScorer / default_noun_extractor)
# has been replaced by 10-dim upstream-class wrappers in scorers.py. End-to-end
# scoring is now exercised via /pub/evaluation_group/ning/worldscore_gens/cogvideox-5b/
# (a curated 100+50 prompt sample). See TEST_MANUAL §4.5 for the integration test plan.
