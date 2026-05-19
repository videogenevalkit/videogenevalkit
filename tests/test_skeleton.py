"""Smoke tests — verify the skeleton imports cleanly and the wiring holds.

These run in the core env only; they don't exercise upstream benchmarks.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path


def test_imports():
    """All public surfaces import cleanly in the core env (no upstream deps)."""
    import videvalkit
    from videvalkit import (BaseAggregator, BaseBenchmark, BaseScorer, PromptItem,
                            RawResult, Summary, VideoSpec, WorkspaceLayout)
    from videvalkit.scheduler import Scheduler, SchedulerConfig
    from videvalkit.storage import ApiCallLogger, Workspace
    from videvalkit.configs import (SUPPORTED_AGGREGATORS, SUPPORTED_BENCHMARKS,
                                    SUPPORTED_JUDGES)

    assert {"vbench", "vbench2", "videobench", "worldjen",
            "t2vcompbench", "physics_iq", "vbench_pp", "v_reasonbench"} \
        <= set(SUPPORTED_BENCHMARKS)
    assert "gemma-4-31b-local" in SUPPORTED_JUDGES
    assert set(SUPPORTED_AGGREGATORS) >= {"weighted_sum", "phas", "bt"}
    assert videvalkit.__version__


def test_workspace_layout_bootstraps_directories():
    from videvalkit.storage import Workspace

    with tempfile.TemporaryDirectory() as td:
        ws = Workspace(td)
        L = ws.layout
        for sub in (L.videos_dir, L.prompts_dir, L.results_dir, L.api_logs_dir,
                    L.api_calls_dir, L.api_stats_dir, L.api_zips_dir):
            assert sub.is_dir(), sub


def test_api_call_logger_writes_in_expected_schema():
    """api_log writer matches video_eval/results/api_logs schema."""
    from videvalkit.storage import Workspace, ApiCallLogger

    with tempfile.TemporaryDirectory() as td:
        ws = Workspace(td)
        logger = ApiCallLogger(ws.layout, provider="google", model="gemma-4-31b-it")
        logger.log(
            request={"prompt_id": 0, "video_fn": "000.mp4", "model_name": "pangu3"},
            response={"usage": {"total_tokens": 4161}, "response": "stub"},
        )
        matches = list(ws.layout.api_calls_dir.rglob("*.jsonl"))
        assert len(matches) == 1
        rec = json.loads(matches[0].read_text().strip())
        assert set(rec.keys()) >= {"timestamp", "model", "user", "request", "response"}
        assert rec["model"] == "gemma-4-31b-it"
        assert rec["request"]["video_fn"] == "000.mp4"


def test_scheduler_routes_cpu_scorer_in_process():
    from videvalkit.core.scorer import BaseScorer, ScoreContext, ScoreResult
    from videvalkit.scheduler import Scheduler

    class FixedScorer(BaseScorer):
        name = "fixed"
        kind = "cpu"
        def score(self, ctx: ScoreContext) -> ScoreResult:
            return ScoreResult(score=0.42, meta={"dim": ctx.dimension})

    ctx = ScoreContext(video_path=Path("/tmp/fake.mp4"), prompt_text="a cat",
                       prompt_id="000", dimension="aesthetic_quality",
                       model_name="cogvideox")
    res = Scheduler().submit_scorer(FixedScorer(), ctx)
    assert res.score == 0.42


def test_weighted_sum_aggregator_basic():
    from videvalkit.aggregators.weighted_sum import WeightedSumAggregator
    from videvalkit.core.types import RawResult

    raw = [
        RawResult(benchmark="vbench", model="m1", dimension="d1", prompt_id="0", score=0.8),
        RawResult(benchmark="vbench", model="m1", dimension="d1", prompt_id="1", score=0.6),
        RawResult(benchmark="vbench", model="m1", dimension="d2", prompt_id="0", score=0.4),
    ]
    s = WeightedSumAggregator(weights={"d1": 2.0, "d2": 1.0}).aggregate(raw)
    assert abs(s.per_dimension["d1"] - 0.7) < 1e-9
    assert abs(s.per_dimension["d2"] - 0.4) < 1e-9
    assert abs(s.overall - 0.6) < 1e-9


def test_phas_aggregator_uses_hand_weights():
    """PHAS with the WorldJen hand-tuned weights collapses to a known value."""
    from videvalkit.aggregators.phas import PHASAggregator, WORLDJEN_HAND_WEIGHTS
    from videvalkit.core.types import RawResult

    # One prompt, all 16 dims scored at 4.0 → PHAS_base = 4.0; variance = 0 → no penalty.
    raw = []
    for d in WORLDJEN_HAND_WEIGHTS:
        raw.append(RawResult(benchmark="worldjen", model="m1", dimension=d,
                             prompt_id="000", score=4.0))
    s = PHASAggregator().aggregate(raw)
    assert abs(float(s.overall) - 4.0) < 1e-6
    assert s.n_prompts == 1
    assert set(s.per_dimension) == set(WORLDJEN_HAND_WEIGHTS)


def test_phas_variance_penalty():
    """When dim scores vary widely, the variance penalty bites."""
    from videvalkit.aggregators.phas import PHASAggregator, WORLDJEN_HAND_WEIGHTS
    from videvalkit.core.types import RawResult

    # Half the dims at 5.0, half at 1.0 → high variance → penalty applies.
    dims = list(WORLDJEN_HAND_WEIGHTS)
    raw = []
    for i, d in enumerate(dims):
        raw.append(RawResult(benchmark="worldjen", model="m1", dimension=d,
                             prompt_id="000", score=5.0 if i % 2 == 0 else 1.0))
    s = PHASAggregator().aggregate(raw)
    # base would be a weighted mean roughly around 3.0; penalty <= 0.3
    assert 0 < float(s.overall) <= 5.0
    base_no_penalty = PHASAggregator(var_penalty_mult=0, var_penalty_cap=0).aggregate(raw)
    assert float(base_no_penalty.overall) >= float(s.overall)


def test_bradley_terry_ranks_consistently():
    """A always beats B, B always beats C → A > B > C BT."""
    from videvalkit.aggregators.bt import compute_bradley_terry

    matchups = [("A", "B")] * 20 + [("B", "C")] * 20 + [("A", "C")] * 20
    bt = compute_bradley_terry(matchups)
    assert bt["A"] > bt["B"] > bt["C"]


def test_bt_aggregator_cross_models():
    """BradleyTerryAggregator.aggregate over multiple models computes cross stats."""
    from videvalkit.aggregators.bt import BradleyTerryAggregator
    from videvalkit.core.types import RawResult

    raw = []
    for pid in range(20):
        # A scores 5, B scores 3
        raw.append(RawResult(benchmark="vbench", model="A", dimension="d1",
                             prompt_id=str(pid), score=5.0))
        raw.append(RawResult(benchmark="vbench", model="B", dimension="d1",
                             prompt_id=str(pid), score=3.0))
    s = BradleyTerryAggregator(bootstrap=50).aggregate(raw)
    cross = s.meta["bt_cross_models"]
    assert cross["bt_point"]["A"] > cross["bt_point"]["B"]
    assert "bt_with_ci" in cross
    ci_a = cross["bt_with_ci"]["A"]
    tol = 1e-6
    assert ci_a["lower_95"] - tol <= ci_a["mean"] <= ci_a["upper_95"] + tol


def test_vlm_judge_skeleton_logs_through_api_logger():
    """OpenAICompatibleVLMJudge logs through ApiCallLogger when wired up."""
    from videvalkit.scorers.vlm_judge.openai_compatible import OpenAICompatibleVLMJudge
    from videvalkit.storage import Workspace, ApiCallLogger

    with tempfile.TemporaryDirectory() as td:
        ws = Workspace(td)
        logger = ApiCallLogger(ws.layout, provider="google", model="gemma-4-31b-it")
        judge = OpenAICompatibleVLMJudge(
            name="test", endpoint="http://localhost:8003/v1",
            model="google/gemma-4-31b-it", provider="google", logger=logger,
        )
        # Don't actually call the endpoint; just exercise the _log path directly.
        judge._log(
            request_meta={"kind": "multimodal", "video_path": "/tmp/x.mp4"},
            payload={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            response={"usage": {"total_tokens": 10}, "content": "ok"},
            raw={},
        )
        calls = list(ws.layout.api_calls_dir.rglob("*.jsonl"))
        assert len(calls) == 1
        rec = json.loads(calls[0].read_text().strip())
        assert rec["request"]["kind"] == "multimodal"


def test_rate_limit_token_bucket_blocks_then_releases():
    """TokenBucket.try_acquire respects the per-minute cap."""
    from videvalkit.scheduler.rate_limit import TokenBucket

    b = TokenBucket(rate_per_min=2)  # only 2 req allowed instantly
    assert b.try_acquire()
    assert b.try_acquire()
    assert not b.try_acquire(), "third call should be rate-limited"


def test_worldjen_dimensions_complete():
    """WorldJen exposes all 16 dimensions across 4 categories."""
    from videvalkit.benchmarks.worldjen.dimensions import (
        WORLDJEN_CATEGORIES, WORLDJEN_DIMENSIONS, WORLDJEN_DEFINITIONS,
        WORLDJEN_DIMENSION_MODES, WORLDJEN_FRAMES_PER_MODE,
    )
    from videvalkit.benchmarks.worldjen import WorldJenBenchmark

    assert len(WORLDJEN_DIMENSIONS) == 16
    assert set(WORLDJEN_DIMENSIONS) == set(WORLDJEN_DEFINITIONS)
    # every dim has a sampling mode
    assert all(d in WORLDJEN_DIMENSION_MODES for d in WORLDJEN_DIMENSIONS)
    # every category lists known dims
    flat = [d for ds in WORLDJEN_CATEGORIES.values() for d in ds]
    assert set(flat) == set(WORLDJEN_DIMENSIONS)
    # frames per mode covers used modes
    used = set(WORLDJEN_DIMENSION_MODES.values())
    assert used <= set(WORLDJEN_FRAMES_PER_MODE)
    # adapter class points at same set
    assert WorldJenBenchmark.dimensions == list(WORLDJEN_DIMENSIONS)


def test_vbench_dimensions_complete():
    """VBench adapter advertises the canonical 16 dims."""
    from videvalkit.benchmarks.vbench.benchmark import (
        VBenchBenchmark, VBENCH_DIMENSIONS, VBENCH_QUALITY_DIMS, VBENCH_SEMANTIC_DIMS,
    )
    assert len(VBENCH_DIMENSIONS) == 16
    assert len(VBENCH_QUALITY_DIMS) == 7
    assert len(VBENCH_SEMANTIC_DIMS) == 9
    assert VBenchBenchmark.dimensions == VBENCH_DIMENSIONS


def test_videobench_dimensions_complete():
    """Video-Bench exposes 9 dims."""
    from videvalkit.benchmarks.videobench.benchmark import VIDEOBENCH_DIMENSIONS
    assert len(VIDEOBENCH_DIMENSIONS) == 9


def test_entry_dispatch_unknown_benchmark_errors():
    from videvalkit.benchmarks import entry
    try:
        entry._load_adapter("nonexistent")
    except KeyError as e:
        assert "nonexistent" in str(e)
    else:
        raise AssertionError("expected KeyError")


def test_strip_code_fences():
    from videvalkit.utils.video import strip_code_fences
    assert strip_code_fences("hello") == "hello"
    assert strip_code_fences("```json\n{\"a\":1}\n```").strip() == '{"a":1}'
    assert strip_code_fences("```\nplain\n```").strip() == "plain"


def test_runner_validation_errors():
    """Runner rejects unknown benchmark / missing judge cleanly."""
    from videvalkit.runner import run
    try:
        run(benchmark="not_a_thing", videos="/tmp", workspace="/tmp/ws")
    except KeyError as e:
        assert "not_a_thing" in str(e)
    else:
        raise AssertionError("expected KeyError")
