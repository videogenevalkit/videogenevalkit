"""Tests for vbench/worldscore lift-out metrics.

Per docs/VIDEO_METRICS_DESIGN.md §3.3-§3.4 + §5.

v0.2 ships:
  * 7 vbench quality-axis lifts [functional given vbench checkpoints]
  * 1 worldscore motion-magnitude lift [shell]
  * Registry + capability tagging + class shapes

Structural tests [registry / mapping / class] run always. The actual
bit-exact contract test [run upstream both ways, assert ≤ 1e-6] needs vbench
checkpoints + GPU — marked needs_gpu, skipped in CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from videvalkit.metrics import SUPPORTED_METRICS, get_metric, list_metrics
from videvalkit.metrics.vbench_dim import (
    VBENCH_DIM_BY_METRIC,
    VBenchDimMetric,
)


GENERIC_LIFTS = [
    "aesthetic-quality", "imaging-quality", "motion-smoothness",
    "temporal-flickering", "subject-consistency", "background-consistency",
    "dynamic-degree", "motion-magnitude",
]


# ----------------------------------------------------- registry ---
class TestLiftRegistry:
    def test_8_lifts_registered(self):
        for name in GENERIC_LIFTS:
            assert name in SUPPORTED_METRICS, f"{name} not registered"

    def test_all_judge_free(self):
        for name in GENERIC_LIFTS:
            assert SUPPORTED_METRICS[name]["needs_judge"] is False

    def test_all_per_video_reference_free(self):
        for name in GENERIC_LIFTS:
            assert SUPPORTED_METRICS[name]["kind"] == "per_video_reference_free"

    def test_source_points_to_bench(self):
        # 7 vbench + 1 worldscore
        assert SUPPORTED_METRICS["motion-smoothness"]["source"].startswith("vbench/")
        assert SUPPORTED_METRICS["motion-magnitude"]["source"].startswith("worldscore/")

    def test_also_used_by_present(self):
        for name in GENERIC_LIFTS:
            assert "also_used_by" in SUPPORTED_METRICS[name]

    def test_tags_in_vocab(self):
        from videvalkit.configs.capability_taxonomy import ALL_TAGS
        for name in GENERIC_LIFTS:
            for t in SUPPORTED_METRICS[name]["tags"]:
                assert t in ALL_TAGS, f"{name} tag {t} not in vocab"


# ----------------------------------------------------- vbench dim mapping ---
class TestVBenchDimMapping:
    def test_7_dims_mapped(self):
        assert len(VBENCH_DIM_BY_METRIC) == 7

    def test_metric_name_to_underscore_dim(self):
        assert VBENCH_DIM_BY_METRIC["motion-smoothness"] == "motion_smoothness"
        assert VBENCH_DIM_BY_METRIC["subject-consistency"] == "subject_consistency"

    def test_get_metric_returns_correct_dim(self):
        m = get_metric("motion-smoothness")
        assert isinstance(m, VBenchDimMetric)
        assert m.dim == "motion_smoothness"

    def test_each_vbench_lift_instantiates(self):
        for name in VBENCH_DIM_BY_METRIC:
            m = get_metric(name)
            assert m.name == name
            assert m.dim == VBENCH_DIM_BY_METRIC[name]

    def test_invalid_dim_metric_rejected(self):
        with pytest.raises(ValueError, match="not a known vbench lift"):
            VBenchDimMetric(metric_name="not-a-real-dim")


# ----------------------------------------------------- capability wiring ---
class TestCapabilityWiring:
    def test_motion_smoothness_resolves_via_capability(self):
        """The lift metric should now contribute to motion.smoothness
        capability [in addition to the bench dim]."""
        from videvalkit.core.capability import resolve_capability
        contributors = resolve_capability("motion.smoothness")
        names = [c.source_name for c in contributors]
        # The metric [canonical] should dedup with vbench/motion_smoothness
        # bench-dim [same canonical source]. After dedup, metric preferred.
        assert any("motion-smoothness" in n for n in names)


# ----------------------------------------------------- worldscore shell ---
class TestWorldscoreShell:
    def test_motion_magnitude_shell_raises(self, tmp_path):
        m = get_metric("motion-magnitude")
        with pytest.raises(NotImplementedError, match="SEA-RAFT runner"):
            m.compute(videos=tmp_path)


# ----------------------------------------------------- bit-exact contract ---
# The real bit-exact test needs vbench checkpoints + GPU. Marked needs_gpu,
# skipped in CI. Documents the contract per VIDEO_METRICS_DESIGN §5.3.
@pytest.mark.needs_gpu
def test_motion_smoothness_bit_exact_lift(tmp_path):
    """eval --bench vbench --dim motion_smoothness == metric --name
    motion-smoothness, ≤ 1e-6, on the same videos.

    Requires vbench checkpoints + GPU + sample videos. Skipped in CI.
    """
    pytest.skip("needs vbench checkpoints + GPU + sample videos — run on server")
    # Sketch of the contract verification [to run on GPU server]:
    #   videos = [.../v1.mp4, .../v2.mp4, .../v3.mp4]
    #   metric = get_metric("motion-smoothness")
    #   metric_result = metric.compute(videos_dir)
    #   bench_result = run_via_bench("vbench", ["motion_smoothness"], videos_dir)
    #   assert abs(metric_result.score - bench_result.per_dim["motion_smoothness"]) < 1e-6
