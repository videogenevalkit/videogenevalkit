"""Tests for distribution-level metrics shell.

Per docs/VIDEO_METRICS_DESIGN.md §4 / §6 / §9.

v0.2 ships:
  * Class shapes + registry entries for FVD / VFID / KVD / CLIP-FVD
  * Reproducibility plumbing
  * Sample-size guards
  * **NOT** the actual backbones [follow-up PR]

So tests here verify the SHELL [registration + sample-size guard + error msg
quality], not the actual scoring. Paper-alignment tests land with backbone PR.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from videvalkit.core.distribution_metric import (
    BaseDistributionMetric,
    DistributionMetricResult,
)
from videvalkit.metrics import (
    SUPPORTED_METRICS,
    get_metric,
    list_metrics,
    metric_info,
)


# ----------------------------------------------------- registry shape ---
class TestRegistry:
    def test_4_distribution_metrics_registered(self):
        names = list_metrics(kind="distribution_reference")
        assert set(names) == {"fvd", "vfid", "kvd", "clip-fvd"}

    def test_all_distribution_metrics_are_judge_free(self):
        for name in list_metrics(kind="distribution_reference"):
            cfg = SUPPORTED_METRICS[name]
            assert cfg["needs_judge"] is False
            assert cfg["compute_kind"] == "local_vision"

    def test_all_distribution_have_realism_distribution_tag(self):
        for name in list_metrics(kind="distribution_reference"):
            cfg = SUPPORTED_METRICS[name]
            assert "realism.distribution" in cfg["tags"]

    def test_clip_fvd_marked_experimental(self):
        assert SUPPORTED_METRICS["clip-fvd"].get("experimental", False) is True

    def test_metric_info_returns_copy(self):
        info = metric_info("fvd")
        assert info["canonical_backbone"] == "i3d-k400"
        # Mutation doesn't affect registry
        info["foo"] = "bar"
        assert "foo" not in SUPPORTED_METRICS["fvd"]


# ----------------------------------------------------- get_metric factory ---
class TestGetMetric:
    def test_get_fvd_returns_instance(self):
        m = get_metric("fvd")
        assert isinstance(m, BaseDistributionMetric)
        assert m.name == "fvd"
        assert m.canonical_backbone == "i3d-k400"

    def test_get_unknown_raises_with_suggestion(self):
        with pytest.raises(KeyError, match="unknown metric"):
            get_metric("fdv")  # typo

    def test_get_close_match_suggestion(self):
        try:
            get_metric("fdv")
        except KeyError as e:
            assert "fvd" in str(e)


# ----------------------------------------------------- list_metrics ---
class TestListMetrics:
    def test_list_all(self):
        names = list_metrics()
        assert len(names) >= 4

    def test_filter_by_no_judge(self):
        names = list_metrics(no_judge=True)
        # All current registered metrics are judge-free
        assert "fvd" in names

    def test_filter_by_kind(self):
        dist = list_metrics(kind="distribution_reference")
        per_prompt = list_metrics(kind="per_prompt_reference_free")
        assert "fvd" in dist
        assert "fvd" not in per_prompt

    def test_filter_by_source(self):
        styl = list_metrics(source="canonical/stylegan-v-port")
        assert "fvd" in styl


# ----------------------------------------------------- sample-size guard ---
class TestSampleSizeGuard:
    def test_above_paper_threshold_returns_none(self):
        out = BaseDistributionMetric.check_sample_size(n_gen=2048)
        assert out is None

    def test_between_500_and_2048_returns_info(self):
        out = BaseDistributionMetric.check_sample_size(n_gen=1000)
        assert out is not None
        assert "below paper-canonical" in out

    def test_between_100_and_500_returns_warn(self):
        out = BaseDistributionMetric.check_sample_size(n_gen=200)
        assert out is not None
        assert "WARN" in out

    def test_below_minimum_returns_error(self):
        out = BaseDistributionMetric.check_sample_size(n_gen=50, min_recommended=100)
        assert out is not None
        assert "ERROR" in out


# ----------------------------------------------------- compute() shell ---
class TestComputeShell:
    """Compute() raises NotImplementedError pointing at backbone-fetch.
    Sample-size guards fire BEFORE the NotImplementedError [shell first]."""

    def test_fvd_compute_with_50_videos_raises_size_error_first(self, tmp_path):
        m = get_metric("fvd")
        with pytest.raises(ValueError, match="ERROR.*below minimum"):
            m.compute(
                gen_videos=[tmp_path / f"v{i}.mp4" for i in range(50)],
                ref_videos=[tmp_path / f"r{i}.mp4" for i in range(2048)],
            )

    def test_fvd_compute_with_200_videos_reaches_not_implemented(self, tmp_path):
        """200 videos > min_recommended (100), so we get past the size guard
        and hit the NotImplementedError from the missing backbone."""
        m = get_metric("fvd")
        with pytest.raises(NotImplementedError, match="backbone fetch|checkpoint"):
            m.compute(
                gen_videos=[tmp_path / f"v{i}.mp4" for i in range(200)],
                ref_videos=[tmp_path / f"r{i}.mp4" for i in range(2048)],
            )

    def test_fvd_unsupported_backbone_rejected(self, tmp_path):
        m = get_metric("fvd")
        with pytest.raises(ValueError, match="not supported for FVD"):
            m.compute(
                gen_videos=[tmp_path / f"v{i}.mp4" for i in range(200)],
                ref_videos=[tmp_path / f"r{i}.mp4" for i in range(200)],
                backbone="not-a-backbone",
            )

    def test_kvd_lower_min_recommended(self, tmp_path):
        """KVD allows smaller N (50) than FVD (100)."""
        m = get_metric("kvd")
        # 60 videos: above KVD threshold, but below FVD's 100
        with pytest.raises(NotImplementedError):  # past size guard
            m.compute(
                gen_videos=[tmp_path / f"v{i}.mp4" for i in range(60)],
                ref_videos=[tmp_path / f"r{i}.mp4" for i in range(200)],
            )
