"""Tests for capability-axis evaluation [T5].

The orchestration [resolve → run → normalize → aggregate] is tested with
mocked metric.compute() so we don't need GPU/checkpoints.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from videvalkit.cli import main
from videvalkit.core.capability_eval import (
    CapabilityEvalResult,
    _minmax_normalize,
    run_capability,
)


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def videos_dir(tmp_path: Path) -> Path:
    d = tmp_path / "videos"
    d.mkdir()
    for i in range(3):
        (d / f"v{i}.mp4").write_bytes(b"fake")
    return d


# ----------------------------------------------------- normalize ---
class TestNormalize:
    def test_in_range(self):
        assert _minmax_normalize(0.5, 0.0, 1.0) == 0.5

    def test_clamps_high(self):
        assert _minmax_normalize(2.0, 0.0, 1.0) == 1.0

    def test_clamps_low(self):
        assert _minmax_normalize(-1.0, 0.0, 1.0) == 0.0

    def test_custom_range(self):
        # 56.4 in [0,100] → 0.564
        assert abs(_minmax_normalize(56.4, 0.0, 100.0) - 0.564) < 1e-9

    def test_degenerate_range(self):
        assert _minmax_normalize(5.0, 3.0, 3.0) == 0.0


# ----------------------------------------------------- run_capability ---
class TestRunCapability:
    def test_runs_per_video_metrics(self, videos_dir):
        """motion.smoothness resolves to motion-smoothness metric [per_video].
        Mock its compute to return a fixed score."""

        class FakeResult:
            score = 0.9

        def fake_get_metric(name):
            class FakeMetric:
                def compute(self, videos, device="auto"):
                    return FakeResult()
            return FakeMetric()

        with patch("videvalkit.metrics.get_metric", new=fake_get_metric):
            result = run_capability("motion.smoothness", videos_dir)
        assert isinstance(result, CapabilityEvalResult)
        assert result.n_contributors_run >= 1
        # score should be the normalized 0.9
        assert abs(result.score - 0.9) < 1e-6

    def test_shell_metric_skipped(self, videos_dir):
        """A metric whose compute raises NotImplementedError is skipped."""

        def fake_get_metric(name):
            class ShellMetric:
                def compute(self, videos, device="auto"):
                    raise NotImplementedError("backbone fetch pending")
            return ShellMetric()

        with patch("videvalkit.metrics.get_metric", new=fake_get_metric):
            result = run_capability("motion.smoothness", videos_dir)
        assert result.n_contributors_run == 0
        assert result.n_skipped >= 1
        assert any("shell" in (c.skip_reason or "") for c in result.contributors)

    def test_per_prompt_metric_skipped(self, videos_dir):
        """align.text2video resolves to clip-score [per_prompt] — needs prompts,
        so capability eval skips it with a clear reason."""
        result = run_capability("align.text2video", videos_dir)
        # clip-score is per_prompt_reference_free → skipped
        skipped = [c for c in result.contributors if c.status == "skipped"]
        assert any("needs refs/prompts/judge" in (c.skip_reason or "")
                   for c in skipped)

    def test_aggregator_mean(self, videos_dir):
        """Two metrics with scores 0.8 and 0.6 → mean 0.7."""
        scores = iter([0.8, 0.6])

        def fake_get_metric(name):
            s = next(scores)

            class M:
                def compute(self, videos, device="auto"):
                    class R:
                        score = s
                    return R()
            return M()

        # motion [top-level] expands to multiple sub-tags; restrict to a tag
        # with exactly the metrics we control. Use motion.magnitude which has
        # dynamic-degree [metric] + motion-magnitude [shell metric].
        # Simpler: patch resolve_capability to return 2 fake metric contributors.
        from videvalkit.core.capability import Contributor

        fake_contributors = [
            Contributor(source_kind="metric", source_name="m1",
                        canonical_source="c1", tags=["motion.smoothness"], cfg={}),
            Contributor(source_kind="metric", source_name="m2",
                        canonical_source="c2", tags=["motion.smoothness"], cfg={}),
        ]
        fake_registry = {
            "m1": {"kind": "per_video_reference_free"},
            "m2": {"kind": "per_video_reference_free"},
        }
        with patch("videvalkit.core.capability.resolve_capability",
                   return_value=fake_contributors), \
             patch("videvalkit.metrics.SUPPORTED_METRICS", fake_registry), \
             patch("videvalkit.metrics.get_metric", new=fake_get_metric):
            result = run_capability("motion.smoothness", videos_dir,
                                    aggregator="mean")
        assert abs(result.score - 0.7) < 1e-6

    def test_unknown_aggregator_raises(self, videos_dir):
        def fake_get_metric(name):
            class M:
                def compute(self, videos, device="auto"):
                    class R:
                        score = 0.5
                    return R()
            return M()
        with patch("videvalkit.metrics.get_metric", new=fake_get_metric):
            with pytest.raises(ValueError, match="unknown capability aggregator"):
                run_capability("motion.smoothness", videos_dir,
                               aggregator="median")


# ----------------------------------------------------- CLI ---
class TestCapabilityEvalCLI:
    def test_cli_runs(self, cli_runner, videos_dir):
        def fake_get_metric(name):
            class M:
                def compute(self, videos, device="auto"):
                    class R:
                        score = 0.85
                    return R()
            return M()

        with patch("videvalkit.metrics.get_metric", new=fake_get_metric):
            result = cli_runner.invoke(main, [
                "capabilities", "eval", "motion.smoothness",
                "--videos", str(videos_dir),
            ])
        assert result.exit_code == 0, result.output
        assert "capability: motion.smoothness" in result.output
        assert "score:" in result.output

    def test_cli_unknown_capability(self, cli_runner, videos_dir):
        result = cli_runner.invoke(main, [
            "capabilities", "eval", "not-a-tag",
            "--videos", str(videos_dir),
        ])
        assert result.exit_code != 0
