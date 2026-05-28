"""Tests for `videvalkit capabilities list/show` CLI commands."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from videvalkit.cli import main


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


class TestCapabilitiesList:
    def test_lists_10_top_level(self, cli_runner):
        result = cli_runner.invoke(main, ["capabilities", "list"])
        assert result.exit_code == 0, result.output
        for top in ("motion", "visual_quality", "text_alignment",
                    "object_fidelity", "subject_consistency",
                    "physical_plausibility", "temporal_coherence",
                    "realism", "compositional", "style"):
            assert top in result.output

    def test_show_sub_expands(self, cli_runner):
        result = cli_runner.invoke(main, ["capabilities", "list", "--show-sub"])
        assert result.exit_code == 0
        # Sub-tags should appear with prefix
        assert "motion.smoothness" in result.output
        assert "comp.spatial" in result.output

    def test_has_contributor_counts(self, cli_runner):
        result = cli_runner.invoke(main, ["capabilities", "list"])
        # motion has at least 1 contributor (vbench/motion_smoothness)
        for line in result.output.splitlines():
            if line.startswith("motion "):
                # Last token should be a non-zero integer
                parts = line.split()
                assert int(parts[-1]) >= 1
                break


class TestCapabilitiesShow:
    def test_show_top_level(self, cli_runner):
        result = cli_runner.invoke(main, ["capabilities", "show", "motion"])
        assert result.exit_code == 0
        assert "motion" in result.output
        # Should describe + expand
        assert "expands to" in result.output

    def test_show_sub_tag(self, cli_runner):
        result = cli_runner.invoke(main, ["capabilities", "show", "motion.smoothness"])
        assert result.exit_code == 0
        # vbench/motion_smoothness is a contributor
        assert "vbench/motion_smoothness" in result.output

    def test_show_unknown_fails(self, cli_runner):
        result = cli_runner.invoke(main, ["capabilities", "show", "not-a-tag"])
        assert result.exit_code != 0
        assert "unknown capability" in result.output.lower()

    def test_show_real_distribution_lists_fvd_when_metrics_ready(
        self, cli_runner,
    ):
        """When metrics module has fvd registered with real.distribution
        tag, the capability resolver should pull it in. Verifies cross-module
        wiring."""
        result = cli_runner.invoke(main, ["capabilities", "show", "real.distribution"])
        assert result.exit_code == 0
        assert "fvd" in result.output
        assert "vfid" in result.output

    def test_show_align_text2video_lists_clip_score(self, cli_runner):
        result = cli_runner.invoke(main, ["capabilities", "show", "align.text2video"])
        assert result.exit_code == 0
        assert "clip-score" in result.output

    def test_show_with_no_contributors_yet(self, cli_runner):
        """A tag with no registered contributors should report cleanly,
        not crash."""
        # Pick a tag known to have no metric/bench-dim tagged yet
        result = cli_runner.invoke(main, ["capabilities", "show", "phys.gravity"])
        assert result.exit_code == 0
        # Either has contributors (physics_iq covers it) or shows the empty message
        assert "phys.gravity" in result.output
