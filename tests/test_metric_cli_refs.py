"""Tests for `videvalkit metric` + `videvalkit refs` CLI command groups."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from videvalkit.cli import main
from videvalkit.refs.registry import BUILTIN_REFS, get_refs, register_ref


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def videos_dir(tmp_path: Path) -> Path:
    d = tmp_path / "vids"; d.mkdir()
    for i in range(3):
        (d / f"v{i}.mp4").write_bytes(b"x")
    return d


# ============================================================
# metric CLI
# ============================================================
class TestMetricList:
    def test_lists_all(self, cli_runner):
        result = cli_runner.invoke(main, ["metric", "list"])
        assert result.exit_code == 0
        assert "fvd" in result.output
        assert "clip-score" in result.output

    def test_no_judge_filter(self, cli_runner):
        result = cli_runner.invoke(main, ["metric", "list", "--no-judge"])
        assert result.exit_code == 0
        assert "fvd" in result.output  # judge-free

    def test_kind_filter(self, cli_runner):
        result = cli_runner.invoke(main, ["metric", "list",
                                          "--kind", "distribution_reference"])
        assert result.exit_code == 0
        assert "fvd" in result.output
        assert "clip-score" not in result.output


class TestMetricShow:
    def test_show_fvd(self, cli_runner):
        result = cli_runner.invoke(main, ["metric", "show", "fvd"])
        assert result.exit_code == 0
        assert "canonical_backbone" in result.output
        assert "i3d-k400" in result.output

    def test_show_unknown(self, cli_runner):
        result = cli_runner.invoke(main, ["metric", "show", "nope"])
        assert result.exit_code != 0


class TestMetricRun:
    def test_run_per_video_metric(self, cli_runner, videos_dir):
        """Run a per-video metric with a mocked compute."""
        class FakeResult:
            def model_dump(self):
                return {"metric": "motion-smoothness", "score": 0.9}

        def fake_get_metric(name):
            class M:
                def compute(self, videos):
                    return FakeResult()
            return M()

        with patch("videvalkit.metrics.get_metric", new=fake_get_metric):
            result = cli_runner.invoke(main, [
                "metric", "run", "--name", "motion-smoothness",
                "--videos", str(videos_dir),
            ])
        assert result.exit_code == 0, result.output
        assert "motion-smoothness" in result.output

    def test_run_distribution_needs_ref(self, cli_runner, videos_dir):
        result = cli_runner.invoke(main, [
            "metric", "run", "--name", "fvd",
            "--gen-videos", str(videos_dir),
        ])
        assert result.exit_code != 0  # needs --ref-videos or --refs

    def test_run_shell_metric_reports_not_functional(self, cli_runner, videos_dir):
        """kvd is a shell → exit 3 with clear message."""
        result = cli_runner.invoke(main, [
            "metric", "run", "--name", "kvd",
            "--gen-videos", str(videos_dir),
            "--ref-videos", str(videos_dir),
            "--allow-tiny-sample",
        ])
        assert result.exit_code == 3
        assert "NOT YET FUNCTIONAL" in result.output

    def test_run_unknown_metric(self, cli_runner, videos_dir):
        result = cli_runner.invoke(main, [
            "metric", "run", "--name", "nope", "--videos", str(videos_dir),
        ])
        assert result.exit_code != 0


# ============================================================
# refs CLI + registry
# ============================================================
class TestRefsRegistry:
    def test_builtin_refs_present(self):
        refs = get_refs()
        assert "ucf101-fvd" in refs
        assert "msr-vtt-val" in refs

    def test_register_local_ref(self, tmp_path, monkeypatch):
        ref_dir = tmp_path / "myref"; ref_dir.mkdir()
        from videvalkit.refs import registry as reg
        monkeypatch.setattr(reg, "USER_REFS_YAML", tmp_path / "refs.yaml")
        reg.register_ref("my-ref", ref_dir, "test set")
        refs = reg.get_refs()
        # need to re-read with patched path
        loaded = reg.load_user_refs()
        assert "my-ref" in loaded


class TestRefsCLI:
    def test_refs_list(self, cli_runner):
        result = cli_runner.invoke(main, ["refs", "list"])
        assert result.exit_code == 0
        assert "ucf101-fvd" in result.output

    def test_refs_show(self, cli_runner):
        result = cli_runner.invoke(main, ["refs", "show", "ucf101-fvd"])
        assert result.exit_code == 0
        assert "n_clips" in result.output

    def test_refs_register(self, cli_runner, tmp_path, monkeypatch):
        ref_dir = tmp_path / "r"; ref_dir.mkdir()
        from videvalkit.refs import registry as reg
        monkeypatch.setattr(reg, "USER_REFS_YAML", tmp_path / "refs.yaml")
        result = cli_runner.invoke(main, [
            "refs", "register", "--name", "x", "--path", str(ref_dir),
        ])
        assert result.exit_code == 0, result.output
        assert "registered" in result.output
