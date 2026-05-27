"""Tests for --no-judge filter on `list benchmarks` and `eval`.

Per docs/JUDGE_SELECTION_DESIGN.md §5.3.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from videvalkit.cli import main
from videvalkit.configs import SUPPORTED_BENCHMARKS


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def videos_dir(tmp_path: Path) -> Path:
    d = tmp_path / "videos"
    d.mkdir()
    return d


@pytest.fixture
def workspace_dir(tmp_path: Path) -> Path:
    return tmp_path / "ws"


# ------------------------------------------------------- list benchmarks ---
class TestListFilter:
    def test_list_benchmarks_no_filter_shows_all(self, cli_runner):
        result = cli_runner.invoke(main, ["list", "benchmarks"])
        assert result.exit_code == 0
        # All registered benches should appear
        for name in SUPPORTED_BENCHMARKS:
            assert name in result.output

    def test_list_benchmarks_no_judge_excludes_judge_benches(self, cli_runner):
        result = cli_runner.invoke(main, ["list", "benchmarks", "--no-judge"])
        assert result.exit_code == 0
        for name, cfg in SUPPORTED_BENCHMARKS.items():
            if cfg.get("needs_judge", False):
                # judge-required bench should be filtered out
                # (only its name appears in the "filtered out" footer if needed)
                lines = [
                    line for line in result.output.splitlines()
                    if line.startswith(f"{name:<16}")
                ]
                assert not lines, f"{name} should be filtered but appeared"
            else:
                assert name in result.output, f"{name} should be visible"

    def test_list_benchmarks_no_judge_shows_skip_footer(self, cli_runner):
        result = cli_runner.invoke(main, ["list", "benchmarks", "--no-judge"])
        # At least 5 benches need judge per v0.2 registry
        n_needs_judge = sum(
            1 for c in SUPPORTED_BENCHMARKS.values()
            if c.get("needs_judge", False)
        )
        if n_needs_judge:
            assert "filtered out" in result.output

    def test_list_benchmarks_has_judge_column(self, cli_runner):
        result = cli_runner.invoke(main, ["list", "benchmarks"])
        # New "judge?" column should be in header
        assert "judge?" in result.output
        # VLM or — should appear in rows
        assert "VLM" in result.output or "—" in result.output


# ----------------------------------------------- eval --no-judge fail-fast ---
class TestEvalNoJudge:
    def test_no_judge_on_judge_bench_fails(
        self, cli_runner, videos_dir, workspace_dir
    ):
        # worldjen needs_judge=True
        result = cli_runner.invoke(main, [
            "eval", "--bench", "worldjen",
            "--videos", str(videos_dir),
            "--workspace", str(workspace_dir),
            "--no-judge",
        ])
        assert result.exit_code != 0
        assert "judge-free" in result.output.lower() or "cannot run" in result.output.lower()

    def test_no_judge_on_judge_free_bench_ok(
        self, cli_runner, videos_dir, workspace_dir
    ):
        """vbench has needs_judge=False, so --no-judge should pass the filter
        gate (the actual run may fail downstream for other reasons; we only
        check the gate)."""
        captured = {}

        def fake_run(**kwargs):
            captured.update(kwargs)
            return {"summary": None, "raw_paths": [], "workspace": str(kwargs["workspace"])}

        with patch("videvalkit.runner.run", new=fake_run):
            result = cli_runner.invoke(main, [
                "eval", "--bench", "vbench",
                "--videos", str(videos_dir),
                "--workspace", str(workspace_dir),
                "--no-judge",
            ])
        assert result.exit_code == 0, result.output

    def test_no_judge_mutex_with_judge(
        self, cli_runner, videos_dir, workspace_dir
    ):
        result = cli_runner.invoke(main, [
            "eval", "--bench", "vbench",
            "--videos", str(videos_dir),
            "--workspace", str(workspace_dir),
            "--no-judge",
            "--judge", "claude-sonnet-4-6",
        ])
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output.lower()

    def test_no_judge_mutex_with_endpoint(
        self, cli_runner, videos_dir, workspace_dir
    ):
        result = cli_runner.invoke(main, [
            "eval", "--bench", "vbench",
            "--videos", str(videos_dir),
            "--workspace", str(workspace_dir),
            "--no-judge",
            "--judge-endpoint", "http://x/v1",
            "--judge-model", "m",
        ])
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output.lower()


# --------------------------------------- registry hygiene: needs_judge present ---
def test_all_benches_have_needs_judge_field():
    """Every bench entry must declare needs_judge: bool. Used by --no-judge filter."""
    missing = [
        name for name, cfg in SUPPORTED_BENCHMARKS.items()
        if "needs_judge" not in cfg
    ]
    assert not missing, f"benches missing needs_judge field: {missing}"


def test_needs_judge_is_bool():
    bad = [
        name for name, cfg in SUPPORTED_BENCHMARKS.items()
        if not isinstance(cfg.get("needs_judge"), bool)
    ]
    assert not bad, f"benches with non-bool needs_judge: {bad}"
