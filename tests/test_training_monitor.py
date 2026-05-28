"""Tests for videvalkit.training monitor + eval-suite + watch CLI."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from videvalkit.cli import main
from videvalkit.training import MonitorConfig, MonitorResult, monitor


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def videos_dir(tmp_path: Path) -> Path:
    d = tmp_path / "videos"
    d.mkdir()
    return d


# ============================================================
# MonitorConfig
# ============================================================
class TestMonitorConfig:
    def test_defaults(self):
        cfg = MonitorConfig()
        assert cfg.profile == "quick"
        assert cfg.benches == []

    def test_save_load_roundtrip(self, tmp_path):
        cfg = MonitorConfig(
            benches=["vbench", "worldjen"],
            metrics=["fvd"],
            profile="standard",
            workspace="/data/ws",
        )
        p = tmp_path / "monitor.yaml"
        cfg.save(p)
        loaded = MonitorConfig.load(p)
        assert loaded.benches == ["vbench", "worldjen"]
        assert loaded.profile == "standard"

    def test_extra_field_rejected(self):
        with pytest.raises(Exception):
            MonitorConfig(unknown_field="x")


# ============================================================
# monitor.preview_prompts
# ============================================================
class TestPreviewPrompts:
    def test_preview_dedups_across_benches(self):
        """preview_prompts returns prompts; dedup by text."""
        cfg = MonitorConfig(benches=["vbench"], profile="full")

        # Mock the bench class to return known prompts
        from videvalkit.core.types import PromptItem

        class FakeBench:
            def list_prompts(self, dimensions=None):
                yield PromptItem(prompt_id="p1", text="a cat")
                yield PromptItem(prompt_id="p2", text="a dog")

        with patch.dict(
            "videvalkit.configs.SUPPORTED_BENCHMARKS",
            {"vbench": {"cls": FakeBench, "needs_judge": False}},
        ):
            prompts = monitor.preview_prompts(cfg)
        texts = [p.text for p in prompts]
        assert "a cat" in texts
        assert "a dog" in texts

    def test_preview_unknown_bench_skipped(self, caplog):
        cfg = MonitorConfig(benches=["not-a-bench"], profile="full")
        prompts = monitor.preview_prompts(cfg)
        assert prompts == []


# ============================================================
# monitor.eval — timeline writing
# ============================================================
class TestMonitorEval:
    def test_eval_writes_timeline(self, tmp_path, videos_dir):
        cfg = MonitorConfig(
            benches=["vbench"], profile="quick", workspace=str(tmp_path / "ws"),
        )

        def fake_run(**kwargs):
            return {
                "summary": {"step_1000": {"overall": 0.75, "per_dimension": {}}},
                "raw_paths": [],
            }

        with patch("videvalkit.runner.run", new=fake_run):
            result = monitor.eval(
                videos_dir, model_name="step_1000", cfg=cfg, step=1000,
            )
        assert isinstance(result, MonitorResult)
        assert result.model_name == "step_1000"
        assert result.overall == 0.75

        timeline = tmp_path / "ws" / "timeline.jsonl"
        assert timeline.exists()
        lines = timeline.read_text().strip().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["model_name"] == "step_1000"
        assert rec["overall"] == 0.75

    def test_eval_appends_multiple_steps(self, tmp_path, videos_dir):
        cfg = MonitorConfig(
            benches=["vbench"], profile="quick", workspace=str(tmp_path / "ws"),
        )

        def fake_run(**kwargs):
            return {"summary": {"m": {"overall": 0.5}}, "raw_paths": []}

        with patch("videvalkit.runner.run", new=fake_run):
            monitor.eval(videos_dir, model_name="step_1000", cfg=cfg, step=1000)
            monitor.eval(videos_dir, model_name="step_2000", cfg=cfg, step=2000)

        timeline = tmp_path / "ws" / "timeline.jsonl"
        lines = timeline.read_text().strip().splitlines()
        assert len(lines) == 2

    def test_eval_bench_failure_captured(self, tmp_path, videos_dir):
        cfg = MonitorConfig(
            benches=["vbench"], profile="quick", workspace=str(tmp_path / "ws"),
        )

        def fake_run(**kwargs):
            raise RuntimeError("boom")

        with patch("videvalkit.runner.run", new=fake_run):
            result = monitor.eval(videos_dir, model_name="m", cfg=cfg)
        # Failure captured, doesn't crash
        assert "error" in result.summary["vbench"]


# ============================================================
# eval-suite CLI
# ============================================================
class TestEvalSuite:
    @pytest.fixture
    def captured_runs(self):
        calls = []

        def fake_run(**kwargs):
            calls.append(kwargs["benchmark"])
            return {"summary": {}, "raw_paths": [], "workspace": "x"}

        with patch("videvalkit.runner.run", new=fake_run):
            yield calls

    def test_multi_bench(self, cli_runner, videos_dir, tmp_path, captured_runs):
        result = cli_runner.invoke(main, [
            "eval-suite",
            "--bench", "vbench", "--bench", "worldscore",
            "--videos", str(videos_dir),
            "--workspace", str(tmp_path / "ws"),
            "--profile", "quick",
        ])
        assert result.exit_code == 0, result.output
        assert "vbench" in captured_runs
        assert "worldscore" in captured_runs

    def test_all_anchored(self, cli_runner, videos_dir, tmp_path, captured_runs):
        result = cli_runner.invoke(main, [
            "eval-suite", "--all-anchored",
            "--videos", str(videos_dir),
            "--workspace", str(tmp_path / "ws"),
        ])
        assert result.exit_code == 0
        assert len(captured_runs) == 6

    def test_no_judge_skips(self, cli_runner, videos_dir, tmp_path, captured_runs):
        result = cli_runner.invoke(main, [
            "eval-suite",
            "--bench", "vbench",       # judge-free
            "--bench", "worldjen",     # needs judge
            "--videos", str(videos_dir),
            "--workspace", str(tmp_path / "ws"),
            "--no-judge",
        ])
        assert result.exit_code == 0
        assert "vbench" in captured_runs
        assert "worldjen" not in captured_runs
        assert "Skipped" in result.output

    def test_no_bench_errors(self, cli_runner, videos_dir, tmp_path):
        result = cli_runner.invoke(main, [
            "eval-suite",
            "--videos", str(videos_dir),
            "--workspace", str(tmp_path / "ws"),
        ])
        assert result.exit_code != 0


# ============================================================
# watch CLI (--once mode)
# ============================================================
class TestWatch:
    def test_watch_once_processes_matches(
        self, cli_runner, tmp_path,
    ):
        # Create fake checkpoint sample dirs
        ckpt_root = tmp_path / "run" / "checkpoints"
        for step in (1000, 2000):
            d = ckpt_root / f"step_{step}" / "samples"
            d.mkdir(parents=True)

        captured = []

        def fake_eval(self, videos_dir, model_name, cfg, step=None):
            captured.append(model_name)
            return MonitorResult(model_name=model_name, overall=0.5)

        with patch("videvalkit.training._Monitor.eval", new=fake_eval):
            result = cli_runner.invoke(main, [
                "watch",
                "--videos-pattern", str(ckpt_root / "step_*" / "samples"),
                "--workspace", str(tmp_path / "ws"),
                "--bench", "vbench",
                "--once",
            ])
        assert result.exit_code == 0, result.output
        # model_name is the parent dir name of "samples" → "step_1000" etc.
        assert "step_1000" in captured
        assert "step_2000" in captured

    def test_watch_once_no_matches(self, cli_runner, tmp_path):
        result = cli_runner.invoke(main, [
            "watch",
            "--videos-pattern", str(tmp_path / "nothing_*" / "samples"),
            "--workspace", str(tmp_path / "ws"),
            "--bench", "vbench",
            "--once",
        ])
        assert result.exit_code == 0
        assert "processed 0" in result.output
