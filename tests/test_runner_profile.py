"""Tests for runner.run() profile + subset wiring + `videvalkit estimate`."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from videvalkit.cli import main


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


@pytest.fixture
def captured_payload():
    """Patch Scheduler.run_in_env to capture the payload dispatched to the
    benchmark adapter."""
    captured = {}

    def fake_run_in_env(self, env_name, benchmark, method, payload):
        captured.update(payload)
        return {"summary": {}, "raw_paths": []}

    with patch("videvalkit.scheduler.Scheduler.run_in_env", new=fake_run_in_env):
        with patch("videvalkit.scheduler.Scheduler.close", new=lambda self: None):
            yield captured


# ============================================================
# runner.run() profile wiring
# ============================================================
class TestRunnerProfile:
    def test_no_profile_defaults_to_full(self, captured_payload):
        from videvalkit.runner import run
        run(benchmark="vbench", videos="/tmp/v", workspace="/tmp/ws")
        assert captured_payload["profile"]["name"] == "full"
        assert captured_payload["profile"]["samples_per_prompt"] == 5

    def test_profile_quick(self, captured_payload):
        from videvalkit.runner import run
        run(benchmark="vbench", videos="/tmp/v", workspace="/tmp/ws", profile="quick")
        assert captured_payload["profile"]["name"] == "quick"
        assert captured_payload["profile"]["frame_sampling"]["n_frames"] == 4

    def test_profile_unknown_raises(self, captured_payload):
        from videvalkit.runner import run
        with pytest.raises(KeyError, match="unknown profile"):
            run(benchmark="vbench", videos="/tmp/v", workspace="/tmp/ws", profile="quik")

    def test_subset_payload_contains_hash(self, tmp_path, captured_payload):
        from videvalkit.runner import run
        subset_path = tmp_path / "s.json"
        subset_path.write_text(json.dumps({
            "schema_version": 1,
            "subset_name": "my_subset",
            "benchmark": "vbench",
            "created": "2026-05-26",
            "n_prompts": 2,
            "prompt_ids": ["p1", "p2"],
        }))
        run(
            benchmark="vbench", videos="/tmp/v", workspace="/tmp/ws",
            subset_path=subset_path,
        )
        assert captured_payload["subset"] is not None
        assert captured_payload["subset"]["name"] == "my_subset"
        assert captured_payload["subset"]["n_prompts"] == 2
        assert len(captured_payload["subset"]["hash"]) == 64

    def test_profile_subset_lookup_missing_warns_not_fails(
        self, captured_payload, caplog,
    ):
        """When profile.subset is set but no subset_v1.json exists for the
        bench [v0.2 reality], we warn and continue with full corpus."""
        from videvalkit.runner import run
        import logging
        with caplog.at_level(logging.WARNING):
            run(
                benchmark="physics_iq", videos="/tmp/v", workspace="/tmp/ws",
                profile="quick",  # physics_iq has no shipped quick_v1.json
            )
        # subset_payload is None (no file found), but run succeeded
        assert captured_payload["subset"] is None
        assert any("no subset file found" in r.message.lower() or
                   "not found" in r.message.lower()
                   for r in caplog.records)


# ============================================================
# CLI --profile / --subset flags
# ============================================================
class TestCLIProfile:
    def test_cli_profile_flag(
        self, cli_runner, videos_dir, workspace_dir, captured_payload,
    ):
        result = cli_runner.invoke(main, [
            "eval", "--bench", "vbench",
            "--videos", str(videos_dir),
            "--workspace", str(workspace_dir),
            "--profile", "standard",
        ])
        assert result.exit_code == 0, result.output
        assert captured_payload["profile"]["name"] == "standard"

    def test_cli_subset_flag(
        self, tmp_path, cli_runner, videos_dir, workspace_dir, captured_payload,
    ):
        subset_path = tmp_path / "s.json"
        subset_path.write_text(json.dumps({
            "schema_version": 1,
            "subset_name": "tiny",
            "benchmark": "vbench",
            "created": "2026-05-26",
            "n_prompts": 1,
            "prompt_ids": ["only-this"],
        }))
        result = cli_runner.invoke(main, [
            "eval", "--bench", "vbench",
            "--videos", str(videos_dir),
            "--workspace", str(workspace_dir),
            "--subset", str(subset_path),
        ])
        assert result.exit_code == 0, result.output
        assert captured_payload["subset"]["name"] == "tiny"

    def test_cli_invalid_profile_rejected(
        self, cli_runner, videos_dir, workspace_dir,
    ):
        result = cli_runner.invoke(main, [
            "eval", "--bench", "vbench",
            "--videos", str(videos_dir),
            "--workspace", str(workspace_dir),
            "--profile", "fast",  # not in Choice
        ])
        assert result.exit_code != 0


# ============================================================
# CLI estimate command
# ============================================================
class TestEstimate:
    def test_estimate_single_bench_default(self, cli_runner):
        result = cli_runner.invoke(main, [
            "estimate", "--bench", "worldjen",
        ])
        assert result.exit_code == 0, result.output
        assert "worldjen" in result.output
        assert "TOTAL" in result.output
        assert "VLM" in result.output  # worldjen needs judge

    def test_estimate_multi_bench(self, cli_runner):
        result = cli_runner.invoke(main, [
            "estimate",
            "--bench", "vbench",
            "--bench", "worldjen",
            "--profile", "quick",
        ])
        assert result.exit_code == 0
        assert "vbench" in result.output
        assert "worldjen" in result.output

    def test_estimate_judge_free_bench(self, cli_runner):
        result = cli_runner.invoke(main, [
            "estimate", "--bench", "vbench", "--profile", "quick",
        ])
        assert result.exit_code == 0
        # vbench is judge-free → 0 judge calls expected
        # find the line for vbench
        lines = [
            l for l in result.output.splitlines()
            if l.strip().startswith("vbench ") or "vbench" in l[:20]
        ]
        # at least one such line
        assert lines

    def test_estimate_n_models(self, cli_runner):
        r1 = cli_runner.invoke(main, [
            "estimate", "--bench", "worldjen", "--profile", "quick", "--n-models", "1",
        ])
        r3 = cli_runner.invoke(main, [
            "estimate", "--bench", "worldjen", "--profile", "quick", "--n-models", "3",
        ])
        # 3 models should produce 3x judge calls
        # parse the TOTAL row — sloppy but works for this test
        import re
        def total_calls(out: str) -> int:
            for line in out.splitlines():
                if "TOTAL" in line:
                    m = re.search(r"(\d+)\s*$", line.strip())
                    if m:
                        return int(m.group(1))
            return -1
        c1 = total_calls(r1.output)
        c3 = total_calls(r3.output)
        assert c3 == c1 * 3

    def test_estimate_unknown_profile_rejected(self, cli_runner):
        result = cli_runner.invoke(main, [
            "estimate", "--bench", "vbench", "--profile", "fast",
        ])
        assert result.exit_code != 0
