"""CLI integration tests for ad-hoc --judge-endpoint flags.

Per docs/JUDGE_SELECTION_DESIGN.md §5.2:
  videvalkit eval --bench X \\
      --judge-endpoint http://x.x.x.x:8003/v1 \\
      --judge-model my-model \\
      --judge-kind openai_compatible \\
      --judge-api-key-env MY_KEY

Tested via click.testing.CliRunner so we don't need to actually run a bench;
we patch runner.run to capture the resolved judge_override.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
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
    (d / "dummy_model").mkdir()
    return d


@pytest.fixture
def workspace_dir(tmp_path: Path) -> Path:
    return tmp_path / "ws"


@pytest.fixture
def captured_run_kwargs():
    """Patch runner.run; capture kwargs without actually running anything."""
    captured: dict[str, Any] = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"summary": None, "raw_paths": [], "workspace": str(kwargs["workspace"])}

    with patch("videvalkit.cli.run", new=fake_run, create=True):
        # cli.eval_cmd uses `from videvalkit.runner import run` inside the func,
        # so patch that import path instead:
        with patch("videvalkit.runner.run", new=fake_run):
            yield captured


# ------------------------------------------------- happy path: full ad-hoc ---
class TestAdHocHappy:
    def test_minimal_ad_hoc(
        self, cli_runner, videos_dir, workspace_dir, captured_run_kwargs
    ):
        result = cli_runner.invoke(main, [
            "eval", "--bench", "worldjen",
            "--videos", str(videos_dir),
            "--workspace", str(workspace_dir),
            "--judge-endpoint", "http://10.0.0.5:8003/v1",
            "--judge-model", "google/gemma-4-31b-it",
        ])
        assert result.exit_code == 0, result.output
        override = captured_run_kwargs.get("judge_override")
        assert override is not None
        assert override["endpoint"] == "http://10.0.0.5:8003/v1"
        assert override["model"] == "google/gemma-4-31b-it"
        # Defaults
        assert override["kind"] == "openai_compatible"
        assert override["provider"] == "adhoc"
        assert override["api_key_env"] is None

    def test_with_kind_and_api_key(
        self, cli_runner, videos_dir, workspace_dir, captured_run_kwargs
    ):
        result = cli_runner.invoke(main, [
            "eval", "--bench", "worldjen",
            "--videos", str(videos_dir),
            "--workspace", str(workspace_dir),
            "--judge-endpoint", "https://api.deepseek.com/v1",
            "--judge-model", "deepseek-v3",
            "--judge-kind", "openai_compatible",
            "--judge-api-key-env", "DEEPSEEK_API_KEY",
        ])
        assert result.exit_code == 0, result.output
        override = captured_run_kwargs["judge_override"]
        assert override["kind"] == "openai_compatible"
        assert override["api_key_env"] == "DEEPSEEK_API_KEY"

    def test_no_judge_passed_when_adhoc(
        self, cli_runner, videos_dir, workspace_dir, captured_run_kwargs
    ):
        cli_runner.invoke(main, [
            "eval", "--bench", "worldjen",
            "--videos", str(videos_dir),
            "--workspace", str(workspace_dir),
            "--judge-endpoint", "http://x/v1",
            "--judge-model", "m",
        ])
        assert captured_run_kwargs.get("judge") is None


# -------------------------------------------- error paths: mutually exclusive ---
class TestMutualExclusion:
    def test_judge_and_endpoint_conflict(
        self, cli_runner, videos_dir, workspace_dir
    ):
        result = cli_runner.invoke(main, [
            "eval", "--bench", "worldjen",
            "--videos", str(videos_dir),
            "--workspace", str(workspace_dir),
            "--judge", "claude-sonnet-4-6",
            "--judge-endpoint", "http://x/v1",
            "--judge-model", "m",
        ])
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output.lower()

    def test_endpoint_without_model_fails(
        self, cli_runner, videos_dir, workspace_dir
    ):
        result = cli_runner.invoke(main, [
            "eval", "--bench", "worldjen",
            "--videos", str(videos_dir),
            "--workspace", str(workspace_dir),
            "--judge-endpoint", "http://x/v1",
        ])
        assert result.exit_code != 0
        assert "must both be supplied" in result.output.lower()

    def test_model_without_endpoint_fails(
        self, cli_runner, videos_dir, workspace_dir
    ):
        result = cli_runner.invoke(main, [
            "eval", "--bench", "worldjen",
            "--videos", str(videos_dir),
            "--workspace", str(workspace_dir),
            "--judge-model", "x",
        ])
        assert result.exit_code != 0


# -------------------------------------------- runner.run accepts override ---
class TestRunnerOverride:
    def test_run_signature_has_judge_override(self):
        import inspect
        from videvalkit.runner import run
        sig = inspect.signature(run)
        assert "judge_override" in sig.parameters

    def test_resolve_via_override_in_run(self, monkeypatch):
        """End-to-end: runner.run with judge_override should produce a payload
        carrying the override dict as the judge cfg (bypass registry)."""
        captured: dict[str, Any] = {}

        # Stub scheduler so we don't actually try to dispatch.
        from videvalkit.scheduler import Scheduler

        def fake_run_in_env(self, env_name, benchmark, method, payload):
            captured.update(payload)
            return {"summary": {}, "raw_paths": []}

        monkeypatch.setattr(Scheduler, "run_in_env", fake_run_in_env)
        monkeypatch.setattr(Scheduler, "close", lambda self: None)

        from videvalkit.runner import run
        adhoc = {
            "kind": "openai_compatible",
            "endpoint": "http://override/v1",
            "model": "override-model",
            "provider": "adhoc",
            "api_key_env": None,
        }
        run(
            benchmark="worldjen",
            videos="/tmp",
            workspace="/tmp/ws",
            judge_override=adhoc,
        )
        assert captured["judge"] is adhoc
