"""Tests for `eval --gpus N` dimension sharding across parallel subprocesses."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from videvalkit.cli import main


@pytest.fixture
def cli_runner():
    return CliRunner()


def _videos(tmp_path: Path) -> Path:
    d = tmp_path / "videos" / "Wan5B"
    d.mkdir(parents=True)
    (d / "x-0.mp4").write_bytes(b"x")
    return tmp_path / "videos"


def test_gpus_shards_explicit_dimensions(cli_runner, tmp_path):
    """--dimensions A B C --gpus 0,1 → 2 subprocesses, each with a dim subset."""
    spawned = []

    def fake_popen(cmd, env=None, **_kwargs):
        spawned.append((cmd, env))
        proc = MagicMock()
        proc.wait.return_value = 0
        return proc

    with patch("subprocess.Popen", side_effect=fake_popen), \
         patch("videvalkit.runner.run") as mock_run:
        mock_run.return_value = {"summary": None, "raw_paths": [], "workspace": ""}
        result = cli_runner.invoke(main, [
            "eval", "--bench", "vbench",
            "--videos", str(_videos(tmp_path)),
            "--workspace", str(tmp_path / "ws"),
            "--models", "Wan5B",
            "--dimensions", "subject_consistency",
            "--dimensions", "background_consistency",
            "--dimensions", "temporal_flickering",
            "--gpus", "0,1",
        ])
    assert result.exit_code == 0, result.output

    # 2 shards spawned
    assert len(spawned) == 2

    # Each shard got CUDA_VISIBLE_DEVICES set
    gpus_used = {env["CUDA_VISIBLE_DEVICES"] for _, env in spawned}
    assert gpus_used == {"0", "1"}

    # Dimensions sharded across, all 3 covered, none doubled
    sharded_dims = []
    for cmd, _ in spawned:
        ds = [cmd[i + 1] for i, t in enumerate(cmd) if t == "--dimensions"]
        sharded_dims.append(ds)
    flat = sum(sharded_dims, [])
    assert sorted(flat) == [
        "background_consistency", "subject_consistency", "temporal_flickering"
    ]
    # Final aggregate pass after shards
    mock_run.assert_called_once()
    final_kwargs = mock_run.call_args.kwargs
    assert sorted(final_kwargs["dimensions"]) == sorted(flat)


def test_gpus_one_gpu_passes_through(cli_runner, tmp_path):
    """--gpus 0 (one GPU) is a noop: no subprocess fan-out, just the normal run()."""
    with patch("subprocess.Popen") as mock_popen, \
         patch("videvalkit.runner.run") as mock_run:
        mock_run.return_value = {"summary": None, "raw_paths": [], "workspace": ""}
        result = cli_runner.invoke(main, [
            "eval", "--bench", "vbench",
            "--videos", str(_videos(tmp_path)),
            "--workspace", str(tmp_path / "ws"),
            "--models", "Wan5B",
            "--dimensions", "subject_consistency",
            "--dimensions", "background_consistency",
            "--gpus", "0",
        ])
    assert result.exit_code == 0, result.output
    mock_popen.assert_not_called()
    mock_run.assert_called_once()


def test_no_gpus_unchanged(cli_runner, tmp_path):
    """Without --gpus, behaviour is unchanged: one in-process run()."""
    with patch("subprocess.Popen") as mock_popen, \
         patch("videvalkit.runner.run") as mock_run:
        mock_run.return_value = {"summary": None, "raw_paths": [], "workspace": ""}
        result = cli_runner.invoke(main, [
            "eval", "--bench", "vbench",
            "--videos", str(_videos(tmp_path)),
            "--workspace", str(tmp_path / "ws"),
            "--models", "Wan5B",
            "--dimensions", "subject_consistency",
        ])
    assert result.exit_code == 0, result.output
    mock_popen.assert_not_called()
    mock_run.assert_called_once()
