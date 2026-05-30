"""Tests for runner auto-staging of external --videos into the workspace.

Bench adapters look for videos at <workspace>/videos/<model>/*.mp4. The runner
must symlink an external --videos tree into that path so adapters find them.
This regression caused silent-null evals (Wan5B vbench run, 2026-05-30).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from videvalkit.runner import run as runner_run


def _make_videos_tree(root: Path, model: str = "Wan5B", n: int = 3) -> Path:
    """Create <root>/<model>/g{i}-0.mp4 placeholder files."""
    d = root / model
    d.mkdir(parents=True)
    for i in range(n):
        (d / f"clip{i}-0.mp4").write_bytes(b"x")
    return root


def test_external_videos_dir_symlinked_into_workspace_per_model(tmp_path):
    """An external --videos tree with one model subdir lands at <ws>/videos/<model>/."""
    ext = _make_videos_tree(tmp_path / "ext", model="Wan5B", n=4)
    ws = tmp_path / "ws"

    captured = {}
    with patch("videvalkit.runner.Scheduler") as MS:
        MS.return_value.run_in_env.return_value = {"summary": None, "raw_paths": []}
        MS.return_value.close.return_value = None
        try:
            runner_run(
                benchmark="vbench",
                videos=str(ext),
                workspace=str(ws),
                models=["Wan5B"],
            )
        except KeyError:
            pass  # bench env resolution may need extra fixtures; staging happens first
        captured["call"] = MS.return_value.run_in_env.call_args

    staged = ws / "videos" / "Wan5B"
    assert staged.is_symlink(), f"expected symlink at {staged}"
    assert staged.resolve() == (ext / "Wan5B").resolve()
    # And the symlinked dir exposes the original videos.
    assert sorted(p.name for p in staged.iterdir()) == [
        "clip0-0.mp4", "clip1-0.mp4", "clip2-0.mp4", "clip3-0.mp4",
    ]


def test_flat_external_videos_dir_staged_under_model_name(tmp_path):
    """If --videos is a flat dir of *.mp4 with no model subdir, runner stages it
    under the first --models name (or 'default')."""
    ext = tmp_path / "flat"
    ext.mkdir()
    for i in range(3):
        (ext / f"v{i}-0.mp4").write_bytes(b"x")
    ws = tmp_path / "ws"

    with patch("videvalkit.runner.Scheduler") as MS:
        MS.return_value.run_in_env.return_value = {"summary": None, "raw_paths": []}
        MS.return_value.close.return_value = None
        try:
            runner_run(
                benchmark="vbench",
                videos=str(ext),
                workspace=str(ws),
                models=["MyModel"],
            )
        except KeyError:
            pass

    staged = ws / "videos" / "MyModel"
    assert staged.is_symlink()
    assert staged.resolve() == ext.resolve()


def test_videos_equal_workspace_videos_dir_is_noop(tmp_path):
    """When --videos already points at <ws>/videos/, no staging happens."""
    ws = tmp_path / "ws"
    ws_videos = ws / "videos"
    ws_videos.mkdir(parents=True)
    (ws_videos / "Wan5B").mkdir()
    (ws_videos / "Wan5B" / "x-0.mp4").write_bytes(b"x")

    with patch("videvalkit.runner.Scheduler") as MS:
        MS.return_value.run_in_env.return_value = {"summary": None, "raw_paths": []}
        MS.return_value.close.return_value = None
        try:
            runner_run(
                benchmark="vbench",
                videos=str(ws_videos),
                workspace=str(ws),
                models=["Wan5B"],
            )
        except KeyError:
            pass

    # The Wan5B dir is the original (not a new symlink).
    assert not (ws_videos / "Wan5B").is_symlink()
