"""Lift-out metric for WorldScore motion_magnitude (SEA-RAFT optical flow).

Per docs/VIDEO_METRICS_DESIGN.md §3.4. Distinct from vbench/dynamic-degree
[RAFT] — worldscore uses SEA-RAFT median flow magnitude.

This is a bit-exact lift: scoring goes through the same
``OpticalFlowScorer`` -> ``OpticalFlowMetric._compute_scores`` upstream call
that the worldscore benchmark's ``motion_magnitude`` dimension uses, and the
frame sampling mirrors ``runners/static_dims.dump_frames`` (49 evenly-spaced
PNG frames). It is therefore functional wherever the worldscore upstream is
staged (``$VIDEVALKIT_WORLDSCORE_ROOT`` / ``~/.cache/videvalkit/upstream/
WorldScore`` + SEA-RAFT weights) — the same precondition as running the bench.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

N_FRAMES_PER_VIDEO = 49  # mirrors worldscore runners/static_dims.N_FRAMES_PER_VIDEO


class MotionMagnitudeResult(BaseModel):
    metric: str = "motion-magnitude"
    score: float
    per_video: dict[str, float] = Field(default_factory=dict)
    n_videos: int
    source: str = "worldscore"
    backbone: str = "SEA-RAFT"


def _dump_frames(mp4_path: Path, tmp_dir: Path, n: int = N_FRAMES_PER_VIDEO) -> list[str]:
    """49 evenly-spaced PNG frames — mirrors static_dims.dump_frames for bit-exactness."""
    import imageio.v3 as iio
    import numpy as np
    from PIL import Image

    arr = iio.imread(str(mp4_path), plugin="pyav")
    if arr.shape[0] == 0:
        return []
    idx = np.linspace(0, arr.shape[0] - 1, min(n, arr.shape[0])).astype(int).tolist()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for i, j in enumerate(idx):
        p = tmp_dir / f"f{i:03d}.png"
        Image.fromarray(arr[j]).save(p)
        paths.append(str(p))
    return paths


class MotionMagnitude:
    """SEA-RAFT median optical-flow magnitude, lifted from worldscore."""

    name = "motion-magnitude"
    requires_judge = False

    def __init__(self) -> None:
        self._scorer = None

    def _ensure_scorer(self):
        if self._scorer is None:
            from videvalkit.benchmarks.worldscore.scorers import OpticalFlowScorer

            self._scorer = OpticalFlowScorer()
        return self._scorer

    def compute(
        self,
        videos: list[Path] | Path,
        device: str = "auto",
        n_frames: int = N_FRAMES_PER_VIDEO,
    ) -> MotionMagnitudeResult:
        vids = [videos] if isinstance(videos, (str, Path)) else list(videos)
        if not vids:
            return MotionMagnitudeResult(score=0.0, per_video={}, n_videos=0)

        # Resolve the requested device through the canonical helper. The
        # OpticalFlowScorer currently auto-detects internally, but calling
        # resolve_device here activates the torch_npu shim when device='npu'
        # so the scorer's hardcoded .cuda() calls redirect to Ascend.
        from videvalkit.core.device import resolve_device
        resolve_device(device)

        scorer = self._ensure_scorer()
        per_video: dict[str, float] = {}
        scores: list[float] = []
        for v in sorted(vids, key=str):
            v = Path(v)
            tmp = Path(tempfile.mkdtemp(prefix="vk_flow_"))
            try:
                paths = _dump_frames(v, tmp, n=n_frames)
                if len(paths) < 2:
                    per_video[str(v)] = 0.0
                    continue
                s = float(scorer.score(paths))
                per_video[str(v)] = s
                scores.append(s)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

        overall = sum(scores) / len(scores) if scores else 0.0
        return MotionMagnitudeResult(
            score=overall, per_video=per_video, n_videos=len(scores)
        )
