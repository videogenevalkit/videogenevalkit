"""Lift-out metric for WorldScore motion_magnitude (SEA-RAFT optical flow).

Per docs/VIDEO_METRICS_DESIGN.md §3.4. Distinct from vbench/dynamic-degree
[RAFT] — worldscore uses SEA-RAFT.

v0.2 ships the class shape; compute() wires to the worldscore runner which
needs the SEA-RAFT checkpoint staged. Until that's wired end-to-end [the
worldscore adapter's runner is more entangled than vbench's clean
dimension_list API], this raises NotImplementedError with a clear pointer.
The registry entry + capability tagging + bit-exact test scaffold are in
place so the metric is discoverable and the wiring is a contained follow-up.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class MotionMagnitudeResult(BaseModel):
    metric: str = "motion-magnitude"
    score: float
    per_video: dict[str, float] = Field(default_factory=dict)
    n_videos: int
    source: str = "worldscore"


class MotionMagnitude:
    name = "motion-magnitude"
    requires_judge = False

    def compute(
        self,
        videos: list[Path] | Path,
        device: str = "auto",
    ) -> MotionMagnitudeResult:
        raise NotImplementedError(
            "motion-magnitude lift awaits worldscore SEA-RAFT runner wiring "
            "[v0.2 follow-up]. The worldscore adapter's runner is more "
            "entangled than vbench's clean dimension_list API; lifting it "
            "cleanly is a contained follow-up. Registry entry + tags + "
            "bit-exact scaffold are in place."
        )
