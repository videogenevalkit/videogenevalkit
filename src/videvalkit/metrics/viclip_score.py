"""ViCLIP-Score — per-clip text-video CLIP cosine.

Per docs/VIDEO_METRICS_DESIGN.md §3.2. T2V-recommended over CLIP-Score
since ViCLIP is trained on video-text pairs (vs CLIP on image-text).

v0.2 ships class shape; full implementation needs ViCLIP checkpoint
[follow-up: feat/viclip-backbone-fetch].
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class ViCLIPScoreResult(BaseModel):
    metric: str = "viclip-score"
    score: float
    per_video: dict[str, float]
    n_pairs: int
    backbone: str = "ViCLIP-L"


class ViCLIPScore:
    """Per-clip ViCLIP cosine — text-video alignment over a 16-frame clip."""

    name = "viclip-score"
    backbone = "ViCLIP-L"
    requires_judge = False

    def compute(
        self,
        videos: list[Path],
        prompts: list[str | dict],
        device: str = "cuda",
        seed: int = 42,
    ) -> ViCLIPScoreResult:
        if len(videos) != len(prompts):
            raise ValueError(
                f"videos and prompts must have same length, "
                f"got {len(videos)} vs {len(prompts)}"
            )
        raise NotImplementedError(
            "viclip-score awaits ViCLIP backbone fetch [v0.2 follow-up]. "
            "Class shape and registry entry in place; backbone load pending."
        )
