"""Lift-out metrics for T2V-CompBench CV dimensions (judge-free).

Per docs/VIDEO_METRICS_DESIGN.md §4 (user 2026-05-20 confirmed).

These wrap the SAME upstream scorer classes the t2vcompbench benchmark adapter
uses [GenerativeNumeracyScorer / SpatialRelationshipsScorer], so the bit-exact
contract holds by construction:

    eval --bench t2vcompbench --dim generative_numeracy
                       ==  [≤ 1e-6]  ==
    metric --name numeracy

Both lifts are GroundingDINO-based [CV], NOT MLLM-judge based → judge-free.
The prompt→meta parsing reuses the upstream pure functions
[_parse_count_noun / _parse_spatial_triple], so a standalone metric needs
only (videos, prompts) — no separate LLM extraction pass.

  numeracy             ← GenerativeNumeracyScorer  [object count vs prompt]
  spatial-relationship ← SpatialRelationshipsScorer [GroundingDINO + depth bbox]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class T2VCompDimResult(BaseModel):
    metric: str
    dim: str
    score: float                        # mean across (video, prompt) pairs
    per_pair: list[dict[str, Any]] = Field(default_factory=list)
    n_pairs: int
    source: str = "t2vcompbench"


def _extract_frames(video_path: Path, n_frames: int = 8) -> list:
    """Uniformly sample n_frames as PIL.Images."""
    import decord
    import numpy as np
    from PIL import Image

    vr = decord.VideoReader(str(video_path))
    total = len(vr)
    if total == 0:
        return []
    idxs = np.linspace(0, total - 1, n_frames, dtype=int).tolist()
    frames = vr.get_batch(idxs).asnumpy()
    return [Image.fromarray(f) for f in frames]


class _T2VCompLift:
    """Common scaffolding for t2vcompbench CV lift metrics."""

    name = "t2vcomp-base"
    dim = ""
    requires_judge = False

    def __init__(self, device: str = "auto"):
        self._requested_device = device
        self._scorer = None

    def _resolve_device(self) -> str:
        if self._requested_device != "auto":
            return self._requested_device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _build_meta(self, prompt: str) -> dict[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError

    def _ensure_scorer(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def compute(
        self,
        videos: list[Path],
        prompts: list[str | dict],
        n_frames: int = 8,
    ) -> T2VCompDimResult:
        if len(videos) != len(prompts):
            raise ValueError(
                f"videos and prompts must match in length, "
                f"got {len(videos)} vs {len(prompts)}"
            )
        norm_prompts = [
            (p.get("text", "") if isinstance(p, dict) else str(p)) for p in prompts
        ]
        self._ensure_scorer()

        per_pair: list[dict[str, Any]] = []
        scores: list[float] = []
        for video_path, prompt_text in zip(videos, norm_prompts):
            meta = self._build_meta(prompt_text)
            frames = _extract_frames(Path(video_path), n_frames=n_frames)
            if not frames:
                per_pair.append({"video": str(video_path), "score": 0.0,
                                 "skip": "no frames"})
                scores.append(0.0)
                continue
            result = self._scorer.score_one(frames, meta)
            score = float(result.get(self.dim, 0.0))
            per_pair.append({"video": str(video_path), "prompt": prompt_text,
                             "score": score})
            scores.append(score)

        overall = sum(scores) / len(scores) if scores else 0.0
        return T2VCompDimResult(
            metric=self.name, dim=self.dim, score=overall,
            per_pair=per_pair, n_pairs=len(scores),
        )


class Numeracy(_T2VCompLift):
    name = "numeracy"
    dim = "generative_numeracy"

    def _ensure_scorer(self):
        if self._scorer is None:
            from videvalkit.benchmarks.t2vcompbench.scorers import (
                GenerativeNumeracyScorer,
            )
            self._scorer = GenerativeNumeracyScorer(device=self._resolve_device())

    def _build_meta(self, prompt: str) -> dict[str, Any]:
        """Derive {objects, numbers} from prompt via upstream parser."""
        from videvalkit.benchmarks.t2vcompbench.scorers import _parse_count_noun
        pairs = _parse_count_noun(prompt)  # [(count, noun), ...]
        return {
            "objects": [noun for _count, noun in pairs],
            "numbers": [count for count, _noun in pairs],
        }


class SpatialRelationship(_T2VCompLift):
    name = "spatial-relationship"
    dim = "spatial_relationships"

    def _ensure_scorer(self):
        if self._scorer is None:
            from videvalkit.benchmarks.t2vcompbench.scorers import (
                SpatialRelationshipsScorer,
            )
            self._scorer = SpatialRelationshipsScorer(device=self._resolve_device())

    def _build_meta(self, prompt: str) -> dict[str, Any]:
        from videvalkit.benchmarks.t2vcompbench.scorers import _parse_spatial_triple
        triple = _parse_spatial_triple(prompt)  # (a, rel, b) | None
        if triple is None:
            return {"object_1": None, "spatial": None, "object_2": None}
        a, rel, b = triple
        return {"object_1": a, "spatial": rel, "object_2": b}
