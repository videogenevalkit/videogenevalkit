"""BaseDistributionMetric — abstract base for FVD-family metrics.

Per docs/VIDEO_METRICS_DESIGN.md §3 (user 2026-05-20 confirmed):

Distribution-level metrics differ from per-prompt metrics in:
  * Input: a SET of generated videos + a SET of reference videos
  * Output: a single scalar for the whole pair of sets
  * Backbone: I3D / VideoMAE / V-JEPA / CLIP-ViT etc.
  * Statistical sensitivity to sample size

This base class is kept SEPARATE from ``BaseScorer`` [which is per-video]
to avoid forcing a single signature on two structurally-different shapes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class DistributionMetricResult(BaseModel):
    """Result of one distribution-metric computation."""

    metric: str
    score: float
    n_gen: int
    n_ref: int
    backbone: str
    backbone_version: str = "unknown"
    clip_sampling: dict[str, Any] = Field(default_factory=dict)
    sample_size_warning: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class BaseDistributionMetric(ABC):
    """Abstract base for FVD-family metrics.

    Subclasses MUST set:
      * ``name``                  — short identifier
      * ``canonical_backbone``    — paper-canonical backbone name
      * ``supported_backbones``   — list of backbone aliases this metric can use
      * ``min_recommended_samples`` — N below which scores are unreliable

    Subclasses MUST implement:
      * ``compute(gen_videos, ref_videos, ...) -> DistributionMetricResult``
    """

    name: str = "base"
    canonical_backbone: str = ""
    supported_backbones: list[str] = []
    min_recommended_samples: int = 100
    requires_reference: bool = True

    @abstractmethod
    def compute(
        self,
        gen_videos: list[Path],
        ref_videos: list[Path] | None,
        backbone: str | None = None,
        clip_sampling: dict | None = None,
        seed: int = 42,
        device: str = "auto",
    ) -> DistributionMetricResult:  # pragma: no cover - abstract
        ...

    @staticmethod
    def check_sample_size(
        n_gen: int, min_recommended: int = 100,
    ) -> str | None:
        """Return a warning string if n_gen is below the recommended threshold.

        Per VIDEO_METRICS_DESIGN.md §9:
        - n_gen >= 2048  →  OK
        - 500 <= n_gen < 2048  →  INFO
        - 100 <= n_gen < 500   →  WARN
        - n_gen < 100          →  ERROR [caller should decide]
        """
        if n_gen >= 2048:
            return None
        if n_gen >= 500:
            return (
                f"n_gen={n_gen} below paper-canonical 2048; expected std "
                f"≈ ±5-15 metric units. Consider --name kvd for small-N stability."
            )
        if n_gen >= min_recommended:
            return (
                f"WARN: n_gen={n_gen} below recommended 500; expected std "
                f"≈ ±20-50 metric units. Score is a trend indicator, not "
                f"paper-comparable. See VIDEO_METRICS_DESIGN.md §9."
            )
        return (
            f"ERROR: n_gen={n_gen} below minimum {min_recommended}; "
            f"score is unreliable. Use --allow-tiny-sample to override "
            f"or switch to --name kvd."
        )
