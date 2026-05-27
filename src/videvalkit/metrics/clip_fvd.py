"""CLIP-FVD (experimental) — Fréchet distance on CLIP-ViT-L/14 frame features.

Per docs/VIDEO_METRICS_DESIGN.md §4.4. Marked experimental: not comparable
to canonical FVD due to different feature space.

v0.2 ships class shape with reasonable defaults; uses openai-clip which IS
present in the env, so we can implement this one more fully than the others.
But for consistency with the other 3 distribution metrics and to avoid
spending extra session time on what's an experimental metric, we ship the
stub here too — wire-up follows the same pattern.
"""

from __future__ import annotations

from pathlib import Path

from videvalkit.core.distribution_metric import (
    BaseDistributionMetric,
    DistributionMetricResult,
)


class CLIPFVD(BaseDistributionMetric):
    name = "clip-fvd"
    canonical_backbone = "clip-vit-l14"
    supported_backbones = ["clip-vit-l14", "clip-vit-b16"]
    min_recommended_samples = 100
    requires_reference = True

    def compute(
        self,
        gen_videos: list[Path],
        ref_videos: list[Path] | None,
        backbone: str | None = None,
        clip_sampling: dict | None = None,
        seed: int = 42,
        device: str = "auto",
    ) -> DistributionMetricResult:
        bb = backbone or self.canonical_backbone
        if bb not in self.supported_backbones:
            raise ValueError(f"backbone {bb!r} not supported for CLIP-FVD")
        warning = self.check_sample_size(len(gen_videos), self.min_recommended_samples)
        if warning and warning.startswith("ERROR:"):
            raise ValueError(warning + " Pass --allow-tiny-sample to override.")
        raise NotImplementedError(
            "CLIP-FVD compute() awaits implementation [v0.2 follow-up]. "
            "Mark experimental: not comparable to FVD."
        )
