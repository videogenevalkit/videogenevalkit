"""VFID — Video FID with InceptionV3 + mean-pool frame aggregation.

Per docs/VIDEO_METRICS_DESIGN.md §4.2.

v0.2 ships class shape; actual compute awaits inception-v3 fetch
[v0.2 follow-up].
"""

from __future__ import annotations

from pathlib import Path

from videvalkit.core.distribution_metric import (
    BaseDistributionMetric,
    DistributionMetricResult,
)


class VFID(BaseDistributionMetric):
    name = "vfid"
    canonical_backbone = "inception-v3"
    supported_backbones = ["inception-v3"]
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
            raise ValueError(f"backbone {bb!r} not supported for VFID")
        warning = self.check_sample_size(len(gen_videos), self.min_recommended_samples)
        if warning and warning.startswith("ERROR:"):
            raise ValueError(warning + " Pass --allow-tiny-sample to override.")
        raise NotImplementedError(
            "VFID compute() awaits inception-v3 backbone fetch [v0.2 follow-up]."
        )
