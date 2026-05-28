"""KVD — Kernel Video Distance (polynomial-kernel MMD² on I3D features).

Per docs/VIDEO_METRICS_DESIGN.md §4.3. More stable than FVD at small N.

v0.2 ships class shape; actual compute awaits I3D-K400 fetch
[shared backbone with FVD].
"""

from __future__ import annotations

from pathlib import Path

from videvalkit.core.distribution_metric import (
    BaseDistributionMetric,
    DistributionMetricResult,
)


class KVD(BaseDistributionMetric):
    name = "kvd"
    canonical_backbone = "i3d-k400"
    supported_backbones = ["i3d-k400"]
    min_recommended_samples = 50
    requires_reference = True

    def compute(
        self,
        gen_videos: list[Path],
        ref_videos: list[Path] | None,
        backbone: str | None = None,
        clip_sampling: dict | None = None,
        seed: int = 42,
        device: str = "auto",
        allow_tiny_sample: bool = False,
    ) -> DistributionMetricResult:
        bb = backbone or self.canonical_backbone
        if bb not in self.supported_backbones:
            raise ValueError(f"backbone {bb!r} not supported for KVD")
        warning = self.check_sample_size(len(gen_videos), self.min_recommended_samples)
        if warning and warning.startswith("ERROR:") and not allow_tiny_sample:
            raise ValueError(warning + " Pass --allow-tiny-sample to override.")
        raise NotImplementedError(
            "KVD compute() awaits I3D-K400 backbone fetch [v0.2 follow-up]. "
            "Backbone shared with FVD."
        )
