"""CLIP-FVD [experimental] — Fréchet distance on CLIP-ViT frame features.

Per docs/VIDEO_METRICS_DESIGN.md §4.4. FUNCTIONAL: openai-clip is in the env,
weights auto-download. NOT comparable to canonical FVD [different feature
space]. Marked experimental.
"""

from __future__ import annotations

import logging
from pathlib import Path

from videvalkit.core.distribution_metric import (
    BaseDistributionMetric,
    DistributionMetricResult,
)
from videvalkit.metrics.utils.frechet import fid_from_features

log = logging.getLogger(__name__)


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
        allow_tiny_sample: bool = False,
    ) -> DistributionMetricResult:
        bb = backbone or self.canonical_backbone
        if bb not in self.supported_backbones:
            raise ValueError(f"backbone {bb!r} not supported for CLIP-FVD")
        if ref_videos is None:
            raise ValueError("CLIP-FVD requires ref_videos")

        n_gen = len(gen_videos)
        warning = self.check_sample_size(n_gen, self.min_recommended_samples)
        if warning and warning.startswith("ERROR:") and not allow_tiny_sample:
            raise ValueError(warning + " Pass --allow-tiny-sample to override.")

        clip_cfg = clip_sampling or {"n_frames": 16}
        dev = self._resolve_device(device)
        from videvalkit.metrics.backbones.clip_vit import CLIPFeatureExtractor
        extractor = CLIPFeatureExtractor(backbone=bb, device=dev)

        gen_feats = extractor.extract_many(gen_videos, **clip_cfg)
        ref_feats = extractor.extract_many(ref_videos, **clip_cfg)
        score = fid_from_features(gen_feats, ref_feats)

        return DistributionMetricResult(
            metric=self.name, score=score, n_gen=n_gen, n_ref=len(ref_videos),
            backbone=bb, backbone_version=f"openai-clip {bb}",
            clip_sampling=clip_cfg, sample_size_warning=warning,
            meta={"note": "experimental; NOT comparable to canonical i3d-FVD"},
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        from videvalkit.metrics.utils.device import resolve_device
        return resolve_device(device)
