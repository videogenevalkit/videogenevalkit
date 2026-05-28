"""FVD — Fréchet Video Distance.

Per docs/VIDEO_METRICS_DESIGN.md §4.1.

Backbones:
  * i3d-k400  — paper canonical [Unterthiner et al. 2018]. Requires the I3D
    Kinetics-400 weights, fetched from videogenevalkit/checkpoints [pending
    HF hosting]. Until then, this backbone raises with a clear message.
  * s3d-k400  — FUNCTIONAL NOW: torchvision S3D pretrained on Kinetics-400.
    Weights auto-download. NOT byte-identical to paper I3D-FVD [different
    backbone], but a legitimate Kinetics-400 video-distribution metric. Use
    for trend monitoring / relative comparison until i3d-k400 lands.

The Fréchet machinery is shared with VFID/CLIP-FVD [metrics/utils/frechet.py],
float64 for reproducibility.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from videvalkit.core.distribution_metric import (
    BaseDistributionMetric,
    DistributionMetricResult,
)
from videvalkit.metrics.utils.frechet import fid_from_features

log = logging.getLogger(__name__)


class FVD(BaseDistributionMetric):
    name = "fvd"
    canonical_backbone = "i3d-k400"
    supported_backbones = ["i3d-k400", "s3d-k400", "videomae-v2-base", "vjepa-l16"]
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
            raise ValueError(
                f"backbone {bb!r} not supported for FVD; "
                f"choose from {self.supported_backbones}"
            )
        if ref_videos is None:
            raise ValueError("FVD requires ref_videos [distribution metric]")

        n_gen = len(gen_videos)
        warning = self.check_sample_size(n_gen, self.min_recommended_samples)
        if warning and warning.startswith("ERROR:") and not allow_tiny_sample:
            raise ValueError(warning + " Pass --allow-tiny-sample to override.")

        if bb in ("i3d-k400", "videomae-v2-base", "vjepa-l16"):
            raise NotImplementedError(
                f"FVD backbone {bb!r} awaits weight fetch from "
                f"videogenevalkit/checkpoints [pending HF hosting]. "
                f"Use --backbone s3d-k400 for a functional Kinetics-400 "
                f"alternative [torchvision S3D, auto-download]."
            )

        # s3d-k400 functional path
        clip_cfg = clip_sampling or {"n_frames": 16, "resize": 224}
        dev = self._resolve_device(device)
        from videvalkit.metrics.backbones.s3d_k400 import S3DFeatureExtractor

        extractor = S3DFeatureExtractor(device=dev)
        gen_feats = extractor.extract_many(gen_videos, **clip_cfg)
        ref_feats = extractor.extract_many(ref_videos, **clip_cfg)
        score = fid_from_features(gen_feats, ref_feats)

        return DistributionMetricResult(
            metric=self.name,
            score=score,
            n_gen=n_gen,
            n_ref=len(ref_videos),
            backbone=bb,
            backbone_version="torchvision-S3D_Weights.KINETICS400_V1",
            clip_sampling=clip_cfg,
            sample_size_warning=warning,
            meta={"note": "s3d-k400 backbone; not byte-identical to paper i3d-k400"},
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
