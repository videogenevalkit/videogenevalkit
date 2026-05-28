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
        # backbone=None means "default" — we may auto-fall-back to s3d-k400
        # for monitoring when the paper i3d-k400 weights aren't placed.
        backbone_explicit = backbone is not None
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

        if bb in ("videomae-v2-base", "vjepa-l16"):
            raise NotImplementedError(
                f"FVD backbone {bb!r} not wired yet. "
                f"Use --backbone s3d-k400 [functional] or i3d-k400 [paper-"
                f"canonical, needs i3d_torchscript.pt placed locally]."
            )

        clip_cfg = clip_sampling or {"n_frames": 16, "resize": 224}
        dev = self._resolve_device(device)

        if bb == "i3d-k400":
            from videvalkit.metrics.backbones.i3d_k400 import (
                I3DFeatureExtractor, i3d_weights_path,
            )
            if i3d_weights_path() is None and not backbone_explicit:
                # Default invocation + no i3d weights → auto-fall-back to s3d-k400.
                # Monitoring [the common case] wants a working trend metric, not
                # a hard failure. Explicit --backbone i3d-k400 still fails [strict
                # paper-repro intent].
                log.warning(
                    "videvalkit: FVD i3d-k400 weights not found; falling back to "
                    "s3d-k400 [functional Kinetics-400 backbone]. This is fine for "
                    "training monitoring / relative comparison. For paper-faithful "
                    "i3d-FVD, place i3d_torchscript.pt [see backbones/i3d_k400.py] "
                    "or pass --backbone i3d-k400 explicitly to require it."
                )
                bb = "s3d-k400"
                from videvalkit.metrics.backbones.s3d_k400 import S3DFeatureExtractor
                extractor = S3DFeatureExtractor(device=dev)
                backbone_version = "torchvision-S3D_Weights.KINETICS400_V1 [auto-fallback]"
                note = "s3d-k400 [auto-fallback from i3d default]; trend metric, not paper i3d-FVD"
            else:
                # weights present, or explicitly requested → paper-canonical [strict]
                extractor = I3DFeatureExtractor(device=dev)
                backbone_version = "i3d-k400-torchscript [StyleGAN-V convention]"
                note = "paper-canonical i3d-k400"
        else:  # s3d-k400 — functional, torchvision auto-download
            from videvalkit.metrics.backbones.s3d_k400 import S3DFeatureExtractor
            extractor = S3DFeatureExtractor(device=dev)
            backbone_version = "torchvision-S3D_Weights.KINETICS400_V1"
            note = "s3d-k400 backbone; not byte-identical to paper i3d-k400"

        gen_feats = extractor.extract_many(gen_videos, **clip_cfg)
        ref_feats = extractor.extract_many(ref_videos, **clip_cfg)
        score = fid_from_features(gen_feats, ref_feats)

        return DistributionMetricResult(
            metric=self.name,
            score=score,
            n_gen=n_gen,
            n_ref=len(ref_videos),
            backbone=bb,
            backbone_version=backbone_version,
            clip_sampling=clip_cfg,
            sample_size_warning=warning,
            meta={"note": note},
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        from videvalkit.metrics.utils.device import resolve_device
        return resolve_device(device)
