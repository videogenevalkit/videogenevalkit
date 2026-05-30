"""KVD — Kernel Video Distance [polynomial-kernel MMD² on video features].

Per docs/VIDEO_METRICS_DESIGN.md §4.3. More stable than FVD at small N.

Shares the S3D-K400 video backbone with FVD [auto-download], so KVD is
FUNCTIONAL out of the box for monitoring. The i3d-k400 backbone [paper] is
also available once i3d_torchscript.pt is placed; default auto-falls-back to
s3d-k400 [same policy as FVD, user decision 2026-05-28].
"""

from __future__ import annotations

import logging
from pathlib import Path

from videvalkit.core.distribution_metric import (
    BaseDistributionMetric,
    DistributionMetricResult,
)
from videvalkit.metrics.utils.mmd import polynomial_mmd2

log = logging.getLogger(__name__)


class KVD(BaseDistributionMetric):
    name = "kvd"
    canonical_backbone = "i3d-k400"
    supported_backbones = ["i3d-k400", "s3d-k400", "videomae-base"]
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
        prompts: str | Path | list[str] | None = None,
        allow_partial_prompts: bool = False,
    ) -> DistributionMetricResult:
        backbone_explicit = backbone is not None
        bb = backbone or self.canonical_backbone
        if bb not in self.supported_backbones:
            raise ValueError(
                f"backbone {bb!r} not supported for KVD; "
                f"choose from {self.supported_backbones}"
            )
        if ref_videos is None:
            raise ValueError("KVD requires ref_videos [distribution metric]")

        # Prompt-aligned mode: enforce gen and ref are over the same prompt
        # set (matched by filename = prompt_id). Without this, the score mixes
        # model-quality with prompt-domain shift and is uninterpretable.
        if prompts is not None:
            from .utils.prompts import load_prompt_ids, verify_prompt_alignment
            pids = (
                load_prompt_ids(prompts) if isinstance(prompts, (str, Path))
                else list(prompts)
            )
            gen_videos, ref_videos, _ = verify_prompt_alignment(
                gen_videos, ref_videos, pids,
                allow_partial=allow_partial_prompts,
            )

        n_gen = len(gen_videos)
        warning = self.check_sample_size(n_gen, self.min_recommended_samples)
        if warning and warning.startswith("ERROR:") and not allow_tiny_sample:
            raise ValueError(warning + " Pass --allow-tiny-sample to override.")

        clip_cfg = clip_sampling or {"n_frames": 16, "resize": 224}
        dev = self._resolve_device(device)

        if bb == "videomae-base":
            from videvalkit.metrics.backbones.videomae import (
                VideoMAEFeatureExtractor,
            )
            extractor = VideoMAEFeatureExtractor(device=dev)
            note = "videomae-base: ViT-B/16 motion-semantic transformer"
        elif bb == "i3d-k400":
            from videvalkit.metrics.backbones.i3d_k400 import (
                I3DFeatureExtractor, i3d_weights_path,
            )
            if i3d_weights_path() is None and not backbone_explicit:
                log.warning(
                    "videvalkit: KVD i3d-k400 weights not found; falling back to "
                    "s3d-k400 [functional]. Fine for monitoring. Place "
                    "i3d_torchscript.pt or pass --backbone i3d-k400 to require it."
                )
                bb = "s3d-k400"
                from videvalkit.metrics.backbones.s3d_k400 import S3DFeatureExtractor
                extractor = S3DFeatureExtractor(device=dev)
                note = "s3d-k400 [auto-fallback]; trend metric"
            else:
                extractor = I3DFeatureExtractor(device=dev)
                note = "paper-canonical i3d-k400"
        else:
            from videvalkit.metrics.backbones.s3d_k400 import S3DFeatureExtractor
            extractor = S3DFeatureExtractor(device=dev)
            note = "s3d-k400 backbone"

        gen_feats = extractor.extract_many(gen_videos, **clip_cfg)
        ref_feats = extractor.extract_many(ref_videos, **clip_cfg)
        score = polynomial_mmd2(gen_feats, ref_feats)

        return DistributionMetricResult(
            metric=self.name, score=score, n_gen=n_gen, n_ref=len(ref_videos),
            backbone=bb, backbone_version=f"{bb} [poly-MMD2]",
            clip_sampling=clip_cfg, sample_size_warning=warning,
            meta={"note": note, "kernel": "polynomial degree=3"},
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        from videvalkit.metrics.utils.device import resolve_device
        return resolve_device(device)
