"""VFID — Video FID via InceptionV3 per-frame features.

Per docs/VIDEO_METRICS_DESIGN.md §4.2. FUNCTIONAL with zero extra deps:
extracts InceptionV3 pool3 [2048-d] features per frame via torchvision
[weights auto-download], pools per video [mean], then computes Fréchet
distance with the shared float64 util [no torch-fidelity needed].
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


class VFID(BaseDistributionMetric):
    name = "vfid"
    canonical_backbone = "inception-v3"
    supported_backbones = ["inception-v3"]
    min_recommended_samples = 100
    requires_reference = True

    def __init__(self):
        self._model = None
        self._tf = None
        self._device = None

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
        bb = backbone or self.canonical_backbone
        if bb not in self.supported_backbones:
            raise ValueError(f"backbone {bb!r} not supported for VFID")
        if ref_videos is None:
            raise ValueError("VFID requires ref_videos")

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

        clip_cfg = clip_sampling or {"n_frames": 8}
        self._ensure_model(self._resolve_device(device))

        gen_feats = self._video_features(gen_videos, **clip_cfg)
        ref_feats = self._video_features(ref_videos, **clip_cfg)
        score = fid_from_features(gen_feats, ref_feats)

        return DistributionMetricResult(
            metric=self.name, score=score, n_gen=n_gen, n_ref=len(ref_videos),
            backbone=bb, backbone_version="torchvision-InceptionV3-IMAGENET1K_V1",
            clip_sampling=clip_cfg, sample_size_warning=warning,
            meta={"aggregation": "per-frame InceptionV3 pool3 [2048-d], mean per video"},
        )

    def _ensure_model(self, device: str):
        if self._model is not None:
            return
        import torch.nn as nn
        from torchvision.models import Inception_V3_Weights, inception_v3
        self._device = device
        weights = Inception_V3_Weights.IMAGENET1K_V1
        model = inception_v3(weights=weights, aux_logits=True)
        model.fc = nn.Identity()   # expose the 2048-d pooled features
        self._model = model.to(device).eval()
        self._tf = weights.transforms()

    def _video_features(self, videos: list[Path], n_frames: int = 8, **_) -> np.ndarray:
        import torch
        feats = []
        for vp in sorted(videos, key=lambda x: str(x)):
            frames = self._read_frames(Path(vp), n_frames=n_frames)
            if frames is None:
                continue
            t = torch.from_numpy(frames).permute(0, 3, 1, 2)   # T,C,H,W uint8
            t = self._tf(t).to(self._device)
            with torch.no_grad():
                f = self._model(t)                              # (T, 2048)
            feats.append(f.float().mean(dim=0).cpu().numpy())   # per-video mean → 2048
        return np.stack(feats, axis=0).astype(np.float64) if feats else np.zeros((0, 2048))

    def _read_frames(self, path: Path, n_frames: int = 8):
        import decord
        vr = decord.VideoReader(str(path))
        total = len(vr)
        if total == 0:
            return None
        idxs = np.linspace(0, total - 1, n_frames, dtype=int).tolist()
        return vr.get_batch(idxs).asnumpy()

    @staticmethod
    def _resolve_device(device: str) -> str:
        from videvalkit.metrics.utils.device import resolve_device
        return resolve_device(device)
