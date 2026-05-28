"""CLIP-Score — per-frame text-image CLIP cosine, averaged across frames.

Per docs/VIDEO_METRICS_DESIGN.md §3.2. Reference-free text-video alignment
baseline. Uses CLIP-ViT-L/14 by default.

Backbone is openai-clip (already in the env), so unlike the FVD-family
shells, this metric is fully functional in v0.2.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel

log = logging.getLogger(__name__)


class CLIPScoreResult(BaseModel):
    metric: str = "clip-score"
    score: float                      # mean across all (video, prompt) pairs
    per_video: dict[str, float]       # video_path → score
    n_pairs: int
    backbone: str = "ViT-L/14"
    n_frames: int


class CLIPScore:
    """Compute mean CLIP-ViT cosine similarity between prompt text and video frames.

    Algorithm:
      1. For each (video, prompt) pair:
         a. Extract N frames uniformly across the clip
         b. Encode each frame via CLIP image encoder
         c. Encode the prompt text once via CLIP text encoder
         d. Cosine similarity per frame; mean across frames → per-video score
      2. Mean across all (video, prompt) pairs → overall score

    Stable + deterministic with fixed seed; mean-pool aggregation is the
    community baseline.
    """

    name = "clip-score"
    backbone = "ViT-L/14"
    requires_judge = False

    def __init__(self, backbone: str = "ViT-L/14"):
        self._backbone_name = backbone
        self._model = None
        self._preprocess = None
        self._device = None

    def _ensure_loaded(self, device: str = "cuda"):
        if self._model is not None:
            return
        import clip
        from .utils.device import resolve_device
        self._device = resolve_device(device)
        self._model, self._preprocess = clip.load(
            self._backbone_name, device=self._device
        )
        self._model.eval()

    def _extract_frames(
        self, video_path: Path, n_frames: int = 8,
    ) -> list[Any]:
        """Uniformly sample n_frames from video; return as PIL.Images."""
        try:
            import decord
        except ImportError:
            raise ImportError(
                "clip-score needs `decord` for frame extraction. "
                "Install via pip install decord."
            )
        from PIL import Image
        import numpy as np

        vr = decord.VideoReader(str(video_path))
        total = len(vr)
        if total == 0:
            return []
        # Uniformly sample
        idxs = np.linspace(0, total - 1, n_frames, dtype=int).tolist()
        frames = vr.get_batch(idxs).asnumpy()  # (n_frames, H, W, 3)
        return [Image.fromarray(f) for f in frames]

    def compute(
        self,
        videos: list[Path],
        prompts: list[str | dict],
        n_frames: int = 8,
        device: str = "cuda",
        seed: int = 42,
    ) -> CLIPScoreResult:
        """Compute CLIP-Score.

        Args:
          videos: list of video paths
          prompts: list of prompt texts (matching length) OR list of dicts
                   with keys "video_path", "text" (more flexible)
          n_frames: frames to sample per video
          device: cuda | cpu
          seed: for reproducibility [no randomness in this metric, kept for API symmetry]

        Returns CLIPScoreResult with per-video + overall mean.
        """
        import clip
        import torch
        import torch.nn.functional as F

        # Normalize inputs
        if len(videos) != len(prompts):
            raise ValueError(
                f"videos and prompts must have same length, "
                f"got {len(videos)} vs {len(prompts)}"
            )

        # Normalize prompt format
        norm_prompts = []
        for p in prompts:
            if isinstance(p, dict):
                norm_prompts.append(p.get("text", ""))
            else:
                norm_prompts.append(str(p))

        self._ensure_loaded(device=device)

        per_video: dict[str, float] = {}
        scores: list[float] = []
        with torch.no_grad():
            for video_path, prompt_text in zip(videos, norm_prompts):
                frames = self._extract_frames(video_path, n_frames=n_frames)
                if not frames:
                    log.warning("clip-score: no frames extracted from %s", video_path)
                    per_video[str(video_path)] = 0.0
                    continue
                # Image features
                img_inputs = torch.stack([
                    self._preprocess(f) for f in frames
                ]).to(self._device)
                img_features = self._model.encode_image(img_inputs).float()
                img_features = F.normalize(img_features, dim=-1)
                # Text features (once)
                text_tokens = clip.tokenize([prompt_text], truncate=True).to(self._device)
                text_features = self._model.encode_text(text_tokens).float()
                text_features = F.normalize(text_features, dim=-1)
                # Per-frame cosine, then mean
                cos = (img_features @ text_features.T).squeeze(-1)
                per_video_score = float(cos.mean().item())
                per_video[str(video_path)] = per_video_score
                scores.append(per_video_score)

        overall = sum(scores) / len(scores) if scores else 0.0
        return CLIPScoreResult(
            score=overall,
            per_video=per_video,
            n_pairs=len(scores),
            backbone=self._backbone_name,
            n_frames=n_frames,
        )
