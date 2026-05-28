"""ViCLIP-Score — per-clip text-video cosine on the ViCLIP video encoder.

Per docs/VIDEO_METRICS_DESIGN.md §3.2. Preferred over CLIP-Score for T2V:
ViCLIP is trained on video-text pairs, so it captures temporal/motion
semantics that a per-frame image-text CLIP misses.

Backbone code is vendored under ``metrics/third_party/viclip`` (MIT). Frame
sampling (8 uniform frames, middle-of-interval) and the CLIP normalization
transform mirror the upstream VBench ``overall_consistency`` path so scores
are comparable.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from pydantic import BaseModel

log = logging.getLogger(__name__)

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class ViCLIPScoreResult(BaseModel):
    metric: str = "viclip-score"
    score: float
    per_video: dict[str, float]
    n_pairs: int
    backbone: str = "ViCLIP-L"
    n_frames: int = 8


def _middle_frame_indices(num_frames: int, vlen: int) -> list[int]:
    """Uniform interval split, take the middle of each — matches VBench."""
    acc = min(num_frames, vlen)
    intervals = np.linspace(start=0, stop=vlen, num=acc + 1).astype(int)
    ranges = [(intervals[i], intervals[i + 1] - 1) for i in range(len(intervals) - 1)]
    idxs = [(s + e) // 2 for s, e in ranges]
    # pad if the clip is shorter than num_frames
    while len(idxs) < num_frames:
        idxs.append(idxs[-1] if idxs else 0)
    return idxs


class ViCLIPScore:
    """Per-clip ViCLIP cosine — text-video alignment over an 8-frame clip."""

    name = "viclip-score"
    backbone = "ViCLIP-L"
    requires_judge = False

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device = None

    def _ensure_loaded(self, device: str = "cuda"):
        if self._model is not None:
            return
        from .backbones.viclip_l import load_viclip

        self._model, self._tokenizer = load_viclip(device=device)
        self._device = next(self._model.parameters()).device

    def _read_clip(self, video_path: Path, n_frames: int = 8):
        import decord
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import center_crop, normalize, resize

        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(str(video_path), num_threads=1)
        vlen = len(vr)
        if vlen == 0:
            return None
        idxs = _middle_frame_indices(n_frames, vlen)
        frames = vr.get_batch(idxs).permute(0, 3, 1, 2)  # (T, C, H, W) uint8
        frames = resize(
            frames, 224, interpolation=InterpolationMode.BICUBIC, antialias=False
        )
        frames = center_crop(frames, 224).float().div(255.0)
        frames = normalize(frames, _CLIP_MEAN, _CLIP_STD)
        return frames.to(self._device)

    def compute(
        self,
        videos: list[Path],
        prompts: list[str | dict],
        n_frames: int = 8,
        device: str = "cuda",
        seed: int = 42,
    ) -> ViCLIPScoreResult:
        import torch

        if len(videos) != len(prompts):
            raise ValueError(
                f"videos and prompts must have same length, "
                f"got {len(videos)} vs {len(prompts)}"
            )
        norm_prompts = [
            (p.get("text", "") if isinstance(p, dict) else str(p)) for p in prompts
        ]
        if not videos:
            return ViCLIPScoreResult(
                score=0.0, per_video={}, n_pairs=0, n_frames=n_frames
            )

        self._ensure_loaded(device=device)

        per_video: dict[str, float] = {}
        scores: list[float] = []
        text_cache: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for video_path, prompt_text in zip(videos, norm_prompts):
                clip = self._read_clip(Path(video_path), n_frames=n_frames)
                if clip is None:
                    log.warning("viclip-score: no frames from %s", video_path)
                    per_video[str(video_path)] = 0.0
                    continue
                vid_feat = self._model.get_vid_features(clip.unsqueeze(0))  # (1, D)
                if prompt_text not in text_cache:
                    txt = self._model.get_text_features(
                        prompt_text, self._tokenizer, {}
                    )
                    text_cache[prompt_text] = txt
                txt_feat = text_cache[prompt_text]  # (1, D), normalized
                score = float((vid_feat @ txt_feat.T)[0][0].cpu())
                per_video[str(video_path)] = score
                scores.append(score)

        overall = sum(scores) / len(scores) if scores else 0.0
        return ViCLIPScoreResult(
            score=overall,
            per_video=per_video,
            n_pairs=len(scores),
            backbone="ViCLIP-L",
            n_frames=n_frames,
        )
