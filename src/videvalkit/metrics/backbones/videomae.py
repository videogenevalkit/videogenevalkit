"""VideoMAE Kinetics-400 video feature extractor.

A motion-semantic feature space for FVD / KVD, complementary to S3D (CNN) and
I3D (paper-canonical). Uses HuggingFace ``MCG-NJU/videomae-base-finetuned-kinetics``
by default — a ViT-B/16 self-supervised video transformer fine-tuned on K400.

Input  : 16 frames × 224×224, uniformly sampled from the clip (matches the
         5s × 24fps × 120-frame T2V standard — 7-8x downsample factor).
Output : 768-d ``[CLS]`` pooled feature per video, float64.

NOTE: VideoMAE-v2 (OpenGVLab/VideoMAEv2-Base) is registered as a supported
backbone string but not implemented here — its weights aren't on the HF
transformers compatibility list yet. ``videomae-base`` (v1, this loader) is
the recommended option until that ships.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_REPO = "MCG-NJU/videomae-base-finetuned-kinetics"
N_FRAMES = 16
RESIZE = 224
FEAT_DIM = 768


class VideoMAEFeatureExtractor:
    """16-frame VideoMAE-base K400 features for FVD/KVD."""

    def __init__(self, device: str = "cpu", repo: str = DEFAULT_REPO):
        import torch
        from transformers import VideoMAEImageProcessor, VideoMAEModel

        self.device = device
        self.repo = repo
        self._processor = VideoMAEImageProcessor.from_pretrained(repo)
        self._model = VideoMAEModel.from_pretrained(repo).to(device).eval()
        self._torch = torch

    def _read_clip(self, path: Path) -> list[np.ndarray] | None:
        import decord
        vr = decord.VideoReader(str(path))
        total = len(vr)
        if total == 0:
            return None
        idxs = np.linspace(0, total - 1, N_FRAMES, dtype=int).tolist()
        arr = vr.get_batch(idxs).asnumpy()      # (T, H, W, 3) uint8
        # The processor expects a list of PIL or numpy frames per video.
        return [arr[i] for i in range(arr.shape[0])]

    def extract_one(self, path: Path, **_kw) -> np.ndarray:
        torch = self._torch
        frames = self._read_clip(Path(path))
        if frames is None:
            return np.zeros(FEAT_DIM, dtype=np.float64)
        inputs = self._processor(frames, return_tensors="pt")
        pixel = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            out = self._model(pixel_values=pixel)
            # last_hidden_state: (B, T_patches, D); pool by mean for a clip vector.
            feat = out.last_hidden_state.mean(dim=1).squeeze(0)
        return feat.float().cpu().numpy().astype(np.float64)

    def extract_many(self, paths: list[Path], **_kw) -> np.ndarray:
        feats = [self.extract_one(Path(p)) for p in sorted(paths, key=str)]
        return np.stack(feats, axis=0) if feats else np.zeros((0, FEAT_DIM))
