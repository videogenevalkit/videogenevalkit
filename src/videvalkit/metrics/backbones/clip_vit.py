"""CLIP-ViT visual feature extractor for CLIP-FVD [openai-clip, in env]."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_BACKBONE_TO_CLIP = {
    "clip-vit-l14": "ViT-L/14",
    "clip-vit-b16": "ViT-B/16",
}


class CLIPFeatureExtractor:
    """Per-frame CLIP image features, mean-pooled per video."""

    def __init__(self, backbone: str = "clip-vit-l14", device: str = "cpu"):
        import clip
        import torch
        if backbone not in _BACKBONE_TO_CLIP:
            raise ValueError(f"unknown clip backbone {backbone!r}")
        self.device = device
        self.model, self.preprocess = clip.load(_BACKBONE_TO_CLIP[backbone], device=device)
        self.model.eval()
        self._torch = torch

    def _read_frames(self, p: Path, n_frames: int = 16):
        import decord
        from PIL import Image
        vr = decord.VideoReader(str(p))
        total = len(vr)
        if total == 0:
            return []
        idxs = np.linspace(0, total - 1, n_frames, dtype=int).tolist()
        return [Image.fromarray(f) for f in vr.get_batch(idxs).asnumpy()]

    def extract_one(self, p: Path, n_frames: int = 16) -> np.ndarray:
        torch = self._torch
        frames = self._read_frames(p, n_frames=n_frames)
        if not frames:
            return np.zeros(self._feat_dim(), dtype=np.float64)
        batch = torch.stack([self.preprocess(f) for f in frames]).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(batch).float()   # (T, D)
        return feats.mean(dim=0).cpu().numpy().astype(np.float64)

    def _feat_dim(self) -> int:
        return getattr(self.model.visual, "output_dim", 768)

    def extract_many(self, paths: list[Path], n_frames: int = 16, **_) -> np.ndarray:
        feats = [self.extract_one(Path(p), n_frames=n_frames)
                 for p in sorted(paths, key=lambda x: str(x))]
        return np.stack(feats, axis=0) if feats else np.zeros((0, self._feat_dim()))
