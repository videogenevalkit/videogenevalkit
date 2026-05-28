"""S3D Kinetics-400 video feature extractor [torchvision].

Functional FVD backbone: weights auto-download via torchvision. Output is a
1024-d per-clip feature [global avg pool of S3D.features].
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class S3DFeatureExtractor:
    """Extract S3D-K400 features from videos. One feature vector per video."""

    def __init__(self, device: str = "cpu"):
        import torch
        from torchvision.models.video import S3D_Weights, s3d

        self.device = device
        self._weights = S3D_Weights.KINETICS400_V1
        self.model = s3d(weights=self._weights).to(device).eval()
        # transforms from the weights metadata [resize/normalize/crop]
        self._tf = self._weights.transforms()
        self._torch = torch

    def _read_clip(self, path: Path, n_frames: int = 16, resize: int = 224):
        """Read n_frames uniformly as a (T, H, W, 3) uint8 array."""
        import decord
        vr = decord.VideoReader(str(path))
        total = len(vr)
        if total == 0:
            return None
        idxs = np.linspace(0, total - 1, n_frames, dtype=int).tolist()
        return vr.get_batch(idxs).asnumpy()  # (T, H, W, 3)

    def extract_one(self, path: Path, n_frames: int = 16, resize: int = 224) -> np.ndarray:
        torch = self._torch
        clip = self._read_clip(path, n_frames=n_frames, resize=resize)
        if clip is None:
            return np.zeros(1024, dtype=np.float64)
        # (T,H,W,C) uint8 → (T,C,H,W) for the weights transform. The
        # VideoClassification transform returns (C, T, H, W).
        t = torch.from_numpy(clip).permute(0, 3, 1, 2)  # T,C,H,W
        t = self._tf(t)                                  # → C,T,H,W (normalized)
        # S3D expects (B, C, T, H, W)
        x = t.unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.model.features(x)               # (1, 1024, T', H', W')
            pooled = feats.mean(dim=[2, 3, 4]).squeeze(0)  # (1024,)
        return pooled.float().cpu().numpy().astype(np.float64)

    def extract_many(
        self, paths: list[Path], n_frames: int = 16, resize: int = 224, **_,
    ) -> np.ndarray:
        feats = [
            self.extract_one(Path(p), n_frames=n_frames, resize=resize)
            for p in sorted(paths, key=lambda x: str(x))
        ]
        return np.stack(feats, axis=0) if feats else np.zeros((0, 1024))
