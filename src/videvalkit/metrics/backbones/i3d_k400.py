"""I3D Kinetics-400 video feature extractor [paper-canonical FVD backbone].

Loads the standard StyleGAN-V / FVD-community I3D torchscript model. The
weights file ``i3d_torchscript.pt`` is NOT bundled [licensing + size]; place
it at one of:

  1. $VIDEVALKIT_FVD_I3D_PATH                              [explicit override]
  2. $VIDEVALKIT_CACHE_HOME/checkpoints/fvd/i3d_torchscript.pt
  3. ~/.cache/videvalkit/checkpoints/fvd/i3d_torchscript.pt   [default]

When present, FVD's i3d-k400 backbone becomes paper-canonical with zero code
change. When absent, FVD raises a clear message pointing here [and to the
functional s3d-k400 alternative].

Interface follows the StyleGAN-V convention:
  detector(videos, rescale, resize, return_features=True) → (B, D) features
  videos: (B, C, T, H, W), float, pixel range [0, 255].
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_DETECTOR_KWARGS = dict(rescale=False, resize=True, return_features=True)


def i3d_weights_path() -> Path | None:
    """Return the resolved I3D weights path if it exists, else None."""
    candidates = []
    env = os.environ.get("VIDEVALKIT_FVD_I3D_PATH")
    if env:
        candidates.append(Path(env))
    # Site-shared ckpt root (set up by scripts/setup_ckpt_dir.sh).
    ckpt_home = os.environ.get("VIDEVALKIT_CKPT_HOME")
    if ckpt_home:
        candidates.append(Path(ckpt_home) / "i3d" / "i3d_torchscript.pt")
    cache_home = os.environ.get(
        "VIDEVALKIT_CACHE_HOME", str(Path.home() / ".cache" / "videvalkit")
    )
    candidates.append(Path(cache_home) / "checkpoints" / "fvd" / "i3d_torchscript.pt")
    for c in candidates:
        if c.is_file():
            return c
    return None


class I3DFeatureExtractor:
    """I3D-K400 torchscript feature extractor [1024-d per clip]."""

    def __init__(self, device: str = "cpu", weights_path: str | Path | None = None):
        import torch

        path = Path(weights_path) if weights_path else i3d_weights_path()
        if path is None or not path.is_file():
            raise FileNotFoundError(
                "I3D-K400 weights not found. FVD's paper-canonical i3d-k400 "
                "backbone needs `i3d_torchscript.pt`. Place it at "
                "$VIDEVALKIT_FVD_I3D_PATH or "
                "~/.cache/videvalkit/checkpoints/fvd/i3d_torchscript.pt "
                "[StyleGAN-V / FVD-community I3D]. Or use --backbone s3d-k400 "
                "for a functional Kinetics-400 alternative."
            )
        self.device = device
        self.detector = torch.jit.load(str(path)).eval().to(device)
        self._torch = torch

    def _read_clip(self, p: Path, n_frames: int = 16, resize: int = 224):
        import decord
        vr = decord.VideoReader(str(p))
        total = len(vr)
        if total == 0:
            return None
        idxs = np.linspace(0, total - 1, n_frames, dtype=int).tolist()
        return vr.get_batch(idxs).asnumpy()  # (T,H,W,3) uint8

    def extract_one(self, p: Path, n_frames: int = 16, resize: int = 224) -> np.ndarray:
        torch = self._torch
        clip = self._read_clip(p, n_frames=n_frames, resize=resize)
        if clip is None:
            return np.zeros(1024, dtype=np.float64)
        # (T,H,W,C) → (1, C, T, H, W) float in [0,255]
        x = (torch.from_numpy(clip).float()
             .permute(3, 0, 1, 2).unsqueeze(0).to(self.device))
        with torch.no_grad():
            feats = self.detector(x, **_DETECTOR_KWARGS)  # (1, 1024)
        return feats.squeeze(0).float().cpu().numpy().astype(np.float64)

    def extract_many(
        self, paths: list[Path], n_frames: int = 16, resize: int = 224, **_,
    ) -> np.ndarray:
        feats = [
            self.extract_one(Path(p), n_frames=n_frames, resize=resize)
            for p in sorted(paths, key=lambda x: str(x))
        ]
        return np.stack(feats, axis=0) if feats else np.zeros((0, 1024))
