"""Frame cache — decode each video at most once per (mode, n_frames) per run.

Frames are written to disk as JPEGs under
    {frames_cache_dir}/{video_hash}/{mode}_{n}/{idx:03d}.jpg
so subsequent scorers that ask for the same (video, mode, n) just read JPEGs.

The on-disk cache is a tiny fraction of the original mp4 size, so disabling
the cache only makes sense when the workspace is on a slow filesystem.
"""

from __future__ import annotations

import logging
from pathlib import Path

from videvalkit.core.layout import WorkspaceLayout
from videvalkit.utils.video import extract_frames, video_hash

log = logging.getLogger(__name__)


class FrameCache:
    """Read-through cache for decoded video frames.

    Keyed by `(video_hash(path), mode, n_frames)`.
    """

    def __init__(self, layout: WorkspaceLayout, enabled: bool = True) -> None:
        self.layout = layout
        self.enabled = enabled

    def get_or_extract(
        self,
        video_path: str | Path,
        mode: str = "holistic",
        n_frames: int = 8,
    ) -> list:
        video_path = Path(video_path)
        if not self.enabled:
            return extract_frames(video_path, mode=mode, n_frames=n_frames)

        cache_dir = self._cache_dir(video_path, mode, n_frames)
        existing = sorted(cache_dir.glob("*.jpg")) if cache_dir.exists() else []
        if len(existing) >= n_frames:
            return self._load(existing[:n_frames])

        frames = extract_frames(video_path, mode=mode, n_frames=n_frames)
        if frames:
            self._save(frames, cache_dir)
        return frames

    def invalidate(self, video_path: str | Path) -> None:
        """Drop all cached variants for a given video."""
        root = self.layout.frames_cache_dir / video_hash(video_path)
        if root.exists():
            import shutil
            shutil.rmtree(root, ignore_errors=True)

    # ---- internals ----------------------------------------------------------
    def _cache_dir(self, video_path: Path, mode: str, n_frames: int) -> Path:
        return self.layout.frames_cache_dir / video_hash(video_path) / f"{mode}_{n_frames}"

    def _save(self, frames: list, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(frames):
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(out_dir / f"{i:03d}.jpg", format="JPEG", quality=88)

    def _load(self, paths: list[Path]) -> list:
        from videvalkit.utils.video import _lazy_pil
        PIL = _lazy_pil()
        return [PIL.open(p).copy() for p in paths]
