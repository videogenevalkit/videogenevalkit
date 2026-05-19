"""Video utilities — frame extraction modes, base64 encoding, content hashing.

Frame sampling modes (matching WorldJen / Video-Bench conventions):
  * "holistic": N frames evenly distributed across the whole video
  * "sampled" : same as holistic; alias kept for readability
  * "micro"   : N frames densely sampled from the first 60 frames (catches
                temporal flickering / motion-smoothness artifacts that average
                out over a longer span)
  * "uniform" : N frames uniformly across a target frame range
"""

from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path


# OpenCV / PIL are heavy; import lazily so the toolkit core env doesn't need them.
_cv2 = None
_PIL = None


def _lazy_cv2():
    global _cv2
    if _cv2 is None:
        import cv2 as _c
        _cv2 = _c
    return _cv2


def _lazy_pil():
    global _PIL
    if _PIL is None:
        import PIL.Image as _p
        _PIL = _p
    return _PIL


def video_hash(path: str | Path, block_size: int = 1 << 20) -> str:
    """Stable cache key: sha256 of (size, first 1 MB)."""
    p = Path(path)
    h = hashlib.sha256()
    h.update(str(p.stat().st_size).encode())
    with p.open("rb") as f:
        h.update(f.read(block_size))
    return h.hexdigest()[:16]


def frame_indices(total: int, mode: str, n_frames: int) -> list[int]:
    """Pick `n_frames` indices into a [0, total) range under the given mode."""
    if total <= 0 or n_frames <= 0:
        return []
    if mode == "micro":
        cap = min(total, 60)
        step = max(1, cap // n_frames)
        return list(range(0, cap, step))[:n_frames]
    return [int(i * total / n_frames) for i in range(n_frames)]


def videobench_frame_indices(
    total: int, fps: float, extract_persecond: int = 2,
) -> list[int]:
    """Upstream Video-Bench ``utils.py:20-56`` frame selection.

    Walk from frame 1 by ``int(fps/extract_persecond)`` until ``total-1``,
    then append frame ``total-1``. Yields ~ ``duration * extract_persecond + 1``
    frames (count varies with video length, unlike the fixed-N modes).
    """
    if total <= 1 or fps <= 0:
        return []
    skip = max(1, int(fps / extract_persecond))
    idxs = list(range(1, total - 1, skip))
    idxs.append(total - 1)
    return idxs


def extract_frames(
    video_path: str | Path,
    mode: str = "holistic",
    n_frames: int = 8,
    extract_persecond: int = 2,
) -> list:
    """Extract frames as PIL.Image.Image list. Empty list on failure.

    ``mode='videobench'`` selects upstream Video-Bench's fps-based pattern
    (fixed-stride from frame 1 + explicit last frame); in this mode
    ``n_frames`` is ignored and the count varies with video duration.

    Implementation note: many generative-video MP4s are encoded with sparse
    keyframes (e.g. Kling outputs ship as 1 I-frame + 120 P-frames). cv2's
    ``cap.set(CAP_PROP_POS_FRAMES, idx)`` seeks to the nearest keyframe in
    that case, which collapses N spaced indices to the same frame and makes
    motion-aware VLM scoring degenerate. We read **sequentially** and pick
    the target indices by counting, which always lands on the exact frame.
    """
    cv2 = _lazy_cv2()
    PIL = _lazy_pil()
    cap = cv2.VideoCapture(str(video_path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if mode == "videobench":
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            indices = videobench_frame_indices(total, fps, extract_persecond)
        else:
            indices = frame_indices(total, mode, n_frames)
        if not indices:
            return []
        wanted = sorted(set(indices))
        wanted_set = set(wanted)
        last_wanted = wanted[-1]
        frame_at: dict[int, "PIL.Image.Image"] = {}
        seq = 0
        while seq <= last_wanted:
            ok, frame = cap.read()
            if not ok:
                break
            if seq in wanted_set:
                frame_at[seq] = PIL.fromarray(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )
            seq += 1
        return [frame_at[idx] for idx in indices if idx in frame_at]
    finally:
        cap.release()


def pil_to_data_url(img, fmt: str = "JPEG", quality: int = 85) -> str:
    """PIL.Image -> 'data:image/jpeg;base64,...' (OpenAI vision-message format)."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"


def strip_code_fences(text: str) -> str:
    """Strip ```json fences / ```` fences from an LLM response, returning bare text."""
    s = text.strip()
    if "```json" in s:
        return s.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in s:
        return s.split("```", 1)[1].split("```", 1)[0].strip()
    return s
