"""Prompt-alignment helpers for distribution metrics.

A trustworthy FVD/VFID/KVD/CLIP-FVD compares video distributions that come
from the **same prompt set** — gen videos and ref videos generated for the
same prompts. Comparing against an unrelated reference set (UCF101 etc.)
mixes "model quality difference" with "prompt-domain difference" and gives
an uninterpretable number.

These helpers enforce the convention:

  * ``load_prompt_ids(path)`` — read prompt_ids from a JSON or JSONL manifest
    (the format ``videvalkit prompts dump`` emits).
  * ``prompt_id_from_filename(path)`` — recover the prompt_id from a video
    file named ``<prompt_id>-<sample_index>.mp4`` (the toolkit's standard
    video layout). Splits on the LAST ``-`` so prompts containing hyphens
    are recovered correctly.
  * ``verify_prompt_alignment(gen, ref, prompt_ids)`` — assert every required
    prompt has at least one matching video in both gen and ref; return the
    paired video lists.

Standard video format assumption: 5s × 24fps × 120 frames. Backbones inside
each metric downsample (S3D 16 / I3D 16 / VideoMAE-v2 16) — videos at other
frame counts work but warn at the metric level.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

STANDARD_FRAMES: int = 120
STANDARD_FPS: int = 24
STANDARD_DURATION_SEC: float = 5.0


def load_prompt_ids(path: str | Path) -> list[str]:
    """Read prompt_ids from a JSON list or JSONL manifest.

    Each record can use ``id``, ``prompt_id`` or ``prompt_en`` as the key
    (in that priority). Mirrors the format ``videvalkit prompts dump`` emits.
    """
    p = Path(path)
    text = p.read_text()
    ids: list[str] = []
    if p.suffix.lower() == ".jsonl" or "\n" in text.strip():
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pid = r.get("id") or r.get("prompt_id") or r.get("prompt_en")
            if pid:
                ids.append(str(pid))
    else:
        data = json.loads(text)
        if isinstance(data, list):
            for r in data:
                pid = r.get("id") or r.get("prompt_id") or r.get("prompt_en")
                if pid:
                    ids.append(str(pid))
        else:
            raise ValueError(f"unsupported prompts manifest shape in {p}")
    if not ids:
        raise ValueError(f"no prompt_ids extracted from {p}")
    return ids


def prompt_id_from_filename(path: str | Path) -> str:
    """Extract prompt_id from ``<prompt_id>-<sample_index>.mp4``.

    Splits on the LAST ``-`` so prompts containing hyphens survive. If no
    trailing ``-N`` suffix is present, returns the bare stem.
    """
    stem = Path(path).stem
    head, _, tail = stem.rpartition("-")
    if head and tail.isdigit():
        return head
    return stem


def _index_by_pid(videos: Iterable[str | Path]) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = defaultdict(list)
    for v in videos:
        out[prompt_id_from_filename(v)].append(Path(v))
    return out


def verify_prompt_alignment(
    gen_videos: Iterable[str | Path],
    ref_videos: Iterable[str | Path],
    prompt_ids: list[str],
    *,
    allow_partial: bool = False,
) -> tuple[list[Path], list[Path], list[str]]:
    """Verify gen and ref both have videos for every required prompt_id.

    Returns ``(gen_aligned, ref_aligned, used_prompt_ids)``: the video lists
    restricted to the matched prompts (preserving multiple samples per prompt),
    and the list of prompt_ids actually covered.

    Raises ``ValueError`` on missing prompts unless ``allow_partial=True``,
    in which case the intersection is used and a count is logged.
    """
    want = list(dict.fromkeys(prompt_ids))   # dedup, preserve order
    g_idx = _index_by_pid(gen_videos)
    r_idx = _index_by_pid(ref_videos)
    missing_in_gen = [p for p in want if p not in g_idx]
    missing_in_ref = [p for p in want if p not in r_idx]
    if (missing_in_gen or missing_in_ref) and not allow_partial:
        sample_g = missing_in_gen[:5]
        sample_r = missing_in_ref[:5]
        raise ValueError(
            f"prompt-aligned distribution metric: "
            f"{len(missing_in_gen)} prompt(s) missing from gen, "
            f"{len(missing_in_ref)} from ref. "
            f"Pass --allow-partial-prompts to use the intersection. "
            f"Examples missing from gen: {sample_g}; from ref: {sample_r}"
        )
    used = [p for p in want if p in g_idx and p in r_idx]
    gen_aligned = [v for p in used for v in g_idx[p]]
    ref_aligned = [v for p in used for v in r_idx[p]]
    if allow_partial and (missing_in_gen or missing_in_ref):
        import logging
        logging.getLogger(__name__).warning(
            "videvalkit: prompt-aligned metric: using %d/%d prompts "
            "(%d missing in gen, %d in ref)",
            len(used), len(want), len(missing_in_gen), len(missing_in_ref),
        )
    return gen_aligned, ref_aligned, used
