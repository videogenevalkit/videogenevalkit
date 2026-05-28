"""artifact-diagnostic — MLLM-as-judge multi-label artifact detection.

v0.2 port of Artifact-Bench (arXiv 2605.18984) as a standalone diagnostic
metric (the full judge-eval benchmark lands in v0.3). For each video the judge
is shown N frames and asked which of the 30 fine-grained artifact types (see
``artifact_taxonomy``) are present. The metric aggregates a per-video artifact
vector into per-artifact frequency, a per-category breakdown, and a top-K list.

This is a ``per_video_with_vlm_judge`` metric: ``needs_judge=True``. The caller
passes a live judge (built by ``scorers.vlm_judge.build_judge``); without one
the metric raises so the ``--no-judge`` path can skip it cleanly.

Taxonomy overlap is expected: ``flickering`` ~ vbench/temporal-flickering,
``identity_drift`` ~ subject-consistency, ``noise_grain`` ~ imaging-quality.
This metric is a coarse multi-label diagnostic, not a replacement for those.

License: Artifact-Bench is academic-research-only.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from videvalkit.metrics.artifact_taxonomy import (
    ARTIFACT_TYPES,
    N_ARTIFACT_TYPES,
    TYPE_TO_CATEGORY,
)

log = logging.getLogger(__name__)


class ArtifactDiagnosticResult(BaseModel):
    metric: str = "artifact-diagnostic"
    n_videos: int
    # mean number of distinct artifact types per video (lower = cleaner)
    mean_artifacts_per_video: float
    # fraction of videos exhibiting each artifact type (0..1)
    per_artifact_rate: dict[str, float] = Field(default_factory=dict)
    # fraction of videos with >=1 artifact in each top category
    per_category_rate: dict[str, float] = Field(default_factory=dict)
    # most frequent artifacts as (type, rate), descending
    top_artifacts: list[tuple[str, float]] = Field(default_factory=list)
    # per-video detail: video_path -> sorted list of present artifact types
    per_video: dict[str, list[str]] = Field(default_factory=dict)
    judge_model: str = "unknown"


def _extract_content(resp: Any) -> str:
    if isinstance(resp, dict):
        if isinstance(resp.get("content"), str):
            return resp["content"]
        try:
            msg = resp["choices"][0]["message"]["content"]
            if isinstance(msg, list):
                return "".join(
                    (p.get("text", "") if isinstance(p, dict) else str(p)) for p in msg
                )
            return str(msg)
        except (KeyError, IndexError, TypeError):
            pass
    raise RuntimeError(f"unexpected judge response shape: {str(resp)[:200]}")


def _parse_present_artifacts(content: str) -> list[str]:
    """Pull the set of present artifact-type ids out of a judge response.

    Accepts ``{"present": [...]}`` JSON (optionally fenced / with prose), or a
    bare JSON array. Unknown ids are dropped; the result is validated against
    the controlled taxonomy and de-duplicated in taxonomy order.
    """
    valid = set(ARTIFACT_TYPES)
    found: set[str] = set()

    def _collect(seq: Any) -> None:
        if isinstance(seq, list):
            for x in seq:
                if isinstance(x, str) and x.strip() in valid:
                    found.add(x.strip())

    s = content.strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.MULTILINE).strip()
    # Try a JSON object with a "present" key, then a bare array.
    parsed = None
    for cand in (s, s[s.index("{"):s.rindex("}") + 1] if "{" in s and "}" in s else None,
                 s[s.index("["):s.rindex("]") + 1] if "[" in s and "]" in s else None):
        if not cand:
            continue
        try:
            parsed = json.loads(cand)
        except Exception:
            continue
        if isinstance(parsed, dict):
            _collect(parsed.get("present", []))
        elif isinstance(parsed, list):
            _collect(parsed)
        if found:
            break
    # Last resort: substring-match known ids in the raw text.
    if not found:
        for t in valid:
            if re.search(rf"\b{re.escape(t)}\b", s):
                found.add(t)
    return [t for t in ARTIFACT_TYPES if t in found]


def _build_prompt() -> str:
    lines = [
        "You are a strict video-quality inspector for AI-generated videos.",
        "Look at the frames and decide which visual ARTIFACTS are present.",
        "Choose only from this controlled list of artifact type ids:",
        "",
    ]
    for t in ARTIFACT_TYPES:
        lines.append(f"  - {t}  ({TYPE_TO_CATEGORY[t]})")
    lines += [
        "",
        "Report ONLY artifacts you can actually see. If the video looks clean, "
        "return an empty list.",
        'Respond with strict JSON: {"present": ["<id>", ...], "notes": "<short>"}',
        "Use the exact ids above; do not invent new ids.",
    ]
    return "\n".join(lines)


class ArtifactDiagnostic:
    """Multi-label MLLM artifact detection over the Artifact-Bench taxonomy."""

    name = "artifact-diagnostic"
    requires_judge = True

    def compute(
        self,
        videos: list[Path] | Path,
        judge: Any = None,
        n_frames: int = 8,
        mode: str = "holistic",
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ArtifactDiagnosticResult:
        if judge is None:
            raise NotImplementedError(
                "artifact-diagnostic needs a VLM judge. Pass --judge <name> "
                "(or run with a judge endpoint); it is excluded by --no-judge."
            )
        vids = [videos] if isinstance(videos, (str, Path)) else list(videos)
        if not vids:
            return ArtifactDiagnosticResult(
                n_videos=0, mean_artifacts_per_video=0.0,
                judge_model=getattr(judge, "model", "unknown"),
            )

        prompt = _build_prompt()
        per_video: dict[str, list[str]] = {}
        type_counts = dict.fromkeys(ARTIFACT_TYPES, 0)
        category_hit_counts: dict[str, int] = {}
        n_scored = 0
        for v in sorted(vids, key=str):
            v = Path(v)
            try:
                resp = judge.chat_with_frames(
                    video_path=str(v), prompt=prompt, mode=mode,
                    n_frames=n_frames, max_tokens=max_tokens, temperature=temperature,
                )
                present = _parse_present_artifacts(_extract_content(resp))
            except Exception as e:  # noqa: BLE001 — one bad video shouldn't kill the run
                log.warning("artifact-diagnostic: %s failed: %s", v, e)
                continue
            per_video[str(v)] = present
            n_scored += 1
            for t in present:
                type_counts[t] += 1
            for cat in {TYPE_TO_CATEGORY[t] for t in present}:
                category_hit_counts[cat] = category_hit_counts.get(cat, 0) + 1

        denom = n_scored or 1
        per_artifact_rate = {t: type_counts[t] / denom for t in ARTIFACT_TYPES}
        per_category_rate = {
            cat: category_hit_counts.get(cat, 0) / denom
            for cat in sorted({TYPE_TO_CATEGORY[t] for t in ARTIFACT_TYPES})
        }
        mean_artifacts = sum(len(v) for v in per_video.values()) / denom
        top = sorted(
            ((t, r) for t, r in per_artifact_rate.items() if r > 0),
            key=lambda kv: kv[1], reverse=True,
        )[:10]

        return ArtifactDiagnosticResult(
            n_videos=n_scored,
            mean_artifacts_per_video=mean_artifacts,
            per_artifact_rate=per_artifact_rate,
            per_category_rate=per_category_rate,
            top_artifacts=top,
            per_video=per_video,
            judge_model=getattr(judge, "model", "unknown"),
        )


assert N_ARTIFACT_TYPES == 30, f"expected 30 artifact types, got {N_ARTIFACT_TYPES}"
