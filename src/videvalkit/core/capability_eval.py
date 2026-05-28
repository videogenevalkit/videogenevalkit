"""Capability-axis evaluation — run all metrics for a capability tag, then
min-max normalize + dedup + aggregate into one capability score.

Per docs/CAPABILITY_TAGS_DESIGN.md §8 (user 2026-05-20 confirmed) — the T5
aggregator that makes the third entry point [`eval --capability X`] produce
an actual number, alongside `eval --bench` and `metric --name`.

Pipeline:
  resolve_capability(tag) → contributors
    → keep runnable ones [per_video_reference_free + functional]
    → run each metric.compute(videos)
    → min-max normalize per metric to [0,1]
    → mean within sub-tag → mean across sub-tags → capability score

Metrics needing refs / prompts / judge, or shells [NotImplementedError],
are SKIPPED with a recorded reason — `eval --capability` is a quick
per-video health read, not a paper number. For ref/prompt/judge metrics use
`metric --name` directly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class CapabilityContributorResult(BaseModel):
    name: str
    raw_score: float | None = None
    normalized: float | None = None
    range: tuple[float, float] | None = None
    status: str = "ok"               # ok | skipped
    skip_reason: str | None = None


class CapabilityEvalResult(BaseModel):
    capability: str
    score: float                     # aggregated normalized score [0,1] or 0 if none
    n_contributors_run: int
    n_skipped: int
    contributors: list[CapabilityContributorResult] = Field(default_factory=list)
    aggregator: str = "mean"
    meta: dict[str, Any] = Field(default_factory=dict)


# Metric kinds that `eval --capability` can run with only a videos dir.
_RUNNABLE_KINDS = {"per_video_reference_free"}


def _list_videos(videos_dir: Path) -> list[Path]:
    exts = {".mp4", ".mov", ".webm", ".avi", ".mkv"}
    return sorted(
        p for p in videos_dir.rglob("*") if p.suffix.lower() in exts
    )


def _minmax_normalize(raw: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (raw - lo) / (hi - lo)))


def run_capability(
    capability: str,
    videos_dir: str | Path,
    aggregator: str = "mean",
    device: str = "auto",
    normalize_ranges: dict[str, tuple[float, float]] | None = None,
) -> CapabilityEvalResult:
    """Run all runnable metrics for ``capability`` on ``videos_dir``.

    ``normalize_ranges`` optionally overrides per-metric [lo, hi] for min-max;
    defaults to [0, 1] for each metric [most vbench dims are already 0-1].
    """
    from videvalkit.core.capability import resolve_capability
    from videvalkit.metrics import SUPPORTED_METRICS, get_metric

    videos_dir = Path(videos_dir)
    videos = _list_videos(videos_dir)
    contributors = resolve_capability(capability)

    results: list[CapabilityContributorResult] = []
    normalized_scores: list[float] = []

    for c in contributors:
        # Only metric contributors are runnable here [bench-dims would need the
        # full bench eval path; they dedup with the metric anyway].
        if c.source_kind != "metric":
            results.append(CapabilityContributorResult(
                name=c.source_name, status="skipped",
                skip_reason="bench-dim contributor [use eval --bench]",
            ))
            continue
        name = c.source_name
        cfg = SUPPORTED_METRICS.get(name, {})
        kind = cfg.get("kind")
        if kind not in _RUNNABLE_KINDS:
            results.append(CapabilityContributorResult(
                name=name, status="skipped",
                skip_reason=f"kind={kind} needs refs/prompts/judge "
                            f"[use metric --name {name}]",
            ))
            continue
        # Try to run it
        try:
            metric = get_metric(name)
            out = metric.compute(videos, device=device) if _accepts_device(metric) \
                else metric.compute(videos)
            raw = _extract_scalar(out)
            lo, hi = (normalize_ranges or {}).get(name, (0.0, 1.0))
            norm = _minmax_normalize(raw, lo, hi)
            results.append(CapabilityContributorResult(
                name=name, raw_score=raw, normalized=norm, range=(lo, hi),
            ))
            normalized_scores.append(norm)
        except NotImplementedError as e:
            results.append(CapabilityContributorResult(
                name=name, status="skipped",
                skip_reason=f"shell [not yet functional]: {str(e)[:80]}",
            ))
        except Exception as e:
            results.append(CapabilityContributorResult(
                name=name, status="skipped",
                skip_reason=f"error: {str(e)[:80]}",
            ))

    # Aggregate
    if normalized_scores:
        if aggregator == "mean":
            score = sum(normalized_scores) / len(normalized_scores)
        elif aggregator == "max":
            score = max(normalized_scores)
        elif aggregator == "min":
            score = min(normalized_scores)
        else:
            raise ValueError(f"unknown capability aggregator {aggregator!r}")
    else:
        score = 0.0

    return CapabilityEvalResult(
        capability=capability,
        score=score,
        n_contributors_run=len(normalized_scores),
        n_skipped=sum(1 for r in results if r.status == "skipped"),
        contributors=results,
        aggregator=aggregator,
        meta={"n_videos": len(videos), "videos_dir": str(videos_dir)},
    )


def _accepts_device(metric: Any) -> bool:
    import inspect
    try:
        sig = inspect.signature(metric.compute)
        return "device" in sig.parameters
    except (ValueError, TypeError):
        return False


def _extract_scalar(out: Any) -> float:
    """Pull a single scalar from a metric result object."""
    # Most result models have a `.score` field
    if hasattr(out, "score"):
        return float(out.score)
    if isinstance(out, (int, float)):
        return float(out)
    if isinstance(out, dict) and "score" in out:
        return float(out["score"])
    raise ValueError(f"cannot extract scalar from metric result {type(out)}")
