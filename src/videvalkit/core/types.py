"""Shared dataclasses passed across Benchmark / Scorer / Aggregator boundaries.

Kept in one file because they are tightly coupled and small.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PromptItem(BaseModel):
    """One prompt expected by a benchmark.

    A benchmark's prompt list is iterated as a sequence of these.
    A prompt may belong to a single dimension (VBench style) or to multiple
    (Video-Bench's cross-dim prompts) — `dimensions` is therefore a list.
    """

    model_config = ConfigDict(frozen=True)

    prompt_id: str
    text: str
    dimensions: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class VideoSpec(BaseModel):
    """A single video file expected on disk, with the metadata needed to score it."""

    model_config = ConfigDict(frozen=True)

    path: Path
    prompt_id: str
    model_name: str
    dimension: str | None = None
    sample_index: int = 0
    meta: dict[str, Any] = Field(default_factory=dict)


class RawResult(BaseModel):
    """One (video, scorer) result, before any cross-prompt aggregation.

    This is the unit written to `results/raw/{benchmark}/{model}/{dim}/{prompt_id}.json`.
    Each benchmark's adapter must normalize upstream outputs into this schema.
    """

    benchmark: str
    model: str
    dimension: str
    prompt_id: str
    score: float | dict[str, Any] | list[float]
    judge: str | None = None
    scorer: str | None = None
    video_path: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    meta: dict[str, Any] = Field(default_factory=dict)


class Summary(BaseModel):
    """Per-(benchmark, model) aggregated result.

    Stored at `results/summary/{benchmark}/{model}.json`.
    """

    benchmark: str
    model: str
    per_dimension: dict[str, float]
    overall: float | dict[str, float]
    n_videos: int
    n_prompts: int
    aggregator: str
    meta: dict[str, Any] = Field(default_factory=dict)


WorkloadKind = Literal["gpu_metric", "vlm_judge_http", "vlm_judge_api", "cpu"]
"""Used by Scheduler to route a Scorer to the right worker pool."""
