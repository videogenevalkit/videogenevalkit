"""Eval profile abstraction — bundle subset + judge tier + frame sampling +
samples_per_prompt + estimated cost into a named lane (quick / standard / full).

Per docs/QUICK_EVAL_DESIGN.md §3 (user 2026-05-20 confirmed):

  quick     — training monitoring, 5-10 min/bench, ρ ≥ 0.85 vs full
  standard  — ablation / iteration,  30-60 min/bench, ρ ≥ 0.95
  full      — paper / leaderboard,  hours/bench, ρ = 1.00 [definition]

Profile is ORTHOGONAL to judge: --profile X chooses subset + frame_sampling;
--judge Y chooses VLM. Combination --profile full --judge paper is the
paper-faithful reproduction lane.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class FrameSamplingSpec(BaseModel):
    """How to sample frames from each video for VLM judges."""

    model_config = ConfigDict(extra="forbid")

    n_frames: int = 8
    mode: Literal["uniform", "middle_frame", "first_last", "random"] = "uniform"


class EstimatedCost(BaseModel):
    """Rough wallclock + token cost estimates for `videvalkit estimate`."""

    model_config = ConfigDict(extra="forbid")

    wallclock_min: float = 0.0
    gpu_hours: float = 0.0
    judge_calls: int = 0
    judge_tokens_in: int = 0
    judge_tokens_out: int = 0


class ProfileSpec(BaseModel):
    """One eval profile."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    subset: str | None = None           # path or named subset; None = full set
    judge: str = "default"               # "default" | "paper" | <registry name>
    frame_sampling: FrameSamplingSpec = Field(default_factory=FrameSamplingSpec)
    samples_per_prompt: int = 1
    estimated: EstimatedCost = Field(default_factory=EstimatedCost)


# Built-in profiles. Subset paths are placeholders [subset_v1 JSONs land in D3
# after offline calibration runs].
SUPPORTED_PROFILES: dict[str, ProfileSpec] = {
    "quick": ProfileSpec(
        name="quick",
        description="Training-time monitoring — fast & stable, 5–10 min per bench",
        subset="quick_v1",  # logical name; resolved per-bench via subset registry
        judge="default",
        frame_sampling=FrameSamplingSpec(n_frames=4, mode="uniform"),
        samples_per_prompt=1,
        estimated=EstimatedCost(
            wallclock_min=8, gpu_hours=0.15,
            judge_calls=60, judge_tokens_in=30000, judge_tokens_out=6000,
        ),
    ),
    "standard": ProfileSpec(
        name="standard",
        description="Reliable eval — Spearman ρ ≥ 0.95 vs full, 30–60 min per bench",
        subset="standard_v1",
        judge="default",
        frame_sampling=FrameSamplingSpec(n_frames=8, mode="uniform"),
        samples_per_prompt=1,
        estimated=EstimatedCost(
            wallclock_min=40, gpu_hours=0.8,
            judge_calls=300, judge_tokens_in=150000, judge_tokens_out=30000,
        ),
    ),
    "full": ProfileSpec(
        name="full",
        description="Paper-faithful reproduction — full prompt set",
        subset=None,  # no subset = full corpus
        judge="default",
        frame_sampling=FrameSamplingSpec(n_frames=8, mode="uniform"),
        samples_per_prompt=5,
        estimated=EstimatedCost(
            wallclock_min=480, gpu_hours=8.0,
            judge_calls=8000, judge_tokens_in=4000000, judge_tokens_out=800000,
        ),
    ),
}


def resolve_profile(name: str | None) -> ProfileSpec:
    """Resolve a profile name to its ProfileSpec.

    ``None`` is treated as ``"full"`` for backward compatibility — existing
    ``videvalkit eval --bench X`` invocations without ``--profile`` continue
    to run the full corpus.
    """
    if name is None:
        return SUPPORTED_PROFILES["full"]
    if name not in SUPPORTED_PROFILES:
        from difflib import get_close_matches
        suggestions = get_close_matches(name, list(SUPPORTED_PROFILES), n=3)
        msg = f"unknown profile {name!r}"
        if suggestions:
            msg += f"; did you mean: {', '.join(suggestions)}?"
        msg += f"\n  available: {list(SUPPORTED_PROFILES)}"
        raise KeyError(msg)
    return SUPPORTED_PROFILES[name]
