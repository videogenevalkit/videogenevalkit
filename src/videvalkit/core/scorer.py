"""BaseScorer — the minimum unit of evaluation: (video, prompt) -> score.

Scorers are intentionally below the Benchmark level so they can be reused.
A Benchmark composes one or more Scorers (one per dimension typically).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from videvalkit.core.types import WorkloadKind


class ScoreContext(BaseModel):
    """Everything a Scorer needs to score one (video, prompt) pair.

    Decoded frames / extracted features go in `cache` so neighboring scorers
    on the same video can reuse them within a single dispatch batch.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    video_path: Path
    prompt_text: str
    prompt_id: str
    dimension: str
    model_name: str
    sample_index: int = 0
    cache: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)


class ScoreResult(BaseModel):
    """Output of one BaseScorer.score() call."""

    score: float | dict[str, Any]
    raw: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)


class BaseScorer(ABC):
    """A single, atomically schedulable scoring step.

    Subclasses declare:
      * `name`   — human-readable identifier
      * `kind`   — workload class (`Scheduler` routes by this)
      * `score(ctx)` — the actual work

    Subclasses MAY override:
      * `required_inputs()` — strings like {"frames", "video"} so the runner
        knows what to materialize before dispatching
      * `setup()` / `teardown()` — load/unload weights, open HTTP sessions
    """

    name: str = "base"
    kind: WorkloadKind = "cpu"

    def required_inputs(self) -> set[str]:
        """Names of artifacts the runner should materialize into ctx.cache."""
        return {"video"}

    def setup(self) -> None:
        """One-shot init in the worker that will own this scorer."""

    def teardown(self) -> None:
        """One-shot cleanup."""

    @abstractmethod
    def score(self, ctx: ScoreContext) -> ScoreResult:  # pragma: no cover - abstract
        ...
