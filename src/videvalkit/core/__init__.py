"""Core abstractions: Benchmark / Scorer / Aggregator + shared types."""

from videvalkit.core.benchmark import BaseBenchmark
from videvalkit.core.scorer import BaseScorer, ScoreContext, ScoreResult
from videvalkit.core.aggregator import BaseAggregator
from videvalkit.core.types import PromptItem, RawResult, Summary, VideoSpec
from videvalkit.core.layout import WorkspaceLayout

__all__ = [
    "BaseBenchmark",
    "BaseScorer",
    "ScoreContext",
    "ScoreResult",
    "BaseAggregator",
    "PromptItem",
    "RawResult",
    "Summary",
    "VideoSpec",
    "WorkspaceLayout",
]
