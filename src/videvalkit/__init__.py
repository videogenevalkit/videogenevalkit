"""videvalkit — unified evaluation toolkit for generative video benchmarks."""

# Add upstream-repo subdirs to sys.path so adapters can `from vbench import …`
# without a per-upstream pip install. Upstream repos are cloned by
# `videvalkit fetch-upstream`; missing ones are silently skipped here and
# surface as a clear ImportError when the user invokes that adapter.
from videvalkit import _upstream_paths  # noqa: F401

from videvalkit.core.benchmark import BaseBenchmark
from videvalkit.core.scorer import BaseScorer, ScoreContext, ScoreResult
from videvalkit.core.aggregator import BaseAggregator
from videvalkit.core.types import PromptItem, RawResult, Summary, VideoSpec
from videvalkit.core.layout import WorkspaceLayout
from videvalkit.runner import run

__version__ = "0.1.0"

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
    "run",
]
