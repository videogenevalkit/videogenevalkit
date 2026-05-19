"""BaseAggregator — collapse raw per-(video, dim) scores into final numbers.

This is the toolkit's first-class layer for cross-benchmark / cross-model
analysis (PHAS weighting, Bradley-Terry rating, weighted sum, etc.).

A Benchmark adapter may call into one of these from its `aggregate()`,
or the runner can apply a cross-benchmark aggregator post-hoc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from videvalkit.core.types import RawResult, Summary


class BaseAggregator(ABC):
    """Combine many RawResult objects into one or more Summary objects.

    Aggregators have two flavors:
      * **Within-benchmark**: per (benchmark, model) -> Summary. Used by adapters.
      * **Cross-benchmark**:  reads multiple benchmarks' summaries, produces
        unified PHAS / BT rankings. Operates on a list[Summary] instead.
    """

    name: str = "base"

    @abstractmethod
    def aggregate(
        self,
        results: list[RawResult],
        **kwargs: Any,
    ) -> Summary:  # pragma: no cover - abstract
        ...

    def aggregate_cross(
        self,
        summaries: list[Summary],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Optional: cross-benchmark / cross-model aggregation.

        Returns a free-form dict (BT ratings, PHAS weights, etc.) — the caller
        decides how to serialize it.
        """
        raise NotImplementedError(
            f"{self.name}.aggregate_cross is not implemented; this aggregator is within-benchmark only"
        )
