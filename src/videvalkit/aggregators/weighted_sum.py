"""WeightedSumAggregator — the simplest, used as default for most adapters.

Configurable per-dimension weights + per-dimension min/max normalization
(VBench v1 uses this exact shape via `scripts/constant.py`).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from videvalkit.core.aggregator import BaseAggregator
from videvalkit.core.types import RawResult, Summary


class WeightedSumAggregator(BaseAggregator):
    name = "weighted_sum"

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        norm_range: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.weights = weights or {}
        self.norm_range = norm_range or {}

    def aggregate(self, results: list[RawResult], **kwargs: Any) -> Summary:
        if not results:
            raise ValueError("no results to aggregate")
        bench = results[0].benchmark
        model = results[0].model
        # mean across prompts for each dim
        per_dim_sum: dict[str, float] = defaultdict(float)
        per_dim_count: dict[str, int] = defaultdict(int)
        for r in results:
            if isinstance(r.score, (int, float)):
                per_dim_sum[r.dimension] += float(r.score)
                per_dim_count[r.dimension] += 1
        per_dim = {d: per_dim_sum[d] / per_dim_count[d] for d in per_dim_sum if per_dim_count[d]}
        per_dim_norm = {d: self._normalize(d, v) for d, v in per_dim.items()}
        overall = self._weighted_overall(per_dim_norm)
        return Summary(
            benchmark=bench,
            model=model,
            per_dimension=per_dim_norm,
            overall=overall,
            n_videos=len(results),
            n_prompts=len({r.prompt_id for r in results}),
            aggregator=self.name,
            meta={"raw_per_dim": per_dim},
        )

    def _normalize(self, dim: str, v: float) -> float:
        if dim not in self.norm_range:
            return v
        lo, hi = self.norm_range[dim]
        if hi == lo:
            return v
        return max(0.0, min(1.0, (v - lo) / (hi - lo)))

    def _weighted_overall(self, per_dim: dict[str, float]) -> float:
        if not per_dim:
            return 0.0
        weights = {d: self.weights.get(d, 1.0) for d in per_dim}
        w_sum = sum(weights.values()) or 1.0
        return sum(per_dim[d] * weights[d] for d in per_dim) / w_sum
