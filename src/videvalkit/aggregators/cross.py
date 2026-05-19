"""Cross-benchmark aggregation — combine multiple benchmarks' summaries.

Given a list of `Summary` objects from any combination of VBench / VBench2 /
Video-Bench / WorldJen runs over the **same set of models**, produces:

  * Per-benchmark normalized scores (z-score across models within benchmark)
  * A unified ranking (mean of normalized scores)
  * Bradley-Terry rating + 95% CI from the implied pairwise preferences
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from videvalkit.core.types import Summary


def _overall_float(s: Summary) -> float:
    o = s.overall
    if isinstance(o, dict):
        # Prefer "Total" if present (VBench-2.0); else the largest entry.
        if "Total" in o:
            return float(o["Total"])
        return float(max(o.values())) if o else 0.0
    return float(o)


def combine_summaries(summaries: list[Summary]) -> dict[str, Any]:
    """Cross-benchmark aggregation.

    Returns::

        {
          "per_benchmark": {bench: {model: overall_score}},
          "normalized":    {bench: {model: zscore}},
          "unified":       {model: mean_zscore},
          "ranking":       [{"model": "...", "score": float, "rank": int}, ...],
          "bt":            {model: {"mean": float, "lower_95": float, "upper_95": float}},
        }
    """
    per_bench: dict[str, dict[str, float]] = defaultdict(dict)
    for s in summaries:
        per_bench[s.benchmark][s.model] = _overall_float(s)

    # Within-benchmark z-score normalization
    normalized: dict[str, dict[str, float]] = {}
    for bench, mp in per_bench.items():
        vals = np.array(list(mp.values()))
        mu = float(vals.mean()) if len(vals) else 0.0
        sd = float(vals.std(ddof=0)) if len(vals) > 1 else 1.0
        sd = sd or 1.0
        normalized[bench] = {m: (v - mu) / sd for m, v in mp.items()}

    # Unified per-model: mean across benchmarks it participated in
    all_models = sorted({m for mp in per_bench.values() for m in mp})
    unified: dict[str, float] = {}
    for m in all_models:
        zs = [normalized[b][m] for b in normalized if m in normalized[b]]
        unified[m] = float(np.mean(zs)) if zs else 0.0

    ranking = sorted(
        [{"model": m, "score": s} for m, s in unified.items()],
        key=lambda r: r["score"], reverse=True,
    )
    for i, r in enumerate(ranking):
        r["rank"] = i + 1

    # Implied BT from per-benchmark scores: each (bench, model_pair) → matchup
    bt = _implied_bt(per_bench)

    return {
        "per_benchmark": dict(per_bench),
        "normalized":    normalized,
        "unified":       unified,
        "ranking":       ranking,
        "bt":            bt,
    }


def _implied_bt(per_bench: dict[str, dict[str, float]]) -> dict[str, Any]:
    from videvalkit.aggregators.bt import (
        bootstrap_bt, compute_bradley_terry, matchups_from_per_prompt_scores,
    )
    # Reuse the prompt-bootstrap helper by treating each benchmark as a "prompt".
    per_prompt = {model: {bench: score for bench, models in per_bench.items()
                          if model in models for score in [models[model]]}
                  for bench in per_bench for model in per_bench[bench]}
    # rebuild per_prompt {model: {bench: score}} cleanly
    per_prompt = {}
    for bench, mp in per_bench.items():
        for model, score in mp.items():
            per_prompt.setdefault(model, {})[bench] = float(score)
    matchups = matchups_from_per_prompt_scores(per_prompt)
    if not matchups:
        return {}
    point = compute_bradley_terry([(m["winner"], m["loser"]) for m in matchups])
    ci = bootstrap_bt(matchups, n_bootstrap=200)
    return {"point": point, "with_ci": ci, "n_matchups": len(matchups)}
