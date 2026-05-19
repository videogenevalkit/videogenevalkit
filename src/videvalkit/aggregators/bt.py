"""BradleyTerryAggregator — pairwise-comparison rating with bootstrap CIs.

Ported from `worldjen_local/scoring/bootstrap_bt_rating.py`.

Two layers:
  * `compute_bradley_terry(matchups)` — iterative MLE → {model: elo-like score}
  * `bootstrap_bt(matchups, n_bootstrap)` — prompt-level resampling → CI

Entry point used by the toolkit runner is `aggregate_cross(summaries)`:
each Summary must carry per-(model, prompt) mean scores in its `.meta`
under the key `"per_prompt"` so we can derive (winner, loser) pairs.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any

import numpy as np

from videvalkit.core.aggregator import BaseAggregator
from videvalkit.core.types import RawResult, Summary


def compute_bradley_terry(
    matchups: list[tuple[str, str]],
    n_iters: int = 500,
) -> dict[str, float]:
    """Iterative MLE for Bradley-Terry; returns Elo-scaled scores centered at 1500."""
    models = sorted({m for pair in matchups for m in pair})
    if len(models) < 2:
        return {m: 1500.0 for m in models}
    idx = {m: i for i, m in enumerate(models)}
    n = len(models)
    W = np.zeros((n, n))
    for winner, loser in matchups:
        W[idx[winner], idx[loser]] += 1

    strength = np.ones(n)
    for _ in range(n_iters):
        new_s = np.zeros(n)
        for i in range(n):
            wins_i = W[i].sum()
            denom = sum(
                (W[i, j] + W[j, i]) / (strength[i] + strength[j])
                for j in range(n)
                if j != i and (W[i, j] + W[j, i]) > 0
            )
            new_s[i] = wins_i / denom if denom > 0 and wins_i > 0 else 1e-6
        s = new_s.sum()
        strength = new_s / s * n if s > 0 else strength

    log_s = np.log(strength + 1e-9)
    log_s -= log_s.mean()
    return {models[i]: 1500 + log_s[i] * 400 / math.log(10) for i in range(n)}


def matchups_from_per_prompt_scores(
    per_prompt: dict[str, dict[str, float]],
    tie_eps: float = 0.01,
) -> list[dict[str, Any]]:
    """Convert {model: {prompt_id: score}} into pairwise matchups.

    Output items: {"prompt_id", "winner", "loser", "model_a", "model_b", "score_diff"}.
    """
    out: list[dict[str, Any]] = []
    models = sorted(per_prompt.keys())
    if len(models) < 2:
        return out
    all_prompts = sorted({pid for m in models for pid in per_prompt[m].keys()})
    for pid in all_prompts:
        present = [m for m in models if pid in per_prompt[m]]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                ma, mb = present[i], present[j]
                sa, sb = per_prompt[ma][pid], per_prompt[mb][pid]
                if abs(sa - sb) < tie_eps:
                    continue
                winner, loser = (ma, mb) if sa > sb else (mb, ma)
                out.append({
                    "prompt_id": pid, "winner": winner, "loser": loser,
                    "model_a": ma, "model_b": mb, "score_diff": abs(sa - sb),
                })
    return out


def bootstrap_bt(
    matchups: list[dict[str, Any]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Prompt-level bootstrap → per-model 95% CI on BT rating."""
    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for m in matchups:
        by_prompt[m["prompt_id"]].append(m)
    prompts = list(by_prompt.keys())
    for _ in range(n_bootstrap):
        sampled = rng.choices(prompts, k=len(prompts))
        pairs = [(m["winner"], m["loser"]) for p in sampled for m in by_prompt[p]]
        if len(pairs) < 2:
            continue
        bt = compute_bradley_terry(pairs)
        for model, r in bt.items():
            samples[model].append(r)
    out: dict[str, dict[str, float]] = {}
    for model, vals in samples.items():
        arr = np.array(vals)
        out[model] = {
            "mean":     float(arr.mean()),
            "std":      float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "lower_95": float(np.percentile(arr, 2.5)),
            "upper_95": float(np.percentile(arr, 97.5)),
            "n_samples": len(arr),
        }
    return out


class BradleyTerryAggregator(BaseAggregator):
    name = "bt"

    def __init__(self, bootstrap: int = 1000, seed: int = 42, tie_eps: float = 0.01) -> None:
        self.bootstrap = bootstrap
        self.seed = seed
        self.tie_eps = tie_eps

    def aggregate(self, results: list[RawResult], **kwargs: Any) -> Summary:
        """BT is inherently cross-model; with one model we just return the mean."""
        if not results:
            raise ValueError("no results")
        # Group per (model, prompt_id) into mean dim score.
        by_model_prompt: dict[tuple[str, str], list[float]] = defaultdict(list)
        for r in results:
            if isinstance(r.score, (int, float)):
                by_model_prompt[(r.model, r.prompt_id)].append(float(r.score))
        per_prompt: dict[str, dict[str, float]] = defaultdict(dict)
        for (m, pid), vals in by_model_prompt.items():
            per_prompt[m][pid] = float(np.mean(vals))

        if len(per_prompt) < 2:
            model, pp = next(iter(per_prompt.items()))
            return Summary(
                benchmark=results[0].benchmark,
                model=model,
                per_dimension={},
                overall=float(np.mean(list(pp.values()))) if pp else 0.0,
                n_videos=len(results),
                n_prompts=len(pp),
                aggregator=self.name,
                meta={"note": "BT needs >= 2 models; returned mean score"},
            )
        cross = self.aggregate_cross_from_per_prompt(per_prompt)
        # Wrap into a Summary for the first model (caller can request the others)
        first_model = sorted(per_prompt.keys())[0]
        return Summary(
            benchmark=results[0].benchmark,
            model=first_model,
            per_dimension={},
            overall=cross["bt_with_ci"][first_model],
            n_videos=len(results),
            n_prompts=len({r.prompt_id for r in results}),
            aggregator=self.name,
            meta={"bt_cross_models": cross},
        )

    def aggregate_cross(self, summaries: list[Summary], **kwargs: Any) -> dict[str, Any]:
        """Build per_prompt scores from a list of within-benchmark Summary objects.

        Each Summary's `.meta["raw_per_dim"]` / `.meta["per_prompt"]` is consulted;
        absent that, we fall back to overall as the only number we have.
        """
        per_prompt: dict[str, dict[str, float]] = {}
        for s in summaries:
            pp = s.meta.get("per_prompt") or {}
            if pp:
                per_prompt[s.model] = pp
            else:
                per_prompt[s.model] = {"_overall": float(s.overall) if isinstance(s.overall, (int, float)) else 0.0}
        return self.aggregate_cross_from_per_prompt(per_prompt)

    def aggregate_cross_from_per_prompt(
        self,
        per_prompt: dict[str, dict[str, float]],
    ) -> dict[str, Any]:
        matchups = matchups_from_per_prompt_scores(per_prompt, tie_eps=self.tie_eps)
        point = compute_bradley_terry([(m["winner"], m["loser"]) for m in matchups])
        ci = bootstrap_bt(matchups, n_bootstrap=self.bootstrap, seed=self.seed)
        return {
            "bt_point":   {k: float(v) for k, v in point.items()},
            "bt_with_ci": ci,
            "n_matchups": len(matchups),
            "n_prompts":  len({m["prompt_id"] for m in matchups}),
        }
