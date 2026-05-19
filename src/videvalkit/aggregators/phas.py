"""PHASAggregator — Per-dimension Human-Anchored Score.

Two roles:
  1. **Apply weights** (default mode): given per-(model, prompt, dim) scores
     and a 16-dim weight vector, produce per-(model, prompt) PHAS and
     per-model averages. Variance penalty applied per-video.

  2. **Calibrate weights** (optional): given pairwise human preferences,
     fit a non-negative ridge logistic regressor to derive weights.
     This is a heavyweight call (sklearn + scipy.optimize), kept behind
     `.calibrate()` so the common path stays light.

Default weights are the WorldJen hand-tuned baseline from
`worldjen_local/human_eval/calibrate_phas.py:HAND`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from videvalkit.core.aggregator import BaseAggregator
from videvalkit.core.types import RawResult, Summary

# ── PHAS weights (production calibration) ────────────────────────────── #
# Non-negative ridge logistic regression on 1653 calibration annotations
# (2696 total; 7 annotators, A1-retest excluded). CV 64.4 %, val 62.6 %.
# Source: upstream WorldJen-benchmarking-subsystem/vlm_eval/unified_analyzer.py:48-69
# (verbatim — sum is 1.0 ± 1e-3 by construction).
WORLDJEN_CALIBRATED_WEIGHTS: dict[str, float] = {
    "subject_consistency":   0.0907,
    "scene_consistency":     0.0000,
    "motion_smoothness":     0.0743,
    "temporal_flickering":   0.0144,
    "inertial_consistency":  0.0542,
    "physical_mechanics":    0.0575,
    "object_permanence":     0.0000,
    "human_fidelity":        0.0000,
    "dynamic_degree":        0.0000,
    "semantic_adherence":    0.2638,
    "spatial_relationship":  0.0000,
    "semantic_drift":        0.1261,
    "composition_framing":   0.1304,
    "lighting_volumetric":   0.0534,
    "color_harmony":         0.0816,
    "structural_gestalt":    0.0535,
}
assert abs(sum(WORLDJEN_CALIBRATED_WEIGHTS.values()) - 1.0) < 1e-3, \
    "calibrated PHAS weights must sum to 1"

# Legacy hand-tuned baseline (kept for back-compat; not the production set).
WORLDJEN_HAND_WEIGHTS: dict[str, float] = {
    "subject_consistency": 0.07, "scene_consistency": 0.06,
    "motion_smoothness": 0.05,   "temporal_flickering": 0.04,
    "inertial_consistency": 0.09, "physical_mechanics": 0.09,
    "object_permanence": 0.09,   "human_fidelity": 0.07,
    "dynamic_degree": 0.04,      "semantic_adherence": 0.09,
    "spatial_relationship": 0.06, "semantic_drift": 0.06,
    "composition_framing": 0.05, "lighting_volumetric": 0.05,
    "color_harmony": 0.04,       "structural_gestalt": 0.05,
}

# Variance penalty: phas_final = phas_base * (1 - min(CAP, mean_var * MULT))
VAR_PENALTY_MULT = 0.05
VAR_PENALTY_CAP  = 0.30


class PHASAggregator(BaseAggregator):
    name = "phas"

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        weights_path: str | Path | None = None,
        var_penalty_mult: float = VAR_PENALTY_MULT,
        var_penalty_cap: float = VAR_PENALTY_CAP,
    ) -> None:
        # Default to the upstream-published **calibrated** weights so our
        # PHAS scores are directly comparable to the WorldJen leaderboard.
        # Pass ``weights=WORLDJEN_HAND_WEIGHTS`` to use the legacy hand-tuned
        # baseline instead.
        self.weights = dict(weights) if weights else dict(WORLDJEN_CALIBRATED_WEIGHTS)
        if weights_path:
            self._load_weights(Path(weights_path))
        self.var_penalty_mult = var_penalty_mult
        self.var_penalty_cap = var_penalty_cap

    def _load_weights(self, path: Path) -> None:
        data = json.loads(path.read_text())
        # Accept either a flat {dim: w} or the calibrate_phas.py JSON shape.
        if "phas_calibration" in data:
            data = data["phas_calibration"]["nonneg_ridge"]["weights"]
        self.weights = {k: float(v) for k, v in data.items()}

    def aggregate(self, results: list[RawResult], **kwargs: Any) -> Summary:
        """Aggregate one (benchmark, model)'s RawResult list into a Summary.

        results must all share `benchmark` and `model`; we compute:
          - per_dimension: mean dim score across prompts
          - per_prompt:    PHAS per prompt (in .meta)
          - overall:       mean PHAS across prompts (variance-penalized)
        """
        if not results:
            raise ValueError("no results")
        bench = results[0].benchmark
        model = results[0].model

        # Bucket by (prompt_id, dim)
        by_prompt_dim: dict[tuple[str, str], list[float]] = defaultdict(list)
        for r in results:
            if isinstance(r.score, (int, float)) and not np.isnan(float(r.score)):
                by_prompt_dim[(r.prompt_id, r.dimension)].append(float(r.score))
        # Per-(prompt, dim) mean
        per_prompt_dim = {k: float(np.mean(v)) for k, v in by_prompt_dim.items()}

        prompts = sorted({pid for pid, _ in per_prompt_dim})
        per_prompt_phas: dict[str, float] = {}
        for pid in prompts:
            dims_for_p = {d: per_prompt_dim[(pid, d)]
                          for (p, d) in per_prompt_dim if p == pid}
            phas_base = self._weighted_mean(dims_for_p)
            # Variance across dimensions for this video.
            vals = list(dims_for_p.values())
            var = float(np.var(vals)) if len(vals) > 1 else 0.0
            penalty = min(self.var_penalty_cap, var * self.var_penalty_mult)
            per_prompt_phas[pid] = phas_base * (1.0 - penalty)

        # Per-dim average across prompts
        per_dim: dict[str, float] = {}
        for d in self.weights.keys():
            vals = [per_prompt_dim[(pid, d)] for pid in prompts if (pid, d) in per_prompt_dim]
            if vals:
                per_dim[d] = float(np.mean(vals))

        overall = float(np.mean(list(per_prompt_phas.values()))) if per_prompt_phas else 0.0

        return Summary(
            benchmark=bench,
            model=model,
            per_dimension=per_dim,
            overall=overall,
            n_videos=len(results),
            n_prompts=len(prompts),
            aggregator=self.name,
            meta={
                "per_prompt": per_prompt_phas,
                "weights":    self.weights,
                "var_penalty": {"mult": self.var_penalty_mult, "cap": self.var_penalty_cap},
            },
        )

    def aggregate_cross(self, summaries: list[Summary], **kwargs: Any) -> dict[str, Any]:
        """Cross-model summary: rank by overall PHAS, expose per_prompt for BT."""
        ranking = sorted(summaries, key=lambda s: float(s.overall), reverse=True)
        return {
            "ranking": [
                {"model": s.model, "phas": float(s.overall), "n_prompts": s.n_prompts}
                for s in ranking
            ],
            "per_prompt": {s.model: s.meta.get("per_prompt", {}) for s in summaries},
            "weights": self.weights,
        }

    def _weighted_mean(self, dim_scores: dict[str, float]) -> float:
        if not dim_scores:
            return 0.0
        # Use only weights for dims we actually observed (renormalize).
        active = {d: self.weights.get(d, 0.0) for d in dim_scores if self.weights.get(d, 0.0) > 0}
        w_sum = sum(active.values())
        if w_sum <= 0:
            return float(np.mean(list(dim_scores.values())))
        return sum(dim_scores[d] * active[d] for d in active) / w_sum

    # ---- optional: weight calibration --------------------------------------
    @staticmethod
    def calibrate(
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        dim_names: list[str],
        Cs: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
        n_splits: int = 5,
    ) -> dict[str, Any]:
        """Non-negative ridge logistic regression for PHAS weights.

        Inputs match `worldjen_local/human_eval/calibrate_phas.py:build_matrix`:
          X: (n, 16) dim-score differences (Model A - Model B)
          y: (n,)    1 if Model A won
          w: (n,)    annotation weights

        Returns {"weights": {dim: w}, "C": float, "cv_accuracy": float}.
        Requires scipy + sklearn (worldjen env only).
        """
        try:
            from scipy.optimize import minimize
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("PHAS calibration requires scipy; install in worldjen env") from e

        n = len(y)
        rng = np.random.default_rng(42)
        fold_idx = np.array_split(rng.permutation(n), n_splits)

        def obj(beta, X_, y_, w_, C):
            logits = np.clip(X_ @ beta, -30, 30)
            prob = np.clip(1 / (1 + np.exp(-logits)), 1e-9, 1 - 1e-9)
            ll = -np.sum(w_ * (y_ * np.log(prob) + (1 - y_) * np.log(1 - prob)))
            return ll + (1.0 / (2 * C)) * np.sum(beta ** 2)

        def grad(beta, X_, y_, w_, C):
            logits = np.clip(X_ @ beta, -30, 30)
            prob = 1 / (1 + np.exp(-logits))
            return X_.T @ (w_ * (prob - y_)) + beta / C

        best = (-1.0, None, None)  # (cv_acc, C, beta)
        for C in Cs:
            accs: list[float] = []
            beta_last: np.ndarray | None = None
            for fi in range(n_splits):
                val = fold_idx[fi]
                trn = np.concatenate([fold_idx[j] for j in range(n_splits) if j != fi])
                res = minimize(
                    obj, x0=np.ones(X.shape[1]) * 0.1,
                    args=(X[trn], y[trn], w[trn], C),
                    jac=grad, method="L-BFGS-B",
                    bounds=[(0, None)] * X.shape[1],
                    options={"maxiter": 2000, "ftol": 1e-10},
                )
                beta_last = res.x
                preds = (1 / (1 + np.exp(-X[val] @ res.x)) >= 0.5).astype(int)
                accs.append(float((preds == y[val]).mean()))
            mean_acc = float(np.mean(accs))
            if mean_acc > best[0]:
                best = (mean_acc, C, beta_last)
        cv_acc, C_best, beta_best = best
        if beta_best is None:
            raise RuntimeError("PHAS calibration produced no solution")
        # Normalize weights to sum to 1.
        s = beta_best.sum()
        if s > 0:
            beta_best = beta_best / s
        return {
            "weights":     dict(zip(dim_names, beta_best.tolist())),
            "C":           float(C_best),
            "cv_accuracy": cv_acc,
        }
