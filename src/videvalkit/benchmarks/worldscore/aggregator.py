"""WorldScore aggregator — mirrors upstream's ``run_evaluate.py`` exactly.

Per-instance normalization uses ``aspect_info`` from
``worldscore/benchmark/utils/utils.py``:
  - z-score for clip_score, clip_iqa+, clip_aesthetic, motion_accuracy, optical_flow
  - min-max with empirical bounds for object_detection, reprojection_error,
    gram_matrix, camera_error (geomean over R/T), motion_smoothness
    (arith-mean over MSE/SSIM/LPIPS)

Per-dim aggregation: flat mean of normalized scores × 100 (run_evaluate.py:54).

Headlines (run_evaluate.py:165-166):
  WorldScore-Static  = mean of the 7 static dim values
  WorldScore-Dynamic = mean of ALL 10 dim values (static + dynamic)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any


# Mirror upstream's ``worldscore_list``.
SPLIT_DIMS: dict[str, list[str]] = {
    "static":  ["camera_control", "object_control", "content_alignment",
                "3d_consistency", "photometric_consistency",
                "style_consistency", "subjective_quality"],
    "dynamic": ["motion_accuracy", "motion_magnitude", "motion_smoothness"],
}

# Mirror upstream's ``aspect_info`` (worldscore/benchmark/utils/utils.py).
NORM_CFG: dict[str, dict] = {
    "camera_control": {"kind": "minmax_vector",
        "empirical_max": [15.0, 0.5], "empirical_min": [0.0, 0.0],
        "higher_is_better": [False, False], "agg": "geomean"},
    "object_control": {"kind": "minmax",
        "empirical_max": 1.0, "empirical_min": 0.0, "higher_is_better": True},
    "content_alignment": {"kind": "zscore",
        "avg": 26.67, "std": 0.8875, "z_max": 1.2741, "z_min": -1.5950,
        "range": [0.25, 0.75], "higher_is_better": True},
    "3d_consistency": {"kind": "minmax",
        "empirical_max": 1.0719, "empirical_min": 0.0, "higher_is_better": False},
    "photometric_consistency": {"kind": "minmax",
        "empirical_max": 1.1920, "empirical_min": 0.0, "higher_is_better": False},
    "style_consistency": {"kind": "minmax",
        "empirical_max": 0.0070, "empirical_min": 0.0, "higher_is_better": False},
    "motion_magnitude": {"kind": "zscore",
        "avg": 3.2425, "std": 3.4505, "z_max": 2.7498, "z_min": -0.8638,
        "range": [0.25, 0.75], "higher_is_better": True},
    "motion_smoothness": {"kind": "minmax_vector",
        "empirical_max": [82.4014, 1.0, 0.0228],
        "empirical_min": [0.0, 0.9224, 0.0],
        "higher_is_better": [False, True, False], "agg": "arith_mean"},
    "motion_accuracy": {"kind": "zscore",
        "avg": -0.1965, "std": 2.3687, "z_max": 1.5180, "z_min": -1.6147,
        "range": [0.25, 0.75], "higher_is_better": True},
    # subjective_quality is the mean of two metric components below.
    "subjective_quality.clip_aesthetic": {"kind": "zscore",
        "avg": 5.5952, "std": 0.2561, "z_max": 1.9741, "z_min": -2.9342,
        "range": [0.25, 0.75], "higher_is_better": True},
    "subjective_quality.clip_iqa+": {"kind": "zscore",
        "avg": 0.5842, "std": 0.0441, "z_max": 1.8703, "z_min": -1.8342,
        "range": [0.25, 0.75], "higher_is_better": True},
}


def _norm_zscore(s: float, cfg: dict) -> float:
    x = ((s - cfg["avg"]) / cfg["std"] - cfg["z_min"]) / (cfg["z_max"] - cfg["z_min"])
    if not cfg["higher_is_better"]: x = 1.0 - x
    r = cfg["range"]
    n = r[0] + (r[1] - r[0]) * x
    return max(0.0, min(1.0, n))


def _norm_minmax(s: float, cfg: dict) -> float:
    s = max(cfg["empirical_min"], min(cfg["empirical_max"], s))
    rng = cfg["empirical_max"] - cfg["empirical_min"]
    n = (s - cfg["empirical_min"]) / rng if rng else 0.0
    if not cfg["higher_is_better"]: n = 1.0 - n
    return n


def _norm_minmax_vector(vec, cfg: dict) -> float:
    parts = []
    for s, mx, mn, hib in zip(vec, cfg["empirical_max"], cfg["empirical_min"],
                              cfg["higher_is_better"]):
        s = max(mn, min(mx, s))
        rng = mx - mn
        n = (s - mn) / rng if rng else 0.0
        if not hib: n = 1.0 - n
        parts.append(n)
    if not parts: return 0.0
    if cfg.get("agg") == "geomean":
        p = 1.0
        for v in parts: p *= max(v, 0.0)
        return p ** (1.0 / len(parts))
    return sum(parts) / len(parts)


def normalize_instance(dim: str, raw_score, sub_metric: str | None = None) -> float | None:
    """Normalize one per-video raw score to [0, 1] using upstream's formula."""
    key = f"{dim}.{sub_metric}" if sub_metric and f"{dim}.{sub_metric}" in NORM_CFG else dim
    cfg = NORM_CFG.get(key)
    if cfg is None: return None
    try:
        if cfg["kind"] == "zscore":
            return _norm_zscore(float(raw_score), cfg)
        if cfg["kind"] == "minmax":
            return _norm_minmax(float(raw_score), cfg)
        if cfg["kind"] == "minmax_vector":
            if isinstance(raw_score, (list, tuple)):
                return _norm_minmax_vector(
                    raw_score[: len(cfg["empirical_max"])], cfg,
                )
    except Exception:
        return None
    return None


def compute_worldscore(rows: list[dict]) -> dict[str, Any]:
    """Aggregate per-video rows into per-dim means + WorldScore-Static/Dynamic.

    Each input row should have keys: ``prompt_id``, ``dimension``, ``score``,
    and optionally ``metric`` for subjective_quality. ``split`` is inferred
    from the prompt_id prefix (``ws_dyn_`` vs ``ws_sta_``) if absent.

    Returns ``{"per_dim", "headlines", "method"}``.
    """
    # Group by (prompt_id, dim, sub_metric)
    by_key: dict[tuple, Any] = {}
    for r in rows:
        pid = r.get("prompt_id") or r.get("entry_id")
        if not pid: continue
        dim = r.get("dimension")
        if dim is None: continue
        sub = r.get("metric") if dim == "subjective_quality" else None
        key = (pid, dim, sub)
        val = r.get("score_capped", r.get("score"))
        if val is None: continue
        split = r.get("split") or ("dynamic" if pid.startswith("ws_dyn_") else "static")
        by_key[key] = (val, split)

    norm_by_dim: dict[str, list[float]] = defaultdict(list)
    raw_by_dim: dict[str, list[float]] = defaultdict(list)
    subj_per_video: dict[str, dict[str, float]] = defaultdict(dict)

    static_set = set(SPLIT_DIMS["static"])
    dynamic_set = set(SPLIT_DIMS["dynamic"])

    for (pid, dim, sub), (val, split) in by_key.items():
        if dim in static_set and split != "static": continue
        if dim in dynamic_set and split != "dynamic": continue
        if dim == "subjective_quality":
            subj_per_video[pid][sub] = val
        else:
            n = normalize_instance(dim, val)
            if n is None: continue
            norm_by_dim[dim].append(n)
            raw_by_dim[dim].append(val if not isinstance(val, list) else val[0])

    # subjective_quality: mean of both component norms per video, then flat mean
    for pid, sub in subj_per_video.items():
        components = []
        if "clip_aesthetic" in sub:
            n = normalize_instance("subjective_quality", sub["clip_aesthetic"], "clip_aesthetic")
            if n is not None: components.append(n)
        if "clip_iqa+" in sub:
            n = normalize_instance("subjective_quality", sub["clip_iqa+"], "clip_iqa+")
            if n is not None: components.append(n)
        if components:
            norm_by_dim["subjective_quality"].append(sum(components) / len(components))

    per_dim: dict[str, dict] = {}
    for dim in SPLIT_DIMS["static"] + SPLIT_DIMS["dynamic"]:
        norms = norm_by_dim.get(dim, [])
        raws = raw_by_dim.get(dim, [])
        per_dim[dim] = {
            "n": len(norms),
            "raw_mean": round(sum(raws) / len(raws), 4) if raws else None,
            "norm_mean_x100": round(sum(norms) / len(norms) * 100, 2) if norms else None,
        }

    static_vals = [per_dim[d]["norm_mean_x100"] for d in SPLIT_DIMS["static"]
                   if per_dim[d]["norm_mean_x100"] is not None]
    all10_vals  = [per_dim[d]["norm_mean_x100"]
                   for d in SPLIT_DIMS["static"] + SPLIT_DIMS["dynamic"]
                   if per_dim[d]["norm_mean_x100"] is not None]
    headlines = {
        "WorldScore-Static":  round(sum(static_vals) / len(static_vals), 2) if static_vals else None,
        "WorldScore-Dynamic": round(sum(all10_vals)  / len(all10_vals), 2)  if all10_vals  else None,
    }

    return {
        "per_dim": per_dim,
        "headlines": headlines,
        "method": "upstream run_evaluate.py: per-dim flat mean × 100; "
                  "Static = mean of 7 static dims; Dynamic = mean of all 10 dims",
    }
