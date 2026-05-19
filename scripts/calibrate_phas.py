"""calibrate_phas.py — fit PHAS dimension weights from human pairwise preferences.

Ported from `worldjen_local/human_eval/calibrate_phas.py`. Reads:

  * `--annotations CSV` — pairwise human evals with columns
        [Prompt ID, Model A, Model B, Winner, Loser, Weight, Source, ...]
    where Source ∈ {"calibration", "validation"}.
  * `--workspace` — toolkit workspace; reads raw VLM scores from
        `results/raw/worldjen/{model}/{dim}/{pid}.json`.

Writes:
  * `<workspace>/results/summary/phas_weights.json` — calibrated weights +
    cv_accuracy + per-dim non-negative ridge coefficients.

Usage::

    python scripts/calibrate_phas.py \
        --workspace /path/to/ws \
        --annotations /path/to/anonymized_human_evals.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from videvalkit.aggregators.phas import PHASAggregator
from videvalkit.benchmarks.worldjen.dimensions import WORLDJEN_DIMENSIONS
from videvalkit.core.types import RawResult


def _load_vlm_scores(workspace: Path) -> dict[tuple[str, str], dict[str, float]]:
    """Return {(model, prompt_id): {dim: mean_score}} from the workspace raw JSONs."""
    raw_dir = workspace / "results" / "raw" / "worldjen"
    out: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for jf in raw_dir.glob("*/*/*.json"):
        try:
            r = RawResult.model_validate_json(jf.read_text())
        except Exception:
            continue
        if isinstance(r.score, (int, float)) and not np.isnan(float(r.score)):
            out[(r.model, r.prompt_id)][r.dimension] = float(r.score)
    return out


def _build_matrix(
    annotations: list[dict],
    vlm_scores: dict[tuple[str, str], dict[str, float]],
    dim_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct (X, y, w) per `worldjen_local/human_eval/calibrate_phas.py:build_matrix`."""
    X, y, w = [], [], []
    for r in annotations:
        pid, ma, mb = r["Prompt ID"], r["Model A"], r["Model B"]
        sa = vlm_scores.get((ma, pid), {})
        sb = vlm_scores.get((mb, pid), {})
        if not sa or not sb:
            continue
        X.append([sa.get(d, 0.0) - sb.get(d, 0.0) for d in dim_names])
        y.append(1 if r["Winner"] == ma else 0)
        w.append(float(r.get("Weight", 1.0)))
    return np.array(X, dtype=float), np.array(y, dtype=int), np.array(w, dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True, type=Path)
    ap.add_argument("--annotations", required=True, type=Path,
                    help="Pairwise human eval CSV")
    ap.add_argument("--source-filter", default="calibration",
                    help="Only use rows where Source == this value (default: calibration)")
    ap.add_argument("--exclude-ids", default="A1-retest",
                    help="Comma list of annotator IDs to exclude")
    args = ap.parse_args()

    exclude = {x.strip() for x in args.exclude_ids.split(",") if x.strip()}
    with args.annotations.open() as f:
        rows = list(csv.DictReader(f))
    annotations = [
        r for r in rows
        if r.get("Source") == args.source_filter
        and r.get("Annotator", r.get("User", "")) not in exclude
    ]
    if not annotations:
        raise SystemExit(f"no annotations matched Source={args.source_filter!r}")

    vlm = _load_vlm_scores(args.workspace)
    if not vlm:
        raise SystemExit(f"no VLM raw scores under {args.workspace}/results/raw/worldjen/")

    X, y, w = _build_matrix(annotations, vlm, WORLDJEN_DIMENSIONS)
    if len(y) < 10:
        raise SystemExit(f"too few usable annotations ({len(y)})")

    result = PHASAggregator.calibrate(X, y, w, WORLDJEN_DIMENSIONS)
    out_path = args.workspace / "results" / "summary" / "phas_weights.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "phas_calibration": {
            "n_annotations": int(len(y)),
            "source":        args.source_filter,
            "dimensions":    WORLDJEN_DIMENSIONS,
            "nonneg_ridge":  result,
        }
    }, indent=2, ensure_ascii=False))
    print(f"Wrote {out_path}")
    print(f"  cv_accuracy = {result['cv_accuracy']:.4f}  C = {result['C']}")
    for d, wgt in sorted(result["weights"].items(), key=lambda kv: -kv[1]):
        print(f"    {d:24s}  {wgt:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
