"""Cross-benchmark report — gather summaries from multiple benchmark runs in one workspace.

Usage::

    python scripts/cross_report.py --workspace /path/to/ws --output cross.json

Equivalent to:
    videvalkit aggregate --workspace /path/to/ws --output cross.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from videvalkit.aggregators import combine_summaries
from videvalkit.core.types import Summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True, type=Path)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    root = args.workspace / "results" / "summary"
    if not root.exists():
        raise SystemExit(f"no summaries under {root}")
    summaries: list[Summary] = []
    for jf in root.glob("*/*.json"):
        summaries.append(Summary.model_validate_json(jf.read_text()))
    rep = combine_summaries(summaries)
    out = args.output or (args.workspace / "results" / "leaderboard" / "cross_benchmark.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str))
    print(f"Wrote {out}")
    for r in rep["ranking"]:
        print(f"  #{r['rank']:>2}  {r['model']:30s}  z={r['score']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
