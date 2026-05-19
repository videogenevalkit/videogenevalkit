"""WorldJen end-to-end smoke driver — runs the toolkit against your local vLLM.

Usage::

    # 1. Make sure vLLM is up on the ports judges.py expects:
    #    Qwen3-32B  at http://localhost:8004/v1
    #    Gemma-4-31B-IT at http://localhost:8003/v1
    # 2. Lay out videos under workspace/videos/{model_name}/{prompt_id}.mp4
    # 3. Drop your prompts jsonl at workspace/prompts/worldjen/prompts.jsonl
    #    with lines like {"prompt_id": "000", "prompt": "..."}.
    # 4. Run:
    python scripts/run_smoke_worldjen.py \
        --workspace /path/to/ws \
        --models pangu3,wan14b \
        --judge gemma-4-31b-local \
        --judge-llm qwen3-32b-local \
        --limit 5

The script defers entirely to the runtime registries (SUPPORTED_JUDGES etc.) —
edit those to change endpoints/models.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from videvalkit.benchmarks.worldjen import WorldJenBenchmark
from videvalkit.configs import SUPPORTED_JUDGES
from videvalkit.storage import Workspace


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True, type=Path)
    ap.add_argument("--prompts-file", type=Path, default=None,
                    help="Defaults to workspace/prompts/worldjen/prompts.jsonl")
    ap.add_argument("--models", required=True, type=str,
                    help="Comma-separated model names whose videos are under "
                         "workspace/videos/{model}/")
    ap.add_argument("--judge", default="gemma-4-31b-local",
                    choices=list(SUPPORTED_JUDGES))
    ap.add_argument("--judge-llm", default="qwen3-32b-local",
                    choices=list(SUPPORTED_JUDGES))
    ap.add_argument("--dimensions", default=None,
                    help="Comma list of dim names; default = all 16")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N prompts (smoke / cost cap)")
    args = ap.parse_args()

    # Line-buffer stdout so progress streams in real time (not just on flush).
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    ws = Workspace(args.workspace)
    prompts_file = args.prompts_file or (ws.layout.prompts_dir / "worldjen" / "prompts.jsonl")
    if not prompts_file.exists():
        sys.exit(f"missing prompts file: {prompts_file}")

    if args.limit:
        truncated = ws.layout.prompts_dir / "worldjen" / "prompts_truncated.jsonl"
        truncated.parent.mkdir(parents=True, exist_ok=True)
        with prompts_file.open() as src, truncated.open("w") as dst:
            for i, line in enumerate(src):
                if i >= args.limit:
                    break
                dst.write(line)
        prompts_file = truncated

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    dims = [d.strip() for d in args.dimensions.split(",")] if args.dimensions else None

    bench = WorldJenBenchmark()
    raw = bench.evaluate(
        layout=ws.layout,
        models=models,
        dimensions=dims,
        judge=SUPPORTED_JUDGES[args.judge],
        judge_llm=SUPPORTED_JUDGES[args.judge_llm],
        prompts_file=prompts_file,
        max_concurrency=args.concurrency,
    )

    from collections import defaultdict
    by_model = defaultdict(list)
    for r in raw:
        by_model[r.model].append(r)
    summaries = []
    for model, items in by_model.items():
        s = bench.aggregate(items, aggregator="phas")
        ws.write_summary(s)
        summaries.append(s)
        print(f"[{model}] PHAS={s.overall:.4f}  n_prompts={s.n_prompts}")

    # Cross-benchmark report (just for one benchmark here; reusable interface)
    from videvalkit.aggregators import combine_summaries
    report = combine_summaries(summaries)
    out = ws.layout.leaderboard_path("worldjen", "json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
