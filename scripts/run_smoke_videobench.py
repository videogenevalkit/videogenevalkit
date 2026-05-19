"""Video-Bench smoke driver — no upstream package install required.

Uses our OpenAI-compatible VLM judge to score each (model, prompt, dim) on the
shared `{workspace}/videos/{model}/{prompt_id}.mp4` layout. Falls back to a
generic prompt template when no per-dim system_prompt file is found.

Usage::

    python scripts/run_smoke_videobench.py \
        --workspace /path/to/ws \
        --models pangu_model3_141 \
        --judge gemma-4-31b-local \
        --dimensions aesthetic_quality,imaging_quality \
        --limit 1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from videvalkit.benchmarks.videobench import VideoBenchBenchmark
from videvalkit.configs import SUPPORTED_JUDGES
from videvalkit.storage import Workspace


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True, type=Path)
    ap.add_argument("--prompts-file", type=Path, default=None)
    ap.add_argument("--models", required=True, type=str)
    ap.add_argument("--judge", default="gemma-4-31b-local", choices=list(SUPPORTED_JUDGES))
    ap.add_argument("--dimensions", default=None,
                    help="Comma list of dim names; default = all 9")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

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
    prompts_file = args.prompts_file or (ws.layout.prompts_dir / "videobench" / "prompts.jsonl")
    if not prompts_file.exists():
        sys.exit(f"missing prompts file: {prompts_file}")

    if args.limit:
        truncated = ws.layout.prompts_dir / "videobench" / "prompts_truncated.jsonl"
        truncated.parent.mkdir(parents=True, exist_ok=True)
        with prompts_file.open() as src, truncated.open("w") as dst:
            for i, line in enumerate(src):
                if i >= args.limit:
                    break
                dst.write(line)
        prompts_file = truncated

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    dims = [d.strip() for d in args.dimensions.split(",")] if args.dimensions else None

    bench = VideoBenchBenchmark()
    raw = bench.evaluate(
        layout=ws.layout, models=models, dimensions=dims,
        judge=SUPPORTED_JUDGES[args.judge],
        prompts_file=prompts_file,
        max_concurrency=args.concurrency,
        shared_video=True,
    )
    from collections import defaultdict
    by_model = defaultdict(list)
    for r in raw:
        by_model[r.model].append(r)
    for model, items in by_model.items():
        s = bench.aggregate(items)
        ws.write_summary(s)
        print(f"[{model}] overall={s.overall:.4f}  n_videos={s.n_videos}  n_prompts={s.n_prompts}")
        for d, v in sorted(s.per_dimension.items()):
            print(f"    {d:30s}  {v:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
