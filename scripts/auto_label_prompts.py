"""Auto-label any prompts.jsonl with VBench-compatible auxiliary_info.

Uses a local LLM (Qwen3-32B via vLLM by default) to read each prompt and
emit structured ground-truth labels for VBench v1 and/or VBench-2.0's
prompt-dependent dimensions. Output is drop-in compatible with upstream's
`VBench_full_info.json` / `VBench2_full_info.json`.

Run::

    python scripts/auto_label_prompts.py \\
        --prompts /pub/evaluation_group/ning/video_eval/worldjen_local/data/prompts/prompts_110.jsonl \\
        --out-dir /pub/evaluation_group/ning/toolkit_test_ws/prompts/auto_labels \\
        --benchmarks vbench,vbench2 \\
        --judge qwen3-32b-local
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from videvalkit.configs import SUPPORTED_JUDGES
from videvalkit.utils.prompt_labeler import (
    VBENCH2_DIM_SCHEMAS,
    VBENCH_V1_DIM_SCHEMAS,
    label_prompts_vbench2,
    label_prompts_vbench_v1,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, type=Path,
                    help="Input prompts.jsonl (toolkit format)")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Output dir for auto-labeled *_full_info.json files")
    ap.add_argument("--benchmarks", default="vbench,vbench2",
                    help="Comma-separated: vbench (v1), vbench2 (v2.0)")
    ap.add_argument("--judge", default="qwen3-32b-local",
                    choices=list(SUPPORTED_JUDGES),
                    help="LLM judge to use (default: local Qwen3-32B at :8004)")
    ap.add_argument("--vbench-dims", default=None,
                    help="Comma list of VBench v1 dims (default: all prompt-dependent).")
    ap.add_argument("--vbench2-dims", default=None,
                    help="Comma list of VBench-2.0 dims (default: all prompt-dependent).")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    judge_cfg = SUPPORTED_JUDGES[args.judge]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    benches = {b.strip() for b in args.benchmarks.split(",") if b.strip()}

    if "vbench" in benches:
        dims = args.vbench_dims.split(",") if args.vbench_dims else None
        out = args.out_dir / "vbench_full_info.json"
        print(f"\n[VBench v1] labeling -> {out}")
        label_prompts_vbench_v1(args.prompts, out, judge_cfg, dims=dims)

    if "vbench2" in benches:
        dims = args.vbench2_dims.split(",") if args.vbench2_dims else None
        out = args.out_dir / "vbench2_full_info.json"
        print(f"\n[VBench-2.0] labeling -> {out}")
        label_prompts_vbench2(args.prompts, out, judge_cfg, dims=dims)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
