#!/usr/bin/env python
"""NPU adaptation smoke test — run on an Ascend 910B box.

Activates the torch_npu runtime, probes the ops most likely to be missing
(conv3d, grid_sample, scaled-dot-product attention), then runs each
"easy-tier" metric on tiny synthetic input with --device npu and reports
PASS / FAIL per metric (with the failing op when it crashes).

    python scripts/npu_smoke.py                 # all easy-tier metrics
    python scripts/npu_smoke.py --only fvd,vfid # subset
    python scripts/npu_smoke.py --device cpu    # sanity-run the harness anywhere

This is a diagnostic, not a pytest — it expects real hardware + weights and
prints a human-readable table.
"""

from __future__ import annotations

import argparse
import tempfile
import traceback
from pathlib import Path

# canonical (judge-free, we control device) + vbench lifts (via upstream + shim)
CANONICAL_DIST = ["fvd", "vfid", "kvd", "clip-fvd"]          # gen + ref videos
CANONICAL_PROMPT = ["clip-score", "viclip-score"]            # videos + prompts
VBENCH_EASY = [
    "temporal-flickering", "subject-consistency",
    "background-consistency", "aesthetic-quality", "imaging-quality",
]
VBENCH_VERIFY = ["dynamic-degree", "motion-smoothness"]      # optical-flow / interp
ALL = CANONICAL_DIST + CANONICAL_PROMPT + VBENCH_EASY + VBENCH_VERIFY

results: list[tuple[str, str, str]] = []  # (name, status, detail)


def _make_video(path: Path, n_frames: int = 16, hw: int = 64, seed: int = 0) -> None:
    import imageio
    import numpy as np
    rng = np.random.default_rng(seed)
    w = imageio.get_writer(str(path), fps=4, codec="libx264")
    try:
        for _ in range(n_frames):
            w.append_data(rng.integers(0, 256, (hw, hw, 3), dtype=np.uint8))
    finally:
        w.close()


def probe_ops(device: str) -> None:
    import torch
    print(f"\n--- op probes on {device} ---")
    for name, fn in [
        ("matmul", lambda: torch.randn(64, 64, device=device) @ torch.randn(64, 64, device=device)),
        ("conv3d", lambda: torch.nn.Conv3d(3, 8, 3).to(device)(
            torch.randn(1, 3, 8, 16, 16, device=device))),
        ("grid_sample", lambda: torch.nn.functional.grid_sample(
            torch.randn(1, 3, 16, 16, device=device),
            torch.rand(1, 16, 16, 2, device=device) * 2 - 1, align_corners=False)),
        ("sdpa", lambda: torch.nn.functional.scaled_dot_product_attention(
            *[torch.randn(1, 4, 16, 16, device=device) for _ in range(3)])),
    ]:
        try:
            fn()
            print(f"  {name:14} OK")
        except Exception as e:
            print(f"  {name:14} FAIL  {type(e).__name__}: {str(e)[:80]}")


def run_metric(name: str, device: str, gen: Path, ref: Path, prompts_for: dict) -> None:
    from videvalkit.metrics import get_metric, metric_info
    info = metric_info(name) or {}
    kind = info.get("kind", "")
    try:
        m = get_metric(name)
        if kind == "distribution_reference":
            r = m.compute(gen_videos=sorted(gen.glob("*.mp4")),
                          ref_videos=sorted(ref.glob("*.mp4")),
                          device=device, allow_tiny_sample=True)
        elif kind == "per_prompt_reference_free":
            vids = sorted(gen.glob("*.mp4"))
            r = m.compute(videos=vids, prompts=["a colorful test clip"] * len(vids),
                          device=device)
        else:  # per_video_reference_free (vbench lifts)
            r = m.compute(videos=sorted(gen.glob("*.mp4")), device=device)
        score = getattr(r, "score", "?")
        results.append((name, "PASS", f"score={score}"))
        print(f"  {name:24} PASS  score={score}")
    except Exception:
        tb = traceback.format_exc().strip().splitlines()[-1]
        results.append((name, "FAIL", tb))
        print(f"  {name:24} FAIL  {tb[:90]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="npu")
    ap.add_argument("--only", default="", help="comma-separated metric subset")
    args = ap.parse_args()

    from videvalkit.core.device import ensure_npu_runtime, resolve_device
    if args.device == "npu":
        active = ensure_npu_runtime()
        print(f"torch_npu runtime active: {active}")
    dev = resolve_device(args.device)
    print(f"resolved device: {dev}")

    probe_ops(dev)

    wanted = [m.strip() for m in args.only.split(",") if m.strip()] or ALL
    with tempfile.TemporaryDirectory() as td:
        gen, ref = Path(td) / "gen", Path(td) / "ref"
        gen.mkdir()
        ref.mkdir()
        for i in range(4):
            _make_video(gen / f"g{i}.mp4", seed=i)
            _make_video(ref / f"r{i}.mp4", seed=100 + i)
        print(f"\n--- metrics on {dev} ---")
        for name in wanted:
            run_metric(name, dev, gen, ref, {})

    n_pass = sum(1 for _, s, _ in results if s == "PASS")
    print(f"\n==== {n_pass}/{len(results)} metrics PASSED on {dev} ====")
    for name, status, detail in results:
        if status == "FAIL":
            print(f"  FAIL {name}: {detail}")


if __name__ == "__main__":
    main()
