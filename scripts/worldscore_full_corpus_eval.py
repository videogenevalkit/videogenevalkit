"""Full-corpus run of the integrated WorldScoreBenchmark adapter.

Processes every video in our 100 + 103 sample via the toolkit's adapter API:
  - 100 dynamic videos -> scored on the 3 dynamic dims (motion_accuracy,
    motion_magnitude, motion_smoothness)
  - 103 static sub-prompt videos -> scored on the 7 static dims
    (camera_control via the parent-entry's GT cameras, object_control,
    content_alignment, 3d_consistency, photometric_consistency,
    style_consistency, subjective_quality)

The adapter (benchmark.py) infers split from prompt_id prefix and emits only
the dims that belong to that split, so split-routing is checked by the
adapter at evaluate-time.

Output:
  RawResults are written incrementally to
  /pub/evaluation_group/ning/worldscore_gens/cogvideox-5b/eval_10dim/
    toolkit_full_raw.jsonl
  Per-dim counts and the final headlines are printed at end.

Run:
  cd /pub/evaluation_group/ning/toolkit
  CUDA_VISIBLE_DEVICES=1 \\
    /pub/evaluation_group/ning/benchmark/envs/videvalkit/bin/python \\
    scripts/worldscore_full_corpus_eval.py
"""
from __future__ import annotations
import json
import os
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from videvalkit.benchmarks.worldscore.scorers import setup_upstream_paths
setup_upstream_paths()

import torch  # noqa: E402
import numpy as np  # noqa: E402

from videvalkit.benchmarks.worldscore import (  # noqa: E402
    WorldScoreBenchmark, WORLDSCORE_STATIC_DIMS, WORLDSCORE_DYNAMIC_DIMS,
)
from videvalkit.core.types import PromptItem, VideoSpec  # noqa: E402


GEN_ROOT     = Path("/pub/evaluation_group/ning/worldscore_gens/cogvideox-5b")
EVAL_OUT     = GEN_ROOT / "eval_10dim"
EVAL_OUT.mkdir(parents=True, exist_ok=True)

STATIC_ENT   = Path("/pub/evaluation_group/ning/prompt/worldscore_static_sample50.jsonl")
STATIC_FLAT  = Path("/pub/evaluation_group/ning/prompt/worldscore_static_sample50_flat.jsonl")
DYNAMIC      = Path("/pub/evaluation_group/ning/prompt/worldscore_dynamic_sample100.jsonl")
REF_DIR      = GEN_ROOT / "refs"
RAW_OUT      = EVAL_OUT / "toolkit_full_raw.jsonl"
SUMMARY_OUT  = EVAL_OUT / "toolkit_full_summary.json"

N_FRAMES     = 49


def build_gt_cameras(camera_path, n_frames=N_FRAMES):
    from worldscore.benchmark.helpers.camera_generator import CameraGen
    cg_root = "/tmp/wscam_cg"
    os.makedirs(cg_root, exist_ok=True)
    cg = CameraGen({
        "benchmark_root": cg_root, "focal_length": 500, "camera_speed": 1,
        "frames": n_frames, "model": "cogvideox_5b_t2v",
    })
    _, cameras_interp = cg.generate_cameras(list(camera_path), cg_root, verbose=False)
    cg.clear()
    idx = np.linspace(0, len(cameras_interp) - 1, n_frames).astype(int).tolist()
    return torch.tensor(np.array([cameras_interp[i] for i in idx]))


def load_static_prompts():
    """Yield one PromptItem per sub-prompt; cameras_gt is pre-built per parent
    entry and reused across that entry's sub-prompts."""
    # Build parent_entry -> (camera_path, content_list, cameras_gt) lookup.
    entries: dict[str, dict] = {}
    for ln in STATIC_ENT.open():
        e = json.loads(ln)
        entries[e["entry_id"]] = e

    cams_cache: dict[str, torch.Tensor] = {}

    for ln in STATIC_FLAT.open():
        sp = json.loads(ln)
        pid = sp["prompt_id"]
        parent = sp["parent_entry"]
        ent = entries.get(parent, {})
        camera_path = ent.get("camera_path") or sp.get("camera_path")
        if parent not in cams_cache and camera_path:
            try:
                cams_cache[parent] = build_gt_cameras(camera_path, N_FRAMES)
            except Exception as e:
                print(f"  ! cameras_gt build failed for {parent}: {e}")
                cams_cache[parent] = None
        cams = cams_cache.get(parent)
        yield PromptItem(
            prompt_id=pid, text=sp["prompt"],
            dimensions=list(WORLDSCORE_STATIC_DIMS),
            meta={
                "split": "static",
                "parent_entry": parent,
                "step": sp.get("step", 0),
                "content_list": ent.get("content_list"),
                "camera_path": camera_path,
                "cameras_gt": cams,
            },
        )


def load_dynamic_prompts():
    for ln in DYNAMIC.open():
        d = json.loads(ln)
        yield PromptItem(
            prompt_id=d["prompt_id"], text=d.get("prompt", ""),
            dimensions=list(WORLDSCORE_DYNAMIC_DIMS),
            meta={"split": "dynamic", "objects": d.get("objects") or []},
        )


def main():
    bench = WorldScoreBenchmark()
    print(f"adapter: {bench.name}  ({len(bench.dimensions)} dims)")
    print(f"  static  : {list(WORLDSCORE_STATIC_DIMS)}")
    print(f"  dynamic : {list(WORLDSCORE_DYNAMIC_DIMS)}")
    print()

    fout = RAW_OUT.open("w")
    n_static_done = 0; n_dynamic_done = 0
    dim_counter: Counter = Counter()
    routing_violations: list[str] = []
    all_raw = []
    t_start = time.time()

    # ---- STATIC ---------------------------------------------------------- #
    static_prompts = list(load_static_prompts())
    print(f"static: {len(static_prompts)} sub-prompts to score")
    for i, p in enumerate(static_prompts, 1):
        mp4 = GEN_ROOT / "static" / f"{p.prompt_id}.mp4"
        if not mp4.exists():
            print(f"  [{i:3d}/{len(static_prompts)}] {p.prompt_id}  ! mp4 missing"); continue
        vs = VideoSpec(path=mp4, prompt_id=p.prompt_id, model_name="cogvideox-5b")
        t0 = time.time()
        try:
            raw = bench.evaluate(
                videos=[vs], prompts=[p],
                dimensions=list(WORLDSCORE_STATIC_DIMS),
                n_frames=N_FRAMES,
                reference_image_dir=str(REF_DIR),
            )
        except Exception as e:
            print(f"  [{i:3d}/{len(static_prompts)}] {p.prompt_id}  ! eval failed: {e}")
            continue
        # Verify routing: every emitted row must be a static dim.
        for r in raw:
            if r.dimension not in WORLDSCORE_STATIC_DIMS:
                routing_violations.append(f"static video {p.prompt_id} emitted {r.dimension}")
            dim_counter[r.dimension] += 1
            fout.write(json.dumps({
                "prompt_id": r.prompt_id, "dimension": r.dimension,
                "metric": (r.meta or {}).get("metric"),
                "split": (r.meta or {}).get("split"),
                "score": r.score, "scorer": r.scorer,
            }) + "\n")
            fout.flush()
            all_raw.append(r)
        n_static_done += 1
        eta = (time.time() - t_start) / max(1, i) * (len(static_prompts) - i)
        if i % 5 == 0 or i == 1:
            print(f"  [{i:3d}/{len(static_prompts)}] {p.prompt_id}  "
                  f"{len(raw)} rows  ({time.time()-t0:.1f}s, eta {eta/60:.1f}min)")

    print(f"\nstatic done: {n_static_done} videos, "
          f"{sum(dim_counter[d] for d in WORLDSCORE_STATIC_DIMS)} total static rows")

    # ---- DYNAMIC --------------------------------------------------------- #
    dyn_prompts = list(load_dynamic_prompts())
    print(f"\ndynamic: {len(dyn_prompts)} prompts to score")
    t_dyn_start = time.time()
    for i, p in enumerate(dyn_prompts, 1):
        mp4 = GEN_ROOT / "dynamic" / f"{p.prompt_id}.mp4"
        if not mp4.exists():
            print(f"  [{i:3d}/{len(dyn_prompts)}] {p.prompt_id}  ! mp4 missing"); continue
        vs = VideoSpec(path=mp4, prompt_id=p.prompt_id, model_name="cogvideox-5b")
        t0 = time.time()
        try:
            raw = bench.evaluate(
                videos=[vs], prompts=[p],
                dimensions=list(WORLDSCORE_DYNAMIC_DIMS),
                n_frames=N_FRAMES,
            )
        except Exception as e:
            print(f"  [{i:3d}/{len(dyn_prompts)}] {p.prompt_id}  ! eval failed: {e}")
            continue
        for r in raw:
            if r.dimension not in WORLDSCORE_DYNAMIC_DIMS:
                routing_violations.append(f"dynamic video {p.prompt_id} emitted {r.dimension}")
            dim_counter[r.dimension] += 1
            fout.write(json.dumps({
                "prompt_id": r.prompt_id, "dimension": r.dimension,
                "metric": (r.meta or {}).get("metric"),
                "split": (r.meta or {}).get("split"),
                "score": r.score, "scorer": r.scorer,
            }) + "\n")
            fout.flush()
            all_raw.append(r)
        n_dynamic_done += 1
        eta = (time.time() - t_dyn_start) / max(1, i) * (len(dyn_prompts) - i)
        if i % 5 == 0 or i == 1:
            print(f"  [{i:3d}/{len(dyn_prompts)}] {p.prompt_id}  "
                  f"{len(raw)} rows  ({time.time()-t0:.1f}s, eta {eta/60:.1f}min)")

    fout.close()
    elapsed = time.time() - t_start
    print(f"\ndynamic done: {n_dynamic_done} videos, "
          f"{sum(dim_counter[d] for d in WORLDSCORE_DYNAMIC_DIMS)} total dynamic rows")

    # ---- routing verification ------------------------------------------- #
    print("\n--- split-routing verification ---")
    print(f"  static dim emissions:")
    for d in WORLDSCORE_STATIC_DIMS:
        print(f"    {d:<26} {dim_counter[d]:>4} rows  (expected ≈ {n_static_done} "
              f"or 2× for subjective_quality)")
    print(f"  dynamic dim emissions:")
    for d in WORLDSCORE_DYNAMIC_DIMS:
        print(f"    {d:<26} {dim_counter[d]:>4} rows  (expected ≈ {n_dynamic_done})")
    if routing_violations:
        print(f"\n  ✗ {len(routing_violations)} routing violations:")
        for v in routing_violations[:10]: print(f"      {v}")
    else:
        print("  ✓ no routing violations")

    # ---- aggregate ------------------------------------------------------ #
    print("\n--- aggregation (per-dim mean × 100, then headlines) ---")
    summary = bench.aggregate(all_raw)
    for d in list(WORLDSCORE_STATIC_DIMS) + list(WORLDSCORE_DYNAMIC_DIMS):
        v = summary.per_dimension.get(d)
        print(f"  {d:<26} {f'{v:.2f}' if v is not None else '—':>8}")
    print(f"\nheadlines: {json.dumps(summary.meta['headlines'])}")
    print(f"method:    {summary.meta['method']}")

    # ---- persist summary ----------------------------------------------- #
    SUMMARY_OUT.write_text(json.dumps({
        "model": "cogvideox-5b-t2v",
        "n_static_videos": n_static_done,
        "n_dynamic_videos": n_dynamic_done,
        "elapsed_seconds": round(elapsed, 1),
        "dim_row_counts": dict(dim_counter),
        "routing_violations": routing_violations,
        "per_dimension": summary.per_dimension,
        "headlines": summary.meta["headlines"],
        "method": summary.meta["method"],
        "n_frames": N_FRAMES,
    }, indent=2))
    print(f"\nwrote {SUMMARY_OUT}")
    print(f"wrote {RAW_OUT}")
    print(f"\n=== DONE in {elapsed/60:.1f} min ===")


if __name__ == "__main__":
    main()
