"""End-to-end smoke test of the integrated WorldScore adapter, all 10 dims.

Runs every upstream dim on 1 static sample and 1 dynamic sample:

  static  (7 dims, 49 frames each) →
    camera_control, object_control, content_alignment,
    3d_consistency, photometric_consistency,
    style_consistency, subjective_quality

  dynamic (3 dims, 49 frames each) →
    motion_accuracy, motion_magnitude, motion_smoothness

For camera_control we generate the GT cameras on the fly via upstream's
CameraGen using the sub-prompt's `camera_path`. For style_consistency we
look up the per-entry reference image extracted by runners/extract_refs.py.

Run:
  cd /pub/evaluation_group/ning/toolkit
  CUDA_VISIBLE_DEVICES=1 \\
    /pub/evaluation_group/ning/benchmark/envs/videvalkit/bin/python \\
    scripts/worldscore_full_10dim_smoke.py
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path

# Ensure upstream paths + mamba_ssm shim are live before any heavy import.
from videvalkit.benchmarks.worldscore.scorers import setup_upstream_paths
setup_upstream_paths()

import torch  # noqa: E402

from videvalkit.benchmarks.worldscore import (  # noqa: E402
    WorldScoreBenchmark, WORLDSCORE_STATIC_DIMS, WORLDSCORE_DYNAMIC_DIMS,
)
from videvalkit.core.types import PromptItem, VideoSpec  # noqa: E402


GEN_ROOT = Path("/pub/evaluation_group/ning/worldscore_gens/cogvideox-5b")
STATIC_FLAT = Path("/pub/evaluation_group/ning/prompt/worldscore_static_sample50_flat.jsonl")
DYNAMIC    = Path("/pub/evaluation_group/ning/prompt/worldscore_dynamic_sample100.jsonl")
STATIC_ENT = Path("/pub/evaluation_group/ning/prompt/worldscore_static_sample50.jsonl")
REF_DIR    = GEN_ROOT / "refs"

N_FRAMES = 49   # production setting; matches upstream's interpframe_num


def _pick(jsonl: Path, gen_subdir: Path):
    for line in jsonl.open():
        e = json.loads(line.strip())
        pid = e.get("prompt_id") or e.get("entry_id")
        mp4 = gen_subdir / f"{pid}.mp4"
        if mp4.exists():
            return e, mp4
    raise RuntimeError(f"no usable sample found in {jsonl}")


def build_gt_cameras(camera_path: list[str], n_frames: int) -> torch.Tensor:
    """Mirror runners/camera_error.py: build interpolated GT cameras via CameraGen.

    Returns a tensor matching the frames we'll pass to CameraErrorMetric.
    """
    from worldscore.benchmark.helpers.camera_generator import CameraGen
    import numpy as np
    cg_root = "/tmp/wssmoke_cg"
    os.makedirs(cg_root, exist_ok=True)
    cg = CameraGen({
        "benchmark_root": cg_root,
        "focal_length": 500,
        "camera_speed": 1,
        "frames": n_frames,
        "model": "cogvideox_5b_t2v",
    })
    _, cameras_interp = cg.generate_cameras(camera_path, cg_root, verbose=False)
    cg.clear()
    idx = np.linspace(0, len(cameras_interp) - 1, n_frames).astype(int).tolist()
    cams = torch.tensor(np.array([cameras_interp[i] for i in idx]))
    return cams


def main() -> None:
    print("=" * 80)
    print("WorldScore adapter full-10-dim smoke  (1 static + 1 dynamic, 49 frames)")
    print("=" * 80)

    bench = WorldScoreBenchmark()
    print(f"adapter dims: {len(bench.dimensions)}/10")
    print(f"  static  ({len(WORLDSCORE_STATIC_DIMS)}): {list(WORLDSCORE_STATIC_DIMS)}")
    print(f"  dynamic ({len(WORLDSCORE_DYNAMIC_DIMS)}): {list(WORLDSCORE_DYNAMIC_DIMS)}")
    print()

    # ---- pick samples ---------------------------------------------------- #
    s_entry, s_mp4 = _pick(STATIC_FLAT, GEN_ROOT / "static")
    d_entry, d_mp4 = _pick(DYNAMIC,    GEN_ROOT / "dynamic")
    print(f"static sample:  {s_entry['prompt_id']}  ({s_mp4.name})")
    print(f"                {s_entry['prompt'][:80]}...")
    print(f"                camera_path={s_entry['camera_path']}  parent={s_entry['parent_entry']}")
    print(f"dynamic sample: {d_entry['prompt_id']}  ({d_mp4.name})")
    print(f"                {d_entry['prompt'][:80]}...")
    print(f"                objects={d_entry['objects']}  motion_type={d_entry.get('motion_type')}")
    print()

    # ---- camera_control needs GT cameras → generate via upstream CameraGen
    print("building GT cameras for camera_control via upstream CameraGen ...")
    t0 = time.time()
    cameras_gt = build_gt_cameras(list(s_entry["camera_path"]), N_FRAMES)
    print(f"  {tuple(cameras_gt.shape)} cameras_gt in {time.time()-t0:.1f}s\n")

    # ---- object_control needs upstream's "scene_name, obj1, obj2" prompt
    # The static entry's content_list[step] follows that exact format.
    # The static-flat JSONL doesn't carry content_list; look it up from the
    # entry-level JSONL.
    entry_content_list = None
    for ln in STATIC_ENT.open():
        ee = json.loads(ln)
        if ee["entry_id"] == s_entry["parent_entry"]:
            entry_content_list = ee.get("content_list", [])
            break
    content_for_objdet = (entry_content_list or [None])[s_entry.get("step", 0)]
    print(f"object_control content string: {content_for_objdet!r}\n")

    # ---- build PromptItems + VideoSpecs --------------------------------- #
    s_prompt = PromptItem(
        prompt_id=s_entry["prompt_id"], text=s_entry["prompt"],
        dimensions=list(WORLDSCORE_STATIC_DIMS),
        meta={
            "split": "static",
            "parent_entry": s_entry["parent_entry"],
            "step": s_entry.get("step", 0),
            "content_list": entry_content_list,    # whole list, _run_dim picks step
            "cameras_gt": cameras_gt,
        },
    )
    d_prompt = PromptItem(
        prompt_id=d_entry["prompt_id"], text=d_entry["prompt"],
        dimensions=list(WORLDSCORE_DYNAMIC_DIMS),
        meta={
            "split": "dynamic",
            "objects": d_entry["objects"],
        },
    )
    s_video = VideoSpec(path=s_mp4, prompt_id=s_prompt.prompt_id, model_name="cogvideox-5b")
    d_video = VideoSpec(path=d_mp4, prompt_id=d_prompt.prompt_id, model_name="cogvideox-5b")

    # ---- evaluate static (7 dims) --------------------------------------- #
    print(f"evaluating static (7 dims) on {s_mp4.name} ...")
    t0 = time.time()
    static_raw = bench.evaluate(
        videos=[s_video], prompts=[s_prompt],
        dimensions=list(WORLDSCORE_STATIC_DIMS),
        n_frames=N_FRAMES,
        reference_image_dir=str(REF_DIR),
    )
    print(f"  static eval done in {time.time()-t0:.1f}s, {len(static_raw)} RawResults")

    # ---- evaluate dynamic (3 dims) -------------------------------------- #
    print(f"\nevaluating dynamic (3 dims) on {d_mp4.name} ...")
    t0 = time.time()
    dyn_raw = bench.evaluate(
        videos=[d_video], prompts=[d_prompt],
        dimensions=list(WORLDSCORE_DYNAMIC_DIMS),
        n_frames=N_FRAMES,
    )
    print(f"  dynamic eval done in {time.time()-t0:.1f}s, {len(dyn_raw)} RawResults")

    # ---- print every per-instance row ----------------------------------- #
    print("\n--- per-instance raw scores ---")
    for r in static_raw + dyn_raw:
        sm = (r.meta or {}).get("metric") or ""
        if isinstance(r.score, (int, float)):
            sval = f"{r.score:.4f}"
        else:
            sval = f"{[round(v, 4) for v in r.score]}"
        tag = (" /" + sm) if sm else ""
        print(f"  {r.prompt_id:<14} {r.dimension + tag:<40} score={sval}  ({r.scorer})")

    # ---- aggregate ------------------------------------------------------ #
    print("\n--- aggregation (mirrors upstream run_evaluate.py) ---")
    summary = bench.aggregate(static_raw + dyn_raw)
    for d in list(WORLDSCORE_STATIC_DIMS) + list(WORLDSCORE_DYNAMIC_DIMS):
        v = summary.per_dimension.get(d)
        vstr = f"{v:.2f}" if v is not None else "—"
        print(f"  {d:<28} norm × 100 = {vstr}")
    print(f"\nheadlines: {json.dumps(summary.meta['headlines'])}")
    print(f"method:    {summary.meta['method']}")
    print()
    print("=" * 80)
    print(f"SUCCESS — all {len(WORLDSCORE_STATIC_DIMS) + len(WORLDSCORE_DYNAMIC_DIMS)} "
          "WorldScore dims completed end-to-end via upstream classes.")


if __name__ == "__main__":
    main()
