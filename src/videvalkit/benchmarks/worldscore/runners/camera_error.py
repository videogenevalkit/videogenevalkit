"""Score camera_control (CameraErrorMetric via DROID-SLAM) on 50 static entries.

For each entry:
  1. Concat its sub-prompt mp4s into a single frame sequence
  2. Use CameraGen to build GT cameras from the entry's camera_path
  3. Sample 32 frames + matching GT cameras
  4. Run CameraErrorMetric → (R_error_deg, T_error)
"""
import sys, os, time, types, json, shutil, glob
import os as _vk_os
from pathlib import Path as _VkPath
_VK_CACHE = _VkPath(_vk_os.environ.get("VIDEVALKIT_CACHE_HOME", _VkPath.home() / ".cache" / "videvalkit"))
from pathlib import Path
from statistics import mean

for m in ["mamba_ssm","mamba_ssm.ops","mamba_ssm.ops.selective_scan_interface"]:
    if m not in sys.modules:
        sm = types.ModuleType(m); sm.selective_scan_fn = lambda *a,**k:None; sm.selective_scan_ref = lambda *a,**k:None
        sys.modules[m] = sm

WS_ROOT = _vk_os.environ.get("VIDEVALKIT_WORLDSCORE_ROOT", str(_VkPath.home() / ".cache" / "videvalkit" / "upstream" / "WorldScore"))
PATCHED_DROID = f"{WS_ROOT}/worldscore/benchmark/metrics/third_party/droid_slam"
sys.path.insert(0, WS_ROOT)
sys.path.insert(0, PATCHED_DROID)
os.chdir(WS_ROOT)

import numpy as np, torch
import imageio.v3 as iio
from PIL import Image
from worldscore.benchmark.helpers.camera_generator import CameraGen
from worldscore.benchmark.metrics.third_party.camera_error_metrics import CameraErrorMetric

GEN_STATIC = _VkPath(_vk_os.environ.get("VIDEVALKIT_WS_GEN_STATIC", str(_VK_CACHE / "smoke-data" / "worldscore" / "videos" / "cogvideox-5b" / "static")))
PROMPTS  = _VkPath(_vk_os.environ.get("VIDEVALKIT_WS_STATIC_PROMPTS", str(_VK_CACHE / "smoke-data" / "worldscore" / "prompts" / "static-entry.jsonl")))
OUT_DIR  = _VkPath(_vk_os.environ.get("VIDEVALKIT_WS_OUT_DIR", "./worldscore_eval_out"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SAMP = 147  # frames sampled for SLAM (covers 3-sub-prompt big-world entries fully)
PER_SCENE_FRAMES = 49


def main():
    print(f"[{time.strftime('%H:%M:%S')}] init")
    entries = [json.loads(l) for l in PROMPTS.open()]
    print(f"[{time.strftime('%H:%M:%S')}] {len(entries)} static entries")

    # init CameraErrorMetric once (heavy load)
    metric = CameraErrorMetric()
    metric._args.disable_vis = True
    print(f"[{time.strftime('%H:%M:%S')}] CameraErrorMetric ready")

    raw = OUT_DIR / "raw_results_camera_v3.jsonl"
    results = []
    n_err = 0
    t0_all = time.time()

    with raw.open("w") as fout:
        for i, entry in enumerate(entries):
            eid = entry["entry_id"]
            cp = list(entry["camera_path"])
            print(f"[{time.strftime('%H:%M:%S')}] [{i+1:2d}/{len(entries)}] {eid}  camera_path={cp}")

            # 1. Find this entry's sub-prompt mp4s
            mp4s = sorted(GEN_STATIC.glob(f"{eid}_s*.mp4"))
            if not mp4s:
                n_err += 1
                print(f"  ! no mp4s for {eid}")
                continue

            # 2. Concat frames from all sub-prompt mp4s
            try:
                all_frames = np.concatenate([iio.imread(str(m), plugin="pyav") for m in mp4s], axis=0)
            except Exception as e:
                n_err += 1
                print(f"  ! frame-read failed: {e}")
                continue

            # 3. Generate GT cameras via CameraGen
            cg_root = f"/tmp/cg_{eid}"
            os.makedirs(cg_root, exist_ok=True)
            try:
                cg = CameraGen({
                    "benchmark_root": cg_root,
                    "focal_length": 500,
                    "camera_speed": 1,
                    "frames": PER_SCENE_FRAMES,
                    "model": "cogvideox_5b_t2v",
                })
                _, cameras_interp = cg.generate_cameras(cp, cg_root, verbose=False)
                cg.clear()
            except Exception as e:
                n_err += 1
                print(f"  ! CameraGen failed: {e}")
                shutil.rmtree(cg_root, ignore_errors=True)
                continue

            # 4. Sample N matching frames + cameras
            T_video = all_frames.shape[0]
            T_cam = len(cameras_interp)
            n = min(N_SAMP, T_video, T_cam)
            vid_idx = np.linspace(0, T_video-1, n).astype(int).tolist()
            cam_idx = np.linspace(0, T_cam-1, n).astype(int).tolist()
            cameras_gt = torch.tensor(np.array([cameras_interp[i] for i in cam_idx]))

            tmp = Path(f"/tmp/wseval_cam_{eid}")
            tmp.mkdir(parents=True, exist_ok=True)
            paths = []
            for k, j in enumerate(vid_idx):
                p = tmp / f"f{k:03d}.png"
                Image.fromarray(all_frames[j]).save(p)
                paths.append(str(p))

            # 5. Score with CameraErrorMetric
            try:
                t0 = time.time()
                err = metric._compute_scores(rendered_images=paths, cameras_gt=cameras_gt)
                # err is (R_score_deg, T_score) tuple
                if hasattr(err, '__len__') and len(err) == 2:
                    R_err, T_err = float(err[0]), float(err[1])
                else:
                    R_err = float(err)
                    T_err = None
                dt = time.time() - t0
                row = {
                    "entry_id": eid, "split": "static", "dimension": "camera_control",
                    "camera_path": cp, "is_big_world": entry.get("is_big_world", False),
                    "score": [R_err, T_err], "elapsed_seconds": round(dt, 1),
                }
                fout.write(json.dumps(row) + "\n"); fout.flush()
                results.append(row)
                print(f"  R_err={R_err:.4f}° T_err={T_err if T_err is None else f'{T_err:.4f}'}  ({dt:.1f}s)")
            except Exception as e:
                n_err += 1
                print(f"  ! score failed: {type(e).__name__}: {str(e)[:120]}")
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
                shutil.rmtree(cg_root, ignore_errors=True)
                metric.droid = None

    elapsed = time.time() - t0_all
    print(f"\n[{time.strftime('%H:%M:%S')}] DONE in {elapsed/60:.1f} min  errors={n_err}")
    if results:
        Rs = [r['score'][0] for r in results if r['score'][0] is not None]
        Ts = [r['score'][1] for r in results if r['score'][1] is not None]
        summary = {
            "model": "cogvideox-5b-t2v",
            "dimension": "camera_control",
            "n_entries": len(entries),
            "n_scored": len(results),
            "n_errors": n_err,
            "elapsed_seconds": round(elapsed, 1),
            "mean_R_error_deg": round(mean(Rs), 4) if Rs else None,
            "mean_T_error": round(mean(Ts), 4) if Ts else None,
        }
        (OUT_DIR / "summary_camera.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
