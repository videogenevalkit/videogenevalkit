"""Score motion_smoothness (VFIMamba + SSIM + LPIPS + MSE) on 100 dynamic videos.

Per upstream: even-indexed frames are conditioning pairs, odd-indexed are the
ground-truth middle frame; for each (I0, I2) pair, predict the mid and compare
to the actual mid using SSIM, LPIPS, MSE. Returns (mse, ssim, lpips) triple.
"""
import sys, os, time, types, json, shutil
import os as _vk_os
from pathlib import Path as _VkPath
_VK_CACHE = _VkPath(_vk_os.environ.get("VIDEVALKIT_CACHE_HOME", _VkPath.home() / ".cache" / "videvalkit"))
from pathlib import Path
from statistics import mean

from videvalkit.benchmarks.worldscore.scorers import install_mamba_ssm_shim
install_mamba_ssm_shim()

WS_ROOT = _vk_os.environ.get("VIDEVALKIT_WORLDSCORE_ROOT", str(_VkPath.home() / ".cache" / "videvalkit" / "upstream" / "WorldScore"))
sys.path.insert(0, WS_ROOT)
sys.path.insert(0, f"{WS_ROOT}/worldscore/benchmark/metrics/third_party/VFIMamba")
os.chdir(WS_ROOT)

import numpy as np, torch
import imageio.v3 as iio
from PIL import Image

GEN_DYN  = _VkPath(_vk_os.environ.get("VIDEVALKIT_WS_GEN_DYNAMIC", str(_VK_CACHE / "smoke-data" / "worldscore" / "videos" / "cogvideox-5b" / "dynamic")))
OUT_DIR  = _VkPath(_vk_os.environ.get("VIDEVALKIT_WS_OUT_DIR", "./worldscore_eval_out"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Subsample frames per video to keep runtime tractable.
# 49 source frames -> use 17 (even=9, odd=8 -> 8 triplets per video).
N_FRAMES_PER_VIDEO = 49  # Full upstream sampling


def dump_frames(mp4_path: Path, tmp_dir: Path, n: int = N_FRAMES_PER_VIDEO):
    arr = iio.imread(str(mp4_path), plugin="pyav")
    if arr.shape[0] == 0: return []
    idx = np.linspace(0, arr.shape[0]-1, min(n, arr.shape[0])).astype(int).tolist()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, j in enumerate(idx):
        p = tmp_dir / f"f{i:03d}.png"
        Image.fromarray(arr[j]).save(p)
        paths.append(str(p))
    return paths


def main():
    print(f"[{time.strftime('%H:%M:%S')}] init")
    from worldscore.benchmark.metrics.third_party.motion_smoothness_metrics import MotionSmoothnessMetric
    metric = MotionSmoothnessMetric()
    print(f"[{time.strftime('%H:%M:%S')}] metric ready")

    videos = sorted(GEN_DYN.glob("*.mp4"))
    print(f"[{time.strftime('%H:%M:%S')}] {len(videos)} dynamic videos")

    raw = OUT_DIR / "raw_results_motion_smoothness_v2.jsonl"
    rows = []
    n_err = 0
    t0_all = time.time()
    with raw.open("w") as fout:
        for i, mp4 in enumerate(videos):
            pid = mp4.stem
            tmp = Path(f"/tmp/wseval_ms_{pid}")
            try:
                paths = dump_frames(mp4, tmp)
                if len(paths) < 4:
                    raise ValueError(f"only {len(paths)} frames")
                t0 = time.time()
                mse, ssim, lpips = metric._compute_scores(rendered_images=paths)
                dt = time.time() - t0
                row = {
                    "prompt_id": pid, "split": "dynamic",
                    "dimension": "motion_smoothness",
                    "score": [float(mse), float(ssim), float(lpips)],
                    "elapsed_seconds": round(dt, 1),
                    "video_path": str(mp4),
                }
                fout.write(json.dumps(row) + "\n"); fout.flush()
                rows.append(row)
                if i % 5 == 0 or i == len(videos) - 1:
                    eta = (time.time() - t0_all) / max(1, i+1) * (len(videos) - i - 1)
                    print(f"[{time.strftime('%H:%M:%S')}] [{i+1:3d}/{len(videos)}] {pid}  mse={mse:.2f} ssim={ssim:.4f} lpips={lpips:.4f}  ({dt:.1f}s, eta={eta/60:.1f}min)")
            except Exception as e:
                n_err += 1
                print(f"  ! {pid}: {type(e).__name__}: {str(e)[:120]}")
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

    elapsed = time.time() - t0_all
    print(f"\n[{time.strftime('%H:%M:%S')}] DONE in {elapsed/60:.1f} min  errors={n_err}")
    if rows:
        mses  = [r['score'][0] for r in rows]
        ssims = [r['score'][1] for r in rows]
        lpips = [r['score'][2] for r in rows]
        summary = {
            "model": "cogvideox-5b-t2v", "dimension": "motion_smoothness",
            "n_videos": len(videos), "n_scored": len(rows), "n_errors": n_err,
            "elapsed_seconds": round(elapsed, 1),
            "mean_mse":   round(mean(mses), 4),
            "mean_ssim":  round(mean(ssims), 4),
            "mean_lpips": round(mean(lpips), 4),
        }
        (OUT_DIR / "summary_motion_smoothness.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
