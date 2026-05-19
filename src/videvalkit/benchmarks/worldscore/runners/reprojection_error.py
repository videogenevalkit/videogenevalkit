"""Score 3d_consistency (ReprojectionErrorMetric via DROID-SLAM) on all 203 videos."""
import sys, os, time, types, json, shutil
import os as _vk_os
from pathlib import Path as _VkPath
_VK_CACHE = _VkPath(_vk_os.environ.get("VIDEVALKIT_CACHE_HOME", _VkPath.home() / ".cache" / "videvalkit"))
from pathlib import Path
from collections import defaultdict
from statistics import mean

# Inject stubs for unavailable deps
for m in ["mamba_ssm","mamba_ssm.ops","mamba_ssm.ops.selective_scan_interface"]:
    if m not in sys.modules:
        sm = types.ModuleType(m); sm.selective_scan_fn = lambda *a,**k:None; sm.selective_scan_ref = lambda *a,**k:None
        sys.modules[m] = sm

WS_ROOT = _vk_os.environ.get("VIDEVALKIT_WORLDSCORE_ROOT", str(_VkPath.home() / ".cache" / "videvalkit" / "upstream" / "WorldScore"))
# Use WorldScore-PATCHED droid_slam (has modified terminate() returning valid_errors)
PATCHED_DROID = f"{WS_ROOT}/worldscore/benchmark/metrics/third_party/droid_slam"
sys.path.insert(0, WS_ROOT)
sys.path.insert(0, PATCHED_DROID)
os.chdir(WS_ROOT)

import numpy as np
import imageio.v3 as iio
from PIL import Image

GEN_ROOT  = _VkPath(_vk_os.environ.get("VIDEVALKIT_WS_GEN_ROOT", str(_VK_CACHE / "smoke-data" / "worldscore" / "videos" / "cogvideox-5b")))
PROMPTS_DIR = _VkPath(_vk_os.environ.get("VIDEVALKIT_WS_PROMPTS_DIR", str(_VK_CACHE / "smoke-data" / "worldscore" / "prompts")))
OUT_DIR     = GEN_ROOT / "eval_10dim"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS = OUT_DIR / "raw_results_3d_v2.jsonl"
N_FRAMES = 49  # Full upstream sampling


def sample_frames_to_disk(mp4_path: Path, tmp_dir: Path) -> list[str]:
    arr = iio.imread(str(mp4_path), plugin="pyav")
    if arr.shape[0] == 0: return []
    idx = np.linspace(0, arr.shape[0]-1, min(N_FRAMES, arr.shape[0])).astype(int).tolist()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, j in enumerate(idx):
        p = tmp_dir / f"f{i:03d}.png"
        Image.fromarray(arr[j]).save(p)
        paths.append(str(p))
    return paths


def main():
    print(f"[{time.strftime('%H:%M:%S')}] init")
    videos = []
    for split, vdir in [("dynamic", GEN_ROOT / "dynamic"), ("static", GEN_ROOT / "static")]:
        for mp4 in sorted(vdir.glob("*.mp4")):
            videos.append((mp4.stem, mp4, split))
    print(f"[{time.strftime('%H:%M:%S')}] {len(videos)} videos to score")

    from worldscore.benchmark.metrics.third_party.reprojection_error_metrics import ReprojectionErrorMetric
    m = ReprojectionErrorMetric()
    m._args.disable_vis = True
    print(f"[{time.strftime('%H:%M:%S')}] metric ready")

    scores = []
    n_err = 0
    t0_all = time.time()
    with RESULTS.open("w") as fout:
        for i, (pid, mp4, split) in enumerate(videos):
            if i % 10 == 0:
                e = time.time() - t0_all
                avg = e / max(1, i) if i else 0
                eta = avg * (len(videos) - i)
                print(f"[{time.strftime('%H:%M:%S')}] [{i+1:3d}/{len(videos)}] {pid}  elapsed={e/60:.1f}min  eta={eta/60:.1f}min")

            tmp = Path(f"/tmp/wseval3d_{pid}")
            try:
                paths = sample_frames_to_disk(mp4, tmp)
                if len(paths) < 8:
                    raise ValueError(f"only {len(paths)} frames")
                s = m._compute_scores(rendered_images=paths)
                scores.append(float(s))
                fout.write(json.dumps({
                    "prompt_id": pid, "split": split, "dimension": "3d_consistency",
                    "score": float(s), "video_path": str(mp4),
                }) + "\n")
                fout.flush()
            except Exception as e:
                n_err += 1
                print(f"  ! {pid}: {type(e).__name__}: {str(e)[:120]}")
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
                # Ensure fresh DROID instance per video
                m.droid = None

    e = time.time() - t0_all
    print(f"\n[{time.strftime('%H:%M:%S')}] DONE in {e/60:.1f} min  errors={n_err}")
    summary = {
        "model": "cogvideox-5b-t2v",
        "dimension": "3d_consistency",
        "n_videos": len(videos),
        "n_errors": n_err,
        "n_scored": len(scores),
        "elapsed_seconds": round(e, 1),
        "mean_score": round(mean(scores), 4) if scores else None,
    }
    (OUT_DIR / "summary_3d.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
