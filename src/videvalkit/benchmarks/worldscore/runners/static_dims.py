"""Re-run the 6 dims that currently use the videvalkit-adapter proxy
(or are missing a metric component), using upstream classes directly.

Dims rerun here:
  - content_alignment   (upstream torchmetrics CLIPScoreMetric, ViT-B/16)
  - object_control      (upstream ObjectDetectionMetric, GroundingDINO + class-match rate)
  - photometric_consistency (upstream OpticalFlowAverageEndPointErrorMetric, SEA-RAFT AEPE)
  - motion_magnitude    (upstream OpticalFlowMetric, SEA-RAFT median flow)
  - subjective_quality_iqa+ (upstream CLIPImageQualityAssessmentPlusMetric, pyiqa clipiqa+)
  - subjective_quality_aesthetic (upstream CLIPAestheticScoreMetric, pyiqa laion_aes)
  - motion_accuracy_rate (re-extract proper class-match rate using spaCy + pred_phrases)

Memory mode: loads each model on demand, frees between dims to fit on a single GPU.

Usage:
  CUDA_VISIBLE_DEVICES=0 python run_ws_upstream_all.py --dims clip,objdet,searaft,aesthetic
"""
import sys, os, time, types, json, shutil, argparse, gc
import os as _vk_os
from pathlib import Path as _VkPath
_VK_CACHE = _VkPath(_vk_os.environ.get("VIDEVALKIT_CACHE_HOME", _VkPath.home() / ".cache" / "videvalkit"))
from pathlib import Path
from statistics import mean

from videvalkit.benchmarks.worldscore.scorers import install_mamba_ssm_shim
install_mamba_ssm_shim()

WS_ROOT = _vk_os.environ.get("VIDEVALKIT_WORLDSCORE_ROOT", str(_VkPath.home() / ".cache" / "videvalkit" / "upstream" / "WorldScore"))
sys.path.insert(0, WS_ROOT)
sys.path.insert(0, f"{WS_ROOT}/worldscore/benchmark/metrics/third_party")
sys.path.insert(0, f"{WS_ROOT}/worldscore/benchmark/metrics/third_party/SEA-RAFT")
os.chdir(WS_ROOT)

import numpy as np, torch
import imageio.v3 as iio
from PIL import Image

GEN_ROOT  = _VkPath(_vk_os.environ.get("VIDEVALKIT_WS_GEN_ROOT", str(_VK_CACHE / "smoke-data" / "worldscore" / "videos" / "cogvideox-5b")))
OUT_DIR  = GEN_ROOT / "eval_10dim"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_WS_PROMPTS = _VkPath(_vk_os.environ.get("VIDEVALKIT_WS_PROMPTS_DIR", str(_VK_CACHE / "smoke-data" / "worldscore" / "prompts")))
STATIC_FLAT = _WS_PROMPTS / "static-flat.jsonl"
STATIC_ENT  = _WS_PROMPTS / "static-entry.jsonl"
DYNAMIC     = _WS_PROMPTS / "dynamic.jsonl"

N_FRAMES_PER_VIDEO = 49  # Full upstream sampling (49 frames per video)
REF_DIR = _VkPath(_vk_os.environ.get("VIDEVALKIT_WS_REFS_DIR", str(GEN_ROOT / "refs")))


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.ipc_collect()


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


def load_static_items():
    """Yield (prompt_id, mp4_path, content_list_for_objdet, prompt_text)."""
    # Build a content_list lookup keyed by prompt_id (sub-prompt)
    entry_lookup = {}
    for line in STATIC_ENT.open():
        entry = json.loads(line)
        eid = entry["entry_id"]
        content_list = entry.get("content_list", [])
        for i, content in enumerate(content_list):
            entry_lookup[f"{eid}_s{i}"] = content
    items = []
    for line in STATIC_FLAT.open():
        sp = json.loads(line)
        pid = sp["prompt_id"]
        mp4 = GEN_ROOT / "static" / f"{pid}.mp4"
        if not mp4.exists(): continue
        content = entry_lookup.get(pid, sp.get("prompt", ""))
        text = sp.get("prompt", "")
        items.append({
            "prompt_id": pid, "split": "static",
            "mp4": mp4, "content_for_objdet": content, "prompt_text": text,
        })
    return items


def load_dynamic_items():
    items = []
    for line in DYNAMIC.open():
        d = json.loads(line)
        pid = d["prompt_id"]
        mp4 = GEN_ROOT / "dynamic" / f"{pid}.mp4"
        if not mp4.exists(): continue
        items.append({
            "prompt_id": pid, "split": "dynamic",
            "mp4": mp4, "prompt_text": d.get("prompt", ""),
            "objects": d.get("objects", []),
        })
    return items


# ============================================================================
# Dim 3: content_alignment — upstream CLIPScoreMetric (torchmetrics ViT-B/16)
# ============================================================================
def run_clip_score():
    """content_alignment: upstream torchmetrics CLIPScore(openai/clip-vit-base-patch16)."""
    from worldscore.benchmark.metrics import CLIPScoreMetric
    metric = CLIPScoreMetric()
    items = load_static_items()  # content_alignment is a STATIC dim
    out = OUT_DIR / "raw_results_clip_score.jsonl"
    print(f"[{time.strftime('%H:%M:%S')}] CLIPScoreMetric ready, scoring {len(items)} static videos")
    t0 = time.time()
    rows = []
    n_err = 0
    with out.open("w") as fout:
        for i, it in enumerate(items):
            tmp = Path(f"/tmp/clip_{it['prompt_id']}")
            try:
                paths = dump_frames(it["mp4"], tmp)
                if not paths: raise ValueError("no frames")
                # Use the scene-only prompt (not composed_prompt with style suffix)
                score = float(metric._compute_scores(paths, it["prompt_text"]))
                row = {"prompt_id": it["prompt_id"], "split": "static",
                       "dimension": "content_alignment", "metric": "clip_score",
                       "score": score, "prompt_text": it["prompt_text"]}
                fout.write(json.dumps(row) + "\n"); fout.flush()
                rows.append(row)
                if i % 10 == 0:
                    eta = (time.time() - t0) / max(1, i+1) * (len(items) - i - 1)
                    print(f"[{time.strftime('%H:%M:%S')}] [{i+1:3d}/{len(items)}] {it['prompt_id']}  score={score:.4f}  eta={eta/60:.1f}min")
            except Exception as e:
                n_err += 1
                print(f"  ! {it['prompt_id']}: {type(e).__name__}: {str(e)[:120]}")
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    print(f"[{time.strftime('%H:%M:%S')}] clip_score DONE in {(time.time()-t0)/60:.1f} min  errors={n_err}")
    del metric; free_gpu()
    return rows


# ============================================================================
# Dim 2: object_control — upstream ObjectDetectionMetric (GroundingDINO + class-match rate)
# ============================================================================
def run_object_detection():
    from worldscore.benchmark.metrics import ObjectDetectionMetric
    metric = ObjectDetectionMetric()
    items = load_static_items()
    out = OUT_DIR / "raw_results_object_detection.jsonl"
    print(f"[{time.strftime('%H:%M:%S')}] ObjectDetectionMetric ready, scoring {len(items)} static videos")
    t0 = time.time()
    rows = []
    n_err = 0
    with out.open("w") as fout:
        for i, it in enumerate(items):
            tmp = Path(f"/tmp/obj_{it['prompt_id']}")
            try:
                paths = dump_frames(it["mp4"], tmp)
                if not paths: raise ValueError("no frames")
                # upstream expects text_prompt like "SceneName, object1, object2"
                # and then strips the first comma-separated chunk
                text = it["content_for_objdet"]
                if not text or "," not in text:
                    text = f"scene, {it['prompt_text']}"  # fallback
                score = float(metric._compute_scores(paths, text))
                row = {"prompt_id": it["prompt_id"], "split": "static",
                       "dimension": "object_control", "metric": "object_detection",
                       "score": score, "text_prompt_used": text}
                fout.write(json.dumps(row) + "\n"); fout.flush()
                rows.append(row)
                if i % 5 == 0:
                    eta = (time.time() - t0) / max(1, i+1) * (len(items) - i - 1)
                    print(f"[{time.strftime('%H:%M:%S')}] [{i+1:3d}/{len(items)}] {it['prompt_id']}  rate={score:.3f}  eta={eta/60:.1f}min")
            except Exception as e:
                n_err += 1
                print(f"  ! {it['prompt_id']}: {type(e).__name__}: {str(e)[:120]}")
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    print(f"[{time.strftime('%H:%M:%S')}] object_detection DONE in {(time.time()-t0)/60:.1f} min  errors={n_err}")
    del metric; free_gpu()
    return rows


# ============================================================================
# Dim 5: photometric_consistency — upstream OpticalFlowAverageEndPointErrorMetric (SEA-RAFT AEPE)
# ============================================================================
def run_aepe():
    from worldscore.benchmark.metrics import OpticalFlowAverageEndPointErrorMetric
    metric = OpticalFlowAverageEndPointErrorMetric()
    items = load_static_items()
    out = OUT_DIR / "raw_results_aepe.jsonl"
    print(f"[{time.strftime('%H:%M:%S')}] OpticalFlowAEPEMetric (SEA-RAFT) ready, scoring {len(items)} static videos")
    t0 = time.time()
    rows = []
    n_err = 0
    with out.open("w") as fout:
        for i, it in enumerate(items):
            tmp = Path(f"/tmp/aepe_{it['prompt_id']}")
            try:
                paths = dump_frames(it["mp4"], tmp)
                if len(paths) < 2: raise ValueError(f"only {len(paths)} frames")
                score = float(metric._compute_scores(paths))
                row = {"prompt_id": it["prompt_id"], "split": "static",
                       "dimension": "photometric_consistency", "metric": "optical_flow_aepe",
                       "score": score}
                fout.write(json.dumps(row) + "\n"); fout.flush()
                rows.append(row)
                if i % 10 == 0:
                    eta = (time.time() - t0) / max(1, i+1) * (len(items) - i - 1)
                    print(f"[{time.strftime('%H:%M:%S')}] [{i+1:3d}/{len(items)}] {it['prompt_id']}  aepe={score:.4f}  eta={eta/60:.1f}min")
            except Exception as e:
                n_err += 1
                print(f"  ! {it['prompt_id']}: {type(e).__name__}: {str(e)[:120]}")
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    print(f"[{time.strftime('%H:%M:%S')}] aepe DONE in {(time.time()-t0)/60:.1f} min  errors={n_err}")
    del metric; free_gpu()
    return rows


# ============================================================================
# Dim 9: motion_magnitude — upstream OpticalFlowMetric (SEA-RAFT median flow)
# ============================================================================
def run_flow():
    from worldscore.benchmark.metrics import OpticalFlowMetric
    metric = OpticalFlowMetric()
    items = load_dynamic_items()  # motion_magnitude is dynamic
    out = OUT_DIR / "raw_results_flow.jsonl"
    print(f"[{time.strftime('%H:%M:%S')}] OpticalFlowMetric (SEA-RAFT) ready, scoring {len(items)} dynamic videos")
    t0 = time.time()
    rows = []
    n_err = 0
    with out.open("w") as fout:
        for i, it in enumerate(items):
            tmp = Path(f"/tmp/flow_{it['prompt_id']}")
            try:
                paths = dump_frames(it["mp4"], tmp)
                if len(paths) < 2: raise ValueError(f"only {len(paths)} frames")
                score = float(metric._compute_scores(paths))
                row = {"prompt_id": it["prompt_id"], "split": "dynamic",
                       "dimension": "motion_magnitude", "metric": "optical_flow",
                       "score": score}
                fout.write(json.dumps(row) + "\n"); fout.flush()
                rows.append(row)
                if i % 10 == 0:
                    eta = (time.time() - t0) / max(1, i+1) * (len(items) - i - 1)
                    print(f"[{time.strftime('%H:%M:%S')}] [{i+1:3d}/{len(items)}] {it['prompt_id']}  flow={score:.4f}  eta={eta/60:.1f}min")
            except Exception as e:
                n_err += 1
                print(f"  ! {it['prompt_id']}: {type(e).__name__}: {str(e)[:120]}")
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    print(f"[{time.strftime('%H:%M:%S')}] flow DONE in {(time.time()-t0)/60:.1f} min  errors={n_err}")
    del metric; free_gpu()
    return rows


# ============================================================================
# Dim 7a: subjective_quality clip_aesthetic — upstream IQACLIPAestheticScoreMetric (pyiqa laion_aes)
# ============================================================================
def run_clip_aesthetic():
    from worldscore.benchmark.metrics import IQACLIPAestheticScoreMetric
    metric = IQACLIPAestheticScoreMetric()
    items = load_static_items()
    out = OUT_DIR / "raw_results_clip_aesthetic.jsonl"
    print(f"[{time.strftime('%H:%M:%S')}] IQACLIPAestheticScoreMetric (pyiqa laion_aes) ready")
    t0 = time.time()
    rows = []
    n_err = 0
    with out.open("w") as fout:
        for i, it in enumerate(items):
            tmp = Path(f"/tmp/aest_{it['prompt_id']}")
            try:
                paths = dump_frames(it["mp4"], tmp)
                if not paths: raise ValueError("no frames")
                score = float(metric._compute_scores(paths))
                row = {"prompt_id": it["prompt_id"], "split": "static",
                       "dimension": "subjective_quality", "metric": "clip_aesthetic",
                       "score": score}
                fout.write(json.dumps(row) + "\n"); fout.flush()
                rows.append(row)
                if i % 10 == 0:
                    eta = (time.time() - t0) / max(1, i+1) * (len(items) - i - 1)
                    print(f"[{time.strftime('%H:%M:%S')}] [{i+1:3d}/{len(items)}] {it['prompt_id']}  aest={score:.3f}  eta={eta/60:.1f}min")
            except Exception as e:
                n_err += 1
                print(f"  ! {it['prompt_id']}: {type(e).__name__}: {str(e)[:120]}")
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    print(f"[{time.strftime('%H:%M:%S')}] clip_aesthetic DONE in {(time.time()-t0)/60:.1f} min  errors={n_err}")
    del metric; free_gpu()
    return rows


# ============================================================================
# Dim 7b: subjective_quality clip_iqa+ — upstream CLIPImageQualityAssessmentPlusMetric
# ============================================================================
def run_clip_iqa_plus():
    from worldscore.benchmark.metrics import CLIPImageQualityAssessmentPlusMetric
    metric = CLIPImageQualityAssessmentPlusMetric()
    items = load_static_items()
    out = OUT_DIR / "raw_results_clip_iqa_plus.jsonl"
    print(f"[{time.strftime('%H:%M:%S')}] CLIPImageQualityAssessmentPlusMetric (pyiqa clipiqa+) ready")
    t0 = time.time()
    rows = []
    n_err = 0
    with out.open("w") as fout:
        for i, it in enumerate(items):
            tmp = Path(f"/tmp/iqa_{it['prompt_id']}")
            try:
                paths = dump_frames(it["mp4"], tmp)
                if not paths: raise ValueError("no frames")
                score = float(metric._compute_scores(paths))
                row = {"prompt_id": it["prompt_id"], "split": "static",
                       "dimension": "subjective_quality", "metric": "clip_iqa+",
                       "score": score}
                fout.write(json.dumps(row) + "\n"); fout.flush()
                rows.append(row)
                if i % 10 == 0:
                    eta = (time.time() - t0) / max(1, i+1) * (len(items) - i - 1)
                    print(f"[{time.strftime('%H:%M:%S')}] [{i+1:3d}/{len(items)}] {it['prompt_id']}  iqa={score:.4f}  eta={eta/60:.1f}min")
            except Exception as e:
                n_err += 1
                print(f"  ! {it['prompt_id']}: {type(e).__name__}: {str(e)[:120]}")
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    print(f"[{time.strftime('%H:%M:%S')}] clip_iqa+ DONE in {(time.time()-t0)/60:.1f} min  errors={n_err}")
    del metric; free_gpu()
    return rows


# ============================================================================
# ============================================================================
# Dim 6: style_consistency — upstream GramMatrixMetric with WorldScore reference image
# ============================================================================
def run_gram():
    from worldscore.benchmark.metrics import GramMatrixMetric
    metric = GramMatrixMetric()
    items = load_static_items()
    out = OUT_DIR / "raw_results_gram.jsonl"
    print(f"[{time.strftime('%H:%M:%S')}] GramMatrixMetric ready, scoring {len(items)} static videos")
    t0 = time.time()
    rows = []
    n_err = 0
    n_no_ref = 0
    with out.open("w") as fout:
        for i, it in enumerate(items):
            tmp = Path(f"/tmp/gram_{it['prompt_id']}")
            try:
                paths = dump_frames(it["mp4"], tmp)
                if not paths: raise ValueError("no frames")
                # Reference image is keyed by parent entry_id (e.g. ws_sta_1958 from sub-prompt ws_sta_1958_s0)
                parent = it["prompt_id"].rsplit("_s", 1)[0]
                ref = REF_DIR / f"{parent}.png"
                if not ref.exists():
                    n_no_ref += 1
                    print(f"  ! no ref for {it['prompt_id']} (looking for {ref})")
                    continue
                score = float(metric._compute_scores(str(ref), paths))
                row = {"prompt_id": it["prompt_id"], "split": "static",
                       "dimension": "style_consistency", "metric": "gram_matrix",
                       "score": score, "reference_image": str(ref)}
                fout.write(json.dumps(row) + "\n"); fout.flush()
                rows.append(row)
                if i % 10 == 0:
                    eta = (time.time() - t0) / max(1, i+1) * (len(items) - i - 1)
                    print(f"[{time.strftime('%H:%M:%S')}] [{i+1:3d}/{len(items)}] {it['prompt_id']}  gram={score:.6f}  eta={eta/60:.1f}min")
            except Exception as e:
                n_err += 1
                print(f"  ! {it['prompt_id']}: {type(e).__name__}: {str(e)[:120]}")
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    print(f"[{time.strftime('%H:%M:%S')}] gram DONE in {(time.time()-t0)/60:.1f} min  errors={n_err}  no_ref={n_no_ref}")
    del metric; free_gpu()
    return rows


DIMS = {
    "clip":      run_clip_score,
    "objdet":    run_object_detection,
    "aepe":      run_aepe,
    "flow":      run_flow,
    "aesthetic": run_clip_aesthetic,
    "iqaplus":   run_clip_iqa_plus,
    "gram":      run_gram,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=str, default=",".join(DIMS.keys()),
                        help="Comma-separated list of dims to run: " + ",".join(DIMS.keys()))
    args = parser.parse_args()
    requested = [d.strip() for d in args.dims.split(",") if d.strip()]
    for d in requested:
        if d not in DIMS:
            print(f"unknown dim: {d}; choose from {list(DIMS.keys())}")
            sys.exit(1)
    print(f"[{time.strftime('%H:%M:%S')}] running dims: {requested}")
    for d in requested:
        print(f"\n{'='*70}\n  {d}\n{'='*70}")
        DIMS[d]()


if __name__ == "__main__":
    main()
