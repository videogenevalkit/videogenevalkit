"""Motion accuracy v2 — exact upstream class-matching rate via spaCy + 49 frames.

Differences from v1:
  1. rate = upstream's `count / len(prompt_list)` via the same two-pass spaCy
     noun matching as motion_accuracy_metrics.py (lines 113-152 / 182-205).
  2. Frames sampled = 49 (full video), not 17.
"""
import sys, os, time, types, json, shutil, re, argparse
import os as _vk_os
from pathlib import Path as _VkPath
_VK_CACHE = _VkPath(_vk_os.environ.get("VIDEVALKIT_CACHE_HOME", _VkPath.home() / ".cache" / "videvalkit"))
from pathlib import Path
from statistics import mean

from videvalkit.benchmarks.worldscore.scorers import install_mamba_ssm_shim
install_mamba_ssm_shim()

WS_ROOT = _vk_os.environ.get("VIDEVALKIT_WORLDSCORE_ROOT", str(_VkPath.home() / ".cache" / "videvalkit" / "upstream" / "WorldScore"))
PATCHED_DROID = f"{WS_ROOT}/worldscore/benchmark/metrics/third_party/droid_slam"
sys.path.insert(0, WS_ROOT)
sys.path.insert(0, PATCHED_DROID)
sys.path.insert(0, f"{WS_ROOT}/worldscore/benchmark/metrics/third_party/SEA-RAFT")
sys.path.insert(0, f"{WS_ROOT}/worldscore/benchmark/metrics/third_party")
os.chdir(WS_ROOT)

import numpy as np, torch
import torch.nn.functional as F
import cv2
import imageio.v3 as iio
from PIL import Image
from torchvision.transforms import ToPILImage

# Upstream's exact NLP helpers — copied for rate calculation
import spacy
nlp = spacy.load("en_core_web_sm")

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from segment_anything import sam_model_registry, SamPredictor
from sam2.build_sam import build_sam2_video_predictor
import hydra
from hydra.core.global_hydra import GlobalHydra

from core.raft import RAFT
from core.utils.utils import load_ckpt
from core.parser import parse_args as raft_parse_args


GEN_DYN  = _VkPath(_vk_os.environ.get("VIDEVALKIT_WS_GEN_DYNAMIC", str(_VK_CACHE / "smoke-data" / "worldscore" / "videos" / "cogvideox-5b" / "dynamic")))
PROMPTS  = _VkPath(_vk_os.environ.get("VIDEVALKIT_WS_DYNAMIC_PROMPTS", str(_VK_CACHE / "smoke-data" / "worldscore" / "prompts" / "dynamic.jsonl")))
OUT_DIR  = _VkPath(_vk_os.environ.get("VIDEVALKIT_WS_OUT_DIR", "./worldscore_eval_out"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_FRAMES = 49  # Full sampling, matching upstream


# =================== Upstream spaCy class-matching helpers ===================

def standardize_string(s):
    return re.sub(r"\s*([_\-*/+])\s*", r"\1", s)


def extract_noun_adjective(phrase):
    doc = nlp(phrase)
    adjectives, nouns = [], []
    for token in doc:
        if token.pos_ in ["ADJ", "JJ", "JJR", "JJS"]:
            adjectives.append(token.text.lower())
        elif token.pos_ in ["NOUN", "PROPN", "NN", "NNS", "NNP", "NNPS"]:
            nouns.append(token.text.lower())
        elif token.pos_ == "VERB" and token.tag_ == "VBN":
            adjectives.append(token.text.lower())
    return adjectives, nouns


def get_match_point(result, prompt_list):
    matched_prompts = set()
    if "##" in result:
        result = result.replace("##", "")
        for prompt in prompt_list:
            if result in prompt:
                matched_prompts.add(prompt)
                return 1, matched_prompts
    else:
        result_adjectives, result_nouns = extract_noun_adjective(result)
        for prompt in prompt_list:
            prompt_adjectives, prompt_nouns = extract_noun_adjective(prompt)
            noun_matches = sum(1 for noun in result_nouns if noun in prompt_nouns)
            if noun_matches == 0:
                continue
            matched_prompts.add(prompt)
            return 1, matched_prompts
    return 0, matched_prompts


def compute_upstream_rate(pred_phrases, objects_list):
    """Exact upstream rate calc per motion_accuracy_metrics.py:182-206."""
    prompt_list = [p.lower() for p in objects_list]
    count = 0
    result_cache = set()
    remaining_prompts = set(prompt_list)

    # First pass: exact matches
    for phrase in pred_phrases:
        result = re.sub(r'\(.*?\)', '', phrase).strip().lower()
        result = standardize_string(result)
        if result in result_cache or result not in remaining_prompts:
            continue
        result_cache.add(result)
        if result in remaining_prompts:
            count += 1
            remaining_prompts.remove(result)

    # Second pass: partial matches
    for phrase in pred_phrases:
        result = re.sub(r'\(.*?\)', '', phrase).strip().lower()
        result = standardize_string(result)
        if result not in result_cache:
            point, matched_prompts = get_match_point(result, list(remaining_prompts))
            count += point
            remaining_prompts -= matched_prompts

    rate = min(count / max(1, len(prompt_list)), 1.0)
    return rate, count


# =================== GroundingDINO + SAM-ViT-H ===================

def load_grounding_dino_model(config_file, checkpoint, device):
    args = SLConfig.fromfile(config_file)
    args.device = device
    args.bert_base_uncased_path = None
    model = build_model(args)
    sd = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(clean_state_dict(sd["model"]), strict=False)
    model.eval().to(device)
    return model


def load_gdino_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.RandomResize([800], max_size=1333), T.ToTensor(),
                           T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image, _ = transform(image_pil, None)
    return image_pil, image


def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cuda"):
    """Returns boxes_filt + pred_phrases (with logits in parens, matching upstream)."""
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]
    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    pred_phrases = []
    for logit in logits_filt:
        phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(phrase + f"({str(logit.max().item())[:4]})")  # with logit string, matching upstream
    return boxes_filt, pred_phrases


# =================== SAM2 video propagation ===================

def vos_inference(predictor, video_dir, mask_dir, score_thresh=0.0):
    inference_state = predictor.init_state(video_path=video_dir, async_loading_frames=False)
    H, W = inference_state["video_height"], inference_state["video_width"]
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])
    if not mask_files:
        return []
    for object_id, fname in enumerate(mask_files):
        mask = np.array(Image.open(os.path.join(mask_dir, fname))) > 0
        predictor.add_new_mask(inference_state=inference_state,
                               frame_idx=0, obj_id=object_id, mask=mask)
    video_segments = {}
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
        out_mask = np.zeros((H, W), dtype=np.bool_)
        for i, _ in enumerate(obj_ids):
            out_mask |= (mask_logits[i] > score_thresh).cpu().numpy().reshape(H, W)
        kernel = np.ones((21, 21), np.uint8)
        out_mask = cv2.dilate(out_mask.astype(np.uint8), kernel, iterations=1).astype(np.bool_)
        video_segments[frame_idx] = out_mask
    return [video_segments[i] for i in sorted(video_segments)]


def mask_resize(mask, target_shape):
    m = (mask.astype(np.uint8) > 0).astype(np.uint8)
    m = cv2.resize(m, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return m > 0


# =================== SEA-RAFT optical flow ===================

def init_searaft(device="cuda"):
    args = argparse.Namespace(
        cfg=f"{WS_ROOT}/worldscore/benchmark/metrics/third_party/SEA-RAFT/config/eval/spring-M.json",
        path=f"{WS_ROOT}/worldscore/benchmark/metrics/checkpoints/Tartan-C-T-TSKH-spring540x960-M.pth",
    )
    args = raft_parse_args(args)
    model = RAFT(args)
    load_ckpt(model, args.path)
    model.to(device).eval()
    return model, args


def load_image_for_raft(imfile, device="cuda"):
    image = cv2.imread(imfile)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)[None].to(device)


def compute_flow(model, args, image1, image2, device="cuda"):
    img1 = F.interpolate(image1, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    img2 = F.interpolate(image2, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    with torch.amp.autocast(device_type="cuda"):
        output = model(img1, img2, iters=args.iters, test_mode=True)
    flow_final = output['flow'][-1]
    flow_down = F.interpolate(flow_final, scale_factor=0.5 ** args.scale, mode='bilinear', align_corners=False) * (0.5 ** args.scale)
    return flow_down.cpu().numpy().squeeze().transpose(1, 2, 0)


def score_one(
    frame_paths: list[str],
    objects: list[str],
    gdino,
    sam_predictor,
    sam2_predictor,
    raft_model,
    raft_args,
    box_threshold: float = 0.4,
    text_threshold: float = 0.4,
    device: str = "cuda",
) -> dict:
    """Score one video's motion_accuracy. Mirrors upstream's pipeline:
    GroundingDINO -> spaCy class-match rate -> SAM-ViT-H first-frame masks ->
    SAM2 video propagation -> SEA-RAFT per-pair flow -> max(obj)-max(bg)
    alignment averaged across pairs, multiplied by rate.

    Returns ``{score, base_score, rate, match_count, pred_phrases, n_pairs}``
    or ``{score: 0.0, ...}`` if no objects detected.
    """
    import shutil
    frame_dir = Path(frame_paths[0]).parent
    pid = frame_dir.name
    mask_dir = Path(f"/tmp/wseval_ma_{pid}/masks")
    mask_dir.mkdir(parents=True, exist_ok=True)
    for f in os.listdir(mask_dir):
        if f.endswith(".png"): os.remove(os.path.join(mask_dir, f))

    caption = ", ".join(o.lower() for o in objects)
    _, image_t = load_gdino_image(frame_paths[0])
    boxes_filt, pred_phrases = get_grounding_output(
        gdino, image_t, caption, box_threshold, text_threshold, device,
    )
    if boxes_filt.size(0) == 0:
        return {"score": 0.0, "base_score": None, "rate": 0.0, "match_count": 0,
                "pred_phrases": [], "n_pairs": 0, "n_objects": len(objects)}
    rate, count = compute_upstream_rate(pred_phrases, objects)

    image_pil = Image.open(frame_paths[0]).convert("RGB")
    W, H = image_pil.size
    img_rgb = np.array(image_pil)
    sam_predictor.set_image(img_rgb)
    boxes_filt = boxes_filt.clone()
    for k in range(boxes_filt.size(0)):
        boxes_filt[k] = boxes_filt[k] * torch.Tensor([W, H, W, H])
        boxes_filt[k][:2] -= boxes_filt[k][2:] / 2
        boxes_filt[k][2:] += boxes_filt[k][:2]
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(
        boxes_filt, img_rgb.shape[:2]).to(device)
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None, point_labels=None,
        boxes=transformed_boxes, multimask_output=False,
    )
    if masks.size(0) == 0:
        return {"score": 0.0, "base_score": None, "rate": rate, "match_count": count,
                "pred_phrases": pred_phrases, "n_pairs": 0, "n_objects": len(objects)}
    for k, mask in enumerate(masks):
        ToPILImage()(mask.cpu().float()).save(os.path.join(mask_dir, f"{k:03d}.png"))

    seg_masks = vos_inference(sam2_predictor, str(frame_dir), str(mask_dir), score_thresh=0.0)
    if len(seg_masks) < 2:
        return {"score": 0.0, "base_score": None, "rate": rate, "match_count": count,
                "pred_phrases": pred_phrases, "n_pairs": 0, "n_objects": len(objects)}

    scores_pair = []
    with torch.no_grad():
        for k in range(len(frame_paths) - 1):
            if k >= len(seg_masks) - 1: break
            img1 = load_image_for_raft(frame_paths[k], device)
            img2 = load_image_for_raft(frame_paths[k+1], device)
            m = seg_masks[k]
            if m.shape != img1.shape[2:]:
                m = mask_resize(m, img1.shape[2:])
            flow = compute_flow(raft_model, raft_args, img1, img2, device)
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            obj_mag, bg_mag = mag[m], mag[~m]
            if obj_mag.size == 0 or bg_mag.size == 0: continue
            scores_pair.append(float(obj_mag.max() - bg_mag.max()))
    shutil.rmtree(mask_dir.parent, ignore_errors=True)
    if not scores_pair:
        return {"score": 0.0, "base_score": None, "rate": rate, "match_count": count,
                "pred_phrases": pred_phrases, "n_pairs": 0, "n_objects": len(objects)}
    base = sum(scores_pair) / len(scores_pair)
    return {"score": float(base * rate), "base_score": float(base), "rate": float(rate),
            "match_count": int(count), "pred_phrases": pred_phrases,
            "n_pairs": len(scores_pair), "n_objects": len(objects)}


def main():
    device = "cuda"
    print(f"[{time.strftime('%H:%M:%S')}] init")

    raft_model, raft_args = init_searaft(device)
    print(f"[{time.strftime('%H:%M:%S')}] SEA-RAFT loaded")

    GlobalHydra.instance().clear()
    from hydra import initialize_config_dir
    initialize_config_dir(config_dir=f"{WS_ROOT}/worldscore/benchmark/metrics/third_party", version_base=None)
    sam2_predictor = build_sam2_video_predictor(
        config_file="sam2/configs/sam2.1/sam2.1_hiera_b+.yaml",
        ckpt_path=f"{WS_ROOT}/worldscore/benchmark/metrics/checkpoints/sam2.1_hiera_base_plus.pt",
        apply_postprocessing=False,
        hydra_overrides_extra=["++model.non_overlap_masks=false"],
    )
    print(f"[{time.strftime('%H:%M:%S')}] SAM2 loaded")

    gdino = load_grounding_dino_model(
        f"{WS_ROOT}/worldscore/benchmark/metrics/third_party/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        f"{WS_ROOT}/worldscore/benchmark/metrics/checkpoints/groundingdino_swint_ogc.pth",
        device,
    )
    sam_vith = sam_model_registry["vit_h"](
        checkpoint=f"{WS_ROOT}/worldscore/benchmark/metrics/checkpoints/sam_vit_h_4b8939.pth"
    ).to(device)
    sam_predictor = SamPredictor(sam_vith)
    print(f"[{time.strftime('%H:%M:%S')}] GroundingDINO + SAM-ViT-H loaded")

    entries = [json.loads(l) for l in PROMPTS.open()]
    print(f"[{time.strftime('%H:%M:%S')}] {len(entries)} dynamic entries")

    raw = OUT_DIR / "raw_results_motion_accuracy_v2.jsonl"
    rows = []
    n_err = 0
    t0_all = time.time()
    BOX_TH = 0.4
    TEXT_TH = 0.4

    with raw.open("w") as fout:
        for i, entry in enumerate(entries):
            pid = entry["prompt_id"]
            objects_list = entry["objects"]
            mp4 = GEN_DYN / f"{pid}.mp4"
            if not mp4.exists():
                n_err += 1; continue

            frame_dir = Path(f"/tmp/wseval_ma2_{pid}/frames")
            mask_dir = Path(f"/tmp/wseval_ma2_{pid}/masks")
            try:
                # 1. Extract frames (full 49)
                arr = iio.imread(str(mp4), plugin="pyav")
                if arr.shape[0] == 0:
                    raise ValueError("empty video")
                frame_dir.mkdir(parents=True, exist_ok=True)
                mask_dir.mkdir(parents=True, exist_ok=True)
                idx = np.linspace(0, arr.shape[0]-1, min(N_FRAMES, arr.shape[0])).astype(int)
                frame_paths = []
                for k, j in enumerate(idx):
                    p = frame_dir / f"{k:03d}.jpg"
                    Image.fromarray(arr[j]).save(p, quality=95)
                    frame_paths.append(str(p))

                # 2. GroundingDINO on frame 0 → boxes + pred_phrases
                t1 = time.time()
                caption = ", ".join(o.lower() for o in objects_list)
                _, image_t = load_gdino_image(frame_paths[0])
                boxes_filt, pred_phrases = get_grounding_output(
                    gdino, image_t, caption, BOX_TH, TEXT_TH, device,
                )

                # 3. UPSTREAM RATE via spaCy class-matching
                if boxes_filt.size(0) == 0:
                    rate, count = 0.0, 0
                else:
                    rate, count = compute_upstream_rate(pred_phrases, objects_list)

                if boxes_filt.size(0) == 0:
                    row = {"prompt_id": pid, "split": "dynamic", "dimension": "motion_accuracy",
                           "score": 0.0, "base_score": None, "rate": 0.0, "match_count": 0,
                           "n_objects": len(objects_list), "n_pred_phrases": 0,
                           "pred_phrases": [], "motion_type": entry.get("motion_type")}
                    fout.write(json.dumps(row) + "\n"); fout.flush()
                    rows.append(row)
                    print(f"[{time.strftime('%H:%M:%S')}] [{i+1:3d}/{len(entries)}] {pid}  NO DETECTION → 0.0")
                    continue

                # 4. SAM with boxes → first-frame masks
                image_pil = Image.open(frame_paths[0]).convert("RGB")
                W, H = image_pil.size
                img_rgb = np.array(image_pil)
                sam_predictor.set_image(img_rgb)
                boxes_filt = boxes_filt.clone()
                for k in range(boxes_filt.size(0)):
                    boxes_filt[k] = boxes_filt[k] * torch.Tensor([W, H, W, H])
                    boxes_filt[k][:2] -= boxes_filt[k][2:] / 2
                    boxes_filt[k][2:] += boxes_filt[k][:2]
                transformed_boxes = sam_predictor.transform.apply_boxes_torch(
                    boxes_filt, img_rgb.shape[:2]).to(device)
                masks, _, _ = sam_predictor.predict_torch(
                    point_coords=None, point_labels=None,
                    boxes=transformed_boxes, multimask_output=False,
                )
                if masks.size(0) == 0:
                    raise RuntimeError("SAM returned no masks")

                for k, mask in enumerate(masks):
                    ToPILImage()(mask.cpu().float()).save(os.path.join(mask_dir, f"{k:03d}.png"))

                # 5. SAM2 propagation across all frames
                t2 = time.time()
                seg_masks = vos_inference(sam2_predictor, str(frame_dir), str(mask_dir), score_thresh=0.0)
                if len(seg_masks) < 2:
                    raise RuntimeError(f"sam2 returned only {len(seg_masks)} masks")

                # 6. SEA-RAFT flow + per-pair motion_alignment
                t3 = time.time()
                scores_pair = []
                with torch.no_grad():
                    for k in range(len(frame_paths) - 1):
                        if k >= len(seg_masks) - 1: break
                        img1 = load_image_for_raft(frame_paths[k], device)
                        img2 = load_image_for_raft(frame_paths[k+1], device)
                        mask = seg_masks[k]
                        if mask.shape != img1.shape[2:]:
                            mask = mask_resize(mask, img1.shape[2:])
                        flow = compute_flow(raft_model, raft_args, img1, img2, device)
                        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                        obj_mag = mag[mask]
                        bg_mag = mag[~mask]
                        if obj_mag.size == 0 or bg_mag.size == 0:
                            continue
                        # Upstream uses max for both
                        align = float(obj_mag.max() - bg_mag.max())
                        scores_pair.append(align)

                if not scores_pair:
                    raise RuntimeError("no valid scoring pairs")

                base_score = sum(scores_pair) / len(scores_pair)
                score = base_score * rate
                dt = time.time() - t1
                row = {
                    "prompt_id": pid, "split": "dynamic", "dimension": "motion_accuracy",
                    "score": float(score),
                    "base_score": float(base_score),
                    "rate": float(rate),
                    "match_count": int(count),
                    "n_objects": int(len(objects_list)),
                    "n_pred_phrases": int(len(pred_phrases)),
                    "pred_phrases": pred_phrases,
                    "n_pairs_scored": len(scores_pair),
                    "motion_type": entry.get("motion_type"),
                    "elapsed_seconds": round(dt, 1),
                }
                fout.write(json.dumps(row) + "\n"); fout.flush()
                rows.append(row)
                eta = (time.time() - t0_all) / max(1, i+1) * (len(entries) - i - 1)
                if i % 5 == 0 or i == len(entries) - 1:
                    print(f"[{time.strftime('%H:%M:%S')}] [{i+1:3d}/{len(entries)}] {pid}  score={score:.3f} (rate={rate:.2f}, count={count}/{len(objects_list)})  ({dt:.1f}s, eta={eta/60:.1f}min)")

            except Exception as e:
                n_err += 1
                print(f"  ! {pid}: {type(e).__name__}: {str(e)[:120]}")
            finally:
                shutil.rmtree(frame_dir.parent, ignore_errors=True)

    elapsed = time.time() - t0_all
    print(f"\n[{time.strftime('%H:%M:%S')}] DONE in {elapsed/60:.1f} min  errors={n_err}")
    if rows:
        ss = [r["score"] for r in rows]
        rates = [r["rate"] for r in rows]
        summary = {
            "model": "cogvideox-5b-t2v", "dimension": "motion_accuracy",
            "n_videos": len(entries), "n_scored": len(rows), "n_errors": n_err,
            "elapsed_seconds": round(elapsed, 1),
            "mean_score": round(mean(ss), 4),
            "mean_rate": round(mean(rates), 4),
            "method": "upstream spaCy class-matching + 49 frames",
        }
        (OUT_DIR / "summary_motion_accuracy_v2.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
