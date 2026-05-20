"""WorldScore adapter — 10-of-10 upstream-class implementation.

Mirrors the open-source WorldScore repo
(https://github.com/Howieeeee/WorldScore, snapshot at
``toolkit/validation/upstream_repos/WorldScore/``). Every dim's score comes
from the upstream metric class directly; per-instance normalization and the
two headlines follow ``worldscore/run_evaluate.py`` exactly.

  Static dims (7) -- evaluated on the static-split videos
  ───────────────────────────────────────────────────────
  1. camera_control          CameraErrorMetric  (DROID-SLAM)
  2. object_control          ObjectDetectionMetric  (GroundingDINO + spaCy)
  3. content_alignment       CLIPScoreMetric  (openai/clip-vit-base-patch16)
  4. 3d_consistency          ReprojectionErrorMetric  (DROID-SLAM)
  5. photometric_consistency OpticalFlowAverageEndPointErrorMetric  (SEA-RAFT AEPE)
  6. style_consistency       GramMatrixMetric  (VGG-19, WorldScore ref img)
  7. subjective_quality      IQACLIPAesthetic + CLIPImageQualityAssessmentPlus

  Dynamic dims (3) -- evaluated on the dynamic-split videos
  ─────────────────────────────────────────────────────────
  8.  motion_accuracy        SEA-RAFT + SAM2 + SAM-ViT-H + GroundingDINO
  9.  motion_magnitude       OpticalFlowMetric  (SEA-RAFT)
  10. motion_smoothness      MotionSmoothnessMetric  (VFIMamba + SSIM + LPIPS + MSE)

  Headlines (mirroring ``run_evaluate.py:165-166``)
  ─────────────────────────────────────────────────
  WorldScore-Static  = mean of the 7 static dim values (× 100)
  WorldScore-Dynamic = mean of ALL 10 dim values (× 100)

Frame sampling: 49 per video (= upstream's ``interpframe_num``).
"""
from __future__ import annotations

import json
import logging
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from videvalkit.core.benchmark import BaseBenchmark
from videvalkit.core.layout import WorkspaceLayout
from videvalkit.core.types import PromptItem, RawResult, Summary, VideoSpec

log = logging.getLogger(__name__)


# Match upstream's ``worldscore_list``
WORLDSCORE_STATIC_DIMS: tuple[str, ...] = (
    "camera_control",
    "object_control",
    "content_alignment",
    "3d_consistency",
    "photometric_consistency",
    "style_consistency",
    "subjective_quality",
)
WORLDSCORE_DYNAMIC_DIMS: tuple[str, ...] = (
    "motion_accuracy",
    "motion_magnitude",
    "motion_smoothness",
)
WORLDSCORE_DIMENSIONS: list[str] = list(WORLDSCORE_STATIC_DIMS) + list(WORLDSCORE_DYNAMIC_DIMS)


WORLDSCORE_DIM_DEFINITIONS: dict[str, str] = {
    "camera_control":          "DROID-SLAM-estimated camera trajectory vs GT camera_path; (R_err°, T_err) tuple, both lower-is-better.",
    "object_control":          "GroundingDINO detection rate of the prompt's required objects, evaluated as upstream's spaCy class-match count / n_objects.",
    "content_alignment":       "Mean per-frame CLIPScore (openai/clip-vit-base-patch16) against the prompt; range [0, 100].",
    "3d_consistency":          "DROID-SLAM reprojection error across the rendered frames; lower-is-better.",
    "photometric_consistency": "SEA-RAFT bidirectional Average End-Point Error (AEPE) between adjacent frames; lower-is-better.",
    "style_consistency":       "VGG-19 gram-matrix MSE between each frame and the WorldScore reference image (input_image.png); lower-is-better.",
    "subjective_quality":      "Mean of (a) pyiqa laion_aes (CLIP+MLP aesthetic) and (b) pyiqa clipiqa+ (CLIP-IQA+).",
    "motion_accuracy":         "SAM2-propagated object mask × SEA-RAFT flow: max(obj_flow) - max(bg_flow) averaged across frame pairs, multiplied by the spaCy class-match rate.",
    "motion_magnitude":        "SEA-RAFT median flow magnitude across adjacent frame pairs; higher = more motion.",
    "motion_smoothness":       "VFIMamba interpolates the middle of each (f[i-1], f[i+1]); SSIM + LPIPS + MSE against the actual f[i].",
}

WORLDSCORE_FRAME_DEFAULT: int = 49  # matches upstream's interpframe_num


class WorldScoreBenchmark(BaseBenchmark):
    name = "worldscore"
    env_name = "videvalkit"
    dimensions = WORLDSCORE_DIMENSIONS
    static_dimensions = list(WORLDSCORE_STATIC_DIMS)
    dynamic_dimensions = list(WORLDSCORE_DYNAMIC_DIMS)
    video_layout = "{split}/{prompt_id}.mp4"

    SAMPLES_PER_PROMPT: int = 1

    # ---- prompts ------------------------------------------------------------
    def list_prompts(
        self,
        dimensions: list[str] | None = None,
        prompts_file: str | Path | None = None,
    ) -> Iterator[PromptItem]:
        """Yield prompts from the WorldScore-style JSONL.

        Expected row keys:
          prompt_id, split ("static"|"dynamic"), prompt, objects, camera_path,
          optionally: motion_type, content_list, parent_entry, style.
        """
        wanted = set(dimensions) if dimensions else set(self.dimensions)
        if prompts_file is None:
            yield PromptItem(
                prompt_id="ws_dyn_0000",
                text="a placeholder dynamic prompt for worldscore smoke",
                dimensions=sorted(wanted & set(WORLDSCORE_DYNAMIC_DIMS)),
                meta={"split": "dynamic", "objects": ["scene"]},
            )
            return
        with Path(prompts_file).open() as f:
            for line in f:
                line = line.strip()
                if not line: continue
                e = json.loads(line)
                pid = str(e.get("prompt_id") or e.get("entry_id"))
                split = e.get("split") or ("dynamic" if pid.startswith("ws_dyn_") else "static")
                relevant_dims = WORLDSCORE_STATIC_DIMS if split == "static" else WORLDSCORE_DYNAMIC_DIMS
                yield PromptItem(
                    prompt_id=pid,
                    text=e.get("prompt", ""),
                    dimensions=sorted(set(relevant_dims) & wanted),
                    meta={k: v for k, v in e.items() if k not in {"prompt_id", "prompt"}},
                )

    def list_required_videos(
        self,
        prompts: list[PromptItem],
        models: list[str],
        layout: WorkspaceLayout,
        samples_per_prompt: int | None = None,
    ) -> list[VideoSpec]:
        n = samples_per_prompt if samples_per_prompt is not None else self.SAMPLES_PER_PROMPT
        specs: list[VideoSpec] = []
        for m in models:
            for p in prompts:
                split = (p.meta or {}).get("split")
                if not split:
                    split = "dynamic" if p.prompt_id.startswith("ws_dyn_") else "static"
                for idx in range(n):
                    rel = self.video_layout.format(split=split, prompt_id=p.prompt_id)
                    specs.append(VideoSpec(
                        path=layout.videos_dir / m / rel,
                        prompt_id=p.prompt_id,
                        model_name=m,
                        sample_index=idx,
                    ))
        return specs

    # ---- evaluate -----------------------------------------------------------
    def evaluate(
        self,
        videos: list[VideoSpec] | None = None,
        layout: WorkspaceLayout | None = None,
        dimensions: list[str] | None = None,
        judge: dict[str, Any] | None = None,
        models: list[str] | None = None,
        prompts: list[PromptItem] | None = None,
        n_frames: int = WORLDSCORE_FRAME_DEFAULT,
        device: str | None = None,
        reference_image_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> list[RawResult]:
        """Score each video on the requested dims, via upstream metric classes.

        ``reference_image_dir`` (style_consistency only): directory containing
        ``{entry_id}.png`` reference images extracted from the WorldScore HF
        dataset (``Howieeeee/WorldScore``); see runners/extract_refs.py.
        """
        if not videos: return []
        wanted = set(dimensions) if dimensions else set(self.dimensions)
        unknown = wanted - set(self.dimensions)
        if unknown:
            raise ValueError(f"unknown worldscore dimension(s): {unknown}")

        # Resolve per-prompt metadata
        prompt_meta: dict[str, dict] = {}
        for p in (prompts or []):
            prompt_meta[p.prompt_id] = {"text": p.text, **(p.meta or {})}

        from videvalkit.benchmarks.worldscore.scorers import (
            CLIPScoreScorer, ObjectDetectionScorer, OpticalFlowAEPEScorer,
            OpticalFlowScorer, GramMatrixScorer, CLIPAestheticScorer,
            CLIPIQAPlusScorer, ReprojectionErrorScorer, CameraErrorScorer,
            MotionSmoothnessScorer, MotionAccuracyScorer,
            extract_frames_to_disk,
        )

        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        # Per-dim scorers, lazy-built only if requested.
        builders = {
            "camera_control":          CameraErrorScorer,
            "object_control":          ObjectDetectionScorer,
            "content_alignment":       CLIPScoreScorer,
            "3d_consistency":          ReprojectionErrorScorer,
            "photometric_consistency": OpticalFlowAEPEScorer,
            "style_consistency":       GramMatrixScorer,
            "motion_accuracy":         MotionAccuracyScorer,
            "motion_magnitude":        OpticalFlowScorer,
            "motion_smoothness":       MotionSmoothnessScorer,
        }
        scorers: dict[str, Any] = {}
        for d in wanted:
            if d == "subjective_quality":
                scorers["subjective_quality.clip_aesthetic"] = CLIPAestheticScorer()
                scorers["subjective_quality.clip_iqa+"] = CLIPIQAPlusScorer()
            elif d in builders:
                scorers[d] = builders[d]()

        # style_consistency auto-resolve: the smoke-data / WorldScore layout
        # ships reference frames in a `refs/` folder beside the split dirs
        # (`<root>/<model>/refs/<entry_id>.png`). If the caller did not pass
        # `reference_image_dir`, look for that sibling folder.
        if reference_image_dir is None and "style_consistency" in wanted:
            for vs in videos:
                cand = Path(vs.path).resolve().parent.parent / "refs"
                if cand.is_dir():
                    reference_image_dir = cand
                    log.info("worldscore: auto-resolved reference_image_dir -> %s", cand)
                    break

        results: list[RawResult] = []
        for vs in videos:
            meta = prompt_meta.get(vs.prompt_id, {})
            split = meta.get("split") or ("dynamic" if vs.prompt_id.startswith("ws_dyn_") else "static")
            # Static dims only on static videos; dynamic on dynamic
            allowed = set(WORLDSCORE_STATIC_DIMS if split == "static" else WORLDSCORE_DYNAMIC_DIMS)

            tmp = Path(tempfile.mkdtemp(prefix=f"ws_{vs.prompt_id}_"))
            try:
                frame_paths = extract_frames_to_disk(vs.path, tmp / "frames", n=n_frames)
                if not frame_paths:
                    log.warning("worldscore: skip %s (no frames)", vs.path); continue

                for dim in sorted(wanted & allowed):
                    try:
                        score, scorer_name, extra = _run_dim(
                            dim, frame_paths, meta, scorers,
                            reference_image_dir=reference_image_dir,
                        )
                    except Exception as e:
                        log.warning("worldscore: %s/%s failed (%s)", vs.prompt_id, dim, e)
                        continue
                    if dim == "subjective_quality":
                        # Emit one row per sub-metric (clip_aesthetic, clip_iqa+).
                        for sub_metric, sub_score in extra.items():
                            results.append(RawResult(
                                benchmark=self.name, model=vs.model_name,
                                dimension="subjective_quality", prompt_id=vs.prompt_id,
                                score=sub_score, scorer=sub_metric,
                                video_path=str(vs.path), meta={"metric": sub_metric, "split": split},
                            ))
                    elif score is None:
                        continue
                    else:
                        results.append(RawResult(
                            benchmark=self.name, model=vs.model_name,
                            dimension=dim, prompt_id=vs.prompt_id,
                            score=score, scorer=scorer_name,
                            video_path=str(vs.path),
                            meta={**extra, "split": split},
                        ))
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        return results

    # ---- aggregate ----------------------------------------------------------
    def aggregate(
        self,
        raw: list[RawResult],
        aggregator: str = "worldscore_upstream",
        **kwargs: Any,
    ) -> Summary:
        """Apply upstream normalization (``aspect_info``) and compute the two
        WorldScore headlines exactly as ``run_evaluate.py`` does."""
        from videvalkit.benchmarks.worldscore.aggregator import compute_worldscore

        rows = [{
            "prompt_id": r.prompt_id, "dimension": r.dimension,
            "score": r.score, "metric": (r.meta or {}).get("metric"),
            "split": (r.meta or {}).get("split"),
        } for r in raw]
        out = compute_worldscore(rows)

        per_dim_mean = {d: info["norm_mean_x100"] for d, info in out["per_dim"].items()
                        if info["norm_mean_x100"] is not None}
        overall = out["headlines"].get("WorldScore-Dynamic", 0.0) or 0.0
        model = raw[0].model if raw else ""
        return Summary(
            benchmark=self.name, model=model,
            per_dimension=per_dim_mean,
            overall=overall,
            n_videos=len({(r.model, r.prompt_id) for r in raw}),
            n_prompts=len({r.prompt_id for r in raw}),
            aggregator=aggregator,
            meta={
                "method": out["method"],
                "headlines": out["headlines"],
                "frame_count": WORLDSCORE_FRAME_DEFAULT,
                "static_dims": list(WORLDSCORE_STATIC_DIMS),
                "dynamic_dims": list(WORLDSCORE_DYNAMIC_DIMS),
            },
        )


def _run_dim(dim, frame_paths, prompt_meta, scorers, *, reference_image_dir=None):
    """Per-dim dispatch: returns ``(score, scorer_name, extra_meta_dict)``.

    For subjective_quality returns ``(score=None, scorer="", {sub_metric: score})``;
    the caller emits one row per sub-metric.
    """
    if dim == "content_alignment":
        text = prompt_meta.get("text", "")
        return scorers[dim].score(frame_paths, text), "clip_score_vit_b16", {}
    if dim == "object_control":
        # Upstream drops the first comma-separated chunk, then matches the rest.
        # Use ``content_list[i]`` if available (preferred), else fall back.
        content = (prompt_meta.get("content_list") or [None])[prompt_meta.get("step", 0) or 0]
        if not content:
            objs = prompt_meta.get("objects") or []
            content = "scene, " + ", ".join(objs) if objs else prompt_meta.get("text", "")
        return scorers[dim].score(frame_paths, content), "grounding_dino", {"text_prompt": content}
    if dim == "photometric_consistency":
        return scorers[dim].score(frame_paths), "searaft_aepe", {}
    if dim == "motion_magnitude":
        return scorers[dim].score(frame_paths), "searaft_median_flow", {}
    if dim == "style_consistency":
        if not reference_image_dir:
            raise RuntimeError("style_consistency requires reference_image_dir")
        parent = prompt_meta.get("parent_entry") or prompt_meta.get("entry_id") \
            or prompt_meta.get("prompt_id", "").rsplit("_s", 1)[0]
        ref = Path(reference_image_dir) / f"{parent}.png"
        if not ref.exists():
            raise FileNotFoundError(f"no reference image: {ref}")
        return scorers[dim].score(frame_paths, str(ref)), "gram_matrix_vgg19", {"ref": str(ref)}
    if dim == "3d_consistency":
        return scorers[dim].score(frame_paths), "droid_reproj", {}
    if dim == "camera_control":
        cameras_gt = prompt_meta.get("cameras_gt")
        if cameras_gt is None:
            raise RuntimeError(
                "camera_control needs ground-truth cameras (`cameras_gt`) in the "
                "prompt meta. They are derived from the entry's `camera_path` by "
                "the prep runner — run it first:\n"
                "    python -m videvalkit.benchmarks.worldscore.runners.camera_error\n"
                "(it concatenates the sub-prompt clips and builds GT cameras via "
                "CameraGen). camera_control is the one worldscore dim that needs "
                "this prep step; the other 9 run directly."
            )
        R, T = scorers[dim].score(frame_paths, cameras_gt)
        return [R, T], "droid_camera_err", {"R_err_deg": R, "T_err": T}
    if dim == "motion_smoothness":
        triple = scorers[dim].score(frame_paths)
        return list(triple), "vfimamba_ssim_lpips_mse", {}
    if dim == "motion_accuracy":
        objects = prompt_meta.get("objects") or []
        out = scorers[dim].score(frame_paths, objects)
        return out["score"], "searaft_sam2_motion_align", out
    if dim == "subjective_quality":
        aes = scorers["subjective_quality.clip_aesthetic"].score(frame_paths)
        iqa = scorers["subjective_quality.clip_iqa+"].score(frame_paths)
        return None, "", {"clip_aesthetic": aes, "clip_iqa+": iqa}
    raise ValueError(f"unhandled dim: {dim}")
