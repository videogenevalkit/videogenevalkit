"""Subprocess wrappers around the upstream T2V-CompBench V2 scripts.

Each wrapper:
  1. Stages our toolkit videos into the upstream-expected layout
     (``<work>/videos/<t2v_model>/0001.mp4, 0002.mp4, ...`` — filenames map
     to JSON-index by ``int(name[:4]) - 1``).
  2. Writes a synthetic ``meta_data/<dim>.json`` whose entries match the
     upstream schema, populated from each ``PromptItem.meta`` (which can
     be filled either by LLM extraction or by ``extract.py`` upfront).
  3. Subprocesses into the upstream script using the **benchmark env**
     Python (which has groundingdino / segment_anything / depth_anything /
     dot all installed).
  4. Parses the upstream-written per-video CSV
     (``<output_path>/<t2v_model>_<dim>_video.csv``) and converts each row
     into a ``RawResult``.

All checkpoints are stored under ``$VIDEVALKIT_T2VCOMPBENCH_CKPTS`` (default:
``~/.cache/videvalkit/checkpoints/t2vcompbench/``, populated by
``videvalkit fetch-checkpoints --bench t2vcompbench``) and the upstream code
itself under ``$VIDEVALKIT_T2VCOMPBENCH_REPO`` (default:
``~/.cache/videvalkit/upstream/T2V-CompBench/``, populated by
``videvalkit fetch-upstream --bench t2vcompbench`` which does a
``git clone https://github.com/KaiyueSun98/T2V-CompBench --branch V2``).

The subprocess Python is ``$VIDEVALKIT_T2VCOMPBENCH_PY`` (default: ``sys.executable``).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from videvalkit.core.types import PromptItem, RawResult, VideoSpec

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Paths — env-var configurable; sensible defaults under ~/.cache/videvalkit/
# --------------------------------------------------------------------------- #

_CACHE_HOME = Path(os.environ.get("VIDEVALKIT_CACHE_HOME",
                                  Path.home() / ".cache" / "videvalkit"))

DEFAULT_REPO = Path(
    os.environ.get(
        "VIDEVALKIT_T2VCOMPBENCH_REPO",
        str(_CACHE_HOME / "upstream" / "T2V-CompBench"),
    )
)
DEFAULT_CKPTS = Path(
    os.environ.get(
        "VIDEVALKIT_T2VCOMPBENCH_CKPTS",
        str(_CACHE_HOME / "checkpoints" / "t2vcompbench"),
    )
)
DEFAULT_UPSTREAM_PY = Path(
    os.environ.get(
        "VIDEVALKIT_T2VCOMPBENCH_PY",
        sys.executable,
    )
)


def _check_setup() -> None:
    """Confirm the four files we always need are in place; cheap and fast."""
    must = {
        "upstream-repo": DEFAULT_REPO,
        "checkpoint-dir": DEFAULT_CKPTS,
        "upstream-python": DEFAULT_UPSTREAM_PY,
        "gd-ckpt": DEFAULT_CKPTS / "groundingdino_swint_ogc.pth",
    }
    missing = {k: v for k, v in must.items() if not v.exists()}
    if missing:
        raise RuntimeError(
            "t2vcompbench upstream not set up; missing: "
            + ", ".join(f"{k}={v}" for k, v in missing.items())
        )


# --------------------------------------------------------------------------- #
# Workspace staging
# --------------------------------------------------------------------------- #

def _stage_videos(
    work: Path, model_tag: str, videos: list[VideoSpec],
    prompts: list[PromptItem],
) -> dict[str, int]:
    """Stage videos into ``<work>/videos/<model_tag>/`` with upstream's
    ``NNNN.mp4`` naming. Returns a mapping ``prompt_id -> 1-based index``."""
    video_dir = work / "videos" / model_tag
    video_dir.mkdir(parents=True, exist_ok=True)
    # Sort prompts deterministically by prompt_id so JSON index is stable.
    sorted_prompts = sorted(prompts, key=lambda p: p.prompt_id)
    prompt_id_to_idx: dict[str, int] = {}
    for i, p in enumerate(sorted_prompts, start=1):
        # Find the matching video (any sample_index — we take sample 0)
        match = next(
            (v for v in videos if v.prompt_id == p.prompt_id), None,
        )
        if match is None:
            log.warning("t2vcompbench/upstream: no video for prompt %s", p.prompt_id)
            continue
        dst = video_dir / f"{i:04d}.mp4"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(match.path.resolve())
        prompt_id_to_idx[p.prompt_id] = i
    return prompt_id_to_idx


def _build_env(extra_pythonpath: list[str] | None = None) -> dict[str, str]:
    """Build an env for the upstream subprocess.

    Two non-obvious things:
      1. Drop any inherited ``PYTHONPATH`` (caller's py3.13 site-packages
         will break py3.10 imports).
      2. **Prepend** the bench env's ``site-packages`` to ``PYTHONPATH``.
         Without this, ``Grounded-Segment-Anything/`` (on the path so the
         spatial script's ``from GroundingDINO...`` works) shadows the
         installed ``segment_anything`` package with an empty
         namespace-package directory.
    """
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    bench_site = (DEFAULT_UPSTREAM_PY.parent.parent
                  / "lib" / "python3.10" / "site-packages")
    pp = [
        str(bench_site),                # <-- prepended so site-packages wins
        str(DEFAULT_REPO / "Grounded-Segment-Anything/GroundingDINO"),
        str(DEFAULT_REPO / "Grounded-Segment-Anything"),
        *(extra_pythonpath or []),
    ]
    env["PYTHONPATH"] = ":".join(p for p in pp if p)
    # Force direct huggingface.co — hf-mirror in our shell is currently broken
    # and breaks DepthAnything.from_pretrained() and similar HF Hub fetches.
    env["HF_ENDPOINT"] = "https://huggingface.co"
    return env


# --------------------------------------------------------------------------- #
# generative_numeracy
# --------------------------------------------------------------------------- #

def run_upstream_numeracy(
    videos: list[VideoSpec],
    prompts: list[PromptItem],
    work_root: Path,
    *,
    t2v_model_tag: str = "videvalkit_run",
) -> list[RawResult]:
    """Run upstream ``compbench_eval_numeracy.py``.

    Each ``PromptItem.meta`` MUST carry the upstream schema for this dim,
    either as top-level keys or under ``meta["t2vcompbench"]["numeracy"]``::

        {"objects": "girl,boy", "numbers": "1,1"}

    (Use ``extract.extract_numeracy`` to fill it from raw text when needed.)
    Returns one ``RawResult`` per (model, prompt_id) with the upstream's
    per-video score (mean of 16 frame scores) and the per-frame array in
    ``meta["per_frame_scores"]``.
    """
    _check_setup()
    if not videos or not prompts:
        return []

    work = work_root / "t2vcompbench_numeracy"
    work.mkdir(parents=True, exist_ok=True)

    # 1. Stage videos
    pid2idx = _stage_videos(work, t2v_model_tag, videos, prompts)
    if not pid2idx:
        log.warning("t2vcompbench/upstream/numeracy: no usable videos")
        return []

    # 2. Build meta JSON in the index order we used
    sorted_pids = sorted(pid2idx, key=lambda pid: pid2idx[pid])
    meta_entries: list[dict[str, Any]] = []
    for pid in sorted_pids:
        p = next(p for p in prompts if p.prompt_id == pid)
        m = _take_dim_meta(p, "numeracy")
        objs = m.get("objects") if isinstance(m, dict) else None
        nums = m.get("numbers") if isinstance(m, dict) else None
        if not objs or not nums:
            log.warning("t2vcompbench/upstream/numeracy: prompt %s missing "
                        "meta.numeracy.objects/numbers; using prompt text raw",
                        pid)
            objs, nums = "", ""
        meta_entries.append({
            "prompt":  p.text,
            "objects": objs,
            "numbers": nums,
        })
    meta_file = work / "meta_numeracy.json"
    meta_file.write_text(json.dumps(meta_entries, indent=2))

    output_path = work / "csv_numeracy"
    output_path.mkdir(parents=True, exist_ok=True)
    output_dir = work / "output_numeracy"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Subprocess
    video_dir = work / "videos" / t2v_model_tag
    script = (DEFAULT_REPO
              / "Grounded-Segment-Anything/GroundingDINO/demo/compbench_eval_numeracy.py")
    config = (DEFAULT_REPO
              / "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    ckpt = DEFAULT_CKPTS / "groundingdino_swint_ogc.pth"

    cmd = [
        str(DEFAULT_UPSTREAM_PY), str(script),
        "--config_file", str(config),
        "--checkpoint_path", str(ckpt),
        "--video-path", str(video_dir),
        "--read-prompt-file", str(meta_file),
        "--output-path", str(output_path),
        "--output_dir", str(output_dir),
        "--t2v-model", t2v_model_tag,
    ]
    log.info("t2vcompbench/upstream/numeracy: running %s", " ".join(cmd[:3]) + " ...")
    proc = subprocess.run(
        cmd, env=_build_env(), cwd=str(DEFAULT_REPO),
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        log.error("upstream numeracy failed (rc=%d): %s",
                  proc.returncode, proc.stderr[-1000:])
        raise RuntimeError("upstream numeracy subprocess failed")

    # 4. Parse per-video CSV
    video_csv = output_path / f"{t2v_model_tag}_numeracy_video.csv"
    return _parse_per_video_csv(
        video_csv, dim="generative_numeracy",
        model_name=videos[0].model_name if videos else t2v_model_tag,
        prompts=prompts, pid2idx=pid2idx,
    )


# --------------------------------------------------------------------------- #
# spatial_relationships
# --------------------------------------------------------------------------- #

def run_upstream_spatial(
    videos: list[VideoSpec],
    prompts: list[PromptItem],
    work_root: Path,
    *,
    t2v_model_tag: str = "videvalkit_run",
) -> list[RawResult]:
    """Run upstream ``compbench_eval_spatial_relationships.py``.

    The script invokes ``run_depth`` internally to produce 3D depth images,
    then performs 2D and 3D scoring. Outputs include 2D and 3D per-video
    CSVs plus a combined ``<t2v_model>_spatial_score.csv`` whose last row
    holds the headline number.

    Each ``PromptItem.meta`` MUST carry the upstream schema for this dim::

        {"object_1": "apple", "spatial": "left", "object_2": "pear"}

    (Use ``extract.extract_spatial`` to fill it from raw text when needed.)
    """
    _check_setup()
    if not videos or not prompts:
        return []

    work = work_root / "t2vcompbench_spatial"
    work.mkdir(parents=True, exist_ok=True)

    # 1. Stage videos with NNNN.mp4 naming, matching the JSON entry order.
    pid2idx = _stage_videos(work, t2v_model_tag, videos, prompts)
    if not pid2idx:
        return []

    # 2. Build the meta JSON
    sorted_pids = sorted(pid2idx, key=lambda pid: pid2idx[pid])
    meta_entries: list[dict[str, Any]] = []
    for pid in sorted_pids:
        p = next(p for p in prompts if p.prompt_id == pid)
        m = _take_dim_meta(p, "spatial") or {}
        meta_entries.append({
            "prompt":   p.text,
            "object_1": m.get("object_1") or "",
            "spatial":  m.get("spatial") or "",
            "object_2": m.get("object_2") or "",
        })
    meta_file = work / "meta_spatial.json"
    meta_file.write_text(json.dumps(meta_entries, indent=2))

    output_path = work / "csv_spatial"
    output_path.mkdir(parents=True, exist_ok=True)
    output_dir = work / "output_spatial"
    output_dir.mkdir(parents=True, exist_ok=True)
    depth_folder = work / "output_spatial_depth"
    depth_folder.mkdir(parents=True, exist_ok=True)

    video_dir = work / "videos" / t2v_model_tag
    script = DEFAULT_REPO / "Grounded-Segment-Anything/compbench_eval_spatial_relationships.py"
    gd_config = DEFAULT_REPO / "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    gd_ckpt = DEFAULT_CKPTS / "groundingdino_swint_ogc.pth"
    sam_ckpt = DEFAULT_CKPTS / "sam_vit_h_4b8939.pth"

    cmd = [
        str(DEFAULT_UPSTREAM_PY), str(script),
        "--config", str(gd_config),
        "--grounded_checkpoint", str(gd_ckpt),
        "--sam_checkpoint", str(sam_ckpt),
        "--video-path", str(video_dir),
        "--depth_folder", str(depth_folder),
        "--read-prompt-file", str(meta_file),
        "--output-path", str(output_path),
        "--output_dir", str(output_dir),
        "--t2v-model", t2v_model_tag,
    ]
    extra_pp = [str(DEFAULT_REPO / "Depth-Anything")]
    log.info("t2vcompbench/upstream/spatial: running ...")
    proc = subprocess.run(
        cmd, env=_build_env(extra_pp), cwd=str(DEFAULT_REPO),
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        log.error("upstream spatial failed (rc=%d): %s",
                  proc.returncode, proc.stderr[-1500:])
        raise RuntimeError("upstream spatial subprocess failed")

    # 3. Parse the combined spatial score CSV (per-video headline number).
    # The upstream writes `<t2v_model>_spatial_score.csv` with rows
    # describing per-video scores plus aggregate stats. The 2D and 3D
    # per-video CSVs (`*_2dvideo.csv`, `*_3dvideo.csv`) give the per-row
    # scores we need.
    raw_results: list[RawResult] = []
    raw_results.extend(_parse_2dor3d_csv(
        output_path / f"{t2v_model_tag}_2dvideo.csv",
        dim_name="spatial_relationships",
        sub_label="2d", model_name=videos[0].model_name,
        prompts=prompts, pid2idx=pid2idx,
    ))
    raw_results.extend(_parse_2dor3d_csv(
        output_path / f"{t2v_model_tag}_3dvideo.csv",
        dim_name="spatial_relationships",
        sub_label="3d", model_name=videos[0].model_name,
        prompts=prompts, pid2idx=pid2idx,
    ))
    return raw_results


def _parse_2dor3d_csv(
    csv_path: Path, *, dim_name: str, sub_label: str, model_name: str,
    prompts: list[PromptItem], pid2idx: dict[str, int],
) -> list[RawResult]:
    """Parse spatial's per-video 2D or 3D CSV.

    Upstream CSV shape (3 cols typical): id, Score_frame_*, Score_*. The
    last row is a summary 'score: <avg>' that we skip.
    """
    if not csv_path.exists():
        log.warning("upstream %s/%s csv missing: %s", dim_name, sub_label, csv_path)
        return []
    idx2pid = {idx: pid for pid, idx in pid2idx.items()}
    out: list[RawResult] = []
    with csv_path.open() as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return []
        for row in reader:
            if not row or not row[0].strip().isdigit():
                continue
            idx = int(row[0])
            pid = idx2pid.get(idx)
            if pid is None:
                continue
            per_frame_str = row[1] if len(row) > 1 else "[]"
            score_str = row[2] if len(row) > 2 else "0.0"
            try:
                per_frame = json.loads(per_frame_str)
            except Exception:
                per_frame = []
            try:
                score = float(score_str)
            except Exception:
                continue
            out.append(RawResult(
                benchmark="t2vcompbench",
                model=model_name,
                dimension=dim_name,
                prompt_id=pid,
                score=score,
                scorer=f"upstream:compbench_eval_spatial_relationships.py:{sub_label}",
                video_path=None,
                meta={"per_frame_scores": per_frame, "upstream_index": idx,
                      "sub_label": sub_label},
            ))
    return out


# --------------------------------------------------------------------------- #
# motion_binding
# --------------------------------------------------------------------------- #

def run_upstream_motion_binding(
    videos: list[VideoSpec],
    prompts: list[PromptItem],
    work_root: Path,
    *,
    t2v_model_tag: str = "videvalkit_run",
    total_frame: int = 16,
    fps: int = 8,
) -> list[RawResult]:
    """Run upstream motion_binding (two scripts in sequence).

    Step 1: ``Grounded-Segment-Anything/compbench_motion_binding_seg.py``
    extracts 1st-frame fg/bg SAM masks per (video, object).

    Step 2: ``dot/compbench_eval_motion_binding.py`` does dense tracking
    (DOT) on the standard-fps video and combines fore/back point motion
    against the expected directions ``d_1`` / ``d_2`` from the meta JSON.
    Outputs ``csv_motion_binding/<t2v_model>_motion_score.csv`` with the
    headline numbers.

    ``PromptItem.meta`` MUST carry::

        {"object_1": "fish", "d_1": "right",
         "object_2": "<noun or null>", "d_2": "<dir or null>"}
    """
    _check_setup()
    if not videos or not prompts:
        return []

    work = work_root / "t2vcompbench_motion"
    work.mkdir(parents=True, exist_ok=True)

    # 1. Stage videos
    pid2idx = _stage_videos(work, t2v_model_tag, videos, prompts)
    if not pid2idx:
        return []

    # 2. Meta JSON
    sorted_pids = sorted(pid2idx, key=lambda pid: pid2idx[pid])
    meta_entries: list[dict[str, Any]] = []
    for pid in sorted_pids:
        p = next(p for p in prompts if p.prompt_id == pid)
        m = _take_dim_meta(p, "motion") or {}
        # Upstream meta_data/motion_binding.json uses "" (empty string)
        # for missing object_2 / d_2, not null. Match exactly.
        meta_entries.append({
            "prompt":   p.text,
            "object_1": m.get("object_1") or "",
            "d_1":      m.get("d_1") or "",
            "object_2": m.get("object_2") or "",
            "d_2":      m.get("d_2") or "",
        })
    meta_file = work / "meta_motion.json"
    meta_file.write_text(json.dumps(meta_entries, indent=2))

    video_dir = work / "videos" / t2v_model_tag
    seg_out = work / "output_motion_binding_seg"
    seg_out.mkdir(parents=True, exist_ok=True)
    csv_out = work / "csv_motion_binding"
    csv_out.mkdir(parents=True, exist_ok=True)
    eval_out = work / "output_motion_binding"
    eval_out.mkdir(parents=True, exist_ok=True)

    gd_config = DEFAULT_REPO / "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    gd_ckpt = DEFAULT_CKPTS / "groundingdino_swint_ogc.pth"
    sam_ckpt = DEFAULT_CKPTS / "sam_vit_h_4b8939.pth"

    # ----- Step 1: SAM segmentation of 1st frame --------------------- #
    seg_script = DEFAULT_REPO / "Grounded-Segment-Anything/compbench_motion_binding_seg.py"
    seg_cmd = [
        str(DEFAULT_UPSTREAM_PY), str(seg_script),
        "--config", str(gd_config),
        "--grounded_checkpoint", str(gd_ckpt),
        "--sam_checkpoint", str(sam_ckpt),
        "--video-path", str(video_dir),
        "--t2v-model", t2v_model_tag,
        "--total_frame", str(total_frame),
        "--fps", str(fps),
        "--read-prompt-file", str(meta_file),
        "--output_dir", str(seg_out),
    ]
    log.info("t2vcompbench/upstream/motion_binding step 1: SAM masks ...")
    proc = subprocess.run(
        seg_cmd, env=_build_env(), cwd=str(DEFAULT_REPO),
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        log.error("motion_binding seg failed (rc=%d): %s",
                  proc.returncode, proc.stderr[-1500:])
        raise RuntimeError("upstream motion_binding seg failed")

    # ----- Step 2: DOT tracking + scoring --------------------------- #
    dot_dir = DEFAULT_REPO / "dot"
    dot_script = dot_dir / "compbench_eval_motion_binding.py"
    # Seg step writes the 8-fps "standard" video under
    # <video_dir>/../video_standard/<t2v_model>/. That is what step 2 wants.
    standard_video_dir = video_dir.parent / "video_standard" / t2v_model_tag
    # DOT config + ckpts
    dot_estimator = DEFAULT_CKPTS / "cvo_raft_patch_8.pth"
    dot_refiner = DEFAULT_CKPTS / "movi_f_raft_patch_4_alpha.pth"
    dot_tracker = DEFAULT_CKPTS / "movi_f_cotracker2_patch_4_wind_8.pth"
    # DOT supports being called with these as paths.
    dot_cmd = [
        str(DEFAULT_UPSTREAM_PY), str(dot_script),
        "--video-path", str(standard_video_dir),
        "--mask_folder", str(seg_out),
        "--read-prompt-file", str(meta_file),
        "--t2v-model", t2v_model_tag,
        "--output-path", str(csv_out),
        "--output_dir", str(eval_out),
        "--estimator_path", str(dot_estimator),
        "--refiner_path", str(dot_refiner),
        "--tracker_path", str(dot_tracker),
    ]
    log.info("t2vcompbench/upstream/motion_binding step 2: DOT tracking ...")
    extra_pp = [str(dot_dir)]
    proc = subprocess.run(
        dot_cmd, env=_build_env(extra_pp), cwd=str(dot_dir),
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        log.error("motion_binding dot failed (rc=%d): %s",
                  proc.returncode, proc.stderr[-1500:])
        raise RuntimeError("upstream motion_binding dot failed")

    # 3. Parse the per-video score CSV
    score_csv = csv_out / f"{t2v_model_tag}_motion_score.csv"
    return _parse_motion_score_csv(
        score_csv, model_name=videos[0].model_name,
        prompts=prompts, pid2idx=pid2idx,
    )


def _parse_motion_score_csv(
    csv_path: Path, *, model_name: str,
    prompts: list[PromptItem], pid2idx: dict[str, int],
) -> list[RawResult]:
    """Parse upstream's motion_score CSV. Shape: each row is one video score."""
    if not csv_path.exists():
        log.error("upstream motion_binding: per-video CSV missing: %s", csv_path)
        return []
    idx2pid = {idx: pid for pid, idx in pid2idx.items()}
    out: list[RawResult] = []
    with csv_path.open() as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or not row[0].strip().isdigit():
                continue
            idx = int(row[0])
            pid = idx2pid.get(idx)
            if pid is None:
                continue
            # last column is the per-video final score; convention from upstream
            try:
                score = float(row[-1])
            except Exception:
                continue
            out.append(RawResult(
                benchmark="t2vcompbench",
                model=model_name,
                dimension="motion_binding",
                prompt_id=pid,
                score=score,
                scorer="upstream:compbench_eval_motion_binding.py",
                video_path=None,
                meta={"upstream_index": idx,
                      "raw_csv_row": row},
            ))
    return out


# --------------------------------------------------------------------------- #
# MLLM dims (4) via upstream LLaVA-1.6-34b
# --------------------------------------------------------------------------- #

DEFAULT_LLAVA_MODEL_PATH = os.environ.get(
    "VIDEVALKIT_T2VCOMPBENCH_LLAVA",
    "liuhaotian/llava-v1.6-34b",
)
"""LLaVA model passed to upstream's --model-path. Either a HF id (lets
huggingface_hub resolve from cache) or an absolute path. We default to
the HF id and rely on the model being already in
``~/.cache/huggingface/hub/models--liuhaotian--llava-v1.6-34b/``."""

# Per-dim configuration. Each entry specifies the upstream script path
# under ``LLaVA/llava/eval/``, the meta_data JSON filename, and the
# expected per-video CSV name.
_MLLM_DIM_CONFIG = {
    "consistent_attribute": {
        "script":     "LLaVA/llava/eval/compbench_eval_consistent_attr.py",
        "meta_keys":  ["phrases"],
        "csv_name":   "{model}_consistent_attr_score.csv",
        "log_tag":    "consistent_attr",
    },
    "dynamic_attribute": {
        "script":     "LLaVA/llava/eval/compbench_eval_dynamic_attr.py",
        "meta_keys":  ["state 0", "state 1", "tag"],
        "csv_name":   "{model}_dynamic_attr_score.csv",
        "log_tag":    "dynamic_attr",
    },
    "action_binding": {
        "script":     "LLaVA/llava/eval/compbench_eval_action_binding.py",
        "meta_keys":  ["phrase_0", "phrase_1"],
        "csv_name":   "{model}_action_binding_score.csv",
        "log_tag":    "action_binding",
    },
    "object_interactions": {
        "script":     "LLaVA/llava/eval/compbench_eval_interaction.py",
        "meta_keys":  [],   # only "prompt" used
        "csv_name":   "{model}_object_interactions_score.csv",
        "log_tag":    "interaction",
    },
}

T2VCOMPBENCH_UPSTREAM_MLLM_DIMS = tuple(_MLLM_DIM_CONFIG.keys())


def run_upstream_mllm(
    dim: str,
    videos: list[VideoSpec],
    prompts: list[PromptItem],
    work_root: Path,
    *,
    t2v_model_tag: str = "videvalkit_run",
    llava_model_path: str | None = None,
) -> list[RawResult]:
    """Run one of the 4 MLLM dims via upstream LLaVA-1.6-34b.

    Each entry in ``PromptItem.meta`` should carry the keys upstream's
    ``meta_data/<dim>.json`` expects (see ``_MLLM_DIM_CONFIG[dim]["meta_keys"]``).

    Use ``extract.extract_<dim>`` or load directly from
    ``meta_data/<dim>.json`` when reproducing the paper.

    Returns one RawResult per video with the per-video Score column.
    """
    _check_setup()
    if dim not in _MLLM_DIM_CONFIG:
        raise KeyError(f"unknown MLLM dim {dim!r}; known: {list(_MLLM_DIM_CONFIG)}")
    cfg = _MLLM_DIM_CONFIG[dim]
    if not videos or not prompts:
        return []

    work = work_root / f"t2vcompbench_{cfg['log_tag']}"
    work.mkdir(parents=True, exist_ok=True)

    # 1. Stage videos
    pid2idx = _stage_videos(work, t2v_model_tag, videos, prompts)
    if not pid2idx:
        return []

    # 2. Build meta JSON (upstream schema differs per dim)
    sorted_pids = sorted(pid2idx, key=lambda pid: pid2idx[pid])
    meta_entries: list[dict[str, Any]] = []
    for pid in sorted_pids:
        p = next(p for p in prompts if p.prompt_id == pid)
        m = p.meta if isinstance(p.meta, dict) else {}
        # Match upstream's exact key naming for the schema fields.
        entry: dict[str, Any] = {"prompt": p.text}
        for k in cfg["meta_keys"]:
            entry[k] = m.get(k) if k in m else m.get(k.replace(" ", "_"), "")
        meta_entries.append(entry)
    meta_file = work / f"meta_{cfg['log_tag']}.json"
    meta_file.write_text(json.dumps(meta_entries, indent=2, ensure_ascii=False))

    output_path = work / f"csv_{cfg['log_tag']}"
    output_path.mkdir(parents=True, exist_ok=True)

    video_dir = work / "videos" / t2v_model_tag
    script = DEFAULT_REPO / cfg["script"]
    llava_path = llava_model_path or DEFAULT_LLAVA_MODEL_PATH

    cmd = [
        str(DEFAULT_UPSTREAM_PY), str(script),
        "--model-path", llava_path,
        "--video-path", str(video_dir),
        "--read-prompt-file", str(meta_file),
        "--output-path", str(output_path),
        "--t2v-model", t2v_model_tag,
    ]
    # Need LLaVA's repo dir on PYTHONPATH so `from llava... import ...` works.
    extra_pp = [str(DEFAULT_REPO / "LLaVA")]
    log.info("t2vcompbench/upstream/%s: running ...", cfg["log_tag"])
    proc = subprocess.run(
        cmd, env=_build_env(extra_pp), cwd=str(DEFAULT_REPO / "LLaVA"),
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        log.error("upstream %s failed (rc=%d): %s",
                  cfg["log_tag"], proc.returncode, proc.stderr[-1500:])
        raise RuntimeError(f"upstream {cfg['log_tag']} subprocess failed")

    # 3. Parse per-video score CSV. Headline column is "Score" (last column).
    csv_path = output_path / cfg["csv_name"].format(model=t2v_model_tag)
    return _parse_mllm_csv(
        csv_path, dim=dim, model_name=videos[0].model_name,
        prompts=prompts, pid2idx=pid2idx,
    )


def _parse_mllm_csv(
    csv_path: Path, *, dim: str, model_name: str,
    prompts: list[PromptItem], pid2idx: dict[str, int],
) -> list[RawResult]:
    """Parse upstream's MLLM per-video CSV.

    Columns: ``name, prompt, seed0_answer..., seed0_score, ..., seed_score, Score``.
    We only need ``name`` (e.g. ``0001.png``) and ``Score`` (last col).
    """
    if not csv_path.exists():
        log.error("upstream %s: per-video CSV missing: %s", dim, csv_path)
        return []
    idx2pid = {idx: pid for pid, idx in pid2idx.items()}
    out: list[RawResult] = []
    with csv_path.open() as f:
        reader = csv.reader(f)
        header = None
        for row in reader:
            if header is None:
                header = row
                continue
            if not row or not row[0].strip():
                continue
            # Skip the trailing "score: <avg>" row
            if row[0].strip().lower().startswith("score"):
                continue
            # name like "0001.png" or "0001.mp4"
            name = row[0].strip()
            stem = name.split(".")[0]
            try:
                idx = int(stem)
            except ValueError:
                continue
            pid = idx2pid.get(idx)
            if pid is None:
                continue
            try:
                score = float(row[-1])
            except (ValueError, IndexError):
                continue
            out.append(RawResult(
                benchmark="t2vcompbench",
                model=model_name,
                dimension=dim,
                prompt_id=pid,
                score=score,
                scorer=f"upstream:llava-v1.6-34b",
                video_path=None,
                meta={"upstream_index": idx, "raw_csv_row": row},
            ))
    return out


# --------------------------------------------------------------------------- #
# Per-video CSV parsing (shared)
# --------------------------------------------------------------------------- #

def _take_dim_meta(p: PromptItem, dim_key: str) -> dict[str, Any] | None:
    """Look up upstream meta for a dim, either flat in p.meta or under p.meta['t2vcompbench'][dim_key]."""
    if not isinstance(p.meta, dict):
        return None
    inner = p.meta.get("t2vcompbench")
    if isinstance(inner, dict) and dim_key in inner:
        return inner[dim_key]
    return p.meta


def _parse_per_video_csv(
    csv_path: Path, *, dim: str, model_name: str,
    prompts: list[PromptItem], pid2idx: dict[str, int],
) -> list[RawResult]:
    """Read upstream's per-video CSV (id, Score_frame_1, Score_1) and emit
    one RawResult per row."""
    if not csv_path.exists():
        log.error("upstream %s: per-video CSV missing at %s", dim, csv_path)
        return []
    idx2pid = {idx: pid for pid, idx in pid2idx.items()}
    out: list[RawResult] = []
    with csv_path.open() as f:
        reader = csv.reader(f)
        header = next(reader)  # id, Score_frame_1, Score_1
        if header[:1] != ["id"]:
            log.warning("upstream %s: unexpected CSV header %s", dim, header)
        for row in reader:
            if not row or not row[0].strip().isdigit():
                continue  # skip the trailing "score: <avg>" summary row
            idx = int(row[0])
            pid = idx2pid.get(idx)
            if pid is None:
                continue
            per_frame_str = row[1] if len(row) > 1 else "[]"
            score_str = row[2] if len(row) > 2 else "0.0"
            try:
                per_frame = json.loads(per_frame_str)
            except Exception:
                per_frame = []
            score = float(score_str)
            out.append(RawResult(
                benchmark="t2vcompbench",
                model=model_name,
                dimension=dim,
                prompt_id=pid,
                score=score,
                scorer="upstream:compbench_eval_numeracy.py"
                if dim == "generative_numeracy"
                else f"upstream:compbench_eval_{dim}.py",
                video_path=None,
                meta={"per_frame_scores": per_frame, "upstream_index": idx},
            ))
    return out
