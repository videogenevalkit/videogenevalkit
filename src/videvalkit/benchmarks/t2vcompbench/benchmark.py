"""T2V-CompBench V2 adapter — 7 compositional dims, MLLM + detector + tracker.

Upstream: https://github.com/KaiyueSun98/T2V-CompBench/tree/V2

Three scorer families per the upstream README:
  * MLLM (LLaVA / Grid-LLaVA / D-LLaVA): consistent_attribute, dynamic_attribute,
    action_binding, object_interactions
  * Detection (GroundingDINO + Depth Anything + GroundingSAM):
    spatial_relationships, generative_numeracy
  * Tracking (GroundingSAM + DOT): motion_binding

Inputs: video frames sampled to either a 6-frame image grid (Grid-LLaVA) or a
16-frame sequence (~8 fps for tracking).

This adapter is a thin shim. `evaluate()` calls the upstream scripts in
T2V-CompBench/eval_<dim>.py; we route MLLM calls through the toolkit's
VLM judge when `judge` is provided so the same LLaVA-Video endpoint can
serve both VBench-2.0 and T2V-CompBench.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from videvalkit.core.benchmark import BaseBenchmark
from videvalkit.core.layout import WorkspaceLayout
from videvalkit.core.types import PromptItem, RawResult, Summary, VideoSpec

log = logging.getLogger(__name__)


T2VCOMPBENCH_DIMENSIONS = [
    "consistent_attribute",
    "dynamic_attribute",
    "action_binding",
    "object_interactions",
    "spatial_relationships",
    "generative_numeracy",
    "motion_binding",
]

T2VCOMPBENCH_SCORER_KIND = {
    "consistent_attribute":  "mllm",
    "dynamic_attribute":     "mllm",
    "action_binding":        "mllm",
    "object_interactions":   "mllm",
    "spatial_relationships": "detection",
    "generative_numeracy":   "detection",
    "motion_binding":        "tracking",
}

# Per-dim short definitions (Sun et al., 2024, Sec. 3).
T2VCOMPBENCH_DEFINITIONS = {
    "consistent_attribute":  "Static attributes (color, shape, texture) bound to the right subject and stable across frames.",
    "dynamic_attribute":     "Attribute that evolves over time (e.g. 'leaf gradually turning yellow').",
    "action_binding":        "Specific action bound to specific subject in a multi-subject scene.",
    "object_interactions":   "Physical or causal interaction between subjects executed correctly.",
    "spatial_relationships": "2D / 3D positional relations (left/right/above/below/front/behind) hold.",
    "generative_numeracy":   "Generated count of target objects matches the prompt.",
    "motion_binding":        "Subject moves as instructed, distinct from camera translation.",
}

# 1400-prompt suite is split per dimension. Each upstream prompt file is
# `T2V-CompBench/prompts/<dim>_examples.txt` (one prompt per line).
_BUNDLED_FALLBACK_PROMPTS: dict[str, list[str]] = {
    "consistent_attribute":  ["a blue car parked next to a white fence"],
    "dynamic_attribute":     ["a green leaf gradually turning yellow"],
    "action_binding":        ["the cat on the left climbs a tree, the dog on the right runs"],
    "object_interactions":   ["a person hands a cup of coffee to another person"],
    "spatial_relationships": ["a red apple to the left of a green pear"],
    "generative_numeracy":   ["three yellow ducks in a row on a pond"],
    "motion_binding":        ["a small fish swims from the left side of the frame to the right"],
}


class T2VCompBenchBenchmark(BaseBenchmark):
    name = "t2vcompbench"
    env_name = "videvalkit"
    dimensions = T2VCOMPBENCH_DIMENSIONS
    video_layout = "{model}/{prompt_id}-{sample_index}.mp4"

    # Upstream V2 uses 1 sample per prompt for MLLM dims, more for tracking.
    SAMPLES_PER_PROMPT: int = 1

    # Frame sampling per scorer family.
    MLLM_GRID_FRAMES = 6        # Grid-LLaVA tile size
    MLLM_SEQ_FRAMES = 16        # D-LLaVA frame-by-frame
    TRACKING_FRAMES = 16        # ~8 fps over 2 s

    # ---- prompts ------------------------------------------------------------
    def list_prompts(
        self,
        dimensions: list[str] | None = None,
        prompts_file: str | Path | None = None,
    ) -> Iterator[PromptItem]:
        """Yield prompts for each requested dimension.

        If `prompts_file` is supplied it must be a JSONL where each entry has
        ``{"prompt_id": ..., "text": ..., "dimension": ...}``. Otherwise a
        single example per dim is yielded from the bundled fallback so
        smoke tests don't need external data.
        """
        wanted = set(dimensions) if dimensions else set(self.dimensions)
        if prompts_file is not None:
            with Path(prompts_file).open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    e = json.loads(line)
                    dim = e.get("dimension")
                    if dim not in wanted:
                        continue
                    yield PromptItem(
                        prompt_id=str(e["prompt_id"]),
                        text=e["text"],
                        dimensions=[dim],
                        meta={"scorer_kind": T2VCOMPBENCH_SCORER_KIND.get(dim)},
                    )
            return
        for dim, examples in _BUNDLED_FALLBACK_PROMPTS.items():
            if dim not in wanted:
                continue
            for i, text in enumerate(examples):
                yield PromptItem(
                    prompt_id=f"{dim}_{i:03d}",
                    text=text,
                    dimensions=[dim],
                    meta={"scorer_kind": T2VCOMPBENCH_SCORER_KIND[dim]},
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
                for idx in range(n):
                    rel = self.video_layout.format(
                        model=m, prompt_id=p.prompt_id, sample_index=idx,
                    )
                    specs.append(VideoSpec(
                        path=layout.videos_dir / rel,
                        prompt_id=p.prompt_id,
                        model_name=m,
                        dimension=(p.dimensions[0] if p.dimensions else None),
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
        n_frames: int = 8,
        mode: str = "upstream",
        **kwargs: Any,
    ) -> list[RawResult]:
        """Score videos on the requested T2V-CompBench dimensions.

        Two modes:

        * ``mode='upstream'`` (default, **paper-faithful**) — every dim is
          run by shelling into the actual upstream scripts under
          ``validation/upstream_repos/T2V-CompBench/`` via the wrappers in
          :mod:`videvalkit.benchmarks.t2vcompbench.upstream`. MLLM dims use
          upstream's LLaVA-1.6-34B chain (3 seeds, chatml_direct, 6-frame
          3x2 grid for consistent/action/interaction, 8-frame sequence for
          dynamic_attr); CV dims use upstream's GroundingDINO_SwinT_OGC +
          SAM-H + Depth-Anything V1 + DOT-cotracker2 pipelines. Scores are
          byte-equivalent to upstream's per-video CSVs.

        * ``mode='toolkit'`` (legacy / smoke-test fallback) — uses the local
          toolkit scorers (single-rubric VLM call for MLLM dims, lighter
          HF-only stack for CV). Faster but produces different numbers from
          upstream; useful for fast smoke / dev tests when the full
          subprocess + checkpoint stack is unavailable.

        Args:
            mode: ``'upstream'`` or ``'toolkit'``.
            judge: required only in ``'toolkit'`` mode for MLLM dims; in
                ``'upstream'`` mode MLLM scoring runs in subprocess and
                ``judge`` is ignored. Optionally used in ``'upstream'``
                mode as an LLM extractor to populate ``PromptItem.meta``
                with the upstream schema when prompts lack it.
        """
        if not videos:
            return []
        if mode not in ("upstream", "toolkit"):
            raise ValueError(f"unknown mode={mode!r}; expected 'upstream' or 'toolkit'")

        wanted = set(dimensions) if dimensions else set(self.dimensions)
        unknown = wanted - set(self.dimensions)
        if unknown:
            raise ValueError(f"unknown t2vcompbench dimension(s): {unknown}")

        if mode == "upstream":
            return self._evaluate_upstream(
                videos=videos, layout=layout, wanted=wanted,
                prompts=prompts or [], judge=judge,
            )

        # ---- toolkit / legacy path ----
        from videvalkit.benchmarks.t2vcompbench.scorers import (
            T2VCOMPBENCH_CV_DIMS, T2VCOMPBENCH_MLLM_DIMS,
            GenerativeNumeracyScorer, MotionBindingScorer,
            SpatialRelationshipsScorer, score_video_dim,
        )
        mllm_dims = set(T2VCOMPBENCH_MLLM_DIMS)
        cv_dims = set(T2VCOMPBENCH_CV_DIMS)
        mllm_wanted = wanted & mllm_dims
        cv_wanted = wanted & cv_dims

        prompt_text: dict[str, str] = {p.prompt_id: p.text for p in (prompts or [])}
        # Short-circuit: no prompts → no scorers loaded, no GPU mem wasted.
        if not any(prompt_text.get(vs.prompt_id) for vs in videos):
            for vs in videos:
                log.warning("t2vcompbench: missing prompt text for %s; "
                            "all dims skipped for this video", vs.prompt_id)
            return []

        # Build judge only if MLLM dims are requested.
        judge_obj = None
        if mllm_wanted:
            if judge is None:
                raise ValueError(
                    "t2vcompbench MLLM dims need a judge — pass "
                    "`judge=SUPPORTED_JUDGES['<vlm>']` (e.g. gemma-4-31b-local)"
                )
            from videvalkit.scorers.vlm_judge.factory import build_judge
            judge_obj = build_judge(judge, layout=layout)
            if hasattr(judge_obj, "setup"):
                judge_obj.setup()

        # Lazy-load CV scorers on first use (avoid GPU load when no work).
        cv_objs: dict[str, Any] = {}

        def _get_cv(dim: str):
            if dim in cv_objs:
                return cv_objs[dim]
            ctor = {
                "generative_numeracy":   GenerativeNumeracyScorer,
                "spatial_relationships": SpatialRelationshipsScorer,
                "motion_binding":        MotionBindingScorer,
            }[dim]
            cv_objs[dim] = ctor(device=self._device())
            return cv_objs[dim]

        results: list[RawResult] = []
        for vs in videos:
            text = prompt_text.get(vs.prompt_id, "")
            if not text:
                log.warning("t2vcompbench: missing prompt text for %s; "
                            "all dims skipped for this video", vs.prompt_id)
                continue
            dims_for_this_video = (
                {vs.dimension} & wanted if vs.dimension else wanted
            )

            # ---- MLLM path ----
            for dim in sorted(dims_for_this_video & mllm_wanted):
                try:
                    r = score_video_dim(
                        judge_obj, vs.path, text, dim, n_frames=n_frames,
                    )
                except Exception as e:
                    log.warning("t2vcompbench: %s on %s failed: %s",
                                dim, vs.prompt_id, e)
                    continue
                results.append(RawResult(
                    benchmark=self.name, model=vs.model_name,
                    dimension=dim, prompt_id=vs.prompt_id,
                    score=r["score"],
                    scorer=f"vlm:{r['judge_model']}",
                    video_path=str(vs.path),
                    meta={
                        "justification": r["justification"],
                        "n_frames":      r["n_frames"],
                        "mode":          r["mode"],
                    },
                ))

            # ---- CV path (frames shared across the 3 CV dims) ----
            cv_to_run = dims_for_this_video & cv_wanted
            if cv_to_run:
                from videvalkit.utils.video import extract_frames
                try:
                    frames = extract_frames(vs.path, n_frames=n_frames)
                except Exception as e:
                    log.warning("t2vcompbench: cannot extract frames from %s: %s",
                                vs.path, e)
                    continue

                # Build CV meta from raw prompt text. If caller provided an
                # LLM judge we use it; otherwise fall back to the deterministic
                # heuristic in extract.py (each extractor handles the no-judge
                # case by returning an empty dict / skipping).
                from videvalkit.benchmarks.t2vcompbench.extract import (
                    extract_motion, extract_numeracy, extract_spatial,
                )
                _DIM_TO_EXTRACTOR = {
                    "generative_numeracy":   extract_numeracy,
                    "spatial_relationships": extract_spatial,
                    "motion_binding":        extract_motion,
                }

                for dim in sorted(cv_to_run):
                    cv_obj = _get_cv(dim)
                    try:
                        # Resolve meta from raw prompt via LLM extractor when
                        # a judge is available, else via the deterministic
                        # regex heuristic in extract.py.
                        meta = _DIM_TO_EXTRACTOR[dim](judge_obj, text)
                        r = cv_obj.score_one(frames, meta)
                    except Exception as e:
                        log.warning("t2vcompbench: %s on %s failed: %s",
                                    dim, vs.prompt_id, e)
                        continue
                    score_key = dim  # score's dict key matches the dim name
                    scorer_tag = {
                        "generative_numeracy":   "cv:grounding_dino_count",
                        "spatial_relationships": "cv:gd_bbox_depth",
                        "motion_binding":        "cv:gd_bbox_raft_flow",
                    }[dim]
                    meta = {k: v for k, v in r.items() if k != score_key}
                    results.append(RawResult(
                        benchmark=self.name, model=vs.model_name,
                        dimension=dim, prompt_id=vs.prompt_id,
                        score=float(r[score_key]),
                        scorer=scorer_tag,
                        video_path=str(vs.path),
                        meta=meta,
                    ))
        return results

    # ---- upstream-mode dispatch --------------------------------------------
    def _evaluate_upstream(
        self,
        *,
        videos: list[VideoSpec],
        layout: WorkspaceLayout | None,
        wanted: set[str],
        prompts: list[PromptItem],
        judge: dict[str, Any] | None,
    ) -> list[RawResult]:
        """Dispatch each requested dim to its upstream.py subprocess wrapper.

        Strategy: group videos by model, then for each (model, dim) call the
        matching upstream entry-point with a per-(model, dim) work dir under
        ``layout.root`` (or a tempdir if no layout).
        """
        import tempfile
        from videvalkit.benchmarks.t2vcompbench import upstream as up
        from videvalkit.benchmarks.t2vcompbench.extract import (
            extract_motion, extract_numeracy, extract_spatial,
        )

        # Resolve work root.
        if layout is not None:
            work_root = layout.root / "_t2vcompbench_upstream"
            work_root.mkdir(parents=True, exist_ok=True)
            _tmp = None
        else:
            _tmp = tempfile.TemporaryDirectory(prefix="t2vcompbench_upstream_")
            work_root = Path(_tmp.name)

        # Lazy LLM-extractor for CV-meta only when judge is provided AND
        # the prompt doesn't already carry the upstream schema.
        from videvalkit.scorers.vlm_judge.factory import build_judge
        llm_for_extract = None
        if judge is not None:
            try:
                llm_for_extract = build_judge(judge, layout=layout)
                if hasattr(llm_for_extract, "setup"):
                    llm_for_extract.setup()
            except Exception as e:
                log.warning("t2vcompbench/upstream: cannot build extractor judge "
                            "(%s); will use deterministic regex heuristic", e)
                llm_for_extract = None

        # Helper: populate upstream meta if missing.
        def _ensure_meta(p: PromptItem, dim: str) -> PromptItem:
            if not isinstance(p.meta, dict):
                p = PromptItem(prompt_id=p.prompt_id, text=p.text,
                               dimensions=p.dimensions, meta={})
            meta = dict(p.meta)
            if dim == "generative_numeracy":
                if not (meta.get("objects") and meta.get("numbers")):
                    extracted = extract_numeracy(llm_for_extract, p.text)
                    objs = extracted.get("objects", [])
                    nums = extracted.get("numbers", [])
                    meta["objects"] = ",".join(str(o) for o in objs) if isinstance(objs, list) else str(objs)
                    meta["numbers"] = ",".join(str(n) for n in nums) if isinstance(nums, list) else str(nums)
            elif dim == "spatial_relationships":
                if not meta.get("object_1"):
                    meta.update(extract_spatial(llm_for_extract, p.text))
            elif dim == "motion_binding":
                if not meta.get("object_1"):
                    meta.update(extract_motion(llm_for_extract, p.text))
            return PromptItem(prompt_id=p.prompt_id, text=p.text,
                              dimensions=p.dimensions, meta=meta)

        # Group videos by model.
        by_model: dict[str, list[VideoSpec]] = {}
        for v in videos:
            by_model.setdefault(v.model_name, []).append(v)

        results: list[RawResult] = []
        prompts_by_id = {p.prompt_id: p for p in prompts}
        try:
            for model_tag, vids_for_model in by_model.items():
                # Prompts present in this batch (preserving order).
                pids_in_batch = []
                seen = set()
                for v in vids_for_model:
                    if v.prompt_id not in seen and v.prompt_id in prompts_by_id:
                        pids_in_batch.append(v.prompt_id)
                        seen.add(v.prompt_id)
                if not pids_in_batch:
                    log.warning("t2vcompbench/upstream: no prompt text for any "
                                "video of model %s; skipping", model_tag)
                    continue
                prompts_for_model = [prompts_by_id[pid] for pid in pids_in_batch]

                for dim in sorted(wanted):
                    # Filter videos to those whose intended dim matches (or all
                    # if vs.dimension is None).
                    vids_for_dim = [
                        v for v in vids_for_model
                        if v.dimension is None or v.dimension == dim
                    ]
                    if not vids_for_dim:
                        continue
                    pids_for_dim = [
                        pid for pid in pids_in_batch
                        if any(v.prompt_id == pid for v in vids_for_dim)
                    ]
                    prompts_for_dim = [prompts_by_id[pid] for pid in pids_for_dim]

                    # CV dims need meta; MLLM dims rely on PromptItem.meta as-is.
                    if dim in ("generative_numeracy", "spatial_relationships",
                               "motion_binding"):
                        prompts_for_dim = [_ensure_meta(p, dim) for p in prompts_for_dim]

                    try:
                        if dim == "generative_numeracy":
                            r = up.run_upstream_numeracy(
                                vids_for_dim, prompts_for_dim, work_root,
                                t2v_model_tag=model_tag,
                            )
                        elif dim == "spatial_relationships":
                            r = up.run_upstream_spatial(
                                vids_for_dim, prompts_for_dim, work_root,
                                t2v_model_tag=model_tag,
                            )
                        elif dim == "motion_binding":
                            r = up.run_upstream_motion_binding(
                                vids_for_dim, prompts_for_dim, work_root,
                                t2v_model_tag=model_tag,
                            )
                        else:
                            # MLLM dims (4): consistent_attribute, dynamic_attribute,
                            # action_binding, object_interactions.
                            r = up.run_upstream_mllm(
                                dim, vids_for_dim, prompts_for_dim, work_root,
                                t2v_model_tag=model_tag,
                            )
                    except Exception as e:
                        log.error("t2vcompbench/upstream %s on model %s failed: %s",
                                  dim, model_tag, e)
                        continue
                    results.extend(r)
        finally:
            if _tmp is not None:
                _tmp.cleanup()

        return results

    def _device(self) -> str:
        try:
            import torch                    # noqa: PLC0415
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    # ---- aggregate ----------------------------------------------------------
    def aggregate(
        self,
        raw: list[RawResult],
        aggregator: str = "weighted_sum",
        **kwargs: Any,
    ) -> Summary:
        """Mean per dim; overall = mean of per-dim means (upstream reports
        each dim independently, no official cross-dim weighting)."""
        from collections import defaultdict
        per_dim: dict[str, list[float]] = defaultdict(list)
        for r in raw:
            if isinstance(r.score, (int, float)):
                per_dim[r.dimension].append(float(r.score))
        per_dim_mean = {d: sum(vs) / len(vs) for d, vs in per_dim.items() if vs}
        overall = (sum(per_dim_mean.values()) / len(per_dim_mean)
                   if per_dim_mean else 0.0)
        model = raw[0].model if raw else ""
        return Summary(
            benchmark=self.name,
            model=model,
            per_dimension=per_dim_mean,
            overall=overall,
            n_videos=len({(r.model, r.prompt_id, r.dimension) for r in raw}),
            n_prompts=len({r.prompt_id for r in raw}),
            aggregator=aggregator,
            meta={"note": "Upstream reports each dim independently; cross-dim mean is toolkit convention."},
        )
