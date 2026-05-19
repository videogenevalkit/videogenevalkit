"""VBench v1 adapter — wraps `Vchitect/VBench` (16 dims, classic metric scorers).

Strategy: thin adapter, call upstream's Python API.

  1. Stage videos: VBench expects either `mode='custom_input'` (any layout) or
     `mode='vbench_standard'` (videos under {model}/{dim}/{prompt}-{idx}.mp4).
     We use `custom_input` because our toolkit layout differs.
  2. For each (model, dim), invoke `VBench(...).evaluate(videos_path=..., dimension_list=[dim], name=..., mode='custom_input')`.
  3. Parse the resulting JSON files VBench writes and convert to RawResult.

This adapter runs in the `videvalkit-vbench` conda env where `vbench` is installed.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from videvalkit.core.benchmark import BaseBenchmark
from videvalkit.core.layout import WorkspaceLayout
from videvalkit.core.types import PromptItem, RawResult, Summary, VideoSpec

log = logging.getLogger(__name__)


VBENCH_DIMENSIONS = [
    "subject_consistency", "background_consistency", "temporal_flickering",
    "motion_smoothness", "dynamic_degree", "aesthetic_quality", "imaging_quality",
    "object_class", "multiple_objects", "human_action", "color",
    "spatial_relationship", "scene", "temporal_style", "appearance_style",
    "overall_consistency",
]

# Quality vs Semantic split, mirroring VBench/scripts/constant.py weights.
VBENCH_QUALITY_DIMS = {
    "subject_consistency", "background_consistency", "temporal_flickering",
    "motion_smoothness", "dynamic_degree", "aesthetic_quality", "imaging_quality",
}
VBENCH_SEMANTIC_DIMS = set(VBENCH_DIMENSIONS) - VBENCH_QUALITY_DIMS


# ─── upstream cal_final_score.py constants (verbatim from VBench/scripts/constant.py) ─── #
# Per-dim weights — note `dynamic degree` is 0.5, all others 1.0.
VBENCH_DIM_WEIGHT: dict[str, float] = {
    "subject_consistency":   1.0,
    "background_consistency": 1.0,
    "temporal_flickering":   1.0,
    "motion_smoothness":     1.0,
    "aesthetic_quality":     1.0,
    "imaging_quality":       1.0,
    "dynamic_degree":        0.5,
    "object_class":          1.0,
    "multiple_objects":      1.0,
    "human_action":          1.0,
    "color":                 1.0,
    "spatial_relationship":  1.0,
    "scene":                 1.0,
    "appearance_style":      1.0,
    "temporal_style":        1.0,
    "overall_consistency":   1.0,
}

# Prompt-dependent dims that upstream `check_dimension_requires_extra_info`
# (vbench/__init__.py:22-26) hard-asserts cannot run in ``mode='custom_input'``.
# These dims need `vbench_standard` (or a custom `full_info_path` matching
# upstream's schema).
VBENCH_PROMPT_DEPENDENT_DIMS: set[str] = {
    "object_class", "multiple_objects", "scene",
    "appearance_style", "color", "spatial_relationship",
}


# Per-dim (Min, Max) for re-scaling raw scores to [0, 1] before weighting.
VBENCH_NORMALIZE: dict[str, tuple[float, float]] = {
    "subject_consistency":   (0.1462, 1.0),
    "background_consistency": (0.2615, 1.0),
    "temporal_flickering":   (0.6293, 1.0),
    "motion_smoothness":     (0.706,  0.9975),
    "dynamic_degree":        (0.0,    1.0),
    "aesthetic_quality":     (0.0,    1.0),
    "imaging_quality":       (0.0,    1.0),
    "object_class":          (0.0,    1.0),
    "multiple_objects":      (0.0,    1.0),
    "human_action":          (0.0,    1.0),
    "color":                 (0.0,    1.0),
    "spatial_relationship":  (0.0,    1.0),
    "scene":                 (0.0,    0.8222),
    "appearance_style":      (0.0009, 0.2855),
    "temporal_style":        (0.0,    0.364),
    "overall_consistency":   (0.0,    0.364),
}


class VBenchBenchmark(BaseBenchmark):
    name = "vbench"
    env_name = "videvalkit-vbench"
    dimensions = VBENCH_DIMENSIONS
    video_layout = "{model}/{prompt_id}-{sample_index}.mp4"

    # Mirror upstream constant.py SEMANTIC_WEIGHT=1, QUALITY_WEIGHT=4 .
    QUALITY_WEIGHT: float = 4.0
    SEMANTIC_WEIGHT: float = 1.0

    # ---- prompts ------------------------------------------------------------
    def list_prompts(self, dimensions: list[str] | None = None) -> Iterator[PromptItem]:
        """VBench's prompt set is bundled inside its installed package.

        Loads `VBench_full_info.json` from the installed `vbench` module.
        """
        info = self._load_vbench_full_info()
        wanted = set(dimensions) if dimensions else set(self.dimensions)
        for entry in info:
            dims = [d for d in entry.get("dimension", []) if d in wanted]
            if not dims:
                continue
            yield PromptItem(
                prompt_id=str(entry.get("prompt_id", entry.get("prompt_en", ""))),
                text=entry.get("prompt_en") or entry.get("prompt", ""),
                dimensions=dims,
                meta=entry,
            )

    def _load_vbench_full_info(self) -> list[dict[str, Any]]:
        try:
            vbench_mod = importlib.import_module("vbench")
        except ImportError as e:
            raise RuntimeError(
                "vbench package not installed; run inside `videvalkit-vbench` env"
            ) from e
        vbench_dir = Path(vbench_mod.__file__).parent
        info_path = vbench_dir / "VBench_full_info.json"
        if not info_path.exists():
            # Some versions place it under prompts/
            alt = vbench_dir / "prompts" / "VBench_full_info.json"
            if alt.exists():
                info_path = alt
            else:
                raise FileNotFoundError(f"VBench_full_info.json not found near {vbench_dir}")
        return json.loads(info_path.read_text())

    def list_required_videos(
        self,
        prompts: list[PromptItem],
        models: list[str],
        layout: WorkspaceLayout,
        samples_per_prompt: int = 5,
    ) -> list[VideoSpec]:
        """VBench default: 5 samples per prompt."""
        specs: list[VideoSpec] = []
        for m in models:
            for p in prompts:
                for idx in range(samples_per_prompt):
                    rel = self.video_layout.format(
                        model=m, prompt_id=p.prompt_id, sample_index=idx
                    )
                    specs.append(VideoSpec(
                        path=layout.videos_dir / rel, prompt_id=p.prompt_id,
                        model_name=m, sample_index=idx,
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
        samples_per_prompt: int = 5,
        full_info_path: str | Path | None = None,
        mode: str = "custom_input",
        **kwargs: Any,
    ) -> list[RawResult]:
        """
        Optional kwargs:
            full_info_path -- absolute path to a VBench_full_info.json-compatible
                              JSON (e.g. from `scripts/auto_label_prompts.py`).
                              When set, override the upstream default and pass
                              this to VBench(full_info_dir=...). Required for
                              prompt-dependent dims in `vbench_standard` mode.
            mode           -- "custom_input" (default) or "vbench_standard".
        """
        if layout is None:
            raise ValueError("VBench.evaluate requires layout=WorkspaceLayout")
        try:
            from vbench import VBench
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "Install vbench in `videvalkit-vbench` env first (see envs/vbench.yaml)"
            ) from e

        dims = list(dimensions) if dimensions else list(self.dimensions)

        # Pre-flight: upstream's `check_dimension_requires_extra_info` will
        # assert in `custom_input` mode for the 6 prompt-dependent dims.
        # Surface the constraint here with a clearer, actionable message
        # rather than letting the upstream assert fire mid-staging.
        if mode == "custom_input":
            unsupported = sorted(set(dims) & VBENCH_PROMPT_DEPENDENT_DIMS)
            if unsupported:
                raise ValueError(
                    f"VBench dims {unsupported} cannot run in mode='custom_input' "
                    "(upstream asserts at vbench/__init__.py:22-26). "
                    "Either pass mode='vbench_standard' (and stage videos with "
                    "the {prompt_text}-{idx}.mp4 filenames upstream expects), "
                    "or pass full_info_path=<your auto-labelled VBench_full_info.json> "
                    "and mode='vbench_standard'. See "
                    "scripts/auto_label_prompts.py for label-generation help."
                )

        models_set = sorted({v.model_name for v in (videos or [])} or set(models or []))
        if not models_set:
            raise ValueError("VBench.evaluate: no models specified")

        from videvalkit.storage.workspace import Workspace
        ws = Workspace(layout.root)

        results: list[RawResult] = []
        for model in models_set:
            for dim in dims:
                if all(ws.has_raw(self.name, model, dim, p) for p in self._model_prompt_ids(model, videos, layout)):
                    log.info("vbench: skipping %s/%s (already complete)", model, dim)
                    results.extend([ws.load_raw(self.name, model, dim, p)
                                    for p in self._model_prompt_ids(model, videos, layout)
                                    if ws.has_raw(self.name, model, dim, p)])
                    continue
                staged_dir = self._stage_videos_for_dim(model, dim, videos, layout)
                with tempfile.TemporaryDirectory() as out_dir:
                    log.info("vbench: %s / %s -> %s (out=%s)", model, dim, staged_dir, out_dir)
                    # Prefer GPU when available; the env var heuristic was wrong
                    # because an unset CUDA_VISIBLE_DEVICES still means "all GPUs".
                    try:
                        import torch
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                    except ImportError:
                        device = "cpu"
                    full_info_dir = (
                        str(full_info_path) if full_info_path
                        else str(self._vbench_full_info_path())
                    )
                    vb = VBench(
                        output_path=out_dir,
                        full_info_dir=full_info_dir,
                        device=device,
                    )
                    vb.evaluate(
                        videos_path=str(staged_dir),
                        name=f"{model}_{dim}",
                        dimension_list=[dim],
                        mode=mode,
                    )
                    results.extend(self._collect_dim_results(model, dim, Path(out_dir), ws))
        return results

    def _model_prompt_ids(
        self, model: str, videos: list[VideoSpec] | None, layout: WorkspaceLayout,
    ) -> list[str]:
        if videos:
            return sorted({v.prompt_id for v in videos if v.model_name == model})
        # Discover from disk
        d = layout.videos_dir / model
        if not d.exists():
            return []
        return sorted({p.stem.split("-")[0] for p in d.glob("*.mp4")})

    def _vbench_full_info_path(self) -> Path:
        vbench_mod = importlib.import_module("vbench")
        return Path(vbench_mod.__file__).parent

    def _stage_videos_for_dim(
        self,
        model: str,
        dimension: str,
        videos: list[VideoSpec] | None,
        layout: WorkspaceLayout,
    ) -> Path:
        """Stage videos into a tmpdir layout VBench's `custom_input` mode accepts.

        VBench's custom_input mode just scans for *.mp4 in a directory, so we
        symlink all (model, *) videos into a single tmpdir per dim.
        """
        staged_root = layout.frames_cache_dir / f"_vbench_stage_{model}_{dimension}"
        if staged_root.exists():
            shutil.rmtree(staged_root)
        staged_root.mkdir(parents=True)
        sources = videos or [
            VideoSpec(path=p, prompt_id=p.stem.split("-")[0], model_name=model)
            for p in (layout.videos_dir / model).glob("*.mp4")
        ]
        for v in sources:
            if v.model_name != model:
                continue
            link = staged_root / v.path.name
            if not link.exists():
                try:
                    link.symlink_to(v.path.resolve())
                except OSError:
                    shutil.copy2(v.path, link)
        return staged_root

    def _collect_dim_results(
        self, model: str, dim: str, vbench_out: Path, ws: Any,
    ) -> list[RawResult]:
        """Collect RawResults from VBench's per-dim JSON output.

        Three JSON shapes are observed across VBench v0.x — v1.x:

        Shape A (v0.x list-of-list):
            {dim: [mean_score, [{video_path, video_results}, ...]]}

        Shape B (v1.x with named keys):
            {dim: {"score": mean, "videos": [{"video_path": ..., "video_results": ...}, ...]}}

        Shape C (custom_input shorthand):
            {dim: per_video_score_float}   # rare; only the mean is given

        We parse all three and normalize into RawResult.
        """
        results: list[RawResult] = []
        for jf in vbench_out.glob("**/*_eval_results.json"):
            try:
                data = json.loads(jf.read_text())
            except json.JSONDecodeError:
                continue
            for dim_name, payload in data.items():
                if dim_name != dim:
                    continue
                results.extend(self._parse_dim_payload(model, dim, payload, jf, ws))
        return results

    def _parse_dim_payload(
        self, model: str, dim: str, payload: Any, jf: Path, ws: Any,
    ) -> list[RawResult]:
        out: list[RawResult] = []
        # Shape A: [mean, [{video_path, video_results}, ...]]
        if isinstance(payload, list) and len(payload) >= 2 and isinstance(payload[1], list):
            mean_score, per_video = payload[0], payload[1]
            for entry in per_video:
                out.extend(self._row_to_result(model, dim, entry, jf, ws, mean_score=mean_score))
            return out
        # Shape B: {"score": mean, "videos": [...]}
        if isinstance(payload, dict) and "videos" in payload and isinstance(payload["videos"], list):
            mean_score = payload.get("score") or payload.get("mean")
            for entry in payload["videos"]:
                out.extend(self._row_to_result(model, dim, entry, jf, ws, mean_score=mean_score))
            return out
        # Shape C: scalar mean only (no per-video breakdown). Emit one summary row.
        if isinstance(payload, (int, float)):
            r = RawResult(
                benchmark=self.name, model=model, dimension=dim,
                prompt_id="_mean", score=float(payload),
                scorer="vbench_metric", meta={"upstream_file": str(jf), "shape": "C"},
            )
            ws.write_raw(r)
            out.append(r)
            return out
        return out

    def _row_to_result(
        self, model: str, dim: str, entry: dict, jf: Path, ws: Any,
        mean_score: float | None = None,
    ) -> list[RawResult]:
        video_path = entry.get("video_path") or entry.get("video") or ""
        # `video_results` may be a float or a dict {"score": float, ...}
        score = entry.get("video_results")
        if isinstance(score, dict):
            score = score.get("score") or score.get("video_results")
        if score is None:
            score = entry.get("score")
        if score is None:
            return []
        try:
            score_val = float(score)
        except (TypeError, ValueError):
            return []
        prompt_id = Path(video_path).stem.split("-")[0] if video_path else "unknown"
        r = RawResult(
            benchmark=self.name, model=model, dimension=dim,
            prompt_id=prompt_id, score=score_val,
            scorer="vbench_metric", video_path=video_path,
            meta={"upstream_file": str(jf), "mean_score": mean_score},
        )
        ws.write_raw(r)
        return [r]

    # ---- aggregate ----------------------------------------------------------
    def aggregate(
        self,
        raw: list[RawResult],
        aggregator: str = "vbench_weighted",
        **kwargs: Any,
    ) -> Summary:
        """Reproduce upstream `cal_final_score.py:get_final_score`.

        Algorithm:
          1. Mean raw score per dim across all (model, prompt, sample).
          2. Min-max rescale each per-dim mean to [0, 1] using
             ``VBENCH_NORMALIZE`` constants.
          3. Multiply by per-dim weight (``VBENCH_DIM_WEIGHT``;
             ``dynamic_degree`` is 0.5, others 1.0).
          4. Quality score   = Σ(weighted Q dims) / Σ(Q weights)
             Semantic score  = Σ(weighted S dims) / Σ(S weights)
          5. Final = (Q · QUALITY_WEIGHT + S · SEMANTIC_WEIGHT)
                     / (QUALITY_WEIGHT + SEMANTIC_WEIGHT)
             (upstream uses QW=4, SW=1).
        """
        from collections import defaultdict
        per_dim_raw: dict[str, list[float]] = defaultdict(list)
        for r in raw:
            if isinstance(r.score, (int, float)):
                per_dim_raw[r.dimension].append(float(r.score))

        per_dim_mean: dict[str, float] = {
            d: sum(vs) / len(vs) for d, vs in per_dim_raw.items() if vs
        }

        # Step 2-3: normalize + weight
        per_dim_weighted: dict[str, float] = {}
        for d, raw_mean in per_dim_mean.items():
            mn, mx = VBENCH_NORMALIZE.get(d, (0.0, 1.0))
            norm = (raw_mean - mn) / (mx - mn) if mx > mn else raw_mean
            per_dim_weighted[d] = norm * VBENCH_DIM_WEIGHT.get(d, 1.0)

        # Step 4: Quality + Semantic section means (weighted denominator)
        q_present = [d for d in VBENCH_QUALITY_DIMS if d in per_dim_weighted]
        s_present = [d for d in VBENCH_SEMANTIC_DIMS if d in per_dim_weighted]
        q_score = (
            sum(per_dim_weighted[d] for d in q_present)
            / sum(VBENCH_DIM_WEIGHT[d] for d in q_present)
        ) if q_present else 0.0
        s_score = (
            sum(per_dim_weighted[d] for d in s_present)
            / sum(VBENCH_DIM_WEIGHT[d] for d in s_present)
        ) if s_present else 0.0

        # Step 5: final score
        if q_present and s_present:
            overall = (
                (q_score * self.QUALITY_WEIGHT + s_score * self.SEMANTIC_WEIGHT)
                / (self.QUALITY_WEIGHT + self.SEMANTIC_WEIGHT)
            )
        elif q_present:
            overall = q_score
        elif s_present:
            overall = s_score
        else:
            overall = 0.0

        # Report **normalized** per-dim scores in per_dimension (consistent
        # with upstream leaderboard display).
        per_dim_normalized: dict[str, float] = {}
        for d, raw_mean in per_dim_mean.items():
            mn, mx = VBENCH_NORMALIZE.get(d, (0.0, 1.0))
            per_dim_normalized[d] = (raw_mean - mn) / (mx - mn) if mx > mn else raw_mean

        model = raw[0].model if raw else ""
        return Summary(
            benchmark=self.name, model=model,
            per_dimension=per_dim_normalized,
            overall=overall,
            n_videos=len({(r.model, r.prompt_id, r.dimension) for r in raw}),
            n_prompts=len({r.prompt_id for r in raw}),
            aggregator=aggregator,
            meta={
                "quality_score": q_score,
                "semantic_score": s_score,
                "quality_weight": self.QUALITY_WEIGHT,
                "semantic_weight": self.SEMANTIC_WEIGHT,
                "per_dim_raw_mean": per_dim_mean,
                "formula": "(Q*4 + S*1)/5 with per-dim min-max norm and 0.5 weight on dynamic_degree",
            },
        )

    # ---- official export ----------------------------------------------------
    def export_official(self, summary: Summary, out_path: Path) -> None:
        """Emit a JSON in the format VBench's leaderboard accepts.

        Upstream `cal_final_score.py` writes dim names with spaces and dashes,
        e.g. "subject consistency", not "subject_consistency". We translate.
        """
        def _key(d: str) -> str:
            return d.replace("_", " ")

        # Pull section scores out of meta (computed in aggregate()).
        q = summary.meta.get("quality_score")
        s = summary.meta.get("semantic_score")
        payload = {
            "model": summary.model,
            "scores": {_key(d): v for d, v in summary.per_dimension.items()},
            "Total_Score": summary.overall,
            "Quality_Score": q if q is not None
                              else float(sum(summary.per_dimension.get(d, 0.0) for d in VBENCH_QUALITY_DIMS)
                                         / max(1, len([d for d in VBENCH_QUALITY_DIMS if d in summary.per_dimension]))),
            "Semantic_Score": s if s is not None
                               else float(sum(summary.per_dimension.get(d, 0.0) for d in VBENCH_SEMANTIC_DIMS)
                                          / max(1, len([d for d in VBENCH_SEMANTIC_DIMS if d in summary.per_dimension]))),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
