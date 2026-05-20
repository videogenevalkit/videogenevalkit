"""VBench-2.0 adapter — wraps `Vchitect/VBench/VBench-2.0`.

Strategy: thin adapter, call upstream's Python API
    from vbench2 import VBench2
    VBench2(...).evaluate(videos_path=..., dimension_list=[dim], mode='custom_input')

Dimensions live in 5 categories (Creativity / Commonsense / Controllability /
Human Fidelity / Physics). The full per-category dim list is loaded lazily
from the installed package so we stay in sync with upstream releases.

Runs in `videvalkit-vbench2` conda env.
"""

from __future__ import annotations

import importlib
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


VBENCH2_CATEGORIES = ["Creativity", "Commonsense", "Controllability", "Human_Fidelity", "Physics"]

# Canonical CapitalizedNames used by upstream's init_submodules + build_full_dimension_list.
# Our adapter accepts either case; we always translate to canonical before invoking VBench2.
_VBENCH2_CANONICAL_DIMS = [
    "Human_Anatomy", "Human_Identity", "Human_Clothes", "Diversity", "Composition",
    "Dynamic_Spatial_Relationship", "Dynamic_Attribute", "Motion_Order_Understanding",
    "Human_Interaction", "Complex_Landscape", "Complex_Plot", "Camera_Motion",
    "Motion_Rationality", "Instance_Preservation", "Mechanics", "Thermotics",
    "Material", "Multi-View_Consistency",
]

# Dims that work in mode='custom_input' (i.e. arbitrary video folder).
# Others require mode='vbench_standard' which expects upstream's prompt suite.
_VBENCH2_CUSTOM_INPUT_OK = {
    "Human_Anatomy", "Human_Identity", "Human_Clothes", "Diversity", "Multi-View_Consistency",
}

# ─── upstream scripts/constant.py — verbatim category lists ──────────── #
# Category → list of canonical dim names (underscore-form).
_VBENCH2_CATEGORY_DIMS: dict[str, list[str]] = {
    "Creativity":      ["Composition", "Diversity"],
    "Commonsense":     ["Instance_Preservation", "Motion_Rationality"],
    "Controllability": [
        "Camera_Motion", "Complex_Landscape", "Complex_Plot",
        "Dynamic_Attribute", "Dynamic_Spatial_Relationship",
        "Human_Interaction", "Motion_Order_Understanding",
    ],
    "Human_Fidelity":  ["Human_Anatomy", "Human_Clothes", "Human_Identity"],
    "Physics":         ["Material", "Mechanics", "Multi-View_Consistency", "Thermotics"],
}

# Reverse map dim → category for O(1) lookup.
_VBENCH2_DIM_TO_CATEGORY: dict[str, str] = {
    d: cat for cat, ds in _VBENCH2_CATEGORY_DIMS.items() for d in ds
}

# Diversity is the only dim that needs > 1 sample/prompt (the metric is intra-
# group variance). Upstream `vbench2/__init__.py` expects 20 samples/prompt
# for it and 3 for all others; our adapter mirrors that default.
_VBENCH2_SAMPLES_PER_DIM: dict[str, int] = {d: 3 for d in _VBENCH2_CANONICAL_DIMS}
_VBENCH2_SAMPLES_PER_DIM["Diversity"] = 20


def _to_canonical(dim: str) -> str:
    """Map any case variant ('human_anatomy', 'multi-view_consistency', ...) → canonical."""
    lo = dim.lower().replace("-", "_")
    for c in _VBENCH2_CANONICAL_DIMS:
        if c.lower().replace("-", "_") == lo:
            return c
    return dim   # unknown — pass through so upstream's error is informative


class VBench2Benchmark(BaseBenchmark):
    name = "vbench2"
    env_name = "videvalkit-vbench2"
    dimensions: list[str] = []   # populated lazily from upstream
    video_layout = "{model}/{prompt_id}-{sample_index}.mp4"
    categories: list[str] = VBENCH2_CATEGORIES

    def _upstream(self):
        try:
            return importlib.import_module("vbench2")
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "Install VBench-2.0 in `videvalkit-vbench2` env (see envs/vbench2.yaml)"
            ) from e

    def _ensure_dims(self) -> None:
        if self.dimensions:
            return
        # Upstream's canonical 18-dim list. We standardize on Capitalized names
        # so init_submodules can find the right per-dim recipe.
        self.dimensions = list(_VBENCH2_CANONICAL_DIMS)

    # ---- prompts ------------------------------------------------------------
    def list_prompts(self, dimensions: list[str] | None = None) -> Iterator[PromptItem]:
        """Read per-dim prompt files shipped inside the VBench-2.0 repo.

        Upstream lays the prompt files out as ``<repo_root>/prompts/prompt/<Dim>.txt``,
        where ``<repo_root>`` is the parent of the installed ``vbench2/`` package.
        We try a few candidate locations to be robust to install layout.
        """
        self._ensure_dims()
        mod = self._upstream()
        pkg_dir = Path(mod.__file__).resolve().parent       # .../vbench2/
        repo_root = pkg_dir.parent                          # .../VBench-2.0/
        candidates = [
            repo_root / "prompts" / "prompt",   # upstream layout
            repo_root / "prompts",               # adapter v0 layout
            pkg_dir / "prompts" / "prompt",
            pkg_dir / "prompts",
        ]
        prompts_dir: Path | None = next((c for c in candidates if c.is_dir()), None)
        if prompts_dir is None:
            log.warning("vbench2: no prompts dir found under %s; tried: %s",
                        repo_root, [str(c) for c in candidates])
            return

        wanted = set(dimensions) if dimensions else set(self.dimensions)
        for dim in sorted(wanted):
            fp = prompts_dir / f"{dim}.txt"
            if not fp.exists():
                # Some upstream copies use space-separated filenames.
                alt = prompts_dir / f"{dim.replace('_', ' ')}.txt"
                fp = alt if alt.exists() else fp
            if not fp.exists():
                continue
            for idx, line in enumerate(fp.read_text().splitlines()):
                line = line.strip()
                if not line:
                    continue
                yield PromptItem(
                    prompt_id=f"{dim}-{idx:03d}",
                    text=line,
                    dimensions=[dim],
                    meta={"category": self._dim_to_category(dim),
                          "prompt_text": line},
                )

    @staticmethod
    def _dim_to_category(dim: str) -> str:
        """Map canonical dim name → category using upstream's explicit list."""
        return _VBENCH2_DIM_TO_CATEGORY.get(dim, "unknown")

    @staticmethod
    def _samples_for_dim(dim: str) -> int:
        """Upstream: Diversity needs 20 samples/prompt; others 3."""
        return _VBENCH2_SAMPLES_PER_DIM.get(dim, 3)

    def list_required_videos(
        self,
        prompts: list[PromptItem],
        models: list[str],
        layout: WorkspaceLayout,
        samples_per_prompt: int | None = None,
    ) -> list[VideoSpec]:
        """Emit one VideoSpec per (model, prompt, sample).

        If ``samples_per_prompt`` is ``None``, defaults to the upstream
        per-dim count: 20 for Diversity, 3 for all other dims.
        """
        specs: list[VideoSpec] = []
        for m in models:
            for p in prompts:
                # Pick per-dim default sample count when caller didn't override.
                dim = p.dimensions[0] if p.dimensions else None
                n = (samples_per_prompt if samples_per_prompt is not None
                     else (self._samples_for_dim(dim) if dim else 3))
                for idx in range(n):
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
        full_info_path: str | Path | None = None,
        mode_override: str | None = None,
        **kwargs: Any,
    ) -> list[RawResult]:
        """
        Optional kwargs:
            full_info_path -- absolute path to a VBench2_full_info.json-compatible
                              JSON (e.g. from `scripts/auto_label_prompts.py`).
                              When set, override the upstream default.
            mode_override  -- force "vbench_standard" even for dims that the
                              adapter would normally route to custom_input.
        """
        if layout is None:
            raise ValueError("VBench2.evaluate requires layout=WorkspaceLayout")
        mod = self._upstream()
        VBench2 = getattr(mod, "VBench2", None) or getattr(mod, "VBench_2_0", None)
        if VBench2 is None:
            raise RuntimeError("Cannot find VBench2 class in upstream package")
        self._ensure_dims()

        dims = list(dimensions) if dimensions else list(self.dimensions)
        models_set = sorted({v.model_name for v in (videos or [])} or set(models or []))
        if not models_set:
            raise ValueError("VBench2.evaluate: no models specified")

        from videvalkit.storage.workspace import Workspace
        ws = Workspace(layout.root)

        results: list[RawResult] = []
        for model in models_set:
            for dim in dims:
                # Translate to upstream's canonical Capitalized name.
                canonical = _to_canonical(dim)
                mode_to_use = mode_override or (
                    "custom_input" if canonical in _VBENCH2_CUSTOM_INPUT_OK
                    else "vbench_standard"
                )
                staged = self._stage_videos(model, canonical, videos, layout)
                with tempfile.TemporaryDirectory() as out_dir:
                    log.info("vbench2: %s / %s (mode=%s)", model, canonical, mode_to_use)
                    try:
                        import torch
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                    except ImportError:
                        device = "cpu"
                    pkg_dir = Path(mod.__file__).resolve().parent
                    # Upstream's init_submodules uses RELATIVE paths
                    # (e.g. "vbench2/third_party/...") resolved from the VBench-2.0
                    # repo root. Chdir into it for the duration of the call.
                    repo_root = pkg_dir.parent
                    # Several dims load mmdet/mmyolo configs that import
                    # third-party packages by bare name (e.g. `yolo_world`).
                    # Add their containing directories to sys.path so those
                    # `custom_imports` resolve.
                    import sys as _sys
                    extra_paths = [
                        pkg_dir / "third_party" / "YOLO-World",
                        pkg_dir / "third_party" / "ViTDetector",
                        # LLaVA_NeXT self-references via bare `import llava`;
                        # need its parent on sys.path so `llava` resolves.
                        pkg_dir / "third_party" / "LLaVA_NeXT",
                        pkg_dir / "third_party",
                    ]
                    added_paths = []
                    for p in extra_paths:
                        sp = str(p)
                        if p.exists() and sp not in _sys.path:
                            _sys.path.insert(0, sp)
                            added_paths.append(sp)
                    import os as _os
                    prev_cwd = _os.getcwd()
                    # Point VBENCH2_CACHE_DIR at the pretrained_hf subdir for
                    # this dim, if it exists. Each {Dim}/ folder is already
                    # laid out exactly as init_submodules expects (e.g.
                    # YOLO-World/, anomaly_detector/), so no downloads fire.
                    prev_cache = _os.environ.get("VBENCH2_CACHE_DIR")
                    pretrained_hf = repo_root / "pretrained_hf" / canonical
                    if not pretrained_hf.exists():
                        # optional alternate location override
                        alt_root = _os.environ.get("VIDEVALKIT_VBENCH2_PRETRAINED_HF")
                        if alt_root:
                            alt = Path(alt_root) / canonical
                            if alt.exists():
                                pretrained_hf = alt
                    if pretrained_hf.exists():
                        _os.environ["VBENCH2_CACHE_DIR"] = str(pretrained_hf)
                    try:
                        _os.chdir(repo_root)
                        # Filter the prompt registry to only the prompts whose
                        # videos are actually staged. Upstream iterates every
                        # prompt in full_info_dir and builds a per-prompt
                        # `video_list`; un-staged prompts get an empty list,
                        # which Instance_Preservation's compute_anomaly indexes
                        # (`video_paths[0]`) without guarding -> IndexError.
                        # Filtering makes vbench2_standard correct for any
                        # partial/sample video set. On a complete set every
                        # prompt is kept, so this is a no-op (identical result).
                        base_info = (
                            str(full_info_path) if full_info_path
                            else str(pkg_dir / "VBench2_full_info.json")
                        )
                        staged_prompts = set()
                        for _vp in Path(staged).rglob("*.mp4"):
                            _st = _vp.stem
                            staged_prompts.add(
                                _st.rsplit("-", 1)[0]
                                if "-" in _st and _st.rsplit("-", 1)[-1].isdigit()
                                else _st
                            )
                        info_dir = base_info
                        try:
                            _all = json.loads(Path(base_info).read_text())
                            _kept = [e for e in _all
                                     if e.get("prompt_en", "").strip() in staged_prompts]
                            if _kept and len(_kept) < len(_all):
                                _fi = Path(out_dir) / "filtered_full_info.json"
                                _fi.write_text(json.dumps(_kept, ensure_ascii=False))
                                info_dir = str(_fi)
                                log.info("vbench2: filtered registry to %d/%d staged prompts",
                                         len(_kept), len(_all))
                        except Exception as _e:  # fall back to the full registry
                            log.warning("vbench2: registry filter skipped (%s)", _e)
                        vb = VBench2(
                            device=device,
                            full_info_dir=info_dir,
                            output_path=out_dir,
                        )
                        vb.evaluate(
                            videos_path=str(staged),
                            name=f"{model}_{canonical}",
                            dimension_list=[canonical],
                            mode=mode_to_use,
                        )
                    finally:
                        _os.chdir(prev_cwd)
                        for sp in added_paths:
                            try:
                                _sys.path.remove(sp)
                            except ValueError:
                                pass
                        if prev_cache is None:
                            _os.environ.pop("VBENCH2_CACHE_DIR", None)
                        else:
                            _os.environ["VBENCH2_CACHE_DIR"] = prev_cache
                    results.extend(self._collect_dim_results(model, canonical, Path(out_dir), ws))
        return results

    def _stage_videos(
        self, model: str, dim: str, videos: list[VideoSpec] | None, layout: WorkspaceLayout,
    ) -> Path:
        staged = layout.frames_cache_dir / f"_vbench2_stage_{model}_{dim}"
        if staged.exists():
            shutil.rmtree(staged)
        staged.mkdir(parents=True)
        sources = videos or [
            VideoSpec(path=p, prompt_id=p.stem.split("-")[0], model_name=model)
            for p in (layout.videos_dir / model).glob("*.mp4")
        ]
        for v in sources:
            if v.model_name != model:
                continue
            link = staged / v.path.name
            if not link.exists():
                try:
                    link.symlink_to(v.path.resolve())
                except OSError:
                    shutil.copy2(v.path, link)
        return staged

    def _collect_dim_results(
        self, model: str, dim: str, vbench_out: Path, ws: Any,
    ) -> list[RawResult]:
        """Parse VBench-2.0's per-dim JSONs into RawResults.

        Same 3 shapes as VBench v1; we share the parsing logic.
        """
        from videvalkit.benchmarks.vbench.benchmark import VBenchBenchmark as _V1
        v1 = _V1()
        v1.name = self.name  # so emitted RawResults are tagged 'vbench2'
        results: list[RawResult] = []
        # Only the *_eval_results.json file contains scores; full_info.json
        # is a list of prompt metadata that would trip our dict-based parser.
        for jf in vbench_out.glob("**/*_eval_results.json"):
            try:
                data = json.loads(jf.read_text())
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            for dim_name, payload in data.items():
                if dim_name != dim:
                    continue
                results.extend(v1._parse_dim_payload(model, dim, payload, jf, ws))
        # Re-tag scorer
        for r in results:
            r.meta["scorer"] = "vbench2"
        return results

    # ---- aggregate ----------------------------------------------------------
    def aggregate(
        self,
        raw: list[RawResult],
        aggregator: str = "vbench2_category",
        **kwargs: Any,
    ) -> Summary:
        """Reproduce upstream ``scripts/cal_final_score.py``.

        Upstream flow:
          (1) ``upload_data[dim] = scalar`` — per-dim aggregate input.
          (2) Per category C: ``score_C = sum(upload_data[k] for k in C_LIST) / len(C_LIST)``
              (``get_creativity_score`` / ``get_commonsense_score`` / ...).
          (3) Final = mean of the 5 category scores
              (``get_final_score`` at line 78).

        This means: aggregate *per dim* first (avoiding bias from per-dim
        sample-count differences such as Diversity's 20 samples vs Composition's
        3), then divide by **fixed** ``len(LIST)`` (not by present-dim count).
        """
        from collections import defaultdict
        if not raw:
            raise ValueError("no results")
        bench = raw[0].benchmark
        model = raw[0].model

        # Step 1: per-dim scalar = mean over all (video, sample) entries.
        # Filter sentinel -1 entries (e.g. Human_Clothes uses score=-1 to mark
        # "Q1 failed — single-person check returned 'no person'"; upstream's
        # `compute_human_clothes` at lines 113-120 of human_clothes.py skips
        # these from both numerator and denominator before averaging).
        per_dim_vals: dict[str, list[float]] = defaultdict(list)
        for r in raw:
            if isinstance(r.score, (int, float)) and r.score >= 0:
                per_dim_vals[r.dimension].append(float(r.score))
        per_dim_mean: dict[str, float] = {
            d: sum(vs) / len(vs) for d, vs in per_dim_vals.items() if vs
        }

        # Step 2: per-category mean over the canonical LIST (upstream
        # divides by the LIST length, not the present-dim count).
        # Missing dims contribute 0 to the numerator and still count
        # toward the denominator — this is upstream's behaviour.
        per_cat_mean: dict[str, float] = {}
        for cat, cat_dims in _VBENCH2_CATEGORY_DIMS.items():
            present = [d for d in cat_dims if d in per_dim_mean]
            if not present:
                continue
            # Use len(cat_dims) (LIST length) as denominator per upstream.
            per_cat_mean[cat] = (
                sum(per_dim_mean[d] for d in present) / len(cat_dims)
            )

        # Step 3: mean of the 5 category scores (only categories with ≥1
        # present dim are included; upstream evaluates all 5 in normal
        # leaderboard runs, but we handle partial dim selections gracefully).
        overall_scalar = (
            sum(per_cat_mean.values()) / len(per_cat_mean)
            if per_cat_mean else 0.0
        )

        return Summary(
            benchmark=bench, model=model, per_dimension=per_dim_mean,
            overall={"Total": overall_scalar, **per_cat_mean},
            n_videos=len(raw), n_prompts=len({r.prompt_id for r in raw}),
            aggregator=self.name,
            meta={
                "per_category": per_cat_mean,
                "per_dim_mean":  per_dim_mean,
                "formula": "per-dim mean → per-cat mean over LIST len → mean of 5 cats",
            },
        )

    def export_official(self, summary: Summary, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": summary.model,
            "per_dimension": summary.per_dimension,
            "per_category": summary.meta.get("per_category", {}),
            "Total": summary.overall.get("Total") if isinstance(summary.overall, dict) else summary.overall,
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
