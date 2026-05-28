"""Lift-out metrics for VBench quality-axis dimensions.

Per docs/VIDEO_METRICS_DESIGN.md §3.3-§3.4 + §5 (user 2026-05-20 confirmed
重集成 = lift existing bench dims into standalone metrics).

Each metric here wraps the SAME upstream ``VBench(...).evaluate(
dimension_list=[dim], mode="custom_input")`` call that the vbench benchmark
adapter uses. Bit-exact contract [VIDEO_METRICS_DESIGN §5.3] holds because
BOTH paths invoke identical deterministic upstream code with identical params:

    videvalkit eval --bench vbench --dimensions motion_smoothness
                          ==  [≤ 1e-6]  ==
    videvalkit metric --name motion-smoothness

The 7 quality-axis dims work in custom_input mode [no prompts needed]:
  subject_consistency / background_consistency / temporal_flickering /
  motion_smoothness / dynamic_degree / aesthetic_quality / imaging_quality

These are per-video reference-free metrics [need only videos + GPU + the
vbench checkpoints, no judge].
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


# Short-name → upstream vbench dim name
VBENCH_DIM_BY_METRIC: dict[str, str] = {
    "subject-consistency":    "subject_consistency",
    "background-consistency": "background_consistency",
    "temporal-flickering":    "temporal_flickering",
    "motion-smoothness":      "motion_smoothness",
    "dynamic-degree":         "dynamic_degree",
    "aesthetic-quality":      "aesthetic_quality",
    "imaging-quality":        "imaging_quality",
}


class VBenchDimResult(BaseModel):
    metric: str
    dim: str
    score: float                       # mean across videos
    per_video: dict[str, float] = Field(default_factory=dict)
    n_videos: int
    source: str = "vbench"


class VBenchDimMetric:
    """Standalone wrapper over a single VBench quality-axis dimension.

    Instantiated per-dim; ``self.dim`` is the upstream dim name. The compute()
    method runs upstream VBench on the videos dir, restricted to that one dim.
    """

    name = "vbench-dim-base"
    requires_judge = False

    def __init__(self, metric_name: str | None = None):
        # Allow either subclass-set name or explicit metric_name
        n = metric_name or getattr(self, "name", None)
        if n not in VBENCH_DIM_BY_METRIC:
            raise ValueError(
                f"VBenchDimMetric {n!r} is not a known vbench lift; "
                f"choose from {sorted(VBENCH_DIM_BY_METRIC)}"
            )
        self.name = n
        self.dim = VBENCH_DIM_BY_METRIC[n]

    def compute(
        self,
        videos: list[Path] | Path,
        device: str = "auto",
        full_info_path: str | Path | None = None,
    ) -> VBenchDimResult:
        """Run upstream VBench on one dim for the given videos.

        ``videos`` may be a directory OR a list of paths [staged into a temp
        dir]. Returns per-video + mean scores.

        Bit-exact with the bench adapter: same VBench(...).evaluate() call.
        """
        try:
            from vbench import VBench
        except ImportError as e:
            raise RuntimeError(
                "vbench not importable — install in the video-eval / videvalkit "
                "env. This lift metric wraps upstream VBench."
            ) from e

        # Resolve device
        if device == "auto":
            from videvalkit.core.device import get_device
            device = get_device().type if _has_device_module() else _fallback_device()

        # Normalize videos → a directory upstream can read
        videos_dir = self._stage_videos(videos)

        # Resolve full_info_dir [upstream needs it even for custom_input]
        if full_info_path is None:
            import importlib
            vbench_mod = importlib.import_module("vbench")
            full_info_path = (
                Path(vbench_mod.__file__).parent / "VBench_full_info.json"
            )

        with tempfile.TemporaryDirectory() as out_dir:
            vb = VBench(
                output_path=out_dir,
                full_info_dir=str(full_info_path),
                device=device,
            )
            vb.evaluate(
                videos_path=str(videos_dir),
                name=f"metric_{self.name}",
                dimension_list=[self.dim],
                mode="custom_input",
            )
            per_video, mean = self._parse_output(Path(out_dir))

        return VBenchDimResult(
            metric=self.name,
            dim=self.dim,
            score=mean,
            per_video=per_video,
            n_videos=len(per_video),
        )

    def _stage_videos(self, videos: list[Path] | Path) -> Path:
        if isinstance(videos, (str, Path)):
            return Path(videos)
        # list of paths → symlink into a temp dir
        staged = Path(tempfile.mkdtemp(prefix=f"vbench_{self.name}_"))
        for v in videos:
            link = staged / Path(v).name
            if not link.exists():
                link.symlink_to(Path(v).resolve())
        return staged

    def _parse_output(self, out_dir: Path) -> tuple[dict[str, float], float]:
        """Parse VBench's eval_results JSON for this dim into per-video + mean."""
        import json

        # upstream writes <name>_eval_results.json
        candidates = sorted(out_dir.glob("*_eval_results.json"))
        if not candidates:
            log.warning("vbench lift %s: no eval_results json in %s",
                        self.name, out_dir)
            return {}, 0.0
        data = json.loads(candidates[0].read_text())
        # Schema: {dim: [mean_score, [ {video_path, video_results}, ... ]]}
        if self.dim not in data:
            return {}, 0.0
        entry = data[self.dim]
        mean = float(entry[0]) if isinstance(entry, list) and entry else 0.0
        per_video: dict[str, float] = {}
        if isinstance(entry, list) and len(entry) > 1 and isinstance(entry[1], list):
            for rec in entry[1]:
                if isinstance(rec, dict):
                    vp = rec.get("video_path", "")
                    vr = rec.get("video_results")
                    if isinstance(vr, (int, float)):
                        per_video[str(vp)] = float(vr)
        return per_video, mean


def _has_device_module() -> bool:
    try:
        import videvalkit.core.device  # noqa
        return True
    except ImportError:
        return False


def _fallback_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# Concrete per-dim classes [so registry cls paths resolve to distinct types].
def _make_dim_class(metric_name: str) -> type:
    cls_name = "".join(w.capitalize() for w in metric_name.split("-"))

    def __init__(self):
        VBenchDimMetric.__init__(self, metric_name=metric_name)

    return type(cls_name, (VBenchDimMetric,), {"name": metric_name, "__init__": __init__})


SubjectConsistency    = _make_dim_class("subject-consistency")
BackgroundConsistency = _make_dim_class("background-consistency")
TemporalFlickering    = _make_dim_class("temporal-flickering")
MotionSmoothness      = _make_dim_class("motion-smoothness")
DynamicDegree         = _make_dim_class("dynamic-degree")
AestheticQuality      = _make_dim_class("aesthetic-quality")
ImagingQuality        = _make_dim_class("imaging-quality")
