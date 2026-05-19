"""Physics-IQ adapter — 198 scenarios across 5 physical domains.

Upstream: https://github.com/google-deepmind/physics-IQ-benchmark
Paper:    https://arxiv.org/abs/2501.09038

Paradigm: "given the first N seconds, predict the next 5 seconds." The model
under test is conditioned on either:
  * one ``switch-frame`` (image-to-video mode), or
  * an N-second ``conditioning-video`` (video-to-video mode at 8/16/24/30 fps)
and must produce a 5-second continuation. We then compare the generated
5 s against the upstream real-camera reference at the pixel level.

Metrics (CV-only; no VLM-as-judge):
  * Spatial IoU         — IoU of per-pixel motion regions across the clip.
  * Spatiotemporal IoU  — Spatial IoU extended over time.
  * Weighted MSE        — pixel MSE weighted by motion magnitude (penalises
                          wrong force trajectories).
  * (optional) MLLM-score — semantic veto by a generic VLM judge; disabled
                          by default since the headline Physics-IQ Score in
                          the paper is metric-only.

Final ``Physics-IQ Score`` is a single percentage in [0, 100] aggregated
across the three metrics (weighting hard-coded in upstream
``code/run_physics_iq.py``).
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


PHYSICS_IQ_DIMENSIONS = [
    "solid_mechanics",
    "fluid_dynamics",
    "optics",
    "thermodynamics",
    "magnetism",
]

PHYSICS_IQ_DEFINITIONS = {
    "solid_mechanics": "Rigid-body collisions, friction, gravity-induced falls.",
    "fluid_dynamics":  "Liquid flow, splashing, viscosity, surface tension.",
    "optics":          "Reflection, refraction, shadows, caustics, lensing.",
    "thermodynamics":  "Phase transitions (melt / freeze / sublimation), heat propagation.",
    "magnetism":       "Attraction / repulsion of magnetic objects.",
}

PHYSICS_IQ_METRICS = ("spatial_iou", "spatiotemporal_iou", "weighted_mse")

# Upstream specifies a single percentage score per (model, scenario, take).
# Output videos must be trimmed to EXACTLY 5 seconds at the chosen fps.
PHYSICS_IQ_OUTPUT_DURATION_S: float = 5.0
PHYSICS_IQ_REFERENCE_FPS: int = 30
PHYSICS_IQ_REFERENCE_RESOLUTION: tuple[int, int] = (3840, 2160)


class PhysicsIQBenchmark(BaseBenchmark):
    name = "physics_iq"
    env_name = "videvalkit"
    dimensions = PHYSICS_IQ_DIMENSIONS

    # Each scenario is filmed from 3 angles × 2 takes → 6 reference variants.
    # The adapter expects generated videos under the toolkit's standard layout,
    # one per scenario × take. Camera angle is in meta.
    video_layout = "{model}/{prompt_id}-{sample_index}.mp4"
    NUM_SCENARIOS: int = 198
    SAMPLES_PER_SCENARIO: int = 6   # 3 angles × 2 takes; the adapter picks the best
    MODE_DEFAULTS: tuple[str, ...] = ("i2v", "v2v")

    # ---- prompts ------------------------------------------------------------
    def list_prompts(
        self,
        dimensions: list[str] | None = None,
        scenarios_file: str | Path | None = None,
        mode: str = "i2v",
    ) -> Iterator[PromptItem]:
        """Yield one PromptItem per scenario.

        Physics-IQ ships ``descriptions.csv`` listing all 198 scenarios with
        their domain label. We expect a normalised JSONL passthrough at
        ``$WORKSPACE/prompts/physics_iq/scenarios.jsonl`` with shape::

            {"prompt_id": "0001", "domain": "fluid_dynamics", "text": "...",
             "conditioning_path": "switch-frames/0001.png"}
        """
        wanted = set(dimensions) if dimensions else set(self.dimensions)
        if scenarios_file is None:
            # Smoke-test fallback: one example per domain.
            for i, d in enumerate(PHYSICS_IQ_DIMENSIONS):
                if d not in wanted:
                    continue
                yield PromptItem(
                    prompt_id=f"{d[:3]}_{i:04d}",
                    text=f"[{d}] reference scenario {i}",
                    dimensions=[d],
                    meta={"mode": mode, "domain": d},
                )
            return
        with Path(scenarios_file).open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                d = e.get("domain")
                if d not in wanted:
                    continue
                yield PromptItem(
                    prompt_id=str(e["prompt_id"]),
                    text=e.get("text", ""),
                    dimensions=[d],
                    meta={
                        "mode": mode,
                        "domain": d,
                        "conditioning_path": e.get("conditioning_path"),
                    },
                )

    def list_required_videos(
        self,
        prompts: list[PromptItem],
        models: list[str],
        layout: WorkspaceLayout,
        samples_per_prompt: int | None = None,
    ) -> list[VideoSpec]:
        n = samples_per_prompt if samples_per_prompt is not None else self.SAMPLES_PER_SCENARIO
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
                        meta={"take": idx % 2, "camera_angle": idx // 2},
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
        reference_root: str | Path | None = None,
        **kwargs: Any,
    ) -> list[RawResult]:
        """Run upstream ``code/run_physics_iq.py`` on the generated clips.

        Wiring sketch:
          1. Build a temporary dir with the generated 5-second clips renamed
             to upstream's expected filenames.
          2. Subprocess into the upstream script with
             ``--input_folders <our staging> --output_folder <ws> --descriptions_file <csv>``.
          3. Parse the upstream JSON output (a per-(scenario, metric) table)
             into RawResult, one per (model, domain, scenario, metric).
        """
        raise NotImplementedError(
            "PhysicsIQBenchmark.evaluate() is a stub. To enable: "
            "clone https://github.com/google-deepmind/physics-IQ-benchmark, "
            "run `code/download_physics_iq_data.py`, point `reference_root` "
            "at the downloaded dataset, and wire a subprocess call to "
            "`code/run_physics_iq.py`."
        )

    # ---- aggregate ----------------------------------------------------------
    def aggregate(
        self,
        raw: list[RawResult],
        aggregator: str = "weighted_sum",
        **kwargs: Any,
    ) -> Summary:
        """Single-number aggregation: mean across scenarios per domain, then
        cross-domain mean. Final number is a percentage in [0, 100] per
        the upstream convention; the toolkit stores it as a float here."""
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
            n_videos=len({(r.model, r.prompt_id) for r in raw}),
            n_prompts=len({r.prompt_id for r in raw}),
            aggregator=aggregator,
            meta={
                "scale": "0-100 (percentage of physical principles correctly modelled)",
                "metrics": list(PHYSICS_IQ_METRICS),
            },
        )
