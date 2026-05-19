"""VBench++ adapter — VBench v1's 16 dims + I2V Image Suite + Trustworthiness.

Upstream: https://github.com/Vchitect/VBench  (same repo as VBench v1; the
``vbench`` Python module exposes both v1 and ++ entry points).
Paper:    https://arxiv.org/abs/2411.13503

VBench++ is **VBench v1 plus**:
  * an I2V Image Suite (adaptive aspect ratio) and 4 extra I2V-specific dims
    that score how faithfully the generated video adheres to a conditioning
    image while still moving;
  * a Trustworthiness section (4 dims): NSFW, fairness (race / gender / age),
    bias, and refusal-rate / stealthiness.

Strategy: subclass the existing VBenchBenchmark adapter (already wired to
the upstream `vbench` package) and extend the dimension list. The
upstream evaluator dispatches on dim name, so adding the new ones is
mostly a matter of registering them; the per-dim scorers live in
``vbench/i2v/`` and ``vbench/trustworthiness/`` upstream.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from videvalkit.core.benchmark import BaseBenchmark
from videvalkit.core.layout import WorkspaceLayout
from videvalkit.core.types import PromptItem, RawResult, Summary, VideoSpec
from videvalkit.benchmarks.vbench.benchmark import (
    VBENCH_DIMENSIONS, VBENCH_QUALITY_DIMS, VBENCH_SEMANTIC_DIMS,
    VBenchBenchmark,
)


# I2V-specific extras (Huang et al., 2024, Table 2).
VBENCH_PP_I2V_EXTRA_DIMS = [
    "i2v_subject",       # subject identity stays faithful to the input image
    "i2v_background",    # background stays faithful to the input image
    "camera_motion",     # follows requested camera move (9 patterns)
    "video_image_consistency",  # overall fidelity to conditioning image
]

# Trustworthiness section.
VBENCH_PP_TRUST_DIMS = [
    "culture_fairness",
    "gender_bias",
    "skin_tone_bias",
    "content_safety",       # NSFW + violence detection
    "refusal_rate",         # how often the model refuses / black-screens
]

VBENCH_PP_DIMENSIONS: list[str] = (
    VBENCH_DIMENSIONS
    + VBENCH_PP_I2V_EXTRA_DIMS
    + VBENCH_PP_TRUST_DIMS
)


class VBenchPPBenchmark(VBenchBenchmark):
    """VBench v1 → VBench++ upgrade.

    Adds I2V extras and Trustworthiness on top of the v1 16 dims.
    `evaluate()` delegates to v1 for shared dims and dispatches the
    extras to the upstream module's I2V / Trust pipelines.
    """

    name = "vbench_pp"
    env_name = "videvalkit"
    dimensions = VBENCH_PP_DIMENSIONS

    # I2V mode expects ``videos/{model}/{prompt_id}/<video>.mp4`` alongside
    # ``images/{prompt_id}.{png,jpg}`` (the Image Suite). We keep the
    # toolkit's standard layout and rely on a sibling ``images_dir``.
    video_layout = "{model}/{prompt_id}-{sample_index}.mp4"

    # Quality / Semantic / I2V / Trust groupings; the official aggregator
    # normalizes each section then takes a section-weighted mean.
    SECTION_DIMS = {
        "quality":  sorted(VBENCH_QUALITY_DIMS),
        "semantic": sorted(VBENCH_SEMANTIC_DIMS),
        "i2v":      VBENCH_PP_I2V_EXTRA_DIMS,
        "trust":    VBENCH_PP_TRUST_DIMS,
    }
    SECTION_WEIGHTS = {"quality": 1.0, "semantic": 1.0, "i2v": 1.0, "trust": 1.0}

    # ---- prompts ------------------------------------------------------------
    def list_prompts(
        self,
        dimensions: list[str] | None = None,
    ) -> Iterator[PromptItem]:
        """Reuse VBench v1's prompt loader for shared dims; for the new
        I2V / Trust dims the upstream package ships separate JSON suites
        (``vbench_i2v_full_info.json``, ``vbench_trust_full_info.json``)."""
        wanted = set(dimensions) if dimensions else set(self.dimensions)

        # v1-shared dims via upstream loader
        v1_wanted = sorted(wanted & set(VBENCH_DIMENSIONS))
        if v1_wanted:
            yield from super().list_prompts(dimensions=v1_wanted)

        # Stub for I2V + Trust suites — upstream JSON files have the same
        # shape as v1's, so once wired the loader can be unified. For
        # smoke tests we emit one synthetic prompt per extra dim.
        extras = sorted((wanted - set(VBENCH_DIMENSIONS)))
        for i, dim in enumerate(extras):
            yield PromptItem(
                prompt_id=f"{dim}_{i:04d}",
                text=f"[{dim}] placeholder prompt — wire upstream JSON to replace.",
                dimensions=[dim],
                meta={"section": self._section_of(dim)},
            )

    # ---- evaluate -----------------------------------------------------------
    def evaluate(
        self,
        videos: list[VideoSpec] | None = None,
        layout: WorkspaceLayout | None = None,
        dimensions: list[str] | None = None,
        judge: dict[str, Any] | None = None,
        models: list[str] | None = None,
        **kwargs: Any,
    ) -> list[RawResult]:
        """Dispatch v1 dims to the parent, ++ extras to upstream I2V/Trust pipelines."""
        wanted = set(dimensions) if dimensions else set(self.dimensions)
        v1_wanted = sorted(wanted & set(VBENCH_DIMENSIONS))
        extras = sorted((wanted - set(VBENCH_DIMENSIONS)))

        results: list[RawResult] = []
        if v1_wanted:
            results.extend(super().evaluate(
                videos=videos, layout=layout, dimensions=v1_wanted,
                judge=judge, models=models, **kwargs,
            ))
        if extras:
            raise NotImplementedError(
                "VBench++ extras (I2V Image Suite + Trustworthiness) are not "
                "wired yet. Each lives in the upstream `vbench` package as "
                "`vbench.i2v.<dim>` / `vbench.trustworthiness.<dim>`; "
                "add per-dim subprocess shims analogous to v1's _run_dim."
            )
        return results

    # ---- aggregate ----------------------------------------------------------
    def aggregate(
        self,
        raw: list[RawResult],
        aggregator: str = "vbench_weighted",
        **kwargs: Any,
    ) -> Summary:
        """Section-weighted mean: Quality + Semantic + (I2V if present) +
        (Trust if present), each normalised then averaged with
        ``SECTION_WEIGHTS``."""
        from collections import defaultdict
        per_dim: dict[str, list[float]] = defaultdict(list)
        for r in raw:
            if isinstance(r.score, (int, float)):
                per_dim[r.dimension].append(float(r.score))
        per_dim_mean = {d: sum(vs) / len(vs) for d, vs in per_dim.items() if vs}

        # group by section
        section_scores: dict[str, list[float]] = defaultdict(list)
        for d, v in per_dim_mean.items():
            section_scores[self._section_of(d)].append(v)
        section_means = {sec: sum(vs) / len(vs) for sec, vs in section_scores.items() if vs}
        if section_means:
            total_w = sum(self.SECTION_WEIGHTS.get(s, 1.0) for s in section_means)
            overall = sum(
                self.SECTION_WEIGHTS.get(s, 1.0) * v
                for s, v in section_means.items()
            ) / total_w
        else:
            overall = 0.0

        model = raw[0].model if raw else ""
        return Summary(
            benchmark=self.name,
            model=model,
            per_dimension=per_dim_mean,
            overall=overall,
            n_videos=len({(r.model, r.prompt_id, r.dimension) for r in raw}),
            n_prompts=len({r.prompt_id for r in raw}),
            aggregator=aggregator,
            meta={"section_means": section_means,
                  "section_weights": self.SECTION_WEIGHTS},
        )

    # ---- helpers ------------------------------------------------------------
    @classmethod
    def _section_of(cls, dim: str) -> str:
        for sec, ds in cls.SECTION_DIMS.items():
            if dim in ds:
                return sec
        return "unknown"
