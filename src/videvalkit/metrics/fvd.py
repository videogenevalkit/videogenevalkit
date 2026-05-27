"""FVD — Fréchet Video Distance (paper canonical, I3D-K400).

Per docs/VIDEO_METRICS_DESIGN.md §4.1 (user 2026-05-20 confirmed):
implementation source = stylegan-v port; backbone = I3D-K400.

v0.2 ships:
  * Class shape + registration via SUPPORTED_METRICS
  * Reproducibility plumbing: seed, float64 reduce, deterministic clip sampling
  * Sample-size guard
  * **NotImplementedError on compute() until I3D-K400 ckpt + stylegan-v port
    code lands** [follow-up: feat/fvd-backbone-fetch]

The class shape is stable now so dependents [manifest benchmarks, profile,
training monitor, capability resolver] can wire to it without waiting for
the backbone.
"""

from __future__ import annotations

from pathlib import Path

from videvalkit.core.distribution_metric import (
    BaseDistributionMetric,
    DistributionMetricResult,
)


class FVD(BaseDistributionMetric):
    """Fréchet Video Distance — I3D-K400 features + Fréchet distance.

    Paper: Unterthiner et al. 2018, "Towards Accurate Generative Models of
    Video".
    """

    name = "fvd"
    canonical_backbone = "i3d-k400"
    supported_backbones = ["i3d-k400", "videomae-v2-base", "vjepa-l16"]
    min_recommended_samples = 100
    requires_reference = True

    def compute(
        self,
        gen_videos: list[Path],
        ref_videos: list[Path] | None,
        backbone: str | None = None,
        clip_sampling: dict | None = None,
        seed: int = 42,
        device: str = "auto",
    ) -> DistributionMetricResult:
        # Backbone resolution
        bb = backbone or self.canonical_backbone
        if bb not in self.supported_backbones:
            raise ValueError(
                f"backbone {bb!r} not supported for FVD; "
                f"choose from {self.supported_backbones}"
            )

        # Sample size guard
        n_gen = len(gen_videos)
        warning = self.check_sample_size(n_gen, self.min_recommended_samples)
        if warning and warning.startswith("ERROR:"):
            raise ValueError(warning + " Pass --allow-tiny-sample to override.")

        # Backbone load — currently blocked on checkpoint fetch
        # Follow-up: feat/fvd-backbone-fetch will land the actual I3D weights
        # via `videvalkit/checkpoints` HF dataset.
        raise NotImplementedError(
            f"FVD compute() awaits I3D-K400 checkpoint fetch [v0.2 follow-up]. "
            f"Class shape, registry entry, and downstream wiring are in place; "
            f"actual backbone load is pending. See "
            f"docs/VIDEO_METRICS_DESIGN.md §4.1 and the backbone-fetch issue."
        )
