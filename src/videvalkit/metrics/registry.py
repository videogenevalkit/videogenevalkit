"""Standalone metric registry — SUPPORTED_METRICS.

Per docs/VIDEO_METRICS_DESIGN.md §6 (user 2026-05-20 confirmed):

v0.2 ships 20 metrics across two tiers:

  通用 T2V quality (14):
    Distribution (4):   fvd / vfid / kvd / clip-fvd*
    Alignment   (2):    clip-score / viclip-score
    Frame perc. (2):    aesthetic-quality / imaging-quality   [lift]
    Temporal    (6):    motion-smoothness / temporal-flickering /
                        subject-consistency / background-consistency /
                        dynamic-degree / motion-magnitude     [lift]
  专用维度 (6):
    object-binding / spatial-relationship / numeracy /
    motion-accuracy / identity-preservation / artifact-diagnostic

This PR introduces the registry SHELL and the 4 distribution-level entries.
Per-prompt entries land in feat/per-prompt-metrics (#9). Lift-outs land in
feat/metric-lift-outs (#10). Specialized in feat/metric-specialized (#11).

Each entry's ``kind`` field tells the runner how to dispatch:
  * distribution_reference   — gen + ref videos, single overall score
  * per_video_reference_free — videos only, scalar per video
  * per_prompt_reference_free — videos + prompts, scalar per pair
  * per_video_with_ref_image  — videos + ref image
  * per_video_with_vlm_judge  — videos + judge, per-video labels

The ``tags`` field references CAPABILITY_TAGS controlled vocab.
"""

from __future__ import annotations

from typing import Any


SUPPORTED_METRICS: dict[str, dict[str, Any]] = {
    # ============================================================
    # Distribution-level metrics (4) — all judge-free
    # ============================================================
    "fvd": dict(
        kind="distribution_reference",
        source="canonical/stylegan-v-port",
        cls="videvalkit.metrics.fvd:FVD",
        canonical_backbone="i3d-k400",
        supported_backbones=["i3d-k400", "videomae-v2-base", "vjepa-l16"],
        min_recommended_samples=100,
        paper_recommended_samples=2048,
        inputs=["gen_videos", "ref_videos"],
        output_kind="scalar_overall",
        needs_judge=False,
        compute_kind="local_vision",
        tags=["realism.distribution"],
        algorithm="Fréchet distance on I3D-K400 features (paper canonical)",
        paper_alignment_test="tests/test_fvd_paper_alignment.py",
        license="Apache-2.0",
        version="1.0",
    ),
    "vfid": dict(
        kind="distribution_reference",
        source="canonical/new",
        cls="videvalkit.metrics.vfid:VFID",
        canonical_backbone="inception-v3",
        supported_backbones=["inception-v3"],
        min_recommended_samples=100,
        paper_recommended_samples=2048,
        inputs=["gen_videos", "ref_videos"],
        output_kind="scalar_overall",
        needs_judge=False,
        compute_kind="local_vision",
        tags=["realism.distribution"],
        algorithm="Fréchet distance on InceptionV3 per-frame mean-pool features",
        paper_alignment_test="tests/test_vfid_paper_alignment.py",
        license="Apache-2.0",
        version="1.0",
    ),
    "kvd": dict(
        kind="distribution_reference",
        source="canonical/new",
        cls="videvalkit.metrics.kvd:KVD",
        canonical_backbone="i3d-k400",
        supported_backbones=["i3d-k400"],
        min_recommended_samples=50,
        paper_recommended_samples=500,
        inputs=["gen_videos", "ref_videos"],
        output_kind="scalar_overall",
        needs_judge=False,
        compute_kind="local_vision",
        tags=["realism.distribution"],
        algorithm="Polynomial-kernel MMD² on I3D-K400 features (Bińkowski et al. 2018)",
        paper_alignment_test="tests/test_kvd_paper_alignment.py",
        license="Apache-2.0",
        version="1.0",
        notes="More stable than FVD at small N (<500).",
    ),
    "clip-fvd": dict(
        kind="distribution_reference",
        source="canonical/new",
        cls="videvalkit.metrics.clip_fvd:CLIPFVD",
        canonical_backbone="clip-vit-l14",
        supported_backbones=["clip-vit-l14", "clip-vit-b16"],
        min_recommended_samples=100,
        paper_recommended_samples=2048,
        inputs=["gen_videos", "ref_videos"],
        output_kind="scalar_overall",
        needs_judge=False,
        compute_kind="local_vision",
        tags=["realism.distribution"],
        algorithm="Fréchet distance on CLIP-ViT-L/14 frame features",
        paper_alignment_test=None,  # experimental
        license="MIT",
        version="0.1",
        experimental=True,
        notes="Experimental: not comparable to FVD (different feature space).",
    ),

    # ============================================================
    # Text-video alignment (2 of 14) — judge-free, needs CLIP/ViCLIP backbone
    # ============================================================
    "clip-score": dict(
        kind="per_prompt_reference_free",
        source="canonical/new",
        cls="videvalkit.metrics.clip_score:CLIPScore",
        canonical_backbone="clip-vit-l14",
        supported_backbones=["clip-vit-l14", "clip-vit-b16"],
        inputs=["videos", "prompts"],
        output_kind="scalar_per_pair",
        needs_judge=False,
        compute_kind="local_vision",
        tags=["align.text2video"],
        algorithm="Per-frame CLIP-ViT-L/14 image-text cosine, averaged across frames",
        paper_alignment_test=None,
        license="MIT (openai-clip)",
        version="1.0",
    ),
    "viclip-score": dict(
        kind="per_prompt_reference_free",
        source="canonical/new",
        cls="videvalkit.metrics.viclip_score:ViCLIPScore",
        canonical_backbone="viclip-l",
        supported_backbones=["viclip-l"],
        inputs=["videos", "prompts"],
        output_kind="scalar_per_pair",
        needs_judge=False,
        compute_kind="local_vision",
        tags=["align.text2video", "align.prompt_following"],
        algorithm="Per-clip ViCLIP video-text cosine over 16-frame clip",
        paper_alignment_test=None,
        license="MIT",
        version="0.1",
        notes="ViCLIP trained on video-text pairs; more accurate than per-frame CLIP for T2V.",
    ),

}
