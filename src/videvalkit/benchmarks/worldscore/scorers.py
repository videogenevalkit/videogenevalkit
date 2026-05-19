"""WorldScore scorers — thin wrappers around upstream metric classes.

Following the upstream WorldScore code base
(``worldscore/benchmark/metrics/``), each scorer here delegates to the
corresponding upstream metric class. No proxy implementations.

Upstream metric -> dim:
  CLIPScoreMetric (torchmetrics)              -> content_alignment
  ObjectDetectionMetric (GroundingDINO)       -> object_control
  OpticalFlowAverageEndPointErrorMetric (AEPE)-> photometric_consistency
  OpticalFlowMetric (SEA-RAFT median flow)    -> motion_magnitude
  GramMatrixMetric (VGG-19, WS reference img) -> style_consistency
  IQACLIPAestheticScoreMetric (pyiqa laion_aes) AND
  CLIPImageQualityAssessmentPlusMetric        -> subjective_quality (averaged)
  ReprojectionErrorMetric (DROID-SLAM)        -> 3d_consistency
  CameraErrorMetric (DROID-SLAM)              -> camera_control
  MotionSmoothnessMetric (VFIMamba+SSIM+LPIPS)-> motion_smoothness
  MotionAccuracyMetric (SEA-RAFT + SAM2)      -> motion_accuracy

Sub-sample: 49 frames per video (matches upstream's interpframe_num).
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np

# Upstream WorldScore repo path (read-only source). Each runner must chdir to
# this directory so upstream's relative-path checkpoints/configs resolve.
# Populated by ``videvalkit fetch-upstream --bench worldscore``.
WS_ROOT = os.environ.get(
    "VIDEVALKIT_WORLDSCORE_ROOT",
    str(Path.home() / ".cache" / "videvalkit" / "upstream" / "WorldScore"),
)


def install_mamba_ssm_shim() -> None:
    """Inject a pure-PyTorch mamba_ssm replacement before VFIMamba imports.

    VFIMamba's feature_extractor imports ``selective_scan_fn`` from
    ``mamba_ssm.ops.selective_scan_interface``. Installing real mamba_ssm
    requires compiling CUDA kernels and would upgrade torch beyond our pinned
    2.3.1+cu121, breaking lietorch / droid_backends. This shim provides the
    same algorithm in pure PyTorch (slower; correct).
    """
    if "mamba_ssm.ops.selective_scan_interface" in sys.modules:
        return

    import torch
    import torch.nn.functional as F
    from einops import rearrange, repeat

    def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                           delta_softplus=False, return_last_state=False):
        dtype_in = u.dtype
        u = u.float(); delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float(); C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "b g n l -> b (g h) n l", h=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "b g n l -> b (g h) n l", h=dim // C.shape[1])
        last_state = None
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2)
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        return out if not return_last_state else (out, last_state)

    root = types.ModuleType("mamba_ssm")
    ops = types.ModuleType("mamba_ssm.ops")
    iface = types.ModuleType("mamba_ssm.ops.selective_scan_interface")
    iface.selective_scan_fn = selective_scan_ref
    iface.selective_scan_ref = selective_scan_ref
    sys.modules["mamba_ssm"] = root
    sys.modules["mamba_ssm.ops"] = ops
    sys.modules["mamba_ssm.ops.selective_scan_interface"] = iface


def setup_upstream_paths() -> None:
    """Add upstream WorldScore + DROID-SLAM + SEA-RAFT + VFIMamba to sys.path,
    chdir to WS_ROOT so relative checkpoint paths resolve."""
    install_mamba_ssm_shim()
    paths = [
        WS_ROOT,
        f"{WS_ROOT}/worldscore/benchmark/metrics/third_party",
        f"{WS_ROOT}/worldscore/benchmark/metrics/third_party/droid_slam",  # patched terminate()
        f"{WS_ROOT}/worldscore/benchmark/metrics/third_party/SEA-RAFT",
        f"{WS_ROOT}/worldscore/benchmark/metrics/third_party/VFIMamba",
    ]
    for p in paths:
        if p not in sys.path:
            sys.path.insert(0, p)
    if Path(WS_ROOT).is_dir() and os.getcwd() != WS_ROOT:
        os.chdir(WS_ROOT)


def extract_frames_to_disk(video_path: str | Path, tmp_dir: str | Path, n: int = 49) -> list[str]:
    """Sample n evenly-spaced frames to PNG, return paths.

    Upstream's interpframe_num is 49 (49 frames per scene). Use the same here.
    """
    import imageio.v3 as iio
    from PIL import Image
    arr = iio.imread(str(video_path), plugin="pyav")
    if arr.shape[0] == 0:
        return []
    idx = np.linspace(0, arr.shape[0] - 1, min(n, arr.shape[0])).astype(int).tolist()
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    # Integer-only stems (e.g. "000.jpg") — SAM2's video loader sorts frames
    # by int(stem) so prefixed names like "f000" break motion_accuracy.
    for i, j in enumerate(idx):
        p = Path(tmp_dir) / f"{i:03d}.jpg"
        Image.fromarray(arr[j]).save(p, quality=95)
        paths.append(str(p))
    return paths


# =========================================================================
# Per-dim scorer wrappers. Each lazy-loads its upstream class on first use.
# =========================================================================

class _UpstreamScorerBase:
    """Mixin: lazy-load an upstream metric class once per process."""
    upstream_class_path: tuple[str, str] = ()

    def __init__(self) -> None:
        setup_upstream_paths()
        self._metric = None

    def _load(self):
        if self._metric is None:
            mod, cls = self.upstream_class_path
            from importlib import import_module
            self._metric = getattr(import_module(mod), cls)()
        return self._metric


class CLIPScoreScorer(_UpstreamScorerBase):
    """content_alignment — torchmetrics CLIPScore(openai/clip-vit-base-patch16)."""
    upstream_class_path = ("worldscore.benchmark.metrics", "CLIPScoreMetric")

    def score(self, frame_paths: list[str], prompt: str) -> float:
        return float(self._load()._compute_scores(frame_paths, prompt))


class ObjectDetectionScorer(_UpstreamScorerBase):
    """object_control — GroundingDINO with spaCy class-match counting."""
    upstream_class_path = ("worldscore.benchmark.metrics", "ObjectDetectionMetric")

    def score(self, frame_paths: list[str], text_prompt: str) -> float:
        """text_prompt should follow upstream's "scene_name, obj1, obj2" form;
        upstream drops the first comma-separated chunk."""
        return float(self._load()._compute_scores(frame_paths, text_prompt))


class OpticalFlowAEPEScorer(_UpstreamScorerBase):
    """photometric_consistency — SEA-RAFT bidir AEPE."""
    upstream_class_path = ("worldscore.benchmark.metrics", "OpticalFlowAverageEndPointErrorMetric")

    def score(self, frame_paths: list[str]) -> float:
        return float(self._load()._compute_scores(frame_paths))


class OpticalFlowScorer(_UpstreamScorerBase):
    """motion_magnitude — SEA-RAFT median flow."""
    upstream_class_path = ("worldscore.benchmark.metrics", "OpticalFlowMetric")

    def score(self, frame_paths: list[str]) -> float:
        return float(self._load()._compute_scores(frame_paths))


class GramMatrixScorer(_UpstreamScorerBase):
    """style_consistency — VGG-19 gram matrices, requires WorldScore reference image."""
    upstream_class_path = ("worldscore.benchmark.metrics", "GramMatrixMetric")

    def score(self, frame_paths: list[str], reference_image: str) -> float:
        return float(self._load()._compute_scores(reference_image, frame_paths))


class CLIPAestheticScorer(_UpstreamScorerBase):
    """subjective_quality / clip_aesthetic component — pyiqa laion_aes."""
    upstream_class_path = ("worldscore.benchmark.metrics", "IQACLIPAestheticScoreMetric")

    def score(self, frame_paths: list[str]) -> float:
        return float(self._load()._compute_scores(frame_paths))


class CLIPIQAPlusScorer(_UpstreamScorerBase):
    """subjective_quality / clip_iqa+ component — pyiqa clipiqa+."""
    upstream_class_path = ("worldscore.benchmark.metrics", "CLIPImageQualityAssessmentPlusMetric")

    def score(self, frame_paths: list[str]) -> float:
        return float(self._load()._compute_scores(frame_paths))


class ReprojectionErrorScorer(_UpstreamScorerBase):
    """3d_consistency — DROID-SLAM reprojection error."""
    upstream_class_path = ("worldscore.benchmark.metrics", "ReprojectionErrorMetric")

    def __init__(self) -> None:
        super().__init__()
        # Per-video init: m.droid = None to force a fresh SLAM instance per call.

    def score(self, frame_paths: list[str]) -> float:
        m = self._load()
        m._args.disable_vis = True
        out = float(m._compute_scores(frame_paths))
        m.droid = None
        return out


class CameraErrorScorer(_UpstreamScorerBase):
    """camera_control — DROID-SLAM camera trajectory vs GT cameras."""
    upstream_class_path = ("worldscore.benchmark.metrics", "CameraErrorMetric")

    def __init__(self) -> None:
        super().__init__()

    def score(self, frame_paths: list[str], cameras_gt) -> tuple[float, float]:
        """Returns (R_err_deg, T_err)."""
        m = self._load()
        m._args.disable_vis = True
        err = m._compute_scores(rendered_images=frame_paths, cameras_gt=cameras_gt)
        m.droid = None
        if hasattr(err, "__len__") and len(err) == 2:
            return float(err[0]), float(err[1])
        return float(err), float("nan")


class MotionSmoothnessScorer(_UpstreamScorerBase):
    """motion_smoothness — VFIMamba interpolation + SSIM + LPIPS + MSE triple."""
    upstream_class_path = ("worldscore.benchmark.metrics", "MotionSmoothnessMetric")

    def score(self, frame_paths: list[str]) -> tuple[float, float, float]:
        """Returns (MSE, SSIM, LPIPS)."""
        out = self._load()._compute_scores(frame_paths)
        return float(out[0]), float(out[1]), float(out[2])


class MotionAccuracyScorer:
    """motion_accuracy — GroundingDINO + SAM-ViT-H + SAM2 + SEA-RAFT.

    Not derived from a single upstream metric class because the upstream class
    requires pre-rendered first-frame masks; for T2V we must generate them on
    the fly. This wrapper drives the full per-video pipeline using the same
    upstream components and the same spaCy class-match rate calc.
    """
    def __init__(self) -> None:
        setup_upstream_paths()
        self._sam2_predictor = None
        self._gdino = None
        self._sam_predictor = None
        self._raft_model = None
        self._raft_args = None

    def _load(self) -> None:
        if self._sam2_predictor is not None:
            return
        from sam2.build_sam import build_sam2_video_predictor
        from segment_anything import sam_model_registry, SamPredictor
        from groundingdino.models import build_model
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.util.utils import clean_state_dict
        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from core.raft import RAFT
        from core.utils.utils import load_ckpt
        from core.parser import parse_args as raft_parse_args
        import argparse
        import torch

        GlobalHydra.instance().clear()
        initialize_config_dir(
            config_dir=f"{WS_ROOT}/worldscore/benchmark/metrics/third_party",
            version_base=None,
        )
        self._sam2_predictor = build_sam2_video_predictor(
            config_file="sam2/configs/sam2.1/sam2.1_hiera_b+.yaml",
            ckpt_path=f"{WS_ROOT}/worldscore/benchmark/metrics/checkpoints/sam2.1_hiera_base_plus.pt",
            apply_postprocessing=False,
            hydra_overrides_extra=["++model.non_overlap_masks=false"],
        )
        args = SLConfig.fromfile(
            f"{WS_ROOT}/worldscore/benchmark/metrics/third_party/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        )
        args.device = "cuda"; args.bert_base_uncased_path = None
        self._gdino = build_model(args)
        sd = torch.load(
            f"{WS_ROOT}/worldscore/benchmark/metrics/checkpoints/groundingdino_swint_ogc.pth",
            map_location="cpu",
        )
        self._gdino.load_state_dict(clean_state_dict(sd["model"]), strict=False)
        self._gdino.eval().to("cuda")

        sam_vith = sam_model_registry["vit_h"](
            checkpoint=f"{WS_ROOT}/worldscore/benchmark/metrics/checkpoints/sam_vit_h_4b8939.pth"
        ).to("cuda")
        self._sam_predictor = SamPredictor(sam_vith)

        raft_args = argparse.Namespace(
            cfg=f"{WS_ROOT}/worldscore/benchmark/metrics/third_party/SEA-RAFT/config/eval/spring-M.json",
            path=f"{WS_ROOT}/worldscore/benchmark/metrics/checkpoints/Tartan-C-T-TSKH-spring540x960-M.pth",
        )
        raft_args = raft_parse_args(raft_args)
        raft_model = RAFT(raft_args)
        load_ckpt(raft_model, raft_args.path)
        raft_model.to("cuda").eval()
        self._raft_model = raft_model
        self._raft_args = raft_args

    def score(self, frame_paths: list[str], objects: list[str]) -> dict[str, Any]:
        """Run the full upstream-aligned motion_accuracy pipeline on one video.

        Returns ``{"score", "base_score", "rate", "match_count", "pred_phrases", "n_pairs"}``.
        """
        self._load()
        # The full pipeline is too long to inline here; delegate to the runner script
        # which mirrors upstream's motion_accuracy_metrics.py + the t2v branch from
        # motion_accuracy_metrics_flowformer.py for first-frame mask generation.
        from videvalkit.benchmarks.worldscore.runners.motion_accuracy import score_one
        return score_one(
            frame_paths=frame_paths,
            objects=objects,
            gdino=self._gdino,
            sam_predictor=self._sam_predictor,
            sam2_predictor=self._sam2_predictor,
            raft_model=self._raft_model,
            raft_args=self._raft_args,
        )
