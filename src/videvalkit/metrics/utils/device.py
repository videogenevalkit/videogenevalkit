"""Device resolution shared across metrics, with optional Ascend NPU support.

`resolve_device` centralizes the "auto / cuda / npu / mps / cpu" decision so
every metric handles devices the same way. NPU support is best-effort: it needs
`torch_npu` (Ascend) installed; importing it also registers the ``npu`` backend
with torch. On a box without it, ``--device npu`` falls back to cpu with a
warning rather than crashing.

NOTE: the NPU path is plumbed but has not been validated on real Ascend
hardware (the dev box is CUDA-only). The device-agnostic metrics (FVD / VFID /
KVD / CLIP-FVD / CLIP-Score / ViCLIP-Score) are pure torch forward passes, so
they are expected to run once torch_npu is present — verify on-device before
reporting NPU numbers.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def _npu_available() -> bool:
    """True if torch_npu is importable and an NPU is visible (registers backend)."""
    try:
        import torch
        import torch_npu  # noqa: F401  (import registers the 'npu' backend)
        return bool(getattr(torch, "npu", None) and torch.npu.is_available())
    except Exception:
        return False


def _mps_available() -> bool:
    try:
        import torch
        return bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
    except Exception:
        return False


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _fallback_cpu(requested: str) -> str:
    log.warning(
        "videvalkit: device %r requested but unavailable; falling back to cpu",
        requested,
    )
    return "cpu"


def resolve_device(device: str = "auto") -> str:
    """Resolve a device string to a concrete backend.

    - ``auto`` prefers cuda > npu > mps > cpu.
    - An explicit ``cuda`` / ``npu`` / ``mps`` (optionally ``:N``) is honored if
      available, else falls back to cpu with a warning.
    - ``cpu`` (or anything else) passes through unchanged.
    """
    if device == "auto":
        if _cuda_available():
            return "cuda"
        if _npu_available():
            return "npu"
        if _mps_available():
            return "mps"
        return "cpu"

    base = device.split(":")[0]
    if base == "cuda":
        return device if _cuda_available() else _fallback_cpu(device)
    if base == "npu":
        return device if _npu_available() else _fallback_cpu(device)
    if base == "mps":
        return device if _mps_available() else _fallback_cpu(device)
    return device
