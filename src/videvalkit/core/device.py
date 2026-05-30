"""Canonical device resolution + Ascend NPU runtime activation.

One place decides "auto / cuda / npu / mps / cpu" and, when NPU is chosen,
activates the torch_npu runtime. The activation imports
``torch_npu.contrib.transfer_to_npu``, which monkeypatches ``.cuda()`` /
``"cuda"`` device strings to NPU — so upstream code that hardcodes ``.cuda()``
(e.g. the VBench dimension scorers) runs on Ascend without per-call edits.

Safety: the shim is only imported when a device actually resolves to ``npu``
and ``torch_npu`` is importable. On a CUDA/CPU box it is never touched, so
existing cuda runs are unaffected.

NOTE: the NPU path is plumbed but unverified on real Ascend 910B hardware
(the dev box is CUDA-only). Run ``scripts/npu_smoke.py`` on-device to confirm.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_NPU_RUNTIME_READY = False


def ensure_npu_runtime() -> bool:
    """Idempotently activate the torch_npu runtime + ``.cuda()``→npu shim.

    Returns True if torch_npu is active, False if it is unavailable (the caller
    then falls back to cpu). Safe to call repeatedly and on non-NPU boxes.
    """
    global _NPU_RUNTIME_READY
    if _NPU_RUNTIME_READY:
        return True
    try:
        import torch_npu  # noqa: F401  (registers the 'npu' backend)
        # Redirects .cuda() / "cuda" -> npu for upstream code that hardcodes CUDA.
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
        _NPU_RUNTIME_READY = True
        log.info("videvalkit: torch_npu runtime active (.cuda() -> npu)")
        return True
    except Exception as e:  # torch_npu absent / not an Ascend box
        log.warning("videvalkit: torch_npu unavailable (%s); NPU path inactive", e)
        return False


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _npu_available() -> bool:
    """True if torch_npu is importable and an NPU is visible."""
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
    - Resolving to ``npu`` activates the torch_npu runtime shim.
    - ``cpu`` (or anything else) passes through unchanged.
    """
    if device == "auto":
        if _cuda_available():
            return "cuda"
        if _npu_available():
            ensure_npu_runtime()
            return "npu"
        if _mps_available():
            return "mps"
        return "cpu"

    base = device.split(":")[0]
    if base == "cuda":
        return device if _cuda_available() else _fallback_cpu(device)
    if base == "npu":
        if _npu_available() and ensure_npu_runtime():
            return device
        return _fallback_cpu(device)
    if base == "mps":
        return device if _mps_available() else _fallback_cpu(device)
    return device


def get_device():
    """Return a ``torch.device`` for the auto-resolved backend.

    Convenience for callers that want a device object rather than a string.
    """
    import torch
    return torch.device(resolve_device("auto"))
