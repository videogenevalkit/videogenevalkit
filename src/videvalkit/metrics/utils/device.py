"""Device resolution for metrics — re-exports the canonical core helper.

The single source of truth lives in :mod:`videvalkit.core.device`. This module
keeps the historical ``videvalkit.metrics.utils.device`` import path working.
"""

from __future__ import annotations

from videvalkit.core.device import (
    ensure_npu_runtime,
    get_device,
    resolve_device,
)

__all__ = ["resolve_device", "ensure_npu_runtime", "get_device"]
