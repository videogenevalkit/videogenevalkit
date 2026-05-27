"""Standalone metric module — SUPPORTED_METRICS + dispatcher.

Per docs/VIDEO_METRICS_DESIGN.md (user 2026-05-20).

Public API::

    from videvalkit.metrics import SUPPORTED_METRICS, get_metric, list_metrics
    metric = get_metric("fvd")
    result = metric.compute(gen_videos=[...], ref_videos=[...])
"""

from __future__ import annotations

from typing import Any

from videvalkit.metrics.registry import SUPPORTED_METRICS

__all__ = [
    "SUPPORTED_METRICS",
    "get_metric",
    "list_metrics",
    "metric_info",
]


def _import_class(import_path: str) -> type:
    """Import a class from a string ``module.path:ClassName``."""
    if ":" not in import_path:
        raise ValueError(
            f"invalid class path {import_path!r}; expected 'module:Class'"
        )
    module_path, class_name = import_path.split(":", 1)
    import importlib
    module = importlib.import_module(module_path)
    try:
        return getattr(module, class_name)
    except AttributeError as e:
        raise ImportError(
            f"module {module_path} has no attribute {class_name}"
        ) from e


def get_metric(name: str) -> Any:
    """Instantiate the metric class for ``name`` from the registry.

    Raises ``KeyError`` for unknown names; ``ImportError`` if the cls path
    can't be loaded [usually means a checkpoint backbone is missing].
    """
    if name not in SUPPORTED_METRICS:
        from difflib import get_close_matches
        suggestions = get_close_matches(name, list(SUPPORTED_METRICS), n=3)
        msg = f"unknown metric {name!r}"
        if suggestions:
            msg += f"; did you mean: {', '.join(suggestions)}?"
        raise KeyError(msg)
    cfg = SUPPORTED_METRICS[name]
    cls = _import_class(cfg["cls"])
    return cls()


def list_metrics(
    kind: str | None = None,
    no_judge: bool = False,
    source: str | None = None,
) -> list[str]:
    """Return metric names filtered by kind / no-judge / source prefix."""
    out = []
    for name, cfg in SUPPORTED_METRICS.items():
        if kind and cfg.get("kind") != kind:
            continue
        if no_judge and cfg.get("needs_judge", False):
            continue
        if source:
            if not cfg.get("source", "").startswith(source):
                continue
        out.append(name)
    return out


def metric_info(name: str) -> dict[str, Any]:
    """Return a copy of the registry entry for inspection / `metric show`."""
    if name not in SUPPORTED_METRICS:
        raise KeyError(f"unknown metric {name!r}")
    return dict(SUPPORTED_METRICS[name])
