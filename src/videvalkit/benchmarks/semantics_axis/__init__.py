"""Semantics Axis Eval — 21-axis VLM-judge prompt-following benchmark."""

from videvalkit.benchmarks.semantics_axis.benchmark import (
    SEMANTICS_AXIS_DIMENSIONS,
    SEMANTICS_AXIS_GROUPS,
    SemanticsAxisBenchmark,
)

__all__ = [
    "SemanticsAxisBenchmark",
    "SEMANTICS_AXIS_DIMENSIONS",
    "SEMANTICS_AXIS_GROUPS",
]
