"""Configuration registries — VLMEvalKit-style dicts.

Three top-level registries; each entry is a small dict that the runner /
adapter / scorer factory reads. Adding a new benchmark / judge / aggregator
is one entry here plus the implementation class.

Example usage::

    from videvalkit.configs import SUPPORTED_BENCHMARKS, SUPPORTED_JUDGES
    cfg = SUPPORTED_JUDGES["gemma-4-31b-local"]
"""

from videvalkit.configs.benchmarks import BENCHMARK_RELATIONS, SUPPORTED_BENCHMARKS
from videvalkit.configs.judges import SUPPORTED_JUDGES
from videvalkit.configs.aggregators import SUPPORTED_AGGREGATORS

__all__ = [
    "SUPPORTED_BENCHMARKS",
    "SUPPORTED_JUDGES",
    "SUPPORTED_AGGREGATORS",
    "BENCHMARK_RELATIONS",
]
