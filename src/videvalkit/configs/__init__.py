"""Configuration registries — VLMEvalKit-style dicts.

Three top-level registries; each entry is a small dict that the runner /
adapter / scorer factory reads. Adding a new benchmark / judge / aggregator
is one entry here plus the implementation class.

Example usage::

    from videvalkit.configs import SUPPORTED_BENCHMARKS, SUPPORTED_JUDGES
    cfg = SUPPORTED_JUDGES["gemma-4-31b-local"]

``SUPPORTED_JUDGES`` is merged with ``~/.config/videvalkit/judges.yaml`` (and
optionally ``$CWD/.videvalkit/judges.yaml``) at import time so users can add
their own endpoints without forking. See ``configs.judge_loader`` and
``docs/JUDGE_SELECTION_DESIGN.md`` §4.
"""

from videvalkit.configs.aggregators import SUPPORTED_AGGREGATORS
from videvalkit.configs.benchmarks import BENCHMARK_RELATIONS, SUPPORTED_BENCHMARKS
from videvalkit.configs.judge_loader import get_judges
from videvalkit.configs.judges import SUPPORTED_JUDGES as _BUILTIN_JUDGES

# Lazy-merge: SUPPORTED_JUDGES exposed to callers is the union of the
# built-in registry and any user-supplied entries in judges.yaml.
# User entries override built-in entries with the same name.
SUPPORTED_JUDGES = get_judges(_BUILTIN_JUDGES)

__all__ = [
    "SUPPORTED_BENCHMARKS",
    "SUPPORTED_JUDGES",
    "SUPPORTED_AGGREGATORS",
    "BENCHMARK_RELATIONS",
]
