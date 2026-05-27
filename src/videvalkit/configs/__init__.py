"""Configuration registries — VLMEvalKit-style dicts.

Four top-level registries; each entry is a small dict that the runner /
adapter / scorer factory reads. Adding a new benchmark / judge / metric /
aggregator is one entry here plus the implementation class — OR a plugin
under ``~/.videvalkit/`` (see ``videvalkit.plugins``).

Example usage::

    from videvalkit.configs import SUPPORTED_BENCHMARKS, SUPPORTED_JUDGES
    cfg = SUPPORTED_JUDGES["gemma-4-31b-local"]

Registry merge semantics (per docs/INTEGRATION_FRAMEWORK_DESIGN.md §4):
  * ``SUPPORTED_JUDGES``     — built-in + ``~/.config/videvalkit/judges.yaml``
                               (see ``configs.judge_loader``)
  * ``SUPPORTED_BENCHMARKS`` — built-in + entry_points ``videvalkit.benchmarks``
                               + ``~/.videvalkit/benchmarks/`` (see ``plugins.loader``)
  * ``SUPPORTED_AGGREGATORS``— built-in (no plugin layer yet)

Set ``VIDEVALKIT_DISABLE_PLUGINS=1`` to ignore third-party plugin sources
(builtin still loads).
"""

from videvalkit.configs.aggregators import SUPPORTED_AGGREGATORS
from videvalkit.configs.benchmarks import (
    BENCHMARK_RELATIONS,
    SUPPORTED_BENCHMARKS as _BUILTIN_BENCHMARKS,
)
from videvalkit.configs.judge_loader import get_judges
from videvalkit.configs.judges import SUPPORTED_JUDGES as _BUILTIN_JUDGES
from videvalkit.plugins import discover as _plugin_discover

# Lazy-merge: SUPPORTED_JUDGES exposed to callers is the union of the
# built-in registry and any user-supplied entries in judges.yaml.
SUPPORTED_JUDGES = get_judges(_BUILTIN_JUDGES)

# Lazy-merge: SUPPORTED_BENCHMARKS includes plugin sources
# (entry_points + ~/.videvalkit/benchmarks/)
SUPPORTED_BENCHMARKS = _plugin_discover("benchmarks", _BUILTIN_BENCHMARKS)

__all__ = [
    "SUPPORTED_BENCHMARKS",
    "SUPPORTED_JUDGES",
    "SUPPORTED_AGGREGATORS",
    "BENCHMARK_RELATIONS",
]
