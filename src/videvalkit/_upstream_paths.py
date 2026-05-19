"""Resolve upstream-repo locations and add them to ``sys.path``.

Adapters in ``videvalkit.benchmarks.<bench>`` import from upstream packages
(``from vbench import VBench``, ``from worldscore.benchmark... import ...``).
The upstream repos are not on PyPI; users clone them via:

    videvalkit fetch-upstream --all

which writes them to ``~/.cache/videvalkit/upstream/<repo>/``. This module is
imported by ``videvalkit/__init__.py`` and prepends each subdirectory to
``sys.path`` if present, so the adapter imports work without an extra ``-e``
install step per upstream.

Override the upstream root via env var:

    export VIDEVALKIT_CACHE_HOME=/some/other/path
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_CACHE_HOME = Path(os.environ.get(
    "VIDEVALKIT_CACHE_HOME",
    Path.home() / ".cache" / "videvalkit",
))

# Mapping: upstream subdir under ~/.cache/videvalkit/upstream/ -> the path to
# add to sys.path so the relevant top-level package (e.g. ``vbench``) is
# importable. The subdir is the same name as ``videvalkit fetch-upstream``
# clones into.
_UPSTREAM_PATHS = [
    "upstream/VBench",                              # exposes both `vbench` and (via subdir) VBench-2.0's vbench2 module
    "upstream/VBench/VBench-2.0",                   # exposes `vbench2`
    "upstream/Video-Bench",                         # exposes `videobench`
    "upstream/WorldJen-benchmarking-subsystem",     # WorldJen scaffolding (not currently imported as a package)
    "upstream/WorldScore",                          # exposes `worldscore`
    "upstream/T2V-CompBench",                       # T2V-CompBench scripts (subprocess-invoked, not imported)
]


def add_upstream_to_syspath() -> list[Path]:
    """Append each existing upstream subdir to ``sys.path`` (only if it exists).

    Returns the list of paths actually added. Missing repos are silently
    skipped — the corresponding adapter will fail at import time with a
    clear error message that points the user at ``videvalkit fetch-upstream``.
    """
    added: list[Path] = []
    for rel in _UPSTREAM_PATHS:
        p = _CACHE_HOME / rel
        if p.is_dir():
            sp = str(p)
            if sp not in sys.path:
                sys.path.append(sp)
                added.append(p)
    return added


# Run at import time so adapters can use ``from vbench import ...`` etc.
add_upstream_to_syspath()
