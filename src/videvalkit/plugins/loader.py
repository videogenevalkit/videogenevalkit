"""Three-layer plugin discovery: built-in / pip entry_points / local dirs.

Per docs/INTEGRATION_FRAMEWORK_DESIGN.md §4 (user 2026-05-20 confirmed):

    1. Built-in        src/videvalkit/{benchmarks,configs}/        (existing)
    2. pip entry_point [project.entry-points.videvalkit.{group}]   (new)
    3. Local           ~/.videvalkit/{benchmarks,metrics}/         (new)
                       $CWD/.videvalkit/{benchmarks,metrics}/      (project-level override)

Conflict resolution: **same name → later source wins** (top-level key replacement,
no field-level deep merge). Conflicts are logged at INFO so they're never silent.

Entry-point groups recognised:
  * ``videvalkit.benchmarks``  — yields {name: bench_cfg_dict}
  * ``videvalkit.metrics``     — yields {name: metric_cfg_dict}  (v0.2 metrics module)
  * ``videvalkit.judges``      — yields {name: judge_cfg_dict}   (future)
  * ``videvalkit.aggregators`` — yields {name: agg_cfg_dict}     (future)

Local dirs use the **__videvalkit_register__()** convention: any
``<dir>/<plugin_name>/__init__.py`` (or ``benchmark.py``) that defines a
top-level function ``__videvalkit_register__()`` returning a dict like
``{"benchmarks": {...}, "metrics": {...}}`` will be picked up.

Disable all third-party plugin loading with ``VIDEVALKIT_DISABLE_PLUGINS=1``
(debug escape hatch). Built-in registry is always loaded.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger(__name__)

USER_PLUGINS_DIR = Path.home() / ".videvalkit"
PROJECT_PLUGINS_DIR_NAME = ".videvalkit"
_ENV_DISABLE = "VIDEVALKIT_DISABLE_PLUGINS"

SUPPORTED_GROUPS = ("benchmarks", "metrics", "judges", "aggregators")


def _disabled() -> bool:
    return os.environ.get(_ENV_DISABLE, "0") == "1"


# ----------------------------------------------------- entry_points layer ---
def _load_entry_points(group: str) -> dict[str, Any]:
    """Load all entries from ``videvalkit.<group>`` entry_points group."""
    if _disabled():
        return {}
    out: dict[str, Any] = {}
    try:
        from importlib.metadata import entry_points
        eps = entry_points(group=f"videvalkit.{group}")
    except Exception as e:
        log.debug("videvalkit: entry_points lookup for %s failed: %s", group, e)
        return {}
    for ep in eps:
        try:
            value = ep.load()
        except Exception as e:
            log.warning(
                "videvalkit: failed to load entry_point %s = %s: %s",
                ep.name, ep.value, e,
            )
            continue
        # Convention: the loaded value is either:
        #   (a) a class/dict-like that becomes the entry under ep.name
        #   (b) a callable returning a {name: cfg} dict (multi-entry)
        if callable(value) and not isinstance(value, type):
            try:
                multi = value()
                if isinstance(multi, dict):
                    out.update(multi)
                    continue
            except Exception as e:
                log.warning(
                    "videvalkit: entry_point %s callable raised: %s", ep.name, e
                )
                continue
        out[ep.name] = value
    return out


# ----------------------------------------------------- local-dir layer ---
def _load_local_dir(base: Path, group: str) -> dict[str, Any]:
    """Scan ``<base>/<group>/<name>/__init__.py`` looking for
    ``__videvalkit_register__()`` callables.

    A plugin module is loaded once; its register function is called once;
    the returned dict's ``{group: {name: cfg}}`` is the contribution to this
    group. Failures isolated to per-plugin (one bad plugin doesn't kill the
    rest).
    """
    if _disabled():
        return {}
    group_dir = base / group
    if not group_dir.is_dir():
        return {}
    out: dict[str, Any] = {}
    for plugin_path in sorted(group_dir.iterdir()):
        if not plugin_path.is_dir() or plugin_path.name.startswith("."):
            continue
        init_py = plugin_path / "__init__.py"
        if not init_py.is_file():
            # also accept plugin_name/benchmark.py as a shorthand
            init_py = plugin_path / "benchmark.py"
            if not init_py.is_file():
                continue
        mod_name = f"videvalkit_user_plugin_{group}_{plugin_path.name}"
        try:
            spec = importlib.util.spec_from_file_location(mod_name, init_py)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)
        except Exception as e:
            log.warning(
                "videvalkit: failed to import local plugin %s: %s", init_py, e
            )
            continue
        register = getattr(module, "__videvalkit_register__", None)
        if not callable(register):
            log.debug(
                "videvalkit: local plugin %s has no __videvalkit_register__()",
                plugin_path,
            )
            continue
        try:
            registered = register()
        except Exception as e:
            log.warning(
                "videvalkit: %s.__videvalkit_register__() raised: %s",
                plugin_path, e,
            )
            continue
        if not isinstance(registered, dict):
            log.warning(
                "videvalkit: %s.__videvalkit_register__() must return a dict",
                plugin_path,
            )
            continue
        # Pull the matching group's entries
        group_entries = registered.get(group, {})
        if isinstance(group_entries, dict):
            out.update(group_entries)
    return out


# ----------------------------------------------------- public discovery API ---
def discover(group: str, builtin: dict[str, Any]) -> dict[str, Any]:
    """Return the merged registry for ``group`` (one of SUPPORTED_GROUPS).

    Merge order (lowest precedence first):
      1. ``builtin`` (passed in by caller — typically the dict from configs/)
      2. pip entry_points group ``videvalkit.<group>``
      3. ``~/.videvalkit/<group>/``
      4. ``$CWD/.videvalkit/<group>/``

    Same name in two sources → later source wins. Conflicts logged at INFO.
    """
    if group not in SUPPORTED_GROUPS:
        raise ValueError(
            f"unknown plugin group {group!r}; "
            f"supported: {SUPPORTED_GROUPS}"
        )

    merged: dict[str, Any] = dict(builtin)
    sources: list[tuple[str, dict[str, Any]]] = [
        (f"entry_points:videvalkit.{group}", _load_entry_points(group)),
        (f"{USER_PLUGINS_DIR}/{group}", _load_local_dir(USER_PLUGINS_DIR, group)),
        (
            f"{Path.cwd() / PROJECT_PLUGINS_DIR_NAME}/{group}",
            _load_local_dir(Path.cwd() / PROJECT_PLUGINS_DIR_NAME, group),
        ),
    ]
    for source_label, contribution in sources:
        if not contribution:
            continue
        overlaps = set(merged) & set(contribution)
        if overlaps:
            for name in sorted(overlaps):
                log.info(
                    "videvalkit: plugin %s overrides %r (from %s)",
                    source_label, name,
                    "previous source",
                )
        merged.update(contribution)
    return merged


def discover_all(
    builtins: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Convenience: discover all four groups at once.

    ``builtins`` is a dict mapping group name → built-in registry, e.g.::

        {
            "benchmarks": SUPPORTED_BENCHMARKS,
            "judges":     SUPPORTED_JUDGES,
            ...
        }
    """
    return {
        group: discover(group, builtins.get(group, {}))
        for group in SUPPORTED_GROUPS
    }


# ----------------------------------------------------- diagnostics helper ---
def plugin_sources_report() -> dict[str, dict[str, Any]]:
    """Return a structured report of what plugin sources are visible.

    Used by ``videvalkit doctor`` to surface plugin layer health (env disable
    flag, entry_points count, local dir presence, etc.). Does NOT actually
    load plugins — just probes filesystem and entry_point metadata.
    """
    report: dict[str, Any] = {
        "disabled_by_env": _disabled(),
        "env_var": _ENV_DISABLE,
        "user_dir": str(USER_PLUGINS_DIR),
        "user_dir_exists": USER_PLUGINS_DIR.is_dir(),
        "project_dir": str(Path.cwd() / PROJECT_PLUGINS_DIR_NAME),
        "project_dir_exists": (Path.cwd() / PROJECT_PLUGINS_DIR_NAME).is_dir(),
        "groups": {},
    }
    for group in SUPPORTED_GROUPS:
        entry_count = 0
        try:
            from importlib.metadata import entry_points
            entry_count = len(list(entry_points(group=f"videvalkit.{group}")))
        except Exception:
            pass
        report["groups"][group] = {
            "entry_points": entry_count,
            "user_plugins": _count_local_plugins(USER_PLUGINS_DIR, group),
            "project_plugins": _count_local_plugins(
                Path.cwd() / PROJECT_PLUGINS_DIR_NAME, group
            ),
        }
    return report


def _count_local_plugins(base: Path, group: str) -> int:
    d = base / group
    if not d.is_dir():
        return 0
    return sum(
        1 for p in d.iterdir()
        if p.is_dir() and not p.name.startswith(".") and (
            (p / "__init__.py").is_file() or (p / "benchmark.py").is_file()
        )
    )
