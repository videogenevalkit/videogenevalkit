"""User-configurable judge endpoints — merge ~/.config/videvalkit/judges.yaml
with the built-in SUPPORTED_JUDGES registry.

Designed per docs/JUDGE_SELECTION_DESIGN.md §4 (user 2026-05-20 confirmed):

  builtin (configs/judges.py)
      ↓ merge by top-level key (user wins)
  ~/.config/videvalkit/judges.yaml
      ↓ merge
  $CWD/.videvalkit/judges.yaml

User entries with the same name as a builtin **replace** the builtin entry
(no field-level deep merge — keeps semantics simple).

Invalid entries (failing pydantic validation) are **skipped with a warning**,
not raised, so one bad entry never breaks the whole registry.

Disable user yaml loading entirely with the env var
``VIDEVALKIT_JUDGE_USER_YAML=0``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

log = logging.getLogger(__name__)

USER_YAML_PATH = Path.home() / ".config" / "videvalkit" / "judges.yaml"
PROJECT_YAML_PATH_RELATIVE = Path(".videvalkit") / "judges.yaml"

_ENV_DISABLE = "VIDEVALKIT_JUDGE_USER_YAML"


class JudgeConfig(BaseModel):
    """Schema for one user-supplied judge entry.

    Validated when a user judges.yaml is loaded. Unknown fields are allowed
    (``extra="allow"``) so that future SDK kwargs propagate without code change.
    """

    model_config = ConfigDict(extra="allow")

    kind: Literal["openai_compatible", "gemini", "anthropic"]
    model: str
    provider: str = "unknown"
    endpoint: str | None = None
    api_key_env: str | None = None
    sys_prompt: str | None = None


def _user_yaml_disabled() -> bool:
    return os.environ.get(_ENV_DISABLE, "1") == "0"


def _read_yaml(path: Path) -> dict[str, dict[str, Any]]:
    """Read one yaml file, return its ``judges:`` block or empty dict.

    Returns ``{}`` for missing file, empty file, or yaml whose top level
    does not contain a ``judges`` mapping. Always returns a dict, never raises.
    """
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        log.warning("videvalkit: skipping malformed judges yaml %s: %s", path, e)
        return {}
    if not isinstance(data, dict):
        return {}
    judges = data.get("judges", {})
    if not isinstance(judges, dict):
        log.warning("videvalkit: %s has no 'judges:' mapping; skipping", path)
        return {}
    return judges


def _validate_entries(
    raw: dict[str, dict[str, Any]], source: Path
) -> dict[str, dict[str, Any]]:
    """Validate each entry; drop invalid ones with a warning."""
    out: dict[str, dict[str, Any]] = {}
    for name, cfg in raw.items():
        if not isinstance(cfg, dict):
            log.warning(
                "videvalkit: judges.yaml %s entry %r is not a mapping; skipping",
                source,
                name,
            )
            continue
        try:
            validated = JudgeConfig(**cfg)
        except ValidationError as e:
            log.warning(
                "videvalkit: judges.yaml %s entry %r failed validation; skipping (%s)",
                source,
                name,
                e.errors()[0]["msg"] if e.errors() else str(e),
            )
            continue
        # Re-export as a plain dict so callers see the same shape as builtin
        # SUPPORTED_JUDGES entries (which are also plain dicts).
        out[name] = validated.model_dump(exclude_none=True)
    return out


def load_user_judges() -> dict[str, dict[str, Any]]:
    """Return merged user-judges dict (~/.config + $CWD/.videvalkit, in order).

    Returns an empty dict if user yaml loading is disabled or no files exist.
    Never raises — bad files / bad entries are warned and skipped.
    """
    if _user_yaml_disabled():
        return {}
    merged: dict[str, dict[str, Any]] = {}
    # Order matters: later sources override earlier ones.
    for path in (USER_YAML_PATH, Path.cwd() / PROJECT_YAML_PATH_RELATIVE):
        raw = _read_yaml(path)
        if not raw:
            continue
        merged.update(_validate_entries(raw, path))
    return merged


def get_judges(
    builtin: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return the full judge registry: builtin entries merged with user yaml.

    User entries override builtin entries with the same name (top-level key
    replacement; no field-level deep merge).
    """
    # Lazy import avoids circular dep with configs/__init__.py.
    if builtin is None:
        from videvalkit.configs.judges import SUPPORTED_JUDGES as _builtin
        builtin = _builtin
    merged = dict(builtin)
    merged.update(load_user_judges())
    return merged


def resolve_judge(
    benchmark: str,
    judge_name: str | None = None,
    judge_override: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Resolve the ``--judge`` argument to a concrete judge config dict.

    Per docs/JUDGE_SELECTION_DESIGN.md §3:

    +----------------------+---------------------------------------------+
    | ``judge_name``       | resolves to                                 |
    +======================+=============================================+
    | ``None``             | benchmark's ``default_judge`` (back-compat) |
    | ``"default"``        | benchmark's ``default_judge`` (explicit)    |
    | ``"paper"``          | benchmark's ``paper_judge`` (★ new in M-alias) |
    | ``"<registry name>"``| direct lookup in merged ``SUPPORTED_JUDGES``|
    | (``judge_override``) | use that dict verbatim, bypass registry     |
    +----------------------+---------------------------------------------+

    Returns ``None`` for benchmarks where ``needs_judge=False`` and the user
    did not pass anything. Raises ``ValueError`` / ``KeyError`` for
    unresolvable inputs (with suggestions where helpful).
    """
    # Lazy import avoids circular dep with configs/__init__.py
    from videvalkit.configs.benchmarks import SUPPORTED_BENCHMARKS

    if judge_override is not None:
        # ad-hoc endpoint via CLI flags; trust the caller's dict
        return judge_override

    if benchmark not in SUPPORTED_BENCHMARKS:
        raise KeyError(
            f"unknown benchmark {benchmark!r}; "
            f"known: {sorted(SUPPORTED_BENCHMARKS)}"
        )
    bench_cfg = SUPPORTED_BENCHMARKS[benchmark]
    needs_judge = bench_cfg.get("needs_judge", False)

    # No judge needed and none requested
    if not needs_judge and not judge_name:
        return None

    judges = get_judges()

    # Semantic keyword: "paper"
    if judge_name == "paper":
        paper_j = bench_cfg.get("paper_judge")
        if paper_j is None:
            raise ValueError(
                f"benchmark {benchmark!r} has no paper_judge declared. "
                f"Available for this bench: default={bench_cfg.get('default_judge')!r}"
            )
        if paper_j not in judges:
            raise ValueError(
                f"benchmark {benchmark!r} paper_judge={paper_j!r} not found "
                f"in registry (built-in + user yaml)."
            )
        return judges[paper_j]

    # Semantic keyword: "default"
    if judge_name == "default":
        default_j = bench_cfg.get("default_judge")
        if default_j is None:
            raise ValueError(
                f"benchmark {benchmark!r} has no default_judge declared"
            )
        return judges[default_j]

    # No keyword → fall back to bench default
    if judge_name is None:
        default_j = bench_cfg.get("default_judge")
        if default_j is None:
            if needs_judge:
                raise ValueError(
                    f"benchmark {benchmark!r} needs a judge; pass --judge"
                )
            return None
        return judges[default_j]

    # Direct registry lookup
    if judge_name not in judges:
        from difflib import get_close_matches
        suggestions = get_close_matches(judge_name, list(judges), n=3)
        msg = f"unknown judge {judge_name!r}"
        if suggestions:
            msg += f"; did you mean: {', '.join(suggestions)}?"
        msg += (
            f"\n  available: {sorted(judges)[:10]}"
            f"{'...' if len(judges) > 10 else ''}"
            f"\n  Add custom endpoints in ~/.config/videvalkit/judges.yaml"
        )
        raise KeyError(msg)
    return judges[judge_name]
