"""Reference video-set management for distribution metrics.

Per docs/VIDEO_METRICS_DESIGN.md §8. Distribution metrics [FVD/VFID/KVD/
CLIP-FVD] need a reference video set. This module lets users register and
look up named ref sets.

Lookup order [first match wins]:
  1. $CWD/.videvalkit/refs.yaml
  2. ~/.config/videvalkit/refs.yaml

refs.yaml schema:
  refs:
    my-ref:
      path: /data/my_reference_videos/
      description: my held-out test set
      n_clips: 500
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

log = logging.getLogger(__name__)

USER_REFS_YAML = Path.home() / ".config" / "videvalkit" / "refs.yaml"
PROJECT_REFS_YAML_REL = Path(".videvalkit") / "refs.yaml"

# Built-in known reference sets [v0.2 placeholders — actual data hosted on
# videogenevalkit/reference-videos HF dataset, fetched via fetch-refs].
BUILTIN_REFS: dict[str, dict[str, Any]] = {
    "ucf101-fvd": dict(
        hf_repo="videogenevalkit/reference-videos",
        hf_subdir="ucf101-fvd",
        n_clips=13320, size_gb=5.0,
        description="UCF101 + paper preprocessing [FVD canonical ref]",
    ),
    "ucf101-fvd-subset-500": dict(
        hf_repo="videogenevalkit/reference-videos",
        hf_subdir="ucf101-fvd-subset-500",
        n_clips=500, size_gb=0.25,
        description="UCF101 500-clip subset [quick-profile ref]",
    ),
    "msr-vtt-val": dict(
        hf_repo="videogenevalkit/reference-videos",
        hf_subdir="msr-vtt-val",
        n_clips=2990, size_gb=2.0,
        description="MSR-VTT validation set",
    ),
}


class RefSpec(BaseModel):
    model_config = ConfigDict(extra="allow")
    path: str | None = None
    description: str = ""
    n_clips: int | None = None
    hf_repo: str | None = None
    hf_subdir: str | None = None


def _read_refs_yaml(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        log.warning("videvalkit: malformed refs.yaml %s: %s", path, e)
        return {}
    refs = data.get("refs", {}) if isinstance(data, dict) else {}
    return refs if isinstance(refs, dict) else {}


def load_user_refs() -> dict[str, dict[str, Any]]:
    """User refs from ~/.config + $CWD/.videvalkit, later overrides earlier."""
    merged: dict[str, dict[str, Any]] = {}
    for p in (USER_REFS_YAML, Path.cwd() / PROJECT_REFS_YAML_REL):
        merged.update(_read_refs_yaml(p))
    return merged


def get_refs() -> dict[str, dict[str, Any]]:
    """Built-in + user refs [user overrides built-in by name]."""
    out = dict(BUILTIN_REFS)
    out.update(load_user_refs())
    return out


def resolve_ref_path(name: str) -> Path:
    """Resolve a named ref to a local directory path.

    Raises if the ref isn't registered, or is a built-in not yet fetched.
    """
    refs = get_refs()
    if name not in refs:
        from difflib import get_close_matches
        sug = get_close_matches(name, list(refs), n=3)
        msg = f"unknown ref {name!r}"
        if sug:
            msg += f"; did you mean: {', '.join(sug)}?"
        raise KeyError(msg)
    cfg = refs[name]
    # Explicit local path [user-registered]
    if cfg.get("path"):
        p = Path(cfg["path"])
        if not p.is_dir():
            raise FileNotFoundError(f"ref {name!r} path does not exist: {p}")
        return p
    # Built-in — check the fetch cache
    import os
    cache = Path(os.environ.get("VIDEVALKIT_CACHE_HOME",
                                Path.home() / ".cache" / "videvalkit")) / "refs" / name
    if cache.is_dir():
        return cache
    raise FileNotFoundError(
        f"ref {name!r} is a built-in set not yet fetched. Run "
        f"`videvalkit fetch-refs --name {name}` [needs HF repo "
        f"{cfg.get('hf_repo')}], or register a local path with "
        f"`videvalkit refs register --name {name} --path <dir>`."
    )


def register_ref(name: str, path: str | Path, description: str = "") -> Path:
    """Append a ref entry to the user refs.yaml [~/.config/videvalkit/refs.yaml]."""
    p = Path(path).resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"ref path not a directory: {p}")
    USER_REFS_YAML.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if USER_REFS_YAML.is_file():
        existing = yaml.safe_load(USER_REFS_YAML.read_text()) or {}
    existing.setdefault("refs", {})
    existing["refs"][name] = {"path": str(p), "description": description}
    USER_REFS_YAML.write_text(yaml.safe_dump(existing, sort_keys=False))
    return USER_REFS_YAML
