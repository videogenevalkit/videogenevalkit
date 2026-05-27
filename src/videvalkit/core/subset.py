"""Subset application — filter PromptItem stream by a saved prompt_id list.

Per docs/QUICK_EVAL_DESIGN.md §4 (user 2026-05-20 confirmed):

Subsets are version-pinned JSON files shipped per-bench under:
  src/videvalkit/benchmarks/<bench>/subsets/<name>_v<N>.json

Schema:
  {
    "schema_version": 1,
    "subset_name": "quick_v1",
    "benchmark": "worldjen",
    "created": "2026-05-20",
    "n_prompts": 48,
    "selection_method": "stratified_seeded",
    "selection_seed": 42,
    "calibration": {
      "method": "spearman",
      "validation_models": [...],
      "spearman_rho_overall": 0.91,
      "spearman_rho_per_dim": {...},
      "max_dim_disagreement": 0.08
    },
    "prompt_ids": ["wj_0042", "wj_0157", ...]
  }

v0.2 ships the loader + filter machinery; subset_v1.json files per bench
land in D3 [offline calibration runs needed first].
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from videvalkit.core.types import PromptItem

log = logging.getLogger(__name__)

SUBSET_SCHEMA_VERSION = 1


class SubsetCalibration(BaseModel):
    """Optional calibration metadata captured when subset was built."""

    model_config = ConfigDict(extra="forbid")

    method: str = "spearman"
    validation_models: list[str] = Field(default_factory=list)
    spearman_rho_overall: float | None = None
    spearman_rho_per_dim: dict[str, float] = Field(default_factory=dict)
    max_dim_disagreement: float | None = None


class SubsetSpec(BaseModel):
    """Subset file schema. Loaded from <bench>/subsets/<name>.json."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    subset_name: str
    benchmark: str
    created: str
    n_prompts: int
    selection_method: str = "stratified_seeded"
    selection_seed: int = 42
    calibration: SubsetCalibration | None = None
    stratification: dict[str, Any] = Field(default_factory=dict)
    prompt_ids: list[str]


class Subset:
    """Loaded subset, ready to filter PromptItem iterables."""

    def __init__(self, spec: SubsetSpec, source_path: Path | None = None):
        self._spec = spec
        self._source_path = source_path
        self._id_set = set(spec.prompt_ids)

    @classmethod
    def from_file(cls, path: str | Path) -> "Subset":
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"subset file not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        spec = SubsetSpec(**raw)
        if spec.schema_version != SUBSET_SCHEMA_VERSION:
            raise ValueError(
                f"subset {p} schema_version={spec.schema_version} "
                f"not supported [expected {SUBSET_SCHEMA_VERSION}]"
            )
        return cls(spec, source_path=p)

    @property
    def spec(self) -> SubsetSpec:
        return self._spec

    @property
    def name(self) -> str:
        return self._spec.subset_name

    @property
    def benchmark(self) -> str:
        return self._spec.benchmark

    @property
    def n_prompts(self) -> int:
        return self._spec.n_prompts

    def filter_prompts(
        self, prompts: list[PromptItem],
    ) -> list[PromptItem]:
        """Return only prompts whose ``prompt_id`` is in the subset.

        Order: input order is preserved. Logs a warning if any subset
        prompt_id isn't found in the input [stale subset / wrong bench].
        """
        found: list[PromptItem] = []
        seen: set[str] = set()
        for p in prompts:
            if p.prompt_id in self._id_set:
                found.append(p)
                seen.add(p.prompt_id)
        missing = self._id_set - seen
        if missing:
            log.warning(
                "videvalkit: subset %s expects %d prompt_ids; %d not found in "
                "input. Stale subset or wrong bench? Missing examples: %s",
                self._spec.subset_name, len(self._id_set),
                len(missing), sorted(missing)[:5],
            )
        return found

    def hash(self) -> str:
        """SHA-256 of canonicalized prompt_ids list. For provenance in
        result.json / timeline.jsonl per QUICK_EVAL_DESIGN §11.7."""
        canonical = json.dumps(sorted(self._spec.prompt_ids)).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


def find_subset(
    benchmark: str, subset_name: str,
    search_dirs: list[Path] | None = None,
) -> Subset:
    """Look up a named subset for a benchmark.

    Search order:
      1. ``$CWD/.videvalkit/subsets/<bench>/<name>.json``         [user-defined]
      2. ``~/.videvalkit/subsets/<bench>/<name>.json``            [user-defined]
      3. ``src/videvalkit/benchmarks/<bench>/subsets/<name>.json`` [built-in]

    Returns the first match. Raises ``FileNotFoundError`` if none found.
    """
    candidates: list[Path] = []
    if search_dirs:
        for d in search_dirs:
            # Accept either "<dir>/<subset_name>.json" OR "<dir>/<bench>/<subset_name>.json"
            candidates.append(d / f"{subset_name}.json")
            candidates.append(d / benchmark / f"{subset_name}.json")

    # CWD
    cwd_sub = (
        Path.cwd() / ".videvalkit" / "subsets" / benchmark / f"{subset_name}.json"
    )
    candidates.append(cwd_sub)
    # User
    user_sub = (
        Path.home() / ".videvalkit" / "subsets" / benchmark / f"{subset_name}.json"
    )
    candidates.append(user_sub)
    # Built-in
    try:
        import videvalkit.benchmarks as _bm_mod
        builtin_dir = Path(_bm_mod.__file__).parent / benchmark / "subsets"
        candidates.append(builtin_dir / f"{subset_name}.json")
    except Exception:
        pass

    for cand in candidates:
        if cand.is_file():
            return Subset.from_file(cand)

    raise FileNotFoundError(
        f"subset {subset_name!r} not found for benchmark {benchmark!r}. "
        f"Searched: {candidates}"
    )
