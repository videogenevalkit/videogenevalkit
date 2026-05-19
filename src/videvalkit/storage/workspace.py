"""Workspace — thin facade over WorkspaceLayout that also reads/writes JSON.

Adapters call `ws.write_raw(result)` to persist incrementally — this makes
runs resumable: if the toolkit crashes, the runner can scan `results/raw/`
and skip what's already done.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from videvalkit.core.layout import WorkspaceLayout
from videvalkit.core.types import RawResult, Summary


class Workspace:
    def __init__(self, root: str | Path) -> None:
        self.layout = WorkspaceLayout(Path(root)).ensure()

    # ---- raw per-(video, scorer) results ------------------------------------
    def write_raw(self, r: RawResult) -> Path:
        path = self.layout.raw_path(r.benchmark, r.model, r.dimension, r.prompt_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(r.model_dump_json(indent=2))
        return path

    def has_raw(self, benchmark: str, model: str, dimension: str, prompt_id: str) -> bool:
        return self.layout.raw_path(benchmark, model, dimension, prompt_id).exists()

    def load_raw(self, benchmark: str, model: str, dimension: str, prompt_id: str) -> RawResult:
        return RawResult.model_validate_json(
            self.layout.raw_path(benchmark, model, dimension, prompt_id).read_text()
        )

    def scan_raw(self, benchmark: str, model: str | None = None) -> list[RawResult]:
        base = self.layout.results_dir / "raw" / benchmark
        if not base.exists():
            return []
        results = []
        glob = "*/*/*.json" if model is None else f"{model}/*/*.json"
        for p in base.glob(glob):
            try:
                results.append(RawResult.model_validate_json(p.read_text()))
            except Exception:
                continue
        return results

    # ---- summary -------------------------------------------------------------
    def write_summary(self, s: Summary) -> Path:
        path = self.layout.summary_path(s.benchmark, s.model)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(s.model_dump_json(indent=2))
        return path

    # ---- generic helpers -----------------------------------------------------
    def write_json(self, relpath: str | Path, payload: Any) -> Path:
        path = self.layout.root / Path(relpath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        return path
