"""api_log writer — sinks every VLM-judge API call to disk in the same schema
that `video_eval/results/api_logs/` already uses, so existing analysis tooling
(and your eyes) keep working.

Schema (one JSON object per line, jsonl):
    {
      "timestamp": ISO8601,
      "model": str,
      "user": str,
      "request": {...},
      "response": {"usage": {...}, "response": str | dict}
    }

Files are partitioned:
    calls/{provider}/{model}/{YYYY-MM}/{YYYY-MM-DD}.{ts}.jsonl
    stats/{provider}/{model}/{user}_{YYYY-MM}.{ts}.jsonl
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from videvalkit.core.layout import WorkspaceLayout


class ApiCallRecord(BaseModel):
    """One (request, response) pair logged from a VLM judge call."""

    model_config = ConfigDict(extra="allow")

    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model: str
    user: str = Field(default_factory=lambda: os.environ.get("USER", "unknown"))
    request: dict[str, Any]
    response: dict[str, Any]


class ApiCallLogger:
    """Append-only writer; one logger instance per (provider, model, run).

    Thread-safety: the underlying file open() + write is atomic for short
    payloads on POSIX, but a real multi-process run should funnel through a
    single logger process. Skeleton uses plain append.
    """

    def __init__(
        self,
        layout: WorkspaceLayout,
        provider: str,
        model: str,
        user: str | None = None,
        session_ts: str | None = None,
    ) -> None:
        self.layout = layout
        self.provider = provider
        self.model = model
        self.user = user or os.environ.get("USER", "unknown")
        self.session_ts = session_ts or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._calls_path: Path | None = None
        self._stats_path: Path | None = None

    # ---- public API ---------------------------------------------------------
    def log(self, request: dict[str, Any], response: dict[str, Any]) -> None:
        rec = ApiCallRecord(model=self.model, user=self.user, request=request, response=response)
        line = rec.model_dump_json() + "\n"
        path = self._ensure_calls_path()
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
        self._update_stats(rec)

    def close(self) -> None:
        # Hook for future flush/rotation; nothing to do in the skeleton.
        pass

    # ---- file path layout ---------------------------------------------------
    @staticmethod
    def _sanitize(s: str) -> str:
        """Strip leading provider prefix and replace `/` with `-` so path components
        don't get split into nested dirs (e.g. `google/gemma-4-31b-it` is one model,
        not two directory levels)."""
        return s.replace("/", "-").strip("-")

    def _ensure_calls_path(self) -> Path:
        if self._calls_path is None:
            now = datetime.now()
            ym = now.strftime("%Y-%m")
            d = now.strftime("%Y-%m-%d")
            base = (self.layout.api_calls_dir / self._sanitize(self.provider)
                    / self._sanitize(self.model) / ym)
            base.mkdir(parents=True, exist_ok=True)
            self._calls_path = base / f"{d}.{self.session_ts}.jsonl"
        return self._calls_path

    def _ensure_stats_path(self) -> Path:
        if self._stats_path is None:
            now = datetime.now()
            ym = now.strftime("%Y-%m")
            base = (self.layout.api_stats_dir / self._sanitize(self.provider)
                    / self._sanitize(self.model))
            base.mkdir(parents=True, exist_ok=True)
            self._stats_path = base / f"{self.user}_{ym}.{self.session_ts}.jsonl"
        return self._stats_path

    def _update_stats(self, rec: ApiCallRecord) -> None:
        """Append a compact stats line (usage only) for fast aggregation."""
        usage = (rec.response or {}).get("usage", {})
        stats_line = json.dumps(
            {
                "timestamp": rec.timestamp,
                "model": rec.model,
                "user": rec.user,
                "usage": usage,
            },
            ensure_ascii=False,
        ) + "\n"
        path = self._ensure_stats_path()
        with path.open("a", encoding="utf-8") as f:
            f.write(stats_line)
