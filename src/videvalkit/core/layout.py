"""Workspace directory layout — the on-disk contract every component honors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkspaceLayout:
    """Resolves all conventional paths under a workspace root.

    Single source of truth: every component (storage writer, scorer, runner)
    asks the layout where to read/write, never hard-codes a path.
    """

    root: Path

    # ---- inputs --------------------------------------------------------------
    @property
    def videos_dir(self) -> Path:
        """`videos/{model}/{prompt_id}-{idx}.mp4` (user-provided)."""
        return self.root / "videos"

    @property
    def prompts_dir(self) -> Path:
        """`prompts/{benchmark}/prompts.jsonl`."""
        return self.root / "prompts"

    # ---- caches --------------------------------------------------------------
    @property
    def frames_cache_dir(self) -> Path:
        """Decoded frames cache, shared across scorers operating on the same video."""
        return self.root / "frames_cache"

    @property
    def model_cache_dir(self) -> Path:
        """Pretrained weight cache (CLIP, RAFT, LLaVA-Video, ...)."""
        return self.root / "model_cache"

    # ---- outputs -------------------------------------------------------------
    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    def raw_path(self, benchmark: str, model: str, dimension: str, prompt_id: str) -> Path:
        return self.results_dir / "raw" / benchmark / model / dimension / f"{prompt_id}.json"

    def summary_path(self, benchmark: str, model: str) -> Path:
        return self.results_dir / "summary" / benchmark / f"{model}.json"

    def leaderboard_path(self, benchmark: str, ext: str = "json") -> Path:
        return self.results_dir / "leaderboard" / f"{benchmark}_export.{ext}"

    # ---- api logs (mirror of video_eval/results/api_logs schema) -------------
    @property
    def api_logs_dir(self) -> Path:
        return self.root / "api_logs"

    @property
    def api_calls_dir(self) -> Path:
        return self.api_logs_dir / "calls"

    @property
    def api_stats_dir(self) -> Path:
        return self.api_logs_dir / "stats"

    @property
    def api_zips_dir(self) -> Path:
        return self.api_logs_dir / "zips"

    # ---- bootstrap -----------------------------------------------------------
    def ensure(self) -> "WorkspaceLayout":
        """Create all directories that should exist before a run."""
        for p in [
            self.videos_dir,
            self.prompts_dir,
            self.frames_cache_dir,
            self.model_cache_dir,
            self.results_dir,
            self.api_logs_dir,
            self.api_calls_dir,
            self.api_stats_dir,
            self.api_zips_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)
        return self
