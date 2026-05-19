"""BaseBenchmark — the top-level adapter abstraction.

Each upstream benchmark (VBench / VBench-2.0 / Video-Bench / WorldJen) is
exposed by subclassing this and filling in the four required methods.

The toolkit handles the rest: dispatching across the right conda env,
materializing inputs in the workspace, capturing API logs, and aggregating
final scores.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from videvalkit.core.layout import WorkspaceLayout  # noqa: F401  (re-exported via docs)
from videvalkit.core.types import PromptItem, RawResult, Summary, VideoSpec


class BaseBenchmark(ABC):
    """Adapter over an upstream video-generation benchmark.

    Subclasses MUST set:
      * `name`         — short identifier, e.g. ``"vbench"``
      * `env_name`     — conda env name the adapter runs in
      * `dimensions`   — full list of dimension names

    Subclasses MUST implement:
      * `list_prompts()`         — yield PromptItem for each prompt
      * `list_required_videos()` — say what videos must exist on disk
      * `evaluate()`             — invoke the upstream pipeline (usually inside env_name)
      * `aggregate()`            — collapse raw results into a Summary

    Subclasses MAY override:
      * `export_official()`      — produce a leaderboard-compatible artifact
      * `video_layout`           — override the default video glob pattern
    """

    name: str = "base"
    env_name: str | None = None
    dimensions: list[str] = []
    video_layout: str = "{model}/{prompt_id}-{sample_index}.mp4"

    @abstractmethod
    def list_prompts(
        self, dimensions: list[str] | None = None
    ) -> Iterator[PromptItem]:  # pragma: no cover - abstract
        ...

    @abstractmethod
    def list_required_videos(
        self,
        prompts: list[PromptItem],
        models: list[str],
        layout: WorkspaceLayout,
        samples_per_prompt: int = 1,
    ) -> list[VideoSpec]:  # pragma: no cover - abstract
        ...

    @abstractmethod
    def evaluate(
        self,
        videos: list[VideoSpec],
        layout: WorkspaceLayout,
        dimensions: list[str] | None = None,
        judge: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[RawResult]:  # pragma: no cover - abstract
        """Run the upstream benchmark's scoring on the given videos.

        Implementations typically:
          1. Map toolkit-standard inputs onto the upstream pipeline's expected layout
             (often via symlinks under a temp dir).
          2. Invoke the upstream Python API (preferred) or CLI.
          3. Normalize the upstream output into a list of RawResult.

        Long-running adapters should yield RawResult by writing each one to disk
        via the workspace layout as soon as it is produced (resume-friendly),
        in addition to returning the full list.
        """

    @abstractmethod
    def aggregate(
        self,
        raw: list[RawResult],
        aggregator: str = "weighted_sum",
        **kwargs: Any,
    ) -> Summary:  # pragma: no cover - abstract
        ...

    def export_official(self, summary: Summary, out_path: Path) -> None:
        """Optional: export the summary in the upstream's leaderboard-submission format."""
        raise NotImplementedError(
            f"{self.name}.export_official is not implemented yet"
        )

    # ---- runner integration: convenience end-to-end --------------------------
    def evaluate_and_aggregate(
        self,
        videos_root: str | None = None,
        workspace_root: str | None = None,
        models: list[str] | None = None,
        dimensions: list[str] | None = None,
        judge: dict[str, Any] | None = None,
        aggregator: str = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Default end-to-end: evaluate() → aggregate() → write summary.

        Called by Runner.run() through the subprocess entry point. Returns a
        JSON-safe dict so it can cross the env boundary intact.
        """
        from videvalkit.core.layout import WorkspaceLayout
        from videvalkit.storage.workspace import Workspace

        if workspace_root is None:
            raise ValueError("workspace_root is required")
        ws = Workspace(workspace_root)
        layout = ws.layout

        raw = self.evaluate(
            layout=layout, dimensions=dimensions, judge=judge, models=models, **kwargs,
        )
        if not raw:
            return {"summary": None, "raw_paths": [], "n_results": 0}

        # group by model and aggregate per model
        from collections import defaultdict
        by_model: dict[str, list[RawResult]] = defaultdict(list)
        for r in raw:
            by_model[r.model].append(r)

        summaries: dict[str, dict[str, Any]] = {}
        for model, items in by_model.items():
            s = self.aggregate(items, aggregator=aggregator or "weighted_sum")
            ws.write_summary(s)
            summaries[model] = s.model_dump(mode="json")
        return {
            "summary": summaries,
            "raw_paths": [
                str(layout.raw_path(self.name, r.model, r.dimension, r.prompt_id))
                for r in raw
            ],
            "n_results": len(raw),
        }
