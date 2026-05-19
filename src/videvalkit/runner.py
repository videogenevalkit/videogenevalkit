"""High-level orchestrator — the function users call from Python.

Walks one (benchmark, models, videos) → (raw_results, summary) end-to-end:

  1. resolve benchmark + judge + aggregator from the config registries
  2. bootstrap the workspace
  3. dispatch the adapter into its conda env via the Scheduler
  4. collect raw results, run the aggregator, write summary
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from videvalkit.configs import (
    SUPPORTED_AGGREGATORS,
    SUPPORTED_BENCHMARKS,
    SUPPORTED_JUDGES,
)
from videvalkit.scheduler import Scheduler, SchedulerConfig
from videvalkit.storage import Workspace


def run(
    benchmark: str,
    videos: str | Path,
    workspace: str | Path,
    models: list[str] | None = None,
    dimensions: list[str] | None = None,
    judge: str | None = None,
    aggregator: str | None = None,
    scheduler_config: SchedulerConfig | None = None,
    **adapter_kwargs: Any,
) -> dict[str, Any]:
    """Run one benchmark end-to-end.

    Returns a dict with keys: ``summary`` (Summary), ``raw_paths`` (list[str]),
    ``workspace`` (str).
    """
    if benchmark not in SUPPORTED_BENCHMARKS:
        raise KeyError(f"unknown benchmark {benchmark!r}; known: {list(SUPPORTED_BENCHMARKS)}")
    bench_cfg = SUPPORTED_BENCHMARKS[benchmark]

    judge_name = judge or bench_cfg.get("default_judge")
    if bench_cfg["needs_judge"] and judge_name is None:
        raise ValueError(f"{benchmark} requires a judge; none supplied")
    judge_cfg = SUPPORTED_JUDGES[judge_name] if judge_name else None

    aggregator_name = aggregator or bench_cfg["default_aggregator"]
    if aggregator_name not in SUPPORTED_AGGREGATORS:
        raise KeyError(f"unknown aggregator {aggregator_name!r}")

    ws = Workspace(workspace)
    sched = Scheduler(scheduler_config or SchedulerConfig())

    # Stage out the inputs link in case caller pointed at a different dir.
    if Path(videos).resolve() != ws.layout.videos_dir.resolve():
        adapter_kwargs.setdefault("external_videos_dir", str(Path(videos).resolve()))

    payload = {
        "videos_root": str(ws.layout.videos_dir),
        "workspace_root": str(ws.layout.root),
        "models": models,
        "dimensions": dimensions,
        "judge": judge_cfg,
        "aggregator": aggregator_name,
        **adapter_kwargs,
    }

    try:
        result = sched.run_in_env(
            env_name=bench_cfg["env"],
            benchmark=benchmark,
            method="evaluate_and_aggregate",
            payload=payload,
        )
    finally:
        sched.close()

    return {
        "summary": result.get("summary"),
        "raw_paths": result.get("raw_paths", []),
        "workspace": str(ws.layout.root),
    }
