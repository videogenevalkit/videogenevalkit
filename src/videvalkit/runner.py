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
    judge_override: dict[str, Any] | None = None,
    profile: str | None = None,
    subset_path: str | Path | None = None,
    aggregator: str | None = None,
    scheduler_config: SchedulerConfig | None = None,
    **adapter_kwargs: Any,
) -> dict[str, Any]:
    """Run one benchmark end-to-end.

    Returns a dict with keys: ``summary`` (Summary), ``raw_paths`` (list[str]),
    ``workspace`` (str).

    ``judge_override`` is an ad-hoc judge config dict that bypasses the registry
    (used by CLI ``--judge-endpoint`` / ``--judge-model`` / ``--judge-kind``).
    Mutually exclusive with ``judge``.

    ``profile`` selects an eval profile (``quick`` / ``standard`` / ``full``);
    defaults to ``"full"`` for back-compat. Profile choice affects frame
    sampling, samples-per-prompt, and the default subset to apply.

    ``subset_path`` overrides the profile's default subset by loading a Subset
    file directly. Use this for one-off custom subsets.
    """
    if benchmark not in SUPPORTED_BENCHMARKS:
        raise KeyError(f"unknown benchmark {benchmark!r}; known: {list(SUPPORTED_BENCHMARKS)}")
    bench_cfg = SUPPORTED_BENCHMARKS[benchmark]

    # Judge resolution: "paper" / "default" / "<name>" / None → concrete cfg dict.
    # ad-hoc judge_override bypasses the registry. See JUDGE_SELECTION_DESIGN §3.
    from videvalkit.configs.judge_loader import resolve_judge
    judge_cfg = resolve_judge(
        benchmark=benchmark, judge_name=judge, judge_override=judge_override
    )

    # Profile resolution: None → "full" (back-compat). See QUICK_EVAL_DESIGN §3.
    from videvalkit.core.profile import resolve_profile
    profile_spec = resolve_profile(profile)

    # Subset resolution: --subset PATH > profile's named subset > None
    subset_obj = None
    if subset_path is not None:
        from videvalkit.core.subset import Subset
        subset_obj = Subset.from_file(subset_path)
    elif profile_spec.subset is not None:
        from videvalkit.core.subset import find_subset
        try:
            subset_obj = find_subset(benchmark, profile_spec.subset)
        except FileNotFoundError:
            # Profile's named subset not available yet → run full corpus + warn
            # (this is the v0.2 reality until subset_v1.json files land per-bench)
            import logging
            logging.getLogger(__name__).warning(
                "videvalkit: profile %r expects subset %r for bench %r, "
                "but no subset file found. Running full corpus. "
                "Subset files land in D3 follow-up [per-bench calibration].",
                profile_spec.name, profile_spec.subset, benchmark,
            )

    aggregator_name = aggregator or bench_cfg["default_aggregator"]
    if aggregator_name not in SUPPORTED_AGGREGATORS:
        raise KeyError(f"unknown aggregator {aggregator_name!r}")

    ws = Workspace(workspace)
    sched = Scheduler(scheduler_config or SchedulerConfig())

    # Auto-stage videos into the workspace layout when --videos points elsewhere.
    # Bench adapters look for videos at <workspace>/videos/<model>/*.mp4 (the
    # layout the scheduler/resume machinery is built around). No adapter today
    # honors external_videos_dir, so we symlink the external --videos tree into
    # the workspace ourselves — one symlink covers all benches uniformly.
    ext = Path(videos).resolve()
    ws_videos = ws.layout.videos_dir.resolve()
    if ext != ws_videos:
        adapter_kwargs.setdefault("external_videos_dir", str(ext))
        ws_videos.mkdir(parents=True, exist_ok=True)
        children = [p for p in ext.iterdir() if p.is_dir()] if ext.is_dir() else []
        if children:
            # Standard layout: <ext>/<model>/*.mp4 — symlink each model dir.
            for src in children:
                dst = ws_videos / src.name
                if dst.is_symlink() or dst.exists():
                    if dst.is_symlink() and dst.resolve() == src:
                        continue
                    dst.unlink() if dst.is_symlink() or dst.is_file() else None
                dst.symlink_to(src)
        elif ext.is_dir():
            # Flat dir of *.mp4 (no model subdir). Stage under the first model
            # the caller named, or 'default' if none. Lets `metric run`-style
            # flat inputs still work via the bench path.
            model_name = (models or ["default"])[0]
            dst = ws_videos / model_name
            if dst.is_symlink() or dst.exists():
                dst.unlink() if dst.is_symlink() else None
            if not dst.exists():
                dst.symlink_to(ext)

    # Profile-level overrides flow through to adapters via kwargs so each
    # benchmark adapter can honor them (frame_sampling, samples_per_prompt).
    profile_payload = {
        "name": profile_spec.name,
        "frame_sampling": profile_spec.frame_sampling.model_dump(),
        "samples_per_prompt": profile_spec.samples_per_prompt,
    }
    subset_payload = None
    if subset_obj is not None:
        subset_payload = {
            "name": subset_obj.name,
            "n_prompts": subset_obj.n_prompts,
            "prompt_ids": list(subset_obj.spec.prompt_ids),
            "hash": subset_obj.hash(),
        }

    payload = {
        "videos_root": str(ws.layout.videos_dir),
        "workspace_root": str(ws.layout.root),
        "models": models,
        "dimensions": dimensions,
        "judge": judge_cfg,
        "aggregator": aggregator_name,
        "profile": profile_payload,
        "subset": subset_payload,
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
