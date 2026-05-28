"""videvalkit.training — training-loop integration for periodic eval.

Per docs/QUICK_EVAL_DESIGN.md §6 (user 2026-05-20 confirmed):

Lets a training script run a quick eval every N steps without shelling out:

    from videvalkit.training import monitor, MonitorConfig
    cfg = MonitorConfig(
        benches=["vbench", "worldjen"],
        metrics=["fvd", "motion-smoothness"],
        profile="quick",
        workspace="/data/run_42/eval",
    )
    for step in range(0, 100_000, 1000):
        if step % 5000 == 0:
            prompts = monitor.preview_prompts(cfg)          # which prompts to gen
            videos_dir = generate_videos(prompts)
            result = monitor.eval(videos_dir, model_name=f"step_{step}", cfg=cfg)
            tb.add_scalar("eval/overall", result.overall, step)

Design notes:
  * No PyTorch / HF Trainer callback — user calls monitor.eval() explicitly
  * preview_prompts() tells the trainer which prompts to generate for
  * Results appended to <workspace>/timeline.jsonl [one line per checkpoint]
  * Cross-bench overall = mean of per-bench overall [z-score normalisation
    is a v0.3 follow-up; v0.2 uses raw mean for trend monitoring]
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger(__name__)

__all__ = ["MonitorConfig", "MonitorResult", "monitor"]


class MonitorConfig(BaseModel):
    """Configuration for training-time eval monitoring."""

    model_config = ConfigDict(extra="forbid")

    benches: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    metric_refs: dict[str, str] = Field(default_factory=dict)
    profile: str = "quick"
    judge: str | None = None
    workspace: str = "./videvalkit_monitor"

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "MonitorConfig":
        import json as _json
        return cls(**_json.loads(Path(path).read_text()))


class MonitorResult(BaseModel):
    """One monitoring step's result."""

    model_config = ConfigDict(extra="allow")

    model_name: str
    step: int | None = None
    summary: dict[str, Any] = Field(default_factory=dict)   # bench → summary dict
    metrics: dict[str, Any] = Field(default_factory=dict)   # metric → result dict
    overall: float = 0.0


class _Monitor:
    """Stateless monitor object exposed as the module-level ``monitor``."""

    def preview_prompts(self, cfg: "MonitorConfig") -> list[Any]:
        """Return the union of prompts the configured benches+profile will need.

        The training loop uses this to know which prompts to generate videos
        for [so the eval has videos to score]. De-duplicated across benches by
        prompt text.
        """
        from videvalkit.configs import SUPPORTED_BENCHMARKS
        from videvalkit.core.profile import resolve_profile

        profile_spec = resolve_profile(cfg.profile)
        seen_texts: set[str] = set()
        out: list[Any] = []
        for bench_name in cfg.benches:
            if bench_name not in SUPPORTED_BENCHMARKS:
                log.warning("monitor: unknown bench %r, skipping", bench_name)
                continue
            bench_cls = SUPPORTED_BENCHMARKS[bench_name]["cls"]
            try:
                bench = bench_cls()
                prompts = list(bench.list_prompts())
            except Exception as e:
                log.warning(
                    "monitor: could not list prompts for %r: %s", bench_name, e
                )
                continue
            # Apply subset if profile declares one
            if profile_spec.subset is not None:
                from videvalkit.core.subset import find_subset
                try:
                    subset = find_subset(bench_name, profile_spec.subset)
                    prompts = subset.filter_prompts(prompts)
                except FileNotFoundError:
                    pass  # no subset file yet → full
            for p in prompts:
                if p.text not in seen_texts:
                    seen_texts.add(p.text)
                    out.append(p)
        return out

    def eval(
        self,
        videos_dir: str | Path,
        model_name: str,
        cfg: "MonitorConfig",
        step: int | None = None,
    ) -> MonitorResult:
        """Run the configured benches + metrics on ``videos_dir``; append to
        timeline.jsonl; return MonitorResult.

        Each bench runs via runner.run with the configured profile + judge.
        Standalone metrics run via the metrics module [where backbones are
        available; shells raise NotImplementedError, captured as errors].
        """
        from videvalkit.runner import run as run_bench

        videos_dir = Path(videos_dir)
        ws = Path(cfg.workspace)
        ws.mkdir(parents=True, exist_ok=True)

        summary: dict[str, Any] = {}
        bench_overalls: list[float] = []

        # 1. Run each benchmark
        for bench_name in cfg.benches:
            try:
                result = run_bench(
                    benchmark=bench_name,
                    videos=videos_dir,
                    workspace=ws / bench_name,
                    models=[model_name],
                    judge=cfg.judge,
                    profile=cfg.profile,
                )
                bench_summary = result.get("summary") or {}
                summary[bench_name] = bench_summary
                # Extract overall if present [summary is {model: {...}} shape]
                for _model, s in (bench_summary or {}).items():
                    ov = s.get("overall") if isinstance(s, dict) else None
                    if isinstance(ov, (int, float)):
                        bench_overalls.append(float(ov))
            except Exception as e:
                log.warning("monitor: bench %r failed: %s", bench_name, e)
                summary[bench_name] = {"error": str(e)}

        # 2. Run standalone metrics [best-effort; shells will raise]
        metric_results: dict[str, Any] = {}
        for metric_name in cfg.metrics:
            try:
                from videvalkit.metrics import get_metric
                m = get_metric(metric_name)
                # We don't have a generic invocation path yet [needs refs/prompts
                # plumbing]; record that the metric is configured but defer the
                # actual call to when its dispatch lands [M3 cont].
                metric_results[metric_name] = {
                    "status": "configured",
                    "note": "metric dispatch from monitor lands with M3 backbone work",
                }
            except Exception as e:
                metric_results[metric_name] = {"error": str(e)}

        overall = (
            sum(bench_overalls) / len(bench_overalls) if bench_overalls else 0.0
        )

        result = MonitorResult(
            model_name=model_name,
            step=step,
            summary=summary,
            metrics=metric_results,
            overall=overall,
        )

        # 3. Append to timeline.jsonl
        timeline = ws / "timeline.jsonl"
        with timeline.open("a", encoding="utf-8") as f:
            line = {
                "model_name": model_name,
                "step": step,
                "profile": cfg.profile,
                "overall": overall,
                "bench_overalls": {
                    b: summary.get(b) for b in cfg.benches
                },
            }
            f.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")

        return result


# Module-level singleton — `from videvalkit.training import monitor`
monitor = _Monitor()
