"""Scheduler — facade routing each Scorer/Benchmark call to the right pool."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from videvalkit.core.scorer import BaseScorer, ScoreContext, ScoreResult


@dataclass
class SchedulerConfig:
    # GPU pool
    gpu_devices: list[int] | None = None        # default: all CUDA_VISIBLE_DEVICES
    gpu_workers_per_device: int = 1
    # HTTP pool
    http_concurrency: int = 32
    http_timeout_s: float = 120.0
    http_max_retries: int = 5
    # Per-provider rate limits {provider: {"rpm": int, "tpm": int}}
    rate_limits: dict[str, dict[str, int]] = field(default_factory=dict)
    # Env dispatcher
    env_root: str | None = None                  # conda root; None -> use `conda` on PATH


class Scheduler:
    """Single entry point for routing work to the appropriate pool.

    Currently a stub — the actual pool implementations live next to this file
    and will be wired in M1 follow-up. The class is here so adapters can already
    program against the right interface.
    """

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        self.config = config or SchedulerConfig()
        self._gpu_pool = None       # lazily created
        self._http_pool = None      # lazily created
        self._env_dispatcher = None  # lazily created

    # ---- scorer-level dispatch ---------------------------------------------
    def submit_scorer(self, scorer: BaseScorer, ctx: ScoreContext) -> ScoreResult:
        """Dispatch one (scorer, ctx) pair according to scorer.kind.

        Synchronous in the skeleton; the async variant `asubmit_scorer` will
        coexist for the HTTP pool path.
        """
        if scorer.kind == "gpu_metric":
            return self._gpu().submit(scorer, ctx)
        elif scorer.kind in ("vlm_judge_http", "vlm_judge_api"):
            return self._http().submit_sync(scorer, ctx)
        elif scorer.kind == "cpu":
            return scorer.score(ctx)
        raise ValueError(f"unknown scorer.kind={scorer.kind!r}")

    # ---- benchmark-level dispatch (whole adapter call in its env) ----------
    def run_in_env(
        self,
        env_name: str,
        benchmark: str,
        method: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Invoke `videvalkit.benchmarks.entry` inside an isolated conda env."""
        return self._env().run(env_name, benchmark, method, payload)

    # ---- lazy pool accessors -----------------------------------------------
    def _gpu(self):
        if self._gpu_pool is None:
            from videvalkit.scheduler.gpu_pool import GPUWorkerPool

            self._gpu_pool = GPUWorkerPool(
                devices=self.config.gpu_devices,
                workers_per_device=self.config.gpu_workers_per_device,
            )
        return self._gpu_pool

    def _http(self):
        if self._http_pool is None:
            from videvalkit.scheduler.http_pool import HTTPDispatcher

            self._http_pool = HTTPDispatcher(
                concurrency=self.config.http_concurrency,
                timeout_s=self.config.http_timeout_s,
                max_retries=self.config.http_max_retries,
                rate_limits=self.config.rate_limits,
            )
        return self._http_pool

    def _env(self):
        if self._env_dispatcher is None:
            from videvalkit.scheduler.env_dispatcher import CondaEnvDispatcher

            self._env_dispatcher = CondaEnvDispatcher(conda_root=self.config.env_root)
        return self._env_dispatcher

    def close(self) -> None:
        for pool in (self._gpu_pool, self._http_pool):
            if pool is not None and hasattr(pool, "close"):
                pool.close()
