"""Unified workload scheduler.

Three worker pools coexist in one process:
  * GPUWorkerPool       — for `kind="gpu_metric"` scorers (VBench v1 etc.)
  * HTTPDispatcher      — for `kind="vlm_judge_http" | "vlm_judge_api"` scorers
  * CondaEnvDispatcher  — for whole-benchmark subprocess calls into an isolated env

`Scheduler` is the single entry point; callers don't pick a pool.
"""

from videvalkit.scheduler.base import Scheduler, SchedulerConfig
from videvalkit.scheduler.gpu_pool import GPUWorkerPool
from videvalkit.scheduler.http_pool import HTTPDispatcher
from videvalkit.scheduler.env_dispatcher import CondaEnvDispatcher
from videvalkit.scheduler.rate_limit import TokenBucket

__all__ = [
    "Scheduler",
    "SchedulerConfig",
    "GPUWorkerPool",
    "HTTPDispatcher",
    "CondaEnvDispatcher",
    "TokenBucket",
]
