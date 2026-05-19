"""GPUWorkerPool — persistent worker-per-GPU pool for GPU-bound scorers.

One subprocess per GPU slot. Each worker boots once, loads weights for any
scorer it sees, and stays warm for the rest of the run.

Communication: stdlib `multiprocessing.Queue`. Tasks are tuples
`(task_id, scorer_pickle, ctx_pickle)`. Results: `(task_id, ok, payload)`.

The scorer is pickled and shipped on first use; the worker caches it by
name so subsequent calls reuse the loaded model.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import pickle
import queue
import threading
import time
import uuid
from typing import Any

from videvalkit.core.scorer import BaseScorer, ScoreContext, ScoreResult

log = logging.getLogger(__name__)


def _worker_main(
    gpu_id: int,
    in_q: "mp.Queue[Any]",
    out_q: "mp.Queue[Any]",
) -> None:  # pragma: no cover - runs in subprocess
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cache: dict[str, BaseScorer] = {}
    while True:
        item = in_q.get()
        if item is None:
            return
        task_id, scorer_blob, ctx_blob = item
        try:
            scorer: BaseScorer = pickle.loads(scorer_blob)
            ctx: ScoreContext = pickle.loads(ctx_blob)
            if scorer.name in cache:
                scorer = cache[scorer.name]
            else:
                scorer.setup()
                cache[scorer.name] = scorer
            res = scorer.score(ctx)
            out_q.put((task_id, True, pickle.dumps(res)))
        except Exception as e:
            out_q.put((task_id, False, repr(e)))


class GPUWorkerPool:
    """Pool of persistent GPU workers; one subprocess per slot.

    Slots = `len(devices) * workers_per_device`. Each slot is pinned to one
    GPU id via CUDA_VISIBLE_DEVICES.
    """

    def __init__(
        self,
        devices: list[int] | None = None,
        workers_per_device: int = 1,
        spawn: bool = True,
    ) -> None:
        if devices is None:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            devices = [int(d) for d in visible.split(",") if d.strip()] if visible else []
        self.devices = devices
        self.workers_per_device = workers_per_device
        self._ctx = mp.get_context("spawn" if spawn else "fork")
        self._in_q: "mp.Queue[Any]" = self._ctx.Queue()
        self._out_q: "mp.Queue[Any]" = self._ctx.Queue()
        self._workers: list[Any] = []
        self._pending: dict[str, threading.Event] = {}
        self._results: dict[str, Any] = {}
        self._reaper_thread: threading.Thread | None = None
        self._closed = False

    # ---- lifecycle ----------------------------------------------------------
    def _ensure_workers(self) -> None:
        if self._workers:
            return
        if not self.devices:
            log.warning("GPUWorkerPool: no GPUs visible; falling back to in-process execution")
            return
        for gpu_id in self.devices:
            for _ in range(self.workers_per_device):
                p = self._ctx.Process(
                    target=_worker_main, args=(gpu_id, self._in_q, self._out_q), daemon=True,
                )
                p.start()
                self._workers.append(p)
        self._reaper_thread = threading.Thread(target=self._reaper, daemon=True)
        self._reaper_thread.start()

    def _reaper(self) -> None:
        while not self._closed:
            try:
                task_id, ok, payload = self._out_q.get(timeout=1.0)
            except queue.Empty:
                continue
            self._results[task_id] = (ok, payload)
            ev = self._pending.pop(task_id, None)
            if ev:
                ev.set()

    # ---- submit -------------------------------------------------------------
    def submit(self, scorer: BaseScorer, ctx: ScoreContext) -> ScoreResult:
        """Synchronous submit: enqueue task, wait for the matching result."""
        if not self.devices:
            scorer.setup()
            return scorer.score(ctx)
        self._ensure_workers()
        task_id = uuid.uuid4().hex
        ev = threading.Event()
        self._pending[task_id] = ev
        self._in_q.put((task_id, pickle.dumps(scorer), pickle.dumps(ctx)))
        if not ev.wait(timeout=3600):
            raise TimeoutError(f"GPU task {task_id} timed out")
        ok, payload = self._results.pop(task_id)
        if not ok:
            raise RuntimeError(f"GPU worker error: {payload}")
        return pickle.loads(payload)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for _ in self._workers:
            try:
                self._in_q.put(None)
            except Exception:
                pass
        for p in self._workers:
            try:
                p.join(timeout=5)
            except Exception:
                pass
        if self._reaper_thread is not None:
            # let the reaper time-out and exit
            time.sleep(0.05)
        self._workers.clear()
