"""HTTPDispatcher — async dispatcher for VLM-judge / API workloads.

Built on aiohttp + asyncio.Semaphore for concurrency control + TokenBucket
for per-provider RPM/TPM limits + exponential-backoff retry on 429 / 5xx /
network errors.

Adapters typically interact with this via `OpenAICompatibleVLMJudge`, which
calls `dispatcher.post_chat(...)` directly. The Scheduler.submit_scorer
path is also available for the sync-style use.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

import aiohttp

from videvalkit.core.scorer import BaseScorer, ScoreContext, ScoreResult
from videvalkit.scheduler.rate_limit import TokenBucket

log = logging.getLogger(__name__)


class HTTPDispatcherError(RuntimeError):
    pass


class HTTPDispatcher:
    """Async, concurrency-bounded, retrying HTTP client for chat-completions APIs."""

    def __init__(
        self,
        concurrency: int = 32,
        timeout_s: float = 240.0,
        max_retries: int = 5,
        rate_limits: dict[str, dict[str, int]] | None = None,
    ) -> None:
        self.concurrency = concurrency
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._buckets: dict[str, TokenBucket] = {}
        for provider, limit in (rate_limits or {}).items():
            self._buckets[provider] = TokenBucket(
                rate_per_min=limit.get("rpm", 0),
                token_rate_per_min=limit.get("tpm", 0),
            )
        self._sem: asyncio.Semaphore | None = None
        self._session: aiohttp.ClientSession | None = None

    # ---- session lifecycle --------------------------------------------------
    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout_s)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    def _ensure_sem(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.concurrency)
        return self._sem

    async def aclose(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    def close(self) -> None:
        """Sync close (best-effort). Prefer `aclose` from inside an event loop."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.aclose())
            else:
                loop.run_until_complete(self.aclose())
        except RuntimeError:
            pass

    # ---- chat-completions POST ---------------------------------------------
    async def post_chat(
        self,
        endpoint: str,
        payload: dict[str, Any],
        api_key: str | None = None,
        provider: str = "openai_compat",
        tokens_estimate: int = 0,
    ) -> dict[str, Any]:
        """POST to {endpoint}/chat/completions with retry + rate-limiting.

        Returns the parsed JSON response. Raises HTTPDispatcherError after
        exhausting retries.
        """
        url = endpoint.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        sem = self._ensure_sem()
        bucket = self._buckets.get(provider)
        session = await self._ensure_session()

        for attempt in range(self.max_retries + 1):
            if bucket is not None:
                await bucket.acquire(tokens_estimate)
            async with sem:
                try:
                    async with session.post(url, json=payload, headers=headers) as resp:
                        if resp.status == 200:
                            return await resp.json()
                        text = (await resp.text())[:2000]
                        if resp.status == 429 or 500 <= resp.status < 600:
                            wait = self._backoff(attempt, base=10 if resp.status == 429 else 5)
                            log.warning(
                                "HTTP %s from %s (attempt %d/%d), backing off %.1fs: %s",
                                resp.status, url, attempt + 1, self.max_retries + 1, wait, text,
                            )
                            await asyncio.sleep(wait)
                            continue
                        raise HTTPDispatcherError(
                            f"POST {url} -> HTTP {resp.status}: {text}"
                        )
                except asyncio.TimeoutError:
                    wait = self._backoff(attempt, base=15)
                    log.warning(
                        "Timeout on %s (attempt %d/%d), backing off %.1fs",
                        url, attempt + 1, self.max_retries + 1, wait,
                    )
                    await asyncio.sleep(wait)
                except aiohttp.ClientError as e:
                    wait = self._backoff(attempt, base=5)
                    log.warning(
                        "Client error on %s (attempt %d/%d): %s; backoff %.1fs",
                        url, attempt + 1, self.max_retries + 1, e, wait,
                    )
                    await asyncio.sleep(wait)
        raise HTTPDispatcherError(
            f"POST {url} failed after {self.max_retries + 1} attempts"
        )

    @staticmethod
    def _backoff(attempt: int, base: float = 5.0, cap: float = 120.0) -> float:
        """Exponential backoff with full jitter."""
        return min(cap, base * (2 ** attempt)) * (0.5 + random.random() * 0.5)

    # ---- BaseScorer integration --------------------------------------------
    def submit_sync(self, scorer: BaseScorer, ctx: ScoreContext) -> ScoreResult:
        """Sync helper for callers that don't run inside an event loop."""
        return asyncio.run(self.asubmit(scorer, ctx))

    async def asubmit(self, scorer: BaseScorer, ctx: ScoreContext) -> ScoreResult:
        """Dispatch one scorer call. The scorer is responsible for calling
        `self.post_chat` itself (injected via `scorer._dispatcher`)."""
        # Inject dispatcher if the scorer wants async I/O.
        if hasattr(scorer, "set_dispatcher"):
            scorer.set_dispatcher(self)
        return scorer.score(ctx)

    async def amap(self, scorer: BaseScorer, contexts: list[ScoreContext]) -> list[ScoreResult]:
        return await asyncio.gather(*[self.asubmit(scorer, c) for c in contexts])

    def bucket_for(self, provider: str) -> TokenBucket | None:
        return self._buckets.get(provider)
