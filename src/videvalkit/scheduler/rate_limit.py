"""TokenBucket — per-provider rate limiter for VLM API calls.

Async-aware: callers `await bucket.acquire(n_tokens)` and the bucket sleeps
until both the request-per-minute and token-per-minute budgets allow it.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    """Two parallel rate limits: RPM (requests) and TPM (tokens-per-minute).

    Both default to 0 = unlimited. Refill is continuous (fractional).
    """

    rate_per_min: int = 0
    token_rate_per_min: int = 0

    _last_refill: float = field(default=0.0, init=False)
    _req_balance: float = field(default=0.0, init=False)
    _tok_balance: float = field(default=0.0, init=False)
    _lock: asyncio.Lock = field(default=None, init=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._last_refill = time.monotonic()
        self._req_balance = float(self.rate_per_min)
        self._tok_balance = float(self.token_rate_per_min)

    def _ensure_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now
        if self.rate_per_min:
            self._req_balance = min(
                float(self.rate_per_min),
                self._req_balance + self.rate_per_min * elapsed / 60.0,
            )
        if self.token_rate_per_min:
            self._tok_balance = min(
                float(self.token_rate_per_min),
                self._tok_balance + self.token_rate_per_min * elapsed / 60.0,
            )

    def try_acquire(self, tokens_estimate: int = 0) -> bool:
        """Sync non-blocking try-acquire. Used by callers that prefer their own loop."""
        self._refill()
        if self.rate_per_min and self._req_balance < 1.0:
            return False
        if self.token_rate_per_min and self._tok_balance < tokens_estimate:
            return False
        if self.rate_per_min:
            self._req_balance -= 1.0
        if self.token_rate_per_min:
            self._tok_balance -= tokens_estimate
        return True

    async def acquire(self, tokens_estimate: int = 0) -> None:
        """Async-blocking acquire: waits until both budgets allow the request."""
        if not self.rate_per_min and not self.token_rate_per_min:
            return
        lock = self._ensure_lock()
        while True:
            async with lock:
                self._refill()
                req_ok = (not self.rate_per_min) or self._req_balance >= 1.0
                tok_ok = (not self.token_rate_per_min) or self._tok_balance >= tokens_estimate
                if req_ok and tok_ok:
                    if self.rate_per_min:
                        self._req_balance -= 1.0
                    if self.token_rate_per_min:
                        self._tok_balance -= tokens_estimate
                    return
                # Sleep just long enough for the smaller deficit to refill.
                wait_req = (1.0 - self._req_balance) * 60.0 / self.rate_per_min if self.rate_per_min else 0
                wait_tok = (
                    (tokens_estimate - self._tok_balance) * 60.0 / self.token_rate_per_min
                    if self.token_rate_per_min else 0
                )
                wait = max(wait_req, wait_tok, 0.01)
            await asyncio.sleep(min(wait, 1.0))
