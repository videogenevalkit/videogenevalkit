"""OpenAI-compatible VLM judge — works for vLLM, SGLang, Ollama, OpenAI, etc.

The scorer composes:
  * Frame extraction from video path (mode + n_frames configurable per call)
  * A user message: text + N image_url blocks (base64 data URLs)
  * POST to {endpoint}/chat/completions through HTTPDispatcher
  * Optional ApiCallLogger sink (input/output JSON per call)

Two entry points:
  * `judge.chat_text(...)` — text-only call (for VQA generation)
  * `judge.chat_with_frames(video_path, prompt, mode, n_frames)` — multimodal
  * `score(ctx)` — BaseScorer interface; uses ctx.meta to pick mode + prompt
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from videvalkit.core.scorer import BaseScorer, ScoreContext, ScoreResult
from videvalkit.utils.video import (
    extract_frames,
    pil_to_data_url,
    strip_code_fences,
)


class OpenAICompatibleVLMJudge(BaseScorer):
    """OpenAI-style chat completions client, sync or async.

    Construction kwargs (also accepted from `SUPPORTED_JUDGES[name]`):
      * endpoint   — base URL ending in "/v1"
      * model      — model id the endpoint expects
      * api_key_env — env var holding the bearer token (None for local)
      * provider   — used by ApiCallLogger for partitioning
      * sys_prompt — optional system prompt path
      * logger     — ApiCallLogger instance for input/output JSON logs
      * dispatcher — HTTPDispatcher (injected by Scheduler in async use)
    """

    kind = "vlm_judge_http"

    def __init__(
        self,
        name: str,
        endpoint: str,
        model: str,
        api_key_env: str | None = None,
        provider: str = "openai_compat",
        sys_prompt: str | Path | None = None,
        logger: Any | None = None,
        dispatcher: Any | None = None,
        frame_cache: Any | None = None,
        request_timeout_s: float = 240.0,
        **_unused: Any,
    ) -> None:
        self.name = name
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.provider = provider
        self.sys_prompt_path = Path(sys_prompt) if sys_prompt else None
        self.logger = logger
        self.dispatcher = dispatcher
        self.frame_cache = frame_cache
        self.request_timeout_s = request_timeout_s
        self._sys_prompt: str | None = None

    # ---- public callable API ------------------------------------------------
    def set_dispatcher(self, dispatcher: Any) -> None:
        self.dispatcher = dispatcher

    def set_logger(self, logger: Any) -> None:
        self.logger = logger

    def setup(self) -> None:
        if self.sys_prompt_path and self._sys_prompt is None:
            self._sys_prompt = self.sys_prompt_path.read_text(encoding="utf-8")

    def chat_text(
        self,
        user: str,
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Synchronous text-only completion."""
        messages: list[dict[str, Any]] = []
        if system or self._sys_prompt:
            messages.append({"role": "system", "content": system or self._sys_prompt})
        messages.append({"role": "user", "content": user})
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **(extra or {}),
        }
        return self._post_sync(payload, request_meta={"kind": "text", "user": user[:200]})

    def _get_frames(self, video_path, mode: str, n_frames: int) -> list:
        if self.frame_cache is not None:
            return self.frame_cache.get_or_extract(video_path, mode=mode, n_frames=n_frames)
        return extract_frames(video_path, mode=mode, n_frames=n_frames)

    def chat_with_frames(
        self,
        video_path: str | Path,
        prompt: str,
        mode: str = "holistic",
        n_frames: int = 8,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        system: str | None = None,
    ) -> dict[str, Any]:
        """Synchronous multimodal call (frames + text)."""
        frames = self._get_frames(video_path, mode=mode, n_frames=n_frames)
        if not frames:
            raise RuntimeError(f"no frames extracted from {video_path}")
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in frames:
            content.append(
                {"type": "image_url", "image_url": {"url": pil_to_data_url(img)}}
            )
        messages: list[dict[str, Any]] = []
        if system or self._sys_prompt:
            messages.append({"role": "system", "content": system or self._sys_prompt})
        messages.append({"role": "user", "content": content})
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        return self._post_sync(
            payload,
            request_meta={
                "kind": "multimodal",
                "video_path": str(video_path),
                "n_frames": len(frames),
                "mode": mode,
            },
        )

    # ---- async variants -----------------------------------------------------
    async def achat_text(
        self,
        user: str,
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if system or self._sys_prompt:
            messages.append({"role": "system", "content": system or self._sys_prompt})
        messages.append({"role": "user", "content": user})
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **(extra or {}),
        }
        return await self._post_async(payload, request_meta={"kind": "text", "user": user[:200]})

    async def achat_with_frames(
        self,
        video_path: str | Path,
        prompt: str,
        mode: str = "holistic",
        n_frames: int = 8,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        system: str | None = None,
    ) -> dict[str, Any]:
        frames = await asyncio.to_thread(self._get_frames, video_path, mode, n_frames)
        if not frames:
            raise RuntimeError(f"no frames extracted from {video_path}")
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in frames:
            content.append(
                {"type": "image_url", "image_url": {"url": pil_to_data_url(img)}}
            )
        messages: list[dict[str, Any]] = []
        if system or self._sys_prompt:
            messages.append({"role": "system", "content": system or self._sys_prompt})
        messages.append({"role": "user", "content": content})
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        return await self._post_async(
            payload,
            request_meta={
                "kind": "multimodal",
                "video_path": str(video_path),
                "n_frames": len(frames),
                "mode": mode,
            },
        )

    # ---- BaseScorer interface ----------------------------------------------
    def score(self, ctx: ScoreContext) -> ScoreResult:
        """Default `score()` for adapters that just want one shot per video.

        Adapters with multi-stage logic (WorldJen has 16 dims per video)
        typically call `chat_with_frames` directly with a per-dim prompt.
        """
        self.setup()
        prompt = ctx.meta.get("prompt") or self._sys_prompt or ctx.prompt_text
        mode = ctx.meta.get("mode", "holistic")
        n_frames = int(ctx.meta.get("n_frames", 8))
        resp = self.chat_with_frames(ctx.video_path, prompt, mode=mode, n_frames=n_frames)
        try:
            parsed = json.loads(strip_code_fences(resp["content"]))
        except (json.JSONDecodeError, KeyError):
            parsed = {"raw_content": resp.get("content", "")}
        score = self._extract_score(parsed)
        return ScoreResult(score=score, raw=parsed, meta={"usage": resp.get("usage", {})})

    @staticmethod
    def _extract_score(parsed: Any) -> float | dict[str, Any]:
        """Best-effort numeric score extraction."""
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            vals = [item.get("score") for item in parsed if isinstance(item.get("score"), (int, float))]
            if vals:
                return sum(vals) / len(vals)
        if isinstance(parsed, dict) and isinstance(parsed.get("score"), (int, float)):
            return float(parsed["score"])
        return parsed if isinstance(parsed, dict) else {"raw": parsed}

    # ---- internals ----------------------------------------------------------
    def _api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f"missing env var {self.api_key_env} for {self.name}")
        return key

    def _post_sync(self, payload: dict[str, Any], request_meta: dict[str, Any]) -> dict[str, Any]:
        """Synchronous POST via `requests` — used when no async loop is available."""
        try:
            import requests
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("install `requests` for sync VLM judge calls") from e

        url = f"{self.endpoint}/chat/completions"
        headers = {"Content-Type": "application/json"}
        key = self._api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"

        r = requests.post(url, json=payload, headers=headers, timeout=self.request_timeout_s)
        r.raise_for_status()
        data = r.json()
        normalized = self._normalize_response(data)
        self._log(request_meta=request_meta, payload=payload, response=normalized, raw=data)
        return normalized

    async def _post_async(self, payload: dict[str, Any], request_meta: dict[str, Any]) -> dict[str, Any]:
        if self.dispatcher is None:
            raise RuntimeError(f"async path requires a dispatcher to be set on {self.name}")
        key = self._api_key()
        data = await self.dispatcher.post_chat(
            endpoint=self.endpoint,
            payload=payload,
            api_key=key,
            provider=self.provider,
        )
        normalized = self._normalize_response(data)
        self._log(request_meta=request_meta, payload=payload, response=normalized, raw=data)
        return normalized

    @staticmethod
    def _normalize_response(raw: dict[str, Any]) -> dict[str, Any]:
        """Pull the assistant message + usage into a flat dict (matches clients.py)."""
        try:
            msg = raw["choices"][0]["message"]
            content = msg.get("content", "") or ""
        except (KeyError, IndexError):
            content = ""
        return {
            "content": content,
            "usage": raw.get("usage", {}) or {},
            "model": raw.get("model", ""),
        }

    def _log(
        self,
        request_meta: dict[str, Any],
        payload: dict[str, Any],
        response: dict[str, Any],
        raw: dict[str, Any],
    ) -> None:
        if self.logger is None:
            return
        # Trim images from the logged request to keep jsonl small.
        logged_payload = dict(payload)
        msgs = logged_payload.get("messages", [])
        clean_messages: list[Any] = []
        for m in msgs:
            content = m.get("content")
            if isinstance(content, list):
                content = [c if c.get("type") != "image_url" else {"type": "image_url", "image_url": "<elided>"}
                           for c in content]
            clean_messages.append({"role": m.get("role"), "content": content})
        logged_payload["messages"] = clean_messages
        self.logger.log(
            request={**request_meta, "payload": logged_payload},
            response={"usage": response.get("usage", {}), "response": response.get("content", "")},
        )
