"""AnthropicVLMJudge — judge backed by the official anthropic SDK.

Anthropic doesn't yet accept video natively; we pass frames as a sequence
of image content blocks (base64 jpeg). This matches what WorldJen's A4
cross-VLM ablation does for Claude Sonnet.
"""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any

from videvalkit.core.scorer import BaseScorer, ScoreContext, ScoreResult
from videvalkit.utils.video import extract_frames, strip_code_fences


class AnthropicVLMJudge(BaseScorer):
    kind = "vlm_judge_api"

    def __init__(
        self,
        name: str,
        model: str = "claude-sonnet-4-6",
        api_key_env: str = "ANTHROPIC_API_KEY",
        sys_prompt: str | Path | None = None,
        logger: Any | None = None,
        frame_cache: Any | None = None,
        request_timeout_s: float = 240.0,
    ) -> None:
        self.name = name
        self.model = model
        self.provider = "anthropic"
        self.api_key_env = api_key_env
        self.sys_prompt_path = Path(sys_prompt) if sys_prompt else None
        self._sys_prompt: str | None = None
        self.logger = logger
        self.frame_cache = frame_cache
        self.request_timeout_s = request_timeout_s
        self._client = None

    def setup(self) -> None:
        if self.sys_prompt_path and self._sys_prompt is None:
            self._sys_prompt = self.sys_prompt_path.read_text(encoding="utf-8")

    def set_logger(self, logger: Any) -> None:
        self.logger = logger

    def set_dispatcher(self, dispatcher: Any) -> None:
        pass  # SDK manages transport

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "AnthropicVLMJudge requires `anthropic` — install in the worldjen env"
            ) from e
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"missing env var {self.api_key_env}")
        self._client = anthropic.Anthropic(api_key=api_key, timeout=self.request_timeout_s)
        return self._client

    # ---- public callable API ------------------------------------------------
    def chat_text(
        self,
        user: str,
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        client = self._ensure_client()
        sys_prompt = system or self._sys_prompt or ""
        resp = client.messages.create(
            model=self.model,
            system=sys_prompt,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        normalized = self._normalize(resp)
        self._log({"kind": "text", "user": user[:200]},
                  [{"role": "user", "content": user}], normalized)
        return normalized

    def chat_with_frames(
        self,
        video_path: str | Path,
        prompt: str,
        mode: str = "holistic",
        n_frames: int = 8,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        client = self._ensure_client()
        frames = (
            self.frame_cache.get_or_extract(video_path, mode=mode, n_frames=n_frames)
            if self.frame_cache is not None
            else extract_frames(video_path, mode=mode, n_frames=n_frames)
        )
        if not frames:
            raise RuntimeError(f"no frames extracted from {video_path}")
        content: list[dict[str, Any]] = []
        for img in frames:
            if img.mode != "RGB":
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.b64encode(buf.getvalue()).decode("ascii"),
                }
            })
        content.append({"type": "text", "text": prompt})
        resp = client.messages.create(
            model=self.model,
            system=self._sys_prompt or "",
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        normalized = self._normalize(resp)
        log_content = [{"type": "image", "source": "<elided>"} for _ in frames] + [{"type": "text", "text": prompt}]
        self._log(
            {"kind": "multimodal", "video_path": str(video_path), "n_frames": len(frames), "mode": mode},
            [{"role": "user", "content": log_content}], normalized,
        )
        return normalized

    async def achat_text(self, user: str, system: str | None = None, **kw: Any) -> dict[str, Any]:
        import asyncio
        return await asyncio.to_thread(self.chat_text, user, system=system, **kw)

    async def achat_with_frames(
        self, video_path, prompt, mode="holistic", n_frames=8, **kw,
    ) -> dict[str, Any]:
        import asyncio
        return await asyncio.to_thread(
            self.chat_with_frames, video_path, prompt, mode=mode, n_frames=n_frames, **kw,
        )

    # ---- BaseScorer interface ----------------------------------------------
    def score(self, ctx: ScoreContext) -> ScoreResult:
        self.setup()
        prompt = ctx.meta.get("prompt") or self._sys_prompt or ctx.prompt_text
        mode = ctx.meta.get("mode", "holistic")
        n_frames = int(ctx.meta.get("n_frames", 8))
        resp = self.chat_with_frames(ctx.video_path, prompt, mode=mode, n_frames=n_frames)
        try:
            parsed = json.loads(strip_code_fences(resp["content"]))
        except (json.JSONDecodeError, KeyError):
            parsed = {"raw_content": resp.get("content", "")}
        return ScoreResult(score=self._extract_score(parsed), raw=parsed,
                           meta={"usage": resp.get("usage", {})})

    @staticmethod
    def _extract_score(parsed: Any) -> float | dict[str, Any]:
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            vals = [item.get("score") for item in parsed
                    if isinstance(item.get("score"), (int, float))]
            if vals:
                return sum(vals) / len(vals)
        if isinstance(parsed, dict) and isinstance(parsed.get("score"), (int, float)):
            return float(parsed["score"])
        return parsed if isinstance(parsed, dict) else {"raw": parsed}

    @staticmethod
    def _normalize(resp: Any) -> dict[str, Any]:
        try:
            blocks = resp.content or []
            text = next((b.text for b in blocks if getattr(b, "type", "") == "text"), "")
        except Exception:
            text = ""
        usage = {}
        try:
            u = resp.usage
            usage = {
                "prompt_tokens":     getattr(u, "input_tokens", 0),
                "completion_tokens": getattr(u, "output_tokens", 0),
                "total_tokens":      getattr(u, "input_tokens", 0) + getattr(u, "output_tokens", 0),
            }
        except Exception:
            pass
        return {"content": text, "usage": usage, "model": getattr(resp, "model", "")}

    def _log(self, request_meta, messages, response):
        if self.logger is None:
            return
        self.logger.log(
            request={**request_meta, "messages": messages, "model": self.model},
            response={"usage": response.get("usage", {}), "response": response.get("content", "")},
        )
