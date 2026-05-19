"""GeminiVLMJudge — judge backed by google-genai SDK.

Gemini accepts mp4 video parts directly (via File API or inline bytes for
small clips). We use inline bytes for simplicity; for clips > 20 MB the
caller should split or fall back to File API uploads.

Logs every call through the injected ApiCallLogger so the api_logs schema
stays consistent across providers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from videvalkit.core.scorer import BaseScorer, ScoreContext, ScoreResult
from videvalkit.utils.video import extract_frames, pil_to_data_url, strip_code_fences


class GeminiVLMJudge(BaseScorer):
    kind = "vlm_judge_api"

    def __init__(
        self,
        name: str,
        model: str = "gemini-3-flash-preview",
        api_key_env: str = "GEMINI_API_KEY",
        sys_prompt: str | Path | None = None,
        logger: Any | None = None,
        frame_cache: Any | None = None,
        request_timeout_s: float = 240.0,
    ) -> None:
        self.name = name
        self.model = model
        self.provider = "google"
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
        # Gemini SDK manages its own transport; ignore but accept for parity.
        pass

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from google import genai  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "GeminiVLMJudge requires `google-genai` — install it in the worldjen env"
            ) from e
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"missing env var {self.api_key_env}")
        self._client = genai.Client(api_key=api_key)
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
        sys_prompt = system or self._sys_prompt
        # google-genai uses system_instruction kwarg in generate_content
        contents = [{"role": "user", "parts": [{"text": user}]}]
        config = {"max_output_tokens": max_tokens, "temperature": temperature}
        if sys_prompt:
            config["system_instruction"] = sys_prompt
        resp = client.models.generate_content(
            model=self.model, contents=contents, config=config,
        )
        normalized = self._normalize(resp)
        self._log({"kind": "text", "user": user[:200]}, contents, normalized)
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
        """Send the video as a sequence of frame images.

        Gemini will accept the video directly as a file_uri/inline_data part,
        but to keep behaviour identical across providers we standardize on
        frame sequences.
        """
        client = self._ensure_client()
        frames = (
            self.frame_cache.get_or_extract(video_path, mode=mode, n_frames=n_frames)
            if self.frame_cache is not None
            else extract_frames(video_path, mode=mode, n_frames=n_frames)
        )
        if not frames:
            raise RuntimeError(f"no frames extracted from {video_path}")
        # google-genai accepts inline_data parts with mime_type
        import base64, io
        parts: list[dict[str, Any]] = [{"text": prompt}]
        for img in frames:
            if img.mode != "RGB":
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(buf.getvalue()).decode("ascii"),
                }
            })
        contents = [{"role": "user", "parts": parts}]
        config = {"max_output_tokens": max_tokens, "temperature": temperature}
        if self._sys_prompt:
            config["system_instruction"] = self._sys_prompt
        resp = client.models.generate_content(
            model=self.model, contents=contents, config=config,
        )
        normalized = self._normalize(resp)
        # Trim images from the logged contents.
        log_parts = [{"text": prompt}] + [{"inline_data": "<elided>"} for _ in frames]
        self._log(
            {"kind": "multimodal", "video_path": str(video_path), "n_frames": len(frames), "mode": mode},
            [{"role": "user", "parts": log_parts}], normalized,
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

    # ---- internals ----------------------------------------------------------
    @staticmethod
    def _normalize(resp: Any) -> dict[str, Any]:
        """Pull text + usage from a genai GenerateContentResponse."""
        try:
            content = resp.text or ""
        except Exception:
            content = ""
        usage = {}
        try:
            u = resp.usage_metadata
            usage = {
                "prompt_tokens":     getattr(u, "prompt_token_count", 0),
                "completion_tokens": getattr(u, "candidates_token_count", 0),
                "total_tokens":      getattr(u, "total_token_count", 0),
            }
        except Exception:
            pass
        return {"content": content, "usage": usage, "model": getattr(resp, "model_version", "")}

    def _log(self, request_meta, contents, response):
        if self.logger is None:
            return
        self.logger.log(
            request={**request_meta, "contents": contents, "model": self.model},
            response={"usage": response.get("usage", {}), "response": response.get("content", "")},
        )
