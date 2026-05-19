"""Judge factory — turn a `SUPPORTED_JUDGES[name]` dict into a live scorer.

Centralizes:
  * picking the right scorer class for the `kind` field
  * attaching an ApiCallLogger that writes to the workspace's api_logs/
  * propagating env-based API keys

Adapter code calls `build_judge(cfg, layout=...)` and never has to know
about specific scorer classes.
"""

from __future__ import annotations

from typing import Any

from videvalkit.core.layout import WorkspaceLayout
from videvalkit.storage.api_log import ApiCallLogger


def build_judge(cfg: dict[str, Any], layout: WorkspaceLayout | None = None) -> Any:
    """Construct a judge scorer from a registry config dict.

    `cfg` is one entry from `SUPPORTED_JUDGES`. If `layout` is provided,
    an ApiCallLogger is attached so every chat call is mirrored to
    `api_logs/calls/{provider}/{model}/...`.
    """
    cfg = dict(cfg)  # don't mutate caller's dict
    kind = cfg.pop("kind", "openai_compatible")
    provider = cfg.get("provider", "openai_compat")
    model = cfg.get("model", "unknown")
    name = cfg.setdefault("name", model)

    logger: ApiCallLogger | None = None
    frame_cache = None
    if layout is not None:
        logger = ApiCallLogger(layout=layout, provider=provider, model=model)
        from videvalkit.utils.frame_cache import FrameCache
        frame_cache = FrameCache(layout)
    cfg.setdefault("logger", logger)
    if frame_cache is not None:
        cfg.setdefault("frame_cache", frame_cache)

    if kind == "openai_compatible":
        from videvalkit.scorers.vlm_judge.openai_compatible import OpenAICompatibleVLMJudge
        return OpenAICompatibleVLMJudge(**cfg)
    keep_keys = {"name", "model", "api_key_env", "sys_prompt", "logger", "frame_cache"}
    if kind == "gemini":
        from videvalkit.scorers.vlm_judge.gemini import GeminiVLMJudge
        return GeminiVLMJudge(**{k: v for k, v in cfg.items() if k in keep_keys})
    if kind == "anthropic":
        from videvalkit.scorers.vlm_judge.anthropic import AnthropicVLMJudge
        return AnthropicVLMJudge(**{k: v for k, v in cfg.items() if k in keep_keys})
    raise ValueError(f"unknown judge kind: {kind!r}")
