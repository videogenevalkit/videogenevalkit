"""Cross-benchmark reusable scorers.

Two families:
  * `scorers.vlm_judge` — LLM/VLM-as-judge. Any OpenAI-compatible endpoint
    (vLLM, SGLang, Ollama, official OpenAI, DeepSeek API) plus Gemini /
    Anthropic SDKs.
  * `scorers.metric`    — model-based feature/metric scorers (CLIP, RAFT,
    aesthetic, MUSIQ, ...). Populated as VBench v1 adapter is wired.
"""

from videvalkit.scorers.vlm_judge.openai_compatible import OpenAICompatibleVLMJudge

__all__ = ["OpenAICompatibleVLMJudge"]
