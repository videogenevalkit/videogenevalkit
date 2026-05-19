"""VLM judge registry.

Each entry is a kwargs dict for instantiating one of the scorer classes
under `videvalkit.scorers.vlm_judge`. The `kind` field tells the runner
which class to instantiate; everything else is passed to its constructor.

Conventions:
  * `kind`         — "openai_compatible" | "gemini" | "anthropic"
  * `endpoint`     — base URL, e.g. "http://localhost:8003/v1"
  * `model`        — model name as the endpoint expects
  * `provider`     — used by api_log writer for partitioning
  * `api_key_env`  — env var to read auth from; None for local endpoints
"""


SUPPORTED_JUDGES = {
    # ---- local vLLM endpoints (matches worldjen_local/run.sh) ----------------
    "gemma-4-31b-local": dict(
        kind="openai_compatible",
        endpoint="http://localhost:8003/v1",
        model="google/gemma-4-31b-it",
        provider="google",
        api_key_env=None,
    ),
    "qwen3-32b-local": dict(
        kind="openai_compatible",
        endpoint="http://localhost:8004/v1",
        model="Qwen/Qwen3-32B",
        provider="Qwen",
        api_key_env=None,
    ),
    "qwen3-vl-32b-local": dict(
        kind="openai_compatible",
        endpoint="http://localhost:8005/v1",
        model="Qwen/Qwen3-VL-32B-Instruct",
        provider="Qwen",
        api_key_env=None,
    ),
    "local-llava-video-7b": dict(
        kind="openai_compatible",
        endpoint="http://localhost:8006/v1",
        model="lmms-lab/LLaVA-Video-7B-Qwen2",
        provider="lmms-lab",
        api_key_env=None,
    ),
    # ---- managed APIs -------------------------------------------------------
    "gemini-3-flash": dict(
        kind="gemini",
        model="gemini-3-flash-preview",
        provider="google",
        api_key_env="GEMINI_API_KEY",
    ),
    "gemini-2.5-pro": dict(
        kind="gemini",
        model="gemini-2.5-pro",
        provider="google",
        api_key_env="GEMINI_API_KEY",
    ),
    "claude-sonnet-4-6": dict(
        kind="anthropic",
        model="claude-sonnet-4-6",
        provider="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
    ),
    "gpt-4o": dict(
        kind="openai_compatible",
        endpoint="https://api.openai.com/v1",
        model="gpt-4o-2024-11-20",
        provider="openai",
        api_key_env="OPENAI_API_KEY",
    ),
}
