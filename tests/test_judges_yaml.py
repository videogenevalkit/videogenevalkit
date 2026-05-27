"""Unit tests for videvalkit.configs.judge_loader.

Covers per JUDGE_SELECTION_DESIGN §9.1:
  * resolve_judge / load_user_judges 全分支
  * JudgeConfig schema 校验
  * merge 优先级（builtin → user yaml）

Plus invariants:
  * 缺文件 / 空文件 / 错配项不抛 fatal
  * VIDEVALKIT_JUDGE_USER_YAML=0 关闭加载
  * 同名 user 条目覆盖 builtin（top-level 覆盖，无 deep-merge）
"""

from __future__ import annotations

import os
from pathlib import Path
from textwrap import dedent

import pytest
from pydantic import ValidationError

from videvalkit.configs import judge_loader as jl
from videvalkit.configs.judge_loader import (
    JudgeConfig,
    get_judges,
    load_user_judges,
)


# ---------------------------------------------------------------- fixtures ---
@pytest.fixture
def tmp_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect USER_YAML_PATH to a tmp dir; return that path (file may
    or may not exist depending on test)."""
    p = tmp_path / "judges.yaml"
    monkeypatch.setattr(jl, "USER_YAML_PATH", p)
    # CWD-side project yaml: also redirect away to avoid accidental pick-up
    monkeypatch.chdir(tmp_path)
    # ensure env not disabled
    monkeypatch.delenv(jl._ENV_DISABLE, raising=False)
    return p


@pytest.fixture
def builtin_minimal() -> dict[str, dict]:
    return {
        "gemma-4-31b-local": {
            "kind": "openai_compatible",
            "endpoint": "http://localhost:8003/v1",
            "model": "google/gemma-4-31b-it",
            "provider": "google",
            "api_key_env": None,
        }
    }


# -------------------------------------------------------------- JudgeConfig ---
class TestJudgeConfig:
    def test_openai_compat_minimal(self):
        c = JudgeConfig(
            kind="openai_compatible",
            model="my-model",
            endpoint="http://x/v1",
        )
        assert c.kind == "openai_compatible"
        assert c.model == "my-model"
        assert c.endpoint == "http://x/v1"
        assert c.provider == "unknown"  # default
        assert c.api_key_env is None

    def test_gemini(self):
        c = JudgeConfig(kind="gemini", model="gemini-2.5-pro",
                        provider="google", api_key_env="GEMINI_API_KEY")
        assert c.kind == "gemini"

    def test_anthropic(self):
        c = JudgeConfig(kind="anthropic", model="claude-sonnet-4-6",
                        provider="anthropic", api_key_env="ANTHROPIC_API_KEY")
        assert c.kind == "anthropic"

    def test_invalid_kind(self):
        with pytest.raises(ValidationError):
            JudgeConfig(kind="cohere", model="x")

    def test_missing_model(self):
        with pytest.raises(ValidationError):
            JudgeConfig(kind="openai_compatible", endpoint="http://x/v1")

    def test_extra_fields_allowed(self):
        """Forward-compat: SDK kwargs unknown today must round-trip."""
        c = JudgeConfig(kind="openai_compatible", model="x",
                        endpoint="http://x/v1", request_timeout_s=60)
        d = c.model_dump()
        assert d["request_timeout_s"] == 60


# ------------------------------------------------------- load_user_judges ---
class TestLoadUserJudges:
    def test_missing_file_returns_empty(self, tmp_yaml: Path):
        assert not tmp_yaml.exists()
        assert load_user_judges() == {}

    def test_empty_file_returns_empty(self, tmp_yaml: Path):
        tmp_yaml.write_text("")
        assert load_user_judges() == {}

    def test_no_judges_key_returns_empty(self, tmp_yaml: Path):
        tmp_yaml.write_text("foo: bar\n")
        assert load_user_judges() == {}

    def test_malformed_yaml_warns_and_returns_empty(
        self, tmp_yaml: Path, caplog: pytest.LogCaptureFixture
    ):
        tmp_yaml.write_text("this: is: not: yaml: [unclosed\n")
        assert load_user_judges() == {}
        assert any("malformed" in r.message for r in caplog.records)

    def test_one_valid_entry(self, tmp_yaml: Path):
        tmp_yaml.write_text(
            dedent("""\
                judges:
                  my-vllm:
                    kind: openai_compatible
                    endpoint: http://10.0.0.5:8003/v1
                    model: google/gemma-4-31b-it
                    provider: google
                    api_key_env: null
            """)
        )
        out = load_user_judges()
        assert "my-vllm" in out
        assert out["my-vllm"]["kind"] == "openai_compatible"
        assert out["my-vllm"]["endpoint"] == "http://10.0.0.5:8003/v1"

    def test_bad_entry_skipped_with_warn(
        self, tmp_yaml: Path, caplog: pytest.LogCaptureFixture
    ):
        tmp_yaml.write_text(
            dedent("""\
                judges:
                  good:
                    kind: openai_compatible
                    model: x
                    endpoint: http://x/v1
                  bad-no-kind:
                    model: y
                    endpoint: http://y/v1
                  bad-kind:
                    kind: cohere
                    model: z
            """)
        )
        out = load_user_judges()
        assert "good" in out
        assert "bad-no-kind" not in out
        assert "bad-kind" not in out
        assert sum("failed validation" in r.message for r in caplog.records) == 2

    def test_disabled_via_env(
        self, tmp_yaml: Path, monkeypatch: pytest.MonkeyPatch
    ):
        tmp_yaml.write_text(
            "judges:\n  x:\n    kind: openai_compatible\n    model: y\n"
            "    endpoint: http://x/v1\n"
        )
        monkeypatch.setenv(jl._ENV_DISABLE, "0")
        assert load_user_judges() == {}

    def test_extra_fields_pass_through(self, tmp_yaml: Path):
        tmp_yaml.write_text(
            dedent("""\
                judges:
                  my:
                    kind: openai_compatible
                    model: x
                    endpoint: http://x/v1
                    request_timeout_s: 120
            """)
        )
        out = load_user_judges()
        assert out["my"]["request_timeout_s"] == 120


# -------------------------------------------------------------- get_judges ---
class TestGetJudges:
    def test_no_user_yaml_returns_builtin(self, tmp_yaml: Path, builtin_minimal):
        out = get_judges(builtin_minimal)
        assert out == builtin_minimal

    def test_user_adds_new_entry(self, tmp_yaml: Path, builtin_minimal):
        tmp_yaml.write_text(
            dedent("""\
                judges:
                  new-vllm:
                    kind: openai_compatible
                    endpoint: http://x/v1
                    model: y
            """)
        )
        out = get_judges(builtin_minimal)
        assert set(out.keys()) == {"gemma-4-31b-local", "new-vllm"}
        # builtin entry preserved
        assert out["gemma-4-31b-local"]["model"] == "google/gemma-4-31b-it"
        # new entry merged
        assert out["new-vllm"]["model"] == "y"

    def test_user_overrides_builtin(self, tmp_yaml: Path, builtin_minimal):
        """User entry with same name fully replaces builtin (no deep merge)."""
        tmp_yaml.write_text(
            dedent("""\
                judges:
                  gemma-4-31b-local:
                    kind: openai_compatible
                    endpoint: http://my-internal:8003/v1
                    model: google/gemma-4-31b-it
                    provider: internal
            """)
        )
        out = get_judges(builtin_minimal)
        assert len(out) == 1
        assert out["gemma-4-31b-local"]["endpoint"] == "http://my-internal:8003/v1"
        assert out["gemma-4-31b-local"]["provider"] == "internal"

    def test_project_yaml_overrides_home_yaml(
        self, tmp_yaml: Path, tmp_path: Path, builtin_minimal,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Per §4.1: $CWD/.videvalkit/judges.yaml > ~/.config/videvalkit/judges.yaml."""
        # home yaml says X
        tmp_yaml.write_text(
            dedent("""\
                judges:
                  shared:
                    kind: openai_compatible
                    model: home-model
                    endpoint: http://home/v1
            """)
        )
        # project yaml says Y (same name)
        proj_dir = tmp_path / ".videvalkit"
        proj_dir.mkdir()
        (proj_dir / "judges.yaml").write_text(
            dedent("""\
                judges:
                  shared:
                    kind: openai_compatible
                    model: project-model
                    endpoint: http://project/v1
            """)
        )
        # tmp_yaml fixture already chdir'd to tmp_path
        out = get_judges(builtin_minimal)
        assert out["shared"]["model"] == "project-model"

    def test_disabled_env_returns_only_builtin(
        self, tmp_yaml: Path, builtin_minimal, monkeypatch: pytest.MonkeyPatch,
    ):
        tmp_yaml.write_text(
            "judges:\n  x:\n    kind: openai_compatible\n    model: y\n"
            "    endpoint: http://x/v1\n"
        )
        monkeypatch.setenv(jl._ENV_DISABLE, "0")
        out = get_judges(builtin_minimal)
        assert out == builtin_minimal


# ------------------------------------- integration: real SUPPORTED_JUDGES ---
def test_supported_judges_has_builtin_entries(monkeypatch: pytest.MonkeyPatch):
    """Sanity: when no user yaml exists, SUPPORTED_JUDGES has the 8 builtin
    entries from configs/judges.py."""
    monkeypatch.setenv(jl._ENV_DISABLE, "0")  # be deterministic in CI
    # Need to re-evaluate the module-level merge to honor the env flag.
    import importlib
    from videvalkit import configs as _configs
    importlib.reload(_configs)

    assert "gemma-4-31b-local" in _configs.SUPPORTED_JUDGES
    assert "claude-sonnet-4-6" in _configs.SUPPORTED_JUDGES
    assert "gpt-4o" in _configs.SUPPORTED_JUDGES
    assert len(_configs.SUPPORTED_JUDGES) >= 8  # 8 builtin + paper aliases added later
