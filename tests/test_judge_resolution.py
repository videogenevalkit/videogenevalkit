"""Unit tests for videvalkit.configs.judge_loader.resolve_judge.

Covers per JUDGE_SELECTION_DESIGN §3 (用户 2026-05-20):
  * judge_name=None       → bench's default_judge (back-compat)
  * judge_name="default"  → bench's default_judge (explicit)
  * judge_name="paper"    → bench's paper_judge (★ new)
  * judge_name="<name>"   → direct lookup in merged SUPPORTED_JUDGES
  * judge_override=<dict> → bypass registry, use the dict
  * needs_judge=False bench + None → returns None
  * paper keyword on bench without paper_judge → ValueError
  * unknown judge name → KeyError with close-match suggestions
"""

from __future__ import annotations

import pytest

from videvalkit.configs.judge_loader import resolve_judge


# ---------------------------------------------------------------- fixtures ---
@pytest.fixture(autouse=True)
def _no_user_yaml(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Make sure no user yaml leaks in for these tests."""
    from videvalkit.configs import judge_loader as jl
    monkeypatch.setattr(jl, "USER_YAML_PATH", tmp_path / "nope.yaml")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(jl._ENV_DISABLE, raising=False)


# ----------------------------------------------------- back-compat (None) ---
class TestBackCompat:
    def test_none_falls_back_to_default_for_judge_bench(self):
        cfg = resolve_judge("worldjen", judge_name=None)
        assert cfg is not None
        assert cfg["model"] == "google/gemma-4-31b-it"  # default_judge

    def test_none_returns_none_for_judge_free_bench(self):
        cfg = resolve_judge("vbench", judge_name=None)
        assert cfg is None

    def test_none_returns_none_for_worldscore(self):
        cfg = resolve_judge("worldscore", judge_name=None)
        assert cfg is None


# ---------------------------------------------- explicit "default" keyword ---
class TestDefaultKeyword:
    def test_default_matches_none_for_judge_bench(self):
        a = resolve_judge("videobench", judge_name=None)
        b = resolve_judge("videobench", judge_name="default")
        assert a == b
        assert b["model"] == "gpt-4o-2024-11-20"

    def test_default_on_judge_free_raises(self):
        with pytest.raises(ValueError, match="no default_judge"):
            resolve_judge("vbench", judge_name="default")


# ------------------------------------------------- "paper" keyword (M new) ---
class TestPaperKeyword:
    def test_paper_resolves_to_paper_judge(self):
        # t2vcompbench paper_judge="paper-llava-1.6-34b"
        cfg = resolve_judge("t2vcompbench", judge_name="paper")
        assert cfg["model"] == "liuhaotian/llava-v1.6-34b"

    def test_paper_for_worldjen_resolves_to_gemma(self):
        cfg = resolve_judge("worldjen", judge_name="paper")
        assert cfg["model"] == "google/gemma-4-31b-it"

    def test_paper_for_videobench_resolves_to_gpt4o(self):
        cfg = resolve_judge("videobench", judge_name="paper")
        assert cfg["model"] == "gpt-4o-2024-11-20"

    def test_paper_on_judge_free_bench_raises(self):
        """vbench has no paper_judge declared (needs_judge=False)."""
        with pytest.raises(ValueError, match="no paper_judge"):
            resolve_judge("vbench", judge_name="paper")

    def test_paper_on_worldscore_raises(self):
        """worldscore is pure CV — paper_judge=None per design."""
        with pytest.raises(ValueError, match="no paper_judge"):
            resolve_judge("worldscore", judge_name="paper")


# -------------------------------------------- direct registry name lookup ---
class TestDirectName:
    def test_existing_name_returns_cfg(self):
        cfg = resolve_judge("worldjen", judge_name="claude-sonnet-4-6")
        assert cfg["kind"] == "anthropic"
        assert cfg["model"] == "claude-sonnet-4-6"

    def test_unknown_name_raises_with_suggestion(self):
        with pytest.raises(KeyError, match="unknown judge"):
            resolve_judge("worldjen", judge_name="gemma-4-31")  # typo

    def test_unknown_name_suggestion_close(self):
        try:
            resolve_judge("worldjen", judge_name="claud-sonnet")
        except KeyError as e:
            assert "claude-sonnet-4-6" in str(e)


# ----------------------------------------------------- judge_override path ---
class TestJudgeOverride:
    def test_override_bypasses_registry(self):
        adhoc = {
            "kind": "openai_compatible",
            "endpoint": "http://10.0.0.5:8003/v1",
            "model": "my-internal-model",
            "provider": "internal",
            "api_key_env": None,
        }
        cfg = resolve_judge(
            "worldjen", judge_name=None, judge_override=adhoc
        )
        assert cfg is adhoc  # exact passthrough

    def test_override_takes_precedence_over_name(self):
        """If both judge_name and judge_override given, override wins."""
        adhoc = {"kind": "openai_compatible", "endpoint": "http://x/v1",
                 "model": "m"}
        cfg = resolve_judge(
            "worldjen", judge_name="claude-sonnet-4-6", judge_override=adhoc
        )
        assert cfg is adhoc


# -------------------------------------------------- unknown bench guard ---
class TestUnknownBench:
    def test_unknown_bench_raises_keyerror(self):
        with pytest.raises(KeyError, match="unknown benchmark"):
            resolve_judge("not-a-bench", judge_name=None)


# -------------------------------------- paper_judge field present on benches ---
def test_judge_benches_declare_paper_judge():
    """Sanity: all benches with needs_judge=True must declare paper_judge.
    This is a registry hygiene check, enforced by REVIEW_PROTOCOL acceptance gate."""
    from videvalkit.configs.benchmarks import SUPPORTED_BENCHMARKS
    missing = []
    for name, cfg in SUPPORTED_BENCHMARKS.items():
        if cfg.get("needs_judge", False) and "paper_judge" not in cfg:
            missing.append(name)
    assert not missing, (
        f"benches with needs_judge=True must declare paper_judge: {missing}"
    )


def test_paper_llava_alias_in_registry():
    """The paper-faithful t2vcompbench judge alias should be in SUPPORTED_JUDGES."""
    from videvalkit.configs import SUPPORTED_JUDGES
    assert "paper-llava-1.6-34b" in SUPPORTED_JUDGES
    assert SUPPORTED_JUDGES["paper-llava-1.6-34b"]["model"] == "liuhaotian/llava-v1.6-34b"
