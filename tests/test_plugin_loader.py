"""Unit tests for videvalkit.plugins.loader.

Per docs/INTEGRATION_FRAMEWORK_DESIGN.md §4 (user 2026-05-20):
  * 3 sources: builtin / entry_points / local-dirs
  * Merge order: later wins (top-level key replacement)
  * Conflict logged at INFO
  * Local plugin convention: ``__videvalkit_register__()`` returns
    ``{group: {name: cfg}}``
  * VIDEVALKIT_DISABLE_PLUGINS=1 disables third-party sources
  * Single-plugin failure doesn't kill others
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path

import pytest

from videvalkit.plugins import discover, plugin_sources_report
from videvalkit.plugins import loader as pl


# ---------------------------------------------------------------- fixtures ---
@pytest.fixture
def tmp_plugins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect both USER_PLUGINS_DIR and CWD to a tmp dir.

    Returns the user-plugins dir (the CWD project-plugins dir is at
    ``tmp_path / ".videvalkit"`` by convention).
    """
    user = tmp_path / "user_plugins"
    monkeypatch.setattr(pl, "USER_PLUGINS_DIR", user)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(pl._ENV_DISABLE, raising=False)
    return user


def _write_plugin(
    base: Path, group: str, name: str, register_body: str,
) -> Path:
    """Create a local plugin at <base>/<group>/<name>/__init__.py."""
    plugin_dir = base / group / name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "__init__.py").write_text(textwrap.dedent(register_body))
    return plugin_dir


@pytest.fixture
def builtin() -> dict[str, dict]:
    return {
        "vbench": {"cls": "vbench.Class", "env": "videvalkit"},
        "worldjen": {"cls": "worldjen.Class", "env": "videvalkit"},
    }


# ------------------------------------------- discover: no plugins → builtin ---
class TestNoPluginSources:
    def test_no_local_no_ep_returns_builtin(self, tmp_plugins, builtin):
        out = discover("benchmarks", builtin)
        assert out == builtin

    def test_disabled_env_returns_builtin(
        self, tmp_plugins, builtin, monkeypatch
    ):
        # Even if a local plugin exists, disabled flag should hide it
        _write_plugin(
            tmp_plugins, "benchmarks", "myplugin",
            """
            def __videvalkit_register__():
                return {"benchmarks": {"myplugin": {"cls": "x"}}}
            """,
        )
        monkeypatch.setenv(pl._ENV_DISABLE, "1")
        out = discover("benchmarks", builtin)
        assert out == builtin


# -------------------------------------------------------- local plugin layer ---
class TestLocalDirPlugins:
    def test_user_plugin_merged(self, tmp_plugins, builtin):
        _write_plugin(
            tmp_plugins, "benchmarks", "anubis",
            """
            def __videvalkit_register__():
                return {"benchmarks": {"anubis": {"cls": "anubis.Class", "env": "videvalkit"}}}
            """,
        )
        out = discover("benchmarks", builtin)
        assert "anubis" in out
        assert "vbench" in out  # builtin preserved
        assert out["anubis"]["cls"] == "anubis.Class"

    def test_user_plugin_overrides_builtin(self, tmp_plugins, builtin, caplog):
        """Same-name plugin should override the builtin and log at INFO."""
        _write_plugin(
            tmp_plugins, "benchmarks", "vbench",
            """
            def __videvalkit_register__():
                return {"benchmarks": {"vbench": {"cls": "user.MyVBench", "env": "custom"}}}
            """,
        )
        with caplog.at_level(logging.INFO):
            out = discover("benchmarks", builtin)
        assert out["vbench"]["cls"] == "user.MyVBench"
        assert any("overrides" in r.message for r in caplog.records)

    def test_project_overrides_user(self, tmp_plugins, builtin):
        """$CWD/.videvalkit > ~/.videvalkit (project beats user)."""
        # user-level says A
        _write_plugin(
            tmp_plugins, "benchmarks", "shared",
            """
            def __videvalkit_register__():
                return {"benchmarks": {"shared": {"cls": "user.Shared"}}}
            """,
        )
        # project-level says B (same name)
        proj_base = Path.cwd() / ".videvalkit"
        _write_plugin(
            proj_base, "benchmarks", "shared",
            """
            def __videvalkit_register__():
                return {"benchmarks": {"shared": {"cls": "project.Shared"}}}
            """,
        )
        out = discover("benchmarks", builtin)
        assert out["shared"]["cls"] == "project.Shared"

    def test_plugin_missing_register_func_skipped(self, tmp_plugins, builtin):
        """Plugin module without __videvalkit_register__ silently skipped."""
        _write_plugin(
            tmp_plugins, "benchmarks", "incomplete",
            "x = 1  # no register function\n",
        )
        out = discover("benchmarks", builtin)
        assert "incomplete" not in out

    def test_plugin_raises_doesnt_kill_others(
        self, tmp_plugins, builtin, caplog,
    ):
        # one bad plugin
        _write_plugin(
            tmp_plugins, "benchmarks", "bad",
            """
            def __videvalkit_register__():
                raise RuntimeError("boom")
            """,
        )
        # one good plugin
        _write_plugin(
            tmp_plugins, "benchmarks", "good",
            """
            def __videvalkit_register__():
                return {"benchmarks": {"good": {"cls": "good.Class"}}}
            """,
        )
        with caplog.at_level(logging.WARNING):
            out = discover("benchmarks", builtin)
        assert "good" in out
        assert "bad" not in out
        assert any("raised" in r.message for r in caplog.records)

    def test_plugin_returns_non_dict_skipped(
        self, tmp_plugins, builtin, caplog,
    ):
        _write_plugin(
            tmp_plugins, "benchmarks", "weird",
            """
            def __videvalkit_register__():
                return "not a dict"
            """,
        )
        with caplog.at_level(logging.WARNING):
            out = discover("benchmarks", builtin)
        assert "weird" not in out
        assert any("must return a dict" in r.message for r in caplog.records)

    def test_plugin_import_error_isolated(self, tmp_plugins, builtin, caplog):
        plugin_dir = tmp_plugins / "benchmarks" / "broken"
        plugin_dir.mkdir(parents=True)
        # Syntax error in __init__.py
        (plugin_dir / "__init__.py").write_text("def __videvalkit_register__(:\n")
        with caplog.at_level(logging.WARNING):
            out = discover("benchmarks", builtin)
        assert "broken" not in out


# ------------------------------------------- multiple groups discovered ---
class TestMultipleGroups:
    def test_metrics_group_independent(self, tmp_plugins, builtin):
        _write_plugin(
            tmp_plugins, "metrics", "fvd-variant",
            """
            def __videvalkit_register__():
                return {"metrics": {"fvd-variant": {"kind": "distribution_reference"}}}
            """,
        )
        # benchmarks group unaffected
        out_b = discover("benchmarks", builtin)
        assert "fvd-variant" not in out_b
        # metrics group picks it up
        out_m = discover("metrics", {})
        assert "fvd-variant" in out_m

    def test_judges_group(self, tmp_plugins):
        _write_plugin(
            tmp_plugins, "judges", "doubao-internal",
            """
            def __videvalkit_register__():
                return {"judges": {"doubao-internal": {"kind": "openai_compatible",
                                                       "model": "doubao",
                                                       "endpoint": "http://x/v1"}}}
            """,
        )
        out = discover("judges", {})
        assert "doubao-internal" in out


# ------------------------------------------------- validation / errors ---
class TestErrors:
    def test_unknown_group_raises(self, tmp_plugins):
        with pytest.raises(ValueError, match="unknown plugin group"):
            discover("not-a-group", {})


# ------------------------------------------- diagnostics report ---
class TestSourceReport:
    def test_report_structure(self, tmp_plugins):
        report = plugin_sources_report()
        assert "disabled_by_env" in report
        assert "groups" in report
        for g in ("benchmarks", "metrics", "judges", "aggregators"):
            assert g in report["groups"]
            assert "entry_points" in report["groups"][g]
            assert "user_plugins" in report["groups"][g]

    def test_report_counts_local_plugins(self, tmp_plugins):
        _write_plugin(
            tmp_plugins, "benchmarks", "p1",
            "def __videvalkit_register__():\n    return {}",
        )
        _write_plugin(
            tmp_plugins, "benchmarks", "p2",
            "def __videvalkit_register__():\n    return {}",
        )
        report = plugin_sources_report()
        assert report["groups"]["benchmarks"]["user_plugins"] == 2

    def test_report_disabled_flag_true_when_env_set(
        self, tmp_plugins, monkeypatch,
    ):
        monkeypatch.setenv(pl._ENV_DISABLE, "1")
        report = plugin_sources_report()
        assert report["disabled_by_env"] is True


# ------------------------------------------- SUPPORTED_BENCHMARKS integration ---
def test_supported_benchmarks_via_plugin_layer(monkeypatch):
    """Sanity: SUPPORTED_BENCHMARKS now flows through plugin discovery, but
    when no user plugins exist, it should still contain all built-in entries."""
    monkeypatch.setenv(pl._ENV_DISABLE, "0")
    import importlib
    from videvalkit import configs as _configs
    importlib.reload(_configs)

    # All 10 anchored + stub benchmarks should still appear
    for name in ("vbench", "vbench2", "videobench", "worldjen", "worldscore",
                 "t2vcompbench", "physics_iq", "vbench_pp", "v_reasonbench",
                 "semantics_axis"):
        assert name in _configs.SUPPORTED_BENCHMARKS, f"missing {name}"
