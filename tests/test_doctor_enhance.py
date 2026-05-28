"""Tests for enhanced `videvalkit doctor` — devices / plugins / metrics /
profiles / capability coverage sections."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from videvalkit.cli import main
from videvalkit.diagnostics import (
    check_capability_coverage,
    check_devices,
    check_metrics,
    check_plugins,
    check_profiles,
    run_all,
)


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


# ----------------------------------------------------- individual checks ---
class TestChecks:
    def test_check_devices_shape(self):
        d = check_devices()
        # Either torch present [cuda key] or absent [torch key]
        assert "cuda" in d or "torch" in d

    def test_check_metrics(self):
        m = check_metrics()
        assert m["total"] >= 14  # we have 14+ metrics now
        assert m["judge_free"] >= 1
        assert "by_kind" in m

    def test_check_profiles(self):
        p = check_profiles()
        assert set(p) == {"quick", "standard", "full"}
        assert p["quick"]["frames"] == 4

    def test_check_plugins(self):
        pl = check_plugins()
        assert "groups" in pl or "error" in pl

    def test_check_capability_coverage(self):
        c = check_capability_coverage()
        assert c["total_tags"] == 44
        assert c["covered_tags"] >= 1


# ----------------------------------------------------- run_all ---
class TestRunAll:
    def test_run_all_has_new_sections(self):
        rep = run_all()
        for key in ("devices", "metrics", "profiles", "plugins",
                    "capability_coverage", "benchmarks"):
            assert key in rep, f"run_all missing {key}"

    def test_benchmarks_split(self):
        rep = run_all()
        bm = rep["benchmarks"]
        # vbench is judge-free; worldjen needs judge
        assert "vbench" in bm["judge_free"]
        assert "worldjen" in bm["needs_judge"]


# ----------------------------------------------------- doctor CLI ---
class TestDoctorCLI:
    def test_doctor_human_readable(self, cli_runner):
        result = cli_runner.invoke(main, ["doctor"])
        assert result.exit_code == 0, result.output
        assert "== Devices ==" in result.output
        assert "== Benchmarks" in result.output
        assert "== Metrics" in result.output
        assert "== Eval profiles ==" in result.output
        assert "== Capability coverage ==" in result.output

    def test_doctor_json(self, cli_runner):
        result = cli_runner.invoke(main, ["doctor", "--json"])
        assert result.exit_code == 0
        rep = json.loads(result.output)
        assert "metrics" in rep
        assert "profiles" in rep
        assert "capability_coverage" in rep
