"""Tests for `videvalkit prompts dump` — the cross-env prompt manifest export."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from videvalkit.cli import main


@pytest.fixture
def cli_runner():
    return CliRunner()


class _FakePrompt:
    def __init__(self, pid: str, text: str, dims: list[str]):
        self.prompt_id = pid
        self.text = text
        self.dimensions = dims


def _fake_bench_with_prompts():
    """Return a mock bench class whose .list_prompts yields three test prompts."""
    cls = MagicMock()
    instance = MagicMock()
    instance.list_prompts.return_value = iter([
        _FakePrompt("p1", "a panda eating bamboo", ["subject_consistency"]),
        _FakePrompt("p2", "a yellow car",          ["color", "object_class"]),
        _FakePrompt("p3", "a tranquil garden",      ["scene", "aesthetic_quality"]),
    ])
    cls.return_value = instance
    return cls, instance


def test_prompts_dump_emits_jsonl_to_stdout(cli_runner):
    cls, inst = _fake_bench_with_prompts()
    with patch.dict("videvalkit.cli.SUPPORTED_BENCHMARKS",
                    {"vbench": {"cls": cls, "env": "videvalkit"}}, clear=False):
        result = cli_runner.invoke(main, ["prompts", "dump", "--bench", "vbench"])
    assert result.exit_code == 0, result.output
    lines = [ln for ln in result.output.strip().split("\n") if ln]
    assert len(lines) == 3
    recs = [json.loads(ln) for ln in lines]
    assert recs[0] == {
        "id": "p1", "prompt_en": "a panda eating bamboo",
        "dimensions": ["subject_consistency"],
    }
    assert recs[1]["dimensions"] == ["color", "object_class"]


def test_prompts_dump_to_file(cli_runner, tmp_path):
    cls, inst = _fake_bench_with_prompts()
    out = tmp_path / "prompts.jsonl"
    with patch.dict("videvalkit.cli.SUPPORTED_BENCHMARKS",
                    {"vbench": {"cls": cls, "env": "videvalkit"}}, clear=False):
        result = cli_runner.invoke(
            main, ["prompts", "dump", "--bench", "vbench", "-o", str(out)],
        )
    assert result.exit_code == 0, result.output
    assert "wrote 3 prompts" in result.output  # status goes to stderr; click merges
    body = out.read_text().strip().split("\n")
    assert len(body) == 3
    assert json.loads(body[0])["prompt_en"] == "a panda eating bamboo"


def test_prompts_dump_passes_dim_filter(cli_runner):
    cls, inst = _fake_bench_with_prompts()
    with patch.dict("videvalkit.cli.SUPPORTED_BENCHMARKS",
                    {"vbench": {"cls": cls, "env": "videvalkit"}}, clear=False):
        cli_runner.invoke(main, [
            "prompts", "dump", "--bench", "vbench",
            "--dimensions", "subject_consistency",
            "--dimensions", "color",
        ])
    inst.list_prompts.assert_called_once_with(
        dimensions=["subject_consistency", "color"]
    )
