"""Tests for prompt-aligned distribution metrics (FVD/VFID/KVD/CLIP-FVD)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from videvalkit.metrics.utils.prompts import (
    load_prompt_ids,
    prompt_id_from_filename,
    verify_prompt_alignment,
)


# -------- helpers --------

def _write_jsonl(path: Path, items: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(it) for it in items) + "\n")
    return path


# -------- prompt_id_from_filename --------

class TestPromptIdFromFilename:
    def test_basic_sample_suffix(self):
        assert prompt_id_from_filename("A panda eating-0.mp4") == "A panda eating"

    def test_no_sample_suffix(self):
        assert prompt_id_from_filename("plain prompt.mp4") == "plain prompt"

    def test_prompt_contains_hyphens(self):
        # Hyphens within prompt survive; only trailing -N is stripped.
        assert prompt_id_from_filename(
            "a sci-fi shot of a low-poly robot-0.mp4"
        ) == "a sci-fi shot of a low-poly robot"

    def test_path_objects(self):
        assert prompt_id_from_filename(Path("/tmp/x/foo-3.mp4")) == "foo"


# -------- load_prompt_ids --------

class TestLoadPromptIds:
    def test_jsonl_id_field(self, tmp_path):
        f = _write_jsonl(tmp_path / "p.jsonl", [
            {"id": "a", "prompt_en": "A cat"},
            {"id": "b", "prompt_en": "A dog"},
        ])
        assert load_prompt_ids(f) == ["a", "b"]

    def test_json_list(self, tmp_path):
        f = tmp_path / "p.json"
        f.write_text(json.dumps([{"prompt_en": "a"}, {"prompt_en": "b"}]))
        assert load_prompt_ids(f) == ["a", "b"]

    def test_empty_raises(self, tmp_path):
        f = _write_jsonl(tmp_path / "empty.jsonl", [])
        with pytest.raises(ValueError, match="no prompt_ids"):
            load_prompt_ids(f)


# -------- verify_prompt_alignment --------

class TestVerifyAlignment:
    def test_full_match(self, tmp_path):
        gen = [tmp_path / f"{p}-0.mp4" for p in ("a", "b", "c")]
        ref = [tmp_path / f"{p}-0.mp4" for p in ("a", "b", "c")]
        for p in gen + ref:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")
        g, r, used = verify_prompt_alignment(gen, ref, ["a", "b", "c"])
        assert len(g) == 3
        assert len(r) == 3
        assert used == ["a", "b", "c"]

    def test_strict_missing_raises(self, tmp_path):
        gen = [tmp_path / "a-0.mp4", tmp_path / "b-0.mp4"]
        ref = [tmp_path / "a-0.mp4"]               # ref missing 'b'
        with pytest.raises(ValueError, match="from ref"):
            verify_prompt_alignment(gen, ref, ["a", "b"])

    def test_partial_uses_intersection(self, tmp_path):
        gen = [tmp_path / "a-0.mp4", tmp_path / "b-0.mp4"]
        ref = [tmp_path / "a-0.mp4"]
        g, r, used = verify_prompt_alignment(
            gen, ref, ["a", "b"], allow_partial=True,
        )
        assert used == ["a"]
        assert len(g) == 1 and len(r) == 1

    def test_multiple_samples_per_prompt_preserved(self, tmp_path):
        gen = [tmp_path / f"a-{i}.mp4" for i in range(3)]
        ref = [tmp_path / f"a-{i}.mp4" for i in range(2)]
        g, r, _ = verify_prompt_alignment(gen, ref, ["a"])
        assert len(g) == 3
        assert len(r) == 2


# -------- end-to-end on a metric.compute() --------

class TestComputeAcceptsPrompts:
    """FVD/VFID/KVD/CLIP-FVD should accept prompts kwarg without crashing on
    the parameter itself; downstream feature-extraction is mocked away."""

    def test_compute_signatures_accept_prompts(self):
        # Just check the signatures advertise the new params.
        import inspect

        from videvalkit.metrics.fvd import FVD
        from videvalkit.metrics.vfid import VFID
        from videvalkit.metrics.kvd import KVD
        from videvalkit.metrics.clip_fvd import CLIPFVD

        for cls in (FVD, VFID, KVD, CLIPFVD):
            sig = inspect.signature(cls.compute)
            assert "prompts" in sig.parameters, f"{cls.__name__}.compute missing 'prompts'"
            assert "allow_partial_prompts" in sig.parameters, \
                f"{cls.__name__}.compute missing 'allow_partial_prompts'"
