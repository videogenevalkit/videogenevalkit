"""Additional integration tests: VBench JSON parsing, frame cache, doctor command."""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import pytest


def test_vbench_parse_shape_a():
    """Shape A: {dim: [mean, [{video_path, video_results}, ...]]}"""
    from videvalkit.benchmarks.vbench.benchmark import VBenchBenchmark
    from videvalkit.storage import Workspace

    with tempfile.TemporaryDirectory() as td:
        ws = Workspace(td)
        out_dir = Path(td) / "vbench_out"
        out_dir.mkdir()
        (out_dir / "x_eval_results.json").write_text(json.dumps({
            "subject_consistency": [
                0.85,
                [{"video_path": "model_a/000-0.mp4", "video_results": 0.9},
                 {"video_path": "model_a/001-0.mp4", "video_results": 0.8}],
            ],
        }))
        bench = VBenchBenchmark()
        results = bench._collect_dim_results("model_a", "subject_consistency", out_dir, ws)
        assert len(results) == 2
        assert results[0].score == 0.9
        assert results[0].prompt_id == "000"


def test_vbench_parse_shape_b():
    """Shape B: {dim: {"score": mean, "videos": [...]}}"""
    from videvalkit.benchmarks.vbench.benchmark import VBenchBenchmark
    from videvalkit.storage import Workspace

    with tempfile.TemporaryDirectory() as td:
        ws = Workspace(td)
        out_dir = Path(td) / "vbench_out"
        out_dir.mkdir()
        (out_dir / "x_eval_results.json").write_text(json.dumps({
            "motion_smoothness": {
                "score": 0.7,
                "videos": [{"video_path": "m/000-0.mp4", "video_results": 0.65}],
            },
        }))
        bench = VBenchBenchmark()
        results = bench._collect_dim_results("m", "motion_smoothness", out_dir, ws)
        assert len(results) == 1
        assert results[0].score == 0.65


def test_vbench_parse_shape_c_scalar_mean():
    """Shape C: scalar mean only — produces one summary row."""
    from videvalkit.benchmarks.vbench.benchmark import VBenchBenchmark
    from videvalkit.storage import Workspace

    with tempfile.TemporaryDirectory() as td:
        ws = Workspace(td)
        out_dir = Path(td) / "vbench_out"
        out_dir.mkdir()
        (out_dir / "x_eval_results.json").write_text(json.dumps({
            "aesthetic_quality": 0.42,
        }))
        bench = VBenchBenchmark()
        results = bench._collect_dim_results("m", "aesthetic_quality", out_dir, ws)
        assert len(results) == 1
        assert results[0].prompt_id == "_mean"
        assert results[0].score == 0.42


def test_vbench_parse_video_results_as_dict():
    """`video_results` may itself be a dict — extract its inner score."""
    from videvalkit.benchmarks.vbench.benchmark import VBenchBenchmark
    from videvalkit.storage import Workspace

    with tempfile.TemporaryDirectory() as td:
        ws = Workspace(td)
        out_dir = Path(td) / "vbench_out"
        out_dir.mkdir()
        (out_dir / "x_eval_results.json").write_text(json.dumps({
            "color": [
                0.5,
                [{"video_path": "m/000-0.mp4", "video_results": {"score": 0.55, "extras": "ok"}}],
            ],
        }))
        bench = VBenchBenchmark()
        results = bench._collect_dim_results("m", "color", out_dir, ws)
        assert len(results) == 1
        assert results[0].score == 0.55


def test_frame_cache_hit_after_miss(monkeypatch):
    """Second call with same args reads from JPEG cache, not video."""
    pytest.importorskip("PIL.Image")
    import PIL.Image
    from videvalkit.utils.frame_cache import FrameCache
    from videvalkit.storage import Workspace

    calls = {"n": 0}

    def fake_extract(path, mode, n_frames):
        calls["n"] += 1
        # real (tiny) RGB PIL images so cache round-tripping through JPEG works
        return [PIL.Image.new("RGB", (16, 16), color=(i * 30, 50, 100))
                for i in range(n_frames)]

    import videvalkit.utils.frame_cache as fc_mod
    monkeypatch.setattr(fc_mod, "extract_frames", fake_extract)

    with tempfile.TemporaryDirectory() as td:
        ws = Workspace(td)
        vid = Path(td) / "fake.mp4"
        vid.write_bytes(b"\x00" * 64)
        cache = FrameCache(ws.layout)

        frames1 = cache.get_or_extract(vid, mode="holistic", n_frames=3)
        assert len(frames1) == 3
        assert calls["n"] == 1

        frames2 = cache.get_or_extract(vid, mode="holistic", n_frames=3)
        assert len(frames2) == 3
        assert calls["n"] == 1, "cache miss on second call"
        # Different (mode, n_frames) should miss
        cache.get_or_extract(vid, mode="micro", n_frames=3)
        assert calls["n"] == 2


def test_doctor_runs_without_workspace():
    """`doctor` against just the registries works (no workspace required)."""
    from videvalkit.diagnostics import run_all

    rep = run_all(workspace=None)
    legacy = {"vbench", "vbench2", "videobench", "worldjen"}
    extras = {"t2vcompbench", "physics_iq", "vbench_pp", "v_reasonbench"}
    assert legacy <= set(rep["envs"])
    assert legacy <= set(rep["adapters"])
    assert extras <= set(rep["adapters"])
    # All adapters import cleanly in the core env (lazy upstream imports).
    for name, r in rep["adapters"].items():
        assert r["ok"] is True, f"{name} failed to import: {r.get('error')}"
    # Local judge endpoints likely unreachable in CI; just check the schema.
    for name, r in rep["judges"].items():
        assert "reachable" in r
        assert "api_key_present" in r


def test_judge_factory_attaches_logger_and_frame_cache():
    """build_judge with layout wires logger AND frame_cache."""
    from videvalkit.scorers.vlm_judge import build_judge
    from videvalkit.storage import Workspace

    with tempfile.TemporaryDirectory() as td:
        ws = Workspace(td)
        cfg = {
            "kind": "openai_compatible",
            "endpoint": "http://localhost:9999/v1",
            "model": "fake",
            "provider": "fake",
            "api_key_env": None,
        }
        judge = build_judge(cfg, layout=ws.layout)
        assert judge.logger is not None
        assert judge.frame_cache is not None
        assert judge.model == "fake"
