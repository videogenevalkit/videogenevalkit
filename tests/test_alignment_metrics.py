"""Tests for text-video alignment metrics (CLIP-Score / ViCLIP-Score)."""

from __future__ import annotations

from pathlib import Path

import pytest

from videvalkit.metrics import SUPPORTED_METRICS, get_metric, list_metrics
from videvalkit.metrics.clip_score import CLIPScore, CLIPScoreResult


# ----------------------------------------------------- registry shape ---
class TestAlignmentRegistry:
    def test_both_registered(self):
        names = list_metrics(kind="per_prompt_reference_free")
        assert "clip-score" in names
        assert "viclip-score" in names

    def test_judge_free(self):
        for name in ("clip-score", "viclip-score"):
            assert SUPPORTED_METRICS[name]["needs_judge"] is False

    def test_tags_align_text2video(self):
        for name in ("clip-score", "viclip-score"):
            assert "align.text2video" in SUPPORTED_METRICS[name]["tags"]

    def test_viclip_extra_prompt_following_tag(self):
        assert "align.prompt_following" in SUPPORTED_METRICS["viclip-score"]["tags"]


# ----------------------------------------------------- get_metric ---
class TestGetMetric:
    def test_get_clip_score(self):
        m = get_metric("clip-score")
        assert isinstance(m, CLIPScore)
        assert m.name == "clip-score"

    def test_get_viclip_score(self):
        m = get_metric("viclip-score")
        assert m.name == "viclip-score"


# ----------------------------------------------------- CLIP-Score input val ---
class TestCLIPScoreInputs:
    def test_mismatched_lengths_raises(self, tmp_path):
        m = get_metric("clip-score")
        with pytest.raises(ValueError, match="same length"):
            m.compute(
                videos=[tmp_path / "v.mp4"],
                prompts=["a", "b"],
            )

    def test_empty_input_returns_zero(self, tmp_path):
        m = get_metric("clip-score")
        # Empty input → CLIP backbone never loaded, returns 0
        result = m.compute(videos=[], prompts=[])
        assert result.score == 0.0
        assert result.n_pairs == 0


# ----------------------------------------------------- ViCLIP-Score shell ---
class TestViCLIPShell:
    def test_viclip_raises_not_implemented(self, tmp_path):
        m = get_metric("viclip-score")
        with pytest.raises(NotImplementedError, match="ViCLIP backbone fetch"):
            m.compute(videos=[tmp_path / "v.mp4"], prompts=["a cat"])


# ----------------------------------------------------- CLIP-Score functional ---
# Mark slow — requires actual CLIP model load and a real video.
# Smoke test only if a test video is present (skipped in CI by default).
def _make_tiny_video(path: Path, n_frames: int = 16, hw: int = 64) -> None:
    """Create a tiny mp4 with uniform random colors per frame."""
    import numpy as np
    try:
        import imageio
    except ImportError:
        pytest.skip("imageio not available")
    np.random.seed(42)
    writer = imageio.get_writer(str(path), fps=4, codec="libx264")
    try:
        for _ in range(n_frames):
            frame = np.random.randint(0, 256, (hw, hw, 3), dtype=np.uint8)
            writer.append_data(frame)
    finally:
        writer.close()


@pytest.mark.slow
def test_clip_score_end_to_end(tmp_path: Path):
    """Smoke test on a tiny random video; verify shape + range, not absolute value."""
    video = tmp_path / "tiny.mp4"
    _make_tiny_video(video)
    m = get_metric("clip-score")
    result = m.compute(
        videos=[video],
        prompts=["a colorful pattern"],
        n_frames=4,
    )
    assert isinstance(result, CLIPScoreResult)
    assert result.n_pairs == 1
    assert -1.0 <= result.score <= 1.0
    assert str(video) in result.per_video
