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


# ----------------------------------------------------- ViCLIP-Score inputs ---
class TestViCLIPInputs:
    def test_mismatched_lengths_raises(self, tmp_path):
        m = get_metric("viclip-score")
        with pytest.raises(ValueError, match="same length"):
            m.compute(videos=[tmp_path / "v.mp4"], prompts=["a", "b"])

    def test_empty_input_returns_zero(self):
        # Empty input short-circuits before the ViCLIP backbone is loaded.
        m = get_metric("viclip-score")
        result = m.compute(videos=[], prompts=[])
        assert result.score == 0.0
        assert result.n_pairs == 0


def test_viclip_middle_frame_indices():
    from videvalkit.metrics.viclip_score import _middle_frame_indices

    # 8 frames from a 16-frame clip → middles of 8 equal intervals.
    idxs = _middle_frame_indices(8, 16)
    assert len(idxs) == 8
    assert idxs == sorted(idxs)
    assert all(0 <= i < 16 for i in idxs)
    # Short clip: still returns exactly num_frames (padded).
    assert len(_middle_frame_indices(8, 3)) == 8


@pytest.mark.needs_gpu
def test_viclip_score_discriminates(tmp_path):
    """ViCLIP must score a matched prompt above a mismatched one.

    Requires CUDA + the ViCLIP weight; skipped where either is absent.
    """
    import shutil

    from videvalkit.metrics.backbones.viclip_l import resolve_viclip_dir

    if resolve_viclip_dir() is None:
        pytest.skip("ViCLIP weights not present")
    src = Path(__file__).parent / "data" / "tiny_clip.mp4"
    if not src.is_file():
        pytest.skip("no sample clip available")
    video = tmp_path / "clip.mp4"
    shutil.copy(src, video)
    m = get_metric("viclip-score")
    matched = m.compute([video], ["the content of this clip"], device="cuda").score
    wrong = m.compute([video], ["a completely unrelated sentence about taxes"],
                      device="cuda").score
    assert -1.0 <= wrong <= matched <= 1.0


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
