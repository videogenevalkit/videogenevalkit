"""Functional tests for FVD [s3d-k400] + VFID [InceptionV3].

Structural tests run always. The end-to-end tests download torchvision
weights [S3D ~30MB, InceptionV3 ~100MB] + run inference, so they're marked
`slow`. They use tiny synthetic videos to verify shape/range, not absolute
paper values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from videvalkit.metrics import SUPPORTED_METRICS, get_metric
from videvalkit.metrics.utils.frechet import (
    compute_statistics,
    fid_from_features,
    frechet_distance,
)


# ----------------------------------------------------- frechet util ---
class TestFrechetUtil:
    def test_identical_distributions_zero(self):
        rng = np.random.default_rng(0)
        feats = rng.normal(size=(50, 16))
        d = fid_from_features(feats, feats)
        assert abs(d) < 1e-6

    def test_shifted_mean_positive(self):
        rng = np.random.default_rng(1)
        a = rng.normal(size=(100, 8))
        b = a + 5.0
        d = fid_from_features(a, b)
        # mean shift of 5 across 8 dims → ~ 8 * 25 = 200
        assert d > 100

    def test_compute_statistics_shape(self):
        feats = np.random.default_rng(2).normal(size=(30, 12))
        mu, sigma = compute_statistics(feats)
        assert mu.shape == (12,)
        assert sigma.shape == (12, 12)

    def test_frechet_symmetric(self):
        rng = np.random.default_rng(3)
        a = rng.normal(size=(40, 6))
        b = rng.normal(size=(40, 6)) + 1.0
        mu1, s1 = compute_statistics(a)
        mu2, s2 = compute_statistics(b)
        d1 = frechet_distance(mu1, s1, mu2, s2)
        d2 = frechet_distance(mu2, s2, mu1, s1)
        assert abs(d1 - d2) < 1e-6


# ----------------------------------------------------- registry/shape ---
class TestRegistry:
    def test_fvd_has_s3d_backbone(self):
        assert "s3d-k400" in SUPPORTED_METRICS["fvd"]["supported_backbones"]

    def test_fvd_canonical_still_i3d(self):
        assert SUPPORTED_METRICS["fvd"]["canonical_backbone"] == "i3d-k400"

    def test_get_fvd_vfid(self):
        assert get_metric("fvd").name == "fvd"
        assert get_metric("vfid").name == "vfid"


# ----------------------------------------------------- guards ---
class TestGuards:
    def test_fvd_i3d_needs_weights(self, tmp_path, monkeypatch):
        """i3d-k400 [paper-canonical] raises FileNotFoundError pointing to
        s3d-k400 when i3d_torchscript.pt is not placed locally."""
        monkeypatch.delenv("VIDEVALKIT_FVD_I3D_PATH", raising=False)
        monkeypatch.setenv("VIDEVALKIT_CACHE_HOME", str(tmp_path / "nocache"))
        m = get_metric("fvd")
        with pytest.raises(FileNotFoundError, match="s3d-k400"):
            m.compute(
                gen_videos=[tmp_path / f"v{i}.mp4" for i in range(200)],
                ref_videos=[tmp_path / f"r{i}.mp4" for i in range(200)],
                backbone="i3d-k400",
            )

    def test_fvd_unsupported_backbone(self, tmp_path):
        m = get_metric("fvd")
        with pytest.raises(ValueError, match="not supported"):
            m.compute(
                gen_videos=[tmp_path / "v.mp4"] * 200,
                ref_videos=[tmp_path / "r.mp4"] * 200,
                backbone="nope",
            )

    def test_fvd_requires_ref(self, tmp_path):
        m = get_metric("fvd")
        with pytest.raises(ValueError, match="requires ref"):
            m.compute(
                gen_videos=[tmp_path / "v.mp4"] * 200,
                ref_videos=None, backbone="s3d-k400",
            )

    def test_fvd_tiny_sample_errors(self, tmp_path):
        m = get_metric("fvd")
        with pytest.raises(ValueError, match="ERROR.*below minimum"):
            m.compute(
                gen_videos=[tmp_path / "v.mp4"] * 50,
                ref_videos=[tmp_path / "r.mp4"] * 200,
                backbone="s3d-k400",
            )


# ----------------------------------------------------- end-to-end (slow) ---
def _make_tiny_video(path: Path, n_frames: int = 16, hw: int = 64, seed: int = 0):
    import imageio
    rng = np.random.default_rng(seed)
    w = imageio.get_writer(str(path), fps=8, codec="libx264")
    try:
        for _ in range(n_frames):
            w.append_data(rng.integers(0, 256, (hw, hw, 3), dtype=np.uint8))
    finally:
        w.close()


@pytest.mark.slow
def test_vfid_end_to_end(tmp_path):
    """VFID on tiny synthetic videos — verify it produces a finite score."""
    gen = tmp_path / "gen"; gen.mkdir()
    ref = tmp_path / "ref"; ref.mkdir()
    for i in range(6):
        _make_tiny_video(gen / f"g{i}.mp4", seed=i)
        _make_tiny_video(ref / f"r{i}.mp4", seed=100 + i)
    m = get_metric("vfid")
    result = m.compute(
        gen_videos=list(gen.glob("*.mp4")),
        ref_videos=list(ref.glob("*.mp4")),
        device="cpu",
        allow_tiny_sample=True,
    )
    assert result.metric == "vfid"
    assert np.isfinite(result.score)
    assert result.score >= 0
    assert result.n_gen == 6 and result.n_ref == 6


@pytest.mark.slow
def test_fvd_s3d_end_to_end(tmp_path):
    """FVD [s3d-k400] on tiny synthetic videos — finite score."""
    gen = tmp_path / "gen"; gen.mkdir()
    ref = tmp_path / "ref"; ref.mkdir()
    for i in range(6):
        _make_tiny_video(gen / f"g{i}.mp4", seed=i)
        _make_tiny_video(ref / f"r{i}.mp4", seed=100 + i)
    m = get_metric("fvd")
    result = m.compute(
        gen_videos=list(gen.glob("*.mp4")),
        ref_videos=list(ref.glob("*.mp4")),
        backbone="s3d-k400",
        device="cpu",
        allow_tiny_sample=True,
    )
    assert result.backbone == "s3d-k400"
    assert np.isfinite(result.score)
