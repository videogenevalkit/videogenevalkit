"""Tests for KVD + CLIP-FVD now functional.

MMD/Fréchet math + registry + guards run always. End-to-end [model download
+ inference on tiny videos] marked slow.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from videvalkit.metrics import get_metric
from videvalkit.metrics.utils.mmd import polynomial_mmd2


# ----------------------------------------------------- mmd util ---
class TestMMD:
    def test_identical_near_zero(self):
        rng = np.random.default_rng(0)
        f = rng.normal(size=(60, 32))
        # identical sets: unbiased MMD2 ~ 0 [not exactly, but small]
        d = polynomial_mmd2(f, f)
        assert abs(d) < 1.0

    def test_shifted_positive(self):
        rng = np.random.default_rng(1)
        a = rng.normal(size=(80, 16))
        b = rng.normal(size=(80, 16)) + 3.0
        assert polynomial_mmd2(a, b) > 0

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match=">=2 samples"):
            polynomial_mmd2(np.zeros((1, 8)), np.zeros((5, 8)))


# ----------------------------------------------------- registry/guards ---
class TestKVD:
    def test_supported_backbones(self):
        m = get_metric("kvd")
        assert "s3d-k400" in m.supported_backbones

    def test_requires_ref(self, tmp_path):
        m = get_metric("kvd")
        with pytest.raises(ValueError, match="requires ref"):
            m.compute(gen_videos=[tmp_path / "v.mp4"] * 60, ref_videos=None)

    def test_default_falls_back_to_s3d(self, tmp_path, monkeypatch):
        from unittest.mock import patch
        monkeypatch.delenv("VIDEVALKIT_FVD_I3D_PATH", raising=False)
        monkeypatch.setenv("VIDEVALKIT_CACHE_HOME", str(tmp_path / "nocache"))
        m = get_metric("kvd")

        class Fake:
            def __init__(self, device="cpu"): pass
            def extract_many(self, paths, **_):
                return np.random.default_rng(0).normal(size=(len(paths), 1024))

        with patch("videvalkit.metrics.backbones.s3d_k400.S3DFeatureExtractor", new=Fake):
            r = m.compute(
                gen_videos=[tmp_path / f"v{i}.mp4" for i in range(60)],
                ref_videos=[tmp_path / f"r{i}.mp4" for i in range(60)],
            )
        assert r.backbone == "s3d-k400"
        assert r.meta["kernel"] == "polynomial degree=3"


class TestCLIPFVD:
    def test_requires_ref(self, tmp_path):
        m = get_metric("clip-fvd")
        with pytest.raises(ValueError, match="requires ref"):
            m.compute(gen_videos=[tmp_path / "v.mp4"] * 200, ref_videos=None)

    def test_unsupported_backbone(self, tmp_path):
        m = get_metric("clip-fvd")
        with pytest.raises(ValueError, match="not supported"):
            m.compute(gen_videos=[tmp_path / "v.mp4"] * 200,
                      ref_videos=[tmp_path / "r.mp4"] * 200, backbone="nope")


# ----------------------------------------------------- e2e (slow) ---
def _tiny(path: Path, seed: int = 0, n: int = 16, hw: int = 64):
    import imageio
    rng = np.random.default_rng(seed)
    w = imageio.get_writer(str(path), fps=8, codec="libx264")
    try:
        for _ in range(n):
            w.append_data(rng.integers(0, 256, (hw, hw, 3), dtype=np.uint8))
    finally:
        w.close()


@pytest.mark.slow
def test_kvd_s3d_end_to_end(tmp_path):
    gen = tmp_path / "g"; gen.mkdir(); ref = tmp_path / "r"; ref.mkdir()
    for i in range(6):
        _tiny(gen / f"g{i}.mp4", seed=i); _tiny(ref / f"r{i}.mp4", seed=100 + i)
    m = get_metric("kvd")
    r = m.compute(gen_videos=list(gen.glob("*.mp4")),
                  ref_videos=list(ref.glob("*.mp4")),
                  backbone="s3d-k400", device="cpu", allow_tiny_sample=True)
    assert np.isfinite(r.score)


@pytest.mark.slow
def test_clip_fvd_end_to_end(tmp_path):
    gen = tmp_path / "g"; gen.mkdir(); ref = tmp_path / "r"; ref.mkdir()
    for i in range(6):
        _tiny(gen / f"g{i}.mp4", seed=i); _tiny(ref / f"r{i}.mp4", seed=100 + i)
    m = get_metric("clip-fvd")
    r = m.compute(gen_videos=list(gen.glob("*.mp4")),
                  ref_videos=list(ref.glob("*.mp4")),
                  device="cpu", allow_tiny_sample=True)
    assert np.isfinite(r.score)
    assert "experimental" in r.meta["note"].lower()
