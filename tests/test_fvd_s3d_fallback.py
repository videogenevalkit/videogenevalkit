"""Test FVD default → s3d-k400 auto-fallback for monitoring.

User decision 2026-05-28: S3D is fine for training monitoring [trend, not
paper-exact]. So default `metric run --name fvd` [no backbone] auto-falls-
back to s3d-k400 when i3d weights are absent, with a warning. Explicit
--backbone i3d-k400 still requires weights [strict paper-repro].
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from videvalkit.metrics import get_metric


@pytest.fixture
def no_i3d(monkeypatch, tmp_path):
    monkeypatch.delenv("VIDEVALKIT_FVD_I3D_PATH", raising=False)
    monkeypatch.setenv("VIDEVALKIT_CACHE_HOME", str(tmp_path / "nocache"))


class TestDefaultFallback:
    def test_default_falls_back_to_s3d(self, no_i3d, tmp_path, caplog):
        """No backbone + no i3d weights → s3d-k400, with a warning.
        We stub S3D extraction so the test stays fast/CPU."""
        m = get_metric("fvd")

        class FakeExtractor:
            def __init__(self, device="cpu"):
                pass

            def extract_many(self, paths, **_):
                return np.random.default_rng(0).normal(size=(len(paths), 1024))

        with patch("videvalkit.metrics.backbones.s3d_k400.S3DFeatureExtractor",
                   new=FakeExtractor):
            with caplog.at_level(logging.WARNING):
                result = m.compute(
                    gen_videos=[tmp_path / f"v{i}.mp4" for i in range(200)],
                    ref_videos=[tmp_path / f"r{i}.mp4" for i in range(200)],
                    # no backbone → default i3d-k400 → fallback
                )
        assert result.backbone == "s3d-k400"
        assert "fallback" in result.meta["note"].lower()
        assert any("falling back to s3d-k400" in r.message for r in caplog.records)

    def test_explicit_i3d_still_strict(self, no_i3d, tmp_path):
        """Explicit --backbone i3d-k400 must still fail when weights absent."""
        m = get_metric("fvd")
        with pytest.raises(FileNotFoundError, match="s3d-k400|i3d_torchscript"):
            m.compute(
                gen_videos=[tmp_path / f"v{i}.mp4" for i in range(200)],
                ref_videos=[tmp_path / f"r{i}.mp4" for i in range(200)],
                backbone="i3d-k400",   # explicit → strict
            )

    def test_explicit_s3d_works(self, no_i3d, tmp_path):
        m = get_metric("fvd")

        class FakeExtractor:
            def __init__(self, device="cpu"):
                pass

            def extract_many(self, paths, **_):
                return np.random.default_rng(1).normal(size=(len(paths), 1024))

        with patch("videvalkit.metrics.backbones.s3d_k400.S3DFeatureExtractor",
                   new=FakeExtractor):
            result = m.compute(
                gen_videos=[tmp_path / f"v{i}.mp4" for i in range(200)],
                ref_videos=[tmp_path / f"r{i}.mp4" for i in range(200)],
                backbone="s3d-k400",
            )
        assert result.backbone == "s3d-k400"
        assert "fallback" not in result.meta["note"].lower()
