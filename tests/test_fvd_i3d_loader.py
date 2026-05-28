"""Tests for the I3D-K400 backbone loader [paper-canonical FVD].

Weights aren't bundled, so the load path is tested via:
  * absent → clear FileNotFoundError [run always]
  * present → load + extract [skipif weights absent; runs on server once
    i3d_torchscript.pt is placed]
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from videvalkit.metrics import get_metric
from videvalkit.metrics.backbones.i3d_k400 import (
    I3DFeatureExtractor,
    i3d_weights_path,
)


# ----------------------------------------------------- weights resolution ---
class TestWeightsPath:
    def test_env_override(self, tmp_path, monkeypatch):
        wp = tmp_path / "my_i3d.pt"
        wp.write_bytes(b"fake")
        monkeypatch.setenv("VIDEVALKIT_FVD_I3D_PATH", str(wp))
        assert i3d_weights_path() == wp

    def test_absent_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VIDEVALKIT_FVD_I3D_PATH", raising=False)
        monkeypatch.setenv("VIDEVALKIT_CACHE_HOME", str(tmp_path / "nocache"))
        assert i3d_weights_path() is None


# ----------------------------------------------------- absent → clear error ---
class TestAbsentWeights:
    def test_extractor_raises_with_pointer(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VIDEVALKIT_FVD_I3D_PATH", raising=False)
        monkeypatch.setenv("VIDEVALKIT_CACHE_HOME", str(tmp_path / "nocache"))
        with pytest.raises(FileNotFoundError, match="i3d_torchscript.pt|s3d-k400"):
            I3DFeatureExtractor(device="cpu")

    def test_fvd_i3d_default_errors_when_weights_absent(self, tmp_path, monkeypatch):
        """metric run --name fvd [default i3d-k400] → FileNotFoundError pointing
        at s3d-k400 when weights aren't placed."""
        monkeypatch.delenv("VIDEVALKIT_FVD_I3D_PATH", raising=False)
        monkeypatch.setenv("VIDEVALKIT_CACHE_HOME", str(tmp_path / "nocache"))
        m = get_metric("fvd")
        gen = [tmp_path / f"v{i}.mp4" for i in range(200)]
        ref = [tmp_path / f"r{i}.mp4" for i in range(200)]
        with pytest.raises(FileNotFoundError, match="s3d-k400"):
            m.compute(gen_videos=gen, ref_videos=ref, backbone="i3d-k400")


# ----------------------------------------------------- present → functional ---
@pytest.mark.skipif(
    i3d_weights_path() is None,
    reason="i3d_torchscript.pt not placed; paper-canonical FVD load not testable",
)
@pytest.mark.slow
def test_i3d_loads_when_present():
    """When weights are placed, the extractor loads. Runs on server once
    i3d_torchscript.pt is staged."""
    ex = I3DFeatureExtractor(device="cpu")
    assert ex.detector is not None
