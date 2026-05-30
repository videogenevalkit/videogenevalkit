"""Tests for the VideoMAE-base FVD/KVD backbone."""

from __future__ import annotations

from videvalkit.metrics.fvd import FVD
from videvalkit.metrics.kvd import KVD


class TestSupportedBackbones:
    def test_fvd_lists_videomae_base(self):
        assert "videomae-base" in FVD().supported_backbones

    def test_kvd_lists_videomae_base(self):
        assert "videomae-base" in KVD().supported_backbones


class TestRegistryConsistency:
    """Class-level supported_backbones must include the new entry too."""

    def test_fvd_registry_supported_backbones(self):
        from videvalkit.metrics.registry import SUPPORTED_METRICS
        assert "videomae-base" in SUPPORTED_METRICS["fvd"]["supported_backbones"]

    def test_kvd_registry_supported_backbones(self):
        from videvalkit.metrics.registry import SUPPORTED_METRICS
        assert "videomae-base" in SUPPORTED_METRICS["kvd"]["supported_backbones"]


class TestVideoMAEImports:
    def test_backbone_module_imports(self):
        from videvalkit.metrics.backbones.videomae import (  # noqa: F401
            VideoMAEFeatureExtractor,
            DEFAULT_REPO,
            FEAT_DIM,
            N_FRAMES,
        )
