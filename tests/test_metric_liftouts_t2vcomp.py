"""Tests for t2vcompbench CV lift-out metrics [numeracy / spatial-relationship].

Meta-parsing [prompt → meta] is pure Python — tested without GPU.
The compute() path needs GroundingDINO + frames — marked needs_gpu.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from videvalkit.metrics import SUPPORTED_METRICS, get_metric, list_metrics
from videvalkit.metrics.t2vcompbench_dim import Numeracy, SpatialRelationship


SPECIALIZED_CV = ["numeracy", "spatial-relationship"]


# ----------------------------------------------------- registry ---
class TestRegistry:
    def test_both_registered(self):
        for name in SPECIALIZED_CV:
            assert name in SUPPORTED_METRICS

    def test_judge_free(self):
        """These are GroundingDINO CV, NOT MLLM judge."""
        for name in SPECIALIZED_CV:
            assert SUPPORTED_METRICS[name]["needs_judge"] is False
            assert SUPPORTED_METRICS[name]["compute_kind"] == "local_vision"

    def test_per_prompt_kind(self):
        for name in SPECIALIZED_CV:
            assert SUPPORTED_METRICS[name]["kind"] == "per_prompt_reference_free"

    def test_source_points_to_t2vcompbench(self):
        assert SUPPORTED_METRICS["numeracy"]["source"].startswith("t2vcompbench/")
        assert SUPPORTED_METRICS["spatial-relationship"]["source"].startswith("t2vcompbench/")

    def test_tags_in_vocab(self):
        from videvalkit.configs.capability_taxonomy import ALL_TAGS
        for name in SPECIALIZED_CV:
            for t in SUPPORTED_METRICS[name]["tags"]:
                assert t in ALL_TAGS


# ----------------------------------------------------- factory ---
class TestFactory:
    def test_get_numeracy(self):
        m = get_metric("numeracy")
        assert isinstance(m, Numeracy)
        assert m.dim == "generative_numeracy"

    def test_get_spatial(self):
        m = get_metric("spatial-relationship")
        assert isinstance(m, SpatialRelationship)
        assert m.dim == "spatial_relationships"


# ----------------------------------------------------- meta parsing (pure) ---
class TestMetaParsing:
    def test_numeracy_meta_from_prompt(self):
        m = Numeracy()
        meta = m._build_meta("three cats and two dogs playing")
        # _parse_count_noun extracts (count, noun) pairs
        assert "objects" in meta
        assert "numbers" in meta
        assert len(meta["objects"]) == len(meta["numbers"])
        # 3 cats, 2 dogs
        assert 3 in meta["numbers"] or "3" in str(meta["numbers"])

    def test_numeracy_meta_invariant(self):
        """objects and numbers always have matching length [upstream parser
        treats 'a'/'an' as count 1, so 'a beautiful sunset' → 1 pair]."""
        m = Numeracy()
        meta = m._build_meta("a beautiful sunset")
        assert len(meta["objects"]) == len(meta["numbers"])

    def test_spatial_meta_from_prompt(self):
        m = SpatialRelationship()
        meta = m._build_meta("a cat on the left of a dog")
        assert "object_1" in meta
        assert "spatial" in meta
        assert "object_2" in meta

    def test_spatial_meta_no_triple(self):
        m = SpatialRelationship()
        meta = m._build_meta("a beautiful landscape")
        # No spatial triple → all None
        assert meta["object_1"] is None


# ----------------------------------------------------- input validation ---
class TestInputValidation:
    def test_mismatched_lengths(self):
        m = Numeracy()
        with pytest.raises(ValueError, match="match in length"):
            # mismatched without triggering scorer load → patch _ensure_scorer
            m._ensure_scorer = lambda: None  # type: ignore
            m.compute(videos=[Path("a.mp4")], prompts=["x", "y"])


# ----------------------------------------------------- capability wiring ---
class TestCapabilityWiring:
    def test_numeracy_in_comp_numeracy(self):
        from videvalkit.core.capability import resolve_capability
        contributors = resolve_capability("comp.numeracy")
        names = [c.source_name for c in contributors]
        assert any("numeracy" in n for n in names)

    def test_spatial_in_comp_spatial(self):
        from videvalkit.core.capability import resolve_capability
        contributors = resolve_capability("comp.spatial")
        names = [c.source_name for c in contributors]
        assert any("spatial-relationship" in n for n in names)


# ----------------------------------------------------- bit-exact (needs GPU) ---
@pytest.mark.needs_gpu
def test_numeracy_bit_exact_lift():
    """metric --name numeracy == eval --bench t2vcompbench --dim
    generative_numeracy, ≤ 1e-6. Needs GroundingDINO + GPU + sample videos."""
    pytest.skip("needs GroundingDINO ckpt + GPU + sample videos — run on server")
