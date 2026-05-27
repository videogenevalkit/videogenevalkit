"""Tests for capability taxonomy + resolver.

Per docs/CAPABILITY_TAGS_DESIGN.md §3 / §7 / §9 (user 2026-05-20).
"""

from __future__ import annotations

import pytest

from videvalkit.configs.capability_taxonomy import (
    ALL_TAGS,
    SUB_TAGS_BY_TOP,
    SUB_TAG_TO_TOP,
    TAG_DESCRIPTIONS,
    TAG_SCHEMA_VERSION,
    TOP_LEVEL_TAGS,
    expand_capability,
    is_valid_tag,
    parent_of,
)
from videvalkit.core.capability import (
    Contributor,
    _dedup_by_canonical,
    all_contributors,
    coverage_report,
    resolve_capability,
)


# ----------------------------------------------------- taxonomy shape ---
class TestTaxonomyShape:
    def test_10_top_level_tags(self):
        assert len(TOP_LEVEL_TAGS) == 10

    def test_34_sub_tags(self):
        total = sum(len(v) for v in SUB_TAGS_BY_TOP.values())
        assert total == 34

    def test_44_total(self):
        assert len(ALL_TAGS) == 44

    def test_all_top_have_subs(self):
        for top in TOP_LEVEL_TAGS:
            assert top in SUB_TAGS_BY_TOP
            assert len(SUB_TAGS_BY_TOP[top]) >= 3

    def test_sub_to_top_consistent(self):
        for sub, top in SUB_TAG_TO_TOP.items():
            assert sub in SUB_TAGS_BY_TOP[top]

    def test_descriptions_complete(self):
        """Every tag has a human-readable description for `capabilities show`."""
        missing = [t for t in ALL_TAGS if t not in TAG_DESCRIPTIONS]
        assert not missing, f"missing TAG_DESCRIPTIONS for: {missing}"

    def test_schema_version_v1(self):
        assert TAG_SCHEMA_VERSION == 1


# ----------------------------------------------------- validation ---
class TestValidation:
    def test_is_valid_top(self):
        assert is_valid_tag("motion")
        assert is_valid_tag("realism")

    def test_is_valid_sub(self):
        assert is_valid_tag("motion.smoothness")
        assert is_valid_tag("comp.numeracy")

    def test_invalid_free_form(self):
        assert not is_valid_tag("Motion")  # case sensitive
        assert not is_valid_tag("speed")
        assert not is_valid_tag("motion.foo")
        assert not is_valid_tag("")

    def test_parent_of_sub(self):
        assert parent_of("motion.smoothness") == "motion"
        assert parent_of("comp.spatial") == "compositional"

    def test_parent_of_top_is_none(self):
        assert parent_of("motion") is None


# ----------------------------------------------------- expand ---
class TestExpand:
    def test_top_expands_to_self_plus_subs(self):
        out = expand_capability("motion")
        assert "motion" in out
        assert "motion.smoothness" in out
        assert "motion.accuracy" in out
        assert len(out) == 1 + 4  # motion + 4 sub

    def test_sub_returns_self_only(self):
        out = expand_capability("motion.smoothness")
        assert out == ["motion.smoothness"]

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown capability tag"):
            expand_capability("not-a-tag")


# ----------------------------------------------------- bench dim_tags ---
class TestBenchDimTags:
    def test_vbench_dim_tags_complete_16(self):
        from videvalkit.configs import SUPPORTED_BENCHMARKS
        vbench = SUPPORTED_BENCHMARKS["vbench"]
        assert "dim_tags" in vbench
        assert len(vbench["dim_tags"]) == 16

    def test_all_dim_tags_from_controlled_vocab(self):
        """Hygiene: every tag assigned to a dim must be in ALL_TAGS."""
        from videvalkit.configs import SUPPORTED_BENCHMARKS
        bad = []
        for bench, cfg in SUPPORTED_BENCHMARKS.items():
            for dim, tags in cfg.get("dim_tags", {}).items():
                for t in tags:
                    if t not in ALL_TAGS:
                        bad.append((bench, dim, t))
        assert not bad, f"dim_tags using free-form tags: {bad}"


# ----------------------------------------------------- resolver ---
class TestResolver:
    def test_resolve_motion_pulls_in_vbench_dims(self):
        contributors = resolve_capability("motion")
        names = [c.source_name for c in contributors]
        assert "vbench/motion_smoothness" in names
        assert "vbench/dynamic_degree" in names

    def test_resolve_sub_tag_narrows(self):
        contributors = resolve_capability("motion.smoothness")
        names = [c.source_name for c in contributors]
        assert "vbench/motion_smoothness" in names
        # dynamic_degree tagged motion.magnitude — should NOT match
        assert "vbench/dynamic_degree" not in names

    def test_resolve_spatial_uses_prefix_form(self):
        # Sub-tag form is "comp.spatial", not "compositional.spatial"
        with pytest.raises(ValueError):
            resolve_capability("compositional.spatial")

        # Correct sub-tag form
        contributors = resolve_capability("comp.spatial")
        names = [c.source_name for c in contributors]
        assert "vbench/spatial_relationship" in names
        assert "t2vcompbench/spatial_relationship" in names

    def test_resolve_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown capability"):
            resolve_capability("Motion")  # case sensitive

    def test_resolve_realism_includes_distribution(self):
        """When metrics module lands, FVD etc. should resolve here.
        For now, no metric contributes; just check no crash."""
        contributors = resolve_capability("realism")
        # Could be empty until metrics module lands — should not crash
        assert isinstance(contributors, list)


# ----------------------------------------------------- dedup ---
class TestDedup:
    def test_dedup_same_canonical_keeps_one(self):
        # Two contributors with same canonical_source
        c1 = Contributor(
            source_kind="bench_dim",
            source_name="vbench/motion_smoothness",
            canonical_source="canon::motion-smoothness",
            tags=["motion.smoothness"],
            cfg={},
        )
        c2 = Contributor(
            source_kind="metric",
            source_name="motion-smoothness",
            canonical_source="canon::motion-smoothness",
            tags=["motion.smoothness"],
            cfg={},
        )
        out = _dedup_by_canonical([c1, c2])
        assert len(out) == 1
        # Should prefer metric over bench_dim
        assert out[0].source_kind == "metric"

    def test_dedup_different_canonical_keeps_both(self):
        c1 = Contributor(source_kind="bench_dim", source_name="a",
                         canonical_source="canon::a", tags=["motion"], cfg={})
        c2 = Contributor(source_kind="bench_dim", source_name="b",
                         canonical_source="canon::b", tags=["motion"], cfg={})
        out = _dedup_by_canonical([c1, c2])
        assert len(out) == 2


# ----------------------------------------------------- coverage report ---
class TestCoverageReport:
    def test_report_has_all_tags(self):
        report = coverage_report()
        for t in ALL_TAGS:
            assert t in report

    def test_motion_has_contributors(self):
        report = coverage_report()
        assert len(report["motion.smoothness"]) >= 1
        assert len(report["motion.magnitude"]) >= 1
