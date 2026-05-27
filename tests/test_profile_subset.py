"""Tests for eval profile + subset abstraction."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest
from pydantic import ValidationError

from videvalkit.core.profile import (
    SUPPORTED_PROFILES,
    EstimatedCost,
    FrameSamplingSpec,
    ProfileSpec,
    resolve_profile,
)
from videvalkit.core.subset import (
    SUBSET_SCHEMA_VERSION,
    Subset,
    SubsetSpec,
    find_subset,
)
from videvalkit.core.types import PromptItem


# ============================================================
# Profile tests
# ============================================================
class TestProfileRegistry:
    def test_three_profiles_registered(self):
        assert set(SUPPORTED_PROFILES) == {"quick", "standard", "full"}

    def test_quick_has_4_frames(self):
        assert SUPPORTED_PROFILES["quick"].frame_sampling.n_frames == 4

    def test_full_has_8_frames_5_samples(self):
        full = SUPPORTED_PROFILES["full"]
        assert full.frame_sampling.n_frames == 8
        assert full.samples_per_prompt == 5
        assert full.subset is None

    def test_quick_has_subset(self):
        assert SUPPORTED_PROFILES["quick"].subset == "quick_v1"

    def test_estimated_wallclock_ordering(self):
        """quick < standard < full"""
        q = SUPPORTED_PROFILES["quick"].estimated.wallclock_min
        s = SUPPORTED_PROFILES["standard"].estimated.wallclock_min
        f = SUPPORTED_PROFILES["full"].estimated.wallclock_min
        assert q < s < f


class TestResolveProfile:
    def test_none_resolves_to_full(self):
        """Back-compat: --profile not passed → full corpus."""
        p = resolve_profile(None)
        assert p.name == "full"

    def test_quick(self):
        assert resolve_profile("quick").name == "quick"

    def test_unknown_raises_with_suggestion(self):
        with pytest.raises(KeyError, match="unknown profile"):
            resolve_profile("quik")

    def test_unknown_suggestion(self):
        try:
            resolve_profile("quik")
        except KeyError as e:
            assert "quick" in str(e)


class TestProfileSpec:
    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            ProfileSpec(name="x", foo="bar")

    def test_valid_minimal(self):
        p = ProfileSpec(name="custom")
        assert p.judge == "default"
        assert p.samples_per_prompt == 1


# ============================================================
# Subset tests
# ============================================================
@pytest.fixture
def valid_subset_dict() -> dict:
    return {
        "schema_version": 1,
        "subset_name": "quick_v1",
        "benchmark": "worldjen",
        "created": "2026-05-20",
        "n_prompts": 3,
        "selection_method": "stratified_seeded",
        "selection_seed": 42,
        "calibration": {
            "method": "spearman",
            "validation_models": ["Kling-v2.6", "Sora", "HunyuanVideo"],
            "spearman_rho_overall": 0.91,
            "spearman_rho_per_dim": {"motion_stability": 0.88},
            "max_dim_disagreement": 0.08,
        },
        "prompt_ids": ["wj_001", "wj_002", "wj_003"],
    }


@pytest.fixture
def subset_file(tmp_path: Path, valid_subset_dict: dict) -> Path:
    p = tmp_path / "quick_v1.json"
    p.write_text(json.dumps(valid_subset_dict))
    return p


class TestSubsetLoad:
    def test_valid_subset_loads(self, subset_file):
        s = Subset.from_file(subset_file)
        assert s.name == "quick_v1"
        assert s.benchmark == "worldjen"
        assert s.n_prompts == 3

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="subset file"):
            Subset.from_file(tmp_path / "nope.json")

    def test_wrong_schema_version_rejected(self, tmp_path, valid_subset_dict):
        valid_subset_dict["schema_version"] = 99
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(valid_subset_dict))
        with pytest.raises(ValueError, match="schema_version=99"):
            Subset.from_file(p)


class TestSubsetFilter:
    def test_filter_keeps_only_listed(self, subset_file):
        s = Subset.from_file(subset_file)
        prompts = [
            PromptItem(prompt_id="wj_001", text="a"),
            PromptItem(prompt_id="wj_999", text="b"),  # not in subset
            PromptItem(prompt_id="wj_002", text="c"),
        ]
        out = s.filter_prompts(prompts)
        assert [p.prompt_id for p in out] == ["wj_001", "wj_002"]

    def test_filter_preserves_input_order(self, subset_file):
        """Order: input order is preserved, NOT subset order."""
        s = Subset.from_file(subset_file)
        prompts = [
            PromptItem(prompt_id="wj_003", text="c"),
            PromptItem(prompt_id="wj_001", text="a"),
            PromptItem(prompt_id="wj_002", text="b"),
        ]
        out = s.filter_prompts(prompts)
        assert [p.prompt_id for p in out] == ["wj_003", "wj_001", "wj_002"]

    def test_filter_warns_on_missing(self, subset_file, caplog):
        s = Subset.from_file(subset_file)
        # Only 1 of 3 expected prompts found → warning
        prompts = [PromptItem(prompt_id="wj_001", text="a")]
        s.filter_prompts(prompts)
        assert any("not found in" in r.message for r in caplog.records)

    def test_hash_deterministic(self, subset_file):
        s1 = Subset.from_file(subset_file)
        s2 = Subset.from_file(subset_file)
        assert s1.hash() == s2.hash()
        assert len(s1.hash()) == 64  # sha256 hex


class TestFindSubset:
    def test_search_dir_finds_subset(self, subset_file):
        """Look up subset by directory hint."""
        s = find_subset(
            "worldjen", "quick_v1",
            search_dirs=[subset_file.parent],
        )
        assert s.name == "quick_v1"

    def test_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            find_subset(
                "no-such-bench", "no-subset",
                search_dirs=[tmp_path],
            )
