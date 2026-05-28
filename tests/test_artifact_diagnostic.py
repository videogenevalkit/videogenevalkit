"""Tests for the artifact-diagnostic metric (Artifact-Bench v0.2 port).

The metric needs a VLM judge; these tests use a fake judge so the parsing +
aggregation logic is covered without an endpoint. A real run is exercised
only when a judge endpoint is configured.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from videvalkit.metrics import SUPPORTED_METRICS, get_metric
from videvalkit.metrics.artifact_diagnostic import (
    ArtifactDiagnostic,
    _parse_present_artifacts,
)
from videvalkit.metrics.artifact_taxonomy import (
    ARTIFACT_CATEGORIES,
    ARTIFACT_FAMILIES,
    ARTIFACT_TYPES,
    N_ARTIFACT_TYPES,
    TYPE_TO_CATEGORY,
)


class FakeJudge:
    model = "fake-vlm"

    def __init__(self, responses: list[list[str]]):
        self._responses = list(responses)
        self.calls = 0

    def chat_with_frames(self, video_path, prompt, **kw):
        present = self._responses[self.calls % len(self._responses)]
        self.calls += 1
        return {"content": json.dumps({"present": present, "notes": "x"})}


# ----------------------------------------------------- taxonomy ---
class TestTaxonomy:
    def test_30_types(self):
        assert N_ARTIFACT_TYPES == 30
        assert len(ARTIFACT_TYPES) == 30
        assert len(set(ARTIFACT_TYPES)) == 30  # unique

    def test_11_families_3_categories(self):
        assert len(ARTIFACT_FAMILIES) == 11
        assert len(ARTIFACT_CATEGORIES) == 3

    def test_every_type_maps_to_a_category(self):
        for t in ARTIFACT_TYPES:
            assert TYPE_TO_CATEGORY[t] in ARTIFACT_CATEGORIES


# ----------------------------------------------------- registry ---
class TestRegistry:
    def test_registered_needs_judge(self):
        cfg = SUPPORTED_METRICS["artifact-diagnostic"]
        assert cfg["needs_judge"] is True
        assert cfg["kind"] == "per_video_with_vlm_judge"

    def test_get_metric(self):
        assert isinstance(get_metric("artifact-diagnostic"), ArtifactDiagnostic)


# ----------------------------------------------------- parsing ---
class TestParsing:
    def test_json_object_present_key(self):
        out = _parse_present_artifacts('{"present": ["flickering", "noise_grain"]}')
        assert out == [t for t in ARTIFACT_TYPES if t in {"flickering", "noise_grain"}]

    def test_bare_array(self):
        assert _parse_present_artifacts('["face_distortion"]') == ["face_distortion"]

    def test_fenced_json(self):
        s = '```json\n{"present": ["flickering"]}\n```'
        assert _parse_present_artifacts(s) == ["flickering"]

    def test_unknown_ids_dropped(self):
        out = _parse_present_artifacts('{"present": ["flickering", "made_up_thing"]}')
        assert out == ["flickering"]

    def test_empty_clean(self):
        assert _parse_present_artifacts('{"present": []}') == []

    def test_returns_taxonomy_order(self):
        # input order shuffled; output must follow ARTIFACT_TYPES order
        out = _parse_present_artifacts('["noise_grain", "flickering"]')
        assert out.index("flickering") < out.index("noise_grain")


# ----------------------------------------------------- compute (mock judge) ---
class TestComputeMock:
    def test_no_judge_raises(self, tmp_path):
        m = get_metric("artifact-diagnostic")
        with pytest.raises(NotImplementedError, match="needs a VLM judge"):
            m.compute([tmp_path / "v.mp4"])

    def test_empty_videos_returns_zero(self):
        m = get_metric("artifact-diagnostic")
        r = m.compute([], judge=FakeJudge([[]]))
        assert r.n_videos == 0
        assert r.mean_artifacts_per_video == 0.0

    def test_aggregation(self, tmp_path):
        v0 = tmp_path / "v0.mp4"
        v1 = tmp_path / "v1.mp4"
        judge = FakeJudge([["flickering", "face_distortion"], []])
        m = get_metric("artifact-diagnostic")
        r = m.compute([v0, v1], judge=judge)

        assert r.n_videos == 2
        assert r.mean_artifacts_per_video == pytest.approx(1.0)
        assert r.per_artifact_rate["flickering"] == pytest.approx(0.5)
        assert r.per_artifact_rate["face_distortion"] == pytest.approx(0.5)
        assert r.per_artifact_rate["noise_grain"] == 0.0
        assert r.per_category_rate["temporal_inconsistencies"] == pytest.approx(0.5)
        assert r.per_category_rate["structural_distortions"] == pytest.approx(0.5)
        assert r.per_category_rate["semantic_incoherence"] == 0.0
        top_names = {t for t, _ in r.top_artifacts}
        assert top_names == {"flickering", "face_distortion"}
        assert r.per_video[str(v0)] == ["flickering", "face_distortion"]
        assert r.per_video[str(v1)] == []
        assert r.judge_model == "fake-vlm"

    def test_bad_video_skipped(self, tmp_path):
        class FlakyJudge:
            model = "flaky"
            def chat_with_frames(self, video_path, prompt, **kw):
                if Path(video_path).name.startswith("bad"):
                    raise RuntimeError("boom")
                return {"content": '{"present": ["flickering"]}'}

        m = get_metric("artifact-diagnostic")
        r = m.compute([tmp_path / "good.mp4", tmp_path / "bad.mp4"], judge=FlakyJudge())
        assert r.n_videos == 1  # bad one skipped, not fatal
        assert r.per_artifact_rate["flickering"] == pytest.approx(1.0)
