"""Tests for ManifestBenchmark — YAML-driven Track A adapter.

Per docs/INTEGRATION_FRAMEWORK_DESIGN.md §3.2.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
from pydantic import ValidationError

from videvalkit.core.manifest_benchmark import (
    MANIFEST_SCHEMA_VERSION,
    ManifestBenchmark,
    ManifestSpec,
    benchmark_from_manifest,
    load_manifest,
)
from videvalkit.core.types import PromptItem


@pytest.fixture
def minimal_manifest_yaml() -> str:
    return dedent("""\
        schema_version: 1
        name: my_bench
        version: 0.1.0
        description: A test bench
        env: videvalkit
        needs_gpu: false
        needs_judge: false
        prompts:
          source: jsonl
          path: prompts.jsonl
        dimensions:
          - name: visual_quality
            weight: 0.3
            scorer: clip-score
            tags: [vq.aesthetic]
          - name: motion
            weight: 0.7
            scorer: motion-smoothness
            tags: [motion.smoothness]
        video_layout: "{model}/{prompt_id}-{sample_index}.mp4"
        aggregator: weighted_sum
    """)


@pytest.fixture
def manifest_dir(tmp_path: Path, minimal_manifest_yaml: str) -> Path:
    (tmp_path / "manifest.yaml").write_text(minimal_manifest_yaml)
    # also drop a prompts.jsonl
    (tmp_path / "prompts.jsonl").write_text(
        '{"prompt_id": "p1", "text": "a cat", "dimensions": ["visual_quality", "motion"]}\n'
        '{"prompt_id": "p2", "text": "a dog running", "dimensions": ["motion"]}\n'
    )
    return tmp_path


# ----------------------------------------------------- schema validation ---
class TestManifestSchema:
    def test_minimal_valid(self, manifest_dir):
        spec = load_manifest(manifest_dir / "manifest.yaml")
        assert spec.name == "my_bench"
        assert spec.schema_version == MANIFEST_SCHEMA_VERSION
        assert len(spec.dimensions) == 2

    def test_wrong_schema_version_rejected(self, tmp_path):
        (tmp_path / "manifest.yaml").write_text(dedent("""\
            schema_version: 2
            name: x
            prompts:
              source: jsonl
              path: p.jsonl
            dimensions: []
        """))
        with pytest.raises(ValidationError, match="schema_version"):
            load_manifest(tmp_path / "manifest.yaml")

    def test_missing_required_field_rejected(self, tmp_path):
        (tmp_path / "manifest.yaml").write_text("schema_version: 1\nname: x\n")
        with pytest.raises(ValidationError):
            load_manifest(tmp_path / "manifest.yaml")

    def test_extra_field_rejected(self, tmp_path):
        (tmp_path / "manifest.yaml").write_text(dedent("""\
            schema_version: 1
            name: x
            unknown_field: yes
            prompts: {source: jsonl, path: p.jsonl}
            dimensions: []
        """))
        with pytest.raises(ValidationError, match="Extra inputs"):
            load_manifest(tmp_path / "manifest.yaml")

    def test_duplicate_dim_names_rejected(self, tmp_path):
        (tmp_path / "manifest.yaml").write_text(dedent("""\
            schema_version: 1
            name: x
            prompts: {source: jsonl, path: p.jsonl}
            dimensions:
              - {name: d1, scorer: clip-score, tags: []}
              - {name: d1, scorer: motion-smoothness, tags: []}
        """))
        with pytest.raises(ValidationError, match="duplicate dimension"):
            load_manifest(tmp_path / "manifest.yaml")

    def test_freeform_tags_rejected(self, tmp_path):
        (tmp_path / "manifest.yaml").write_text(dedent("""\
            schema_version: 1
            name: x
            prompts: {source: jsonl, path: p.jsonl}
            dimensions:
              - name: d1
                scorer: clip-score
                tags: [Motion, my-custom-tag]
        """))
        with pytest.raises(ValidationError, match="controlled vocab"):
            load_manifest(tmp_path / "manifest.yaml")

    def test_inline_scorer_spec(self, tmp_path):
        (tmp_path / "manifest.yaml").write_text(dedent("""\
            schema_version: 1
            name: x
            prompts: {source: jsonl, path: p.jsonl}
            dimensions:
              - name: text_alignment
                scorer:
                  ref: vlm_judge
                  prompt_template: foo.txt
                  mode: middle_frame
                  n_frames: 8
                tags: [align.text2video]
        """))
        spec = load_manifest(tmp_path / "manifest.yaml")
        assert spec.dimensions[0].scorer.ref == "vlm_judge"
        assert spec.dimensions[0].scorer.prompt_template == "foo.txt"


# ----------------------------------------------------- prompt loading ---
class TestPromptLoading:
    def test_jsonl_loads(self, manifest_dir):
        b = benchmark_from_manifest(manifest_dir / "manifest.yaml")
        prompts = list(b.list_prompts())
        assert len(prompts) == 2
        assert prompts[0].prompt_id == "p1"
        assert prompts[0].text == "a cat"

    def test_jsonl_dim_filter(self, manifest_dir):
        b = benchmark_from_manifest(manifest_dir / "manifest.yaml")
        # p2 only has dimension "motion"; filter to "visual_quality" → p1 only
        prompts = list(b.list_prompts(dimensions=["visual_quality"]))
        assert len(prompts) == 1
        assert prompts[0].prompt_id == "p1"

    def test_missing_prompts_file_raises(self, tmp_path, minimal_manifest_yaml):
        (tmp_path / "manifest.yaml").write_text(minimal_manifest_yaml)
        b = benchmark_from_manifest(tmp_path / "manifest.yaml")
        with pytest.raises(FileNotFoundError, match="prompts file"):
            list(b.list_prompts())

    def test_malformed_jsonl_raises(self, tmp_path, minimal_manifest_yaml):
        (tmp_path / "manifest.yaml").write_text(minimal_manifest_yaml)
        (tmp_path / "prompts.jsonl").write_text("not json\n")
        b = benchmark_from_manifest(tmp_path / "manifest.yaml")
        with pytest.raises(ValueError, match="not valid JSON"):
            list(b.list_prompts())

    def test_missing_required_field_raises(self, tmp_path, minimal_manifest_yaml):
        (tmp_path / "manifest.yaml").write_text(minimal_manifest_yaml)
        (tmp_path / "prompts.jsonl").write_text('{"prompt_id": "p1"}\n')
        b = benchmark_from_manifest(tmp_path / "manifest.yaml")
        with pytest.raises(ValueError, match="missing prompt_id/text"):
            list(b.list_prompts())


# ----------------------------------------------------- video staging ---
class TestVideoStaging:
    def test_video_layout_expansion(self, manifest_dir):
        from videvalkit.core.layout import WorkspaceLayout
        b = benchmark_from_manifest(manifest_dir / "manifest.yaml")
        prompts = list(b.list_prompts())
        layout = WorkspaceLayout(manifest_dir / "ws")
        videos = b.list_required_videos(prompts, ["model_a"], layout)
        assert len(videos) == 2  # 2 prompts × 1 model × 1 sample
        # First entry: model_a / p1-0.mp4
        assert "model_a" in str(videos[0].path)
        assert "p1-0.mp4" in str(videos[0].path)

    def test_samples_per_prompt(self, manifest_dir):
        from videvalkit.core.layout import WorkspaceLayout
        b = benchmark_from_manifest(manifest_dir / "manifest.yaml")
        prompts = list(b.list_prompts())
        layout = WorkspaceLayout(manifest_dir / "ws")
        videos = b.list_required_videos(
            prompts, ["model_a"], layout, samples_per_prompt=3
        )
        assert len(videos) == 6  # 2 prompts × 3 samples


# ----------------------------------------------------- evaluate stubs ---
class TestEvaluateScaffold:
    def test_evaluate_needs_metrics_module_raises(self, manifest_dir):
        from videvalkit.core.layout import WorkspaceLayout
        b = benchmark_from_manifest(manifest_dir / "manifest.yaml")
        layout = WorkspaceLayout(manifest_dir / "ws")
        # The clip-score / motion-smoothness scorers need SUPPORTED_METRICS
        # which is M3 work. Currently raises NotImplementedError with
        # clear message.
        with pytest.raises(NotImplementedError, match="metrics module|metric registry"):
            b.evaluate(videos=[], layout=layout)

    def test_evaluate_with_judge_required(self, tmp_path):
        """vlm_judge scorer without judge → ValueError."""
        from videvalkit.core.layout import WorkspaceLayout

        (tmp_path / "manifest.yaml").write_text(dedent("""\
            schema_version: 1
            name: x
            needs_judge: true
            prompts: {source: jsonl, path: p.jsonl}
            dimensions:
              - name: text_alignment
                scorer:
                  ref: vlm_judge
                  prompt_template: t.txt
                tags: [align.text2video]
            default_judge: gemma-4-31b-local
        """))
        b = benchmark_from_manifest(tmp_path / "manifest.yaml")
        layout = WorkspaceLayout(tmp_path / "ws")
        with pytest.raises(ValueError, match="vlm_judge"):
            b.evaluate(videos=[], layout=layout, judge=None)


# ----------------------------------------------------- aggregation ---
class TestAggregation:
    def test_aggregate_weighted_sum(self, manifest_dir):
        from videvalkit.core.types import RawResult
        b = benchmark_from_manifest(manifest_dir / "manifest.yaml")
        raw = [
            RawResult(
                benchmark="my_bench", model="m1", dimension="visual_quality",
                prompt_id="p1", score=0.8,
            ),
            RawResult(
                benchmark="my_bench", model="m1", dimension="motion",
                prompt_id="p1", score=0.6,
            ),
        ]
        summary = b.aggregate(raw, aggregator="weighted_sum")
        assert summary.per_dimension["visual_quality"] == 0.8
        assert summary.per_dimension["motion"] == 0.6
        # weighted: 0.8*0.3 + 0.6*0.7 = 0.24 + 0.42 = 0.66
        assert abs(summary.overall - 0.66) < 1e-6
        assert summary.meta["manifest_schema_version"] == 1


# ----------------------------------------------------- public factory ---
class TestFactory:
    def test_factory_returns_manifest_benchmark(self, manifest_dir):
        b = benchmark_from_manifest(manifest_dir / "manifest.yaml")
        assert isinstance(b, ManifestBenchmark)
        assert b.name == "my_bench"
        assert b.env_name == "videvalkit"
        assert b.dimensions == ["visual_quality", "motion"]

    def test_factory_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="manifest not found"):
            benchmark_from_manifest(tmp_path / "nope.yaml")
