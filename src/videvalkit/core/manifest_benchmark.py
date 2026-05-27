"""ManifestBenchmark — declarative YAML-driven benchmark adapter.

Per docs/INTEGRATION_FRAMEWORK_DESIGN.md §3.2 (user 2026-05-20 confirmed,
重集成轻框架):

A user can declare a new benchmark in a YAML file rather than writing a Python
adapter. The runtime ``ManifestBenchmark`` class implements the 4 abstract
methods of ``BaseBenchmark`` by reading the manifest.

Manifest schema [v1, see CAPABILITY_TAGS_DESIGN style versioning]:

    name: my_bench
    version: 0.1.0
    schema_version: 1
    description: My custom benchmark
    env: videvalkit
    needs_gpu: true
    needs_judge: true

    prompts:
      source: jsonl
      path: prompts.jsonl              # relative to manifest.yaml

    dimensions:
      - name: visual_quality
        weight: 0.3
        scorer: clip-score              # registry name or "<module>:<Class>"
        tags: [vq.aesthetic]
      - name: text_alignment
        weight: 0.4
        scorer:
          ref: vlm_judge
          prompt_template: prompts/text_alignment.txt
          mode: middle_frame
          n_frames: 8
        tags: [align.text2video]

    video_layout: "{model}/{prompt_id}-{sample_index}.mp4"

    aggregator: weighted_sum

    default_judge: gemma-4-31b-local
    paper_judge: paper-llava-1.6-34b
    recommended_judges:
      - gemma-4-31b-local
      - claude-sonnet-4-6

Schemas with ``schema_version != 1`` are rejected — manifest authors must
bump the schema_version field in lockstep with toolkit version (future
work in v0.3+).

Track B Python adapters [src/videvalkit/benchmarks/<name>/benchmark.py]
remain the path for complex multi-stage pipelines; this module exists
for the 80% simple case where each prompt → one scorer call → one score.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from videvalkit.core.benchmark import BaseBenchmark
from videvalkit.core.layout import WorkspaceLayout
from videvalkit.core.types import PromptItem, RawResult, Summary, VideoSpec

MANIFEST_SCHEMA_VERSION = 1


# ----------------------------------------------------- manifest schema ---
class PromptSpec(BaseModel):
    """How to load the prompts for the benchmark."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["jsonl", "hf_dataset"]
    path: str | None = None            # required if source=jsonl
    repo: str | None = None            # required if source=hf_dataset
    split: str = "test"


class ScorerRefDetailed(BaseModel):
    """Inline scorer spec for vlm_judge / advanced configurations."""

    model_config = ConfigDict(extra="allow")  # allow forward-compat kwargs

    ref: Literal["vlm_judge", "metric"] = "metric"
    name: str | None = None             # required if ref=metric
    prompt_template: str | None = None  # required if ref=vlm_judge
    mode: str | None = None             # frame sampling mode
    n_frames: int | None = None


class DimensionSpec(BaseModel):
    """One scoring dimension within a manifest benchmark."""

    model_config = ConfigDict(extra="forbid")

    name: str
    weight: float = 1.0
    scorer: str | ScorerRefDetailed     # registry name OR inline spec
    tags: list[str] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tags")
    @classmethod
    def _validate_tags(cls, v: list[str]) -> list[str]:
        """Reject free-form tags — must come from controlled vocab."""
        from videvalkit.configs.capability_taxonomy import ALL_TAGS
        bad = [t for t in v if t not in ALL_TAGS]
        if bad:
            raise ValueError(
                f"tags {bad} not in controlled vocab. "
                f"See docs/CAPABILITY_TAGS_DESIGN.md §3 for the 44-tag vocab."
            )
        return v


class ManifestSpec(BaseModel):
    """Full manifest schema. Loaded from manifest.yaml."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    name: str
    version: str = "0.1.0"
    description: str = ""
    env: str = "videvalkit"
    needs_gpu: bool = False
    needs_judge: bool = False
    prompts: PromptSpec
    dimensions: list[DimensionSpec]
    video_layout: str = "{model}/{prompt_id}-{sample_index}.mp4"
    aggregator: str = "weighted_sum"
    default_judge: str | None = None
    paper_judge: str | None = None
    recommended_judges: list[str] = Field(default_factory=list)

    @field_validator("schema_version")
    @classmethod
    def _check_schema_version(cls, v: int) -> int:
        if v != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"manifest schema_version={v} not supported by this toolkit "
                f"[expected {MANIFEST_SCHEMA_VERSION}]. Upgrade the toolkit "
                f"or edit the manifest to match."
            )
        return v

    @field_validator("dimensions")
    @classmethod
    def _check_unique_dim_names(cls, v: list[DimensionSpec]) -> list[DimensionSpec]:
        names = [d.name for d in v]
        if len(set(names)) != len(names):
            dupes = [n for n in names if names.count(n) > 1]
            raise ValueError(f"duplicate dimension names: {sorted(set(dupes))}")
        return v


def load_manifest(path: str | Path) -> ManifestSpec:
    """Load and validate a manifest.yaml. Raises ValidationError on bad schema."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"manifest not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return ManifestSpec(**raw)


# ----------------------------------------------------- runtime adapter ---
class ManifestBenchmark(BaseBenchmark):
    """Runtime BaseBenchmark implementation backed by a manifest.yaml.

    Implements the 4 abstract methods by reading the manifest:

    - ``list_prompts()`` ← manifest.prompts [jsonl or HF dataset]
    - ``list_required_videos()`` ← manifest.video_layout template expansion
    - ``evaluate()`` ← per-dim scorer dispatch [registry lookup or inline spec]
    - ``aggregate()`` ← manifest.aggregator + per-dim weights

    Track B [Python BaseBenchmark subclass] is still the path for complex
    benchmarks needing custom staging / multi-stage pipelines.
    """

    name = "manifest_runtime"           # overridden by manifest.name at __init__
    env_name = "videvalkit"

    def __init__(self, manifest: ManifestSpec, manifest_dir: Path | None = None):
        self._manifest = manifest
        self._manifest_dir = manifest_dir or Path.cwd()
        # Adapt class-level attrs per-instance from manifest
        self.name = manifest.name
        self.env_name = manifest.env
        self.dimensions = [d.name for d in manifest.dimensions]
        self.video_layout = manifest.video_layout

    @property
    def manifest(self) -> ManifestSpec:
        return self._manifest

    # -- BaseBenchmark abstract impls -----------------------------------------
    def list_prompts(
        self, dimensions: list[str] | None = None
    ) -> Iterator[PromptItem]:
        """Yield prompts loaded from manifest.prompts.

        For ``source: jsonl``, expects each line to be a JSON object with at
        least ``prompt_id`` and ``text``; optional ``dimensions`` and
        ``meta``.

        For ``source: hf_dataset``, defers to HF datasets library [lazy import].
        """
        prompts_spec = self._manifest.prompts
        if prompts_spec.source == "jsonl":
            yield from self._load_jsonl_prompts(prompts_spec, dimensions)
        elif prompts_spec.source == "hf_dataset":
            yield from self._load_hf_dataset_prompts(prompts_spec, dimensions)
        else:
            raise ValueError(f"unknown prompts source {prompts_spec.source!r}")

    def _load_jsonl_prompts(
        self, spec: PromptSpec, dimensions: list[str] | None,
    ) -> Iterator[PromptItem]:
        import json

        if spec.path is None:
            raise ValueError("prompts.source=jsonl requires prompts.path")
        path = self._manifest_dir / spec.path
        if not path.is_file():
            raise FileNotFoundError(f"prompts file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"prompts.jsonl line {line_no} not valid JSON: {e}"
                    )
                if "prompt_id" not in rec or "text" not in rec:
                    raise ValueError(
                        f"prompts.jsonl line {line_no} missing prompt_id/text"
                    )
                prompt_dims = rec.get("dimensions", [])
                if dimensions is not None:
                    if not any(d in prompt_dims for d in dimensions):
                        continue
                yield PromptItem(
                    prompt_id=rec["prompt_id"],
                    text=rec["text"],
                    dimensions=prompt_dims,
                    meta=rec.get("meta", {}),
                )

    def _load_hf_dataset_prompts(
        self, spec: PromptSpec, dimensions: list[str] | None,
    ) -> Iterator[PromptItem]:
        raise NotImplementedError(
            "prompts.source=hf_dataset not yet implemented; "
            "use source=jsonl for v0.2 [INTEGRATION_FRAMEWORK_DESIGN §11]"
        )

    def list_required_videos(
        self,
        prompts: list[PromptItem],
        models: list[str],
        layout: WorkspaceLayout,
        samples_per_prompt: int = 1,
    ) -> list[VideoSpec]:
        """Expand video_layout template against (model, prompt, sample) tuples."""
        videos: list[VideoSpec] = []
        for model in models:
            for prompt in prompts:
                for sample_idx in range(samples_per_prompt):
                    rel = self._manifest.video_layout.format(
                        model=model,
                        prompt_id=prompt.prompt_id,
                        sample_index=sample_idx,
                    )
                    full = layout.videos_dir / rel
                    videos.append(VideoSpec(
                        path=full,
                        prompt_id=prompt.prompt_id,
                        model_name=model,
                        dimension=None,
                        sample_index=sample_idx,
                    ))
        return videos

    def evaluate(
        self,
        videos: list[VideoSpec],
        layout: WorkspaceLayout,
        dimensions: list[str] | None = None,
        judge: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[RawResult]:
        """Dispatch each (video, dim) → scorer.

        Scorer dispatch:
          - dim.scorer is a string registry name → look up in SUPPORTED_METRICS
            [or treat as inline Python ``"module:Class"`` path]
          - dim.scorer is a ScorerRefDetailed with ref=vlm_judge → use judge cfg
          - dim.scorer with ref=metric needs ``name`` field

        v0.2 ships the dispatch *skeleton*; actual scorer invocation requires
        SUPPORTED_METRICS [M3 in flight]. For now, raises NotImplementedError
        with a clear message pointing at the missing piece.
        """
        # Validate dimensions filter
        dim_specs = self._manifest.dimensions
        if dimensions:
            dim_specs = [d for d in dim_specs if d.name in dimensions]
        if not dim_specs:
            return []

        # Detect what we actually need from each scorer
        needs_metric_registry = any(
            isinstance(d.scorer, str) for d in dim_specs
        )
        needs_judge = any(
            isinstance(d.scorer, ScorerRefDetailed) and d.scorer.ref == "vlm_judge"
            for d in dim_specs
        )

        if needs_metric_registry:
            try:
                from videvalkit.configs.metrics import SUPPORTED_METRICS  # noqa
            except ImportError:
                raise NotImplementedError(
                    f"manifest benchmark {self.name!r} references metric scorers "
                    f"but the metrics module is not yet implemented "
                    f"[v0.2 M3 in progress, see "
                    f"docs/INTEGRATION_FRAMEWORK_DESIGN.md §5]"
                )

        if needs_judge and judge is None:
            raise ValueError(
                f"manifest benchmark {self.name!r} declares a vlm_judge scorer "
                f"but no judge was passed. Use --judge <name> on the CLI or "
                f"set default_judge in the manifest."
            )

        # v0.2 placeholder: actual dispatch lands when SUPPORTED_METRICS lands.
        raise NotImplementedError(
            f"ManifestBenchmark.evaluate() dispatch awaits v0.2 M3 metric "
            f"registry. Scaffolding in place [manifest validated, prompts "
            f"loaded, videos staged]; dispatch will plug into SUPPORTED_METRICS "
            f"once it lands."
        )

    def aggregate(
        self, raw: list[RawResult], aggregator: str = "weighted_sum", **kwargs: Any,
    ) -> Summary:
        """Aggregate raw results using manifest.aggregator + per-dim weights."""
        from collections import defaultdict

        # Group by dim
        by_dim: dict[str, list[float]] = defaultdict(list)
        for r in raw:
            score = r.score if isinstance(r.score, (int, float)) else 0.0
            by_dim[r.dimension].append(float(score))

        per_dimension: dict[str, float] = {
            dim: sum(scores) / len(scores) for dim, scores in by_dim.items() if scores
        }

        # Weighted overall
        weights_by_dim = {d.name: d.weight for d in self._manifest.dimensions}
        if aggregator == "weighted_sum":
            total_w = sum(
                weights_by_dim.get(d, 1.0) for d in per_dimension
            )
            if total_w > 0:
                overall = sum(
                    score * weights_by_dim.get(d, 1.0)
                    for d, score in per_dimension.items()
                ) / total_w
            else:
                overall = 0.0
        else:
            # Other aggregators delegated to SUPPORTED_AGGREGATORS
            from videvalkit.configs.aggregators import SUPPORTED_AGGREGATORS
            if aggregator not in SUPPORTED_AGGREGATORS:
                raise KeyError(
                    f"manifest aggregator {aggregator!r} not in registry"
                )
            # v0.2: simple mean fallback for non-weighted_sum;
            # full aggregator dispatch is a M3 follow-up
            overall = (
                sum(per_dimension.values()) / len(per_dimension)
                if per_dimension else 0.0
            )

        if raw:
            model = raw[0].model
            n_videos = len({r.video_path for r in raw if r.video_path})
            n_prompts = len({r.prompt_id for r in raw})
        else:
            model = "unknown"
            n_videos = n_prompts = 0

        return Summary(
            benchmark=self.name,
            model=model,
            per_dimension=per_dimension,
            overall=overall,
            n_videos=n_videos,
            n_prompts=n_prompts,
            aggregator=aggregator,
            meta={
                "manifest_version": self._manifest.version,
                "manifest_schema_version": self._manifest.schema_version,
            },
        )


# ----------------------------------------------------- factory ---
def benchmark_from_manifest(
    manifest_path: str | Path,
) -> ManifestBenchmark:
    """Public factory: ``manifest_path`` → ``ManifestBenchmark`` instance."""
    path = Path(manifest_path)
    spec = load_manifest(path)
    return ManifestBenchmark(spec, manifest_dir=path.parent)
