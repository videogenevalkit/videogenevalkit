"""Benchmark registry."""

from videvalkit.benchmarks.physics_iq import PhysicsIQBenchmark
from videvalkit.benchmarks.semantics_axis import SemanticsAxisBenchmark
from videvalkit.benchmarks.t2vcompbench import T2VCompBenchBenchmark
from videvalkit.benchmarks.v_reasonbench import VReasonBenchBenchmark
from videvalkit.benchmarks.vbench import VBenchBenchmark
from videvalkit.benchmarks.vbench2 import VBench2Benchmark
from videvalkit.benchmarks.vbench_pp import VBenchPPBenchmark
from videvalkit.benchmarks.videobench import VideoBenchBenchmark
from videvalkit.benchmarks.worldjen import WorldJenBenchmark
from videvalkit.benchmarks.worldscore import WorldScoreBenchmark


# All four benchmarks share one conda env (`videvalkit`). Upstream's
# version pins are overrideable in practice; see envs/videvalkit.yaml.
_SHARED_ENV = "videvalkit"

SUPPORTED_BENCHMARKS = {
    "vbench": dict(
        cls=VBenchBenchmark,
        env=_SHARED_ENV,
        needs_gpu=True,
        needs_judge=False,
        default_aggregator="vbench_weighted",
    ),
    "vbench2": dict(
        cls=VBench2Benchmark,
        env=_SHARED_ENV,
        needs_gpu=True,
        needs_judge=True,
        default_judge="local-llava-video-7b",
        default_aggregator="vbench2_category",
    ),
    "videobench": dict(
        cls=VideoBenchBenchmark,
        env=_SHARED_ENV,
        needs_gpu=False,
        needs_judge=True,
        default_judge="gpt-4o",
        default_aggregator="weighted_sum",
    ),
    "worldjen": dict(
        cls=WorldJenBenchmark,
        env=_SHARED_ENV,
        needs_gpu=False,
        needs_judge=True,
        default_judge="gemma-4-31b-local",
        default_aggregator="phas",
    ),
    "t2vcompbench": dict(
        cls=T2VCompBenchBenchmark,
        env=_SHARED_ENV,
        needs_gpu=True,                # GroundingDINO / Depth / SAM
        needs_judge=True,              # LLaVA-Video for 4 of 7 dims
        default_judge="local-llava-video-7b",
        default_aggregator="weighted_sum",
    ),
    "physics_iq": dict(
        cls=PhysicsIQBenchmark,
        env=_SHARED_ENV,
        needs_gpu=False,               # pixel-level CV; no model weights
        needs_judge=False,             # no VLM judge in the headline metric
        default_aggregator="weighted_sum",
    ),
    "vbench_pp": dict(
        cls=VBenchPPBenchmark,
        env=_SHARED_ENV,               # shares VBench v1 env
        needs_gpu=True,
        needs_judge=True,              # I2V + Trustworthiness use a VLM
        default_judge="local-llava-video-7b",
        default_aggregator="vbench_weighted",
    ),
    "v_reasonbench": dict(
        cls=VReasonBenchBenchmark,
        env=_SHARED_ENV,
        needs_gpu=False,               # deterministic per-task verifiers
        needs_judge=False,
        default_aggregator="weighted_sum",
    ),
    "worldscore": dict(
        cls=WorldScoreBenchmark,
        env=_SHARED_ENV,
        needs_gpu=True,                # VGG16 + pyiqa CLIP run on CUDA when available
        needs_judge=False,             # pure CV / pyiqa pipeline
        default_aggregator="weighted_sum",
    ),
    "semantics_axis": dict(
        cls=SemanticsAxisBenchmark,
        env=_SHARED_ENV,
        needs_gpu=False,               # VLM-judge only, no local CV checkpoints
        needs_judge=True,
        default_judge="gemma-4-31b-local",
        default_aggregator="weighted_sum",
    ),
}


# Public superset relationships, surfaced by `videvalkit list benchmarks`.
# Use when one benchmark's dimension set is a strict superset of another's
# and shares the v1 implementation.
BENCHMARK_RELATIONS: dict[str, dict[str, str]] = {
    "vbench_pp": {"superset_of": "vbench"},
}
