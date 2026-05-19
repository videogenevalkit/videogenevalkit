"""Benchmark adapters.

Each subpackage exposes a single `*Benchmark` class subclassing BaseBenchmark.
Adapters are import-safe in the toolkit core env (no upstream deps imported at
module load time); upstream imports happen lazily inside method bodies and run
inside the benchmark's own conda env via the EnvDispatcher.
"""

from videvalkit.benchmarks.vbench import VBenchBenchmark
from videvalkit.benchmarks.vbench2 import VBench2Benchmark
from videvalkit.benchmarks.videobench import VideoBenchBenchmark
from videvalkit.benchmarks.worldjen import WorldJenBenchmark

__all__ = [
    "VBenchBenchmark",
    "VBench2Benchmark",
    "VideoBenchBenchmark",
    "WorldJenBenchmark",
]
