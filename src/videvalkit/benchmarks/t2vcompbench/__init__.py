from videvalkit.benchmarks.t2vcompbench.benchmark import T2VCompBenchBenchmark
from videvalkit.benchmarks.t2vcompbench.scorers import (
    DIMENSION_PROMPT_TEMPLATES,
    T2VCOMPBENCH_CV_DIMS,
    T2VCOMPBENCH_MLLM_DIMS,
)

__all__ = [
    "T2VCompBenchBenchmark",
    "T2VCOMPBENCH_MLLM_DIMS",
    "T2VCOMPBENCH_CV_DIMS",
    "DIMENSION_PROMPT_TEMPLATES",
]
