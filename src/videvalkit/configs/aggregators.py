"""Aggregator registry."""

from videvalkit.aggregators.weighted_sum import WeightedSumAggregator
from videvalkit.aggregators.phas import PHASAggregator
from videvalkit.aggregators.bt import BradleyTerryAggregator


SUPPORTED_AGGREGATORS = {
    "weighted_sum":     dict(cls=WeightedSumAggregator, kwargs={}),
    # VBench v1's official weighting (constants land in M3)
    "vbench_weighted":  dict(cls=WeightedSumAggregator, kwargs={}),
    # VBench-2.0 categorical mean (lands in M4)
    "vbench2_category": dict(cls=WeightedSumAggregator, kwargs={}),
    "phas":             dict(cls=PHASAggregator, kwargs={}),
    "bt":               dict(cls=BradleyTerryAggregator, kwargs={"bootstrap": 1000}),
}
