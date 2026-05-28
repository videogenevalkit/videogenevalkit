# Guide: Extending

[← Home](../../index.md)

Add a metric, benchmark, or judge — without forking. Three plugin sources:
built-in, pip `entry_points`, or local `~/.videvalkit/`.

---

## Add a judge

Easiest of all — just config. See [Judge Selection](Judge-Selection.md):

```yaml
# ~/.config/videvalkit/judges.yaml
judges:
  my-judge:
    kind: openai_compatible
    endpoint: http://host:8003/v1
    model: my/model
    api_key_env: MY_KEY
```

---

## Add a benchmark — Track A (manifest, simple)

For "each prompt → one scorer call → one score" benchmarks, write a YAML:

```yaml
# ~/.videvalkit/benchmarks/my_bench/manifest.yaml
schema_version: 1
name: my_bench
env: videvalkit
needs_judge: false
prompts:
  source: jsonl
  path: prompts.jsonl
dimensions:
  - name: visual_quality
    weight: 0.3
    scorer: aesthetic-quality        # a registry metric, or module:Class
    tags: [vq.aesthetic]
  - name: motion
    weight: 0.7
    scorer: motion-smoothness
    tags: [motion.smoothness]
video_layout: "{model}/{prompt_id}-{sample_index}.mp4"
aggregator: weighted_sum
```

`ManifestBenchmark` implements the four `BaseBenchmark` methods from this file.
Tags must come from the [controlled vocab](../reference/Capability-Tags.md).

---

## Add a benchmark — Track B (Python, complex)

For multi-stage / staging / subprocess pipelines, subclass `BaseBenchmark`:

```python
# ~/.videvalkit/benchmarks/my_bench/benchmark.py
from videvalkit.core.benchmark import BaseBenchmark

class MyBench(BaseBenchmark):
    name = "my_bench"
    dimensions = [...]
    def list_prompts(self, dimensions=None): ...
    def list_required_videos(self, prompts, models, layout, samples_per_prompt=1): ...
    def evaluate(self, videos, layout, dimensions=None, judge=None, **kw): ...
    def aggregate(self, raw, aggregator="weighted_sum", **kw): ...

def __videvalkit_register__():
    return {"benchmarks": {"my_bench": {"cls": MyBench, "env": "videvalkit",
                                        "needs_judge": False}}}
```

Verify it's discovered: `videvalkit list benchmarks` (and `videvalkit doctor`
shows plugin sources).

---

## Add a metric

A standalone metric subclasses `BaseScorer` (per-video/per-prompt) or
`BaseDistributionMetric` (FVD-family), then registers in `SUPPORTED_METRICS`
or via a plugin's `__videvalkit_register__()` returning `{"metrics": {...}}`.

Required registry fields: `kind`, `source`, `needs_judge`, `compute_kind`,
`tags`, `cls`.

**Lifting a benchmark dim into a standalone metric**: wrap the *same* upstream
call the bench adapter uses, so results are **bit-exact** between
`eval --bench X --dimensions Y` and `metric run --name Y`. See
`metrics/vbench_dim.py` for the pattern.

---

## Verify

```bash
videvalkit doctor                 # plugin sources, registry counts
videvalkit list benchmarks        # your bench appears
videvalkit metric list            # your metric appears
videvalkit capabilities show <tag>  # your tagged metric appears
```

Conflicts (same name across sources) are logged at INFO; later source wins.
`VIDEVALKIT_DISABLE_PLUGINS=1` ignores all third-party sources.
