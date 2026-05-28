# 指南:扩展

[← 首页](../../index.md)

添加指标、基准或评审——无需 fork。三种插件来源:
内置、pip `entry_points`,或本地 `~/.videvalkit/`。

---

## 添加一个评审

最简单——纯配置。见[评审选择](Judge-Selection.md):

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

## 添加基准 — 路线 A(manifest,简单)

对于“每个 prompt → 一次 scorer 调用 → 一个分数”的基准,写一份 YAML:

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
    scorer: aesthetic-quality        # 注册表中的指标,或 module:Class
    tags: [vq.aesthetic]
  - name: motion
    weight: 0.7
    scorer: motion-smoothness
    tags: [motion.smoothness]
video_layout: "{model}/{prompt_id}-{sample_index}.mp4"
aggregator: weighted_sum
```

`ManifestBenchmark` 依据此文件实现 `BaseBenchmark` 的四个方法。
标签必须来自[受控词表](../reference/Capability-Tags.md)。

---

## 添加基准 — 路线 B(Python,复杂)

对于多阶段 / 暂存 / 子进程流水线,继承 `BaseBenchmark`:

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

验证它被发现:`videvalkit list benchmarks`(以及 `videvalkit doctor`
会显示插件来源)。

---

## 添加一个指标

独立指标继承 `BaseScorer`(逐视频/逐 prompt)或
`BaseDistributionMetric`(FVD 家族),然后注册到 `SUPPORTED_METRICS`,
或通过插件的 `__videvalkit_register__()` 返回 `{"metrics": {...}}`。

必填注册表字段:`kind`、`source`、`needs_judge`、`compute_kind`、
`tags`、`cls`。

**把基准维度抽成独立指标**:包裹基准适配器所用的*同一次*上游调用,
使得 `eval --bench X --dimensions Y` 与 `metric run --name Y` 之间结果
**逐位一致(bit-exact)**。参考 `metrics/vbench_dim.py` 的写法。

---

## 验证

```bash
videvalkit doctor                 # 插件来源、注册表计数
videvalkit list benchmarks        # 你的基准出现
videvalkit metric list            # 你的指标出现
videvalkit capabilities show <tag>  # 你打了标签的指标出现
```

冲突(跨来源同名)以 INFO 级别记录;后来的来源胜出。
`VIDEVALKIT_DISABLE_PLUGINS=1` 忽略所有第三方来源。
