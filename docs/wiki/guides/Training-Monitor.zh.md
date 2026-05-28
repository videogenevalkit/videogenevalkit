# 指南:训练监控

[← 首页](../../index.md)

每 N 个训练步跑一次快速评测以追踪质量趋势——无需 shell 外调,也不阻塞训练循环。

---

## CLI:监视一个检查点目录

```bash
videvalkit watch \
  --videos-pattern '/runs/r42/checkpoints/step_*/samples' \
  --bench vbench --profile quick \
  --workspace /runs/r42/eval
```

轮询该 glob,在每个新检查点出现时评测其视频,并向
`<workspace>/timeline.jsonl` 每个检查点追加一行。`--once`
处理当前匹配后退出(不轮询)。

---

## Python API:在训练循环内部

```python
from videvalkit.training import monitor, MonitorConfig

cfg = MonitorConfig(
    benches=["vbench"],
    metrics=["fvd", "motion-smoothness"],
    profile="quick",
    workspace="/runs/r42/eval",
)

for step in range(0, 100_000, 1000):
    train_step()
    if step % 5000 == 0:
        prompts = monitor.preview_prompts(cfg)        # 需要生成哪些 prompt
        videos  = generate_videos(prompts)            # 你的模型
        result  = monitor.eval(videos, model_name=f"step_{step}", cfg=cfg, step=step)
        tb.add_scalar("eval/overall", result.overall, step)
        for bench, summary in result.summary.items():
            ...                                        # 逐基准细节
```

| 方法 | 用途 |
|---|---|
| `monitor.preview_prompts(cfg)` | 所配置基准+配置档需要的 prompt 集 |
| `monitor.eval(videos, model_name, cfg)` | 跑基准/指标,追加到时间线,返回 `MonitorResult` |
| `MonitorConfig.save / load` | 把监控配置与运行一起持久化 |

---

## 记录了什么

`<workspace>/timeline.jsonl` — 每个检查点一行 JSON:

```json
{"model_name": "step_5000", "step": 5000, "profile": "quick",
 "overall": 0.71, "bench_overalls": {"vbench": {...}}}
```

跨基准的 `overall` 是各基准 overall 的均值(用于趋势监控的原始均值;
z-score 归一化是后续改进)。

---

## 监控中的 FVD

未放置论文 I3D 权重时,FVD 默认使用 **S3D-K400** 骨干(自动下载)——
一个有效的 Kinetics-400 趋势指标。非常适合监控;若要论文数字,
请用 `--backbone i3d-k400` 配合权重。见[指标](../reference/Metrics.md)。

---

## 说明

- 没有 PyTorch/Lightning 回调——你显式调用 `monitor.eval()`。
- 用 `--profile quick` 获得稳定、快速的趋势读数。
- watch 使用轮询(非 inotify);选一个与你检查点节奏匹配的间隔。
