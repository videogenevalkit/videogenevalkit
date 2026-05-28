# 指南:配置档与快速评测

[← 首页](../../index.md)

配置档用精度换速度。它们控制 prompt 子集、帧采样和每个 prompt 的样本数——
与你使用哪个评审正交。

---

## 三个配置档

| 配置档 | 子集 | 帧数 | 样本数 | 墙钟时间 | 用途 |
|---|---|---|---|---|---|
| `quick` | 小 | 4 | 1 | ~5–10 分钟 | 训练监控、冒烟、CI |
| `standard` | 中 | 8 | 1 | ~30–60 分钟 | 消融、迭代 |
| `full` | 全 | 8 | 5 | 数小时 | 论文 / 榜单(默认) |

```bash
videvalkit eval --bench vbench --profile quick --videos gen/ --workspace ws/
```

不带 `--profile` = `full`(向后兼容)。

---

## 先估算成本

```bash
videvalkit estimate --bench vbench --bench worldjen --profile quick --judge gpt-4o
```

```
Benchmark         Judge?      Wallclock     GPU-h   Judge calls
----------------------------------------------------------------
vbench            —               6.0 min     0.10             0
worldjen          VLM             8.0 min     0.05            60
----------------------------------------------------------------
TOTAL                            14.0 min     0.15            60
```

---

## 一次跑多个基准

```bash
videvalkit eval-suite --all-anchored --profile quick \
  --videos gen/ --workspace ws/

# 或者挑选 + 跳过需评审的基准
videvalkit eval-suite --bench vbench --bench worldjen --no-judge \
  --videos gen/ --workspace ws/
```

---

## 自定义子集

```bash
videvalkit eval --bench vbench --subset my_subset.json --videos gen/ --workspace ws/
```

子集是一份版本锁定的 prompt_id JSON,带校准元数据(相对全集的 Spearman ρ)。
`--subset` 覆盖配置档的默认子集。

---

## 论文忠实 = full + paper 评审

```bash
videvalkit eval --bench t2vcompbench --profile full --judge paper ...
```

如果是训练期监控,请参见[训练监控](Training-Monitor.md)。
