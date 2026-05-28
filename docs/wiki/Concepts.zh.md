# 核心概念

[← 首页](../index.md)

用五个要点构成心智模型。其余都是细节。

---

## 1. 三个正交维度

videogenevalkit 把*评测什么*与*怎么评测*、*用哪个评审*分离开来。

```
            评测什么              怎么评(成本)         用哪个评审
    ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
    │ --bench   X      │   │ --profile quick  │   │ --judge paper    │
    │ --name    Y      │ × │ --profile standard│ × │ --judge default  │
    │ --capability Z   │   │ --profile full   │   │ --judge <name>   │
    └──────────────────┘   └──────────────────┘   └──────────────────┘
```

它们自由组合。`--profile full --judge paper` 是论文忠实复现路线;
`--profile quick --judge default` 是训练监控路线。

---

## 2. 三个入口(“评测什么”维度)

| 入口 | 粒度 | 输出 | 何时用 |
|---|---|---|---|
| `eval --bench` | 整个基准 | 可与论文对比的逐维度分数 | 对照已发布榜单做汇报 |
| `metric run --name` | 单个指标 | 单个标量 | 你只想要 FVD / CLIP-Score 等 |
| `capabilities eval` | 单项能力 | 跨指标聚合 | “*运动*整体表现如何?” |

从基准中抽出(lift-out)的指标(例如 `motion-smoothness`),无论通过
`--bench vbench --dimensions motion_smoothness` 还是
`metric run --name motion-smoothness` 调用,都是**逐位一致(bit-exact)**的——它们共用一份实现。

---

## 3. 配置档(“怎么评”维度)

| 配置档 | 子集 | 帧数 | 墙钟时间 | 用途 |
|---|---|---|---|---|
| `quick` | 小 | 4 | ~5–10 分钟 | 训练监控、冒烟、CI |
| `standard` | 中 | 8 | ~30–60 分钟 | 消融、迭代 |
| `full` | 全语料 | 8 | 数小时 | 论文 / 榜单(默认) |

`videvalkit estimate` 在运行前预览成本。

---

## 4. 评审(“用哪个”维度)

每个使用评审的基准声明两个槽位:

| 槽位 | 含义 |
|---|---|
| `paper_judge` | 论文所用的 VLM(忠实复现) |
| `default_judge` | 更便宜 / 经验证的替身 |

用 `--judge paper` / `--judge default` / `--judge <registry-name>` 解析,
在 `~/.config/videvalkit/judges.yaml` 添加你自己的,或用 `--judge-endpoint` 临时指定。
没有评审?`--no-judge` 会过滤出无需评审的工作。
见[评审选择](guides/Judge-Selection.md)。

---

## 5. 能力标签

每个指标和基准维度都带有来自固定 44 标签词表(10 个顶层 + 34 个子级)的标签。
这让你能**按能力**而非按基准来评测:

```
motion → motion-smoothness, dynamic-degree, motion-magnitude, ...
          (横跨 vbench、worldscore、独立指标——已去重)
```

见[能力标签](reference/Capability-Tags.md)。

---

## 它们如何拼装到一起

```
videvalkit eval --bench worldjen --judge paper --profile full
        │
        ├─ resolve_judge()  → paper_judge → 具体评审配置
        ├─ resolve_profile()→ 子集 + 帧采样
        ├─ plugin discover  → 基准适配器
        └─ scheduler        → adapter.evaluate() → aggregate() → 汇总
```

完整分层见[架构](Architecture.md)。
