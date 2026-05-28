# 架构

[← 首页](../index.md)

---

## 分层设计

```
┌────────────────────────────────────────────────────────────────────┐
│ L7  CLI / Python API                                                 │
│     eval · eval-suite · metric · capabilities · refs · estimate ·    │
│     watch · doctor · list                                            │
├────────────────────────────────────────────────────────────────────┤
│ L6  编排(Orchestration)                                            │
│     runner.run · resolve_judge · resolve_profile · plugin.discover · │
│     scheduler (env / GPU / HTTP)                                     │
├────────────────────────────────────────────────────────────────────┤
│ L5  注册表(惰性合并:内置 + 用户 + 插件)                         │
│     SUPPORTED_BENCHMARKS · SUPPORTED_JUDGES · SUPPORTED_METRICS ·    │
│     SUPPORTED_AGGREGATORS · capability taxonomy                      │
├────────────────────────────────────────────────────────────────────┤
│ L4  插件发现                                                         │
│     builtin → pip entry_points → ~/.videvalkit/ → $CWD/.videvalkit/  │
├────────────────────────────────────────────────────────────────────┤
│ L3  核心抽象                                                         │
│     BaseBenchmark · BaseScorer · BaseDistributionMetric ·            │
│     BaseAggregator · ManifestBenchmark · Profile · Subset · Capability│
├────────────────────────────────────────────────────────────────────┤
│ L2  适配器与指标                                                     │
│     10 个基准适配器 · 20 个指标 · 3 个评审后端 ·                     │
│     共享骨干网络 (S3D / I3D / InceptionV3 / CLIP)                    │
├────────────────────────────────────────────────────────────────────┤
│ L1  基础设施                                                         │
│     Workspace · ApiCallLogger · FrameCache · frechet/mmd utils       │
├────────────────────────────────────────────────────────────────────┤
│ L0  外部                                                             │
│     上游论文仓库 · HF (checkpoints / refs) · VLM 端点                │
└────────────────────────────────────────────────────────────────────┘
```

每一层只依赖其下一层。新增一个指标/基准/评审只触及
L2(实现)+ L5(一行注册表)——别无其他。

---

## 四个注册表

| 注册表 | 存放 | 合并来源 |
|---|---|---|
| `SUPPORTED_BENCHMARKS` | 基准适配器 + dim_tags + 评审槽位 | 内置 + 插件 |
| `SUPPORTED_JUDGES` | 评审配置 | 内置 + `judges.yaml` |
| `SUPPORTED_METRICS` | 指标规格(kind/source/tags/backbone) | 内置 + 插件 |
| `SUPPORTED_AGGREGATORS` | 跨 prompt 聚合器 | 内置 |

全部在**导入时惰性合并**:内置条目加上从用户配置 / 插件发现的一切。
同名 → 后来的来源胜出(以 INFO 级别记录)。

---

## 插件模型

三层,优先级从低到高:

1. **内置** — `src/videvalkit/{benchmarks,metrics}/`
2. **pip entry_points** — `[project.entry-points."videvalkit.benchmarks"]`
3. **本地目录** — `~/.videvalkit/<group>/` 然后 `$CWD/.videvalkit/<group>/`

本地插件使用 `__videvalkit_register__()` 约定。用
`VIDEVALKIT_DISABLE_PLUGINS=1` 禁用所有第三方来源。

---

## 两条基准集成路线

| 路线 | 何时 | 怎么做 |
|---|---|---|
| **A — Manifest** | 简单:prompt → scorer → score | 一个 `manifest.yaml`([扩展](guides/Extending.md)) |
| **B — Python 适配器** | 复杂:暂存 / 多阶段 / 子进程 | `BaseBenchmark` 子类 |

两者最终汇聚到同一个 runner / workspace / scheduler。

---

## 共享指标基础设施

| 模块 | 用途 |
|---|---|
| `metrics/utils/frechet.py` | Fréchet 距离 (FVD / VFID / CLIP-FVD),float64 |
| `metrics/utils/mmd.py` | 多项式核 MMD²(KVD) |
| `metrics/backbones/s3d_k400.py` | S3D Kinetics-400 视频特征 |
| `metrics/backbones/i3d_k400.py` | I3D-K400 torchscript 加载器(论文版 FVD) |
| `metrics/backbones/clip_vit.py` | CLIP-ViT 帧特征 |

---

## 设计存档

原始设计文档(理念、权衡、决策快照)位于
[`docs/design/`](../design/PRODUCT_DESIGN.md):PRODUCT、JUDGE_SELECTION、INTEGRATION_FRAMEWORK、
VIDEO_METRICS、QUICK_EVAL、CAPABILITY_TAGS、REVIEW_PROTOCOL、NPU_ADAPTATION。
wiki 是可操作的参考;设计文档解释*为什么*。
