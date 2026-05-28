# 扩展性 — 评审与集成

> 合并原 JUDGE_SELECTION 与 INTEGRATION_FRAMEWORK 两份设计文档:两者都回答
> “如何不 fork 就加东西?”。可操作的用法见
> [评审选择](../wiki/guides/Judge-Selection.md)与[扩展](../wiki/guides/Extending.md);
> 本文是理念。

---

## 第 A 部分 — 评审切换

### A.1 问题

一个基准发布的数字取决于*由哪个* VLM 评审。v0.0.1 给每个基准硬编一个评审,
导致(a)忠实复现与(b)便宜迭代互斥,而偷换模型会破坏可比性。

### A.2 三个声明档

每个用评审的基准在注册表条目里声明两个槽位,CLI 再暴露第三条路径:

| 选择子 | 解析到 | 用于 |
|---|---|---|
| `--judge paper` | 基准的 `paper_judge` | 忠实复现 |
| `--judge default` | 基准的 `default_judge` | 便宜 / 已验证的迭代 |
| `--judge <name>` | 任意 `SUPPORTED_JUDGES` 条目 | 某个特定模型 |

`resolve_judge(benchmark, judge_name, judge_override)` 把语义关键字映射到具体
配置。对无需评审的基准用 `--judge paper` 会快速失败——基准绝不偷偷降级。

### A.3 自带评审

两条路径,无需 fork:

- **持久** — `~/.config/videvalkit/judges.yaml` 添加具名条目
  (`kind: openai_compatible | gemini | anthropic`、endpoint、model、`api_key_env`)。
- **临时** — `--judge-endpoint / --judge-model / --judge-kind / --judge-api-key-env`
  一次性使用,与 `--judge` 互斥。

配置优先级(后者覆盖前者,顶层键替换):
`builtin → ~/.config → $CWD/.videvalkit → env → CLI`。

### A.4 无需评审路径

基准和指标上的 `needs_judge` 驱动 `--no-judge`,过滤出完全不需 VLM/LLM 即可运行的
工作(4 个基准、17 个指标)。评审后端由 `build_judge(cfg)` 构建一次并在整次运行中
复用;每次调用都镜像到 workspace API 日志以便离线回放。

---

## 第 B 部分 — 集成框架

### B.1 两条轨道(80/20)

| 轨道 | 何时 | 怎么做 |
|---|---|---|
| **A — Manifest** | “prompt → scorer → score”(约 80%) | 一份 `manifest.yaml`(`schema_version: 1`,≤ 12 个顶层字段),由 `ManifestBenchmark` 实现 |
| **B — Python 适配器** | 暂存 / 多阶段 / 子进程(约 20%) | 一个 `BaseBenchmark` 子类(4 方法) |

两者注册方式相同,最终汇聚到同一 runner / workspace / scheduler——runner 看不出
区别。这是刻意的划分:强迫所有人用 YAML 会让复杂场景痛苦;强迫所有人写 Python
会抬高简单场景的门槛。

### B.2 三层插件发现

优先级从低到高:

1. **builtin** — `src/videvalkit/{benchmarks,metrics}/`
2. **pip entry_points** — `[project.entry-points."videvalkit.benchmarks"]`
3. **本地目录** — `~/.videvalkit/<group>/` 然后 `$CWD/.videvalkit/<group>/`

本地插件用 `__videvalkit_register__()` 约定,返回 `{"benchmarks": {...}}` /
`{"metrics": {...}}`。同名冲突以 INFO 记录(后来的来源胜出);
`VIDEVALKIT_DISABLE_PLUGINS=1` 忽略所有第三方来源。`doctor` 打印每条的解析来源
以供溯源。

### B.3 添加指标

继承 `BaseScorer`(逐视频/逐 prompt)或 `BaseDistributionMetric`(FVD 家族),
或——对某个基准维度——写一个**逐位一致 lift**,包裹同一次上游调用
(见[指标](VIDEO_METRICS_DESIGN.md))。在 `SUPPORTED_METRICS` 中以必填字段注册
(`kind`、`source`、`needs_judge`、`compute_kind`、`tags`、`cls`)。

### B.4 它*不是*什么

没有插件 manifest DSL,没有动态发现魔法——只有三个惰性合并源喂给一个注册表。
脚手架(`videvalkit new`)与契约校验器(`videvalkit validate`)曾被考虑并后移到
v0.3;注册表 + manifest 已让集成成为亚小时级任务。
