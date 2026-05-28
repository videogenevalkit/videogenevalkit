# 命令行参考

[← 首页](../../index.md)

所有命令都是 `videvalkit` 的子命令。运行 `videvalkit <cmd> --help` 查看完整参数。

---

## 评测

| 命令 | 用途 |
|---|---|
| `eval --bench X --videos V --workspace W` | 跑一个基准 |
| `eval-suite --bench A --bench B ... ` / `--all-anchored` | 把多个基准跑进同一个工作区 |
| `capabilities eval <tag> --videos V` | 跑某能力的所有指标并聚合 |
| `metric run --name M ...` | 跑单个独立指标 |
| `aggregate --workspace W` | 跨基准 z-score 报告 |

### `eval` 关键参数

| 参数 | 含义 |
|---|---|
| `--bench` | 基准名(必填) |
| `--videos` / `--workspace` | 输入目录 / 输出目录(必填) |
| `--models` | 模型名(可重复) |
| `--dimensions` | 维度子集(可重复) |
| `--judge` | `paper` / `default` / `<registry name>` |
| `--judge-endpoint / --judge-model / --judge-kind / --judge-api-key-env` | 临时评审 |
| `--no-judge` | 拒绝需要评审的基准 |
| `--profile` | `quick` / `standard` / `full` |
| `--subset` | 子集 JSON 的路径 |
| `--aggregator` | 覆盖默认聚合器 |

---

## 指标与参考集

| 命令 | 用途 |
|---|---|
| `metric list [--kind --no-judge --source]` | 列出指标,可过滤 |
| `metric show <name>` | 显示某指标的注册表条目 |
| `metric run --name M --gen-videos / --videos / --ref-videos / --refs / --prompts / --judge` | 跑一个指标 |
| `refs list / show <name> / register --name --path` | 管理参考视频集 |
| `fetch-refs --name <ref> / --all [--dry-run]` | 把内置参考集下载到 fetch 缓存 |

### `metric run` 按 kind 区分的输入

| 指标 kind | 必需输入 |
|---|---|
| `distribution_reference` | `--gen-videos` + (`--ref-videos` 或 `--refs`) |
| `per_prompt_reference_free` | `--videos` + `--prompts` |
| `per_video_reference_free` | `--videos` |
| `per_video_with_vlm_judge` | `--videos` + `--judge <name>` |

`--allow-tiny-sample` 绕过分布型指标的小样本量(small-N)守卫。

---

## 能力

| 命令 | 用途 |
|---|---|
| `capabilities list [--show-sub]` | 列出 10 个顶层(及 34 个子级)标签 + 贡献者计数 |
| `capabilities show <tag>` | 显示某标签的贡献者 |
| `capabilities eval <tag> --videos V [--aggregator mean/max/min]` | 跨指标能力分 |

---

## 规划与监控

| 命令 | 用途 |
|---|---|
| `estimate --bench A --bench B --profile P` | 预览墙钟 / GPU-h / 评审调用数 |
| `watch --videos-pattern '...' --bench X --profile quick [--once]` | 轮询检查点目录,评测每个新模型 |

Python 训练循环 API:`videvalkit.training.monitor` — 见[训练监控](../guides/Training-Monitor.md)。

---

## 检查与设置

| 命令 | 用途 |
|---|---|
| `doctor [--json]` | 健康检查:设备、基准、指标、配置档、能力覆盖率、插件、评审 |
| `list benchmarks [--no-judge]` / `list judges` / `list aggregators` | 列出注册表 |
| `fetch-smoke-data` / `fetch-checkpoints` / `fetch-upstream` | 拉取数据 / 权重 / 上游仓库 |
| `prepare-workspace --workspace W --videos V` | 初始化工作区 |
