# Fast Benchmark & Scorer Integration — Design Doc

| 字段 | 内容 |
|---|---|
| 版本 | v0.1 (draft) |
| 状态 | Design — pending review |
| 创建 | 2026-05-20 |
| 影响范围 | `core/`, `configs/`, `cli.py`, 新增 `plugins/`, `templates/`, `metrics/` 模块 |
| 关联文档 | [`JUDGE_SELECTION_DESIGN.md`](JUDGE_SELECTION_DESIGN.md)（judge 切换）—— 本文聚焦 benchmark/scorer 的快速接入 |
| 目标读者 | benchmark 集成者 · 算法贡献者 · 内部下游评测团队 |

---

## 1. 背景与问题陈述

### 1.1 当前接入新 benchmark 的实际成本

现有接入路径（以 worldjen / t2vcompbench 为参考）：

1. 在 `src/videvalkit/benchmarks/<name>/` 新建目录
2. 写 `benchmark.py` 实现 `BaseBenchmark` 4 个抽象方法（`list_prompts / list_required_videos / evaluate / aggregate`）—— 现有 6 个 adapter 文件长度 210–752 行
3. 在 `configs/benchmarks.py` 加 entry
4. 写 prompts 文件（jsonl）放到 `benchmarks/<name>/prompts/`
5. 写 smoke test、集成 test
6. 改 `docs/USER_MANUAL_*.md` 与 `docs/TEST_MANUAL.md`
7. 必要时改 `envs/videvalkit.yaml` 或 `post_install.sh`

**典型工作量**：1–3 人天（不含算法本身实现），其中至少 50% 是模板化的 IO / staging / aggregation boilerplate。

### 1.2 当前接入新 scoring 算法（metric / scorer）的实际成本

更尴尬：

- 算法如果属于某个 benchmark 的某个 dim → 写进那个 benchmark 的 `evaluate()`，与 benchmark 强耦合
- 算法如果是独立指标（FVD / FID / CLIP-Score / 自定义 reward model）→ **没有独立的 metrics 模块**。README & DEV_MANUAL §14 都把 `videvalkit metric --name fvd` 标为 "planned 2026-05-18"，目前没实现
- `BaseScorer` 存在但只服务 VLM judge，没暴露成独立可调用单元

### 1.3 用户原始需求

> "后面快速集成新的 benchmark 和评测算法也要提供一个快速集成的方式"

—— 翻译为产品需求：

1. **新 benchmark 接入时间从天级压到小时级**（简单情况 < 1 小时）
2. **新 scoring 算法可作为独立单元注册**，既能被 benchmark 引用也能 CLI 直接调用
3. **不强制 fork repo** —— 支持 pip 包、本地目录、用户配置目录三种来源
4. **接入即可自检** —— 自动跑 contract test 而不是等到生产数据上才发现 schema 不对

---

## 2. 目标 / 非目标

### 2.1 目标

| # | 目标 | 验收 |
|---|---|---|
| G1 | **两条接入轨道**：manifest (YAML) 与 Python adapter，前者覆盖 80% 简单场景 | 新 benchmark "每个 prompt 调一次 scorer 拿一个数" 用 manifest < 1 小时跑通 |
| G2 | **独立 scorer 注册** —— `SUPPORTED_METRICS` 注册表 + `videvalkit metric` CLI | `videvalkit metric --name clip-score --gen-videos ... --ref-videos ...` 一行能跑 |
| G3 | **三层插件发现**：built-in / pip entry_points / 本地用户目录 | 任意一种都不需要改 toolkit 源码 |
| G4 | **脚手架 CLI** —— `videvalkit new bench/metric` 一键生成模板代码 | 输出可直接 `videvalkit validate` 通过 |
| G5 | **Contract test 自动化** —— `videvalkit validate bench <name>` 跑标准化检查 | 5 项检查 < 30 秒返回 |
| G6 | **零回归** —— 现有 6 个 anchored benchmark 不受影响 | 现有 tests/ 全绿 |

### 2.2 非目标

- ❌ **不做** benchmark 自动从 paper 生成（自然语言→adapter 代码）。
- ❌ **不做** 算法的"性能 / 准确率自动评估"（"你的新 scorer 比 CLIP-Score 好不好"），只做 schema/契约校验。
- ❌ **不做** sandboxed 插件执行（信任本地用户、信任 pip 包 maintainer）。
- ❌ **不做** scorer 之间的 DAG 编排（"先跑 FVD，再用 FVD 的结果跑 X"）—— 留给 v0.4，目前每个 scorer 独立。

---

## 3. 双轨接入模型

### 3.1 两条轨道

```
┌────────────────────────────────────────────────────────────────┐
│  Track A: Manifest (YAML)        Track B: Python Adapter        │
│  ───────────────────────         ──────────────────────         │
│  适用：每个 prompt 调一次         适用：需要 staging / 跨 prompt │
│       scorer 拿一个分数            复用 / 多阶段 pipeline /      │
│                                    upstream 子进程调用           │
│                                                                  │
│  写一个 manifest.yaml              写一个 BaseBenchmark 子类     │
│       ↓                                  ↓                       │
│  ManifestBenchmark 运行时类        直接放到三层插件路径之一       │
│  自动 implement 4 个抽象方法                                     │
│       ↓                                  ↓                       │
│              都进同一个 SUPPORTED_BENCHMARKS 注册表              │
│              都被同一个 runner.run() 调用                        │
│              都生成同一种 RawResult / Summary 输出                │
└────────────────────────────────────────────────────────────────┘
```

> **关键观察**：80% 新 benchmark 接入只需要回答四个问题——"prompt 在哪、视频按什么 layout 命名、每个 dim 用哪个 scorer、怎么聚合"。这四个问题完全可以 YAML 化。复杂的 20%（如 worldjen 两阶段 VQA、t2vcompbench paper/toolkit 双模式）走 Python。

### 3.2 Track A：Manifest 驱动

**完整示例**：

```yaml
# ~/.videvalkit/benchmarks/my_bench/manifest.yaml
name: my_bench
version: 0.1.0
description: My custom 5-dim benchmark for short videos
env: videvalkit                    # 复用共享 conda env
needs_gpu: true
needs_judge: true

prompts:
  source: jsonl
  path: prompts.jsonl              # 相对 manifest.yaml 的路径
  # 或者：
  # source: hf_dataset
  # repo: my-org/my-prompts
  # split: test

dimensions:
  - name: visual_quality
    weight: 0.3
    scorer: clip-score              # 引用 SUPPORTED_METRICS
  - name: text_alignment
    weight: 0.4
    scorer:
      ref: vlm_judge                # 用 judge（由 runner.run 传入）
      prompt_template: prompts/text_alignment.txt
      mode: middle_frame            # frame 采样模式
      n_frames: 8
  - name: motion_smoothness
    weight: 0.3
    scorer: my_pkg.scorers:MotionSmoothness    # 任意 import path
    kwargs:
      threshold: 0.5

video_layout: "{model}/{prompt_id}-{sample_index}.mp4"

aggregator: weighted_sum            # 引用 SUPPORTED_AGGREGATORS

default_judge: gemma-4-31b-local    # judge 槽位（与 JUDGE_SELECTION_DESIGN 一致）
paper_judge: paper-llava-1.6-34b
recommended_judges:
  - gemma-4-31b-local
  - claude-sonnet-4-6
```

**运行**：

```bash
# 注册即用，无需 toolkit 源码改动
videvalkit list benchmarks                  # 能看到 my_bench
videvalkit eval --bench my_bench --videos ... --workspace ...
```

**实现要点**：

新增 `src/videvalkit/core/manifest_benchmark.py`：

```python
class ManifestBenchmark(BaseBenchmark):
    """运行时由 YAML manifest 实例化的 benchmark。

    所有 4 个抽象方法的实现都从 manifest 字段派生：
      - list_prompts()         ← manifest.prompts (jsonl 或 HF dataset)
      - list_required_videos() ← manifest.video_layout 模板展开
      - evaluate()             ← 每个 dim 的 scorer 串行/并行调用
      - aggregate()            ← manifest.aggregator + dim weights
    """
    def __init__(self, manifest: ManifestSpec, manifest_path: Path) -> None:
        self.name = manifest.name
        self.env_name = manifest.env
        self.dimensions = [d.name for d in manifest.dimensions]
        self._manifest = manifest
        self._manifest_dir = manifest_path.parent
```

`ManifestSpec` 是 pydantic model，加载时严格 schema 校验（不合法直接报错，列出具体行）。

### 3.3 Track B：Python adapter（保留现有路径）

完全等于现状：

- 写 `BaseBenchmark` 子类
- 加 entry 到 `SUPPORTED_BENCHMARKS`（built-in 路径）**或** 通过下面 §4 的插件机制注入

**唯一新增的辅助**：模板生成（`videvalkit new bench --template python` 输出最小可用骨架）。

---

## 4. 三层插件发现机制

### 4.1 三个来源（优先级低 → 高）

```
1. Built-in        src/videvalkit/benchmarks/<name>/        ← 现有路径，不变
2. pip entry_point  pyproject.toml [project.entry-points]    ← 新增
                   "videvalkit.benchmarks" / ".metrics"
3. Local           ~/.videvalkit/benchmarks/<name>/          ← 新增
                   $CWD/.videvalkit/benchmarks/<name>/       ← 新增（项目级覆盖）
```

同名时 **后者覆盖前者**，与 judge 配置同语义。

### 4.2 第三方 pip 包接入示例

第三方维护的 `videvalkit-anubis` 包：

```toml
# pyproject.toml of videvalkit-anubis
[project.entry-points."videvalkit.benchmarks"]
anubis = "videvalkit_anubis.bench:AnubisBenchmark"

[project.entry-points."videvalkit.metrics"]
anubis-fvd = "videvalkit_anubis.metrics:AnubisFVD"
```

用户 `pip install videvalkit-anubis` 即可：

```bash
videvalkit list benchmarks         # anubis 自动出现
videvalkit eval --bench anubis ...
videvalkit metric --name anubis-fvd ...
```

### 4.3 本地目录接入示例

```
~/.videvalkit/
├── benchmarks/
│   ├── my_bench/
│   │   ├── manifest.yaml          ← Track A
│   │   └── prompts.jsonl
│   └── my_python_bench/
│       └── benchmark.py            ← Track B（含 __videvalkit_register__ 标记）
└── metrics/
    └── my_scorer.py                ← BaseScorer 子类
```

`benchmark.py` 通过约定函数注册：

```python
# ~/.videvalkit/benchmarks/my_python_bench/benchmark.py
from videvalkit.core.benchmark import BaseBenchmark

class MyPythonBench(BaseBenchmark):
    name = "my_python_bench"
    ...

def __videvalkit_register__():
    return {
        "benchmarks": {"my_python_bench": {
            "cls": MyPythonBench,
            "env": "videvalkit",
            "needs_judge": False,
            ...
        }}
    }
```

加载器只调用 `__videvalkit_register__()`，不做模块通配 import 副作用。

### 4.4 加载顺序与冲突处理

实现于 `src/videvalkit/plugins/loader.py`：

```python
def discover_all() -> RegistrySnapshot:
    snapshot = RegistrySnapshot.from_builtin()       # 第 1 层
    snapshot.merge(load_entry_points("videvalkit.benchmarks"))  # 第 2 层
    snapshot.merge(load_local_dirs([USER_DIR, PROJECT_DIR]))    # 第 3 层
    return snapshot
```

冲突日志：同名 entry 被覆盖时打 INFO log，写到 `~/.cache/videvalkit/plugin-conflicts.log`，避免静默。

### 4.5 安全边界

- 仅信任 **本地用户文件** 与 **用户主动 pip install 的包**，不做沙箱
- 加载失败不致命：单个 plugin 报错 → warn + skip，其他正常
- 环境变量 `VIDEVALKIT_DISABLE_PLUGINS=1` 关闭所有第三方源（debug 用）

---

## 5. 独立 Scoring 算法层

### 5.1 概念分层

```
┌─────────────────────────────────────────────────────────┐
│              SUPPORTED_METRICS (新增注册表)               │
│                                                          │
│  每个 entry = 一个独立 scoring 算法 (BaseScorer 子类)    │
│                                                          │
│  用法 1：CLI 直接调用                                     │
│     videvalkit metric --name fvd \                       │
│       --gen-videos a/ --ref-videos b/                    │
│                                                          │
│  用法 2：被 benchmark manifest 引用                       │
│     dimensions:                                          │
│       - name: visual_quality                             │
│         scorer: fvd                                      │
│                                                          │
│  用法 3：被 Python adapter 直接 import                    │
│     from videvalkit.metrics import get_metric            │
│     scorer = get_metric("fvd")                           │
│     score = scorer.score(ctx)                            │
└─────────────────────────────────────────────────────────┘
```

### 5.2 注册表 schema

> **v0.2 更新（用户 2026-05-20 确认）**：metric 注册表加 `source` / `kind` / `also_used_by` / `paper_alignment_test` / `license` 字段，支持 **双入口模型** 与 **lift-out 透明性**。详见 [`VIDEO_METRICS_DESIGN.md`](VIDEO_METRICS_DESIGN.md) §6。

```python
# src/videvalkit/configs/metrics.py
SUPPORTED_METRICS = {
    # Distribution-level（4 个）
    "fvd": dict(
        kind="distribution_reference",
        source="canonical/stylegan-v-port",
        cls="videvalkit.metrics.fvd:FVD",
        canonical_backbone="i3d-k400",
        inputs=["gen_videos", "ref_videos"],
        output_kind="scalar_overall",
        version="1.0",
    ),
    # Text-video alignment（2 个，全新写）
    "clip-score": dict(
        kind="per_prompt_reference_free",
        source="canonical/new",
        cls="videvalkit.metrics.clip_score:CLIPScore",
        inputs=["videos", "prompts"],
        output_kind="scalar_per_pair",
    ),
    # Lift-out（v0.2 共 12 个）—— 短名锁 canonical 实现
    "motion-smoothness": dict(
        kind="per_video_reference_free",
        source="vbench/motion-smoothness",        # 透明展示出处
        cls="videvalkit.metrics.motion_smoothness:MotionSmoothness",
        algorithm="AMT-based frame interpolation reconstruction error",
        inputs=["videos"],
        output_kind="scalar_per_video",
        also_used_by=["vbench", "vbench_pp"],     # 哪些 bench 复用本 metric
        paper_alignment_test="tests/test_metric_motion_smoothness.py",
        license="Apache-2.0 (vbench)",
    ),
    # ... 全部 19 个 entry 见 VIDEO_METRICS_DESIGN.md §6
}
```

**字段语义**：

| 字段 | 用途 |
|---|---|
| `kind` | 输入需求：`distribution_reference` / `per_video_reference_free` / `per_prompt_reference_free` / `per_video_with_ref_image` / `per_video_with_vlm_judge`；CLI 缺参时按此 fail-fast |
| `source` | `canonical/<impl>` 或 `<bench>/<dim>`；`metric show` 透明展示 |
| `needs_judge` | `bool` —— 是否需 VLM/LLM chat 调用（影响 `--no-judge` filter）。用户 2026-05-20 加入 |
| `compute_kind` | `local_vision` / `local_text_only` / `needs_vlm` / `needs_llm`；细粒度信息供 doctor 报告 |
| `tags` | `list[str]` —— capability tags from controlled vocab（44 个，见 [`CAPABILITY_TAGS_DESIGN.md`](CAPABILITY_TAGS_DESIGN.md)）。决定 `eval --capability X` 反向索引 |
| `also_used_by` | 哪些 bench 复用此 metric（lift-out 才有） |
| `paper_alignment_test` | 哪个测试文件保证 paper 对齐 + bit-exact lift |
| `license` | 上游 license |

> **`needs_judge` 与 `--no-judge` CLI filter**（用户 2026-05-20 确认）：v0.2 的 20 个 metric 中 17 个不需 judge（distribution / alignment / frame perceptual / temporal 全部 + numeracy / motion-accuracy / identity-preservation），3 个需 VLM（object-binding / spatial-relationship / artifact-diagnostic）。`videvalkit list metrics --no-judge` 过滤显示 judge-free 子集；`eval --no-judge` 在选了需 judge 的 metric/bench 时 fail-fast。

### 5.3 BaseScorer 复用与边界

现有 `BaseScorer` (`src/videvalkit/core/scorer.py`, 74 行) 接口：

```python
def score(self, ctx: ScoreContext) -> ScoreResult: ...
async def ascore(self, ctx: ScoreContext) -> ScoreResult: ...
```

`ScoreContext` 已经能携带 `video_path / prompt / meta`，**无需修改**。新增独立 metric 直接继承 `BaseScorer` 即可。

VLM judge 是 `BaseScorer` 的一种 `kind="vlm_judge_http"`，独立 metric 是 `kind="gpu_metric" | "cpu"`，调度层 (`scheduler/`) 已经按 `kind` 字段路由。

### 5.4 CLI: `videvalkit metric`

```bash
videvalkit metric list                              # 列所有注册的 metrics
videvalkit metric show fvd                          # 展开 entry 详情
videvalkit metric --name fvd \                      # 跑一次
  --gen-videos path/to/gen/ \
  --ref-videos path/to/ref/ \
  --device cuda:0 \
  --output result.json
videvalkit metric --name clip-score \
  --gen-videos gen/ \
  --prompts prompts.jsonl \
  --output result.json
```

实现：复用现有 `runner.run` 的 workspace + api_log 基础设施，只是 benchmark 维度退化为单 metric。

### 5.5 v0.2 内置 metrics（用户 2026-05-20 确认）

> **重大调整**：原计划 7 个 metric（含 PSNR/SSIM/LPIPS/FID-image）→ 调整为 **19 个 metric**，分两档；ref-based 窄适用指标（PSNR/SSIM/LPIPS/FID-image）**移出 v0.2**（T2V 无 ground-truth，这类指标不适用）。完整 spec 见 [`VIDEO_METRICS_DESIGN.md`](VIDEO_METRICS_DESIGN.md)。

**通用 T2V quality (14 个)**：

| 子组 | metrics |
|---|---|
| Distribution (4) | `fvd` / `vfid` / `kvd` / `clip-fvd`（experimental） |
| Text-video alignment (2) | `clip-score` / `viclip-score` |
| Frame perceptual (2, lift) | `aesthetic-quality` / `imaging-quality` |
| Temporal (6, lift) | `motion-smoothness` / `temporal-flickering` / `subject-consistency` / `background-consistency` / `dynamic-degree` / `motion-magnitude` |

**专用维度 (5 个)**：

| metric | 测什么 |
|---|---|
| `object-binding` (lift) | 名词/物体保真 |
| `spatial-relationship` (lift) | 空间语言 |
| `numeracy` (lift) | 数量准确性 |
| `motion-accuracy` (lift) | 动作准确性 |
| `identity-preservation` (new, ArcFace) | 人物/角色保持 |

**双入口模型**：所有 lift-out metric 都同时通过 `eval --bench X --dimensions Y` 与 `metric --name Y` 调用，共用单一实现，位级一致硬契约（≤ 1e-6 误差），见 VIDEO_METRICS_DESIGN.md §5。

---

## 6. 脚手架 CLI

> **⚠ DEFERRED to v0.3 候选**（用户 2026-05-20 确认）：v0.2 优先集成 19 个具体 metric，scaffolding CLI（`videvalkit new bench/metric/judge`）属于"用户不直接看到的基础设施"，推到 v0.3 candidate 列表。本节设计保留为启动时参考。
>
> v0.2 期间用户加自己的 benchmark / metric 走 `~/.videvalkit/` 目录手写，或 pip entry_points 接入（§4 已实现）—— 不需要 scaffolding 命令辅助。

### 6.1 命令

```bash
videvalkit new bench <name> \
  [--template manifest|python] \         # default: manifest
  [--target builtin|local|package] \     # default: local (~/.videvalkit/...)
  [--with-prompts <jsonl>] \
  [--judge-required]

videvalkit new metric <name> \
  [--template reference|reference_free] \
  [--target local|package]

videvalkit new judge <name>              # 输出 yaml snippet 到 stdout（不写文件）
```

### 6.2 输出（以 `videvalkit new bench my_bench --template manifest --target local` 为例）

```
~/.videvalkit/benchmarks/my_bench/
├── manifest.yaml          ← 填好骨架，TODO 标注 user 必填项
├── prompts.jsonl          ← 3 个示例 prompt（让 smoke 直接能跑）
├── README.md              ← 包含 "下一步" 清单
└── tests/
    └── test_smoke.py      ← 一行 `videvalkit validate bench my_bench`
```

`videvalkit new bench ... --template python` 多生成 `benchmark.py` 骨架并引用 `__videvalkit_register__`。

### 6.3 模板存放

`src/videvalkit/templates/`：

```
templates/
├── benchmark_manifest/
│   ├── manifest.yaml.j2
│   ├── prompts.jsonl
│   └── README.md.j2
├── benchmark_python/
│   ├── benchmark.py.j2
│   ├── __init__.py
│   └── manifest.yaml.j2   # 仍生成，做元数据
└── metric/
    ├── scorer.py.j2
    └── README.md.j2
```

用 jinja2 渲染；jinja2 已经在 worldjen 里用过，是现成依赖。

---

## 7. Contract Test 自动化

> **⚠ DEFERRED to v0.3 候选**（用户 2026-05-20 确认）：与 §6 一同推迟。v0.2 用现有 `tests/test_skeleton.py` 手动覆盖 anchored benchmark；自动 contract validator 推 v0.3。
>
> v0.2 期间新增 metric 的契约保证通过 §5 的双入口位级一致测试强制（见 VIDEO_METRICS_DESIGN.md §5.3），不依赖 generic validator。

### 7.1 `videvalkit validate bench <name>`

7 项检查（基于现有 `tests/test_skeleton.py` 抽象出来）：

| # | 检查 | 失败示例 |
|---|---|---|
| 1 | entry 在合并后的 `SUPPORTED_BENCHMARKS` 里 | 名字写错 / 插件加载失败 |
| 2 | `list_prompts()` 至少返回 1 个 `PromptItem`，schema 合法 | dimensions 字段类型错 |
| 3 | `list_required_videos(prompts, models=["fake"])` 至少返回 1 个 `VideoSpec` | layout 模板拼写错 |
| 4 | `evaluate(...)` 用一个 mock judge + 1 个 fake video 跑通，返回 `list[RawResult]` | upstream import 路径错 |
| 5 | `aggregate(raw)` 返回 `Summary`，per_dimension 覆盖至少一个 dim | aggregator 名错 |
| 6 | 若 `needs_judge=True`，`default_judge` 在 judges 注册表里存在 | judge 名拼写错 |
| 7 | docs/USER_MANUAL 里有对应章节（可选警告） | 文档没更新（warn 不 fail） |

输出：

```
$ videvalkit validate bench my_bench
[1/7] registry resolution                  ✓
[2/7] list_prompts() schema                ✓
[3/7] list_required_videos() schema        ✓
[4/7] evaluate() with mock judge           ✓ (12 raw results)
[5/7] aggregate() schema                   ✓
[6/7] default_judge resolvable             ✓ (gemma-4-31b-local)
[7/7] doc presence                          ⚠ docs/USER_MANUAL.md missing § my_bench

OK with 1 warning. Run `videvalkit validate bench my_bench --strict` to treat warnings as errors.
```

### 7.2 `videvalkit validate metric <name>`

4 项：

1. entry in `SUPPORTED_METRICS`
2. cls importable
3. `score(ctx)` 用 fixture context 跑通
4. 返回 `ScoreResult` 且 `kind` 字段匹配 manifest 声明

### 7.3 CI 集成

`tests/test_plugin_contract.py`：参数化扫所有插件来源（builtin + 仿真 user dir + 仿真 entry_points），对每个 entry 调 `validate_*`。新插件不通过 contract 则 CI 红。

---

## 8. 文件改动清单

### 8.1 新增

| 路径 | 用途 |
|---|---|
| `src/videvalkit/core/manifest_benchmark.py` | `ManifestBenchmark` 运行时类 + `ManifestSpec` pydantic |
| `src/videvalkit/plugins/__init__.py` | |
| `src/videvalkit/plugins/loader.py` | 三层发现 + 合并 + 冲突日志 |
| `src/videvalkit/plugins/discovery.py` | entry_points / local dir 扫描细节 |
| `src/videvalkit/configs/metrics.py` | `SUPPORTED_METRICS` 注册表 |
| `src/videvalkit/metrics/__init__.py` | `get_metric(name)` |
| `src/videvalkit/metrics/fvd.py` | FVD scorer |
| `src/videvalkit/metrics/clip_score.py` | CLIP-Score |
| `src/videvalkit/metrics/{fid,psnr,ssim,lpips,inception_score}.py` | 其余 5 个 metric |
| `src/videvalkit/cli_metric.py` | `metric` 子命令组 |
| `src/videvalkit/cli_new.py` | `new bench/metric/judge` 脚手架命令 |
| `src/videvalkit/cli_validate.py` | `validate bench/metric` |
| `src/videvalkit/templates/` | jinja2 模板（见 §6.3） |
| `tests/test_manifest_benchmark.py` | manifest 加载 + 运行 e2e |
| `tests/test_plugin_loader.py` | 三层发现 + 优先级 + 冲突 |
| `tests/test_plugin_contract.py` | 自动 contract 检查参数化 |
| `tests/test_metrics_*.py` | 每个 metric 一个 smoke test |
| `examples/manifest_benchmark/` | 完整 manifest 示例（可直接 copy） |
| `docs/INTEGRATION_FRAMEWORK_DESIGN.md` | 本文 |
| `docs/INTEGRATION_GUIDE.md` | 用户视角的 "how to add a benchmark in < 1 hour" |

### 8.2 修改

| 路径 | 修改点 |
|---|---|
| `src/videvalkit/configs/__init__.py` | `SUPPORTED_BENCHMARKS` 改为函数式访问，内部 lazy-merge 插件来源；保留 dict 别名 |
| `src/videvalkit/cli.py` | 接入 `metric / new / validate` 子命令组 |
| `src/videvalkit/runner.py` | 调用 `discover_all()` 而非直接读 dict |
| `src/videvalkit/diagnostics.py` | `doctor` 报告插件加载状态 |
| `pyproject.toml` | 添加 `jinja2`（已是 worldjen 间接依赖，提到 top-level）、`entry-points` group 声明 |
| `docs/DEV_MANUAL.md` | §4 加 "插件层" 模块；§5.2 Module B 更新；§14 Standalone Metrics 标 done |
| `docs/USER_MANUAL_*.md` | 加 §X "Adding your own benchmark / metric" |
| `README.md` | TL;DR 加一行 "extend with one YAML file or one Python class" |

### 8.3 删除

无。所有改动 backward-compatible。

---

## 9. 兼容性 & 迁移

| 现有用法 | v0.1 行为 |
|---|---|
| `videvalkit eval --bench vbench ...` | 不变；vbench 仍走 built-in Python adapter |
| `from videvalkit.configs import SUPPORTED_BENCHMARKS` | 返回合并后的快照，含插件来源 |
| 现有 `tests/test_*.py` | 全绿（contract test 是新增） |
| 现有 6 个 anchored adapter 的代码 | 一行不动；它们继续是 Python adapter，与新的 manifest 模式平行存在 |

### 9.1 选择性迁移建议

`physics_iq` 和 `v_reasonbench` 当前是 stub，结构简单 —— 可作为 manifest 模式的首批 dogfood 候选，验证 manifest 表达力。

---

## 10. 里程碑拆解（v0.2，用户 2026-05-20 调整）

| 阶段 | 内容 | 工作量 |
|---|---|---|
| **M1 — Plugin loader** | 三层发现 + 合并 + 单测 + diagnostics 接入 | 1 day |
| **M2 — Manifest benchmark** | `ManifestSpec` + `ManifestBenchmark` + e2e test（用 physics_iq 改造做证明） | 1.5 day |
| **M3 — Metrics module** | **19 个 metric 全交付**（6 新写 + 12 lift-out + 1 ArcFace 新写），双入口位级一致；详见 [`VIDEO_METRICS_DESIGN.md`](VIDEO_METRICS_DESIGN.md) §14 | **8.3 day** |
| ~~M4 — Scaffolding~~ | **DEFERRED to v0.3 候选** | (0) |
| ~~M5 — Contract validator~~ | **DEFERRED to v0.3 候选** | (0) |
| **M6 — Docs & examples** | USER_MANUAL 章节 + 完整 example manifest | 0.5 day |

**总计 ≈ 11.3 person-days**（v0.2 内 INTEGRATION 部分）。比原 6.5 day 增 4.8 day 因 M3 metric 工作量从 2 day 扩到 8.3 day；M4/M5 推迟省 1.5 day。

并行机会：M3（metric lift-out 12 个）与 M2（manifest）可并行；M4/M5 已推迟。

---

## 11. Open Questions

> 评审时需要决策的点：

1. **Manifest schema 版本演进策略**：v0.1 schema 升级到 v0.2 时如何兼容？倾向 **manifest 文件顶部强制写 `schema_version: 1`**，loader 按 schema_version 路由解析器。
2. **是否在 v0.1 提供 `videvalkit publish` 帮助用户把 local benchmark 打包成 pip 包**？倾向 **v0.2**，v0.1 只做 entry_points 文档。
3. **Manifest 里的 `scorer: my_pkg.scorers:MotionSmoothness` 这种 import path 是否需要 pre-validate**（加载 manifest 时就 import，而非 runtime）？倾向 **lazy + 在 `validate bench` 时强制 import 校验**。
4. **是否允许 manifest 引用其他 benchmark 的 prompt 集**（"我的 benchmark 复用 vbench 的 prompts 但用我的 scorer"）？倾向 **v0.1 不支持**，避免循环依赖；v0.2 加 `prompts.source: benchmark_ref`。
5. **Metric 是否需要 cache key 概念**（同一对视频跑两次 FVD 是否复用结果）？倾向 **v0.1 简单 file-based cache + `--no-cache` flag**，复杂的 content-hash 缓存推到后续。
6. **`videvalkit new` 默认 target 是 `local` (`~/.videvalkit/`) 还是 `package` (CWD `./my_bench/`)**？倾向 **local**，因为最低摩擦；用户想发布时显式加 `--target package`。

---

## 12. 风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| Manifest 表达力不足，用户被迫频繁退化到 Python adapter | manifest 沦为玩具 | M2 用 physics_iq + v_reasonbench 真实改造做表达力验证；不通过则补 schema |
| 插件加载顺序细节让用户困惑（"为什么我的本地版本没生效"） | 隐性 bug，难诊断 | `doctor` 详细列每个 entry 的最终来源；`videvalkit list benchmarks --verbose` 显示 source 列 |
| pip entry_points 在 conda env 与 system Python 之间错乱 | 第三方包加载不到 | M1 测试覆盖两种 Python 安装路径；doctor 报告 sys.path 与 entry_points 来源 |
| 接入的第三方 metric 依赖把共享 conda env 撑爆 | 长期复杂度 | 文档明确建议第三方 metric 包声明 `requires` 字段，列重依赖；`videvalkit metric` 在 missing dep 时给出明确 install 提示 |
| Contract test 过严，把合理但非典型的 benchmark 拒之门外 | 误伤 | 7 项中只 5 项强制，2 项 warn-only；提供 `--no-strict` |

---

## 13. 决策快照

> 评审时一次性确认下列默认决策（"同意"即全部采纳）：

- ✅ 双轨：manifest（80% 简单场景）+ Python adapter（20% 复杂场景）并存
- ✅ 三层插件发现：builtin / pip entry_points / 本地用户目录
- ✅ 同名冲突 = top-level 覆盖 + INFO log，不做 deep-merge
- ✅ Manifest schema 顶部强制 `schema_version: 1`
- ✅ 新增 `SUPPORTED_METRICS` 注册表，**复用 `BaseScorer` + 新增 `BaseDistributionMetric`** 并列（v0.2 调整，详见 [`VIDEO_METRICS_DESIGN.md`](VIDEO_METRICS_DESIGN.md) §5）
- ✅ **v0.2 内置 19 个 metric**（用户 2026-05-20 调整）：14 通用 T2V quality + 5 专用维度；PSNR/SSIM/LPIPS/FID-image **移出 v0.2**
- ✅ **双入口模型**：lift-out metric 同时通过 `eval --bench` 与 `metric --name` 调用，位级一致硬契约
- ✅ ~~`videvalkit new` 脚手架~~ **DEFERRED to v0.3 候选**
- ✅ ~~Contract validator~~ **DEFERRED to v0.3 候选**
- ✅ `videvalkit publish` 推到 v0.3+
- ✅ Manifest 跨 benchmark prompt 复用推到 v0.3+
- ✅ Metric 简单 file-based cache + `--no-cache`，复杂 content-hash 推到 v0.3+
- ✅ `VIDEVALKIT_DISABLE_PLUGINS=1` 一键关闭所有第三方源

---

## 14. 与其他设计文档的关系

| 文档 | 关系 |
|---|---|
| [`JUDGE_SELECTION_DESIGN.md`](JUDGE_SELECTION_DESIGN.md) | Judge 是 benchmark 的子组件；本文 §3.2 manifest 的 `default_judge / paper_judge / recommended_judges` 字段直接复用 judge 文档定义 |
| `DEV_MANUAL.md` §5.2 Module B | 现有 6 个 anchored adapter 的设计；本文不改 Module B，新增 Module B' (Plugin) 与 Module I (Metrics) |
| `DEV_MANUAL.md` §14 Standalone Metrics | 该节标 "planned 2026-05-18"；本文 §5 即其落地设计 |
| `TEST_MANUAL.md` | 新插件接入时自动追加章节模板（`videvalkit validate bench` 输出可直接 paste） |

—— end of design v0.1 ——
