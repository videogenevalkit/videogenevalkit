# Quick Eval & Training-Time Monitoring — Design Doc

| 字段 | 内容 |
|---|---|
| 版本 | v0.1 (draft) |
| 状态 | Design — pending review |
| 创建 | 2026-05-20 |
| 影响范围 | `core/`, `configs/`, `cli.py`, 新增 `profiles/`, `subsets/`, `training/` |
| 关联文档 | [`PRODUCT_DESIGN.md`](PRODUCT_DESIGN.md) §5（作为**支柱 D**） · [`JUDGE_SELECTION_DESIGN.md`](JUDGE_SELECTION_DESIGN.md)（profile 复用 judge 三档） · [`INTEGRATION_FRAMEWORK_DESIGN.md`](INTEGRATION_FRAMEWORK_DESIGN.md)（subset 通过 manifest 声明） |
| 目标读者 | 模型研发者（训练监控） · 端到端评测用户（demo / smoke） · benchmark 集成者（要为新 bench 提供 quick subset） |

---

## 1. 背景与问题陈述

### 1.1 现状

`videvalkit eval --bench X` 默认跑**全集**：
- VBench v1 ~1600 prompts × 5 samples × 多个 dim → 单卡 GPU 4–8 小时
- WorldJen 800 prompts × VLM 判 16 dim → 6–10 小时
- T2V-CompBench 1400 prompts × LLaVA-34B → 数十小时

**问题**：
1. **训练时无法用** —— 训一个 step 10 分钟，评测一次要 8 小时，无法在训练过程中插入评测来监控指标
2. **demo / smoke 路径模糊** —— 现有 `fetch-smoke-data` 拉的是 3 个示例视频，但 `eval --bench` 时不知道"我现在是 smoke 还是 prod"，配置耦合
3. **多 bench 一次跑没有时间预算意识** —— 用户不知道 `--bench vbench --bench worldjen --bench t2vcompbench` 跑下来要 24 小时还是 2 小时
4. **bench 选择粒度不够** —— 想跑"6 个 bench 各跑一个 quick 子集快速看趋势" 当前要 6 次 CLI

### 1.2 用户原始需求

> "1. 要快速评测，训练过程的检测，这里要自动化的划分一个靠谱的子集对于每个 bench，能保证快速能出结果"
> "2. 用户可以选择快速评测还是全评，也可以挑选不同的评测 bench 来测"

—— 翻译为产品需求：

| # | 需求 | 验收 |
|---|---|---|
| R1 | 每个 benchmark 提供一个**自动划分、可复现、统计上靠谱**的 quick 子集 | 50–80 prompts，单 bench < 10 min，Spearman ρ vs 全集 ≥ 0.85 |
| R2 | "靠谱"必须可量化 —— 不能只是随机抽样 | calibration 跑过、文档可查、subset 文件可固化 |
| R3 | CLI 一行切换 quick / full | `--profile quick` / `--profile full` |
| R4 | CLI 一行跑多个 bench | `--bench A --bench B` 或 `--all` |
| R5 | 跑之前能预估耗时与成本 | `videvalkit estimate` 命令 |
| R6 | 训练循环里能调用 | Python API `videvalkit.training.monitor(...)` |

---

## 2. 目标 / 非目标

### 2.1 目标

1. 引入 **eval profile** 概念：把 subset + judge 档位 + frame 采样 + sample 数 打包成一个名字（`quick / standard / full`）
2. 每个 anchored benchmark 在 v0.2 提供 **3 个 profile** 的 subset 文件，并标注每个的 Spearman ρ
3. CLI 支持 `--profile <name>` 和 `--bench <name>` 多次/`--all`
4. 新增 `videvalkit estimate` 评估时间与 token 消耗
5. 新增 `videvalkit watch` 训练监控循环
6. 新增 Python `videvalkit.training` 子模块，供 trainer 内嵌
7. Subset 文件**版本化**（subset_v1 / v2 / v3），历史 run 可复现

### 2.2 非目标

- ❌ **不做** 自适应 subset（"基于当前模型表现挑下次跑哪些 prompt"）—— 太复杂，留 v0.4
- ❌ **不做** 训练 loop 集成（pytorch hook、accelerate callback）—— 用户只通过 Python API 主动调用
- ❌ **不做** subset 在线 / 跨用户共享 —— subset 文件就在 repo 里，所有人用同一套
- ❌ **不做** 完全自动的 ρ 校准 pipeline —— 校准是离线 offline 一次性活，结果是 checked-in JSON

---

## 3. 核心概念：Eval Profile

### 3.1 Profile 是什么

一个 profile = 一个**评测成本档位**的命名打包，包含 5 个维度：

```yaml
profiles:
  quick:
    description: "Training-time monitoring — fast & stable, 5–10 min per bench"
    subset: subsets/quick_v1.json       # prompt id 列表 + dim 覆盖元数据
    judge: default                       # JUDGE doc 三档之一
    frame_sampling:
      n_frames: 4                        # 默认 8，quick 减半
      mode: uniform
    samples_per_prompt: 1
    estimated:
      wallclock_min: 8
      gpu_hours: 0.15
      judge_calls: 60
      judge_tokens_in: 30000
      judge_tokens_out: 6000

  standard:
    description: "Reliable eval — Spearman ρ ≥ 0.95 vs full, 30–60 min per bench"
    subset: subsets/standard_v1.json
    judge: default
    frame_sampling:
      n_frames: 8
      mode: uniform
    samples_per_prompt: 1
    estimated:
      wallclock_min: 40
      gpu_hours: 0.8
      judge_calls: 300

  full:
    description: "Paper-faithful reproduction — full prompt set"
    subset: null                         # 不裁剪
    judge: default                       # 或 paper（用 --judge paper 覆盖）
    frame_sampling:
      n_frames: 8
      mode: uniform
    samples_per_prompt: 5                # vbench 类 multi-sample 才有意义
    estimated:
      wallclock_min: 480
      gpu_hours: 8.0
      judge_calls: 8000
```

### 3.2 三个内置 profile（v0.2 固定）

| Profile | 用途 | Subset 规模 | Wallclock 量级 | 期望 ρ vs full |
|---|---|---:|---:|---:|
| `quick` | 训练监控 / smoke / CI | 50–100 prompts | 5–10 min | ≥ 0.85 |
| `standard` | 论文 ablation / 模型迭代评测 | 200–400 prompts | 30–60 min | ≥ 0.95 |
| `full` | 论文最终数 / leaderboard 提交 | 全集（800–2000+） | 数小时 | 1.00 (定义) |

> `paper` 不是一个 profile —— 它是 **judge 维度**（`--judge paper`，见 JUDGE doc）。可以组合：`--profile full --judge paper` 才是严格 paper-faithful。

### 3.3 Profile 的可组合性

`--profile quick --judge claude-sonnet-4-6` 完全合法 —— profile 决定 prompt 子集、frame 数、sample 数；judge 选择正交。

| CLI 组合 | 实际效果 |
|---|---|
| `--profile quick` | subset + 默认 judge（profile 里 judge=default 解析为该 bench 的 default_judge） |
| `--profile quick --judge paper` | subset + paper-faithful judge（如 LLaVA-1.6-34B），慢但精确 |
| `--profile full --judge gpt-4o` | 全 prompt + GPT-4o judge |
| `--profile full --judge paper` | **完全复现 paper** |

---

## 4. Subset 设计

### 4.1 "靠谱" 的定义

一个 subset 文件是"靠谱的"当且仅当：

| 标准 | 量化 |
|---|---|
| **覆盖性** | 每个 dimension 至少 N 个 prompt（N 在 manifest 指定，典型 N=3 quick / N=10 standard） |
| **分层均衡** | 按 dim / category / difficulty stratified |
| **可复现** | 同一 subset 文件无论谁跑都得到同一组 prompt id |
| **可校准** | 在 ≥ 3 个已知模型上，子集分数对全集分数的 Spearman ρ ≥ 阈值 |
| **版本化** | `subset_v1.json` 一旦发布不再修改；改进版本是 `subset_v2.json` |

### 4.2 Subset 文件 schema

```json
{
  "schema_version": 1,
  "subset_name": "quick_v1",
  "benchmark": "worldjen",
  "created": "2026-05-20",
  "n_prompts": 48,
  "selection_method": "stratified_seeded",
  "selection_seed": 42,
  "calibration": {
    "method": "spearman",
    "validation_models": ["Kling-v2.6", "Sora", "HunyuanVideo", "CogVideoX-5B"],
    "spearman_rho_overall": 0.91,
    "spearman_rho_per_dim": {
      "motion_stability": 0.88,
      "logic_physics": 0.85,
      "instruction_adherence": 0.93,
      "aesthetic_quality": 0.94
    },
    "max_dim_disagreement": 0.08
  },
  "stratification": {
    "by": ["dimension"],
    "per_stratum_n": 3
  },
  "prompt_ids": [
    "wj_0042", "wj_0157", "wj_0203", ...
  ]
}
```

### 4.3 Subset 选择算法（v0.2）

**Method 1: stratified seeded random**（v0.2 默认 / fallback）
1. 按 dim（或 dim × difficulty）分层
2. 每层用固定 seed 随机抽 N 个
3. 跑 4 个 validation 模型计算 ρ
4. 若 ρ < 阈值，调整 seed 或加大 N 重抽
5. 保存为 `subset_v1.json`

**Method 2: leaderboard-calibrated** (v0.3+)
1. 利用 `validation/expected/<bench>_leaderboard.json` 里的 ≥ 5 个模型 per-prompt 分数
2. 在所有可能的 size-k 子集中（用启发式 / submodular optimization）找 ρ 最大者
3. 保存 + 文档化

v0.2 用 Method 1，先把通路跑通；v0.3 切到 Method 2 时只换 subset 文件，CLI 不变。

### 4.4 Subset 存放位置

```
src/videvalkit/benchmarks/<bench_name>/subsets/
├── quick_v1.json
├── standard_v1.json
└── README.md       ← 解释每个 subset 的 ρ + 何时更新
```

每个 anchored benchmark 在 v0.2 必须交付 quick_v1 + standard_v1 两个文件。Manifest-driven benchmark（INTEGRATION doc Track A）在 manifest 里**可选**声明 subsets，未声明则 quick = 全集随机 20%。

### 4.5 Subset 校准流程（offline，repo 维护者执行）

```bash
# 1. 拉全集 leaderboard JSON
videvalkit fetch-leaderboard --bench worldjen

# 2. 生成候选 subset
videvalkit subset propose --bench worldjen --target quick \
  --n-prompts 48 --seed 42 \
  --output proposed_quick.json

# 3. 计算 ρ
videvalkit subset calibrate --bench worldjen \
  --subset proposed_quick.json \
  --validation-models Kling-v2.6,Sora,HunyuanVideo,CogVideoX-5B

# 4. 若 ρ ≥ 0.85，固化
mv proposed_quick.json src/videvalkit/benchmarks/worldjen/subsets/quick_v1.json
```

`videvalkit subset propose/calibrate` 是新子命令（v0.2）。

---

## 5. CLI 设计

### 5.1 `eval` 扩展（单 bench）

```bash
# 现有
videvalkit eval --bench worldjen --videos ... --workspace ...

# v0.2 新增
videvalkit eval --bench worldjen --profile quick --videos ... --workspace ...
videvalkit eval --bench worldjen --profile standard ...
videvalkit eval --bench worldjen --profile full --judge paper ...
videvalkit eval --bench worldjen --subset path/to/my_custom_subset.json ...
```

flag 优先级：`--subset` > `--profile` > 默认 (full)。

### 5.2 `eval-suite` 多 bench（合并 JUDGE doc 推迟项）

```bash
# 一次跑多个 bench
videvalkit eval-suite \
  --bench vbench --bench worldjen --bench videobench \
  --videos ~/videos --workspace ~/runs/quick \
  --profile quick

# 全 6 anchored bench
videvalkit eval-suite --all-anchored --profile quick \
  --videos ~/videos --workspace ~/runs/full

# 不同 bench 不同 profile
videvalkit eval-suite \
  --bench vbench=full \
  --bench worldjen=quick \
  --bench t2vcompbench=standard \
  --videos ... --workspace ...

# 不同 bench 不同 judge（结合 JUDGE doc）
videvalkit eval-suite \
  --bench vbench2 --bench worldjen \
  --judge-for vbench2=local-llava-video-7b \
  --judge-for worldjen=claude-sonnet-4-6 \
  --profile standard
```

执行：内部串行调用 `runner.run`，共享同一个 workspace；末尾自动调 `aggregate`。

### 5.3 `estimate` 预估命令

跑之前可以**先算账**：

```bash
videvalkit estimate \
  --bench vbench --bench worldjen --bench t2vcompbench \
  --profile quick \
  --judge gpt-4o \
  --n-models 1

# 输出：
# ┌───────────────┬──────────┬───────┬───────────┬─────────────┐
# │ Benchmark     │ Wallclock│ GPU-h │ API calls │ Est. tokens │
# ├───────────────┼──────────┼───────┼───────────┼─────────────┤
# │ vbench        │ 6 min    │ 0.10  │ 0 (no VLM)│ –           │
# │ worldjen      │ 8 min    │ 0.05  │ 96        │ ~50k in/10k out │
# │ t2vcompbench  │ 12 min   │ 0.25  │ 70        │ ~40k in/8k out │
# ├───────────────┼──────────┼───────┼───────────┼─────────────┤
# │ TOTAL         │ 26 min   │ 0.40  │ 166       │ ~90k/18k    │
# │ Est. $ (gpt-4o│          │       │           │ ~$0.95      │
# └───────────────┴──────────┴───────┴───────────┴─────────────┘
```

实现：从 profile.estimated 字段累加 + judge pricing table（写死 / 可配）。

### 5.4 `watch` 训练监控

```bash
# 监听 checkpoint 目录，每出新 model 就跑一次 quick
videvalkit watch \
  --videos-pattern "/data/train_runs/run_42/checkpoints/step_*/samples/" \
  --workspace /data/train_runs/run_42/eval \
  --bench vbench --bench worldjen \
  --profile quick \
  --judge default \
  --on-new-checkpoint eval \
  --plot live_chart.html
```

行为：
- 轮询（或 inotify）检测新视频目录
- 一旦稳定（10s 无变化）→ 跑 `eval-suite --profile quick`
- 每次结果追加到 `eval/timeline.jsonl`（一行一个 checkpoint）
- 可选生成 HTML / TensorBoard plot

### 5.5 `subset` 子命令组

```bash
videvalkit subset list                          # 列所有 bench 的 subset 文件
videvalkit subset show <bench>/<subset_name>    # 展开 JSON
videvalkit subset propose --bench X ...         # 离线工具：生成候选
videvalkit subset calibrate --bench X ...       # 离线工具：算 ρ
videvalkit subset compare \                     # 对比两个 subset 对同一组数据的差异
  --subset-a quick_v1 --subset-b quick_v2 \
  --on-results path/to/raw/
```

---

## 6. Python API — 训练集成

### 6.1 设计：`videvalkit.training.monitor`

```python
from videvalkit.training import monitor, MonitorConfig

cfg = MonitorConfig(
    benches=["vbench", "worldjen"],
    profile="quick",
    judge="default",
    workspace="/data/train_runs/run_42/eval",
)

# 训练循环里
for step in range(0, 100_000, 1000):
    train_one_step(...)
    if step % 5000 == 0:
        videos_dir = generate_samples(model, prompts=monitor.preview_prompts())
        result = monitor.eval(videos_dir=videos_dir, model_name=f"step_{step}", cfg=cfg)
        # result.summary: {"vbench": Summary, "worldjen": Summary}
        # result.overall: float (cross-bench z-score)
        tensorboard.add_scalar("eval/overall", result.overall, step)
        for bench, s in result.summary.items():
            tensorboard.add_scalar(f"eval/{bench}", s.overall, step)
```

### 6.2 `monitor.preview_prompts()` —— 训练时反向暴露 prompt

训练时需要先生成视频再评测。但 `--profile quick` 选了哪 50 个 prompt？训练 loop 怎么知道要给哪些 prompt 生成视频？

`monitor.preview_prompts(bench=None) -> list[PromptItem]` 返回当前 profile 下的全部 prompt（去重跨 bench），训练代码用它生成视频。

```python
prompts = monitor.preview_prompts()              # 所有 bench 的 union
# 或
prompts = monitor.preview_prompts(bench="worldjen")
generate_videos(prompts, output_dir=videos_dir)
result = monitor.eval(videos_dir=videos_dir, ...)
```

### 6.3 `MonitorConfig` 持久化

`MonitorConfig.save("monitor.yaml")` / `MonitorConfig.load(...)` —— 训练脚本里写一行 `cfg = MonitorConfig.load("monitor.yaml")`，profile / bench / judge 都从 yaml 来。便于 reproduce。

### 6.4 不做的事

- ❌ 不内置 PyTorch Lightning callback / HF Trainer integration —— 用户自己一行 `if step % N == 0: monitor.eval(...)` 就够了
- ❌ 不内嵌训练 loss / metric 显示 —— 用户已有 wandb/tb
- ❌ 不做 distributed eval（多卡同时跑评测） —— 评测本身已经够快了

---

## 7. 文件改动清单

### 7.1 新增

| 路径 | 用途 |
|---|---|
| `src/videvalkit/configs/profiles.py` | `SUPPORTED_PROFILES` = `{"quick": ProfileSpec(...), "standard": ..., "full": ...}` |
| `src/videvalkit/core/profile.py` | `ProfileSpec` pydantic + `resolve_profile()` |
| `src/videvalkit/core/subset.py` | `Subset` 类 + 加载 / 校验 / 应用 (`subset.filter_prompts(prompts)`) |
| `src/videvalkit/benchmarks/*/subsets/quick_v1.json` | × 6 anchored benchmark |
| `src/videvalkit/benchmarks/*/subsets/standard_v1.json` | × 6 anchored benchmark |
| `src/videvalkit/benchmarks/*/subsets/README.md` | 每 bench 一个 |
| `src/videvalkit/cli_estimate.py` | `videvalkit estimate` |
| `src/videvalkit/cli_subset.py` | `videvalkit subset list/show/propose/calibrate/compare` |
| `src/videvalkit/cli_watch.py` | `videvalkit watch` |
| `src/videvalkit/cli_eval_suite.py` | `videvalkit eval-suite`（含 multi-profile / multi-judge） |
| `src/videvalkit/training/__init__.py` | `monitor`, `MonitorConfig`, `MonitorResult` |
| `src/videvalkit/training/loop.py` | watch 内部实现 + Python API |
| `src/videvalkit/pricing.py` | judge token pricing table |
| `tests/test_profiles.py` | profile schema + resolve |
| `tests/test_subset_application.py` | subset 文件 → prompt filter 通路 |
| `tests/test_estimate.py` | estimate 计算正确性 |
| `tests/test_training_api.py` | `monitor.eval` 端到端 |
| `docs/QUICK_EVAL_DESIGN.md` | 本文 |
| `examples/training_monitor.py` | 训练集成完整示例 |
| `examples/watch_run.sh` | watch 命令完整示例 |

### 7.2 修改

| 路径 | 修改点 |
|---|---|
| `src/videvalkit/configs/benchmarks.py` | 每个 entry 加 `subsets` 字段（指向相对 `subsets/*.json`），加 `supports_profiles` 列表 |
| `src/videvalkit/core/benchmark.py` | `evaluate_and_aggregate` 接受 `profile` 与 `subset` 参数，先 `subset.filter_prompts()` 再调 `list_required_videos` |
| `src/videvalkit/runner.py` | `run()` 加 `profile` / `subset` / `frame_sampling` 参数；resolve_profile 收敛在此 |
| `src/videvalkit/cli.py` | `eval` 加 `--profile` / `--subset`；接入 `estimate / watch / subset / eval-suite` 子命令组 |
| `src/videvalkit/diagnostics.py` | `doctor` 列每 bench 是否有 subset 文件 + ρ 是否达标 |
| `docs/PRODUCT_DESIGN.md` | §5 加 "支柱 D"；§6 路线图加 profile 相关条目 |
| `docs/USER_MANUAL_*.md` | 新章节 §X "Quick eval & training monitoring" |
| `docs/DEV_MANUAL.md` | §5.2 Module B 提到 subset 字段；§17 新增 "Profile & Subset" |
| `docs/TEST_MANUAL.md` | 加一节 "每 bench 的 quick / standard subset 校准 ρ" |
| `README.md` | quickstart 加一条 "want fast eval? `--profile quick` (5-10 min/bench)" |

### 7.3 删除

无。所有改动 backward-compatible（不传 `--profile` 等于 `full` = 当前行为）。

---

## 8. 兼容性 & 迁移

| 现有用法 | v0.2 行为 |
|---|---|
| `videvalkit eval --bench X ...`（不传 profile） | 等价于 `--profile full`，**完全等同当前行为** |
| `validation/expected/*.json` 与 leaderboard 对比 | 仅 `--profile full` 下的结果才参与对比；quick/standard 结果不污染 expected |
| 现有 6 个 anchored adapter | 加 `subsets/` 目录；`evaluate` 函数签名不变（subset 应用在调用前） |
| 现有用户脚本 | 一行不动 |

### 8.1 训练监控历史数据兼容性

`timeline.jsonl` 写入 schema 含 `profile_name` + `subset_version`。subset_v1 → v2 升级后旧的 timeline 仍可读，但比较图会标注"基线版本差异"警告。

---

## 9. Calibration 数据来源

### 9.1 Validation 模型来源

每个 bench 需要 ≥ 3 个已发表模型的全集分数才能算 ρ：

| Bench | Validation 模型源 |
|---|---|
| vbench | HF leaderboard（HunyuanVideo / CogVideoX-5B / Kling-v2.6 / Sora 等 20+ 模型，含 per-dim 分数） |
| vbench2 | HF leaderboard（同上但 v2） |
| videobench | paper + 我们自跑（CogVideoX-5B） |
| worldjen | paper 报告 5 模型 + 我们自跑（Kling-v2.6） |
| worldscore | paper 报告 ≥ 5 模型 |
| t2vcompbench | paper 报告 ≥ 8 模型 |

`videvalkit fetch-leaderboard --bench X` 拉对应 JSON 进入 `validation/expected/`。

### 9.2 ρ 阈值

- quick: Spearman ρ ≥ **0.85** （允许某些 dim 跌到 0.75，但 max disagreement ≤ 0.10）
- standard: Spearman ρ ≥ **0.95**

低于阈值则该 subset 不发布。

### 9.3 ρ 不达标的 fallback

某个 bench 的 quick subset 实在校不到 0.85（如 worldjen 的 logic_physics dim 高方差）：
1. 加大 quick 的 n_prompts（48 → 80）
2. 调整 stratification（按 difficulty 而非 dim）
3. 实在不行 → 在 subset 文件里诚实标记 `quality: "low"` 并在 doctor 输出警告

---

## 10. 里程碑拆解

| 阶段 | 内容 | 工作量 |
|---|---|---|
| **D1 — Profile 抽象** | `ProfileSpec` + `resolve_profile` + `runner.run` 接 profile + 单测 | 1 day |
| **D2 — Subset 应用层** | `Subset` 类 + `filter_prompts` + benchmark 接入 + 单测 | 1 day |
| **D3 — 6 个 anchored bench 的 subset 文件生成** | `subset propose` + `subset calibrate` 工具实现 + 离线跑校准 + checked-in 12 个 JSON（6 bench × 2 subset） | 2 day |
| **D4 — CLI: estimate / subset / eval-suite** | 三组子命令 + pricing table | 1 day |
| **D5 — Training API + watch** | `videvalkit.training` + `videvalkit watch` + example 脚本 | 1.5 day |
| **D6 — Docs + dogfood** | USER/DEV manual + README + 训练监控完整 example + 一个真实模型上跑过 watch 7 天 | 1 day |

**总计 ≈ 7.5 person-days**。可拆 6 个独立 PR。

并行机会：D3 与 D4 / D5 可并行；D6 串行。

---

## 11. Open Questions

> 评审时需要决策的点：

1. **三个 profile 名字**：`quick / standard / full` vs `fast / mid / full` vs `dev / staging / prod`？倾向 **quick / standard / full**（语义最直接）。
2. **是否允许用户自定义 profile**（`~/.videvalkit/profiles.yaml`）？倾向 **v0.2 不开放**（避免可重现性混乱），v0.3 再加。`--subset` 已能覆盖临时需求。
3. **`--all-anchored` vs `--bench all`**：哪种语法？倾向 **`--all-anchored`** 因为"all" 还可以指 anchored + stub，明确点。
4. **standard profile 是否真的需要**？quick / full 二档够不够？倾向 **保留 standard**，因为 quick 校不到 0.95 时还有中间档可选。
5. **judge default 在 profile 里的行为**：profile 写 `judge: default` 时是查 bench.default_judge，还是 paper_judge？倾向 **bench.default_judge**（省钱档）；想 paper 用 `--judge paper` 显式覆盖。
6. **watch 是否做 inotify**：跨平台麻烦；倾向 **轮询 (poll, default 60s) 为主**，inotify 作为 Linux 优化路径 v0.3 加。
7. **timeline.jsonl plot 工具**：内置 HTML / 写 TensorBoard / 都不做？倾向 **写 wandb 兼容的 JSON + 一个最简 HTML（plotly cdn）**，TB integration 留 v0.3。
8. **estimate 的 token 价格表更新策略**：写死在 `pricing.py`？倾向 **写死 + 在文件顶部写"最后更新 2026-05"**，明确不保证准确，用户可覆盖。
9. **subset 文件是否要做 hash 校验**（防止用户改了 subset_v1.json 影响复现性）？倾向 **做**，记录在 timeline.jsonl 的每条记录里。

---

## 12. 风险

| 风险 | 影响 | 对策 |
|---|---|---|
| Quick subset ρ 校不到阈值（某些 bench 高方差） | quick profile 不可信 | §9.3 fallback；坦诚在文档标 "quality: low"；不强行发布烂的 subset |
| 用户混淆 profile / judge / mode 三个维度 | 配置错乱 | doc 一张大表说明三者正交；CLI 输出最开始必打 "running with profile=X judge=Y" |
| watch 命令在训练机上长跑稳定性 | 内存泄漏 / 文件句柄 | D6 真跑一遍 7 天监控；不通过则推迟 |
| Subset 文件维护成本（每次新加 bench / 升级算法都要重校） | 长期 toil | 把校准做成 CI nightly job（v0.3）；v0.2 手动 |
| 训练时生成的视频与 subset 期望的视频不一致 | eval 失败 | `monitor.preview_prompts()` 强制声明清单；不在清单里的视频被 ignore + warn |
| 三个 profile 名字与 INTEGRATION doc 里 manifest 的字段命名冲突 | schema 混乱 | profile 字段统一加前缀 `profile_` 或放 `profiles:` 嵌套 dict |
| pricing.py 过期导致 estimate 不准 | 用户决策被误导 | 文档说"最后更新 YYYY-MM"；CLI 输出末尾打 disclaimer |

---

## 13. 决策快照

> 评审本文时一次性确认（"同意"即采纳）：

- ✅ 引入 **eval profile** 概念，3 个内置：quick / standard / full
- ✅ Profile 维度 = subset + judge 档 + frame sampling + samples_per_prompt + estimated
- ✅ "靠谱"定义 = Spearman ρ vs 全集 ≥ 阈值（quick 0.85 / standard 0.95）
- ✅ Subset 文件 version 化（v1 / v2），checked-in JSON，永不静默改
- ✅ v0.2 用 stratified seeded random 方法生成 subset；v0.3 切到 leaderboard-calibrated
- ✅ CLI: `--profile`, `--subset`, `eval-suite`, `estimate`, `watch`, `subset list/show/propose/calibrate/compare`
- ✅ Python API: `videvalkit.training.monitor` + `MonitorConfig` + `preview_prompts()`
- ✅ 6 个 anchored bench 全部交付 quick_v1 + standard_v1（共 12 个 JSON）
- ✅ profile 与 judge **正交**（profile 选 subset/frame，judge 选 VLM）
- ✅ `--profile full --judge paper` = paper-faithful 复现
- ✅ watch 用轮询，inotify 留 v0.3
- ✅ estimate 用写死 pricing + 文档免责
- ✅ subset 文件做 hash 校验记录在 timeline
- ✅ 不做 PyTorch/HF 训练框架直接集成（用户主动 `monitor.eval(...)`）
- ✅ v0.2 不开放用户自定义 profile（reproducibility 优先）；`--subset path/to/x.json` 是 escape hatch

---

## 14. 与其他设计文档的关系

| 文档 | 关系 |
|---|---|
| [`PRODUCT_DESIGN.md`](PRODUCT_DESIGN.md) §5 | 本文是**支柱 D**（Quick Eval & Training Monitor），与 A/B/C 并列；§6 路线图把 D 加进 v0.2 |
| [`JUDGE_SELECTION_DESIGN.md`](JUDGE_SELECTION_DESIGN.md) | Profile 里的 `judge` 字段引用 JUDGE doc 的三档（paper/default/registry-name）；`--profile X --judge Y` CLI 组合两个维度 |
| [`INTEGRATION_FRAMEWORK_DESIGN.md`](INTEGRATION_FRAMEWORK_DESIGN.md) | Manifest-driven benchmark 在 manifest 里**可选**声明 `subsets:` 字段；Track A 的简单 benchmark 也能享受 quick profile |
| `DEV_MANUAL.md` | §5.2 Module B 增加 subset/profile 字段说明；§17 新增章节 |
| `TEST_MANUAL.md` | 新增"每 bench 校准 ρ 表"，与 mean \|Δ\| 表并列 |

### 14.1 v0.2 工作量汇总（四支柱合并视图）

| 支柱 | 文档 | 工作量 |
|---|---|---|
| A (env / install) | DEV_MANUAL + doctor 增强 | 0.5 day |
| B (judge selection) | JUDGE_SELECTION_DESIGN | 2 day |
| C (integration framework) | INTEGRATION_FRAMEWORK_DESIGN | 6.5 day |
| **D (quick eval & training monitor)** | **本文** | **7.5 day** |
| **TOTAL v0.2** | | **≈ 16.5 person-days** |

> v0.2 时间线需要从 PRODUCT_DESIGN 原计划的 3 周扩展到 **5 周**（按单人节奏），或保持 3 周但需 ≥ 2 人并行（B/C 与 D 并行）。

---

—— end of design v0.1 ——
