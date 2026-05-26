# videogenevalkit — Product Design & Roadmap

| 字段 | 内容 |
|---|---|
| 版本 | v0.1 (draft) |
| 状态 | Master design doc — 与下属 design docs 共生维护 |
| 创建 | 2026-05-20 |
| 范围 | 顶层产品愿景 · 系统全景 · 三大设计支柱 · 路线图 · 跨支柱原则 |
| 目标读者 | 全角色：核心开发 · benchmark 集成者 · VLM 集成者 · 端到端评测用户 · 下游部署团队 · 论文复现者 |

> 这份文档是 **入口**。读完它能知道 videogenevalkit 是什么、要解决什么、目前到哪一步、下一步去哪。细节落在三份子文档：
> - [`DEV_MANUAL.md`](DEV_MANUAL.md) — 已实现的工程细节（v0.0.1 现状）
> - [`JUDGE_SELECTION_DESIGN.md`](JUDGE_SELECTION_DESIGN.md) — VLM judge 切换层（设计中）
> - [`INTEGRATION_FRAMEWORK_DESIGN.md`](INTEGRATION_FRAMEWORK_DESIGN.md) — benchmark + scorer 快速集成层（设计中）

---

## 1. 产品愿景

### 1.1 一句话定位

**`videogenevalkit` 是 text-to-video 模型评测的统一入口** —— 一行 CLI、一份 workspace、一套 schema，把 6+ 个分散在不同 upstream repo 的评测套件、8+ 个 VLM judge、5+ 个聚合算法整合到一起，并对外开放扩展。

### 1.2 三句话价值主张

1. **不重造轮子**：所有 benchmark 都是上游 paper repo 的 thin adapter，byte-for-byte 对齐官方 leaderboard（v0.0.1 已自报 mean |Δ| 在 0.005–0.012 区间）。
2. **不绑死后端**：VLM judge / 评分算法 / 聚合器都是注册表里的可替换组件。用本地 vLLM、用 Claude/Gemini/GPT、用 paper 原生模型，CLI 一个 flag 切换。
3. **不强制 fork**：扩展新 benchmark / 新 metric / 新 judge endpoint 不需要改 toolkit 源码 —— manifest YAML、pip entry_points、本地用户目录三种路径任选。

### 1.3 价值边界 —— 我们**不做**什么

| 不做 | 理由 |
|---|---|
| 视频生成本身 | `BaseT2VModel` 在抽象层预留但不实现；视频从外部进来 |
| 重写官方 metric 公式 | adapter 调上游包；我们只管 IO/调度/聚合 |
| 跨机分布式 | `EnvDispatcher` 单机；接口保留给未来 Ray/SSH |
| 评测 judge 自身（"谁更准"） | 留独立 `videvalkit judge-eval` 子项目 |
| 自动 paper→adapter 代码生成 | 不靠谱，留给上游作者 |

---

## 2. 用户与场景

### 2.1 五类目标用户

| # | 用户 | 核心诉求 | 接触的接口 |
|---|---|---|---|
| 1 | **模型研发者** | 训完新 T2V 模型，要对 6 个 benchmark 一次跑完拿可比数 | `videvalkit eval --bench X` × 6 + `aggregate` |
| 2 | **Benchmark 集成者** | 自己有评测算法 / paper，想接入工具链 | manifest YAML / `BaseBenchmark` 子类 / `videvalkit new bench` |
| 3 | **VLM 集成者** | 部署了私有 vLLM / 接了新云 API（Claude 4.7 / 国产模型） | `~/.config/videvalkit/judges.yaml` / `--judge-endpoint` |
| 4 | **论文复现者** | 拿这个 repo 的数字证明自己 paper 复现了 SOTA | `--judge paper` + `videvalkit compare-leaderboard` |
| 5 | **下游部署团队** | 在自己机房跑，资源 / 网络 / API 都不一样 | env tarball + judges.yaml + plugin loader |

### 2.2 三个高频场景

**场景 A — "我训了一个新模型，48 小时内拿到 6-bench 全分数"**
```
1. ssh GPU 机
2. 把 videos/ 放好
3. videvalkit eval --bench vbench / vbench2 / videobench / worldjen / worldscore / t2vcompbench (×6)
4. videvalkit aggregate → cross_benchmark.json
5. 截图发 Slack
```

**场景 B — "我有自己的评测 idea，1 小时内接入"**
```
1. videvalkit new bench my_idea --template manifest
2. 编辑 ~/.videvalkit/benchmarks/my_idea/manifest.yaml （填 prompts、dims、scorer 引用）
3. videvalkit validate bench my_idea     ← contract test 通过
4. videvalkit eval --bench my_idea ...   ← 完全融入主流程
```

**场景 C — "我要忠实复现 paper X 报告的数字"**
```
1. videvalkit fetch-checkpoints --bench X
2. videvalkit eval --bench X --judge paper ...   ← paper 原生 VLM
3. videvalkit compare-leaderboard --workspace ...
```

---

## 3. 系统全景

### 3.1 分层架构（v0.2 目标态）

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       L7. CLI / Python entrypoint                         │
│   videvalkit eval / aggregate / metric / new / validate / judges / doctor │
├──────────────────────────────────────────────────────────────────────────┤
│                       L6. Runner / Orchestration                          │
│         runner.run() · resolve_judge() · discover_all() · scheduler       │
├──────────────────────────────────────────────────────────────────────────┤
│                       L5. Three Unified Registries                        │
│   SUPPORTED_BENCHMARKS · SUPPORTED_JUDGES · SUPPORTED_METRICS ·           │
│   SUPPORTED_AGGREGATORS  ← 都是 lazy-merge (builtin + user + entry_points)│
├──────────────────────────────────────────────────────────────────────────┤
│                       L4. Plugin Discovery                                │
│   builtin (src/)  ·  pip entry_points  ·  ~/.videvalkit/  ·  $CWD/.videvalkit/ │
├──────────────────────────────────────────────────────────────────────────┤
│                       L3. Core Abstractions                               │
│   BaseBenchmark · BaseScorer · BaseAggregator · ManifestBenchmark         │
│   PromptItem · VideoSpec · RawResult · Summary · ScoreContext             │
├──────────────────────────────────────────────────────────────────────────┤
│                       L2. Adapters (横向并列)                             │
│   vbench · vbench2 · videobench · worldjen · worldscore · t2vcompbench    │
│   physics_iq · vbench_pp · v_reasonbench · <user plugins> · ...           │
├──────────────────────────────────────────────────────────────────────────┤
│                       L1. Infrastructure                                  │
│   Workspace · ApiCallLogger · FrameCache · Scheduler (env / GPU / HTTP)   │
│   conda env (envs/videvalkit.yaml + conda-pack tarball)                   │
├──────────────────────────────────────────────────────────────────────────┤
│                       L0. External                                        │
│   upstream paper repos · HF hub (checkpoints + smoke-data + env)          │
│   VLM endpoints (local vLLM / managed API)                                │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.2 一次评测的端到端调用链

```
user CLI: videvalkit eval --bench worldjen --judge paper ...
   │
   ▼
[L7] cli.eval_cmd
   │
   ▼
[L6] runner.run(benchmark, videos, ws, judge="paper", ...)
   │   ├─ resolve_judge("paper", benchmark="worldjen")  ──► JUDGE doc §3
   │   │     paper → paper_judge → "gemma-4-31b-local" → SUPPORTED_JUDGES[...]
   │   │
   │   ├─ discover_all()                                ──► INTEGRATION doc §4
   │   │     builtin + ~/.videvalkit/ + entry_points → SUPPORTED_BENCHMARKS
   │   │
   │   └─ scheduler.run_in_env(env="videvalkit", ...)
   │         │
   │         ▼
   │   [L2] WorldJenBenchmark.evaluate_and_aggregate(...)
   │         │
   │         ├─ list_prompts() → [PromptItem...]
   │         ├─ list_required_videos(prompts, models) → [VideoSpec...]
   │         ├─ evaluate(videos, judge=cfg) → [RawResult...]
   │         │     │  build_judge(cfg) → OpenAICompatibleVLMJudge
   │         │     │  judge.achat_with_frames(...)   ←──┐
   │         │     │                                     │
   │         │     │  api_log.append(...)                │
   │         │     │  frame_cache.get_or_extract(...)    │
   │         │     │                                     │
   │         │     ▼                                     │
   │         │  [L1] HTTPDispatcher / RateLimit  ────────┘
   │         │           │
   │         │           ▼
   │         │       VLM endpoint (vLLM @ :8003 or API)
   │         │
   │         └─ aggregate(raw, "phas") → Summary
   │
   ▼
results/summary/worldjen/{model}.json
api_logs/calls/google/gemma-4-31b-it/...
results/raw/worldjen/{model}/{dim}/{prompt_id}.json
```

---

## 4. 现状盘点（v0.0.1，2026-05-19 validated）

### 4.1 已交付能力（v0.0.1）+ v0.2 目标态对比

| 维度 | v0.0.1 数 | v0.2 目标 | 内容 |
|---|---:|---:|---|
| **Benchmark adapter** | 9 | 9 | 6 anchored + 3 stub（v0.3 做实 stub） |
| **VLM judge** | 8 | 8 + user yaml | 4 local vLLM + 4 managed API + 用户自定义无限扩展 |
| **Standalone metric** | 0 | **20** | 14 通用 (全 judge-free) + 6 专用 (含 artifact-diagnostic)（[`VIDEO_METRICS_DESIGN.md`](VIDEO_METRICS_DESIGN.md)）|
| **Judge-free path** | – | **17/20 metric · 4/9 bench** | `--no-judge` 一键过滤，没 VLM/API 也能跑大半 |
| **Capability tags** | 0 | **44 (10 顶 + 34 子)** | 按能力跨 bench 调用：`eval --capability motion`（[`CAPABILITY_TAGS_DESIGN.md`](CAPABILITY_TAGS_DESIGN.md)） |
| **Aggregator** | 5 | 5 | weighted_sum / vbench_weighted / vbench2_category / phas / bt |
| **Core abstraction** | 3+1 | **4+1** | + `BaseDistributionMetric`（与 `BaseScorer` 并列） |
| **Eval profile** | 0 | **3** | quick / standard / full（[`QUICK_EVAL_DESIGN.md`](QUICK_EVAL_DESIGN.md)）|
| **CLI 子命令** | 8 | **15+** | + judges / metric / refs / estimate / watch / subset / eval-suite |
| **Doc** | 4 | 9 | + JUDGE / INTEGRATION / QUICK_EVAL / VIDEO_METRICS / PRODUCT |

### 4.2 已验证的复现度（mean |Δ| vs 官方 leaderboard）

| Benchmark | Mean \|Δ\| | 备注 |
|---|---:|---|
| VBench v1 | **0.012** | HunyuanVideo, 16/16 dims |
| VBench-2.0 | **0.0055** | 18/18 dims, 4 byte-exact |
| T2V-CompBench | **0.0046** | 6/7 in-tol, paper-exact LLaVA-1.6-34B |
| Video-Bench | judge-substitution offset | static + alignment dims match |
| WorldJen | PHAS Δ -0.47 | decord-vs-cv2 frame variance |
| WorldScore | full pipeline | 10/10 dims wired |

### 4.3 已知 / 已声明但**未**交付

| 项目 | 状态 | 落地于 |
|---|---|---|
| README 提到的 `~/.config/videvalkit/judges.yaml` | 文档承诺，代码缺失 | JUDGE_SELECTION_DESIGN v0.1 |
| DEV_MANUAL §14 "Standalone Metrics Module (planned)" | 文档承诺，代码缺失 | INTEGRATION_FRAMEWORK_DESIGN §5 |
| DEV_MANUAL §16 "User-Configurable VLM Endpoints (planned)" | 文档承诺，代码缺失 | JUDGE_SELECTION_DESIGN §4 |
| 3 个 stub benchmark | 骨架在，dims 未填 | 路线图 v0.3 |
| `videvalkit compare-leaderboard` | README 提到，代码缺失 | 路线图 v0.3 |
| macOS / CPU subset | Linux + CUDA only | 路线图 v0.4 |

---

## 5. 四大设计支柱

> 四份子 design doc 各管一个维度，构成 v0.2 的全部交付物。<br/>
> 第五份（NPU 适配）当前 **Deferred**，不进短期路线图。

### 5.1 支柱 A：环境与一键安装（已交付，巩固期）

**现状**：conda-pack tarball + `pip install -e .` + `post_install.sh`（7 个 build-from-source）+ HF 拉取 smoke/ckpts。

**v0.2 改进点**：
- `doctor` 增强：插件加载状态 + judge 连通性 + 资源前置检查（`paper-llava-1.6-34b` 需要 70GB 显存的 fail-fast）
- env tarball 与 plugin 加载路径解耦验证（避免 `pip install -e .` 装在 conda env 外）
- macOS 上至少能 `videvalkit list / new / validate`（不能跑 eval 但能开发态用）

### 5.2 支柱 B：VLM Judge 切换（设计中 → v0.2 落地）

**详见 [`JUDGE_SELECTION_DESIGN.md`](JUDGE_SELECTION_DESIGN.md)**。要点：

```
每个 benchmark 注册表里声明三档 judge：
   paper_judge      ← 复现 paper 用（LLaVA-1.6-34B / GPT-4o / paper-Gemma）
   default_judge    ← 省事档（小模型 / 已 validate）
   recommended_judges[]  ← help & doc 展示

CLI 三种语法：
   --judge <name>                      ← registry 名（原有）
   --judge paper / default             ← 语义关键字（新）
   --judge-endpoint URL --judge-model M --judge-kind K  ← ad-hoc（新）

配置 5 层优先级：
   builtin → ~/.config/judges.yaml → $CWD/.videvalkit/judges.yaml → env → CLI
```

**工作量**：≈ 2 person-days

### 5.3 支柱 C：Benchmark + Metric 快速集成（设计中 → v0.2 落地）

**详见 [`INTEGRATION_FRAMEWORK_DESIGN.md`](INTEGRATION_FRAMEWORK_DESIGN.md)** + **[`VIDEO_METRICS_DESIGN.md`](VIDEO_METRICS_DESIGN.md)（子专题）**。要点：

```
双轨接入：
   Track A — manifest.yaml (80% 简单场景)
   Track B — Python BaseBenchmark 子类 (20% 复杂场景)

三层插件发现：
   builtin (src/)  →  pip entry_points  →  ~/.videvalkit/  →  $CWD/.videvalkit/

独立 Metric 层（v0.2 重点）：
   SUPPORTED_METRICS 注册表 + videvalkit metric CLI
   ★ v0.2 内置 19 个 metric（用户 2026-05-20 确认）：
      通用 T2V quality (14):
        Distribution (4):  fvd / vfid / kvd / clip-fvd*
        Alignment (2):     clip-score / viclip-score
        Frame perceptual:  aesthetic-quality / imaging-quality       (lift)
        Temporal (6):      motion-smoothness / temporal-flickering /
                           subject-consistency / background-consistency /
                           dynamic-degree / motion-magnitude         (lift)
      专用维度 (5):
        object-binding / spatial-relationship / numeracy /
        motion-accuracy (lift) + identity-preservation (new, ArcFace)
   ★ 双入口：eval --bench / metric --name 共享单一实现，位级一致

DEFERRED to v0.3：videvalkit new (脚手架) / validate (contract test)
```

**工作量**：≈ 11.3 person-days（M1 plugin loader 1d + M2 manifest 1.5d + M3 19 metrics 8.3d + M6 docs 0.5d）

### 5.4 支柱 D：Quick Eval & Training Monitor（设计中 → v0.2 落地）

**详见 [`QUICK_EVAL_DESIGN.md`](QUICK_EVAL_DESIGN.md)**。要点：

```
三档 eval profile：
   quick      ← 训练监控 / smoke / CI (5–10 min, ρ ≥ 0.85 vs full)
   standard   ← ablation / 迭代评测 (30–60 min, ρ ≥ 0.95)
   full       ← paper 复现 / leaderboard (数小时)

CLI：
   videvalkit eval --bench X --profile quick
   videvalkit eval-suite --bench A --bench B --profile quick    ← 多 bench
   videvalkit estimate --bench A --profile quick --judge X      ← 跑前算账
   videvalkit watch ...                                          ← 训练监控循环
   videvalkit subset propose/calibrate/show

Python API：
   videvalkit.training.monitor(...) + MonitorConfig
   支持 metrics= 字段，与支柱 C 的 19 个 metric 协同
```

**工作量**：≈ 7.5 person-days

### 5.5 横切层：Capability Tags（v0.2 末期加入，用户 2026-05-20 拉前）

**详见 [`CAPABILITY_TAGS_DESIGN.md`](CAPABILITY_TAGS_DESIGN.md)**。要点：

```
给 metric / bench dim 打能力 tag，44 个 controlled vocab（10 顶 + 34 子）：

   motion · visual_quality · text_alignment · object_fidelity · subject_consistency ·
   physical_plausibility · temporal_coherence · realism · compositional · style

CLI 新增第三入口：
   videvalkit eval --capability motion --videos X/
       → 跨 bench 跨 metric 跑所有 motion 类，去重 + min-max normalize 后聚合

互不替代 现有两入口：
   --bench X        ← paper-comparable 数
   --name Y         ← 单 metric 标量
   --capability Z   ← 能力画像 ★ 新增
```

**工作量**：≈ 3.1 person-days

### 5.6 四支柱 + 横切的相互依赖

```
支柱 A (env / install)
   ↓ 提供运行时基础
支柱 B (judge selection) ──────┐
                              │ 被 C 引用（manifest 里 default_judge/paper_judge）
支柱 C (integration + metrics) ←┘  ←─── 被 D 引用（profile 里 metrics 字段）
   ↓ 提供 20 个 metric                  
支柱 D (quick eval + training)
   ↓ 提供训练监控闭环
       ↓
   横切：Capability Tags  ←── metric / bench dim 上加 tags 视图
   ↓ 提供按能力聚合的第三入口
[ 端到端可扩展的评测平台 + 能力画像 ]
```

**关键决策**：B / C / D + 横切 Capability Tags 同时进入 v0.2。它们共用 plugin loader / 同样的 YAML schema 风格，分开做反而割裂。

---

## 6. 路线图

### 6.1 版本与时间线

| 版本 | 目标日期 | 主题 | 主要交付 |
|---|---|---|---|
| **v0.0.1** | 2026-05-13 ✓ | Anchored adapters | 6 benchmark · 8 judge · 5 aggregator · 1 env · 4 doc |
| **v0.1.0** | 2026-05-19 ✓ | Validation | mean \|Δ\| 全部公布；conda-pack tarball + 5-min quickstart |
| **v0.2.0** | 2026-07-10 | **Extensibility + Metrics + Capability tags** | 支柱 B / C / D 全量 + capability 横切；20 个 metric 双入口 + 44 个能力 tag；macOS dev 态可用 |
| **v0.3.0** | 2026-07-31 | Expansion | 3 stub benchmark 做实；国产 VLM 接入；2-3 个第三方插件 dogfood；per-dim judge override |
| **v0.4.0** | 2026-09-15 | Productionization | macOS / CPU subset；leaderboard 网站；`videvalkit compare-leaderboard` |
| **v1.0** | 2026-Q4 | Public release | 论文 / 博客 / colab notebook；judge-eval 子项目 first release |

### 6.2 v0.2 内部里程碑（用户 2026-05-20 调整后）

**v0.2 总工时 ≈ 29.7 person-days**（单人节奏 6–7 周；2 人并行 3–4 周）：

| 支柱 | 工作量 | 子项 |
|---|---:|---|
| A (env / install 巩固) | 0.5 d | doctor 增强 |
| B (judge selection) | 2 d | M0 |
| C (integration framework) | 11.3 d | M1 plugin loader (1d) + M2 manifest (1.5d) + **M3 20 metrics 双入口 (8.3d, 含 artifact-diagnostic + needs_judge 字段)** + M6 docs (0.5d)；M4/M5 scaffolding/validator **推 v0.3** |
| D (quick eval + training monitor) | 7.5 d | D1–D6 |
| **横切 · Capability Tags** | **3.1 d** | **T1 taxonomy + T2 注册表打 tag + T3 resolve + T4 CLI + T5 聚合器 + T6 docs** |
| **横切 · Review Protocol** | **1.5 d** | **REVIEW_PROTOCOL.md + PR template + CI 5 jobs + 3 check scripts + maintainer skill** |
| **needs_judge / --no-judge 字段** | 0.8 d | 已含入 C 的 M3 子项 |
| **TOTAL** | **29.7 d** | |

```
v0.2 6-week roadmap (单人节奏)：

Week 1: M0 Judge + M1 Plugin loader
Week 2-3: M3 Metrics (20 个 metric，12 lift + 7 new + artifact-diagnostic)
        + M2 Manifest benchmark（与 M3 并行）
Week 4: D1-D4 Quick eval profile + Subset + Estimate + Eval-suite
Week 5: D5-D6 Watch + Training API + 总集成测试
Week 6: T1-T6 Capability Tags + 横切（taxonomy / resolve / CLI / 聚合器） + macOS dev 态验证 + Docs

可拆 ~16 个独立 PR：
  - judge yaml / paper-alias / ad-hoc endpoint            (B)
  - needs_judge / --no-judge filter                       (B+C 横切)
  - plugin loader / manifest benchmark                    (C)
  - distribution metrics (FVD/VFID/KVD/CLIP-FVD)          (C-M3)
  - alignment metrics (CLIP/ViCLIP)                       (C-M3)
  - lift-out generic 8 个 (vbench × 7 + worldscore × 1)    (C-M3)
  - lift-out specialized 4 + ArcFace identity 1           (C-M3)
  - artifact-diagnostic (Artifact-Bench port)             (C-M3)
  - refs management + CLI                                 (C-M3)
  - eval profile + subset                                 (D)
  - estimate / eval-suite / watch                         (D)
  - training Python API                                   (D)
  - capability tags taxonomy + 注册表打 tag                (Capability 横切)
  - capabilities CLI + resolve + 聚合                      (Capability 横切)
  - review protocol (CI workflow + PR template + 3 scripts) (Review 横切) ★ 新
  - macOS dev 态 + doctor 增强                             (A)
  - docs 汇总                                             (across)
```

### 6.3 v0.3 候选清单（按价值排序）

| # | 项目 | 价值 | 工作量 |
|---|---|---|---|
| 1 | 3 stub benchmark 做实（physics_iq / vbench_pp / v_reasonbench）—— 作为 manifest dogfood | ★★★★★ | 3–5 day |
| 2 | **Artifact-Bench (experimental)** —— v0.3 加入完整 judge-eval 语义 bench；与国产 judge 接入联动 | ★★★★ | 3 day |
| 3 | **Scaffolding CLI** (`videvalkit new bench/metric/judge`) —— 从 v0.2 推延 | ★★★★ | 1 day |
| 4 | **Contract validator** (`videvalkit validate bench/metric`) —— 从 v0.2 推延 | ★★★ | 0.5 day |
| 5 | **更多 metric**：DOVER / MANIQA / BLIP2-Score / Inception-Score / Action-Recognition / OCR / Physics | ★★★★ | 3–4 day |
| 6 | 国产 VLM judge 接入（Doubao / Qwen3-Max / Hunyuan-VL）通过 user yaml example | ★★★★ | 0.5 day |
| 7 | `videvalkit compare-leaderboard` —— 自动 diff vs validation/expected/ | ★★★★ | 1 day |
| 8 | per-dim judge override（JUDGE doc 非目标升级） | ★★★ | 2 day |
| 5 | `videvalkit eval-suite` 多 bench 一次跑（JUDGE doc 推迟项） | ★★★ | 1 day |
| 6 | 第三方 benchmark 插件 dogfood（EvalCrafter / FETV / AIGCBench 任一） | ★★★ | 2–3 day |
| 7 | `videvalkit publish` 帮 local plugin 打 pip 包 | ★★ | 1 day |
| 8 | Resume / retry 优化（断点续跑 raw json 已有，但缺 CLI 命令） | ★★ | 0.5 day |

### 6.4 长期方向（v0.4+）

- **`videvalkit judge-eval` 子项目**：独立工具评估"judge X 在 dim Y 上比 judge Z 准多少"，配套人工标注 pipeline
- **Leaderboard 网站**：自动从 `validation/expected/` + 社区提交聚合（GitHub Pages + JSON 数据）
- **Colab quickstart**：T4/A100 一键 demo notebook（绕开 conda-pack 限制）
- **跨机分布式**：`EnvDispatcher` 接入 Ray / SSH，单机变多机
- **论文**：把整个工具链 + 复现度数据写成 paper（target: NeurIPS 2026 D&B 或 EMNLP）

---

## 7. 跨支柱设计原则

> 这些原则贯穿三个支柱，单独读任一支柱可能不显著，合在一起决定了 toolkit 的"长相"。

### 7.1 Registry-driven，不做 plugin loader DSL

四个注册表 (`SUPPORTED_BENCHMARKS / SUPPORTED_JUDGES / SUPPORTED_METRICS / SUPPORTED_AGGREGATORS`) 是 single source of truth。所有"加东西"的动作 = 在某个注册表里加 entry。**没有 YAML plugin manifest DSL，没有动态发现 magic**，只有"builtin dict + user yaml + entry_points"三个 lazy-merge 源。

### 7.2 Adapter-first，不重实现 upstream

每个 benchmark / 每个 paper-faithful judge 都是上游代码的 thin adapter。toolkit 只管：
- 输入归一化（toolkit 视频路径 ↔ upstream 期望路径，symlink staging）
- 输出归一化（每种 upstream JSON shape → 统一 `RawResult`）
- 跨 benchmark 编排（scheduler / api_log / frame_cache / workspace）
- 聚合（PHAS / BT / cross-bench z-score）

**永远不重写 paper 的 metric 公式**。

### 7.3 80/20 双轨

简单场景 YAML，复杂场景 Python。**不强迫所有用户用 manifest**（会让复杂场景变痛苦），**也不强迫所有用户写 Python**（会让简单场景门槛过高）。manifest 和 Python adapter 进同一注册表，runner 看不出区别。

### 7.4 三档默认显式化（paper / default / custom）

每个有 judge 的 benchmark 强制声明 `paper_judge` 和 `default_judge`，让"忠实复现 paper"与"省钱跑 demo"两条 lane 都是 first-class。**不允许把 paper 模型偷偷换成小模型而不告知用户**（这是 v0.0.1 的隐性问题）。

### 7.5 Workspace as single truth

一次评测的所有 input / output / log / cache 都在 `$WORKSPACE_ROOT/` 下。`results/raw/{bench}/{model}/{dim}/{prompt_id}.json` 是 resume primitive —— 存在就跳过。`api_logs/calls/{provider}/{model}/...` 是 replay primitive —— 离线可回放每一次 API 调用。

### 7.6 Minimal trusted surface

新集成者面对的接口越窄越好：
- `BaseBenchmark`：4 方法
- `BaseScorer`：1 方法 (`score(ctx)`)
- `BaseAggregator`：1 方法
- Manifest schema：≤ 12 个顶层字段

**任何接口扩张都要明确动机**。

### 7.7 Reproducibility is a contract, not a goal

`validation/expected/` 里存每个 benchmark 的官方 leaderboard JSON；CI 跑 `videvalkit compare-leaderboard`（v0.3 落地）每次都要在 tolerance 之内。**复现度是测试，不是承诺**。

### 7.8 Plugin-first，fork 是最后选项

用户扩展的所有正常路径都不需要改 toolkit 源码：
- 加 judge → `~/.config/videvalkit/judges.yaml`
- 加 benchmark → `~/.videvalkit/benchmarks/<name>/manifest.yaml` 或 pip 包
- 加 metric → 同上

Fork & PR 是给上游 anchored benchmark 的修正用的。

---

## 8. 成功度量

> 怎么知道这个产品做得好不好？

| 度量 | 当前 (v0.0.1) | v0.2 目标 | v0.3 目标 | v1.0 目标 |
|---|---:|---:|---:|---:|
| Anchored benchmark 数 | 6 | 6 | 9 (3 stub 做实) | 12+ |
| Total benchmark 数（含插件） | 6 | 6 + n_user | 10+ | 20+ |
| VLM judge 数 | 8 | 8 + n_user | 12+ | 20+ |
| 接入新 benchmark 时长（manifest 简单场景） | n/a | **< 1 hour** | < 30 min | < 15 min |
| 接入新 benchmark 时长（Python 复杂场景） | 1–3 day | 0.5–2 day | < 1 day | < 0.5 day |
| 接入新 metric 时长 | n/a (没 module) | **< 1 hour** | < 30 min | < 15 min |
| 接入新 judge endpoint 时长 | 改源码 + PR | **< 5 min**（yaml） | < 1 min | < 30 sec |
| Quickstart 安装时长 | ~30 min | ~30 min | ~20 min | < 10 min (colab) |
| Mean \|Δ\| vs leaderboard | 0.005–0.012 | 同 | 同 + 自动监控 | 同 + 网站 |
| 第三方贡献（plugin 仓库数） | 0 | 0 | 2–3 | 10+ |
| GitHub stars | – | – | – | 1k+ |

---

## 9. 风险与对策

| 风险 | 影响层 | 对策 |
|---|---|---|
| Upstream paper repo 版本漂移（如 VBench 升级 CUDA 11.8 → 12.4） | 支柱 A | per-bench env 模式保留；`CondaEnvDispatcher` 已就位；CI 跑 monthly probe |
| Manifest 表达力不够，用户大量退化到 Python | 支柱 C | v0.2 用 physics_iq + v_reasonbench 真实改造做表达力验证；不通过则补 schema |
| Paper-faithful judge 资源门槛吓退用户（70GB GPU） | 支柱 B | `--judge default` 永远可用作低成本路径；doctor 提前 fail-fast |
| 插件加载顺序让用户困惑 | 支柱 C | `doctor` 列每个 entry 最终来源；`list --verbose` 显示 source 列 |
| 三档 judge 让用户看不懂数字差异 | 支柱 B | TEST_MANUAL 每个 bench 加 "paper vs default 数值差异表" |
| 用户改了 `~/.videvalkit/`，但忘了，导致复现失败 | 跨支柱 | workspace summary 里记录所有 entry 来源 (provenance trail) |
| README 承诺的功能继续不落地（信誉损失） | 产品层 | **本次 v0.2 一次性把 README/DEV_MANUAL 里所有 "planned" 项落地或显式降级** |
| 用户期望 macOS 能跑完整评测 | 产品层 | 文档头部明确"Linux + CUDA"，macOS 仅 dev 态；v0.4 再扩 subset |

---

## 10. 与外部生态的关系

| 外部 | 当前关系 | 目标关系 |
|---|---|---|
| **Upstream paper repos**（VBench / WorldJen / T2V-CompBench / ...） | git clone + thin wrapper | 长期：上游接受我们提的 PR（修小 bug、改 entry_point） |
| **HuggingFace** | datasets/checkpoints/env-tarball 三个 dataset repo | 长期：托管 leaderboard JSON 与社区贡献的 manifest |
| **vLLM / SGLang / Ollama 生态** | 通过 OpenAI-compatible 协议天然支持 | 文档示例覆盖三种典型部署 |
| **管 API 厂商**（OpenAI / Anthropic / Google / 字节 / 阿里） | OpenAI-compatible 直连；Gemini/Claude 走 SDK | 国产模型在 v0.3 加 user yaml 示例 |
| **PyTorch / CUDA 生态** | 单 env 锁定 torch 2.3.1 + CUDA 12.1 | 跟进上游 paper 用什么版本，不主动升级 |
| **Conda / pip** | conda env + pip install -e . | 不引入第三种包管理器（poetry / uv 等） |
| **GitHub** | repo + issues + PR | v1.0 之后开始接受外部 PR；contributing guide 写明流程 |

---

## 11. 决策快照

> 评审本文时一次性确认（"同意"即采纳）。**v0.2 范围用户 2026-05-20 已确认**：

- ✅ 顶层定位：T2V 评测的**统一入口**，不做生成、不做分布式、不做 judge-eval（独立项目）
- ✅ 五类用户、三大场景如本文 §2
- ✅ **四大支柱 B + C + D 同时进入 v0.2**（A 巩固期；NPU 横切 Deferred）
- ✅ **v0.2 工作量 ≈ 29.7 person-days**（6–7 周单人 / 3–4 周 2 人并行），~16 个独立 PR
- ✅ **v0.2 重集成轻框架**：20 个 metric + lift-out 双入口 + 44 个 capability tag 为核心；scaffolding/validator 推 v0.3
- ✅ **强制 Review Protocol**：所有 PR 必走 3 层 gate (CI / 12 项自查 / peer 5 问)；详见 [`REVIEW_PROTOCOL.md`](REVIEW_PROTOCOL.md)
- ✅ **v0.2 不含**：PSNR/SSIM/LPIPS/FID-image (T2V 不适用) / NPU 适配 (Deferred)
- ✅ v0.3 优先级：3 stub 做实 > scaffolding/validator > 更多 metric > 国产 judge > compare-leaderboard
- ✅ 跨支柱原则 8 条（§7）作为代码 review 标准
- ✅ 成功度量以"接入时长"为核心 KPI（< 1 hour for simple bench/metric）
- ✅ macOS 完整支持推到 v0.4，v0.2 仅做 dev 态
- ✅ v1.0 (Q4 2026) 论文 + 网站 + colab + judge-eval 子项目同时落地
- ✅ 5 份子 design doc 与 DEV_MANUAL 共生维护，互相引用而不复制内容

---

## 12. 文档关系图

```
docs/
├── PRODUCT_DESIGN.md              ← 顶层 (本文，路线图 + 4 支柱 + 横切)
├── JUDGE_SELECTION_DESIGN.md      ← 支柱 B (~2 d)
├── INTEGRATION_FRAMEWORK_DESIGN.md ← 支柱 C 通用 (~3 d；scaffolding/validator 推 v0.3)
├── VIDEO_METRICS_DESIGN.md        ← 支柱 C 子专题 — 20 metric (~8.3 d，计入 C)
├── QUICK_EVAL_DESIGN.md           ← 支柱 D (~7.5 d)
├── CAPABILITY_TAGS_DESIGN.md      ← 横切层 — 44 个能力 tag (~3.1 d)
├── REVIEW_PROTOCOL.md             ← 横切层 — quality gate (~1.5 d) ★ 2026-05-20
├── NPU_ADAPTATION_DESIGN.md       ← 横切，Deferred (~8.5 d，未排期)
└── DEV_MANUAL.md                  ← v0.0.1 现状

.claude/skills/
└── videvalkit-maintainer/         ★ 我的行为 skill 化（intake / review / 协议执行）

层级关系：

                    ┌────────────────────────┐
                    │  README.md             │
                    │  (面向新用户的入口)     │
                    └────────────┬───────────┘
                                 │ 链接到
                                 ▼
                    ┌────────────────────────┐
                    │  PRODUCT_DESIGN.md     │  ←── 你在这里
                    │  (顶层产品 / 路线图)    │
                    └─────┬──────────┬───────┘
                          │          │
              ┌───────────┘          └────────────┐
              ▼                                   ▼
   ┌────────────────────────┐       ┌───────────────────────────────┐
   │ JUDGE_SELECTION_       │       │ INTEGRATION_FRAMEWORK_        │
   │ DESIGN.md              │       │ DESIGN.md                     │
   │ (支柱 B: judge 切换)    │       │ (支柱 C: bench/scorer 接入)   │
   └──────────┬─────────────┘       └────────────┬──────────────────┘
              │                                  │
              └──────────────┬───────────────────┘
                             ▼
                  ┌────────────────────────┐
                  │  DEV_MANUAL.md         │
                  │  (已实现的工程细节)     │
                  └────────────┬───────────┘
                               │
              ┌────────────────┼─────────────────┐
              ▼                ▼                 ▼
   ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
   │ USER_MANUAL_     │ │ USER_MANUAL_     │ │ TEST_MANUAL.md   │
   │ en.md            │ │ cn.md            │ │ (复现度验证)     │
   └──────────────────┘ └──────────────────┘ └──────────────────┘
```

—— end of product design v0.1 ——
