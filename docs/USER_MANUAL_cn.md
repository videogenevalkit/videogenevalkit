# videvalkit 用户手册

| 字段 | 值 |
|---|---|
| 标题 | videvalkit 用户手册（中文版） |
| Python 包名 | `videvalkit` |
| GitHub 仓库 | `videogenevalkit` |
| 读者 | 运行文本到视频（text-to-video, T2V）评测的研究人员、集成工程师及下游团队 |
| 配套文档 | [DEV_MANUAL.md](DEV_MANUAL.md)（架构）- [TEST_MANUAL.md](TEST_MANUAL.md)（论文 Delta 复现验证） |

本手册面向最终用户，介绍 `videvalkit` 的能力、安装方法、数据与权重准备、六个锚定基准的运行方式、judge（评判模型）配置以及 GPU 调度策略。内部架构请参考 DEV_MANUAL；论文 Delta 复现表格请参考 TEST_MANUAL。

本文档为英文版的忠实翻译，目录结构完全一致。所有命令、代码及配置文件示例保持英文原样；技术名词在首次出现时给出中文简释。

---

## 目录

1. [简介](#1-简介)
2. [系统要求](#2-系统要求)
3. [安装](#3-安装)
4. [数据与权重准备](#4-数据与权重准备)
5. [第一次评测](#5-第一次评测)
6. [分基准操作指南](#6-分基准操作指南)
7. [配置评分器（VLM/LLM judges）](#7-配置评分器vlmllm-judges)
8. [配置 GPU](#8-配置-gpu)
9. [跨基准聚合](#9-跨基准聚合)
10. [独立度量](#10-独立度量)
11. [生成你自己的视频](#11-生成你自己的视频)
12. [自定义 prompt（自动标注器）](#12-自定义-prompt自动标注器)
13. [故障排查与常见问题](#13-故障排查与常见问题)
14. [许可证与引用](#14-许可证与引用)

---

## 1. 简介

`videvalkit` 是面向文本到视频生成模型的统一评测工具集。一份配置、一个入口、六个基准（benchmark）。通过同一个命令行接口，你可以在 VBench v1、VBench-2.0、Video-Bench、WorldJen、WorldScore 与 T2V-CompBench 上对模型打分，并在同一个工作区（workspace）内完成跨基准对比。

工具集已交付六个**锚定的**基准适配器（adapter），均处于生产可用状态：

| 适配器 | 上游 | 评测维度 |
|---|---|---|
| `vbench` | Vchitect/VBench（CVPR 2024） | 16 维：7 维 quality + 9 维 semantic |
| `vbench2` | Vchitect/VBench-2.0（CVPR 2025） | 18 维，覆盖 5 个类别（Creativity 创造性、Commonsense 常识、Controllability 可控性、Human Fidelity 人体保真、Physics 物理） |
| `videobench` | Han 等，Video-Bench（CVPR 2025） | 9 维：4 维 alignment + 4 维 静态/动态 quality + video-text consistency |
| `worldjen` | moonmath-ai/WorldJen | 16 维，按 motion_stability、logic_physics、instruction_adherence、aesthetic_quality 四个宏类分组；PHAS 聚合器 |
| `worldscore` | Duan 等，WorldScore | 10 维：7 静态 + 3 动态；纯 CV 评分 |
| `t2vcompbench` | Sun 等，T2V-CompBench V2（ECCV 2024） | 7 维组合性维度；LLaVA-1.6-34B + GD/SAM/DOT |

另有 3 个补充适配器 stub（占位实现）将于后续版本中交付：Physics-IQ、VBench++、V-ReasonBench。它们已列入 `videvalkit list benchmarks --include-stubs`，但尚未达到生产可用状态。

工具集**不**重写上游评分公式。每个适配器都按字节对齐（byte-for-byte）地调用上游包，仅在其上叠加 IO、调度（scheduling）、注册表（registry）、judge 接线与统一输出格式。TEST_MANUAL.md 中的验证清扫记录了与公开排行榜的 mean |Delta|：VBench v1 = 0.012，VBench-2.0 = 0.006，T2V-CompBench = 0.013（7 个维度中 6 个落在 +/-0.020 内）。各基准的复现表与分维 Delta 见 TEST_MANUAL 第 4 节。

---

## 2. 系统要求

| 要求项 | 值 |
|---|---|
| 操作系统 | Linux x86_64（已在 CentOS/RHEL/Ubuntu 上验证） |
| CUDA 驱动 | 12.1 或更高 |
| GPU | 至少 1 块 NVIDIA GPU，显存 >= 24 GB。运行 T2V-CompBench 的 paper-mode（LLaVA-1.6-34B）则要求 >= 80 GB |
| 内存 | >= 64 GB |
| 磁盘 | 完整 smoke 数据 + 全部六个基准的 checkpoint 至少需要 200 GB 可用空间 |
| conda | miniforge 或 miniconda，建议使用较新版本 |
| 网络 | 能够 HTTPS 出站访问 huggingface.co 与 github.com |

24 GB 的下限可覆盖 WorldJen、Video-Bench、VBench v1、VBench-2.0 与 toolkit 模式下的 WorldScore。T2V-CompBench 的 paper-mode（`--mode upstream`，使用打包的 LLaVA-1.6-34B MLLM）需要 80 GB 级别的 GPU（A100/H100）。若不具备，可改用 `--mode toolkit` 并指定较小的 VLM judge，但请接受 DEV_MANUAL 第 15.3.2 节所述的 paper-Delta 警告。

仅 PSNR、SSIM 等成对度量支持纯 CPU 运行，所有基准都需要 GPU。

---

## 3. 安装

工具集以预打包 env tarball（环境压缩包）的形式分发，即生成 `TEST_MANUAL.md` 全部验证结果的同一个环境，按字节快照打包。下载、解压、运行即可。

```bash
# 1. 克隆仓库（取源代码 + 脚本 + 文档）
git clone https://github.com/videogenevalkit/videogenevalkit.git
cd videogenevalkit

# 2. 下载 env tarball（~7.6 GB）
hf download videogenevalkit/env-tarball videvalkit-env.tar.gz \
    --local-dir /tmp

# 3. 解压到持久目录（解压后约 15 GB）
sudo mkdir -p /opt/videvalkit-env
sudo tar xzf /tmp/videvalkit-env.tar.gz -C /opt/videvalkit-env
sudo chown -R $USER /opt/videvalkit-env

# 4. 激活环境并修正内部绝对路径
source /opt/videvalkit-env/bin/activate
conda-unpack

# 5. 在本仓库上以可编辑方式安装工具集源码
pip install --no-deps -e .

# 6. 安装预打包未包含的 7 个 build-from-source / git-only 依赖
bash scripts/post_install.sh
# 如果不需要 WorldScore camera_control 或 VBench-2.0 Human_Anatomy：
# bash scripts/post_install.sh --minimal   (~10 分钟更快)
```

Tarball 内含：
- Python 3.10 + torch 2.3.1+cu121
- 约 350 个固定版本的依赖（transformers 4.51.3、mmcv 2.2.0、decord、opencv、pyiqa、openai-clip、timm、xformers、triton、bitsandbytes、accelerate、peft、ms-swift、qwen-vl-utils 等，以及所有传递依赖的已验证组合）
- 模型 checkpoint（检查点）**不打包** — 首次运行 benchmark 时按需获取（全部约 125 GB）。可用 `videvalkit fetch-checkpoints --bench <name>` 按 benchmark 拉取

`scripts/post_install.sh` 会额外安装：`detectron2`、`SAM-2`、`GroundingDINO`、`segment-anything`、`lietorch`、`droid_backends`（经 DROID-SLAM 克隆构建）和 spaCy 模型 `en_core_web_sm`。这些是 git / 源码构建包，未纳入 tarball 以保证 tarball 大小可控；它们会在 post-install 阶段针对本地 CUDA 重新编译。

### 3.1 验证环境

```bash
videvalkit doctor
```

`doctor` 会检查：环境激活、CUDA 可见性、GPU 显存、HF 缓存位置、smoke 数据是否就绪，以及任何已配置 judge endpoint（端点）的可达性。正常输出全部为绿色检查；任何红色项都会阻塞依赖它的评测。

---

## 4. 数据与权重准备

### 4.1 HuggingFace 鉴权（可选）

工具集获取的多数 checkpoint 都是公开的。少数上游发布的参考视频集是 gated（需授权），需要一个 HuggingFace read-scope token（部分自动批准，部分需要人工审核）。注册方法：

```bash
hf auth login
# 粘贴一个 read-scope token，来源： https://huggingface.co/settings/tokens
```

如果你处在企业代理之后或正在使用 HuggingFace 镜像，在拉取前设置 `HF_ENDPOINT`：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 4.2 拉取 smoke 数据

工具集在同一个 HF 仓库（`videogenevalkit/smoke-data`）中托管了每个基准的代表性 prompt 子集与论文发布的参考视频。可一次性拉取全部六个，也可只取其一：

```bash
# 拉取全部 6 个基准的 smoke 数据（约 3 GB）
videvalkit fetch-smoke-data

# 或仅拉取单个基准
videvalkit fetch-smoke-data --bench worldjen
```

smoke 数据默认落盘到 `~/.cache/videvalkit/smoke-data/<bench>/`。每个基准的 smoke 子集即为论文发布的官方视频集或其代表性子集（每基准约 50-200 个视频），足以端到端验证整条流水线，而无需多日 GPU 任务。

### 4.3 拉取 checkpoints

预训练 checkpoint 按基准拉取。各基准的体量（完整的逐维（per-dim）模型依赖映射见 DEV_MANUAL 第 15.3 节）：

| Bench | 体量 | 说明 |
|---|---:|---|
| `vbench` | 7.7 GB | DINO、CLIP、AMT、RAFT、MUSIQ、GRiT、UMT、tag2text、ViCLIP |
| `vbench2` | 23 GB | LLaVA-Video-7B + Qwen2.5-7B + Qwen2.5-VL-3B + CV 栈 |
| `videobench` | 0 GB（自有） | 仅数据集；全部 9 维都使用你配置的 VLM judge |
| `worldjen` | 16 GB | Gemma-4-31B + Qwen2.5-7B（打包到 `hf-models/`） |
| `worldscore` | 6.3 GB | DROID-SLAM、SEA-RAFT、VFIMamba、SAM2、GroundingDINO、LAION |
| `t2vcompbench` | 72 GB | LLaVA-1.6-34B（68 GB）+ GD + SAM-H + Depth-Anything V1 + DOT |

```bash
# 拉取单个基准的 checkpoint
videvalkit fetch-checkpoints --bench worldscore

# 拉取 t2vcompbench 但跳过 68 GB 的 LLaVA-1.6-34B MLLM 栈
videvalkit fetch-checkpoints --bench t2vcompbench --skip-mllm-upstream

# 拉取全部（约 110 GB）
videvalkit fetch-checkpoints --all

# 预览将下载哪些文件、各自多大，不真正下载
videvalkit fetch-checkpoints --bench t2vcompbench --dry-run
```

注：`LLaVA-1.6-34B` 已打包但属可选。若 GPU 显存不足 80 GB，请加 `--skip-mllm-upstream`，并在 toolkit 模式下用较小 VLM judge 运行 T2V-CompBench。

Checkpoint 默认落盘到 `~/.cache/videvalkit/checkpoints/<bench>/`。可用 `VIDEVALKIT_CHECKPOINT_ROOT` 覆盖。下载支持断点续传。

---

## 5. 第一次评测

下面给出在 WorldJen 上端到端 smoke 运行的规范示例。WorldJen 是最稳妥的入门基准：50 条 prompt、每条 prompt 1 个视频、单 GPU 配合本地 Gemma judge 的整体墙钟时间约 1 小时。smoke 数据中已附带 `fal-ai_kling-video_v2.6_pro_text-to-video`（即 Kling v2.6）的视频。

```bash
videvalkit eval --bench worldjen \
  --videos ~/.cache/videvalkit/smoke-data/worldjen/videos/fal-ai_kling-video_v2.6_pro_text-to-video \
  --workspace runs/first \
  --judge gemma-4-31b-local

videvalkit aggregate --workspace runs/first

cat runs/first/results/summary/worldjen/Kling.json
```

第一条命令以本地 Gemma-4-31B vLLM judge（端口 8003）跑 WorldJen 适配器。WorldJen 分两阶段：阶段 A 用 LLM 生成逐 prompt 的 VQA 问题；阶段 B 用 VLM 在采样帧上作答。smoke 数据中已附带预构建的 `vqa_questions_50prompts.jsonl`，默认情况下阶段 A 会被跳过；如需从头跑阶段 A，请参考第 7 节。

`Summary` JSON 顶层有三个组：

```json
{
  "benchmark": "worldjen",
  "model": "Kling",
  "n_videos": 50,
  "headline": {"metric": "phas", "score": 3.6561},
  "per_dimension": {
    "subject_consistency": 3.782,
    "scene_consistency":   3.718,
    "motion_smoothness":   3.134,
    "...": "..."
  },
  "overall": {
    "PHAS": 3.6561,
    "unweighted_dim_mean": 3.577,
    "n_records": 800
  },
  "meta": {
    "toolkit_commit": "abcd1234",
    "upstream_pkg":   "worldjen==in-tree",
    "judge":          "openai_compatible:google/gemma-4-31b-it",
    "ckpt_checksums": {"...": "..."},
    "runtime":        {"python": "3.10.13", "torch": "2.3.1+cu121"},
    "scorers_used":   {"default": "gemma-4-31b-local"}
  }
}
```

`per_dimension` 是 16 个 WorldJen 维度的分数（1-5 分制）。`overall.PHAS` 是头条分数（带论文调参权重的 PHAS 聚合器）。`meta.scorers_used` 记录实际所用 judge，便于下游工具识别 judge 已被替换的运行。

参照 TEST_MANUAL 第 4.3 节：本次 Kling smoke 运行 PHAS ~3.66，对比论文报告的 Gemma-judge 头条 4.12（Delta -0.47）。差距源自 decord 与 cv2 的取帧差异以及 Gemma 采样配置不同；流水线的正确性已在逐维层面验证。

`aggregate` 步骤写入 `runs/first/results/leaderboard/cross_benchmark.json`（在工作区中含多个基准时尤为有用；单一基准时它只是把该基准的逐模型头条卷起）。

---

## 6. 分基准操作指南

下面每个小节都是完整的可复制粘贴运行示例：拉取 smoke 数据、从 HuggingFace `videogenevalkit/checkpoints` 取出对应 checkpoint、暂存 3-5 个样例视频、执行评测、看分数。所有示例都于 2026-05-19 完成了端到端验证；下文的预期分数即为当次运行（每基准 3 个样例视频）的实际结果。

**6 个基准速查表：**

| 节号 | Bench / 维度 | 评分器 | Checkpoint（来自 `videogenevalkit/checkpoints`） | 用时（3 视频） | 显存 | 预期分数 |
|---|---|---|---|---:|---:|---|
| 6.4 | `worldjen` / 16 维 | Gemma-4-31B vLLM | — 无本地权重 | ~5 min | 0 GB | overall ≈ 3.2 / 5 |
| 6.1 | `vbench` / `subject_consistency` | DINO ViT-B/16 | `vbench/pretrained/dino_model/`（343 MB） | ~30 s | 2 GB | ≈ 0.92 |
| 6.2 | `vbench2` / `Camera_Motion` | CoTracker3（内置） | `vbench2/third_party/cotracker/`（204 MB） | ~1 min | 4 GB | ≈ 0.67 |
| 6.3 | `videobench` / `action_consistency` | Gemma-4-31B vLLM | — 无本地权重 | ~2 min | 0 GB | ≈ 2.0（1-5 原始分） |
| 6.5 | `worldscore` / `motion_magnitude` | SEA-RAFT | `worldscore/Tartan-C-T-TSKH-*` + `raft-things`（150 MB） | ~2 min | 4 GB | ≈ 56.4（×100） |
| 6.6 | `t2vcompbench` / `action_binding`（paper-mode） | LLaVA-1.6-34B | `hf-models/liuhaotian/llava-v1.6-34b/`（68 GB） | LLaVA 加载后 ~5 min | 70 GB | raw 7.22 → norm 0.69 |

运行间方差：CV 类维度（DINO、CoTracker3、SEA-RAFT、GroundingDINO）约 ±0.05；VLM judge 类维度（Gemma-4-31B、LLaVA-1.6-34B）约 ±0.15（temperature=0.2 采样所致）。各维容忍带详见 [`TEST_MANUAL.md`](TEST_MANUAL.md) §3。

**跑超过 3 个视频。** 没有 `--limit` 参数 —— `videvalkit eval` 会对 `--videos` 目录下找到的*每一个*视频打分。下文各示例中的「3 个视频」上限完全来自暂存步骤里的 `| head -3` 过滤。要跑全集，可以去掉 `| head -3`（软链全部视频），或者直接把 `--videos` 指向源目录、跳过暂存步骤。注意 `fetch-smoke-data` 只拉取一个**子样本**（HF 数据集 `videogenevalkit/smoke-data` —— 例如 WorldJen 50 个视频、其余基准每维约 3 个）；去掉 `| head -3` 是跑完该*子样本*中的所有视频，而非完整的官方基准全集。要跑完整全集，需用你自己的模型在各基准的官方 prompt 列表上生成视频（`videvalkit fetch-upstream --bench <name>` 可拉取这些 prompt 清单）。

各维定义与设计源材料：DEV_MANUAL 第 15.3.1 节（逐维依赖）与 TEST_MANUAL 第 4 节（验证结果）。

### 6.1 VBench v1

**概览。** 16 维，7 quality + 9 semantic。所有维度都是纯 CV（无需 VLM judge）。各维默认评分器：`subject_consistency` 用 DINO ViT-B/16，`background_consistency` 用 CLIP ViT-B/32，`motion_smoothness` 用 AMT-S，`dynamic_degree` 用 RAFT，`imaging_quality` 用 MUSIQ，`object_class`/`color`/`multiple_objects`/`spatial_relationship` 用 GRiT，`human_action` 用 UMT，`scene` 用 tag2text，`appearance_style`/`temporal_style`/`overall_consistency` 用 ViCLIP，`aesthetic_quality` 用 LAION-aesthetic + CLIP ViT-L/14。聚合器为 `vbench_weighted`：`Total = 0.54*Quality + 0.46*Semantic`。

**运行示例：在 3 个 HunyuanVideo 视频上跑 `subject_consistency`（~30 s, 2 GB 显存）**

```bash
# 1. 取 smoke 视频 + DINO ViT-B/16 权重 + VBench prompt 注册表
videvalkit fetch-smoke-data  --bench vbench
videvalkit fetch-checkpoints --bench vbench

# 2. 暂存 3 个 HunyuanVideo 样例视频
mkdir -p ~/runs/vbench/videos/HunyuanVideo
ls ~/.cache/videvalkit/smoke-data/vbench/videos/HunyuanVideo/*.mp4 \
  | head -3 | xargs -I{} ln -sf {} ~/runs/vbench/videos/HunyuanVideo/

# 3. 评测
videvalkit eval --bench vbench \
    --videos ~/runs/vbench/videos \
    --workspace ~/runs/vbench/ws \
    --models HunyuanVideo \
    --dimensions subject_consistency \
    --prompts-file ~/.cache/videvalkit/smoke-data/vbench/prompts.jsonl
```

**预期：** `per_dimension.subject_consistency ≈ 0.92`（每视频 8 帧的 DINO 相邻帧 cosine 求均值 × 3 视频）。运行间方差约 ±0.02。

**Checkpoint：** `vbench/pretrained/dino_model/dino_vitbase16_pretrain.pth`（343 MB）。等价的直接 API 调用：`huggingface_hub.snapshot_download(repo_id="videogenevalkit/checkpoints", repo_type="dataset", allow_patterns=["vbench/pretrained/dino_model/*", "vbench/VBench_full_info.json"])`。

**注意事项：** `dynamic_degree` 是噪声最大的一维（RAFT 在 GPU 上的非确定性），该维容忍带需放宽至 ±0.025。`human_action` 依赖 YOLOv5x 权重；请用 `checksums.json` 校验 SHA-256。对于依赖 prompt 的维度（`object_class`、`color`、`spatial_relationship`、`scene`、`human_action`、`multiple_objects`），自定义 prompt 必须携带 `auxiliary_info` 标签；若无，请使用自动标注器（见第 12 节）。

**验证：** HunyuanVideo 全集清扫，16/16 维都落在 ±0.025 内；与 HF 排行榜的 mean |Δ| 为 0.012。

### 6.2 VBench-2.0

**概览。** 18 维，分 5 个类别（Creativity、Commonsense、Controllability、Human Fidelity、Physics）。12 个推理维使用 LLaVA-Video-7B-Qwen2 作为 VLM 评分器；其中 5 维额外引入 Qwen2.5-7B-Instruct 作为 LLM judge。其余 6 维使用打包的 CV 栈：`Camera_Motion` 与 `Multi-View_Consistency` 用 CoTracker3；`Diversity` 用 VGG-19；`Human_Anatomy` 用打包的 ViTDetector；`Human_Identity` 用 ArcFace + RetinaFace；`Instance_Preservation` 用 Qwen2.5-VL-3B（经 ms-swift）。聚合器为 `vbench2_category`：5 个类别均值再算术平均得 `Overall`。

**运行示例：在 3 个 HunyuanVideo 视频上跑 `Camera_Motion`（~1 min, 4 GB 显存）**

```bash
# 1. 取 smoke 视频 + CoTracker3 权重 + VBench-2.0 prompt 注册表
videvalkit fetch-smoke-data  --bench vbench2
videvalkit fetch-checkpoints --bench vbench2

# 2. 暂存 3 个 Camera_Motion 标签的 HunyuanVideo 视频
mkdir -p ~/runs/vbench2/videos/HunyuanVideo/Camera_Motion
ls ~/.cache/videvalkit/smoke-data/vbench2/videos/HunyuanVideo/Camera_Motion/*.mp4 \
  | head -3 | xargs -I{} ln -sf {} ~/runs/vbench2/videos/HunyuanVideo/Camera_Motion/

# 3. 评测
videvalkit eval --bench vbench2 \
    --videos ~/runs/vbench2/videos \
    --workspace ~/runs/vbench2/ws \
    --models HunyuanVideo \
    --dimensions Camera_Motion \
    --extra-kwarg 'mode="vbench2_standard"'
```

**预期：** `per_dimension.Camera_Motion ≈ 0.67`（每个视频的 0/1 二值判定再求均值，3 个视频可能落在 0、0.33、0.67、1.0）。CoTracker3 推理无 temperature，逐次结果一致。

**Checkpoint：** `vbench2/third_party/cotracker/cotracker2.pth`（204 MB）；等价的 `allow_patterns=["vbench2/third_party/cotracker/*", "vbench2/VBench2_full_info.json"]`。

**注意事项：** `Human_Anatomy` 与 `Human_Identity` **不可**替换为其他 VLM（打包的检测器）。`Diversity` 至少需要每条 prompt 2 个 seed。HunyuanVideo 全集清扫的 mean |Δ| 为 0.0055（18/18 维），前提是已应用 cv2 顺序读帧修复与 `-1` 哨兵过滤。默认 judge 为 `local-llava-video-7b`；可通过 `--scorer-vlm gemma-4-31b-local` 替换为更强的推理模型（paper-Δ 容忍带会变宽，参见第 7 节）。

**验证：** HunyuanVideo 全集清扫，18/18 维都落在 ±0.025 内；与 HF 排行榜（2025-03-28 行）的 mean |Δ| 为 0.0055。

### 6.3 Video-Bench

**概览。** 9 维：4 维 alignment（`video_text_consistency`、`object_class_consistency`、`color_consistency`、`action_consistency`、`scene_consistency`），1-3 分制；4 维 quality（`imaging_quality`、`aesthetic_quality`、`temporal_consistency`、`motion_effects`），1-5 分制。9 维共用**同一个** VLM judge——这是按模型数计算最简单的基准。论文使用 GPT-4o（`gpt-4o-2024-08-06`）；工具集默认注册的是 `gpt-4o-2024-11-20`。聚合器为 `videobench_per_dim`：在链式查询（chain-of-query）响应上做算术平均。

**运行示例：在 3 个 CogVideoX-5B 视频上跑 `action_consistency`（~2 min，Gemma judge，无本地权重）**

```bash
# 1. 取 smoke 视频 + prompts.jsonl（无 checkpoint —— VLM judge 负责评分）
videvalkit fetch-smoke-data --bench videobench

# 2. 暂存 3 个 action_consistency 标签的视频
mkdir -p ~/runs/videobench/videos/cogvideox5b
ls ~/.cache/videvalkit/smoke-data/videobench/videos/cogvideox5b/action_consistency/*.mp4 \
  | head -3 | xargs -I{} ln -sf {} ~/runs/videobench/videos/cogvideox5b/

# 3. 评测（Gemma-4-31B vLLM judge 在 localhost:8003）
videvalkit eval --bench videobench \
    --videos ~/runs/videobench/videos \
    --workspace ~/runs/videobench/ws \
    --models cogvideox5b \
    --judge gemma-4-31b-local \
    --dimensions action_consistency \
    --prompts-file ~/.cache/videvalkit/smoke-data/videobench/prompts.jsonl
```

**预期：** `per_dimension.action_consistency ≈ 2.0`（原始 1-5 分）。Gemma temperature=0.2 采样导致逐次方差约 ±0.5。

**Checkpoint：** 无 —— 评分器就是 VLM endpoint。改用 GPT-4o：`--judge gpt-4o`，并设置 `OPENAI_API_KEY`；如需 paper-exact 的 `gpt-4o-2024-08-06` 快照，请在 `~/.config/videvalkit/judges.yaml` 中注册（参见第 7 节）。

**注意事项：** GPT-4o 快照存在漂移；如要复现论文，请新建 `gpt-4o-2024-08-06` 条目并使用 `--judge gpt-4o-2024-08-06`。用 Gemma 替代 GPT-4o 会在 1-5 分制下让动态质量维度漂移 ±2.0 点（`temporal_consistency` 与 `motion_effects` 在 Gemma 下呈双峰分布）。静态与对齐维度在 Gemma 下与论文相差不超过 ±0.2。

### 6.4 WorldJen

**概览。** 16 维，分 4 个宏类（motion_stability、logic_physics、instruction_adherence、aesthetic_quality）。两阶段：阶段 A 用 LLM 生成 VQA 问题（默认 `qwen3-32b-local`，端口 8004）；阶段 B 用 VLM 作答（默认 `gemma-4-31b-local`，端口 8003）。聚合器为 `phas`：分维均值的加权和减去方差项，权重为论文校准值。

**运行示例：在 3 个 Kling 视频上跑全部 16 维（~5 min，Gemma judge，无本地权重）**

```bash
# 1. 取 worldjen smoke（50 个 Kling 视频 + prompts.jsonl + vqa.jsonl）
videvalkit fetch-smoke-data --bench worldjen

# 2. 暂存 3 个 Kling 视频
mkdir -p ~/runs/worldjen/videos/Kling
ls ~/.cache/videvalkit/smoke-data/worldjen/videos/fal-ai_kling-video_v2.6_pro_text-to-video/*.mp4 \
  | head -3 | xargs -I{} ln -sf {} ~/runs/worldjen/videos/Kling/

# 3. 评测（约 250 次 Gemma 调用：阶段 A VQA 生成 + 阶段 B 作答）
videvalkit eval --bench worldjen \
    --videos ~/runs/worldjen/videos \
    --workspace ~/runs/worldjen/ws \
    --models Kling \
    --judge gemma-4-31b-local \
    --aggregator phas
```

**预期：** `~/runs/worldjen/ws/results/summary/worldjen/Kling.json`：

```
overall ≈ 3.2 / 5  （在 3 条 prompt 上对 16 维均值再求均值）
per_category:
  instruction_adherence  ≈ 3.78    (Kling 表现最强)
  aesthetic_quality      ≈ 3.50
  motion_stability       ≈ 3.20
  logic_physics          ≈ 2.58    (Kling 表现最弱)
```

逐维明细（3 条样本 prompt 的均值）：

| 维度 | 分数 | 维度 | 分数 |
|---|---:|---|---:|
| `semantic_adherence`     | 4.47 | `composition_framing` | 3.60 |
| `semantic_drift`         | 4.07 | `lighting_volumetric` | 3.60 |
| `color_harmony`          | 3.83 | `subject_consistency` | 3.43 |
| `scene_consistency`      | 3.73 | `structural_gestalt`  | 2.97 |
| `temporal_flickering`    | 3.73 | `spatial_relationship`| 2.80 |
| `human_fidelity`         | 2.70 | `motion_smoothness`   | 2.60 |
| `inertial_consistency`   | 2.50 | `dynamic_degree`      | 2.57 |
| `physical_mechanics`     | 2.57 | `object_permanence`   | 2.50 |

**Checkpoint：** 无 —— 阶段 A（LLM）与阶段 B（VLM）皆为远端 endpoint。默认 `gemma-4-31b-local` 即 vLLM 在 `http://localhost:8003/v1` 提供的 `google/gemma-4-31b-it`。可替换为 `--judge gemini-3-flash`（托管 API，需 `GOOGLE_API_KEY`）以匹配论文使用的 Gemini 配置。

**论文 calibrated 头条分。** 论文公布的 Kling-v2.6 在全 50 prompt 头条切片上的 PHAS 为 **4.12**，使用 *校准后* 的逐维权重（基于人工标注的非负 ridge 回归）。要复现，请软链全部 50 个 mp4 至 videos 目录并重跑 `--aggregator phas`（校准权重会自动加载）。

**注意事项：** 若工作区已存在 `vqa_questions_50prompts.jsonl`，阶段 A 会被静默跳过——日志中 "judge_llm not given; defaulting to judge for VQA gen" 这一行在文件存在性检查之前就会无条件输出，不能据此判断阶段 A 是否真的跑了。请查看 `<ws>/api_logs/calls/Qwen/` 以确认。对 Gemma 请保持 `max_concurrency=2`，避免 broken-pipe 报错洪流；阶段 A 在 Qwen 上很轻量，可与重负载的阶段 B 并行不冲突。

### 6.5 WorldScore

**概览。** 10 维：7 静态（`camera_control`、`object_control`、`content_alignment`、`3d_consistency`、`photometric_consistency`、`style_consistency`、`subjective_quality`）+ 3 动态（`motion_accuracy`、`motion_magnitude`、`motion_smoothness`）。全部 CV 评分，无需 VLM judge。技术栈：DROID-SLAM（相机 + 3D）、SEA-RAFT（光流 + 光度一致性）、VFIMamba（基于插帧的 motion smoothness）、SAM2（motion-accuracy 掩膜传播）、GroundingDINO + SAM-H（目标检测）、VGG-19（风格 Gram）、LAION + CLIP-IQA+（主观质量）、torchmetrics CLIPScore（内容对齐）。头条指标：`WorldScore-Static`（7 维静态分均值 × 100）与 `WorldScore-Dynamic`（10 维均值 × 100）。

**运行示例：在 3 个 CogVideoX-5B 动态视频上跑 `motion_magnitude`（~2 min, 4 GB 显存）**

WorldScore 的上游代码以相对路径定位权重；`fetch-checkpoints` 后我们将权重软链到上游期望的目录。

```bash
# 1. 取 smoke 视频 + 上游仓库 + SEA-RAFT 权重
videvalkit fetch-smoke-data  --bench worldscore
videvalkit fetch-upstream    --bench worldscore   # git clone WorldScore 仓库
videvalkit fetch-checkpoints --bench worldscore

# 2. 把权重软链到上游期望的位置（一次性）
WS_ROOT=~/.cache/videvalkit/upstream/WorldScore
mkdir -p $WS_ROOT/worldscore/benchmark/metrics/checkpoints
ln -sf ~/.cache/videvalkit/checkpoints/worldscore/*.pth \
       $WS_ROOT/worldscore/benchmark/metrics/checkpoints/
ln -sf ~/.cache/videvalkit/checkpoints/worldscore/*.pkl \
       $WS_ROOT/worldscore/benchmark/metrics/checkpoints/
export VIDEVALKIT_WORLDSCORE_ROOT=$WS_ROOT

# 3. 暂存 3 个动态视频
mkdir -p ~/runs/worldscore/videos/cogvideox-5b/dynamic
ls ~/.cache/videvalkit/smoke-data/worldscore/videos/cogvideox-5b/dynamic/*.mp4 \
  | head -3 | xargs -I{} ln -sf {} ~/runs/worldscore/videos/cogvideox-5b/dynamic/

# 4. 评测（SEA-RAFT 在每视频 49 帧上算光流，取 magnitude 中位数）
videvalkit eval --bench worldscore \
    --videos ~/runs/worldscore/videos \
    --workspace ~/runs/worldscore/ws \
    --models cogvideox-5b \
    --dimensions motion_magnitude \
    --prompts-file ~/.cache/videvalkit/smoke-data/worldscore/prompts/dynamic.jsonl
```

**预期：** `per_dimension.motion_magnitude ≈ 56.4`（上游约定的 ×100 量纲）。仅跑这一维时，头条 `WorldScore-Dynamic ≈ 56.4`。SEA-RAFT 无 temperature，逐次结果一致。

**Checkpoint：**
- `worldscore/Tartan-C-T-TSKH-spring540x960-M.pth`（130 MB）—— SEA-RAFT 光流
- `worldscore/raft-things.pth`（20 MB）—— DROID-SLAM 中使用的 RAFT
- 等价：`allow_patterns=["worldscore/Tartan*", "worldscore/raft-things*"]`

若要跑全部 10 维，还需：`groundingdino_swint_ogc.pth`、`sam_vit_h_4b8939.pth`、`sam2.1_hiera_large.pt`、`VFIMamba.pkl`、`droid.pth`、`sac+logos+ava1-l14-linearMSE.pth`（直接 `--bench worldscore` 不加 `allow_patterns` 即可获取全部，约 6.3 GB）。

**注意事项：** `style_consistency` 对比的是 HF 数据集中提供的 `input_image.png`，**不是**生成视频的第 0 帧；对 T2V 模型而言，参考图由上游 T2I 流水线在相同场景 prompt 上生成。评测前请先跑一次 `runners/extract_refs.py` 以物化逐条目（per-entry）参考图。请把 torch 钉在 `2.3.1+cu121`；安装 `mamba_ssm` 时**不要**让 pip 自动升级 torch（升级会破坏 lietorch/droid_backends/sam2/pyiqa）。适配器内置纯 PyTorch 的 `mamba_ssm.selective_scan` shim，无需重编译 CUDA 扩展即可运行 VFIMamba。

### 6.6 T2V-CompBench

**概览。** 7 维组合性维度。4 维 MLLM（`consistent_attribute`、`action_binding`、`object_interactions`、`dynamic_attribute`）使用 LLaVA-1.6-34B，temperature=0，`chatml_direct` 对话模板，3 seed 取均值。3 维 CV：`generative_numeracy`（GroundingDINO + 计数）、`spatial_relationships`（GD + Depth-Anything V1，2D+3D 联合）、`motion_binding`（GD + SAM-H + DOT = cotracker2 + RAFT estimator + RAFT refiner）。聚合器：7 维无权均值。

**运行示例 A —— `action_binding` paper-mode，3 个 CogVideoX-5B 视频（LLaVA 加载后 ~5 min，需 ≥80 GB 显存）**

```bash
# 1. 取 smoke 视频 + t2vcompbench CV 权重 + LLaVA-1.6-34B
videvalkit fetch-smoke-data  --bench t2vcompbench
videvalkit fetch-checkpoints --bench t2vcompbench   # GroundingDINO + SAM-H + Depth-Anything + DOT（~4 GB）
videvalkit fetch-checkpoints --bench hf-models      # LLaVA-1.6-34B + Qwen2.5-7B + LLaVA-Video-7B + CLIP + DepthAnything（~95 GB）

# 2. clone T2V-CompBench 上游仓库（包含 LLaVA 评测脚本）
videvalkit fetch-upstream --bench t2vcompbench

# 3. 暂存 3 个 action_binding 标签的视频
mkdir -p ~/runs/t2vcomp_paper/videos/CogVideoX-5B
ls ~/.cache/videvalkit/smoke-data/t2vcompbench/videos/CogVideoX-5B/action_binding_*.mp4 \
  | head -3 | xargs -I{} ln -sf {} ~/runs/t2vcomp_paper/videos/CogVideoX-5B/

# 4. 评测（LLaVA-1.6-34B temperature=0，3 seed × 3 视频 = 9 次推理，~5 min）
videvalkit eval --bench t2vcompbench \
    --videos ~/runs/t2vcomp_paper/videos \
    --workspace ~/runs/t2vcomp_paper/ws \
    --models CogVideoX-5B \
    --dimensions action_binding \
    --prompts-file ~/.cache/videvalkit/smoke-data/t2vcompbench/prompts.jsonl \
    --extra-kwarg 'mode="upstream"'
```

**预期：** `per_dimension.action_binding ≈ 7.22`（原始 1-10 分），经上游 `(raw - 1) / 9` 公式归一化 = **0.691**。论文 Table 2 中 CogVideoX-5B 在全 200 视频集的分数为 0.533；我们 3 个视频的 smoke 在 ±0.15 采样带内，符合预期。

**Checkpoint：** `hf-models/liuhaotian/llava-v1.6-34b/`（15 个 safetensors，共 68 GB）；等价 `allow_patterns=["hf-models/liuhaotian/llava-v1.6-34b/*"]`。

**运行示例 B —— `action_binding` toolkit-mode（不需要 LLaVA-34B，可在小显存 GPU 上运行）**

若没有 80 GB 显存，可切换到 `mode="toolkit"`，将 MLLM 维度路由到你配置的 VLM judge：

```bash
videvalkit eval --bench t2vcompbench \
    --videos ~/runs/t2vcomp_toolkit/videos \
    --workspace ~/runs/t2vcomp_toolkit/ws \
    --models CogVideoX-5B \
    --dimensions action_binding \
    --judge gemma-4-31b-local \
    --prompts-file ~/.cache/videvalkit/smoke-data/t2vcompbench/prompts.jsonl \
    --extra-kwarg 'mode="toolkit"'
# → per_dimension.action_binding ≈ 0.33（绝对值不同，因评分器从 LLaVA-34B 换成了 Gemma；此时不再保证 paper-Δ）
```

**注意事项：** `--mode upstream` 是 paper-exact 路径（LLaVA-1.6-34B 子进程 shim），是按字节复现论文 Δ 所必需。`--mode toolkit` 将 4 维 MLLM 路由到你配置的 VLM judge——在 GPU 显存不足 80 GB 时有用，但会接受被记录到 `meta.scorers_used` 的 paper-Δ 警告。

**验证：** 在 upstream 模式下，7 维中 6 维与论文 Table 5 相差 ±0.020 内；`consistent_attribute` 出现 +0.186 漂移，归因于 LLaVA HEAD 修订漂移（论文的 requirements.txt 未钉住 revision）。

---

## 7. 配置评分器（VLM/LLM judges）

参考 DEV_MANUAL 第 16 节。Judges（评判模型）是可插拔的：每个需要调用 VLM/LLM 的基准都通过 `SUPPORTED_JUDGES` 注册表以及一条优先级链完成路由，让你无需 fork 工具集即可逐主机覆写。

### 优先级链（从低到高）

1. `src/videvalkit/configs/judges.py` 中的内置 `SUPPORTED_JUDGES` 默认值
2. 环境变量默认值（如 `VIDEVALKIT_JUDGE_DEFAULT`）
3. `~/.config/videvalkit/judges.yaml`——用户级覆写与新增 judge 名称
4. `<workspace>/judges.yaml`——按项目钉死
5. CLI 标志——最高优先级：`--judge`、`--judge-endpoint`、`--judge-model`、`--judge-kind`、`--judge-api-key-env`

### 端口 8003 / 8004 的本地 vLLM Gemma / Qwen

```yaml
# ~/.config/videvalkit/judges.yaml
gemma-4-31b-local:
  kind: openai_compatible
  endpoint: http://localhost:8003/v1
  model: google/gemma-4-31b-it
  provider: google
  api_key_env: null
  request_timeout_s: 180

qwen3-32b-local:
  kind: openai_compatible
  endpoint: http://localhost:8004/v1
  model: Qwen/Qwen3-32B
  provider: Qwen
  api_key_env: null
```

启动对应的 vLLM 服务（一次性启动后保持驻留）：

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-4-31b-it --port 8003 \
  --max-model-len 32768 --gpu-memory-utilization 0.85 \
  --served-model-name google/gemma-4-31b-it &

CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-32B --port 8004 \
  --max-model-len 32768 --gpu-memory-utilization 0.85 \
  --served-model-name Qwen/Qwen3-32B &
```

`--served-model-name` 必须与 YAML 的 `model` 字段按字节一致；vLLM 通过该名称路由。

### OpenAI GPT-4o

```yaml
# ~/.config/videvalkit/judges.yaml
gpt-4o:
  kind: openai_compatible
  endpoint: https://api.openai.com/v1
  model: gpt-4o-2024-11-20
  provider: openai
  api_key_env: OPENAI_API_KEY
  cost_per_million_input_tokens: 2.50
  cost_per_million_output_tokens: 10.00
  cost_per_image_input: 0.00765

gpt-4o-2024-08-06:           # 用于复现 Video-Bench 论文
  kind: openai_compatible
  endpoint: https://api.openai.com/v1
  model: gpt-4o-2024-08-06
  provider: openai
  api_key_env: OPENAI_API_KEY
```

```bash
export OPENAI_API_KEY=sk-...
videvalkit eval --bench videobench --judge gpt-4o-2024-08-06 ...
```

### Gemini（Google AI Studio）

```yaml
gemini-3-flash:
  kind: gemini
  model: gemini-3-flash-preview
  provider: google
  api_key_env: GEMINI_API_KEY
  cost_per_million_input_tokens: 0.075
  cost_per_million_output_tokens: 0.30
```

```bash
export GEMINI_API_KEY=...
```

### Anthropic Claude

```yaml
claude-sonnet-4-6:
  kind: anthropic
  model: claude-sonnet-4-6
  provider: anthropic
  api_key_env: ANTHROPIC_API_KEY
```

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### CLI 选择

```bash
# 全基准 judge 替换
videvalkit eval --bench vbench2 --videos ... \
  --scorer-vlm gemma-4-31b-local        # 代替默认 lmms-lab/LLaVA-Video-7B-Qwen2

# 逐维覆写（混搭）
videvalkit eval --bench vbench2 --videos ... \
  --scorer-vlm gemma-4-31b-local \
  --scorer-vlm-dim complex_plot=claude-sonnet-4-6 \
  --scorer-vlm-dim human_interaction=gpt-4o-2024-11-20

# 临时 endpoint 覆写（无需改 YAML）
videvalkit eval --bench videobench --videos ... \
  --judge-kind openai_compatible \
  --judge-endpoint http://10.0.1.7:9001/v1 \
  --judge-model Qwen/Qwen3-32B \
  --judge-api-key-env null
```

当你替换某基准的论文默认评分器时，工具集会：
1. 把实际所用评分器写入 `Summary.meta.scorers_used`。
2. 在 `compare-leaderboard` 输出中拒绝声称 Delta-vs-paper。
3. 若你传入 `--allow-judge-substitution`，则将容忍带放宽 3 倍。
4. 在 `api_logs/` 中逐次调用记录 token 用量。

运行后可查看花费：

```bash
videvalkit api-usage --workspace runs/first
```

---

## 8. 配置 GPU

参考 DEV_MANUAL 第 17 节。三种调度模式：

| 模式 | 触发方式 | 行为 |
|---|---|---|
| `single` | `--gpu N` | 整个基准在单一设备上跑 |
| `dim_parallel` | `--gpus 0,1,2`（或 `auto`） | 维度分片到 GPU 池中，每个维度作为独立子进程，并设置 `CUDA_VISIBLE_DEVICES=N` |
| `affinity` | `--gpu-affinity Dim=N,...` | 用户钉特定维度到特定 GPU；其余维度回落到 GPU 池 |

模式根据传入标志推断；没有显式的 `--mode` 切换开关。

### 单 GPU 示例

```bash
videvalkit eval --bench worldjen --gpu 2 --videos ... --workspace ...
```

### 维度并行示例（VBench-2.0 的 18 维分到 3 张 GPU）

```bash
videvalkit eval --bench vbench2 --videos ... --gpus 0,1,2 --gpu-strategy most_free_memory
```

### 亲和示例（把 LLaVA-34B 维度钉到专属 GPU）

```bash
videvalkit eval --bench t2vcompbench --videos ... \
  --gpu-affinity consistent_attribute=0,action_binding=1,object_interactions=2 \
  --gpus 3                                # 其他维度回落到 GPU 3
```

### 自动选择策略

| 策略 | 在获取时选择的 GPU |
|---|---|
| `most_free_memory`（默认） | `argmax(free_mem_gb)` |
| `round_robin` | GPU 池的下一个索引 |
| `least_utilization` | `argmin(utilization_percent)` |

通过 `--gpu-strategy {most_free_memory, round_robin, least_utilization}` 传入，或写入 YAML。

### 通过 `compute.yaml` 设定持久默认值

```yaml
# ~/.config/videvalkit/compute.yaml
compute:
  gpus: auto                              # 对所有基准默认
  auto_strategy: most_free_memory
  reserve_mem_gb: 6                       # 不挑选剩余 <6 GB 的 GPU
```

逐工作区的覆写写入 `<workspace>/config.yaml`，schema 一致，并额外提供 `compute.affinity` 段，用于 `benchmark.dim` 形式的钉死。

调度器在工作区中写入 `compute_log.jsonl`，每次维度启动一行，与 `api_log.jsonl` 联动，用以回答“本次评测花了多少？”——既包含 GPU 分钟数也包含 token 花费。

最小显存提示已写入各基准 manifest（例如 T2V-CompBench 的 `consistent_attribute` 要求 `min_mem_gb=75, exclusive=True`）；调度器拒绝把不满足提示的维度调度到对应 GPU。

---

## 9. 跨基准聚合

当工作区中包含一个或多个基准的 summary 文件时，可将它们聚合成统一榜单：

```bash
videvalkit aggregate --workspace runs/first
# 输出文件：
#   runs/first/results/leaderboard/cross_benchmark.json
# 控制台：
#   #1  seedance20         z=+0.521
#   #2  pangu_model3_141   z=+0.183
#   #3  wan-14B-pe-141     z=-0.704
```

工具集内置五种聚合器：

| 聚合器 | 行为 |
|---|---|
| `weighted_sum` | 用户在维度/基准层面指定权重 |
| `vbench_weighted` | `Total = 0.54 * Quality + 0.46 * Semantic`，VBench v1 头条 |
| `vbench2_category` | 5 类别均值再均值得 `Overall`，VBench-2.0 头条 |
| `phas` | PHAS = `Sum(w_i * mu_i) - lambda * sigma^2`，WorldJen 头条 |
| `bt` | 跨模型的 Bradley-Terry 两两比较排序，`videvalkit aggregate` 跨基准用 |

通过 `aggregate` 的 `--aggregator <name>` 切换，默认 `weighted_sum`，基准间等权。

跨基准输出 schema：

```json
{
  "models": ["seedance20", "pangu_model3_141", "wan-14B-pe-141"],
  "ranked": [
    {"model": "seedance20", "z_score": 0.521, "per_bench": {"worldjen": 4.12, "vbench": 0.815, "..."}}
  ],
  "bt_rating": {"seedance20": 1.42, "...": "..."},
  "meta": {"aggregator": "weighted_sum", "n_benches": 3, "...": "..."}
}
```

---

## 10. 独立度量

参考 DEV_MANUAL 第 14 节。独立度量（standalone metrics）在两组视频上计算单一算法——没有 prompt、没有维度、没有聚合器。适合不需要基准脚手架，只要一个数字（FID、FVD、CLIP-Score、PSNR、SSIM、LPIPS）的场景。

```bash
videvalkit metric \
  --name fvd \
  --gen-videos path/to/generated/ \
  --ref-videos path/to/reference/ \
  --device cuda:0 \
  --out fvd_result.json
```

支持的度量：

| 度量 | 需要参考集 | 备注 |
|---|---|---|
| `fid` | 是 | clean-fid / pytorch-fid（Inception-V3 pool3） |
| `fvd` | 是 | 标准 I3D Kinetics-400 |
| `clipscore` | 否 | OpenAI CLIP ViT-B/32 |
| `psnr` | 是 | 成对视频 PSNR |
| `ssim` | 是 | 成对视频 SSIM |
| `lpips` | 是 | AlexNet 或 VGG 主干 |

注：独立度量模块属于 planned-2026-05-18 通道；注册表与 CLI 已接线，FVD/FID/CLIP-Score 现可使用；PSNR/SSIM/LPIPS 已排期，优先级见 DEV_MANUAL 第 14 节。

返回一个扁平 JSON：`{"fvd": 132.4, "n_gen": 200, "n_ref": 200, "backbone": "i3d-k400", ...}`。

---

## 11. 生成你自己的视频

若你有自己的 T2V 模型，请将生成结果按各基准的预期 layout（目录结构）组织到一个根目录下：

```
<videos_root>/
  <model_name>/
    <prompt_id>.mp4
    ...
```

各基准预期 layout（完整规范见 DEV_MANUAL 第 4 节）：

| Bench | Layout |
|---|---|
| `vbench` | `<root>/<model>/<prompt_id>-<sample_idx>.mp4`（每 prompt 5 个采样） |
| `vbench2` | `<root>/<model>/<dim>/<prompt_id>-<sample_idx>.mp4`（按维度组织，每 prompt 1-3 个采样，视维度而定） |
| `videobench` | `<root>/<model>/<dim>/<prompt_id>.mp4`（按维度组织；上游 zip 包也遵循该结构） |
| `worldjen` | `<root>/<model>/<prompt_id>.mp4`（每 prompt 1 个采样） |
| `worldscore` | `<root>/<model>/{static,dynamic}/<prompt_id>.mp4`（按 split 组织） |
| `t2vcompbench` | `<root>/<model>/<dim>/<prompt_id>.mp4`（按维度组织，完整运行每维度 200 prompt） |

布好后将 `<videos_root>` 传给 `videvalkit eval --videos <root>`，适配器会按预期 layout 遍历。逐基准 prompt 文件位于 `~/.cache/videvalkit/smoke-data/<bench>/prompts/`（用于复现论文），自定义 prompt 则通过 `--prompts-file <path>` 传入。

resume（断点续算）是默认行为：重跑相同命令会跳过已有 `results/raw/*.json` 的 `(model, dim, prompt)` 三元组。如需强制重跑，请删除对应 JSON。

---

## 12. 自定义 prompt（自动标注器）

VBench v1 与 VBench-2.0 中依赖 prompt 的维度（`object_class`、`color`、`spatial_relationship`、`scene`、`human_action`、`multiple_objects`，以及 VBench-2.0 的多数维度）需要每条 prompt 上携带 `auxiliary_info` 标签——上游代码据此组装检测/匹配 prompt。自定义 prompt 默认没有这些标签。

自动标注器借助本地 LLM 补齐缺口：

```bash
python scripts/auto_label_prompts.py \
  --prompts /data/my_prompts/prompts.jsonl \
  --out-dir runs/my_ws/prompts/auto_labels \
  --benchmarks vbench,vbench2 \
  --judge qwen3-32b-local
```

输出目录下生成 `vbench_full_info.json` 与 `vbench2_full_info.json`，每条 prompt 一项，附带自动抽取的 `auxiliary_info` 块。逐维度的 LLM schema 打包在 `videvalkit/prompt_labelers/` 中。

使用自动标注后的文件：

```bash
videvalkit eval --bench vbench \
  --prompts-file runs/my_ws/prompts/auto_labels/vbench_full_info.json \
  --videos ... --workspace ...
```

注意事项：自动标注效果良好但并非完美。局限性见 TEST_MANUAL 第 2.4 节。如要主张 paper-Delta，请始终使用上游原版 prompt。

---

## 13. 故障排查与常见问题

请先运行 `videvalkit doctor`，多数问题会一目了然。

| 表现 | 可能原因 | 处理 |
|---|---|---|
| `fetch-checkpoints` 时 HF 鉴权失败 | 未设置 token，或网络走的 HF 镜像会丢文件 | 用 read-scope token 运行 `hf auth login`；若走镜像，`export HF_ENDPOINT=https://hf-mirror.com`。WorldScore 的 parquet 必须从 `huggingface.co` 直拉（镜像会丢） |
| 拉取过程中磁盘耗尽 | `~/.cache/videvalkit` 落在了小分区 | 拉取前 `export VIDEVALKIT_CACHE_ROOT=/data/videvalkit_cache`；完整 smoke + ckpt 体量约 130 GB |
| LLaVA-1.6-34B 在 <80 GB GPU 上 OOM | T2V-CompBench paper-mode 要求 80 GB | 改用 `--mode toolkit --judge gemma-4-31b-local`（paper-Delta 容忍带变宽） |
| Judge endpoint 不可达 | vLLM 未运行，或 `--served-model-name` 与注册表的 `model` 字段不匹配 | `curl http://host:port/v1/models | jq` 查看真实名称；对齐注册表或启动参数 |
| `cv2` 在 H.264 论文视频上取错帧 | 稀疏关键帧 seek bug | 已在 `utils/video.py:extract_frames` 修复（用顺序 `cap.read()` 配合 `wanted_set` 替代 `cap.set(CAP_PROP_POS_FRAMES)`）。如发现重复帧，请确认工具集版本不低于 0.0.1 post-2026-05-18 |
| WorldJen 阶段 A 总不跑 | 工作区已存在预构建 `vqa_questions_50prompts.jsonl` | 阶段 A 会被静默跳过。若要强制跑，请删除该文件或加 `--no-prebuilt-vqa` |
| 并发下 Gemma 出现 `broken pipe / RemoteProtocolError` | 对 vLLM 的并发突发 | 对所有依赖 Gemma 的基准设置 `--max-concurrency 2`；将共用 Gemma 的基准错峰运行 |
| 安装时 `detectron2` 编译失败 | `nvcc --version` 与 `torch.version.cuda` 不匹配 | 安装与你 CUDA 匹配的预编译 detectron2 wheel |
| `flash-attn` 导入报 ABI 错 | wheel 与 torch 不匹配 | `pip install flash-attn==2.5.8 --no-build-isolation` |
| VBench v1 的 `dynamic_degree` 跨运行抖动 | RAFT 在 GPU 上的非确定性 | 属预期；将该维容忍带放宽至 +/-0.025 |
| `api_logs/` 膨胀到数 GB | 调用量大 | `scripts/clear_cache.py --api-logs --older-than 30d` |
| 评测启动后立刻报 “no videos found” | layout 不匹配 | 对照第 11 节确认 layout；`videobench` 与 `t2vcompbench` 是按维度组织的 |

分数不匹配时的深入排查路径见 TEST_MANUAL 第 5 节。

---

## 14. 许可证与引用

工具集本身（本代码库）使用 **Apache-2.0** 许可证。各个被适配的上游各自持有自己的许可证，集中陈列于仓库根的 `LICENSES/`。

重要上游许可证：

| 上游 | 许可证 |
|---|---|
| VBench、VBench-2.0 | Apache-2.0 |
| Video-Bench、WorldJen | 各论文自身许可证 |
| T2V-CompBench（LLaVA-1.6-34B 子进程路径） | 仅限研究用途 |
| DROID-SLAM、SEA-RAFT、GroundingDINO、SAM | 各自的开源许可证 |
| WorldScore checkpoints | 在各骨干网络条款下的衍生使用 |

### 引用工具集

```bibtex
@software{videogenevalkit2026,
  title  = {videogenevalkit: A unified evaluation toolkit for text-to-video generation},
  author = {Liu, Ning and contributors},
  year   = {2026},
  url    = {https://github.com/videogenevalkit/videogenevalkit}
}
```

### 引用底层基准

发表来自特定适配器的数字时，请**额外引用原始基准论文**，不要只引工具集。6 个锚定基准的 BibTeX 条目位于 `docs/citations.bib`。具体来说：

- VBench v1：Huang 等，CVPR 2024
- VBench-2.0：Zheng 等，CVPR 2025
- Video-Bench：Han 等，CVPR 2025（arXiv:2504.04907）
- WorldJen：WorldJen 团队，项目页 github.com/moonmath-ai
- WorldScore：Duan 等
- T2V-CompBench：Sun 等，ECCV 2024

---

> 用户手册到此结束。论文 Delta 验证结果见 [TEST_MANUAL.md](TEST_MANUAL.md)；工具集内部架构见 [DEV_MANUAL.md](DEV_MANUAL.md)。
