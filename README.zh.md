# videogenevalkit

**面向文本生成视频(T2V)的统一评测工具包。**

*[English](README.md) · [中文](README.zh.md)*

一个 CLI、一份 workspace、一套 schema。**三种方式**给 T2V 模型打分——整个
**基准**、单个**指标**,或整项**能力**——用**你选的评审**。基准分数逐字节对齐
官方榜单。

> 📖 **文档:[wiki](docs/index.md) 是主参考,中英双语** —— 快速上手、指南,以及
> CLI / 指标 / 基准 / 评审参考。设计理念见
> [`docs/design/`](docs/design/PRODUCT_DESIGN.md);长篇安装见
> [`docs/USER_MANUAL.md`](docs/USER_MANUAL.md)。

<p align="left">
  <a href="#快速上手"><img alt="quickstart" src="https://img.shields.io/badge/quickstart-30%20min-blue"></a>
  <a href="#支持哪些"><img alt="benchmarks" src="https://img.shields.io/badge/benchmarks-10-orange"></a>
  <a href="https://huggingface.co/datasets/videogenevalkit/checkpoints"><img alt="HF checkpoints" src="https://img.shields.io/badge/HF-checkpoints-yellow?logo=huggingface"></a>
  <a href="LICENSES/"><img alt="licenses" src="https://img.shields.io/badge/licenses-multi--upstream-lightgrey"></a>
</p>

---

## 三种评测方式

| 入口 | 命令 | 回答的问题 |
|---|---|---|
| **基准** | `videvalkit eval --bench vbench` | “我的模型在 VBench 上得分如何?” |
| **指标** | `videvalkit metric run --name fvd ...` | “这些视频的 FVD 是多少?” |
| **能力** | `videvalkit capabilities eval motion ...` | “跨所有指标看,运动表现如何?” |

从基准中抽出的指标,无论经 `--bench` 还是 `metric run` 触达都**逐位一致**——
单一实现,不漂移。三条设计承诺:*适配而非重写* · *后端可插拔* ·
*插件优先(经 YAML / pip / 本地目录扩展,无需 fork)*。

---

## 支持哪些

**10 个基准适配器** —— 6 个锚定公开榜单 + Semantics-Axis(自研)+ 3 个补充:

| 适配器 | 上游 | 评测什么 |
|---|---|---|
| `vbench` | [Vchitect/VBench](https://github.com/Vchitect/VBench) (CVPR 2024) | 16 维,质量 + 语义;加权和 + min-max 归一 |
| `vbench2` | [Vchitect/VBench-2.0](https://github.com/Vchitect/VBench) | 5 大类 18 维,含物理 |
| `videobench` | [Video-Bench](https://github.com/Video-Bench/Video-Bench) (CVPR 2025) | 9 维:对齐 + 静/动态质量 |
| `worldjen` | [WorldJen](https://github.com/moonmath-ai/WorldJen-benchmarking-subsystem) | 16 维;PHAS 聚合器 |
| `worldscore` | [WorldScore](https://github.com/yhw-yhw/WorldScore) | 10 维:SLAM + RAFT + SAM + IQA 组合 |
| `t2vcompbench` | [T2V-CompBench V2](https://github.com/KaiyueSun98/T2V-CompBench/tree/V2) | 7 个组合性维度;LLaVA-1.6-34B MLLM + CV |
| `semantics_axis` | 自研 | 21 个 prompt-following 轴;VLM 评审,1–5 |

外加 3 个补充适配器:Physics-IQ、VBench++、V-ReasonBench。

**v0.2 还包含:**

| 能力 | 内容 |
|---|---|
| **20 个独立指标** | FVD · VFID · KVD · CLIP-FVD · CLIP-Score · ViCLIP-Score + 8 个基准 lift + 专项维度。**目前 16 个无需评审即可跑**;`artifact-diagnostic` 用 `--judge` 运行。任意一个用 `videvalkit metric run --name X`。 |
| **评审选择** | `--judge paper/default/<name>` · 用户 `judges.yaml` · 临时 `--judge-endpoint` · `--no-judge` 完全离线 |
| **评测配置档** | `--profile quick/standard/full` + `videvalkit estimate` 成本预览 |
| **训练监控** | `videvalkit watch` + `videvalkit.training.monitor` Python API |
| **能力标签** | 44 标签词表;`videvalkit capabilities list/show/eval` |

**开箱 8 个 VLM/LLM 评审**:本地 vLLM(Gemma-4-31B、Qwen3-32B、Qwen3-VL-32B、
LLaVA-Video-7B)+ 云 API(Gemini、GPT-4o、Claude);在
`~/.config/videvalkit/judges.yaml` 添加自己的。**聚合器**:weighted_sum ·
vbench_weighted · vbench2_category · phas · bt。

---

## 可复现性 —— 你实际能得到什么

我们对 6 个锚定基准重跑了官方榜单。**相对已发布数字的平均 |Δ|:**

| 基准 | 模型 | 维度 | 平均 \|Δ\| | 备注 |
|---|---|---:|---:|---|
| **VBench v1** | HunyuanVideo | 16/16 | **0.012** | 与 HF 榜单一致 |
| **VBench-2.0** | HunyuanVideo | 18/18 | **0.0055** | 4 维逐字节一致 |
| **T2V-CompBench** | CogVideoX-5B | 6/7 | **0.0046** | 论文版 LLaVA-1.6-34B;一个离群项已记录 |
| **Video-Bench** | CogVideoX-5B | 9/9 | 偏移 | Gemma 替代 GPT-4o;静态+对齐维度一致 |
| **WorldJen** | Kling-v2.6 | 16/16 | Δ −0.47 | decord 与 cv2 抽帧差异 |
| **WorldScore** | CogVideoX-5B | 10/10 | 已打通 | DROID-SLAM + SEA-RAFT + VFIMamba + SAM2 |

逐维度表见 [`docs/TEST_MANUAL.md`](docs/TEST_MANUAL.md)。

---

## 快速上手

### 1. 克隆 + 安装环境(~10 分钟)

工具包以预打包的 `conda-pack` 环境 tarball 交付——即产出 `docs/TEST_MANUAL.md`
中每个结果的逐字节环境(Python 3.10 + torch 2.3.1+cu121 + ~350 个锁定依赖)。
需要 Linux + CUDA。

```bash
git clone https://github.com/videogenevalkit/videogenevalkit.git
cd videogenevalkit

# 下载并解包环境 tarball(下载 ~7.6 GB,落盘 ~15 GB)
hf download videogenevalkit/env-tarball videvalkit-env.tar.gz --local-dir /tmp
sudo mkdir -p /opt/videvalkit-env
sudo tar xzf /tmp/videvalkit-env.tar.gz -C /opt/videvalkit-env
sudo chown -R $USER /opt/videvalkit-env
source /opt/videvalkit-env/bin/activate
conda-unpack                       # 重写环境内的绝对路径

# 安装工具包 + 源码构建的额外项
pip install --no-deps -e .
bash scripts/post_install.sh       # detectron2, SAM-2, GroundingDINO, ...
videvalkit doctor                  # 验证:环境、GPU、缓存、评审
```

> macOS / 无 GPU:可执行 `videvalkit list / metric list / capabilities list`
> 并开发,但基准/指标的*实际执行*需要 Linux + CUDA。

### 2. 拉取冒烟数据 + 检查点(~15 分钟,~8 GB)

```bash
videvalkit fetch-smoke-data                                    # 视频 + prompt
videvalkit fetch-checkpoints --bench worldscore --bench t2vcompbench
```

LLaVA-1.6-34B(68 GB,T2V-CompBench 论文模式 MLLM)**不**打包——首次论文模式
运行时从上游 HF 解析。

### 3. 跑第一个基准(~5 分钟)

```bash
videvalkit fetch-smoke-data --bench worldjen
mkdir -p ~/runs/worldjen/videos/Kling
ls ~/.cache/videvalkit/smoke-data/worldjen/videos/*/*.mp4 | head -3 \
  | xargs -I{} ln -sf {} ~/runs/worldjen/videos/Kling/

videvalkit eval --bench worldjen \
    --videos ~/runs/worldjen/videos --workspace ~/runs/worldjen/ws \
    --models Kling --judge gemma-4-31b-local --aggregator phas
# → ~/runs/worldjen/ws/results/summary/worldjen/Kling.json
```

没有评审端点?走无需评审路径:

```bash
videvalkit list benchmarks --no-judge        # vbench · worldscore · physics_iq · v_reasonbench
videvalkit eval --bench vbench --no-judge --profile quick --videos gen/ --workspace ws/
```

### 4. 单个指标 / 一项能力

```bash
# 分布型指标(需参考集;FVD 自动下载骨干)
videvalkit metric run --name fvd --gen-videos gen/ --refs ucf101-fvd --allow-tiny-sample

# 跨指标能力分
videvalkit capabilities eval visual_quality --videos gen/
```

完整走查见 [wiki 快速上手](docs/index.md),逐基准配方见
[`docs/USER_MANUAL.md`](docs/USER_MANUAL.md)。

---

## 你**不**需要做的事

- **不用重写评分器。** 适配器逐字节委托上游代码;我们加 IO、调度、注册表和
  统一输出格式。
- **不用论文 API 凭证。** 评审可插拔,默认本地 vLLM。云 API 一行配置切换。
  `--no-judge` 完全跳过。
- **不用每个基准一个 conda 环境。** 一个共享环境覆盖所有适配器。

---

## 文档

| 文档 | 内容 |
|---|---|
| **[wiki](docs/index.md)**(en/zh) | 主参考——快速上手、指南、CLI/指标/基准/评审、架构、路线图 |
| [设计存档](docs/design/PRODUCT_DESIGN.md)(en/zh) | 每个子系统*为何*这样设计 |
| [`USER_MANUAL.md`](docs/USER_MANUAL.md)(en/zh) | 长篇安装 + 逐基准运行配方 |
| [`TEST_MANUAL.md`](docs/TEST_MANUAL.md) | 逐基准验证:相对榜单的 Δ、容差、已知差异 |
| [`DEV_MANUAL.md`](docs/DEV_MANUAL.md) | 深度架构(v0.0.1 时代;以 wiki 为准核对) |

文档可构建为带语言切换的可搜索站点:
`pip install -r requirements-docs.txt && mkdocs serve`。

---

## 许可与引用

工具包为 **Apache-2.0**。每个上游适配器保留各自许可,集中于
[`LICENSES/`](LICENSES/)(VBench/VBench-2.0 Apache-2.0;T2V-CompBench 在 LLaVA
路径上研究专用;DROID-SLAM / SEA-RAFT / GroundingDINO / SAM 按各自条款)。
报告某个适配器的数字时请引用其上游论文。

```bibtex
@software{videogenevalkit2026,
  title  = {videogenevalkit: A unified evaluation toolkit for text-to-video generation},
  year   = {2026},
  url    = {https://github.com/videogenevalkit/videogenevalkit},
}
```
