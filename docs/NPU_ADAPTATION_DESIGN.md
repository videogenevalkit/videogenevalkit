# NPU Adaptation — Cross-Cutting Design Doc

| 字段 | 内容 |
|---|---|
| 版本 | v0.1 (draft) |
| 状态 | **Deferred** — 未来参考，不在短期路线图（v0.2 / v0.3）排期内 |
| 创建 | 2026-05-20 |
| 最后调整 | 2026-05-20：明确不进 v0.2/v0.3 排期；保留为后续候选 |
| 性质 | **横切关注**（cross-cutting concern），不是新支柱 |
| 影响范围 | 4 个支柱全部：env install · judge endpoint · scorer/metric 实现 · profile 资源声明 |
| 关联文档 | [`PRODUCT_DESIGN.md`](PRODUCT_DESIGN.md) §3.1 (L0/L1) · [`JUDGE_SELECTION_DESIGN.md`](JUDGE_SELECTION_DESIGN.md) §4 · [`INTEGRATION_FRAMEWORK_DESIGN.md`](INTEGRATION_FRAMEWORK_DESIGN.md) §3-§5 · [`QUICK_EVAL_DESIGN.md`](QUICK_EVAL_DESIGN.md) §3 |
| 目标读者 | 国产算力部署团队 · 训练侧用 Ascend / MLU 的研发者 · 内部下游集成方 |

---

## 0. 状态说明（重要）

> **本文档当前为"未来候选"，不进入 v0.2 / v0.3 排期。**
>
> - 不会前置 E1/E2 到 v0.2 末期；§16.1 原"工作量更新"中的前置建议**作废**
> - v0.2 仍按原计划执行 4 支柱（A/B/C/D），CUDA-only 路径
> - 待 v0.3 完成后，根据实际用户需求与算力可用性，再决定是否在 v0.4+ 启动 NPU 适配
> - 文档其余内容（架构、CUDA 假设盘点、设备抽象层方案、兼容矩阵）保持有效，作为启动时的设计起点

---

## 1. 背景与定位

### 1.1 为什么现在做 NPU

国产 T2V 模型训练侧大量在昇腾 (Ascend) 集群上跑（CloudVideo, Kling 等多家）。如果评测必须切到 CUDA 才能跑，会产生：

- **数据搬运**：训练机 → CUDA 评测机，几百 GB 视频反复传
- **环境不一致**：训练用 torch_npu，评测用 torch+cu121，可能出现"训练正常，评测时模型 forward 行为微变"
- **算力浪费**：训练机 NPU 闲置等评测
- **采购阻塞**：很多场景拿不到 CUDA 卡

NPU 适配是国产生态的硬需求，不是 nice-to-have。

### 1.2 NPU 谱系

本文 **主要目标 = Ascend**（昇腾 910 / 910B / 910C），兼容以下二级目标：

| 后端 | torch 扩展 | 状态 | 优先级 |
|---|---|---|---|
| NVIDIA CUDA | 原生 torch | ✅ 当前 baseline | — |
| Huawei Ascend (昇腾) | `torch_npu` | ⚠ 待适配 | **P0 (v0.3)** |
| Cambricon MLU (寒武纪) | `torch_mlu` | ⚠ 待适配 | P2 (v0.4 候选) |
| 海光 DCU | HIP-based (近 CUDA) | ⚠ 大部分能直跑 | P2 (v0.4 候选) |
| 摩尔线程 | `torch_musa` | ⚠ 待适配 | P3 |
| Apple MPS / Intel XPU / 其他 | torch native | 仅 dev 态 | P3 |
| CPU fallback | torch native | 部分 metric 可跑 | P1 (v0.3) |

**v0.3 仅交付 Ascend + CPU fallback**，其他 NPU 走"plug-in 路径"由社区贡献。

### 1.3 用户原始需求

> "还有一个就是 npu 的适配也要考虑进去"

—— 翻译为产品需求：

| # | 需求 | 验收 |
|---|---|---|
| N1 | Ascend 用户能在本机跑评测 —— 至少 quick profile 全 6 bench 中 ≥ 4 个能跑 | NPU 装机后 `videvalkit doctor` 报告 NPU 可用，`eval-suite --profile quick --device npu` 成功 |
| N2 | 不能跑的 bench / dim 不能静默失败 —— 必须显式拒绝 + 给出原因 | "worldscore 在 NPU 上不支持（DROID-SLAM CUDA kernel），请用 CUDA 机或跳过该 bench" |
| N3 | VLM judge 在 NPU 集群上跑（vLLM-Ascend / MindIE）应与 CUDA 端等价 | 同一 judge 名走 OpenAI-compat 协议，下游 toolkit 无感知 |
| N4 | 不强迫所有用户装 NPU 依赖 | NPU 是可选 env，CUDA 用户的安装路径完全不变 |
| N5 | 不为 NPU fork 一个独立 repo 或独立 CLI | 同一份代码 / 同一份 CLI，差异在 install 与 device flag |

---

## 2. 目标 / 非目标

### 2.1 目标

1. **设备抽象层**：`videvalkit.core.device` 统一 `get_device(preferred) / device_name() / num_devices()`，把 42 处直接 `"cuda"` 引用收敛
2. **CLI `--device` flag**：`auto / cuda / npu / mps / cpu`，runner 与各 adapter 接收
3. **NPU env 路径**：`envs/videvalkit-npu.yaml` + `scripts/post_install_npu.sh` + 独立 conda-pack tarball
4. **VLM judge 端点中性**：现有 `openai_compatible` backend 已 device-agnostic（HTTP 协议），文档说明如何在 Ascend 上起 vLLM-Ascend / MindIE
5. **NPU 兼容矩阵**：每 bench / 每 dim / 每 metric 明确标注 NPU 支持状态（✅ / ⚠ partial / ❌），列入 TEST_MANUAL
6. **Fail-fast 友好**：选 NPU 跑 worldscore 时直接报错给出建议（不进 4 小时后崩）
7. **零回归**：CUDA 用户路径完全不变

### 2.2 非目标

- ❌ **不做** 自动 CUDA→NPU 算子转换（torch_npu 已提供，但 DROID-SLAM 这种自定义 kernel 没人能自动迁移）
- ❌ **不做** NPU 上的 paper-faithful 复现承诺 —— paper 数大多在 CUDA 上跑，NPU 结果允许在 tolerance 内偏差
- ❌ **不做** 跨设备混合训练 / 评测（NPU + CUDA 混部）—— scheduler 仅单后端
- ❌ **不做** NPU CI（GitHub Actions 没 NPU runner）—— 用 self-hosted runner 或 manual verification
- ❌ **不做** AscendCL / CANN 直接调用 —— 全部走 PyTorch + torch_npu 这层抽象
- ❌ **不发布** Ascend 上的 leaderboard 数字（precision 差异 + Ascend 上没跑过 paper 模型）

---

## 3. 当前 codebase 的 CUDA 假设盘点

### 3.1 数量分布（grep 结果）

| 文件 | CUDA 引用数 | 难度 |
|---|---:|---|
| `worldscore/scorers.py` + `runners/*.py` | 18 | 🔴 高（DROID-SLAM custom kernel + SEA-RAFT autocast） |
| `t2vcompbench/scorers.py` + `benchmark.py` | 8 | 🟡 中（device kwarg 链长，但都 PyTorch high-level） |
| `vbench/benchmark.py`、`vbench2/benchmark.py` | 4 | 🟢 低（仅 `"cuda" if available else "cpu"` 三元式） |
| `worldscore/runners/static_dims.py` | 2 | 🟢 低（`torch.cuda.empty_cache()`，可包一层 no-op fallback） |
| `videobench/*.py`、`worldjen/*.py` | 0 | ✅ 已 device-agnostic（VLM judge over HTTP） |

### 3.2 安装依赖里的 CUDA 假设

| 来源 | 内容 | NPU 处理 |
|---|---|---|
| `envs/videvalkit.yaml` | 硬钉 `cu121` + mmcv `cu121` wheel index | 新 yaml 用 `torch_npu` + Ascend 兼容 mmcv |
| `post_install.sh` Group 1-4 | 7 个 build：detectron2 / SAM-2 / GroundingDINO / segment-anything / lietorch / droid_backends / en_core_web_sm | 5 个能在 NPU 上 build；lietorch + droid_backends ❌ |
| `--minimal` flag | 已能跳 detectron2/lietorch/droid_backends 三大重活 | **NPU 默认走 --minimal 等价路径** |

### 3.3 关键阻塞项

| 阻塞 | 范围 | 影响 |
|---|---|---|
| **DROID-SLAM** (`droid_backends`) | WorldScore 的 camera_control + 3d_consistency dims | NPU 上跑不动；这 2 dim 在 NPU 端标 ❌ |
| **lietorch** | WorldScore SLAM 依赖 | 同上 |
| **detectron2** (CUDA-only build) | VBench-2.0 Human_Anatomy dim | NPU 端标 ❌（或后续找 ascend 移植版） |
| **flash-attn** | VBench-2.0 部分 dim 加速 | NPU 用 torch_npu 自带的注意力实现替代，性能下降 ~30% 但能跑 |
| **xformers** | 多处用作可选加速 | torch_npu 自带优化，xformers skip |
| **bitsandbytes** | LLaVA-34B 4-bit 量化 | NPU 走 fp16 / int8 via MindIE，不用 bnb |

---

## 4. 三阶段策略

```
v0.3 (Minimum NPU)         v0.4 (Expand)              v1.0 (Native)
─────────────────────      ─────────────────────      ─────────────────────
• 设备抽象层 + --device     • 更多 dim 在 NPU 通     • 官方 NPU env tarball
• Ascend env yaml 雏形       • Cambricon / DCU 探    • leaderboard 含 NPU 列
• 4 个 bench quick 跑通       社区 plugin              • paper-faithful 在 NPU
  (vbench/vbench2 部分/        • MindIE judge 文档        上的可控偏差表
   videobench/worldjen)        完善                     • CI 含 self-hosted
• 矩阵 + fail-fast          • CPU fallback 完整         Ascend runner
• vLLM-Ascend judge          (PSNR/SSIM/FVD/CLIP)    • 论文里专门一节讲
  文档                        • 离线 NPU 复现验证        国产算力支持
```

### 4.1 v0.3 NPU 支持矩阵（目标）

| Benchmark | CUDA | Ascend (NPU) | CPU | 备注 |
|---|:-:|:-:|:-:|---|
| **vbench** | ✅ 16/16 | ⚠ ~10/16 | ❌ | DINO/CLIP/RAFT-based dims 可跑；motion_smoothness / aesthetic 待验证 |
| **vbench2** | ✅ 18/18 | ⚠ ~12/18 | ❌ | Human_Anatomy (detectron2) / Camera_Motion (CoTracker3 cuda kernel?) 暂不支持 |
| **videobench** | ✅ 9/9 | ✅ 9/9 | ✅ 9/9 | 纯 VLM judge，HTTP 协议，三端均可（judge 部署在哪都行） |
| **worldjen** | ✅ 16/16 | ✅ 16/16 | ✅ 16/16 | 纯 VLM judge，同上 |
| **worldscore** | ✅ 10/10 | ⚠ ~6/10 | ❌ | camera_control / 3d_consistency / reprojection_error 依赖 DROID-SLAM ❌；其余 7 dim 可跑 |
| **t2vcompbench** | ✅ 7/7 | ⚠ ~5/7 | ❌ | LLaVA-Video-7B 经 MindIE 起；GroundingDINO / SAM-H / Depth-Anything 在 NPU 上可跑（已有移植） |

**v0.3 总体**：6 bench 中 **2 个全通** (videobench / worldjen) + **4 个部分通** (vbench / vbench2 / worldscore / t2vcompbench)。CPU fallback 仅独立 metric。

### 4.2 CPU fallback 范围（v0.3）

仅独立 metric (INTEGRATION doc §5)：
- ✅ PSNR / SSIM / LPIPS / CLIP-Score / Inception-Score / FID / FVD —— 全部能 CPU 跑（慢，仅 dev 用）
- ❌ Benchmark adapter 一律不支持 CPU fallback（视频生成评测在 CPU 上没意义）

---

## 5. 设备抽象层设计

### 5.1 `videvalkit.core.device`（新模块）

```python
# src/videvalkit/core/device.py
from __future__ import annotations
from enum import Enum
import os
import torch


class DeviceKind(str, Enum):
    CUDA = "cuda"
    NPU = "npu"          # Ascend 昇腾
    MLU = "mlu"          # Cambricon 寒武纪（v0.4+ 社区 plugin）
    DCU = "dcu"          # 海光（v0.4+）
    MPS = "mps"          # Apple
    CPU = "cpu"


_PRIORITY = [DeviceKind.CUDA, DeviceKind.NPU, DeviceKind.DCU,
             DeviceKind.MLU, DeviceKind.MPS, DeviceKind.CPU]


def detect_available() -> list[DeviceKind]:
    """返回当前进程可用的设备种类列表，按优先级。"""
    out: list[DeviceKind] = []
    if torch.cuda.is_available():
        out.append(DeviceKind.CUDA)
    if _try_import_torch_npu() and _npu_visible():
        out.append(DeviceKind.NPU)
    # MLU / DCU 同模式 —— v0.4 加
    if torch.backends.mps.is_available():
        out.append(DeviceKind.MPS)
    out.append(DeviceKind.CPU)
    return out


def get_device(preferred: str | None = None) -> torch.device:
    """
    preferred:
      - None / "auto":     按 _PRIORITY 自动选第一个可用
      - "cuda" / "npu" / "cpu" / ...: 指定；不可用则 raise
      - "cuda:0" / "npu:1": 带 index
    """
    available = detect_available()
    if preferred in (None, "auto"):
        return torch.device(available[0].value)
    kind = preferred.split(":")[0]
    if DeviceKind(kind) not in available:
        raise DeviceNotAvailableError(
            f"requested device {preferred!r} not available; "
            f"available: {[d.value for d in available]}"
        )
    return torch.device(preferred)


def device_name(d: torch.device | None = None) -> str:
    d = d or get_device()
    if d.type == "cuda":
        return torch.cuda.get_device_name(d)
    if d.type == "npu":
        import torch_npu  # noqa: F401
        return torch.npu.get_device_name(d)
    return d.type


def empty_cache(d: torch.device | None = None) -> None:
    """device-agnostic cache 清理，无对应实现时 no-op。"""
    d = d or get_device()
    if d.type == "cuda":
        torch.cuda.empty_cache()
    elif d.type == "npu":
        import torch_npu
        torch.npu.empty_cache()
    # else no-op


def autocast_ctx(d: torch.device | None = None, dtype=torch.float16):
    """device-agnostic autocast context manager。"""
    d = d or get_device()
    return torch.amp.autocast(device_type=d.type, dtype=dtype)


def _try_import_torch_npu() -> bool:
    try:
        import torch_npu  # noqa
        return True
    except ImportError:
        return False


def _npu_visible() -> bool:
    return os.environ.get("ASCEND_VISIBLE_DEVICES", "") != "" \
        or _try_import_torch_npu() and torch.npu.is_available()
```

### 5.2 调用方迁移

把现有 42 处 `"cuda" if torch.cuda.is_available() else "cpu"` 类语句统一替换：

```python
# Before
device = "cuda" if torch.cuda.is_available() else "cpu"

# After
from videvalkit.core.device import get_device
device = get_device(preferred=self.requested_device).type
```

也把 `torch.cuda.empty_cache()` → `empty_cache()`，`torch.amp.autocast(device_type="cuda")` → `autocast_ctx()`。

### 5.3 `requested_device` 怎么传进 adapter

```
CLI --device npu
   ↓
runner.run(..., device="npu")
   ↓
adapter.evaluate_and_aggregate(..., device="npu")
   ↓
adapter.evaluate(..., device="npu")
   ↓
内部所有 `get_device("npu")` 调用拿到 npu device
```

`BaseBenchmark.evaluate_and_aggregate` 签名加 `device: str = "auto"` 形参，向后兼容。

---

## 6. CLI 改动

### 6.1 `--device` flag

```bash
# 自动选（CUDA > NPU > MPS > CPU）
videvalkit eval --bench videobench --videos ... --workspace ...
videvalkit eval --bench videobench --device auto ...

# 强制 NPU
videvalkit eval --bench worldjen --device npu ...

# 强制 NPU 的特定卡
videvalkit eval --bench worldjen --device npu:2 ...

# 强制 CPU（仅 metric / device-agnostic bench）
videvalkit metric --name clip-score --device cpu ...

# 拒绝执行的 case
videvalkit eval --bench worldscore --device npu ...
# → Error: worldscore on NPU is partial (6/10 dims supported).
#   Either:
#     (a) use --device cuda
#     (b) restrict dims: --dimensions style,object_count,motion_magnitude,...
#     (c) accept partial run: --allow-partial-npu
```

### 6.2 `doctor` 增强

```
$ videvalkit doctor

== Devices ==
  cuda           --     (not available)
  npu            OK     8 × Ascend 910B (driver 23.0.0, CANN 8.0.0)
  mps            --
  cpu            OK     (always)

== Adapter NPU compatibility ==
  vbench         ⚠ partial  (10/16 dims, see TEST_MANUAL §npu)
  vbench2        ⚠ partial  (12/18)
  videobench     ✓ full     (VLM judge over HTTP)
  worldjen       ✓ full
  worldscore     ⚠ partial  (6/10, DROID-SLAM dims unavailable)
  t2vcompbench   ⚠ partial  (5/7, detectron2 dim unavailable)

== Judges ==
  gemma-4-31b-local  reach=OK   key=--    (openai_compatible)
  ...
```

### 6.3 Bench × device 强制门禁

每 bench 在 registry 声明：

```python
SUPPORTED_BENCHMARKS["worldscore"] = dict(
    cls=WorldScoreBenchmark,
    ...
    device_support={
        "cuda": "full",
        "npu":  "partial",   # 列入下面的 unsupported_dims
        "cpu":  "none",
    },
    unsupported_dims_per_device={
        "npu": ["camera_control", "3d_consistency", "reprojection_error"],
    },
)
```

runner 调度前查表，partial 时：
- 默认要求 `--allow-partial-npu`（避免静默丢 dim）
- 或自动按 `unsupported_dims_per_device` 缩减 `--dimensions`，并 warning

---

## 7. Env 与安装路径

### 7.1 三条独立 env 路径

```
envs/
├── videvalkit.yaml              ← 现有 (CUDA 12.1)
├── videvalkit-npu.yaml          ← 新增 (Ascend, torch_npu)
└── videvalkit-cpu-metrics.yaml  ← 新增 (仅 metric / 开发态)
```

NPU yaml 关键差异：

```yaml
# envs/videvalkit-npu.yaml (摘要)
name: videvalkit-npu
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - pip:
    # 不装 cu121 torch
    - --extra-index-url https://download.pytorch.org/whl/cpu
    - torch==2.3.1
    # Ascend 扩展
    - torch_npu==2.3.1.post2          # 与 torch 版本对应
    # CANN 包通过宿主机驱动提供，不在 conda env 内
    # ...其余 video / numpy / pydantic / aiohttp / 等同 CUDA env
    # 跳过：bitsandbytes, xformers, flash-attn
```

### 7.2 NPU post_install

```bash
# scripts/post_install_npu.sh
set -eu
echo "=== NPU post-install ==="
python -c "import torch, torch_npu; print(f'torch={torch.__version__} torch_npu OK')"

# 跳过：detectron2, lietorch, droid_backends（CUDA-only kernels）
# 安装：SAM-2, GroundingDINO (CPU build → NPU runtime), segment-anything, en_core_web_sm
bash scripts/post_install.sh --minimal

# NPU 特定：装 Ascend 适配版 mmcv（若 dim 需要）
pip install mmcv-ascend==2.2.0  # 假定有此 wheel；否则用社区 fork
```

### 7.3 conda-pack tarball 矩阵

| Tarball | 大小 | HF repo |
|---|---|---|
| `videvalkit-env-cuda121.tar.gz` | ~7.6 GB | `videogenevalkit/env-tarball` 现有 |
| `videvalkit-env-npu-ascend910b.tar.gz` | ~6.0 GB | `videogenevalkit/env-tarball` 新增 (v0.3) |
| `videvalkit-env-cpu-metrics.tar.gz` | ~2.0 GB | 同上，dev 态用 |

下载与 install 流程对称：

```bash
hf download videogenevalkit/env-tarball videvalkit-env-npu-ascend910b.tar.gz --local-dir /tmp
sudo tar xzf /tmp/videvalkit-env-npu-ascend910b.tar.gz -C /opt/videvalkit-env-npu
source /opt/videvalkit-env-npu/bin/activate
conda-unpack
pip install --no-deps -e .
bash scripts/post_install_npu.sh
videvalkit doctor
```

---

## 8. VLM Judge 在 NPU 上

### 8.1 协议层完全 reuse

`openai_compatible` backend 走 HTTP，**与底层硬件解耦**。NPU 上的 judge：

| 后端 | 部署方式 | 端口约定 | 在 `SUPPORTED_JUDGES` 里的形态 |
|---|---|---|---|
| vLLM-Ascend (社区 fork) | `python -m vllm_ascend.entrypoints.openai.api_server ...` | 沿用 8003-8009 | 同 `gemma-4-31b-local` 等 entry，**endpoint 不变** |
| MindIE-LM (华为官方) | `mindie-llm --model X --port 8003 ...` | 同上 | 同上 |
| Hybrid (NPU train 机 + CUDA judge 机) | judge 在另一机 | endpoint 改 host | user yaml 加 entry |

**toolkit 视角无差异** —— 用户只要在 `~/.config/videvalkit/judges.yaml` 写对 endpoint，judge 走的什么硬件由用户自己保障。

### 8.2 文档增量

`docs/USER_MANUAL_*.md` 新增章节 "Running VLM judges on NPU"，覆盖：
1. vLLM-Ascend 安装步骤
2. MindIE-LM 启动命令
3. 各 judge model 在 Ascend 910B 上的实测吞吐（reference data）
4. 已知不兼容情况（如 GPT-OSS 模型的 RoPE scaling 在 vLLM-Ascend 早期版本有 bug）

### 8.3 不做的事

- ❌ 不内置 vLLM-Ascend / MindIE 的部署脚本（属于 judge 部署侧关注，toolkit 只用 endpoint）
- ❌ 不为 NPU 上 judge 跑 paper-faithful 校验 —— 留给用户自行评估

---

## 9. NPU 上的 Profile 与 Subset

跨支柱协同（与 QUICK_EVAL_DESIGN.md §3 衔接）：

```yaml
profiles:
  quick:
    ...
    device_requirements:
      preferred: cuda            # 不是强制
      acceptable: [cuda, npu]
      unacceptable: [cpu, mps]
```

`videvalkit estimate` 在 NPU 上的输出额外标注：

```
┌───────────────┬──────────┬───────┬──────────────────┐
│ Benchmark     │ Wallclock│ NPU-h │ NPU compat       │
├───────────────┼──────────┼───────┼──────────────────┤
│ vbench        │ 8 min    │ 0.13  │ ⚠ 10/16 dims     │
│ worldjen      │ 8 min    │ 0.05  │ ✓ full           │
│ worldscore    │ —        │ —     │ ❌ DROID-SLAM    │
└───────────────┴──────────┴───────┴──────────────────┘

Suggestion: worldscore needs CUDA. Run those on a separate machine.
```

---

## 10. 文件改动清单

### 10.1 新增

| 路径 | 用途 |
|---|---|
| `src/videvalkit/core/device.py` | 设备抽象 (§5.1) |
| `envs/videvalkit-npu.yaml` | Ascend env (§7.1) |
| `envs/videvalkit-cpu-metrics.yaml` | CPU dev env |
| `scripts/post_install_npu.sh` | NPU 后装步骤 |
| `scripts/build_npu_tarball.sh` | 打 conda-pack |
| `src/videvalkit/configs/devices.py` | 设备 × bench 兼容矩阵（registry） |
| `docs/NPU_ADAPTATION_DESIGN.md` | 本文 |
| `docs/USER_MANUAL_npu.md` | NPU 用户专用安装与运行指南 |
| `tests/test_device_abstraction.py` | 抽象层单测 + mock npu 路径 |
| `tests/test_npu_compat_matrix.py` | bench × device gate 单测 |
| `examples/npu_quick_eval.sh` | 端到端 NPU 跑通示例 |

### 10.2 修改（22 处）

| 类别 | 文件 | 改点 |
|---|---|---|
| **核心** | `core/benchmark.py` | `evaluate_and_aggregate` 加 `device` 参数 |
| | `runner.py` | `run()` 加 `device` 形参，转发到 adapter |
| | `cli.py` | `eval` / `eval-suite` / `metric` 加 `--device` flag |
| | `diagnostics.py` | `doctor` 报告设备 + 兼容矩阵 |
| **配置** | `configs/benchmarks.py` | 每个 entry 加 `device_support` + `unsupported_dims_per_device` |
| **vbench** | `benchmarks/vbench/benchmark.py` | 2 处 device 三元 → `get_device()` |
| **vbench2** | `benchmarks/vbench2/benchmark.py` | 2 处同上 |
| **t2vcompbench** | `benchmarks/t2vcompbench/scorers.py` | 5 处 device kwarg 收敛 |
| | `benchmarks/t2vcompbench/benchmark.py` | 1 处 |
| **worldscore** | `benchmarks/worldscore/scorers.py` | 4 处 `.to("cuda")` → 抽象（部分 dim 直接 raise） |
| | `benchmarks/worldscore/runners/*.py` | 14 处 device hardcode → 抽象 |
| | `benchmarks/worldscore/runners/static_dims.py` | `empty_cache()` 改包装 |
| **post_install** | `scripts/post_install.sh` | doc 加一句"NPU 用户用 post_install_npu.sh" |
| **doc** | `README.md` | 加 "NPU support" 一段链接到 USER_MANUAL_npu.md |
| | `docs/PRODUCT_DESIGN.md` | §3.1 L0/L1 加 NPU 设备列；§6 路线图 v0.3 加条 |
| | `docs/DEV_MANUAL.md` | §8.2 加 NPU 兼容矩阵；§5.2 Module B 加 device_support |
| | `docs/TEST_MANUAL.md` | 加"NPU 上各 bench 偏差表"占位 |
| | `docs/USER_MANUAL_{en,cn}.md` | 主文 link 到 USER_MANUAL_npu.md |

### 10.3 删除

无。

---

## 11. 兼容性

| 现有用法 | v0.3 行为 |
|---|---|
| `videvalkit eval --bench X ...`（不传 --device） | `--device auto` → CUDA 机上选 cuda，NPU 机上选 npu，等同当前 |
| 现有 conda env (CUDA 12.1) | 不动 |
| 现有 worldscore 用户 | 不动；NPU 用户拿到 fail-fast 错误而非 4 小时后崩 |
| `import videvalkit; ...` | API 加 `device` 参数，默认 "auto"，旧调用代码不变 |

---

## 12. 里程碑

| 阶段 | 内容 | 工作量 |
|---|---|---|
| **E1 — 设备抽象层** | `core/device.py` + 替换 42 处 hardcoded `"cuda"` + 单测 | 1.5 day |
| **E2 — Bench × device 矩阵 + fail-fast** | `device_support` 字段 + runner 门禁 + doctor 报告 | 1 day |
| **E3 — NPU env** | `videvalkit-npu.yaml` + post_install_npu.sh + Ascend 机器实测 4 个 bench | 2 day |
| **E4 — VLM judge on NPU 文档** | USER_MANUAL_npu.md vLLM-Ascend / MindIE 章节 + 一组 reference 吞吐数 | 0.5 day |
| **E5 — CPU fallback for metrics** | 7 个独立 metric 加 CPU 路径 + 单测 | 1 day |
| **E6 — conda-pack tarball + HF 上传** | 打包 + 测试解压 → conda-unpack → 用户走通 | 1 day |
| **E7 — 实机端到端验证** | Ascend 910B 上跑 videobench / worldjen / vbench-quick / vbench2-quick；记录偏差 | 1 day |
| **E8 — 文档 + 路线图回写** | TEST_MANUAL NPU 偏差表 + PRODUCT_DESIGN 回写 | 0.5 day |

**总计 ≈ 8.5 person-days**。**当前不排期**——留作 v0.4+ 候选，启动时按本节工作量评估。

### 12.1 依赖关系

```
E1 ──► E2 ──► E3 ──► E7 ──► E8
                       │
E4 ──── 与 E3 并行 ───►┘
                       │
E5 ──── 与 E3 并行 ───►┘
                       │
E6 ──── 串行依赖 E3 ──►┘
```

E1/E2 是基础设施（约 2.5 day），可在 v0.2 末期开始；其余进 v0.3。

---

## 13. Open Questions

> 评审时需要决策的点：

1. **Ascend NPU 实机谁来跑**？团队内有 910B 机器还是要找合作伙伴借？倾向 **联系一家国产 T2V 团队（如 Kling / Moonshot）借机 1 周做 E7**。
2. **是否要在 NPU env 里同时支持 910 / 910B / 910C 三代**？倾向 **v0.3 只支持 910B（最常见）**，910C 等社区贡献。
3. **MindIE 与 vLLM-Ascend 是否都做文档**？两者协议都是 OpenAI-compat，但部署命令差异大。倾向 **两者都写一段最简启动命令**，让用户自选。
4. **partial 模式 default 是 fail 还是 warn**？`videvalkit eval --bench worldscore --device npu` 默认应该报错还是自动 drop 不兼容 dim？倾向 **默认报错**（让用户显式 `--allow-partial-npu`），避免静默丢分。
5. **Cambricon MLU 是否进 v0.3**？还是只画占位？倾向 **占位**（写在 device.py 注释里，社区贡献 plugin）。
6. **海光 DCU 大部分 CUDA 代码能直跑（HIP 翻译层），是否声明"近似 CUDA"档**？倾向 **不**，用户 `--device cuda` 在 DCU 上能跑就跑，但我们不背书。
7. **NPU 上跑出的数与 CUDA 数的偏差容忍度**？倾向 **mean \|Δ\| ≤ 0.05** 之内可接受（vs CUDA 上 0.005-0.012，NPU 上松一档）；超出的 dim 在 TEST_MANUAL 列出。
8. **NPU env tarball 是否上 HF**？尺寸 ~6GB，与 CUDA tarball 同 repo。倾向 **上**，与 CUDA 对称。

---

## 14. 风险

| 风险 | 影响 | 对策 |
|---|---|---|
| Ascend 实机难借 | E7 拖延 | 提前联系 2-3 家备选；E1/E2 用 mock torch_npu 先做 |
| torch_npu 版本与 torch 版本绑得死，跟 CUDA env 漂移 | NPU env 维护成本高 | torch 2.3.1 → torch_npu 2.3.1.post2 锚定；升级时 CUDA/NPU 同步 |
| vLLM-Ascend 社区 fork 落后 vLLM 主线 | judge 性能 / 兼容性差 | doc 注明实测版本；用户可换 MindIE |
| NPU 上 LLaVA-1.6-34B 内存 / 速度差异大 | paper-faithful 复现偏差超 0.05 | TEST_MANUAL 单独一节标注；不在 NPU 上承诺 paper-faithful |
| 42 处 device hardcode 替换引入回归 | CUDA 用户体验下降 | E1 单测覆盖每处替换；end-to-end 现有 6 bench smoke test 全跑一遍 |
| 用户 NPU 驱动 / CANN 版本不匹配 | 安装失败 | `doctor` 检查 CANN 版本；明确错误提示 |
| 国产生态多 NPU 厂商（Ascend/MLU/DCU/musa）维护成本爆炸 | 长期 toil | v0.3 只交 Ascend；其余社区 plugin 进 `~/.videvalkit/devices/` |
| MindIE 协议未来与 OpenAI-compat 偏离 | judge 不通 | 监控 MindIE release；偏离时加 adapter 层 |

---

## 15. 决策快照

> 评审本文时一次性确认（"同意"即采纳）：

- ✅ **NPU 适配是横切关注，不是新支柱** —— 跨 A/B/C/D 四支柱
- ✅ v0.3 主要目标：Ascend 910B；其他 NPU (MLU / DCU / musa) 留 v0.4+ plugin
- ✅ 设备抽象层 `videvalkit.core.device` 统一管 device 选择 + 缓存清理 + autocast
- ✅ CLI 新增 `--device auto|cuda|npu|cpu|...`，默认 auto
- ✅ Bench × device 兼容矩阵 first-class（registry 字段），doctor 展示
- ✅ partial-on-NPU 默认 fail-fast，需 `--allow-partial-npu` 显式开
- ✅ NPU env tarball 与 CUDA tarball 并列，~6GB，上 HF
- ✅ vLLM-Ascend 与 MindIE 都写文档，协议都是 OpenAI-compat（toolkit 无感知差异）
- ✅ NPU 上**不承诺** paper-faithful；mean \|Δ\| ≤ 0.05 是松目标
- ✅ v0.3 不含 NPU CI；用 self-hosted 或 manual verification（E7 实机验收）
- ✅ DROID-SLAM / lietorch / detectron2 在 NPU 上明确不支持，相关 dim 标 ❌
- ✅ CPU fallback 仅独立 metric (7 个)；bench adapter 全部不支持 CPU
- ✅ paper-faithful 复现的 leaderboard 数字只在 CUDA 上发布
- ✅ **当前不排期** —— v0.2/v0.3 不含 NPU；本文作为未来启动时的设计起点
- ✅ 工作量约 8.5 person-days，启动时一次性做完，不前置任何子任务

---

## 16. 与其他设计文档的关系

| 文档 | 关系 |
|---|---|
| [`PRODUCT_DESIGN.md`](PRODUCT_DESIGN.md) | §3.1 L0/L1 加 NPU 设备；§6 路线图 v0.3 加 NPU；§7 跨支柱原则新增第 9 条 "Device-agnostic core, hardware-specific install" |
| [`JUDGE_SELECTION_DESIGN.md`](JUDGE_SELECTION_DESIGN.md) | Judge 走 HTTP 协议天然 device-agnostic，无改动；本文 §8 补 NPU 上部署 judge 的文档增量 |
| [`INTEGRATION_FRAMEWORK_DESIGN.md`](INTEGRATION_FRAMEWORK_DESIGN.md) | Manifest schema 加可选 `device_support` 字段；plugin 加 metric 时声明 device 支持；本文 §5 抽象层供 plugin 直接复用 |
| [`QUICK_EVAL_DESIGN.md`](QUICK_EVAL_DESIGN.md) | Profile 加 `device_requirements` 字段；`videvalkit estimate` 在 NPU 输出补 "NPU compat" 列；本文 §9 |

### 16.1 排期说明（修订）

> **2026-05-20 调整**：NPU 适配**不进 v0.2/v0.3 排期**。
>
> - v0.2 保持原计划 16.5 day（A+B+C+D 四支柱，CUDA-only）
> - v0.3 保持原计划（3 stub bench / 国产 judge / compare-leaderboard），不含 NPU
> - 本文 §12 列出的 E1–E8 工作量（≈ 8.5 day）作为未来启动时的预估，**不前置任何任务到 v0.2**
> - 若未来启动 NPU 适配，E1/E2（设备抽象层 ≈ 2.5 day）届时一次性做完，不再尝试与其他支柱并行

---

—— end of design v0.1 ——
