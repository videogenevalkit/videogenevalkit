# NPU 适配 — 设计(大部分后移)

> **状态:大部分后移。** 完整的基准 NPU 支持不在短期路线图内。**已落地部分管道**:
> 设备无关的核心指标(FVD / VFID / KVD / CLIP-FVD / CLIP-Score / ViCLIP-Score)
> 通过 `metrics/utils/device.py`(`resolve_device`)解析 `--device npu`——导入
> `torch_npu` 以注册后端,缺失时回退到 cpu 并告警。此路径**未在真实昇腾硬件上验证**
> (开发机仅 CUDA)——上报 NPU 数字前请先在设备上验证。本存根其余部分记录剩余工作的
> 意图与约束。

---

## 意图

让工具包中与设备无关的部分(多数指标、无需评审的基准、经 HTTP 的 VLM 评审调用)
在昇腾 NPU 上运行,并在某基准依赖 CUDA-only 内核处显式、诚实地降级。

## 为何后移

当前聚焦 Linux + CUDA。NPU 会引入一个环境矩阵和逐算子移植,在出现明确需求之前
并不划算;v0.2 设计中没有任何东西会在日后阻碍它。

## 恢复时须遵守的约束

- **设备选择**应自动优先 CUDA > NPU > MPS > CPU,并支持显式覆盖
  (`--device npu`);当某基准在 NPU 上仅部分支持时快速失败(提供:换设备、
  限制维度,或接受部分运行)。
- **CUDA-only 内核**(如 detectron2、lietorch、droid_backends)没有 NPU 构建——
  受影响的基准维度是不支持,而非悄悄出错。SAM-2、GroundingDINO、segment-anything
  有 CPU/NPU 构建。
- **独立环境**(`videvalkit-npu`)保持 CUDA 环境干净;插件加载器与注册表与设备
  无关,无需改动。
- **VLM 评审**不受影响——它们是对某端点的 HTTP 调用,与本地设备无关。
- **配置档/子集**原样沿用;只是计算后端不同。

需求出现时,把它作为独立里程碑立项,并为每个基准维度建一份设备覆盖矩阵。

## 环境(910B)

独立 env 保持 CUDA 环境干净。草稿模板 + 安装器:

- `envs/videvalkit-npu.yaml` — conda env(python 3.10 + ffmpeg + 设备无关 pip 依赖);
  版本为占位符,需按宿主机 CANN 版本固定。
- `scripts/post_install_npu.sh` — 安装 torch + `torch_npu`(与 CANN 匹配)+ 依赖
  torch 的包(openai-clip、pyiqa、decord/eva-decord);`INSTALL_VBENCH=1` 加上抽出的
  vbench 维度。
- `scripts/npu_smoke.py` — 安装后在设备上跑,输出易档指标的 PASS/FAIL 报告。

NPU 易档 = 6 个 canonical 指标 + 5 个 vbench lift(temporal-flickering、
subject/background-consistency、aesthetic/imaging-quality)。`.cuda()`→npu 重定向由
`core.device.ensure_npu_runtime()`(`torch_npu.contrib.transfer_to_npu`)处理,
设备解析为 npu 时激活。
