# 路线图与状态

[← 首页](../index.md)

> 快照:`v0.2-dev` · 384 个测试全绿 · v0.2 接近完成。

---

## 版本时间线

| 版本 | 日期 | 主题 | 状态 |
|---|---|---|---|
| v0.0.1 | 2026-05-13 | 锚定适配器 | ✅ |
| v0.1.0 | 2026-05-19 | 验证(已发布平均 \|Δ\|) | ✅ |
| **v0.2.0** | ~2026-07 | 可扩展性 + 指标 + 能力 + 训练监控 | 🔵 接近完成 |
| v0.3.0 | 稍后 | 占位基准转实做 · 国产 VLM · 脚手架 CLI | ⏳ |
| v0.4.0 | 稍后 | macOS/CPU 子集 · 榜单站点 · NPU | ⏳ |
| v1.0 | 2026 Q4 | 论文 · colab · 评审评测子项目 | ⏳ |

---

## v0.2 — 已完成

| 领域 | 交付内容 |
|---|---|
| 评审选择 | `--judge paper/default/<name>` · 用户 `judges.yaml` · 临时端点 · `--no-judge` |
| 插件加载器 | 三层发现(内置 / entry_points / 本地目录) |
| Manifest 基准 | YAML Track-A 适配器 |
| 指标(16 个可用) | FVD · VFID · KVD · CLIP-FVD · CLIP-Score · ViCLIP-Score · 7 个 vbench lift · motion-magnitude · numeracy · spatial-relationship · artifact-diagnostic(需评审) |
| 指标/参考集 CLI | `metric list/show/run` · `refs list/show/register` |
| 快速评测 | 3 个配置档 · 子集 · `estimate` · `eval-suite` |
| 训练监控 | `watch` · `videvalkit.training.monitor` Python API |
| 能力标签 | 44 标签词表 · 解析器 · `capabilities list/show/eval` |
| 质量 | 评审协议 + CI(3 个检查脚本)· 增强版 `doctor` |
| 文档 | 本 wiki |

---

## v0.2 — 剩余

| 项 | 状态 / 阻塞 |
|---|---|
| `artifact-diagnostic` 指标 | ✅ 已实现 + mock 测试;配置好 `--judge` 端点即可运行。分类法叶子待与论文核对;完整评审评测基准为 v0.3。 |
| `object-binding` | **经由基准可用**(`eval --bench t2vcompbench --dimensions consistent_attribute`,MLLM 评审)。不是独立 `--videos` 指标——只会重复同一次评审调用。 |
| `motion-accuracy` | **经由基准可用**(`eval --bench worldscore --dimensions motion_accuracy`)。受 prompt 条件约束(预期运动方向)→ 天然 bench-only,非独立指标。 |
| `identity-preservation` | **后移到 i2v 阶段**——其主用途(参考图身份匹配)偏 i2v,出 v0.2 T2V 范围。 |
| 论文版 I3D-FVD 运行 | `i3d_torchscript.pt` 托管(加载器就绪;S3D 回退覆盖监控) |
| `fetch-refs` 跨机器 | `videogenevalkit/reference-videos` HF 数据集上传(命令已交付;`refs register` 是零下载的本地路径) |

解锁杠杆:**启动一个评审端点**(→ artifact-diagnostic)、**托管 I3D
torchscript 权重**(→ 论文版 FVD)。

---

## 指标计分板

| | 数量 | 哪些 |
|---|---|---|
| ✅ 可用 | 16 | fvd, vfid, kvd, clip-fvd, clip-score, viclip-score, 7 个 vbench lift, motion-magnitude, numeracy, spatial-relationship |
| ✅ 已注册(需评审) | 1 | artifact-diagnostic(Artifact-Bench 移植;用 `--judge` 运行) |
| ↪ 仅基准 / ⏳ 后移 | 3 | object-binding, motion-accuracy(经由各自基准);identity-preservation(后移到 i2v) |

详情见[指标参考](reference/Metrics.md)。
