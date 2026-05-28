# 视频指标 — 设计

> 独立指标层的设计理念。实时的逐指标目录(名称、骨干、状态)见
> [指标参考](../wiki/reference/Metrics.md);本文解释其背后的*结构*。

---

## 1. 设计原则

- **两档分类。** 通用 T2V 质量(适用于任意视频)vs. 专项维度(组合性 / 伪影 /
  身份)。这种划分让常用、无需评审的指标易于触达,并把更重、常受评审/prompt
  约束的指标隔离开。
- **双入口,单实现。** 一个指标有两种触达方式——`eval --bench X --dimensions Y`
  和 `metric run --name Y`——两者调用*同一份*代码。不存在会漂移的第二份实现。
- **默认无需评审。** 多数指标无需 VLM/LLM;`needs_judge` 标出例外,
  以便 `--no-judge` 过滤。
- **不交付不适用的指标。** PSNR/SSIM/LPIPS/FID-image 是需要 ground-truth 视频的
  基于参考的逐帧指标;T2V 没有 ground truth,故排除(它们属于 I2V/V2V/重建)。

---

## 2. 两档

**第 1 档 — 通用 T2V 质量**(除注明外均无需评审):

- *分布型*(需参考集):`fvd`、`vfid`、`kvd`、`clip-fvd`。
- *文本-视频对齐*(需 prompt):`clip-score`、`viclip-score`。
- *帧感知*(从 vbench 抽出):`aesthetic-quality`、`imaging-quality`。
- *时序*(从 vbench/worldscore 抽出):`motion-smoothness`、
  `temporal-flickering`、`subject-consistency`、`background-consistency`、
  `dynamic-degree`、`motion-magnitude`。

**第 2 档 — 专项维度**:`numeracy`、`spatial-relationship`(CV,无需评审)、
`artifact-diagnostic`(MLLM 评审),以及 `object-binding` 和 `motion-accuracy`
——后两者受 prompt/评审 约束,因此**仅基准**(经由其基准运行,而非独立
`--videos` 指标),还有 `identity-preservation`(后移到 i2v 阶段)。

---

## 3. 双入口契约(逐位一致 lift)

*lift*(抽出)把某个基准维度暴露为独立指标,做法是包裹基准适配器所发起的
**同一次上游调用**。具体地:

- `motion-smoothness`(独立)包裹与 `eval --bench vbench --dimensions
  motion_smoothness` 相同的
  `VBench.evaluate(dimension_list=["motion_smoothness"], mode="custom_input")` 调用。
- `motion-magnitude` 经由相同的 `OpticalFlowScorer` 包裹 worldscore 基准的
  `OpticalFlowMetric`(SEA-RAFT)。

契约:两个入口之间结果须**逐位一致(≤ 1e-6)**。这作为 `lift-out` PR 关卡强制
执行(见[评审协议](REVIEW_PROTOCOL.md))。收益:能力层与独立 CLI 复用基准机制,
公式零分叉。

分布型指标是例外——它们没有基准来源,直接继承 `BaseDistributionMetric`,
共享 Fréchet(`frechet.py`,float64)与多项式 MMD(`mmd.py`)工具及骨干加载器。

---

## 4. 骨干策略

| 骨干 | 使用者 | 备注 |
|---|---|---|
| S3D-K400 | FVD/KVD 默认 | 自动下载;有效的 Kinetics-400 趋势指标——适合监控 |
| I3D-K400 | FVD(论文) | torchscript;放置权重后即为论文版(`--backbone i3d-k400`) |
| InceptionV3 | VFID | torchvision,自动 |
| CLIP-ViT-L/14 | clip-score, clip-fvd | openai-clip,环境内 |
| ViCLIP-L/14 | viclip-score | 已 vendor;权重自 OpenGVLab 自动获取 |

缺论文权重时 FVD 自动 i3d→s3d 回退,监控开箱即用;论文数字需显式指定 I3D 权重。

---

## 5. 注册表 schema

`SUPPORTED_METRICS` 中每条都带必填字段 `kind`、`source`、`cls`、`needs_judge`、
`compute_kind`、`tags`(一致性检查会强制)。`kind` 驱动 runner 派发:

| `kind` | 输入 |
|---|---|
| `distribution_reference` | 生成视频 + 参考集 |
| `per_prompt_reference_free` | 视频 + prompt |
| `per_video_reference_free` | 视频 |
| `per_video_with_vlm_judge` | 视频 + 评审 |

`tags` 引用[能力词表](CAPABILITY_TAGS_DESIGN.md);`source` 记录来源
(`canonical/...` 或 lift 的 `<bench>/<dim>`);lift 还设 `also_used_by` 供能力
解析器去重。

---

## 6. artifact-diagnostic(Artifact-Bench 移植)

[Artifact-Bench](https://github.com/FrankYang-17/Artifact-Bench)(arXiv
2605.18984)的 v0.2 切片:在 30 类分类法(3 类 → 11 族 → 30 叶,见
`metrics/artifact_taxonomy.py`)上的 MLLM 多标签检测器。它是
`per_video_with_vlm_judge` 指标,研究专用许可,且有意与部分 lift 重叠
(`flickering`~temporal-flickering、`identity_drift`~subject-consistency)。
完整的评审评测基准是 v0.3 交付物;分类法叶子是待与论文核对的工作版。
