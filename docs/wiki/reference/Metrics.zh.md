# 指标参考

[← 首页](../../index.md)

计划中共 20 个独立指标,分两层。**目前 16 个无需评审即可运行**,外加
`artifact-diagnostic`(已注册,用 `--judge` 运行)。其余 3 个
专项维度经由各自基准提供或已后移——见第 2 层。
任意一个都可用 `videvalkit metric run --name <name>` 运行。

> **可用(Functional)** = 今天就能跑(骨干自动下载、已在环境内,或包裹已暂存的上游)。
> 状态以 v0.2-dev 为准。

---

## 第 1 层 — 通用 T2V 质量(14)

### 分布级(4)— 需要参考集,无需评审

| 指标 | 骨干 | 状态 | 标签 |
|---|---|---|---|
| `fvd` | S3D-K400(自动)/ I3D-K400(论文) | ✅ 可用(默认 s3d) | realism.distribution |
| `vfid` | InceptionV3(自动) | ✅ 可用 | realism.distribution |
| `kvd` | S3D-K400 + poly-MMD² | ✅ 可用 | realism.distribution |
| `clip-fvd` | CLIP-ViT-L/14 | ✅ 可用(实验性) | realism.distribution |

> FVD/KVD 监控时默认 S3D-K400(Kinetics-400,自动下载)。若要论文版
> I3D-FVD,放置 `i3d_torchscript.pt` 并传 `--backbone i3d-k400`。
> CLIP-FVD 用的是 CLIP 特征空间——与标准 FVD **不可比**。

### 文本-视频对齐(2)— 需要 prompt,无需评审

| 指标 | 骨干 | 状态 | 标签 |
|---|---|---|---|
| `clip-score` | CLIP-ViT-L/14 | ✅ 可用 | align.text2video |
| `viclip-score` | ViCLIP-L/14(自动获取) | ✅ 可用 | align.text2video, align.prompt_following |

### 帧感知(2)— 从 vbench 抽出,无需评审

| 指标 | 来源 | 状态 | 标签 |
|---|---|---|---|
| `aesthetic-quality` | vbench (LAION) | ✅ 可用* | vq.aesthetic, style.aesthetic |
| `imaging-quality` | vbench (MUSIQ) | ✅ 可用* | vq.imaging, vq.sharpness |

### 时序(6)— 从 vbench/worldscore 抽出,无需评审

| 指标 | 来源 | 状态 | 标签 |
|---|---|---|---|
| `motion-smoothness` | vbench (AMT) | ✅ 可用* | motion.smoothness, temp.flickering |
| `temporal-flickering` | vbench | ✅ 可用* | temp.flickering, vq.artifact_free |
| `subject-consistency` | vbench (DINO) | ✅ 可用* | subj.identity, subj.appearance |
| `background-consistency` | vbench (CLIP) | ✅ 可用* | subj.appearance, temp.continuity |
| `dynamic-degree` | vbench (RAFT) | ✅ 可用* | motion.magnitude |
| `motion-magnitude` | worldscore (SEA-RAFT) | ✅ 可用† | motion.magnitude |

*在有 vbench 检查点时可用;包裹上游 `VBench.evaluate(dimension_list=[dim])` → 与基准路径逐位一致。
†在有 worldscore 上游(`$VIDEVALKIT_WORLDSCORE_ROOT` + SEA-RAFT 权重)时可用;包裹与基准相同的 `OpticalFlowMetric` 调用 → 逐位一致。

---

## 第 2 层 — 专项维度(6)

| 指标 | 来源 | 评审? | 状态 | 标签 |
|---|---|---|---|---|
| `numeracy` | t2vcompbench (GroundingDINO) | 否 | ✅ 可用 | comp.numeracy, obj.count |
| `spatial-relationship` | t2vcompbench (GDINO+Depth) | 否 | ✅ 可用 | comp.spatial |
| `artifact-diagnostic` | Artifact-Bench 移植 | 是 | ✅ 已注册(需 `--judge`) | vq.artifact_free |
| `object-binding` | t2vcompbench (MLLM) | 是 | ↪ 仅基准(`eval --bench t2vcompbench`) | obj.binding, obj.presence |
| `motion-accuracy` | worldscore (RAFT+SAM2) | 是 | ↪ 仅基准(`eval --bench worldscore`) | motion.accuracy, align.action_verb |
| `identity-preservation` | ArcFace(新) | 否 | ⏳ 后移到 i2v 阶段 | subj.identity, subj.character |

---

## 状态图例

| | 含义 |
|---|---|
| ✅ 可用 | 今天能跑(无需评审,或已注册 + 用 `--judge` 运行) |
| ↪ 仅基准 | 受 prompt/评审 条件约束的维度;经由其基准运行,而非独立指标 |
| ⏳ 后移 | 出 v0.2 范围(后续阶段再做) |

---

## v0.2 排除项

PSNR / SSIM / LPIPS / FID-image——这些是**基于参考的逐帧**指标,
需要 ground-truth 视频。T2V 没有 ground truth,因此不适用
(仅对 I2V / V2V / 重建有用)。不交付。

另见:[能力标签](Capability-Tags.md) · [CLI: metric run](CLI.md#metrics--references)
