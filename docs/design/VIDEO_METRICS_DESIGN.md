# Video Metrics — Two-Tier Catalog & Dual-Entry Design Doc

| 字段 | 内容 |
|---|---|
| 版本 | v0.2 (2026-05-20 大幅重构) |
| 状态 | Design — 9 条决策点已用户确认（2026-05-20） |
| 性质 | 支柱 C (INTEGRATION) 的核心子专题 —— v0.2 高优交付 |
| 影响范围 | `metrics/` 新模块 · `BaseDistributionMetric` · CLI `metric` 子命令 · 6 anchored bench 的 lift-out 重构 · refs 与 backbone 管理 |
| 关联文档 | [`INTEGRATION_FRAMEWORK_DESIGN.md`](INTEGRATION_FRAMEWORK_DESIGN.md) §5 · [`QUICK_EVAL_DESIGN.md`](QUICK_EVAL_DESIGN.md) §3, §6 · [`PRODUCT_DESIGN.md`](PRODUCT_DESIGN.md) §4.1 |
| 目标读者 | T2V 模型研发者（训练监控） · paper 复现者 · 算法贡献者 |

---

## 1. 设计原则

> v0.2 metric 工作的产品偏好（用户 2026-05-20 评审确认）：
>
> **"用户不需要看到基础设施，需要看到的是大量现成可用的、统一接口的指标。"**

由此决定：

| 偏好 | 落地 |
|---|---|
| **重集成，轻框架** | Plugin scaffolding (`videvalkit new` / `validate bench`) 推到 v0.3；v0.2 集中交付 **14+6 = 20 个 metric**（含 artifact-diagnostic）|
| **复用现有优秀算法** | 6 个 anchored bench 里现有的 12+ 个高质量 scorer **lift-out** 成 standalone metric |
| **同一算法两个入口** | `eval --bench` (粗粒度) 与 `metric --name` (细粒度) 共用单一实现，位级一致 |
| **三类指标只保留两类** | 通用 T2V quality 与 专用维度；reference-based 窄适用 (PSNR/SSIM/LPIPS/FID-image) 移出 v0.2 |
| **PSNR/SSIM 不适用 T2V quality** | T2V 没有 ground-truth；这类指标仅 I2V/V2V/reconstruction 用，v0.2 不交付 |

---

## 2. 两档分类

```
┌──────────────────────────────────────────────────────────────────────────┐
│  通用 T2V quality（高优首选）                v0.2 交付 14 个（全 judge-free）│
│  ───────────────────────────                ─────────────────              │
│  任何 T2V 模型都该跑的"基础体检"               4 distribution + 2 alignment │
│                                              + 2 frame perceptual + 6 temporal │
├──────────────────────────────────────────────────────────────────────────┤
│  专用维度（按需挑）                          v0.2 交付 6 个                │
│  ─────────────────                          ──────────────                │
│  测某个具体能力（物体保真 / 数量 / 身份 /     4 lift + 1 new + 1 paper-port │
│  artifact 诊断 ...）                         (artifact-diagnostic)         │
│                                              其中 3 个需 VLM judge          │
└──────────────────────────────────────────────────────────────────────────┘

   v0.2 不交付：PSNR / SSIM / LPIPS / FID-image (窄场景 ref-based)
                Camera Pose / 3D Consistency  (CUDA-only kernel)
                OCR / Physics / Action / DOVER / MANIQA / BLIP2 (推 v0.3)
```

---

## 3. 通用 T2V quality（v0.2 必交，14 个）

### 3.1 Distribution-level (4 个，全部新写)

| Metric | Backbone | Canonical impl 来源 | 备注 |
|---|---|---|---|
| `fvd` | I3D-K400 | stylegan-v port | **paper canonical**；T2V 社区报数标配 |
| `vfid` | InceptionV3 + mean-pool 帧聚合 | 自实现 | 比 FVD 简单，per-frame 路径 |
| `kvd` | I3D-K400 (复用 FVD backbone) | 自实现 polynomial kernel | 小 N (< 500) 比 FVD 稳定 |
| `clip-fvd` | CLIP-ViT-L/14 | 自 wrap | **experimental**；与 FVD 不可比 |

输入需求：`distribution_reference` — 需要 gen videos + ref videos。

### 3.2 Text-video alignment (2 个，全部新写)

| Metric | 算法 | 备注 |
|---|---|---|
| `clip-score` | CLIP-ViT-L/14 per-frame text-image 余弦 → mean | 业界 baseline |
| `viclip-score` | ViCLIP per-clip text-video 余弦 | per-clip 比 per-frame 准，T2V 推荐 |

输入需求：`per_prompt_reference_free` — 需要 videos + prompts。

### 3.3 Frame perceptual quality (2 个，lift-out)

| Short canonical | Source | 算法 |
|---|---|---|
| `aesthetic-quality` | `vbench/aesthetic-quality` | LAION-Aesthetic V2 |
| `imaging-quality` | `vbench/imaging-quality` | MUSIQ (image quality assessment) |

输入需求：`per_video_reference_free` — 只需 videos。

### 3.4 Temporal quality (6 个，lift-out)

| Short canonical | Source | 算法 |
|---|---|---|
| `motion-smoothness` | `vbench/motion-smoothness` | AMT 插帧重建误差 |
| `temporal-flickering` | `vbench/temporal-flickering` | 相邻帧 MSE 时序方差 |
| `subject-consistency` | `vbench/subject-consistency` | DINO 特征余弦跨帧 |
| `background-consistency` | `vbench/background-consistency` | CLIP 特征余弦跨帧 |
| `dynamic-degree` | `vbench/dynamic-degree` | RAFT 光流幅度 |
| `motion-magnitude` | `worldscore/motion-magnitude` | SEA-RAFT 光流幅度（与 dynamic-degree 算法不同） |

输入需求：`per_video_reference_free`。

---

## 4. 专用维度（v0.2 按需挑，6 个）

| Short canonical | Source | 算法 | 测什么 | Judge? |
|---|---|---|---|:-:|
| `object-binding` | `t2vcompbench/object-binding` | GroundingDINO + MLLM 检测 prompt 提到的 object | 名词/物体保真 | ✓ VLM |
| `spatial-relationship` | `t2vcompbench/spatial-relationship` | GroundingDINO + bbox 几何 + MLLM | 空间语言（左右上下里外） | ✓ VLM |
| `numeracy` | `t2vcompbench/numeracy` | object count vs prompt 数字 | 数量准确性 | ✗ |
| `motion-accuracy` | `worldscore/motion-accuracy` | RAFT + GroundingDINO + spaCy 动词对齐 | 动作准确性 | ✗ |
| `identity-preservation` | new (ArcFace) | ArcFace 跨帧 + 与 ref image 余弦 | 人物/角色保持 | ✗ |
| **`artifact-diagnostic`** | `canonical/artifact-bench-port` | 30 类 artifact MLLM-as-judge 多选 | **30 维 artifact 频率诊断** | **✓ VLM** |

输入需求：

| Kind | 哪些 metric |
|---|---|
| `per_prompt_reference_free` | object-binding / spatial-relationship / numeracy / motion-accuracy |
| `per_video_with_ref_image` | identity-preservation |
| `per_video_with_vlm_judge` | artifact-diagnostic |

### 4.1 artifact-diagnostic 细节（v0.2 新增，2026-05-20 用户决策）

**来源**：[Artifact-Bench (arXiv 2605.18984, 2026-05)](https://arxiv.org/abs/2605.18984) · paper 的 AID 任务方法论 lift 出来跑用户视频。

**算法**：
1. 加载 paper 的 30 类 artifact taxonomy（3 大类 → 11 失效族 → 30 细粒度）
2. 对每个用户输入视频，让 VLM judge 看视频 + 30 类 artifact 描述
3. 判定每个 artifact 是否出现（0/1 或 logit-based 置信度）
4. 输出：每视频 30 维 artifact 频率向量 + top-K artifact + 总 artifact rate

**与 Artifact-Bench 原 task 的区别**：
- Artifact-Bench 原 task 3 (AID) 是 **6 选多**（从 paper 给的 6 个候选里选哪几个 artifact 存在），用来评测 judge
- 我们的 `artifact-diagnostic` 是 **30 选多**（让 judge 在完整 taxonomy 上判定），用来诊断 T2V 模型
- 因此**不复现 paper AID 数字**，但复用 paper 的 prompt template + taxonomy

**输出 schema**：
```json
{
  "metric": "artifact-diagnostic",
  "per_video": [{
    "video_path": "...",
    "artifact_frequency": {"color_exposure_anomaly": 0, "unnatural_camera_motion": 1, ...},
    "top_3_artifacts": ["...", "...", "..."],
    "overall_artifact_rate": 0.43
  }],
  "aggregate": {
    "per_artifact_rate": {"color_exposure_anomaly": 0.12, ...},
    "top_3_across_all_videos": [...]
  }
}
```

**完整 Artifact-Bench**（judge-eval 语义）推到 v0.3 experimental bench，见 [`PRODUCT_DESIGN.md`](PRODUCT_DESIGN.md) §6.3。

---

## 5. 双入口模型（核心约定）

### 5.1 同一份实现，两个入口

```
                ┌────────────────────────────────────────────┐
                │   src/videvalkit/metrics/<name>.py         │
                │   ────────────────────────────────         │
                │   单一真源（single source of truth）         │
                └────────────────────────────────────────────┘
                          ▲                           ▲
                          │ import                    │ import
            ┌─────────────┴─────────────┐ ┌───────────┴───────────────┐
            │  Bench 入口（粗粒度）       │ │  Metric 入口（细粒度）      │
            │                          │ │                           │
            │  videvalkit eval         │ │  videvalkit metric        │
            │    --bench vbench        │ │    --name motion-smoothness│
            │    --dimensions          │ │    --videos /my/videos    │
            │      motion_smoothness   │ │                           │
            │                          │ │  用户自己的视频；可选          │
            │  用 bench 的 prompt +     │ │  --prompts-from vbench    │
            │  bench 的 video layout    │ │  复用 bench prompt 集      │
            │                          │ │                           │
            │  → paper-comparable 数    │ │  → 独立分数，灵活           │
            └──────────────────────────┘ └───────────────────────────┘
                          │                           │
                          └─────────────┬─────────────┘
                                        ▼
                          ┌──────────────────────────────┐
                          │  位级一致硬契约 (≤ 1e-6 误差)   │
                          │  同视频两路必须跑出同数         │
                          └──────────────────────────────┘
```

### 5.2 实现路径

- 每个 metric 的核心算法实现于 `src/videvalkit/metrics/<name>.py`
- Bench adapter 的对应 dim **不重写**算法，反向 `from videvalkit.metrics.<name> import score_video`
- Standalone CLI `metric --name X` 直接调同一函数

### 5.3 位级一致硬契约

**约束**：对任意视频集 V，

```
eval --bench vbench --dimensions motion_smoothness --videos V
                              ==（位级一致，≤ 1e-6）==
metric --name motion-smoothness --videos V
```

不允许 metric 内部写"简化版本" —— 那是 bug 源。

**强制测试**：`tests/test_metric_lift_bit_exact.py` 对每个 lift-out metric 自动跑两条路径并 assert 数值一致。CI 红时不可合入。

### 5.4 命名 convention

| 形式 | 用法 | 例子 |
|---|---|---|
| **Short canonical** | 全局唯一短名，指向 v0.2 内置 canonical 实现 | `motion-smoothness` |
| **Source-qualified** | 显式指明出处，避免歧义；未来同名跨多 source 时必需 | `vbench/motion-smoothness` |

v0.2 短名 → canonical 映射在 `SUPPORTED_METRICS` 注册表里硬编码；用户 plugin 加同名 metric 时必须用 source-qualified 形式（避免覆盖 canonical）。

---

## 6. `SUPPORTED_METRICS` 注册表 schema

```python
SUPPORTED_METRICS = {
    # ---- Distribution (4) ----
    "fvd": dict(
        kind="distribution_reference",
        source="canonical/stylegan-v-port",
        cls="videvalkit.metrics.fvd:FVD",
        canonical_backbone="i3d-k400",
        supported_backbones=["i3d-k400", "videomae-v2-base", "vjepa-l16"],
        min_recommended_samples=2048,
        inputs=["gen_videos", "ref_videos"],
        output_kind="scalar_overall",
        version="1.0",
    ),
    "vfid": dict(kind="distribution_reference", ...),
    "kvd":  dict(kind="distribution_reference", ...),
    "clip-fvd": dict(kind="distribution_reference", experimental=True, ...),

    # ---- Text-video alignment (2, new) ----
    "clip-score":   dict(kind="per_prompt_reference_free",
                         cls="videvalkit.metrics.clip_score:CLIPScore",
                         inputs=["videos", "prompts"], ...),
    "viclip-score": dict(kind="per_prompt_reference_free", ...),

    # ---- Generic lift-outs (8) ----
    "aesthetic-quality": dict(
        kind="per_video_reference_free",
        source="vbench/aesthetic-quality",
        needs_judge=False,
        compute_kind="local_vision",
        tags=["vq.aesthetic", "style.aesthetic"],         # ← capability tags（用户 2026-05-20）
        cls="videvalkit.metrics.aesthetic_quality:AestheticQuality",
        algorithm="LAION-Aesthetic V2",
        inputs=["videos"],
        output_kind="scalar_per_video",
        also_used_by=["vbench", "vbench_pp"],
        paper_alignment_test="tests/test_metric_aesthetic_quality.py",
        license="Apache-2.0 (vbench)",
    ),
    "imaging-quality": dict(kind="per_video_reference_free", source="vbench/imaging-quality", ...),
    "motion-smoothness": dict(kind="per_video_reference_free", source="vbench/motion-smoothness", ...),
    "temporal-flickering": dict(kind="per_video_reference_free", source="vbench/temporal-flickering", ...),
    "subject-consistency": dict(kind="per_video_reference_free", source="vbench/subject-consistency", ...),
    "background-consistency": dict(kind="per_video_reference_free", source="vbench/background-consistency", ...),
    "dynamic-degree": dict(kind="per_video_reference_free", source="vbench/dynamic-degree", ...),
    "motion-magnitude": dict(kind="per_video_reference_free", source="worldscore/motion-magnitude", ...),

    # ---- Specialized lift-outs (5) ----
    "object-binding":       dict(kind="per_prompt_reference_free", source="t2vcompbench/object-binding", ...),
    "spatial-relationship": dict(kind="per_prompt_reference_free", source="t2vcompbench/spatial-relationship", ...),
    "numeracy":             dict(kind="per_prompt_reference_free", source="t2vcompbench/numeracy", ...),
    "motion-accuracy":      dict(kind="per_prompt_reference_free", source="worldscore/motion-accuracy", ...),
    "identity-preservation": dict(kind="per_video_with_ref_image", source="canonical/new", algorithm="ArcFace cosine", ...),
}
```

字段说明（首次出现的）：

| 字段 | 用途 |
|---|---|
| `kind` | 输入需求；CLI 缺参时 fail-fast |
| `source` | `canonical/...` 或 `<bench>/<dim>`；`metric show` 透明展示 |
| `algorithm` | 一句话算法描述 |
| `also_used_by` | 哪些 bench 复用此 metric（lift-out 才有） |
| `paper_alignment_test` | 哪个测试文件保证 paper 对齐 |
| `license` | 上游 license，避免合并冲突 |
| `tags` | `list[str]` —— capability tag（44 个 controlled vocab，见 [`CAPABILITY_TAGS_DESIGN.md`](CAPABILITY_TAGS_DESIGN.md)）。用户 2026-05-20 加入。每 metric 1–2 个，决定 `eval --capability X` 的反向索引 |

---

## 7. CLI 设计

### 7.1 列表 / 详情

```bash
videvalkit metric list
# Distribution (4):
#   fvd  vfid  kvd  clip-fvd*
#
# Generic T2V (canonical short names):
#   clip-score           (new)
#   viclip-score         (new)
#   aesthetic-quality    → vbench/aesthetic-quality
#   imaging-quality      → vbench/imaging-quality
#   motion-smoothness    → vbench/motion-smoothness
#   temporal-flickering  → vbench/temporal-flickering
#   subject-consistency  → vbench/subject-consistency
#   background-consistency → vbench/background-consistency
#   dynamic-degree       → vbench/dynamic-degree
#   motion-magnitude     → worldscore/motion-magnitude
#
# Specialized (canonical short names):
#   object-binding       → t2vcompbench/object-binding
#   spatial-relationship → t2vcompbench/spatial-relationship
#   numeracy             → t2vcompbench/numeracy
#   motion-accuracy      → worldscore/motion-accuracy
#   identity-preservation (new, ArcFace)
#
#  * = experimental, not paper-canonical

videvalkit metric list --kind per_video_reference_free   # 只看某档
videvalkit metric list --source vbench                   # 只看某出处
```

```bash
videvalkit metric show motion-smoothness
# name              : motion-smoothness
# kind              : per_video_reference_free
# canonical_source  : vbench/motion-smoothness (Vchitect/VBench, Apache-2.0)
# algorithm         : AMT-based frame interpolation reconstruction error
# inputs            : [video]
# output            : scalar per video, range [0,1] higher = smoother
# also_used_by      : [vbench, vbench_pp]
# paper_alignment   : VBench HF leaderboard, tol mean |Δ| ≤ 0.002
# implementation    : src/videvalkit/metrics/motion_smoothness.py
# bit_exact_test    : tests/test_metric_motion_smoothness_lift.py
```

### 7.2 跑一个 / 多个 metric

```bash
# Per-video reference-free（无 prompt 无 ref）
videvalkit metric --name motion-smoothness --videos /my/videos --output result.json
videvalkit metric --name aesthetic-quality --videos /my/videos

# Per-prompt reference-free（需 prompt）
videvalkit metric --name object-binding --videos X/ --prompts prompts.jsonl
videvalkit metric --name clip-score --videos X/ --prompts-from t2vcompbench
  # ↑ 复用某 bench 的 prompt 集，省去自己组装

# Distribution（需 ref）
videvalkit metric --name fvd --gen-videos X/ --refs ucf101-fvd
videvalkit metric --name fvd --gen-videos X/ --ref-videos /my/ref/

# With ref image
videvalkit metric --name identity-preservation --videos X/ --ref-image alice.png

# 多 metric 一次跑（同 kind 自动合并 IO）
videvalkit metric \
  --name motion-smoothness --name aesthetic-quality --name imaging-quality \
  --videos X/ --output multi.json

# 跨 kind 也支持（不同 kind 各自执行）
videvalkit metric \
  --name fvd --name clip-score --name motion-smoothness \
  --gen-videos X/ --refs ucf101-fvd --prompts prompts.jsonl
```

### 7.3 输入需求 fail-fast

```bash
$ videvalkit metric --name object-binding --videos X/
ERROR: 'object-binding' requires --prompts (per_prompt_reference_free).
       Either:
         --prompts prompts.jsonl
         --prompts-from t2vcompbench   (use upstream prompt set)

$ videvalkit metric --name fvd --videos X/
ERROR: 'fvd' requires --gen-videos AND (--refs OR --ref-videos).
       --videos is for non-distribution metrics.
       Did you mean: --gen-videos X/ --refs ucf101-fvd ?
```

### 7.4 Refs 子命令组

```bash
videvalkit refs list                          # 内置 + 用户注册
videvalkit refs show ucf101-fvd               # 展开 manifest + hash
videvalkit refs register --name X --path P    # 注册用户 ref
videvalkit refs verify ucf101-fvd             # 校验 hash
videvalkit fetch-refs --name ucf101-fvd       # 从 HF 拉取
```

---

## 8. Reference Video Set 管理

### 8.1 内置 reference set（v0.2 交付）

| Name | 大小 | 数量 | 来源 |
|---|---:|---:|---|
| `ucf101-fvd` | ~5 GB | 13320 clips | UCF101 + paper 标准前处理 |
| `ucf101-fvd-subset-500` | ~250 MB | 500 clips | UCF101 子集，用于 quick profile |
| `msr-vtt-val` | ~2 GB | 2990 clips | MSR-VTT validation |

v0.3 候选：`webvid-2m-val-10k` (~10 GB) / `kinetics600-val-5k` (~8 GB)。

存放：`~/.cache/videvalkit/refs/<name>/` + `manifest.json`（视频数、平均时长、分辨率、source 引用、license、sha256）。

HF dataset：`videogenevalkit/reference-videos`（v0.2 新建）。

### 8.2 用户自定义 ref

```bash
# 直接传路径
videvalkit metric --name fvd --gen-videos X/ --ref-videos /my/ref/

# 注册后复用
videvalkit refs register --name my-ref --path /data/ref/
videvalkit metric --name fvd --gen-videos X/ --refs my-ref
```

注册结果写 `~/.config/videvalkit/refs.yaml`。

---

## 9. Sample size 警告

分布层 metric 对 N 敏感。`videvalkit metric` 跑前检查：

| n_gen | 级别 | 行为 |
|---|---|---|
| ≥ 2048 | OK | 无警告 |
| 500–2047 | INFO | 给统计推算 |
| 100–499 | WARN | 强警告 + 建议改用 KVD |
| < 100 | **ERROR** | 默认拒绝，需 `--allow-tiny-sample` |

示例：

```
$ videvalkit metric --name fvd --gen-videos /tmp/50_videos --refs ucf101-fvd
ERROR: n_gen=50 below minimum 100 for FVD.
       Use --allow-tiny-sample to override (FVD will be unreliable).
       Consider --name kvd (more stable at small N).
```

---

## 10. 数值复现性

| 控制点 | 实现 |
|---|---|
| Frame sampling | 固定 `n_frames=16, stride=1, start_offset=0` (FVD canonical) |
| Clip resize | 双线性 224×224，固定 `antialias=True` |
| Backbone weights | `videogenevalkit/checkpoints` HF 文件 sha256 校验 |
| Seed | 默认 42；CLI `--seed` 可改；写入 result.json |
| Numerical reduce | 协方差矩阵累加用 float64；sqrtm 用 `scipy.linalg.sqrtm` (双精度) |
| Batch 顺序 | 视频按文件名排序后处理 |
| Device 影响 | CPU/CUDA 结果差 ≤ 1e-3 |

复现性测试见 `tests/test_fvd_reproducibility.py` 等。

---

## 11. 与 Profile / Training Monitor 衔接

QUICK_EVAL_DESIGN §3 的 profile schema 扩展：

```yaml
profiles:
  quick:
    metrics:                                # 新增字段
      - fvd
      - kvd
      - motion-smoothness
      - aesthetic-quality
      - clip-score
    metric_refs:
      fvd: ucf101-fvd-subset-500           # 小 ref set
      kvd: ucf101-fvd-subset-500
    metric_options:
      fvd:
        sample_size_acknowledge: true      # 显式承认是 trend indicator
```

训练监控 Python API：

```python
cfg = MonitorConfig(
    benches=["vbench"],
    metrics=["fvd", "kvd", "motion-smoothness", "aesthetic-quality"],
    metric_refs={"fvd": "ucf101-fvd-subset-500", "kvd": "ucf101-fvd-subset-500"},
    profile="quick",
)
for step in range(0, 100_000, 1000):
    if step % 5000 == 0:
        result = monitor.eval(videos_dir, model_name=f"step_{step}", cfg=cfg)
        for m, r in result.metrics.items():
            tb.add_scalar(f"eval/{m}", r.score, step)
```

> **重要**：Quick profile 跑出来的 FVD 是 trend indicator，不是 paper 数。文档明确写出"expected std ±20-50 units"。

---

## 12. 文件改动清单

### 12.1 新增（v0.2）

| 路径 | 用途 |
|---|---|
| `src/videvalkit/core/distribution_metric.py` | `BaseDistributionMetric` + `DistributionMetricResult` |
| **Distribution-level (4)** | |
| `src/videvalkit/metrics/fvd.py` | stylegan-v port wrap |
| `src/videvalkit/metrics/vfid.py` | 自实现 |
| `src/videvalkit/metrics/kvd.py` | 自实现（复用 FVD backbone） |
| `src/videvalkit/metrics/clip_fvd.py` | experimental |
| **Alignment (2)** | |
| `src/videvalkit/metrics/clip_score.py` | per-frame CLIP-ViT-L/14 |
| `src/videvalkit/metrics/viclip_score.py` | per-clip ViCLIP |
| **Lift-out generic (8)** | |
| `src/videvalkit/metrics/aesthetic_quality.py` | lift from vbench |
| `src/videvalkit/metrics/imaging_quality.py` | lift from vbench |
| `src/videvalkit/metrics/motion_smoothness.py` | lift from vbench |
| `src/videvalkit/metrics/temporal_flickering.py` | lift from vbench |
| `src/videvalkit/metrics/subject_consistency.py` | lift from vbench |
| `src/videvalkit/metrics/background_consistency.py` | lift from vbench |
| `src/videvalkit/metrics/dynamic_degree.py` | lift from vbench |
| `src/videvalkit/metrics/motion_magnitude.py` | lift from worldscore |
| **Lift-out specialized (5)** | |
| `src/videvalkit/metrics/object_binding.py` | lift from t2vcompbench |
| `src/videvalkit/metrics/spatial_relationship.py` | lift from t2vcompbench |
| `src/videvalkit/metrics/numeracy.py` | lift from t2vcompbench |
| `src/videvalkit/metrics/motion_accuracy.py` | lift from worldscore |
| `src/videvalkit/metrics/identity_preservation.py` | new (ArcFace) |
| **Backbones** | |
| `src/videvalkit/metrics/backbones/{i3d_k400, videomae_v2, vjepa, clip_vit, viclip, inception_v3, dino, arcface, sea_raft, amt}.py` | backbone loaders |
| `src/videvalkit/metrics/utils/{clip_sampling, frechet, kernel_mmd}.py` | 共享工具 |
| **Refs** | |
| `src/videvalkit/refs/registry.py` | refs.yaml 加载 + manifest |
| **CLI** | |
| `src/videvalkit/cli_metric.py` | metric 子命令组（list/show/run） |
| `src/videvalkit/cli_refs.py` | refs 子命令组 |
| **Tests** | |
| `tests/test_metric_lift_bit_exact.py` | 13 个 lift metric 的双入口位级一致 |
| `tests/test_fvd_reproducibility.py` | seed/device 复现性 |
| `tests/test_fvd_paper_alignment.py` | UCF101 vs paper 对齐 |
| `tests/test_metric_kind_fail_fast.py` | CLI 输入需求校验 |
| `tests/test_refs_management.py` | refs 注册 / hash |

### 12.2 修改

| 路径 | 修改点 |
|---|---|
| `src/videvalkit/configs/metrics.py` | 19 个 metric entry，含 `source`/`kind`/`also_used_by`/`paper_alignment_test`/`license` 字段 |
| `src/videvalkit/cli.py` | 接入 `metric`/`refs`/`fetch-refs` 子命令组 |
| `src/videvalkit/benchmarks/vbench/benchmark.py` | 8 个 dim 改为 `from videvalkit.metrics.X import score_video` （删除 inline 实现） |
| `src/videvalkit/benchmarks/worldscore/benchmark.py` | 2 个 dim 改为 import |
| `src/videvalkit/benchmarks/t2vcompbench/benchmark.py` | 3 个 dim 改为 import |
| `src/videvalkit/training/__init__.py` | `MonitorConfig` 加 `metrics` + `metric_refs` 字段 |
| `docs/INTEGRATION_FRAMEWORK_DESIGN.md` | §5 `SUPPORTED_METRICS` schema 加 `source`/`kind` 字段说明 |
| `docs/QUICK_EVAL_DESIGN.md` | §3.1 profile schema 加 `metrics` + `metric_refs` |
| `docs/PRODUCT_DESIGN.md` | §4.1 metric 数 7 → 19；§6.2 v0.2 总工时 16.5 → 18.3 day |
| `docs/USER_MANUAL_*.md` | 新章节 "Metrics — quick reference" + "Distribution metrics" + "Lift-out vs bench" |
| `docs/TEST_MANUAL.md` | 加 "FVD on UCF101 paper-alignment" + "Lift bit-exact" 表 |
| `README.md` | TL;DR 加一行 "19 metrics out of the box, dual entry (bench / standalone)" |

### 12.3 新增 HF dataset

`videogenevalkit/reference-videos`（新）：UCF101 + MSR-VTT。

`videogenevalkit/checkpoints` 加 `metrics-backbones/`：I3D / VideoMAE / VJEPA / CLIP / ViCLIP / Inception / DINO / ArcFace / SEA-RAFT / AMT。

---

## 13. 与 Paper / Bench 对齐

### 13.1 Distribution-level paper 对齐

| Metric | 对照源 | tolerance |
|---|---|---|
| FVD on UCF101 | StyleGAN-V paper Table 1 | mean \|Δ\| ≤ 5% |
| FVD on Kinetics600 (CogVideoX-5B) | CogVideoX paper | mean \|Δ\| ≤ 5% |
| KVD on UCF101 | StyleGAN-V supp | mean \|Δ\| ≤ 5% |
| VFID on MSR-VTT | TATS paper | mean \|Δ\| ≤ 10% (无 canonical) |

### 13.2 Lift-out bit-exact 对齐

每个 lift-out metric 在 3 个 sample video 上：

```python
score_bench = run_via_bench_path(videos)         # eval --bench X --dim Y
score_metric = run_via_metric_path(videos)       # metric --name Y
assert abs(score_bench - score_metric) < 1e-6
```

不达标的 metric **不允许进入 v0.2** —— 要么修代码让它对齐，要么不 lift。

---

## 14. 里程碑

| 阶段 | 内容 | 工作量 |
|---|---|---|
| **F1 — DistributionMetric 抽象 + I3D backbone** | base class + I3D loader + clip sampling + Fréchet utils | 0.5 day |
| **F2 — FVD + 复现性测试 + paper 对齐** | stylegan-v port wrap + seed/device 一致性 | 1 day |
| **F3 — VFID + KVD** | 复用 backbone；统计层各 ~80 行 | 0.5 day |
| **F4 — CLIP-FVD (experimental)** | CLIP backbone + 文档说明 | 0.5 day |
| **F5 — CLIP-Score + ViCLIP-Score** | per-frame / per-clip alignment | 0.5 day |
| **F6 — Lift-out generic 8 个** | 从 vbench/worldscore adapter 抽离 + 反向 import + bit-exact test | 2 day |
| **F7 — Lift-out specialized 4 个** | 从 t2vcompbench/worldscore 抽离 + bit-exact test | 1 day |
| **F8 — Identity Preservation (new)** | ArcFace loader + cross-frame + ref image 比较 | 0.5 day |
| **F9 — Refs 管理 + CLI** | refs.yaml + fetch-refs + 上传 HF dataset | 0.5 day |
| **F10 — metric show / list / 命名 alias 解析** | CLI 透明性 + source 字段填充 | 0.3 day |
| **F11 — 训练监控 metrics 接入** | `MonitorConfig.metrics` + profile schema 扩展 | 0.3 day |
| **F12 — 双入口 bit-exact 集成测试** | 13 个 lift metric 自动跑两路径对比 | 0.5 day |

**总计 ≈ 8.3 day**，全部进 INTEGRATION doc 的 M3。

### 14.1 v0.2 工时再核对

| 支柱 | 调整后 |
|---|---:|
| A (env/install 巩固) | 0.5 |
| B (judge selection) | 2 |
| C (integration framework) - **scaffolding 子项推 v0.3** | 4 |
| C - M3 metrics (本文 F1-F12) | 8.3 |
| D (quick eval & training monitor) | 7.5 |
| **TOTAL v0.2** | **22.3 day** |

> 注：原 16.5 day → 18.3 day → **22.3 day**（含本次完整 19 个 metric 的 lift-out 8.3 day，扣回 scaffolding 推 v0.3 省 1.5 day，故净增 4.5 day）。
>
> **时间线**：单人节奏需 5–6 周；2 人并行可 3 周（B+C metrics 一人，D + C 其他子项一人）。

---

## 15. Open Questions

> 用户已确认 9 条决策（§17）；下列为执行细节问题：

1. **lift-out 失败时**：某 metric 抽离后位级一致测试不过怎么办？倾向 **不进 v0.2，留在 bench 内**；不强行 lift 半成品。
2. **lift-out 改了 bench 内部行为是否需要回归测试**：lift 时 `vbench` 整体 mean \|Δ\| 必须不变 —— 倾向 **加 bench-level 回归测试**，每个 lift PR 都跑一遍 vbench 全集 smoke。
3. **`--prompts-from <bench>` 自动加载的 prompt 集是否要 cache**？倾向 **cache 到 `~/.cache/videvalkit/prompt-sets/<bench>/`**，避免每次 lazy import 全 bench。
4. **多 metric 输出 JSON 的 schema**：扁平 `{"fvd": {...}, "motion-smoothness": {...}}` vs 嵌套 by kind？倾向 **扁平**，最简。

---

## 16. 风险

| 风险 | 影响 | 对策 |
|---|---|---|
| Lift-out 改坏了 bench 数（mean \|Δ\| 升高） | paper 复现度回退 | F6/F7 每个 PR 跑 bench-level 回归；不通过不合 |
| 13 个 lift metric 工作量爆炸 | v0.2 延期 | 分批：先 vbench 8 个（高频），t2vcompbench 3 个；worldscore lift 因 DROID-SLAM 风险大，仅 lift `motion-magnitude`/`motion-accuracy` 两个独立的 |
| 双入口 schema 不一致导致用户混淆 | 用户体验 | doc 一张大表对比；CLI `metric show` 强制列出 also_used_by |
| 用户用 `clip-score` 短名得到不同实现 | 数不可比 | 短名锁 canonical；plugin 必须用 source-qualified 形式注册 |
| HF 上 reference-videos repo 大 (~7 GB) 拉取慢 | install 卡 | 默认不拉，按 metric 触发 fetch；提供 mirror 提示 |
| ArcFace identity-preservation 的 license（InsightFace） | 法律 | 用 buffalo_l 公开权重；doc 写明仅研究用 |
| stylegan-v port 不再维护 | 长期风险 | 把 FVD core 算法 vendor 进 repo（~300 LoC），不长期依赖外部 |

---

## 17. 决策快照（用户 2026-05-20 已确认）

- ✅ 两档分类：通用 T2V quality + 专用维度；ref-based 窄适用 (PSNR/SSIM/LPIPS/FID-image) **移出 v0.2**
- ✅ v0.2 交付 **14 + 6 = 20 个 metric**（含 artifact-diagnostic from Artifact-Bench paper-port）
- ✅ **Judge-free path**：17/20 个 metric + 4/9 个 bench 不需 VLM/LLM，`--no-judge` filter 一键过滤
- ✅ 注册表加 `needs_judge: bool` + `compute_kind: str` 字段，`list` / `metric show` / `doctor` 展示
- ✅ 双入口：`eval --bench` 与 `metric --name` 共用单一实现
- ✅ 位级一致硬契约（≤ 1e-6 误差，CI 强制）
- ✅ 命名 short canonical + source-qualified；短名锁 v0.2 canonical
- ✅ `kind` 字段声明输入需求，CLI 缺参 fail-fast
- ✅ `metric show` 必须列 canonical_source / paper_alignment / 实现路径
- ✅ Lift 反例（Human Anatomy / consistent_attribute）明确留 bench 内
- ✅ `--prompts-from <bench>` 复用 bench prompt 集
- ✅ Plugin scaffolding (`videvalkit new`) 推到 v0.3 候选
- ✅ CLIP-FVD experimental 进 v0.2；赶进度可推 v0.3 省 0.5 day
- ✅ Reference set v0.2 内置 UCF101 + MSR-VTT (~7 GB)，HF dataset `videogenevalkit/reference-videos`
- ✅ N < 100 默认 ERROR + `--allow-tiny-sample` 显式
- ✅ FVD canonical = I3D-K400 via stylegan-v port，core 算法 vendor 进 repo

---

## 18. 与其他设计文档的关系

| 文档 | 关系 |
|---|---|
| [`INTEGRATION_FRAMEWORK_DESIGN.md`](INTEGRATION_FRAMEWORK_DESIGN.md) | §5 SUPPORTED_METRICS 加 `source`/`kind`/`also_used_by` 字段说明；§6 scaffolding 推 v0.3；本文 §6 是 §5 的具体落地 |
| [`QUICK_EVAL_DESIGN.md`](QUICK_EVAL_DESIGN.md) | Profile schema 加 `metrics` 字段；训练监控 API 加 `metrics` 参数（本文 §11） |
| [`JUDGE_SELECTION_DESIGN.md`](JUDGE_SELECTION_DESIGN.md) | 无直接交互；VLM-judge-based metric 走 Judge 抽象，本文不重复 |
| [`PRODUCT_DESIGN.md`](PRODUCT_DESIGN.md) | §4.1 metric 数 7 → 19；§6.2 v0.2 工时 16.5 → 22.3；§7.3 跨支柱原则 "80/20 双轨" 加注 "metric 也走双入口" |
| [`NPU_ADAPTATION_DESIGN.md`](NPU_ADAPTATION_DESIGN.md) | Deferred；NPU 启动时分布层 metric backbone 需独立验证 |
| `DEV_MANUAL.md` | §14 标 done；新增 §15 "Metrics & Lift-out" 章节 |
| `TEST_MANUAL.md` | 新增 FVD paper-alignment + lift bit-exact 两张表 |

---

—— end of design v0.2 ——
