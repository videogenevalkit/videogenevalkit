# Capability Tags — Cross-Bench Ability Profile Design

| 字段 | 内容 |
|---|---|
| 版本 | v0.1 (draft) |
| 状态 | Design — 用户 2026-05-20 确认进 v0.2 |
| 性质 | 支柱 C / 支柱 D 之上的**正交视图层**：给 metric / bench dim 打能力 tag，按能力跨 bench 调用 |
| 影响范围 | `configs/{metrics,benchmarks}.py` · `core/capability.py` (新) · `cli` 新增 `capabilities` 子命令 + `--capability` filter |
| 关联文档 | [`VIDEO_METRICS_DESIGN.md`](VIDEO_METRICS_DESIGN.md) · [`INTEGRATION_FRAMEWORK_DESIGN.md`](INTEGRATION_FRAMEWORK_DESIGN.md) · [`QUICK_EVAL_DESIGN.md`](QUICK_EVAL_DESIGN.md) · [`PRODUCT_DESIGN.md`](PRODUCT_DESIGN.md) |
| 目标读者 | T2V 模型研发者（按能力做诊断评测）· benchmark 集成者（给新 metric 打 tag）|

---

## 1. 背景

### 1.1 用户需求

> "每个 bench 不同的指标都要有一个画像或者叫 tag，这样我们后面要评测某一项特定的能力项的时候，可以一键调用这些不同 bench 同样评测画像的指标"

—— 翻译：当前 metric/bench dim 是**按实现来源**组织的（vbench/motion-smoothness, worldscore/motion-magnitude）。用户希望按**能力维度**组织，一键跑跨 bench 同能力的所有 metric。

### 1.2 当前轴 vs 新增轴

```
当前两轴                          新增第三轴（capability）
──────────                       ─────────────────────
bench  axis  →  vbench / ws / ... motion           →  motion-smoothness · dynamic-degree
                                                       · motion-magnitude · motion-accuracy
metric axis  →  fvd / clip-score   object_fidelity  →  object-binding · numeracy · obj-presence
                / motion-smooth                          (跨 vbench / t2vcompbench / standalone)
                / ...
```

**Tag 是 metric 的属性**，多对多——一个 metric 可挂多 tag（如 `motion-smoothness` 既属 `motion.smoothness` 也属 `temp.flickering`），一个 tag 下挂多 metric。

---

## 2. 目标 / 非目标

### 2.1 目标

1. **Controlled vocab**：固定 ~44 个 tag（10 顶层 + 34 子），避免命名漂移
2. **每 metric / bench dim 强制声明 tags**：注册表 schema 加 `tags: [...]` 字段
3. **CLI 按 capability 一键评测**：`eval --capability motion --videos X/` 跨 source 跑所有 motion 类 metric
4. **去重 + 归一**：lift-out canonical 同源去重；跨 source 不同 range 自动 min-max normalize
5. **透明展示**：`capabilities list / show` + `metric show <name>` 显示 tags

### 2.2 非目标

- ❌ **不允许 free-form tag**（避免 motion / Motion / 运动 三种写法）
- ❌ **不允许用户自定义 capability bundle**（v0.4 候选）
- ❌ **不替代 `--bench` / `--name` 入口**（capability 是第三入口，并存）
- ❌ **不在 v0.2 让 plugin 加新 tag**（用 controlled vocab；v0.4 再开）

---

## 3. Tag Taxonomy（两层 · v1 固化）

```
顶层 10 个   ←─ 用户常用粒度
   │
   └── 子 tag 34 个   ←─ metric 精确归属
```

### 3.1 完整 tag 表

| 顶层 | 子 tag | 测什么 |
|---|---|---|
| **motion** | motion.smoothness | 帧间平滑度 |
| | motion.magnitude | 运动幅度 |
| | motion.accuracy | 动作与 prompt 对齐 |
| | motion.naturalness | 运动是否自然 |
| **visual_quality** | vq.aesthetic | 美学分 |
| | vq.imaging | 成像质量 |
| | vq.artifact_free | 无 artifact |
| | vq.sharpness | 清晰度 |
| **text_alignment** | align.text2video | 文-视频对齐 |
| | align.prompt_following | 整体 prompt 遵循度 |
| | align.action_verb | 动词对齐 |
| **object_fidelity** | obj.presence | 物体存在 |
| | obj.count | 数量准确 |
| | obj.attribute | 属性（颜色/材质） |
| | obj.binding | object-attribute 绑定 |
| **subject_consistency** | subj.identity | 主体身份 |
| | subj.appearance | 外观连续 |
| | subj.character | 角色一致 |
| **physical_plausibility** | phys.gravity | 重力合理 |
| | phys.causality | 因果合理 |
| | phys.anatomy | 解剖合理 |
| | phys.kinematics | 运动学合理 |
| **temporal_coherence** | temp.flickering | 闪烁 |
| | temp.continuity | 时序连续 |
| | temp.scene_consistency | 场景一致 |
| **realism** | real.distribution | 分布层真实（FVD 系）|
| | real.detection | 是否 AI 生成 |
| | real.artifact_rate | artifact 频率 |
| **compositional** | comp.multi_object | 多物体场景 |
| | comp.spatial | 空间关系 |
| | comp.numeracy | 数量构成 |
| **style** | style.aesthetic | 美学风格 |
| | style.cg_anime | CG / 动画风格 |
| | style.consistency | 风格一致 |

**合计：10 顶层 + 34 子 = 44 tag**。固化为 v1 controlled vocab。

### 3.2 演进策略

- 顶部声明 `tag_schema_version: 1`
- v2 schema 升级时 loader 按 version 路由
- v0.2 → v0.3 期间 tag 列表只加不删，且需 design doc 增补
- 删 tag 或重命名 → 必须 schema_version bump

---

## 4. v0.2 metric → tag 映射（完整 20 个）

| Metric | Tags |
|---|---|
| `fvd` | realism.distribution |
| `vfid` | realism.distribution |
| `kvd` | realism.distribution |
| `clip-fvd` | realism.distribution |
| `clip-score` | align.text2video |
| `viclip-score` | align.text2video · align.prompt_following |
| `aesthetic-quality` | vq.aesthetic · style.aesthetic |
| `imaging-quality` | vq.imaging · vq.sharpness |
| `motion-smoothness` | motion.smoothness · temp.flickering |
| `temporal-flickering` | temp.flickering · vq.artifact_free |
| `subject-consistency` | subj.identity · subj.appearance |
| `background-consistency` | subj.appearance · temp.continuity |
| `dynamic-degree` | motion.magnitude |
| `motion-magnitude` | motion.magnitude |
| `object-binding` | obj.binding · obj.presence |
| `spatial-relationship` | comp.spatial |
| `numeracy` | comp.numeracy · obj.count |
| `motion-accuracy` | motion.accuracy · align.action_verb |
| `identity-preservation` | subj.identity · subj.character |
| `artifact-diagnostic` | real.artifact_rate · vq.artifact_free（实际跨多 tag，metric show 详注）|

## 5. v0.2 anchored bench dim → tag（示例映射，完整表 ~150 行待落地时填）

**VBench (16 dims) — 完整映射**：

| Dim | Tags |
|---|---|
| subject_consistency | subj.identity |
| background_consistency | subj.appearance · temp.continuity |
| temporal_flickering | temp.flickering |
| motion_smoothness | motion.smoothness |
| dynamic_degree | motion.magnitude |
| aesthetic_quality | vq.aesthetic |
| imaging_quality | vq.imaging |
| object_class | obj.presence |
| multiple_objects | comp.multi_object · obj.count |
| human_action | align.action_verb |
| color | obj.attribute |
| spatial_relationship | comp.spatial |
| scene | comp.multi_object |
| appearance_style | style.consistency |
| temporal_style | style.consistency · temp.scene_consistency |
| overall_consistency | align.prompt_following |

其他 8 bench 的 dim 映射在编码阶段补全（design doc 不强求穷举）。

---

## 6. 注册表 schema 扩展

```python
# src/videvalkit/configs/metrics.py
SUPPORTED_METRICS["motion-smoothness"] = dict(
    kind="per_video_reference_free",
    source="vbench/motion-smoothness",
    needs_judge=False,
    compute_kind="local_vision",
    tags=["motion.smoothness", "temp.flickering"],       # ← 新增
    cls="videvalkit.metrics.motion_smoothness:MotionSmoothness",
    ...
)

# src/videvalkit/configs/benchmarks.py
# bench 不挂 tag；bench 内每个 dim 挂 tag，存在 dim spec 里
SUPPORTED_BENCHMARKS["vbench"] = dict(
    cls=VBenchBenchmark,
    ...
    dim_tags={                                            # ← 新增
        "motion_smoothness": ["motion.smoothness"],
        "dynamic_degree":    ["motion.magnitude"],
        ...
    },
)
```

**`tag_schema_version: 1`** 顶部声明在 `configs/__init__.py`，未来升级有 hook。

---

## 7. CLI 设计

### 7.1 `capabilities` 子命令组

```bash
# 列所有 tag 及其挂载情况
videvalkit capabilities list
# motion                    7 sources  (4 metric · 3 bench dim)
# visual_quality           11          (3 metric · 8 bench dim)
# realism                   5
# physical_plausibility    14
# ...

# 详情：某 tag 下挂什么
videvalkit capabilities show motion
# motion.smoothness:
#   metric/motion-smoothness  ← canonical
#   vbench/motion_smoothness  ← same source (deduped)
# motion.magnitude:
#   metric/motion-magnitude   (worldscore canonical, range [0,100])
#   metric/dynamic-degree     (vbench canonical, range [0,1])
#   vbench/dynamic_degree     ← same source as metric/dynamic-degree (deduped)
#   worldscore/motion_magnitude  ← same source as metric/motion-magnitude (deduped)
# motion.accuracy:
#   metric/motion-accuracy
# motion.naturalness:
#   (none yet —待 v0.3 加 unnatural-motion-detector)

# 反向：列某 metric 的 tag
videvalkit metric show motion-smoothness
# ...
# tags: [motion.smoothness, temp.flickering]
# capability_groups: motion · temporal_coherence
```

### 7.2 `eval --capability`

```bash
# 一键评测某能力
videvalkit eval --capability motion --videos X/ --workspace ws/
# Resolving motion → 4 metrics:
#   motion-smoothness  (motion.smoothness)
#   motion-magnitude   (motion.magnitude)
#   dynamic-degree     (motion.magnitude — different impl, both run)
#   motion-accuracy    (motion.accuracy)
# Running 4 metrics ...

# 多 tag
videvalkit eval \
  --capability motion \
  --capability visual_quality \
  --videos X/

# 子 tag 也能用
videvalkit eval --capability motion.smoothness --videos X/
# → 只跑 motion-smoothness 一个

# 与 --no-judge / --profile 组合
videvalkit eval \
  --capability physical_plausibility \
  --no-judge --profile quick \
  --videos X/
# → 跑 physical_plausibility 标签下 judge-free 的 metric · quick profile
```

### 7.3 与现有入口的关系

```
三个并存的入口：
  videvalkit eval --bench X         ← 跑完整 bench（paper-comparable 数）
  videvalkit metric --name Y        ← 跑单个 metric（标量）
  videvalkit eval --capability Z    ← 跨 bench 跑某能力（aggregate 数）
```

互不替代：bench 入口给 paper 数；metric 入口给单点；capability 入口给能力画像。

---

## 8. 跨 source 聚合策略

### 8.1 默认聚合层级

```
raw scores (各 source 不同 range)
        ↓ min-max normalize per metric to [0,1]
normalized scores
        ↓ dedup by canonical_source（lift 与 metric 同源只算一次）
deduplicated scores
        ↓ mean within sub-tag
sub-tag scores
        ↓ mean within top-level tag (weighted by # sub-tags 现实落到几个)
top-level capability score
```

### 8.2 例子

`eval --capability motion --videos X/` 输出：

```json
{
  "capability": "motion",
  "sub_capabilities": {
    "motion.smoothness": {
      "score_normalized": 0.92,
      "n_contributors": 1,
      "contributors": [
        {"source": "canonical/vbench-port", "name": "motion-smoothness",
         "raw": 0.92, "range": [0, 1], "normalized": 0.92}
      ]
    },
    "motion.magnitude": {
      "score_normalized": 0.64,
      "n_contributors": 2,
      "contributors": [
        {"source": "canonical/worldscore-port", "name": "motion-magnitude",
         "raw": 58.4, "range": [0, 100], "normalized": 0.584},
        {"source": "canonical/vbench-port", "name": "dynamic-degree",
         "raw": 0.71, "range": [0, 1], "normalized": 0.71}
      ]
    },
    "motion.accuracy": {
      "score_normalized": 0.81,
      "n_contributors": 1,
      "contributors": [
        {"source": "canonical/worldscore-port", "name": "motion-accuracy",
         "raw": 0.81, "range": [0, 1], "normalized": 0.81}
      ]
    }
  },
  "overall_motion": 0.79
}
```

### 8.3 可选聚合器

CLI flag `--capability-aggregator` 切换：

| Aggregator | 行为 |
|---|---|
| `mean` (默认) | 子 tag 内 mean → 顶 tag mean |
| `max` | 子 tag 内 max（最优）→ 顶 tag mean |
| `min` | 子 tag 内 min（最差）→ 顶 tag mean |
| `weighted` | 子 tag 内 mean，顶 tag 内按 contributor 数加权 mean |
| `bt` | Bradley-Terry 跨模型 ranking 模式（多模型时用）|

---

## 9. 去重规则

```python
# 同 canonical_source 的多个入口只算一次
# 例：vbench/motion_smoothness (bench dim) 与 metric/motion-smoothness (standalone)
#     都指向 src/videvalkit/metrics/motion_smoothness.py
#     dedup_key = "canonical/vbench-port::motion-smoothness"
#     → 在 capability 聚合时只算一次

# 不同 canonical_source 即使语义近似也都算
# 例：vbench/dynamic_degree (RAFT) 与 worldscore/motion_magnitude (SEA-RAFT)
#     不同算法、不同 source → 都算，作为 motion.magnitude 子 tag 的两个 contributor
```

去重日志写到 `result.json` 的 `deduplicated_contributors: [...]`，可追溯。

---

## 10. 文件改动清单

### 10.1 新增

| 路径 | 用途 |
|---|---|
| `src/videvalkit/core/capability.py` | `CapabilityTag` enum + `resolve_capability(tag) → list[MetricRef]` + 聚合器 |
| `src/videvalkit/configs/capability_taxonomy.py` | 44 个 tag 的 controlled vocab + 描述 |
| `src/videvalkit/cli_capability.py` | `capabilities list/show` 子命令 + `eval --capability` 解析 |
| `tests/test_capability_resolve.py` | tag → metrics 反向索引 |
| `tests/test_capability_aggregation.py` | min-max normalize + dedup + mean |
| `tests/test_capability_taxonomy_consistency.py` | 每 metric/bench dim 至少 1 tag + tag 全在 controlled vocab |
| `docs/CAPABILITY_TAGS_DESIGN.md` | 本文 |
| `examples/capability_eval.sh` | 完整 demo |

### 10.2 修改

| 路径 | 改点 |
|---|---|
| `src/videvalkit/configs/metrics.py` | 20 个 entry 加 `tags: [...]` 字段 |
| `src/videvalkit/configs/benchmarks.py` | 9 个 entry 加 `dim_tags: {dim_name: [tags]}` 字段 |
| `src/videvalkit/configs/__init__.py` | 加 `tag_schema_version: 1` |
| `src/videvalkit/cli.py` | `eval` 加 `--capability` (多次) + `--capability-aggregator` flag |
| `src/videvalkit/runner.py` | `run()` 接受 `capability` 参数 |
| `src/videvalkit/diagnostics.py` | doctor 报告每 tag 的 coverage（有几个 contributor）|
| `docs/VIDEO_METRICS_DESIGN.md` | §6 schema 加 `tags` 字段 |
| `docs/INTEGRATION_FRAMEWORK_DESIGN.md` | §5.2 加 `tags` 字段说明 |
| `docs/QUICK_EVAL_DESIGN.md` | profile 加 `capabilities` 字段（可选） |
| `docs/PRODUCT_DESIGN.md` | §4.1 加 capability 行；§5 加 sub-pillar；§6.2 工时 |
| `docs/USER_MANUAL_*.md` | 加 "Capability tags" 章节 |
| `README.md` | TL;DR 加一行 "evaluate by ability, not just by bench" |

### 10.3 删除

无。

---

## 11. 兼容性

| 现有用法 | v0.2 行为 |
|---|---|
| `videvalkit eval --bench X ...` | 不变 |
| `videvalkit metric --name Y` | 不变 |
| 现有用户脚本 | 一行不动 |
| 注册表 import | 加字段不破坏现有 entry；旧 entry tags 字段缺失 → loader 给 warning + tag-as-uncategorized 兜底 |

---

## 12. 里程碑（v0.2）

| 阶段 | 内容 | 工作量 |
|---|---|---|
| **T1 — Taxonomy 固化** | 44 个 tag controlled vocab + 描述 + 文档 | 0.3 d |
| **T2 — 注册表加 tags 字段** | 20 metric + 9 bench dim 完整打 tag | 0.5 d |
| **T3 — Resolve 层** | `resolve_capability` + dedup + min-max normalize | 0.5 d |
| **T4 — CLI** | `capabilities list/show` + `eval --capability` + aggregator | 1 d |
| **T5 — 聚合器与测试** | min-max + dedup + 跨 source · 单测 + 集成测 | 0.5 d |
| **T6 — 文档** | README / USER_MANUAL / metric show 输出 | 0.3 d |

**合计 ≈ 3.1 day**，进入 v0.2，与 M3 metric 工作并行（M3 metric 落 tag 是 T2 的输入）。

---

## 13. Open Questions

| # | 问题 | 默认 |
|---|---|---|
| 1 | Tag 数量（44 个）够不够？ | v0.2 固化 44，v0.3 再扩 |
| 2 | bench dim 没合适 tag 怎么办？ | 给 `uncategorized` 临时 tag + log warning，等 v0.3 设计新 tag |
| 3 | 跨 range score normalize → min-max 还是 z-score？ | min-max（与 vbench_weighted 同语义） |
| 4 | `--capability X` 与 `--no-judge` 同时 → 自动过滤还是 fail-fast？ | **自动过滤** + log 列出被跳过的 |
| 5 | 同 source dedup 时哪个名字优先（短名 vs source-qualified）？ | 短名优先（用户更易识别） |
| 6 | capability 评测结果是否进 cross_benchmark.json？ | 进 `results/capability/<tag>.json` 独立文件 |

---

## 14. 风险

| 风险 | 影响 | 对策 |
|---|---|---|
| 44 个 tag 用户记不住 | 用户体验 | `capabilities list` 默认显示 10 顶层 + counts；`--show-sub` 展开 |
| metric 打 tag 主观，多人意见不一 | 维护成本 | design doc 列完整清单 + PR review 强制走 owner approval |
| 跨 source min-max normalize 引入新数（用户认不出）| 信任 | 默认输出含 raw + normalized + contributor list；`--no-normalize` 可关 |
| 同 source dedup 漏掉，metric 跑两遍 | 计算浪费 | dedup_key 严格匹配 `canonical_source::name`；CI 加重复跑检测 |
| Tag schema v1 → v2 升级用户脚本 break | 演进 | `tag_schema_version: 1` 顶部强制；v2 加 migration 工具 |
| Plugin metric 没声明 tag | 集成痛点 | manifest schema 要求 `tags` 字段；缺则 `videvalkit validate` warn（v0.3）|

---

## 15. 决策快照（用户 2026-05-20 已确认）

- ✅ Tag taxonomy 两层：10 顶层 + 34 子 = 44 个 controlled vocab
- ✅ metric / bench dim entry 强制声明 `tags: [...]`
- ✅ CLI 新增 `capabilities list/show` + `eval --capability X`（与 `--bench` / `--name` 并存）
- ✅ 默认聚合：min-max normalize → dedup by canonical_source → sub-tag mean → top-tag mean
- ✅ 不允许 free-form tag · 不允许用户自定义 capability bundle（v0.4 候选）
- ✅ Plugin metric 用 controlled vocab；v0.4 再开自定义 tag
- ✅ **进 v0.2**，工时 +3.1 day · v0.2 总 25.1 → 28.2 day
- ✅ `tag_schema_version: 1` 顶部强制声明，未来升级有 hook

---

## 16. 与其他设计文档的关系

| 文档 | 关系 |
|---|---|
| [`VIDEO_METRICS_DESIGN.md`](VIDEO_METRICS_DESIGN.md) | §6 SUPPORTED_METRICS schema 加 `tags` 字段；本文 §4 列完整 20 metric → tag 映射 |
| [`INTEGRATION_FRAMEWORK_DESIGN.md`](INTEGRATION_FRAMEWORK_DESIGN.md) | §5.2 schema 加 `tags` 字段说明 |
| [`JUDGE_SELECTION_DESIGN.md`](JUDGE_SELECTION_DESIGN.md) | 与 `--no-judge` 正交，但 `--capability X --no-judge` 是常用组合 |
| [`QUICK_EVAL_DESIGN.md`](QUICK_EVAL_DESIGN.md) | profile 可在内部声明 `capabilities: [motion, visual_quality]` 字段（v0.2 可选）|
| [`PRODUCT_DESIGN.md`](PRODUCT_DESIGN.md) | §4.1 加 capability 行；§5 加 sub-pillar；§6.2 工时；§6.3 v0.4 候选加 "用户自定义 capability bundle" |
| `DEV_MANUAL.md` | §15 新增 "Capability Tags" 章节 |
| `TEST_MANUAL.md` | 加 "capability coverage 一致性" 章节（每 tag 至少 1 contributor） |

---

—— end of design v0.1 ——
