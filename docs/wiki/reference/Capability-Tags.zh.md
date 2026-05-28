# 能力标签参考

[← 首页](../../index.md)

一套固定的 **44 标签**受控词表(10 个顶层 + 34 个子级),给每个指标和
基准维度打标,从而能**按能力**评测。

```bash
videvalkit capabilities list [--show-sub]
videvalkit capabilities show motion
videvalkit capabilities eval motion --videos gen/
```

---

## 10 个顶层标签

| 顶层 | 子标签 | 衡量 |
|---|---|---|
| **motion** | smoothness · magnitude · accuracy · naturalness | 物体如何运动 |
| **visual_quality** | aesthetic · imaging · artifact_free · sharpness | 帧级质量 |
| **text_alignment** | text2video · prompt_following · action_verb | 是否遵循 prompt |
| **object_fidelity** | presence · count · attribute · binding | 物体是否正确 |
| **subject_consistency** | identity · appearance · character | 主体是否随时间保持一致 |
| **physical_plausibility** | gravity · causality · anatomy · kinematics | 物理/解剖是否真实 |
| **temporal_coherence** | flickering · continuity · scene_consistency | 帧间连贯性 |
| **realism** | distribution · detection · artifact_rate | 整体真实感 |
| **compositional** | multi_object · spatial · numeracy | 多物体场景 |
| **style** | aesthetic · cg_anime · consistency | 艺术风格 |

子标签规范形式为 `<prefix>.<leaf>`,例如 `motion.smoothness`、
`comp.spatial`、`real.distribution`。完整词表在
`src/videvalkit/configs/capability_taxonomy.py`。

---

## 解析如何运作

`capabilities eval <tag>`:

1. **解析** — 顶层展开为其全部子标签;收集打了其中任一标签的每个指标 +
   基准维度。
2. **去重** — 抽出的指标与其来源基准维度共享同一规范来源
   → 只计一次(优先用指标)。
3. **运行** — 每个可运行指标(逐视频、无需评审)在视频上计算。
4. **归一化** — 每个指标 min-max 到 [0, 1]。
5. **聚合** — 跨贡献者取均值(或 max/min)→ 能力分。

需要参考集 / prompt / 评审的指标,以及 shell,会**带原因被跳过**
(`eval --capability` 是快速的逐视频读数;那些请用 `metric run`)。

---

## 规则

- **仅受控词表** — 自由格式标签在加载时被拒绝。
- **带版本** — `tag_schema_version = 1`;词表变更则提升版本。
- v0.2:插件使用现有词表;自定义标签是 v0.4 候选项。

---

## 示例

```
$ videvalkit capabilities show motion
motion
  How things move — speed, smoothness, accuracy, naturalness
  expands to: [motion, motion.smoothness, motion.magnitude, motion.accuracy, motion.naturalness]

  source_kind   name                          tags
  ----------------------------------------------------------------
  bench_dim     vbench/motion_smoothness      motion.smoothness
  bench_dim     vbench/dynamic_degree         motion.magnitude
  bench_dim     worldscore/motion_magnitude   motion.magnitude
  ...
```

每个指标的标签见[指标](Metrics.md)。
