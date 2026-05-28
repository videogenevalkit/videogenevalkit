# 能力标签 — 设计

> 按能力评测这条轴的理念。标签清单与 CLI 见
> [能力标签参考](../wiki/reference/Capability-Tags.md)。

---

## 1. 问题

不同基准切分质量的方式不同,而同一项能力(比如运动)由散落在 VBench、WorldScore
和独立代码中的多个指标衡量。一个问“运动表现如何?”的用户不该需要知道哪个基准
拥有哪个指标。能力轴是第三入口——与 `--bench`(可与论文对比)和 `--name`
(单个标量)并列——让你*按能力*评测。

## 2. 受控词表

固定的 **44 标签**词表:10 个顶层能力,每个带子标签(共 34 个)。规范形式为
`<prefix>.<leaf>`(如 `motion.smoothness`、`comp.spatial`、`real.distribution`)。
10 个顶层:motion、visual_quality、text_alignment、object_fidelity、
subject_consistency、physical_plausibility、temporal_coherence、realism、
compositional、style。

**为何受控而非自由格式**:自由格式标签会漂移成同义词
(`motion`/`movement`/`dynamics`),破坏跨基准分组。不在词表内的标签在加载时被拒,
一致性检查会核验每个指标/维度标签都在词表内。词表**带版本**
(`tag_schema_version = 1`);任何变更都提升版本。

## 3. 解析

`capabilities eval <tag>`:

1. **解析** — 顶层标签展开为其全部子标签;收集带其中任一标签的每个指标和基准维度。
2. **去重** — 抽出的指标与其来源基准维度共享同一规范来源,故只计一次(优先用指标)。
   这正是 lift 记录 `also_used_by` 的原因。
3. **运行** — 每个可运行贡献者(逐视频、无需评审)在视频上计算;需要
   参考集/prompt/评审 的指标会*带原因*被跳过(能力轴是快速的逐视频读数)。
4. **归一化** — 每个指标 min-max 到 [0, 1]。
5. **聚合** — 跨贡献者取均值(或 max/min)→ 一个能力分。

## 4. 范围

v0.2 交付词表、解析器,以及 `capabilities list/show/eval`。插件复用现有词表;
用户自定义标签是后续候选项。打标是指标/维度上的附加元数据——它绝不改变指标如何
计算,只改变它如何被分组。
