# Capability Tags — Quick Reference

> 完整设计见 `docs/CAPABILITY_TAGS_DESIGN.md`。本文是写 metric 时打 tag 的速查表。
>
> **v0.2 controlled vocab，44 个 tag (10 顶 + 34 子)，只能从这里挑，不允许 free-form。**

---

## 10 顶层 + 34 子 tag

| 顶层 | 子 tag | 测什么 |
|---|---|---|
| **motion** | `motion.smoothness` | 帧间平滑度 |
| | `motion.magnitude` | 运动幅度 |
| | `motion.accuracy` | 动作与 prompt 对齐 |
| | `motion.naturalness` | 运动是否自然 |
| **visual_quality** | `vq.aesthetic` | 美学分 |
| | `vq.imaging` | 成像质量 |
| | `vq.artifact_free` | 无 artifact |
| | `vq.sharpness` | 清晰度 |
| **text_alignment** | `align.text2video` | 文-视频对齐 |
| | `align.prompt_following` | 整体 prompt 遵循度 |
| | `align.action_verb` | 动词对齐 |
| **object_fidelity** | `obj.presence` | 物体存在 |
| | `obj.count` | 数量准确 |
| | `obj.attribute` | 属性（颜色/材质） |
| | `obj.binding` | object-attribute 绑定 |
| **subject_consistency** | `subj.identity` | 主体身份 |
| | `subj.appearance` | 外观连续 |
| | `subj.character` | 角色一致 |
| **physical_plausibility** | `phys.gravity` | 重力合理 |
| | `phys.causality` | 因果合理 |
| | `phys.anatomy` | 解剖合理 |
| | `phys.kinematics` | 运动学合理 |
| **temporal_coherence** | `temp.flickering` | 闪烁 |
| | `temp.continuity` | 时序连续 |
| | `temp.scene_consistency` | 场景一致 |
| **realism** | `real.distribution` | 分布层真实（FVD 系）|
| | `real.detection` | 是否 AI 生成 |
| | `real.artifact_rate` | artifact 频率 |
| **compositional** | `comp.multi_object` | 多物体场景 |
| | `comp.spatial` | 空间关系 |
| | `comp.numeracy` | 数量构成 |
| **style** | `style.aesthetic` | 美学风格 |
| | `style.cg_anime` | CG / 动画风格 |
| | `style.consistency` | 风格一致 |

**合计 10 + 34 = 44 tag**，固化为 v1 controlled vocab。

---

## v0.2 现有 metric 的 tag 映射

| Metric | Tags |
|---|---|
| `fvd` | `realism.distribution` |
| `vfid` | `realism.distribution` |
| `kvd` | `realism.distribution` |
| `clip-fvd` | `realism.distribution` |
| `clip-score` | `align.text2video` |
| `viclip-score` | `align.text2video` · `align.prompt_following` |
| `aesthetic-quality` | `vq.aesthetic` · `style.aesthetic` |
| `imaging-quality` | `vq.imaging` · `vq.sharpness` |
| `motion-smoothness` | `motion.smoothness` · `temp.flickering` |
| `temporal-flickering` | `temp.flickering` · `vq.artifact_free` |
| `subject-consistency` | `subj.identity` · `subj.appearance` |
| `background-consistency` | `subj.appearance` · `temp.continuity` |
| `dynamic-degree` | `motion.magnitude` |
| `motion-magnitude` | `motion.magnitude` |
| `object-binding` | `obj.binding` · `obj.presence` |
| `spatial-relationship` | `comp.spatial` |
| `numeracy` | `comp.numeracy` · `obj.count` |
| `motion-accuracy` | `motion.accuracy` · `align.action_verb` |
| `identity-preservation` | `subj.identity` · `subj.character` |
| `artifact-diagnostic` | `real.artifact_rate` · `vq.artifact_free` |

---

## 打 tag 流程（写新 metric 时）

1. 算法本质上测什么 → 找顶层 tag
2. 选 1-2 个最具体的子 tag（避免挂太多）
3. 在 metric registry entry 写 `tags=["motion.smoothness", "temp.flickering"]`
4. 不允许 free-form：必须从上面 44 个里选
5. 如果觉得 44 个不够 → 不要加 free-form tag，先 PR `docs/CAPABILITY_TAGS_DESIGN.md` 提议加新 tag（走 schema-change PR 流程）

---

## 反向：按 tag 找 metric (CLI)

```bash
videvalkit capabilities show motion.smoothness
# motion.smoothness:
#   metric/motion-smoothness   (canonical)
#   vbench/motion_smoothness   (same source, deduped)
```
