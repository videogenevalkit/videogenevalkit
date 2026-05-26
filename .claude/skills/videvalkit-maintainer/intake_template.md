# Paper / Repo Intake Report Template

> 用法：用户发新 paper / repo 时，**第一反应**用这个模板回报告，**不立即写代码**。等用户 approve 后再进 coding flow（且仅在 GPU+Linux 阶段）。

---

## Intake Report · `<Paper/Repo Name>`

**Paper**: `<arXiv URL / PDF path>`, `<YYYY-MM>`
**GitHub**: `<repo URL>`
**HF Data** (if any): `<repo>`
**机构 / 作者**: `<short list>`

### 1. 一句话本质

`<1-2 sentences summarizing what this evaluation method does and why it exists>`

### 2. 分类

```
[ ] 通用 T2V quality       (任何 T2V 模型都该跑的基础体检)
[ ] 专用维度               (测某个具体能力)
[ ] 完整 benchmark         (prompts + dims + scoring 全套)
[ ] judge-eval bench       (评测 judge 而非 T2V 模型 — 罕见)
[ ] 已被覆盖 (跳过)
```

### 3. 已有覆盖检查

| 维度 | 对比 | 结论 |
|---|---|---|
| vs 现有 20 metric | `<重叠/替代/互补/无关>` | `<...>` |
| vs 现有 9 benchmark | `<同上>` | `<...>` |
| vs 现有 8 judge | `<同上>` | `<...>` |
| vs 44 capability tag | `<可挂哪几个 tag>` | `<...>` |

### 4. 接入轨道建议

```
[ ] A · Manifest YAML (Track A)        — prompts/dims/scorers 全可声明
[ ] B · Python adapter (Track B)       — 复杂 staging / 多阶段 / upstream 子进程
[ ] C · Standalone metric (lift / new) — 算法 lift 出来作独立 metric
[ ] D · 仅作 dim within existing bench  — 不独立暴露
```

**推荐**: `<A/B/C/D 之一或组合>` —— `<reason>`

### 5. 依赖与风险

| 项 | 详情 |
|---|---|
| Checkpoint | `<X GB · 哪里拿 (HF/Google Drive/...)>` |
| 算子 | `<PyTorch high-level / custom CUDA kernel / pure CV>` |
| Judge 需求 | `<None / VLM / LLM / 都要>` |
| 算力 | `<CPU only / GPU X GB / multi-GPU>` |
| License | `<Apache / MIT / research-only / unclear ← 必须查清>` |
| Paper 对齐难度 | `<低/中/高 + 原因>` |
| Code 质量 | `<paper 发布时间 + repo 维护状态 + 关键文件 LoC>` |
| 与已有 metric 重叠 | `<列出重叠的 metric/dim 名字>` |

### 6. 工时估计

| 子项 | 工时 |
|---|---:|
| 读 paper + repo + 设计接入 | `<X> d` |
| 写 adapter / metric 代码 | `<X> d` |
| Paper alignment 测试 | `<X> d` |
| 文档 / USER_MANUAL | `<X> d` |
| **合计** | **`<X> d`** |

### 7. 推荐优先级

```
[ ] P0 — 立即接 (高价值 + 低风险，吸收进当前 milestone)
[ ] P1 — 当前 sprint (v0.2 内)
[ ] P2 — v0.3 候选
[ ] P3 — backlog (调研即可)
```

### 8. 接入策略（可选拆分）

如适用，提多个 option 让 user 选：

**Option A**: `<title>` — `<scope, effort, version>`
**Option B**: `<title>` — `<scope, effort, version>`
**Option C**: `<title>` — `<scope, effort, version>`

**Recommend**: `<which option(s) + reason>`

### 9. 下一步

- [ ] **同意 Option X** → 我开始 [ design 落地 / coding (如已在 GPU+Linux 阶段) ]
- [ ] **修改方案** → user 指出哪里改
- [ ] **需要补信息** → `<what's missing>`

---

## 报告写完后的事

无论 approve / defer / reject：
1. **存 memory**: `/Users/yue/.claude/projects/-Users-yue-Documents-t2v-benchmark/memory/integrated_<name>.md`
   - paper info / decision / 风险 / 涉及的 doc 位置
2. **更新 doc** (若 approved):
   - `docs/VIDEO_METRICS_DESIGN.md` §4 或 `docs/PRODUCT_DESIGN.md` §6.3 加入
   - 工时累加到 v0.2 / v0.3 totals
3. **不立即写代码** —— 等 GPU+Linux phase
