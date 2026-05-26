# Review Protocol — Quality Gate for All PRs

| 字段 | 内容 |
|---|---|
| 版本 | v0.1 |
| 状态 | 强制执行 — 所有 PR 必走 |
| 创建 | 2026-05-20 |
| 范围 | v0.2 全部代码 PR · v0.3+ 持续适用 |
| 关联 | [`PRODUCT_DESIGN.md`](PRODUCT_DESIGN.md) §11 · `.github/pull_request_template.md` · `.claude/skills/videvalkit-maintainer/` |

> 这份是 **单一真源**。任何关于"PR 怎么过、什么算 ready to merge、CI 跑什么"的问题都查这里。

---

## 0. 为什么需要这份协议

v0.2 起 videogenevalkit 进入功能密集期：

- 4 支柱 + 1 横切，14+ 个独立 PR
- 长周期协同开发，多人参与
- 每加一个 metric / bench / judge 都引入位级一致、paper 对齐、license 等硬约束
- 没有强制 review gate → 一周漂移 → 半月返工

**目标**：任何 PR 在 "合并 → 跑通 → 与 paper 对齐 → 文档 / memory 同步" 全链路都不出意外。

---

## 1. 三层 Review 模型

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: 自动 CI                                              │
│   ruff · pytest · bit-exact · paper-alignment · schema valid  │
│         ↓ 全绿才能进 Layer 2                                  │
├─────────────────────────────────────────────────────────────┤
│ Layer 2: PR 自查 (作者填 template)                            │
│   12 项 checklist · 作者必须勾完才申请 review                  │
│         ↓ 自查无遗漏才能进 Layer 3                            │
├─────────────────────────────────────────────────────────────┤
│ Layer 3: Peer review                                         │
│   至少 1 人按 acceptance gate（§3）+ 5 个问题（§6）过一遍      │
│         ↓ approved 才能 merge                                │
└─────────────────────────────────────────────────────────────┘
```

任意一层不过 → 不允许 merge，无例外。

---

## 2. Layer 1 — 自动 CI 检查清单

`.github/workflows/ci.yml` 配置（v0.2 必落地）：

```yaml
jobs:
  lint:
    - ruff check src/ tests/ scripts/
    - ruff format --check src/ tests/

  unit-tests:
    - pytest tests/ -x --tb=short
    # 包括但不限于：
    #   tests/test_skeleton.py
    #   tests/test_registry_schema.py        ← needs_judge / tags / source 必填
    #   tests/test_capability_taxonomy_consistency.py ← tag 在 44 个 controlled vocab 里
    #   tests/test_judge_resolution.py
    #   tests/test_metric_kind_fail_fast.py

  bit-exact-lift:                            # lift-out PR 必跑
    if: contains(github.event.pull_request.labels.*.name, 'lift-out')
    - pytest tests/test_metric_lift_bit_exact.py -v

  paper-alignment:                           # new metric/bench PR 必跑
    if: contains(github.event.pull_request.labels.*.name, 'new-metric') ||
        contains(github.event.pull_request.labels.*.name, 'new-bench')
    - pytest tests/test_paper_alignment*.py -v

  doc-checks:                                # 所有 PR 都跑
    - python scripts/check_design_doc_consistency.py
    - python scripts/check_doc_links.py
    - python scripts/check_pr_template_filled.py    ← PR body 必含 12 项 checklist
```

**关键脚本**（v0.2 新增）：

| 脚本 | 检查 |
|---|---|
| `check_design_doc_consistency.py` | 跨 5 份子 doc 数字 / 字段 / metric 数 / 工时一致 |
| `check_doc_links.py` | docs/*.md 内部链接 (`[X](Y)`) 全部活的，外部链接 200 |
| `check_pr_template_filled.py` | PR body 含 `## What / Why / How tested / Checklist / Risks` 五段 |
| `test_registry_schema.py` | SUPPORTED_BENCHMARKS / SUPPORTED_JUDGES / SUPPORTED_METRICS 每条 entry 字段完整 |
| `test_capability_taxonomy_consistency.py` | 每 metric 至少 1 tag · tag 来自 44 controlled vocab · 每 tag 至少 1 contributor |
| `test_metric_lift_bit_exact.py` | 12 个 lift metric 的双入口数值差 ≤ 1e-6 |

---

## 3. Layer 2 — PR 类型 × Acceptance Gate

每个 PR 必须打 **type label**（CI 据此选哪些 gate 必跑）：

| Type label | 必过 acceptance gate |
|---|---|
| **`new-metric`** (从 0 写新) | ✓ paper-alignment ± tolerance · ✓ `tags` 字段 (controlled vocab) · ✓ `needs_judge` / `compute_kind` 字段 · ✓ `metric show` 输出含 source / license · ✓ USER_MANUAL_{en,cn} 加 snippet |
| **`lift-out`** (从 bench 抽出) | ✓ **bit-exact ≤ 1e-6** (硬约束) · ✓ bench-level 回归 (原 bench mean \|Δ\| 不变) · ✓ 双入口集成测试 · ✓ 同 new-metric 5 项基础字段 |
| **`new-bench`** | ✓ smoke + 完整 integration test · ✓ `dim_tags` 完整 (该 bench 所有 dim 都打 tag) · ✓ doctor 加条目 · ✓ TEST_MANUAL 加 paper-alignment 行 (mean \|Δ\| ≤ 5%) |
| **`new-judge`** | ✓ `judges test` 通过 (reach + auth) · ✓ 至少 1 个 bench 跑通用例 · ✓ pricing.py 加价格（若 managed API） · ✓ user yaml example 加段 |
| **`schema-change`** | ✓ migration test · ✓ `tag_schema_version` / `manifest_schema_version` bump · ✓ 老 entry loader 兼容性测试 · ✓ doc 顶部加 changelog |
| **`cli-change`** | ✓ --help 输出更新 · ✓ 至少 1 个 example 加到 USER_MANUAL · ✓ doctor 不破 · ✓ 自动补全脚本（若有）同步 |
| **`doc-only`** | ✓ 跨 5 份子 design doc 一致性 · ✓ 链接不死 · ✓ memory 同步 (若涉及决策) |
| **`infra`** (CI / build / packaging) | ✓ 在 fresh env 装一遍跑过 · ✓ Linux + macOS dev 态都过 · ✓ 不破现有 doctor |
| **`refactor`** (无功能变更) | ✓ 全 test 不少 1 个 · ✓ 至少 1 个性能基线（无大 regression） · ✓ git history 干净（每 commit 可编译）|

> 一个 PR 可以打多 label（如 `lift-out + cli-change`），必过 gate 是并集。

---

## 4. Layer 2 — PR 自查 checklist (PR template)

完整模板在 `.github/pull_request_template.md`。摘要：

```markdown
## What
<1-2 sentences>

## Why
<关联到哪份 design doc 的哪节>

## How tested
- [ ] 新增 / 修改的 test 列表
- [ ] 手动 smoke 跑过

## Type label
<打勾：new-metric / lift-out / new-bench / new-judge / schema-change /
       cli-change / doc-only / infra / refactor>

## Checklist (12 项)
- [ ] Lint pass · pytest 全绿
- [ ] 注册表字段完整 (kind / source / needs_judge / compute_kind / tags / license)
- [ ] 双入口位级一致 (lift-out 必勾)
- [ ] Paper alignment 数字记录 (new-metric / new-bench 必勾)
- [ ] Capability tags 来自 44 controlled vocab
- [ ] 影响的 design doc 已更新（列出）
- [ ] USER_MANUAL_{en,cn} 用法示例加了
- [ ] TEST_MANUAL paper-alignment 数刷新（若适用）
- [ ] README / PPT 数字无需改（或已改）
- [ ] License 字段填写
- [ ] memory 跨会话决策已存（若适用）
- [ ] 不引入新的 long-running CI 步骤（> 5 min）

## Risks
<本 PR 可能 break 什么 + 缓解>

## Out of scope
<故意不做的相邻事项>
```

**作者义务**：所有 12 项必须显式 ✓ 或显式 N/A + 一句原因。空 / 没勾 → CI 直接 `check_pr_template_filled.py` 红。

---

## 5. Layer 3 — Peer Review 视角

Reviewer 按 **5 个问题** 顺序问：

```
1. 这是哪份 design doc 里规划的？哪一节？
   ↓ 找不到
   - 要么 push back 要求先补 design
   - 要么打 P3 标签到 v0.3+ backlog
   
2. 它对应的 type label 的 acceptance gate 全过了吗？(§3)
   ↓ 缺一项
   - request changes，列出哪项

3. 它会 break 现有用户哪些 path？
   ↓ 风险大
   - 要求迁移文档 + 灰度 flag

4. CI 全绿吗？
   ↓ 红
   - 不看代码细节，要求先修绿

5. 30 天后我能否在 design doc / memory 里查到这次决策？
   ↓ 否
   - 要求补 doc / memory entry
```

**最少 reviewer 数**：v0.2 期 **1 人**（项目小，避免 review 阻塞）；v1.0 起 **2 人**。

---

## 6. 周期 Review（跨 PR 的一致性维护）

| 周期 | 责任人 | 检查内容 |
|---|---|---|
| **每 PR** | 作者 + reviewer | Layer 1-3 全套 |
| **每周** | maintainer | 跑 `check_design_doc_consistency.py`；扫 memory 看新决策是否需 doc 化 |
| **每 milestone (M0-M6 / D1-D6 / T1-T6 / F1-F12)** | maintainer | 完整 smoke：9 bench + 20 metric + 3 profile + 44 tag 都跑一次小样本；TEST_MANUAL 数字刷新 |
| **每 release (v0.x.0)** | maintainer + 项目 lead | TEST_MANUAL paper-alignment 数字全刷新；validation/expected/ 重跑；CHANGELOG.md 写好 |

---

## 7. 例外处理

允许跳 gate 的**唯一**情况：**项目 lead 显式打 `bypass-review` label + PR description 写明理由**。次数与原因每月汇总，看是否需要调整协议。

不允许：
- ❌ "我急" → 不行，按 PR 拆小
- ❌ "这是 trivial 改动" → trivial 改动走全套 CI 也只多 2 分钟
- ❌ "reviewer 太慢" → 拆小 PR 让 reviewer 易看，或加 reviewer 数

---

## 8. Maintainer agent 的承诺

我（长期 maintainer agent）在每次接到 paper/repo → approve → 进 coding 时，会：

1. **自动按 PR template 填好** PR description（不让 reviewer 现编）
2. **预跑 Layer 1 所有 CI 项** 在本地（虽然 CI 会再跑一次，但本地先过避免 PR 红来红去）
3. **自检 §3 对应 type label 的 acceptance gate**（lift PR 跑 bit-exact、new metric PR 跑 paper align）
4. **同步更新 design doc + memory**（避免事后忘）
5. **打好 type label + 关联 issue / milestone**

具体见 `.claude/skills/videvalkit-maintainer/SKILL.md`。

---

## 9. 文件改动清单（v0.2 落地本协议）

### 新增

| 路径 | 用途 |
|---|---|
| `docs/REVIEW_PROTOCOL.md` | 本文 (单一真源) |
| `.github/pull_request_template.md` | PR 模板 (作者必填) |
| `.github/labels.yml` | 9 个 type label 定义 (`new-metric`, `lift-out`, ...) |
| `scripts/check_design_doc_consistency.py` | 跨 doc 一致性 |
| `scripts/check_doc_links.py` | 内外链接 |
| `scripts/check_pr_template_filled.py` | PR body 5 段是否齐 |
| `tests/test_registry_schema.py` | 注册表必填字段 |
| `tests/test_capability_taxonomy_consistency.py` | tag controlled vocab |
| `tests/test_metric_lift_bit_exact.py` | 双入口位级一致 |
| `.claude/skills/videvalkit-maintainer/` | 我自身行为的 skill 化（见单独 README）|

### 修改

| 路径 | 改点 |
|---|---|
| `.github/workflows/ci.yml` | 新增 lint / unit / bit-exact / paper-align / doc-checks 5 job |
| `docs/PRODUCT_DESIGN.md` | §6.2 工时 +1.5 d (28.2 → 29.7)；§11 决策快照加 review protocol 行 |

---

## 10. 决策快照

- ✅ 三层 review：自动 CI + PR 自查 + peer review，任何一层不过不可 merge
- ✅ 9 种 PR type label，每种独立 acceptance gate
- ✅ PR template 12 项 checklist，强制全勾或 N/A + 理由
- ✅ Peer review 用 5 问法
- ✅ v0.2 期 1 reviewer；v1.0 起 2 reviewer
- ✅ Bypass 唯一路径：项目 lead `bypass-review` label
- ✅ Maintainer agent 自动按本协议执行（skill 化）
- ✅ 工时 +1.5 d 入 v0.2，总 28.2 → 29.7 day

—— end of protocol v0.1 ——
