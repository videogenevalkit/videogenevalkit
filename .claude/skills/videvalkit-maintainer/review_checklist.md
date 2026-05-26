# PR Review Checklist — Quick Reference

> 全协议在 `docs/REVIEW_PROTOCOL.md`。本文是 maintainer agent 自查时用的速查表。

---

## 1. PR Type Label × Acceptance Gate

每个 PR 必打 type label，CI 据此选哪些 gate 必跑：

| Label | 必过 acceptance gate |
|---|---|
| **`new-metric`** | paper-alignment ± tol · `tags` 字段 · `needs_judge` / `compute_kind` · `metric show` 含 source/license · USER_MANUAL snippet |
| **`lift-out`** | **bit-exact ≤ 1e-6 (硬约束)** · bench-level 回归 · 双入口集成测试 · 同 new-metric 基础字段 |
| **`new-bench`** | smoke + integration test · `dim_tags` 完整 · doctor 加条目 · TEST_MANUAL paper-alignment 行 (mean \|Δ\| ≤ 5%) |
| **`new-judge`** | `judges test` 通过 · ≥ 1 bench 用例 · pricing.py 加价格（managed API） · user yaml example |
| **`schema-change`** | migration test · `tag_schema_version` / `manifest_schema_version` bump · 老 entry loader 兼容性 · doc 顶部 changelog |
| **`cli-change`** | --help 更新 · ≥ 1 USER_MANUAL example · doctor 不破 · 自动补全脚本同步 |
| **`doc-only`** | 跨 5 份子 design doc 一致 · 链接不死 · memory 同步（若涉决策） |
| **`infra`** | fresh env 装一遍跑过 · Linux + macOS dev 都过 · doctor 不破 |
| **`refactor`** | 全 test 不少 1 个 · 性能基线无大 regression · git history 干净 |

可打多 label，必过 gate 取并集。

---

## 2. 12 项自查 checklist (PR template body)

每项必须 explicit ✓ 或 explicit N/A + 一句原因：

1. [ ] Lint pass · pytest 全绿
2. [ ] 注册表 entry 字段完整 (`kind` / `source` / `needs_judge` / `compute_kind` / `tags` / `license`)
3. [ ] 双入口位级一致 ≤ 1e-6 (`lift-out` 必勾)
4. [ ] Paper alignment mean \|Δ\| recorded (`new-metric`/`new-bench` 必勾，附数字)
5. [ ] Capability tags 来自 44 controlled vocab
6. [ ] 影响的 design doc 已更新（列出）
7. [ ] USER_MANUAL_{en,cn} 用法示例加了
8. [ ] TEST_MANUAL paper-alignment 表已刷新（若适用）
9. [ ] README / PPT 数字无需改（或已改）
10. [ ] License 字段填写
11. [ ] memory 跨会话决策已存（若适用）
12. [ ] 不引入新的长 CI 步骤（> 5 min）

---

## 3. Reviewer 5 问

合并前 reviewer 按顺序问：

```
1. 这是哪份 design doc 里规划的？哪一节？
   ↓ 找不到 → push back 要求先补 design，或打 P3 backlog
2. type label 的 acceptance gate 全过了吗？
   ↓ 缺一项 → request changes
3. 会 break 现有用户哪些 path？
   ↓ 风险大 → 要求迁移文档 + 灰度 flag
4. CI 全绿吗？
   ↓ 红 → 不看代码细节，先修绿
5. 30 天后我能否在 doc / memory 查到这次决策？
   ↓ 否 → 要求补 doc / memory entry
```

---

## 4. Maintainer agent 的承诺

每次 user approve 后进 coding（仅 GPU+Linux phase），我自动：

1. ✓ 按 PR template 填好 description（不让 reviewer 现编）
2. ✓ 本地预跑 Layer 1 全部 CI 项
3. ✓ 自检对应 type label 的 acceptance gate
4. ✓ 同步更新 design doc + memory
5. ✓ 打 type label + 关联 issue / milestone

---

## 5. CI Job 速查

`.github/workflows/ci.yml` 的 jobs:

| Job | 时机 | 内容 |
|---|---|---|
| `lint` | 所有 PR | ruff check / format |
| `unit-tests` | 所有 PR | pytest 全套 |
| `bit-exact-lift` | `lift-out` label | `tests/test_metric_lift_bit_exact.py` |
| `paper-alignment` | `new-metric`/`new-bench` label | `tests/test_paper_alignment*.py` |
| `doc-checks` | 所有 PR | `check_design_doc_consistency.py` + `check_doc_links.py` + `check_pr_template_filled.py` |

---

## 6. Bypass 唯一路径

`bypass-review` label + PR description 写明理由 + 项目 lead 显式 approve。次数 / 原因每月汇总。

trivial 改 / 急 / reviewer 慢 **都不是** bypass 理由。
