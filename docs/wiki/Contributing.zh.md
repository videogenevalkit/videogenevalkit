# 贡献指南

[← 首页](../index.md)

每一处改动都要经过三层评审关卡。完整协议:
[`docs/design/REVIEW_PROTOCOL.md`](../design/REVIEW_PROTOCOL.md)。

---

## 分支模型

```
main                    稳定线
  ↑
v0.2-dev                集成分支(所有 v0.2 PR 落在这里)
  ↑  ↑  ↑
feat/<short-name>       每个 PR 一个分支
```

PR 以 `v0.2-dev` 为目标。`v0.2-dev → main` 在 v0.2 完成后一次性大合并。
不对 `v0.2-dev` / `main` 强推(force-push)。

---

## 三层评审

| 层 | 关卡 |
|---|---|
| **1. CI** | ruff · pytest · doc-links · 设计文档一致性 · PR 模板检查 |
| **2. 自查** | PR 模板中的 12 项清单,全部勾选或标 N/A |
| **3. 同行评审** | 一位评审者执行 5 问检查 |

三关全过,PR 才能合并。

---

## PR 类型标签(至少选 1 个)

| 标签 | 额外关卡 |
|---|---|
| `new-metric` | 论文对齐 ± 容差 · 标签 · `metric show` 字段 |
| `lift-out` | 相对基准路径**逐位一致 ≤ 1e-6** · 基准回归 |
| `new-bench` | 冒烟 + 集成 · `dim_tags` · TEST_MANUAL 行 |
| `new-judge` | `judges test` · ≥1 次基准运行 · 定价 |
| `schema-change` | 迁移测试 · 版本号提升 |
| `cli-change` | 更新 `--help` · USER_MANUAL 示例 |
| `doc-only` / `infra` / `refactor` | 见协议 |

---

## 5 问同行检查

1. 哪份设计文档 / wiki 页面规划了此改动?
2. 该类型标签的验收关卡通过了吗?
3. 它会破坏哪条现有路径?
4. CI 是绿的吗?
5. 30 天后该决策是否可追溯(文档 / memory)?

---

## 推送前的本地检查

```bash
ruff check src/ tests/ scripts/
pytest tests/ -m "not slow and not needs_gpu"
python scripts/check_design_doc_consistency.py
python scripts/check_doc_links.py
```

**务必在合并*之前*确认测试全绿**,而非合并之后。

---

## 提交风格

约定式提交:`feat:` / `fix:` / `docs:` / `test:` / `chore:` / `refactor:`。
AI 协助的提交请附 `Co-Authored-By:` 尾注。
