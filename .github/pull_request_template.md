<!--
videogenevalkit PR template — REQUIRED for all PRs.
Full protocol: docs/REVIEW_PROTOCOL.md
CI checks this body has all 5 sections (## What / Why / How tested / Type label / Checklist / Risks).
-->

## What

<!-- 1-2 sentences. What does this PR do? -->


## Why

<!-- Which design doc + section motivates this? e.g.:
   docs/VIDEO_METRICS_DESIGN.md §4.1 artifact-diagnostic
   docs/QUICK_EVAL_DESIGN.md §3.2 standard profile
   docs/CAPABILITY_TAGS_DESIGN.md §4 metric→tag mapping
-->


## How tested

- [ ] Tests added/modified:
  - `tests/test_<...>.py` — <一句描述>
- [ ] Manual smoke run:
  ```bash
  # paste actual command(s) you ran
  ```


## Type label

<!-- Pick one or more — CI uses these to decide which acceptance gates run.
     See docs/REVIEW_PROTOCOL.md §3 for what each label requires. -->

- [ ] `new-metric`     — writing a new standalone metric from scratch
- [ ] `lift-out`       — extracting an existing bench dim into a standalone metric (bit-exact required)
- [ ] `new-bench`      — adding a new benchmark adapter
- [ ] `new-judge`      — adding a judge endpoint / SDK backend
- [ ] `schema-change`  — modifying registry schema, manifest schema, or capability taxonomy
- [ ] `cli-change`     — adding/modifying CLI subcommand or flag
- [ ] `doc-only`       — design doc / USER_MANUAL / README only
- [ ] `infra`          — CI, build, packaging, env yaml
- [ ] `refactor`       — internal refactor with no user-visible change


## Checklist (12 项)

<!-- Tick every box explicitly. If N/A, write "N/A — <one-line reason>" instead. -->

- [ ] Lint pass · pytest 全绿
- [ ] 注册表 entry 字段完整 (`kind` / `source` / `needs_judge` / `compute_kind` / `tags` / `license`)
- [ ] Double-entry bit-exact ≤ 1e-6 (`lift-out` 必勾；其他 N/A)
- [ ] Paper alignment mean |Δ| recorded (`new-metric` / `new-bench` 必勾；附数字)
- [ ] Capability tags 来自 44 controlled vocab (CAPABILITY_TAGS_DESIGN.md §3)
- [ ] 影响的 design doc 已更新（列出）:
  - <none / docs/...md §...>
- [ ] USER_MANUAL_en.md + USER_MANUAL_cn.md 用法示例已加（或不适用）
- [ ] TEST_MANUAL.md paper-alignment 表已刷新（若 new-metric / new-bench）
- [ ] README / PPT 数字无需改（或本 PR 已改）
- [ ] License 字段填了上游 / 本身 license
- [ ] memory 跨会话决策已存到 `~/.claude/projects/.../memory/`（若有）
- [ ] 不引入新的长 CI 步骤（> 5 min）


## Risks

<!-- What might this break? How to mitigate? -->


## Out of scope

<!-- Adjacent things this PR deliberately does NOT do. -->


---

<!-- 🤖 Reviewer's 5 questions:
1. 这是哪份 design doc 里规划的？
2. type label 的 acceptance gate 全过了吗？
3. 会 break 哪些现有 path？
4. CI 全绿吗？
5. 30 天后能在 doc / memory 查到这次决策吗？

Full protocol: docs/REVIEW_PROTOCOL.md
-->
