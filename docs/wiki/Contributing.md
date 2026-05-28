# Contributing

[← Home](../index.md)

Every change goes through a three-layer review gate. Full protocol:
[`docs/design/REVIEW_PROTOCOL.md`](../design/REVIEW_PROTOCOL.md).

---

## Branch model

```
main                    stable line
  ↑
v0.2-dev                integration branch (all v0.2 PRs land here)
  ↑  ↑  ↑
feat/<short-name>       one branch per PR
```

PRs target `v0.2-dev`. `v0.2-dev → main` is one big merge after v0.2 is done.
No force-push to `v0.2-dev` / `main`.

---

## Three review layers

| Layer | Gate |
|---|---|
| **1. CI** | ruff · pytest · doc-links · design-doc consistency · PR-template check |
| **2. Self-check** | 12-item checklist in the PR template, all ticked or N/A |
| **3. Peer** | one reviewer applies the 5-question check |

A PR cannot merge unless all three pass.

---

## PR type labels (pick ≥1)

| Label | Extra gate |
|---|---|
| `new-metric` | paper-alignment ± tol · tags · `metric show` fields |
| `lift-out` | **bit-exact ≤ 1e-6** vs bench path · bench regression |
| `new-bench` | smoke + integration · `dim_tags` · TEST_MANUAL row |
| `new-judge` | `judges test` · ≥1 bench run · pricing |
| `schema-change` | migration test · version bump |
| `cli-change` | `--help` updated · USER_MANUAL example |
| `doc-only` / `infra` / `refactor` | see protocol |

---

## The 5-question peer check

1. Which design doc / wiki page planned this?
2. Did the type-label's acceptance gate pass?
3. What existing path does it break?
4. Is CI green?
5. Will the decision be traceable in 30 days (doc / memory)?

---

## Local checks before pushing

```bash
ruff check src/ tests/ scripts/
pytest tests/ -m "not slow and not needs_gpu"
python scripts/check_design_doc_consistency.py
python scripts/check_doc_links.py
```

**Always confirm tests are green BEFORE merging**, not after.

---

## Commit style

Conventional commits: `feat:` / `fix:` / `docs:` / `test:` / `chore:` / `refactor:`.
Include a `Co-Authored-By:` trailer for AI-assisted commits.
