# Review Protocol

> The mandatory quality gate for every PR. The contributor-facing summary is in
> [Contributing](../wiki/Contributing.md); this is the full protocol and its
> rationale.

---

## 0. Why

The toolkit grows by adding benchmarks, metrics, and judges — each a chance to
silently break paper-alignment, the dual-entry bit-exact contract, or the registry
invariants. A uniform gate keeps correctness from depending on any one reviewer's
memory.

## 1. Three layers

| Layer | Gate |
|---|---|
| **1. CI** | ruff · pytest · doc-links · design-doc consistency · PR-template check |
| **2. Self-check** | the 12-item checklist in the PR template, all ticked or N/A |
| **3. Peer** | one reviewer applies the 5-question check |

A PR cannot merge unless all three pass. **Confirm the full regression is green
*before* the merge command, not after** — a hook failure means the commit did not
happen.

## 2. PR type labels (pick ≥ 1)

Each label adds an acceptance gate beyond CI:

| Label | Extra gate |
|---|---|
| `new-metric` | paper-alignment ± tol · tags in vocab · `metric show` fields |
| `lift-out` | **bit-exact ≤ 1e-6** vs the bench path · bench regression |
| `new-bench` | smoke + integration · `dim_tags` · TEST_MANUAL row |
| `new-judge` | `judges test` · ≥ 1 bench run · pricing |
| `schema-change` | migration test · version bump |
| `cli-change` | `--help` updated · USER_MANUAL example |
| `doc-only` / `infra` / `refactor` | lighter gates per this protocol |

## 3. The 5-question peer check

1. Which design doc / wiki page planned this?
2. Did the type label's acceptance gate pass?
3. What existing path does it break?
4. Is CI green?
5. Will the decision be traceable in 30 days (doc / memory)?

## 4. Periodic consistency

Beyond per-PR review, `check_design_doc_consistency.py` guards cross-cutting
invariants (capability taxonomy size = 44, metric tags in vocab, required registry
fields, every judge-using bench declares `paper_judge`). This catches drift that no
single PR review would.

## 5. Maintainer-agent commitment

The long-term maintenance agent follows: intake report → user approval → bit-exact
+ paper-aligned integration → one PR per item, each through this gate. The behavior
is skill-ized under `.claude/skills/videvalkit-maintainer/`.
