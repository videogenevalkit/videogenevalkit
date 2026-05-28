# videogenevalkit Wiki

**Unified evaluation toolkit for text-to-video generation.**
One CLI · one workspace · one schema. 10 benchmarks · 8+ judges · 20 metrics · 44 capability tags.

> Status: `v0.2-dev` · 24 PRs merged · 363 tests green · ~97% of v0.2 scope.
> This wiki is the primary documentation. Design rationale lives in [`docs/design/`](../design/).

---

## Quick navigation

| Section | Pages |
|---|---|
| **Start here** | [Getting Started](Getting-Started.md) · [Core Concepts](Concepts.md) |
| **Guides** | [Judge Selection](guides/Judge-Selection.md) · [Profiles & Quick Eval](guides/Profiles-and-Quick-Eval.md) · [Training Monitor](guides/Training-Monitor.md) · [Extending](guides/Extending.md) |
| **Reference** | [CLI](reference/CLI.md) · [Benchmarks](reference/Benchmarks.md) · [Metrics](reference/Metrics.md) · [Judges](reference/Judges.md) · [Capability Tags](reference/Capability-Tags.md) |
| **Project** | [Architecture](Architecture.md) · [Roadmap & Status](Roadmap.md) · [Contributing](Contributing.md) |

### Full documentation map

This wiki is the current, operational reference. Two other doc sets exist:

| Doc set | Location | Status | Use for |
|---|---|---|---|
| **Wiki** (this) | `docs/wiki/` | ✅ current (v0.2) | day-to-day usage, reference, architecture |
| **Manuals** | `docs/DEV_MANUAL.md` · `TEST_MANUAL.md` · `USER_MANUAL_{en,cn}.md` | ⚠️ v0.0.1-era + banners | deep architecture rationale (DEV), paper-alignment validation tables (TEST), long-form install (USER) |
| **Design archive** | `docs/design/` | 🔒 frozen | *why* each v0.2 subsystem was built the way it is (8 design docs + slide deck) |

> Rule of thumb: **need to *do* something → wiki**; **need to know *why* → design**; **need validation numbers → TEST_MANUAL**.

---

## What it is

videogenevalkit unifies the fragmented T2V-evaluation landscape — VBench, VBench-2.0,
Video-Bench, WorldJen, WorldScore, T2V-CompBench and more — behind a single interface.

**Three ways to evaluate** (all functional):

| Entry point | Command | Answers |
|---|---|---|
| **Benchmark** | `videvalkit eval --bench vbench` | "How does my model score on VBench?" |
| **Metric** | `videvalkit metric run --name fvd ...` | "What's the FVD of these videos?" |
| **Capability** | `videvalkit capabilities eval motion ...` | "How good is motion across all metrics?" |

**Three design promises:**

| Promise | Meaning |
|---|---|
| Adapter, not reimplementation | Each benchmark wraps upstream code byte-for-byte |
| Pluggable backends | judges / metrics / aggregators are swappable registry entries |
| Plugin-first | extend via YAML / pip / local dir — no fork required |

---

## 30-second tour

```bash
# health check — see devices, benchmarks, metrics, profiles, coverage
videvalkit doctor

# run a benchmark with a quick profile, no judge needed
videvalkit eval --bench vbench --profile quick --videos gen/ --workspace ws/

# run a single metric (FVD auto-downloads its backbone)
videvalkit metric run --name fvd --gen-videos gen/ --refs my-ref --allow-tiny-sample

# evaluate a whole capability across metrics
videvalkit capabilities eval motion --videos gen/
```

See [Getting Started](Getting-Started.md) for the full walkthrough.
