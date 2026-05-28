# videogenevalkit Wiki

**Unified evaluation toolkit for text-to-video generation.**
One CLI · one workspace · one schema. 10 benchmarks · 8+ judges · 20 metrics · 44 capability tags.

> Status: `v0.2-dev` · 384 tests green · v0.2 nearly complete.
> This wiki is the primary documentation. Design rationale lives in [`docs/design/`](design/PRODUCT_DESIGN.md).

---

## Quick navigation

| Section | Pages |
|---|---|
| **Start here** | [Getting Started](wiki/Getting-Started.md) · [Core Concepts](wiki/Concepts.md) |
| **Guides** | [Judge Selection](wiki/guides/Judge-Selection.md) · [Profiles & Quick Eval](wiki/guides/Profiles-and-Quick-Eval.md) · [Training Monitor](wiki/guides/Training-Monitor.md) · [Extending](wiki/guides/Extending.md) |
| **Reference** | [CLI](wiki/reference/CLI.md) · [Benchmarks](wiki/reference/Benchmarks.md) · [Metrics](wiki/reference/Metrics.md) · [Judges](wiki/reference/Judges.md) · [Capability Tags](wiki/reference/Capability-Tags.md) |
| **Project** | [Architecture](wiki/Architecture.md) · [Roadmap & Status](wiki/Roadmap.md) · [Contributing](wiki/Contributing.md) |

### Full documentation map

This wiki is the current, operational reference. Two other doc sets exist:

| Doc set | Location | Status | Use for |
|---|---|---|---|
| **Wiki** (this) | `docs/wiki/` | ✅ current (v0.2) | day-to-day usage, reference, architecture |
| **Manuals** | `docs/DEV_MANUAL.md` · `TEST_MANUAL.md` · `USER_MANUAL.md` (en/zh) | ⚠️ v0.0.1-era + banners | deep architecture rationale (DEV), paper-alignment validation tables (TEST), long-form install (USER) |
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

See [Getting Started](wiki/Getting-Started.md) for the full walkthrough.
