# Product Design

> **Why this exists.** The wiki tells you *what* the toolkit does and *how* to
> use it; this design archive records *why* it is built the way it is. For
> status, the metric scoreboard, and the version timeline, see the
> [Roadmap](../wiki/Roadmap.md). For the layered architecture and call flow,
> see [Architecture](../wiki/Architecture.md).

---

## 1. Vision

**videogenevalkit is the unified entry point for text-to-video evaluation** —
one CLI, one workspace, one schema — pulling the fragmented T2V landscape
(VBench, VBench-2.0, Video-Bench, WorldJen, WorldScore, T2V-CompBench, …)
behind a single interface and keeping it open to extension.

Three value propositions:

1. **Don't reinvent.** Every benchmark is a thin adapter over the upstream
   paper repo, byte-for-byte aligned to the official leaderboard.
2. **Don't lock the backend.** Judges / scorers / aggregators are swappable
   registry entries — local vLLM, Claude/Gemini/GPT, or the paper's native
   model, switched by one CLI flag.
3. **Don't force a fork.** New benchmark / metric / judge extends via manifest
   YAML, pip `entry_points`, or a local user dir — no toolkit edits.

### Scope boundaries — what we deliberately do *not* do

| Out of scope | Reason |
|---|---|
| Video generation itself | `BaseT2VModel` is reserved but unimplemented; videos come from outside |
| Rewriting official metric formulas | adapters call upstream packages; we own IO / scheduling / aggregation only |
| Cross-machine distribution | the dispatcher is single-host; the interface is reserved for future Ray/SSH |
| Evaluating the judges themselves ("who is more accurate") | left to a separate `judge-eval` subproject |
| Auto paper→adapter codegen | unreliable; left to upstream authors |

---

## 2. Users & scenarios

| User | Core need | Interface |
|---|---|---|
| Model developer | run a new T2V model across all benchmarks for comparable numbers | `eval --bench X` ×N + `aggregate` |
| Benchmark integrator | wire their own eval method into the toolchain | manifest YAML / `BaseBenchmark` subclass |
| VLM integrator | plug a private vLLM or a new cloud API | `judges.yaml` / `--judge-endpoint` |
| Paper reproducer | prove a faithful reproduction of published numbers | `--judge paper` |
| Downstream deployer | run in their own cluster with different resources | env tarball + `judges.yaml` + plugin loader |

Three high-frequency flows: **(A)** new model → run every benchmark → aggregate
→ report; **(B)** new eval idea → one manifest → fully in the pipeline; **(C)**
faithful paper reproduction → `--judge paper` at `--profile full`.

---

## 3. System overview

Seven layers, each depending only on the one below; adding a metric/bench/judge
touches L2 (implementation) + L5 (one registry row) and nothing else. The full
diagram and end-to-end call chain live in [Architecture](../wiki/Architecture.md).
The load-bearing idea: **four registries** (`SUPPORTED_BENCHMARKS / _JUDGES /
_METRICS / _AGGREGATORS`) are the single source of truth, lazy-merged from
builtin + user config + plugins.

---

## 4. The four pillars (+ one cross-cut)

| Pillar | Rationale | Deep dive |
|---|---|---|
| **A — Env & one-click install** | a reproducible conda-pack snapshot + `pip install -e .`; the toolkit must run identically across clusters | — |
| **B — Judge switching** | reproduction vs. cost are *both* first-class: every judge-using bench declares `paper_judge` and `default_judge`, never silently swapping models | [Extensibility](EXTENSIBILITY_DESIGN.md) |
| **C — Fast integration + metrics** | a benchmark or a standalone metric should be addable without a fork; lifted metrics are bit-exact with the bench path | [Extensibility](EXTENSIBILITY_DESIGN.md) · [Metrics](VIDEO_METRICS_DESIGN.md) |
| **D — Quick eval & training monitor** | training loops need a fast, calibrated read, distinct from paper-faithful runs | [Quick Eval](QUICK_EVAL_DESIGN.md) |
| **× — Capability tags** | evaluate *by ability* across benchmarks, not just per-benchmark | [Capability Tags](CAPABILITY_TAGS_DESIGN.md) |

B, C, D and the capability cross-cut shipped together in v0.2: they share the
plugin loader and one YAML-schema style, so splitting them would only fragment
the design.

---

## 5. Cross-pillar principles

These run through every pillar and define how the toolkit "feels".

1. **Registry-driven, no plugin DSL.** "Adding something" = one entry in one
   registry. Three lazy-merge sources (builtin dict + user yaml + entry_points),
   no dynamic-discovery magic.
2. **Adapter-first.** Each benchmark / paper-faithful judge is a thin adapter:
   normalize input paths, normalize output to one `RawResult` shape, orchestrate,
   aggregate. **Never rewrite a paper's metric formula.**
3. **80/20 dual track.** Simple cases use a manifest YAML; complex cases subclass
   in Python. Both enter the same registry; the runner can't tell them apart.
4. **Explicit `paper` / `default` / custom judge tiers.** A bench never silently
   downgrades the paper model to a small one.
5. **Workspace as single truth.** All input / output / log / cache for a run live
   under one workspace; raw per-prompt JSON is the resume primitive, the API log
   is the replay primitive.
6. **Minimal trusted surface.** `BaseBenchmark` = 4 methods, `BaseScorer` = 1,
   `BaseAggregator` = 1, manifest ≤ 12 top-level fields. Surface growth needs a
   stated motive.
7. **Reproducibility is a contract, not a goal.** Official leaderboard JSON is
   checked in; CI diffs against it within tolerance.
8. **Plugin-first; fork is the last resort.** Every normal extension path avoids
   editing toolkit source.

---

## 6. Decision snapshot

- Positioning: the **unified entry point** for T2V eval — not generation, not
  distribution, not judge-eval (separate project).
- Pillars **B + C + D + capability tags ship together** (A is in a consolidation
  phase; NPU is deferred).
- **Integration over framework**: standalone metrics + bit-exact lift dual-entry
  + capability tags are the core; scaffolding/validator pushed to v0.3.
- **Mandatory review protocol**: every PR passes the 3-layer gate — see
  [Review Protocol](REVIEW_PROTOCOL.md).
- **Excluded**: PSNR/SSIM/LPIPS/FID-image (not applicable to T2V); NPU (deferred).

---

## 7. Design archive map

| Doc | Covers |
|---|---|
| **PRODUCT_DESIGN** (this) | vision · scope · pillars · cross-pillar principles |
| [VIDEO_METRICS_DESIGN](VIDEO_METRICS_DESIGN.md) | two-tier catalog · dual-entry · bit-exact lift · registry schema |
| [EXTENSIBILITY_DESIGN](EXTENSIBILITY_DESIGN.md) | judge switching + benchmark/metric integration + plugin model |
| [QUICK_EVAL_DESIGN](QUICK_EVAL_DESIGN.md) | eval profiles · subset calibration · training monitor |
| [CAPABILITY_TAGS_DESIGN](CAPABILITY_TAGS_DESIGN.md) | controlled tag vocabulary · resolver · versioning |
| [REVIEW_PROTOCOL](REVIEW_PROTOCOL.md) | the 3-layer quality gate |
| [NPU_ADAPTATION_DESIGN](NPU_ADAPTATION_DESIGN.md) | deferred future-plan stub |

Operational reference is the [wiki](../index.md); these docs explain *why*.
