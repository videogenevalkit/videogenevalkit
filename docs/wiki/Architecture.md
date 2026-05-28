# Architecture

[← Home](../index.md)

---

## Layered design

```
┌────────────────────────────────────────────────────────────────────┐
│ L7  CLI / Python API                                                 │
│     eval · eval-suite · metric · capabilities · refs · estimate ·    │
│     watch · doctor · list                                            │
├────────────────────────────────────────────────────────────────────┤
│ L6  Orchestration                                                    │
│     runner.run · resolve_judge · resolve_profile · plugin.discover · │
│     scheduler (env / GPU / HTTP)                                     │
├────────────────────────────────────────────────────────────────────┤
│ L5  Registries (lazy-merged: builtin + user + plugins)              │
│     SUPPORTED_BENCHMARKS · SUPPORTED_JUDGES · SUPPORTED_METRICS ·    │
│     SUPPORTED_AGGREGATORS · capability taxonomy                      │
├────────────────────────────────────────────────────────────────────┤
│ L4  Plugin discovery                                                 │
│     builtin → pip entry_points → ~/.videvalkit/ → $CWD/.videvalkit/  │
├────────────────────────────────────────────────────────────────────┤
│ L3  Core abstractions                                                │
│     BaseBenchmark · BaseScorer · BaseDistributionMetric ·            │
│     BaseAggregator · ManifestBenchmark · Profile · Subset · Capability│
├────────────────────────────────────────────────────────────────────┤
│ L2  Adapters & metrics                                               │
│     10 benchmark adapters · 20 metrics · 3 judge backends ·          │
│     shared backbones (S3D / I3D / InceptionV3 / CLIP)                │
├────────────────────────────────────────────────────────────────────┤
│ L1  Infrastructure                                                   │
│     Workspace · ApiCallLogger · FrameCache · frechet/mmd utils       │
├────────────────────────────────────────────────────────────────────┤
│ L0  External                                                         │
│     upstream paper repos · HF (checkpoints / refs) · VLM endpoints   │
└────────────────────────────────────────────────────────────────────┘
```

Each layer depends only on the one below. Adding a metric/bench/judge touches
L2 (implementation) + L5 (one registry row) — nothing else.

---

## The four registries

| Registry | Holds | Merge sources |
|---|---|---|
| `SUPPORTED_BENCHMARKS` | bench adapters + dim_tags + judge slots | builtin + plugins |
| `SUPPORTED_JUDGES` | judge configs | builtin + `judges.yaml` |
| `SUPPORTED_METRICS` | metric specs (kind/source/tags/backbone) | builtin + plugins |
| `SUPPORTED_AGGREGATORS` | cross-prompt aggregators | builtin |

All are **lazy-merged at import**: built-in entries plus anything discovered
from user config / plugins. Same-name → later source wins (logged at INFO).

---

## Plugin model

Three layers, lowest precedence first:

1. **Built-in** — `src/videvalkit/{benchmarks,metrics}/`
2. **pip entry_points** — `[project.entry-points."videvalkit.benchmarks"]`
3. **Local dirs** — `~/.videvalkit/<group>/` then `$CWD/.videvalkit/<group>/`

Local plugins use the `__videvalkit_register__()` convention. Disable all
third-party sources with `VIDEVALKIT_DISABLE_PLUGINS=1`.

---

## Two benchmark integration tracks

| Track | When | How |
|---|---|---|
| **A — Manifest** | simple: prompt → scorer → score | one `manifest.yaml` ([Extending](guides/Extending.md)) |
| **B — Python adapter** | complex: staging / multi-stage / subprocess | `BaseBenchmark` subclass |

Both converge on the same runner / workspace / scheduler.

---

## Shared metric infrastructure

| Module | Purpose |
|---|---|
| `metrics/utils/frechet.py` | Fréchet distance (FVD / VFID / CLIP-FVD), float64 |
| `metrics/utils/mmd.py` | polynomial-kernel MMD² (KVD) |
| `metrics/backbones/s3d_k400.py` | S3D Kinetics-400 video features |
| `metrics/backbones/i3d_k400.py` | I3D-K400 torchscript loader (paper FVD) |
| `metrics/backbones/clip_vit.py` | CLIP-ViT frame features |

---

## Design archives

The original design documents (rationale, trade-offs, decision snapshots) live in
[`docs/design/`](../design/PRODUCT_DESIGN.md): PRODUCT, JUDGE_SELECTION, INTEGRATION_FRAMEWORK,
VIDEO_METRICS, QUICK_EVAL, CAPABILITY_TAGS, REVIEW_PROTOCOL, NPU_ADAPTATION.
The wiki is the operational reference; the design docs explain *why*.
