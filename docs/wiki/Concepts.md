# Core Concepts

[← Home](../index.md)

The mental model in five ideas. Everything else is detail.

---

## 1. Three orthogonal axes

videogenevalkit separates *what you evaluate* from *how* and *with which judge*.

```
            WHAT                  HOW (cost)           WHICH judge
    ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
    │ --bench   X      │   │ --profile quick  │   │ --judge paper    │
    │ --name    Y      │ × │ --profile standard│ × │ --judge default  │
    │ --capability Z   │   │ --profile full   │   │ --judge <name>   │
    └──────────────────┘   └──────────────────┘   └──────────────────┘
```

These compose freely. `--profile full --judge paper` is the paper-faithful lane;
`--profile quick --judge default` is the training-monitor lane.

---

## 2. Three entry points (the WHAT axis)

| Entry | Granularity | Output | Use when |
|---|---|---|---|
| `eval --bench` | whole benchmark | paper-comparable per-dim scores | reporting against a published leaderboard |
| `metric run --name` | one metric | a single scalar | you want just FVD / CLIP-Score / etc. |
| `capabilities eval` | one ability | cross-metric aggregate | "how good is *motion* overall?" |

A metric that's lifted out of a benchmark (e.g. `motion-smoothness`) is
**bit-exact** whether called via `--bench vbench --dimensions motion_smoothness`
or `metric run --name motion-smoothness` — they share one implementation.

---

## 3. Profiles (the HOW axis)

| Profile | Subset | Frames | Wallclock | Use |
|---|---|---|---|---|
| `quick` | small | 4 | ~5–10 min | training monitoring, smoke, CI |
| `standard` | medium | 8 | ~30–60 min | ablation, iteration |
| `full` | full corpus | 8 | hours | paper / leaderboard (default) |

`videvalkit estimate` previews cost before you run.

---

## 4. Judges (the WHICH axis)

Every judge-using benchmark declares two slots:

| Slot | Meaning |
|---|---|
| `paper_judge` | the VLM the paper used (faithful reproduction) |
| `default_judge` | a cheaper / validated stand-in |

Resolve with `--judge paper` / `--judge default` / `--judge <registry-name>`,
add your own in `~/.config/videvalkit/judges.yaml`, or go ad-hoc with
`--judge-endpoint`. Don't have one? `--no-judge` filters to judge-free work.
See [Judge Selection](guides/Judge-Selection.md).

---

## 5. Capability tags

Every metric and benchmark-dimension carries tags from a fixed 44-tag vocabulary
(10 top-level + 34 sub). This lets you evaluate **by ability** instead of by
benchmark:

```
motion → motion-smoothness, dynamic-degree, motion-magnitude, ...
          (across vbench, worldscore, standalone metrics — deduped)
```

See [Capability Tags](reference/Capability-Tags.md).

---

## How it fits together

```
videvalkit eval --bench worldjen --judge paper --profile full
        │
        ├─ resolve_judge()  → paper_judge → concrete judge config
        ├─ resolve_profile()→ subset + frame sampling
        ├─ plugin discover  → benchmark adapter
        └─ scheduler        → adapter.evaluate() → aggregate() → summary
```

Full layering in [Architecture](Architecture.md).
