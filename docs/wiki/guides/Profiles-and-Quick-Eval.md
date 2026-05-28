# Guide: Profiles & Quick Eval

[← Home](../Home.md)

Profiles trade accuracy for speed. They control the prompt subset, frame
sampling, and samples-per-prompt — orthogonal to which judge you use.

---

## The three profiles

| Profile | Subset | Frames | Samples | Wallclock | Use |
|---|---|---|---|---|---|
| `quick` | small | 4 | 1 | ~5–10 min | training monitor, smoke, CI |
| `standard` | medium | 8 | 1 | ~30–60 min | ablation, iteration |
| `full` | full | 8 | 5 | hours | paper / leaderboard (default) |

```bash
videvalkit eval --bench vbench --profile quick --videos gen/ --workspace ws/
```

No `--profile` = `full` (back-compatible).

---

## Estimate cost first

```bash
videvalkit estimate --bench vbench --bench worldjen --profile quick --judge gpt-4o
```

```
Benchmark         Judge?      Wallclock     GPU-h   Judge calls
----------------------------------------------------------------
vbench            —               6.0 min     0.10             0
worldjen          VLM             8.0 min     0.05            60
----------------------------------------------------------------
TOTAL                            14.0 min     0.15            60
```

---

## Run several benchmarks at once

```bash
videvalkit eval-suite --all-anchored --profile quick \
  --videos gen/ --workspace ws/

# or pick + skip judge benches
videvalkit eval-suite --bench vbench --bench worldjen --no-judge \
  --videos gen/ --workspace ws/
```

---

## Custom subsets

```bash
videvalkit eval --bench vbench --subset my_subset.json --videos gen/ --workspace ws/
```

A subset is a version-pinned JSON of prompt_ids with calibration metadata
(Spearman ρ vs full). `--subset` overrides the profile's default subset.

---

## Paper-faithful = full + paper judge

```bash
videvalkit eval --bench t2vcompbench --profile full --judge paper ...
```

For training-time monitoring instead, see [Training Monitor](Training-Monitor.md).
