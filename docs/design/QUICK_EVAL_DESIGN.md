# Quick Eval & Training Monitor — Design

> Rationale for fast, calibrated evaluation. The how-to is
> [Profiles & Quick Eval](../wiki/guides/Profiles-and-Quick-Eval.md) and
> [Training Monitor](../wiki/guides/Training-Monitor.md).

---

## 1. Problem

Paper-faithful runs take hours — far too slow to call every N training steps. But
a fast eval is only useful if its ranking *agrees* with the full eval; an
uncalibrated subset can mislead. So the design needs two things: a speed knob, and
evidence that the fast knob preserves ranking.

## 2. Goals / non-goals

- **Goal**: a profile that runs in minutes and tracks the full-eval trend; a way to
  prove the agreement; a training-loop API that doesn't shell out or block.
- **Non-goal**: replacing paper-faithful eval, or inventing new metrics. Profiles
  are orthogonal to *what* (bench/metric) and *which judge*.

## 3. Eval profiles

Three named points on the accuracy↔speed curve, controlling prompt subset, frame
sampling, and samples-per-prompt:

| Profile | Subset | Frames | Samples | Target |
|---|---|---|---|---|
| `quick` | small | 4 | 1 | training monitor / smoke / CI; ρ ≥ 0.85 vs full |
| `standard` | medium | 8 | 1 | ablation / iteration; ρ ≥ 0.95 |
| `full` | full | 8 | 5 | paper / leaderboard (default) |

No `--profile` = `full`, so existing behavior is unchanged.

## 4. Subset calibration

A subset is a **version-pinned JSON** of `prompt_id`s plus provenance (SHA-256) and
a calibration record. The methodology:

1. Pull the full-corpus leaderboard scores.
2. Propose a candidate subset (stratified by dimension).
3. Compute Spearman ρ between subset ranking and full ranking across models.
4. Freeze the subset only if ρ ≥ the profile's threshold (0.85 for quick).

Pinning + provenance means a `quick` number is reproducible and its agreement with
`full` is a recorded fact, not a hope. `--subset <file>` overrides a profile's
default subset.

## 5. Planning & multi-bench

- `estimate` previews wallclock / GPU-hours / judge-call count (and token/$ when a
  judge is set) *before* a run — so cost is visible up front.
- `eval-suite` runs several benchmarks (or `--all-anchored`) into one workspace,
  each with its own profile/judge if needed.

## 6. Training-loop integration

`videvalkit.training.monitor` is an explicit API, not a framework callback:

- `monitor.preview_prompts(cfg)` → the prompt set the configured benches+profile
  need (generate exactly these).
- `monitor.eval(videos, model_name, cfg, step)` → run benches/metrics, append one
  line to `timeline.jsonl`, return a `MonitorResult`.

`MonitorConfig` carries a `metrics=` field so the standalone metrics layer plugs
straight in. The CLI `watch` polls a checkpoint glob and evaluates each new
checkpoint. Cross-bench `overall` is a raw mean of per-bench overalls (a trend
signal; z-score normalization is a later refinement). FVD in monitoring uses the
S3D backbone by default — a valid trend metric without the paper I3D weight.
