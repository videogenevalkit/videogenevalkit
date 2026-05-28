# Guide: Training Monitor

[← Home](../../index.md)

Run a quick eval every N training steps to track quality trends — without
shelling out or blocking the loop.

---

## CLI: watch a checkpoint directory

```bash
videvalkit watch \
  --videos-pattern '/runs/r42/checkpoints/step_*/samples' \
  --bench vbench --profile quick \
  --workspace /runs/r42/eval
```

Polls the glob, evaluates each new checkpoint's videos as it appears, and
appends one line per checkpoint to `<workspace>/timeline.jsonl`. `--once`
processes current matches and exits (no polling).

---

## Python API: inside the training loop

```python
from videvalkit.training import monitor, MonitorConfig

cfg = MonitorConfig(
    benches=["vbench"],
    metrics=["fvd", "motion-smoothness"],
    profile="quick",
    workspace="/runs/r42/eval",
)

for step in range(0, 100_000, 1000):
    train_step()
    if step % 5000 == 0:
        prompts = monitor.preview_prompts(cfg)        # which prompts to generate
        videos  = generate_videos(prompts)            # your model
        result  = monitor.eval(videos, model_name=f"step_{step}", cfg=cfg, step=step)
        tb.add_scalar("eval/overall", result.overall, step)
        for bench, summary in result.summary.items():
            ...                                        # per-bench detail
```

| Method | Purpose |
|---|---|
| `monitor.preview_prompts(cfg)` | the prompt set the configured benches+profile need |
| `monitor.eval(videos, model_name, cfg)` | run benches/metrics, append to timeline, return `MonitorResult` |
| `MonitorConfig.save / load` | persist the monitor config alongside the run |

---

## What gets recorded

`<workspace>/timeline.jsonl` — one JSON line per checkpoint:

```json
{"model_name": "step_5000", "step": 5000, "profile": "quick",
 "overall": 0.71, "bench_overalls": {"vbench": {...}}}
```

Cross-bench `overall` is the mean of per-bench overalls (raw mean for trend
monitoring; z-score normalization is a later refinement).

---

## FVD in monitoring

FVD defaults to the **S3D-K400** backbone (auto-downloads) when paper I3D
weights aren't placed — a valid Kinetics-400 trend metric. Perfect for
monitoring; for paper numbers use `--backbone i3d-k400` with weights. See
[Metrics](../reference/Metrics.md).

---

## Notes

- No PyTorch/Lightning callback — you call `monitor.eval()` explicitly.
- Use `--profile quick` for stable, fast trend reads.
- Watch uses polling (no inotify); pick an interval matching your checkpoint cadence.
