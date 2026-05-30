# Guide: Training Monitor

[← Home](../../index.md)

Track quality trends across training checkpoints — without disturbing the
training process. Two questions decide the right setup:

1. **Which metrics make a good monitoring signal?** Most don't.
2. **How does the trainer reach an eval that lives in a different conda env?**

---

## 1. Pick metrics that *can* be monitored

A monitoring metric needs to (a) carry meaningful signal at small samples and
(b) move continuously with training. Most VBench dimensions fail one or both.

| Bucket | Examples | Why |
|---|---|---|
| 🟢 **Recommended for monitoring** | **fvd**, **vfid**, **kvd**, **clip-score**, **viclip-score**, vbench's quality-axis dims (`subject_consistency`, `background_consistency`, `motion_smoothness`, `imaging_quality`, `aesthetic_quality`) | Continuous scalars; mean variance shrinks as 1/√N; smooth across checkpoints. |
| 🟡 **Optional, noisy with small N** | `temporal_flickering`, `dynamic_degree` | Discrete-ish 0–1 scores; usable only with full or large prompt sets. |
| 🔴 **Avoid for monitoring** | `object_class`, `multiple_objects`, `scene`, `color`, `spatial_relationship`, `appearance_style`, `temporal_style`, `human_action`, `overall_consistency` | Discrete detection-style + per-dim prompt subsets are small + need ≥5 samples per prompt. Save for milestone evals on the full corpus. |

**Rule of thumb**: if a metric is *prompt-conditioned* and *discrete* (detector
hit/miss, classifier yes/no), do not monitor it — save it for full-corpus
evaluation at milestones.

---

## 2. Cross-env reality (training vs. eval)

T2V trainers (Wan, CogVideoX, …) and `videvalkit` live in **different conda
envs** with conflicting torch / mmcv / vbench pins. The trainer **cannot
`import videvalkit`** directly. The toolkit accepts this:

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  prompts.jsonl   │    │ <ws>/samples/    │    │ timeline.jsonl   │
│  (fixed prompt   │ →  │ step_N/<id>.mp4  │ →  │ (eval scores by  │
│   set, JSON)     │    │ (trainer writes) │    │  step, JSON)     │
└──────────────────┘    └──────────────────┘    └──────────────────┘
        ↑                       ↑                         ↑
   eval env (once)         training env (every          eval env
                            checkpoint)                 (consumer)
```

The trainer reads JSON, writes mp4s. The eval consumer reads mp4s, writes
JSON. **Nothing imports across env boundaries.**

---

## 3. Async mode — `videvalkit watch` (recommended)

Run a watcher in a separate terminal / tmux (in the eval env). It polls a
checkpoint glob and scores each new checkpoint into a per-run `timeline.jsonl`.

```bash
videvalkit watch \
  --videos-pattern '/runs/r42/checkpoints/step_*/samples' \
  --bench vbench --dimensions subject_consistency \
                 --dimensions motion_smoothness \
                 --dimensions imaging_quality \
  --workspace /runs/r42/eval \
  --gpus 0,1,2,3
```

`--gpus` shards dimensions across N GPUs — a full-corpus vbench eval that
took 50 min sequential drops to ~10 min on 5 GPUs (~5x).

In the training loop, just dump videos to the right path:

```python
# In trainer (wan22 env, no videvalkit dependency)
import json
prompts = json.load(open("/runs/r42/prompts.jsonl"))
for step in train_steps:
    if step % 5000 == 0:
        out_dir = f"/runs/r42/checkpoints/step_{step}/samples"
        for p in prompts:
            generate(p["prompt_en"], out=f"{out_dir}/{p['id']}.mp4")
```

The watcher picks it up and scores. The trainer optionally tails
`timeline.jsonl` for W&B / TensorBoard logging.

---

## 4. Sync mode — subprocess (when the trainer wants the number inline)

For sparser milestones where you want the score back in the training loop:

```python
# Trainer subprocess into the eval env's python — no shared imports
import subprocess, json
EVAL_PY = "/root/miniconda3/envs/video-eval/bin/python"

def eval_checkpoint(samples_dir, step):
    r = subprocess.run([
        EVAL_PY, "-m", "videvalkit.cli", "eval",
        "--bench", "vbench",
        "--videos", samples_dir,
        "--workspace", "/runs/r42/eval",
        "--models", f"step_{step}",
        "--dimensions", "subject_consistency",
        "--dimensions", "motion_smoothness",
        "--dimensions", "imaging_quality",
        "--gpus", "0,1,2,3",
    ], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)
```

Trainer pays a subprocess startup cost (~few seconds) per call but gets the
result back synchronously.

---

## 5. Distribution metrics as the headline training signal

`fvd` / `vfid` / `kvd` measure distance from a fixed reference distribution
and are the most stable trend signal across checkpoints. Pair with a fixed
reference set (`refs register --name <name> --path <dir>`).

```bash
# Per-checkpoint inside watch — fast, judge-free, continuous
videvalkit metric run --name fvd \
  --gen-videos /runs/r42/checkpoints/step_5000/samples \
  --refs my-ref --allow-tiny-sample
videvalkit metric run --name vfid --gen-videos ... --refs my-ref
videvalkit metric run --name clip-score --videos ... --prompts ...
```

FVD defaults to **S3D-K400** when paper I3D weights aren't placed — a valid
trend signal. For paper-canonical numbers at milestones, place
`i3d_torchscript.pt` and pass `--backbone i3d-k400`.

---

## 6. The Python API

`videvalkit.training.monitor` is an explicit API (not a framework callback) —
use it when the trainer DOES live in an env that can import videvalkit:

```python
from videvalkit.training import monitor, MonitorConfig

cfg = MonitorConfig(
    benches=[],                              # skip bench paths if dims are noisy
    metrics=["fvd", "vfid", "clip-score"],   # the trustworthy monitoring trio
    workspace="/runs/r42/eval",
)
for step in range(0, 100_000, 5000):
    videos = generate_videos(load_prompts())
    result = monitor.eval(videos, model_name=f"step_{step}", cfg=cfg, step=step)
    tb.add_scalar("eval/fvd", result.summary["fvd"]["score"], step)
```

Use the subprocess pattern above when the trainer cannot import videvalkit.

---

## 7. Why not subsets?

Earlier the recommended path was "`--profile quick` with a small prompt
subset". This turned out to be a dead end for vbench: each dim has a
**different prompt pool**, so a stratified 44-prompt subset gives every dim
only 4 samples — too few for discrete dims, and the per-dim subsets aren't
comparable to each other. The wiki used to push this; it doesn't anymore.
What survives:

- The `Subset` infrastructure (find_subset, SubsetSpec) remains usable for
  bench-specific calibrated subsets if anyone ships one with a real Spearman ρ
  measurement.
- For T2V training monitoring, **the right path is FVD/VFID + a couple of
  continuous vbench quality dims on the full prompt set with `--gpus` parallel
  eval**.
