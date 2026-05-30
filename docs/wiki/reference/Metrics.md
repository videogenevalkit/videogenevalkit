# Metrics Reference

[← Home](../../index.md)

20 planned standalone metrics in two tiers. **16 run judge-free today**, plus
`artifact-diagnostic` (registered, runs with a `--judge`). The remaining 3
specialized dims are available via their bench or deferred — see Tier 2.
Run any with `videvalkit metric run --name <name>`.

> **Functional** = works today (backbone auto-downloads, is in-env, or wraps a
> staged upstream). Status as of v0.2-dev.

---

## Tier 1 — General T2V quality (14)

### Standard input format

T2V monitoring uses **5 s × 24 fps = 120 frames** as the assumed video length.
Each backbone downsamples internally to its required clip length (S3D 16,
I3D 16, VideoMAE-v2 16, …). Other lengths work but a warning is logged.

### Prompt-aligned distribution metrics (v0.2+)

`fvd` / `vfid` / `kvd` / `clip-fvd` accept an optional `--prompts <file>`
flag. When given, the metric enforces that **gen and ref videos come from
the same prompt set** (filename = `<prompt_id>-<sample>.mp4`). Without this
guard, comparing model output to an unrelated reference (UCF101 etc.) mixes
"model quality" with "prompt-domain shift" — the resulting FVD is
uninterpretable.

```bash
videvalkit metric run --name fvd \
  --gen-videos runs/r42/step_5000/samples \
  --ref-videos baselines/wan5b_ref/samples \
  --prompts prompts.jsonl   # the prompts both sets were generated from
```

`--allow-partial-prompts`: use the intersection of prompt_ids present in
both sets instead of erroring on any mismatch.

### Distribution-level (4) — need a reference set, judge-free

| Metric | Backbone | Status | Tags |
|---|---|---|---|
| `fvd` | S3D-K400 (auto) / I3D-K400 (paper) | ✅ functional (s3d default) | realism.distribution |
| `vfid` | InceptionV3 (auto) | ✅ functional | realism.distribution |
| `kvd` | S3D-K400 + poly-MMD² | ✅ functional | realism.distribution |
| `clip-fvd` | CLIP-ViT-L/14 | ✅ functional (experimental) | realism.distribution |

> FVD/KVD default to S3D-K400 (Kinetics-400, auto-download) for monitoring. For
> paper-canonical I3D-FVD, place `i3d_torchscript.pt` and pass `--backbone i3d-k400`.
> CLIP-FVD uses a CLIP feature space — **not** comparable to canonical FVD.

### Text-video alignment (2) — need prompts, judge-free

| Metric | Backbone | Status | Tags |
|---|---|---|---|
| `clip-score` | CLIP-ViT-L/14 | ✅ functional | align.text2video |
| `viclip-score` | ViCLIP-L/14 (auto-fetch) | ✅ functional | align.text2video, align.prompt_following |

### Frame perceptual (2) — lift from vbench, judge-free

| Metric | Source | Status | Tags |
|---|---|---|---|
| `aesthetic-quality` | vbench (LAION) | ✅ functional* | vq.aesthetic, style.aesthetic |
| `imaging-quality` | vbench (MUSIQ) | ✅ functional* | vq.imaging, vq.sharpness |

### Temporal (6) — lift from vbench/worldscore, judge-free

| Metric | Source | Status | Tags |
|---|---|---|---|
| `motion-smoothness` | vbench (AMT) | ✅ functional* | motion.smoothness, temp.flickering |
| `temporal-flickering` | vbench | ✅ functional* | temp.flickering, vq.artifact_free |
| `subject-consistency` | vbench (DINO) | ✅ functional* | subj.identity, subj.appearance |
| `background-consistency` | vbench (CLIP) | ✅ functional* | subj.appearance, temp.continuity |
| `dynamic-degree` | vbench (RAFT) | ✅ functional* | motion.magnitude |
| `motion-magnitude` | worldscore (SEA-RAFT) | ✅ functional† | motion.magnitude |

*functional given the vbench checkpoints; wraps upstream `VBench.evaluate(dimension_list=[dim])` → bit-exact with the bench path.
†functional given the worldscore upstream (`$VIDEVALKIT_WORLDSCORE_ROOT` + SEA-RAFT weights); wraps the same `OpticalFlowMetric` call as the bench → bit-exact.

---

## Tier 2 — Specialized dimensions (6)

| Metric | Source | Judge? | Status | Tags |
|---|---|---|---|---|
| `numeracy` | t2vcompbench (GroundingDINO) | no | ✅ functional | comp.numeracy, obj.count |
| `spatial-relationship` | t2vcompbench (GDINO+Depth) | no | ✅ functional | comp.spatial |
| `artifact-diagnostic` | Artifact-Bench port | yes | ✅ registered (needs `--judge`) | vq.artifact_free |
| `object-binding` | t2vcompbench (MLLM) | yes | ↪ bench-only (`eval --bench t2vcompbench`) | obj.binding, obj.presence |
| `motion-accuracy` | worldscore (RAFT+SAM2) | yes | ↪ bench-only (`eval --bench worldscore`) | motion.accuracy, align.action_verb |
| `identity-preservation` | ArcFace (new) | no | ⏳ deferred to i2v phase | subj.identity, subj.character |

---

## Status legend

| | Meaning |
|---|---|
| ✅ functional | runs today (judge-free, or registered + runs with `--judge`) |
| ↪ bench-only | prompt/judge-conditioned dim; run it via its benchmark, not as a standalone metric |
| ⏳ deferred | out of v0.2 scope (revisit in a later phase) |

---

## Excluded from v0.2

PSNR / SSIM / LPIPS / FID-image — these are **reference-based per-frame** metrics
that need a ground-truth video. T2V has no ground truth, so they don't apply
(useful only for I2V / V2V / reconstruction). Not shipped.

See also: [Capability Tags](Capability-Tags.md) · [CLI: metric run](CLI.md#metrics--references)
