# Metrics Reference

[← Home](../Home.md)

20 standalone metrics in two tiers. **16 functional, 4 not-yet-registered.**
Run any with `videvalkit metric run --name <name>`.

> **Functional** = works today (backbone auto-downloads, is in-env, or wraps a
> staged upstream). Status as of v0.2-dev.

---

## Tier 1 — General T2V quality (14)

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
| `object-binding` | t2vcompbench (MLLM) | yes | ⛔ not registered (judge) | obj.binding, obj.presence |
| `motion-accuracy` | worldscore (RAFT+GDINO) | no | ⛔ not registered (runner) | motion.accuracy, align.action_verb |
| `identity-preservation` | ArcFace (new) | no | ⛔ not registered (insightface) | subj.identity, subj.character |
| `artifact-diagnostic` | Artifact-Bench port | yes | ⛔ not registered (judge+port) | real.artifact_rate, vq.artifact_free |

---

## Status legend

| | Meaning |
|---|---|
| ✅ functional | runs today |
| ⚪ shell | registered, `NOT YET FUNCTIONAL`, pending weights/runner |
| ⛔ not registered | needs an external prereq (judge endpoint / insightface / runner wiring) |

---

## Excluded from v0.2

PSNR / SSIM / LPIPS / FID-image — these are **reference-based per-frame** metrics
that need a ground-truth video. T2V has no ground truth, so they don't apply
(useful only for I2V / V2V / reconstruction). Not shipped.

See also: [Capability Tags](Capability-Tags.md) · [CLI: metric run](CLI.md#metrics--references)
