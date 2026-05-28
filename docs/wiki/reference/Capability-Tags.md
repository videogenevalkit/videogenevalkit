# Capability Tags Reference

[← Home](../Home.md)

A fixed **44-tag** controlled vocabulary (10 top-level + 34 sub) tags every
metric and benchmark-dimension, enabling evaluation **by ability**.

```bash
videvalkit capabilities list [--show-sub]
videvalkit capabilities show motion
videvalkit capabilities eval motion --videos gen/
```

---

## The 10 top-level tags

| Top-level | Sub-tags | Measures |
|---|---|---|
| **motion** | smoothness · magnitude · accuracy · naturalness | how things move |
| **visual_quality** | aesthetic · imaging · artifact_free · sharpness | frame-level quality |
| **text_alignment** | text2video · prompt_following · action_verb | follows the prompt |
| **object_fidelity** | presence · count · attribute · binding | objects right |
| **subject_consistency** | identity · appearance · character | same subject over time |
| **physical_plausibility** | gravity · causality · anatomy · kinematics | physics/anatomy real |
| **temporal_coherence** | flickering · continuity · scene_consistency | frame-to-frame coherence |
| **realism** | distribution · detection · artifact_rate | overall realism |
| **compositional** | multi_object · spatial · numeracy | multi-object scenes |
| **style** | aesthetic · cg_anime · consistency | artistic style |

Sub-tag canonical form is `<prefix>.<leaf>`, e.g. `motion.smoothness`,
`comp.spatial`, `real.distribution`. Full vocab in
`src/videvalkit/configs/capability_taxonomy.py`.

---

## How resolution works

`capabilities eval <tag>`:

1. **Resolve** — top-level expands to all its sub-tags; collect every metric +
   bench-dim tagged with any of them.
2. **Dedup** — a lifted metric and its origin bench-dim share a canonical source
   → counted once (metric preferred).
3. **Run** — each runnable metric (per-video, judge-free) computes on the videos.
4. **Normalize** — min-max per metric to [0, 1].
5. **Aggregate** — mean (or max/min) across contributors → capability score.

Metrics needing refs / prompts / judge, and shells, are **skipped with a reason**
(`eval --capability` is a quick per-video read; use `metric run` for those).

---

## Rules

- **Controlled vocab only** — free-form tags are rejected at load time.
- **Versioned** — `tag_schema_version = 1`; vocab changes bump the version.
- v0.2: plugins use the existing vocab; custom tags are a v0.4 candidate.

---

## Example

```
$ videvalkit capabilities show motion
motion
  How things move — speed, smoothness, accuracy, naturalness
  expands to: [motion, motion.smoothness, motion.magnitude, motion.accuracy, motion.naturalness]

  source_kind   name                          tags
  ----------------------------------------------------------------
  bench_dim     vbench/motion_smoothness      motion.smoothness
  bench_dim     vbench/dynamic_degree         motion.magnitude
  bench_dim     worldscore/motion_magnitude   motion.magnitude
  ...
```

See [Metrics](Metrics.md) for each metric's tags.
