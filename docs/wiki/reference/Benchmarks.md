# Benchmarks Reference

[← Home](../Home.md)

10 registered benchmark adapters. Each wraps upstream code byte-for-byte.
Run with `videvalkit eval --bench <name>`.

---

## Anchored benchmarks (production-ready, 6)

| Benchmark | Dims | Judge? | Default judge | Paper judge | Scores |
|---|---|:-:|---|---|---|
| `vbench` | 16 | ✗ | — | — | quality + semantic, weighted-sum |
| `vbench2` | 18 | ✓ VLM | local-llava-video-7b | local-llava-video-7b | 5 categories incl. Physics |
| `videobench` | 9 | ✓ VLM | gpt-4o | gpt-4o | alignment + dynamic quality |
| `worldjen` | 16 | ✓ VLM | gemma-4-31b-local | gemma-4-31b-local | PHAS 4-category |
| `worldscore` | 10 | ✗ | — | — | SLAM + RAFT + SAM stack |
| `t2vcompbench` | 7 | ✓ VLM | local-llava-video-7b | paper-llava-1.6-34b | compositional |

## Supplementary (4)

| Benchmark | Dims | Judge? | Notes |
|---|---|:-:|---|
| `physics_iq` | — | ✗ | pixel-level CV physics |
| `vbench_pp` | ⊃ vbench | ✓ VLM | I2V + Trustworthiness |
| `v_reasonbench` | — | ✗ | deterministic verifiers |
| `semantics_axis` | — | ✓ VLM | VLM-judge semantic axis |

---

## Judge-free subset

Four benchmarks need **no** VLM/LLM judge — runnable fully offline:

```bash
videvalkit list benchmarks --no-judge
# vbench · worldscore · physics_iq · v_reasonbench
```

---

## Choosing the judge

For judge-using benchmarks:

```bash
videvalkit eval --bench t2vcompbench --judge paper      # LLaVA-1.6-34B (paper)
videvalkit eval --bench t2vcompbench --judge default    # local-llava-video-7b
videvalkit eval --bench t2vcompbench --judge gpt-4o     # any registry name
```

See [Judge Selection](../guides/Judge-Selection.md).

---

## Reproducibility

`--profile full --judge paper` is the paper-faithful lane. Reported mean |Δ| vs
official leaderboards (v0.0.1/v0.1.0 validation): VBench v1 0.012, VBench-2.0
0.0055, T2V-CompBench 0.0046. See `docs/TEST_MANUAL.md` for per-dim tables.

---

## Add your own

Simple benchmarks: a `manifest.yaml` (Track A). Complex: a `BaseBenchmark`
subclass (Track B). See [Extending](../guides/Extending.md).
