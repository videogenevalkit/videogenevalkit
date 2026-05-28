# Guide: Judge Selection

[← Home](../Home.md) · see also [Judges Reference](../reference/Judges.md)

---

## Decide which lane you're in

| You want... | Use |
|---|---|
| Reproduce a paper's numbers | `--judge paper` |
| Cheap / fast iteration | `--judge default` |
| A specific model | `--judge <name>` |
| Your private endpoint | `judges.yaml` or `--judge-endpoint` |
| No judge at all | `--no-judge` |

---

## Reproduce a paper

```bash
videvalkit eval --bench t2vcompbench --profile full --judge paper \
  --videos gen/ --workspace ws/
```

`paper` resolves to each benchmark's `paper_judge` (e.g. LLaVA-1.6-34B for
T2V-CompBench). Needs the paper VLM available; fails fast if missing.

---

## Use your own endpoint (persistent)

Create `~/.config/videvalkit/judges.yaml` once:

```yaml
judges:
  my-qwen3-vl:
    kind: openai_compatible
    endpoint: http://10.20.30.40:8005/v1
    model: Qwen/Qwen3-VL-32B-Instruct
    provider: Qwen
    api_key_env: null
```

Then:

```bash
videvalkit eval --bench worldjen --judge my-qwen3-vl ...
videvalkit list judges            # confirm it appears
```

---

## Use an endpoint once (ad-hoc)

```bash
videvalkit eval --bench worldjen \
  --judge-endpoint http://10.0.0.5:8003/v1 \
  --judge-model google/gemma-4-31b-it \
  --judge-kind openai_compatible
```

---

## Run without any judge

```bash
videvalkit list benchmarks --no-judge      # vbench, worldscore, physics_iq, v_reasonbench
videvalkit eval --bench vbench --no-judge --videos gen/ --workspace ws/
```

`--no-judge` on a judge-requiring bench errors with the judge-free alternatives.

---

## Verify connectivity

```bash
videvalkit doctor            # the Judges section shows reachability + key presence
```
