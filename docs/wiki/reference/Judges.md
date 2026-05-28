# Judges Reference

[← Home](../Home.md)

A *judge* is a VLM/LLM that scores semantic dimensions. Judges are pluggable
registry entries; the same endpoint can serve multiple benchmarks.

---

## Built-in judges (8)

| Name | Kind | Model | Auth |
|---|---|---|---|
| `gemma-4-31b-local` | openai_compatible | google/gemma-4-31b-it | local :8003 |
| `qwen3-32b-local` | openai_compatible | Qwen/Qwen3-32B | local :8004 |
| `qwen3-vl-32b-local` | openai_compatible | Qwen/Qwen3-VL-32B-Instruct | local :8005 |
| `local-llava-video-7b` | openai_compatible | lmms-lab/LLaVA-Video-7B-Qwen2 | local :8006 |
| `gemini-3-flash` | gemini | gemini-3-flash-preview | GEMINI_API_KEY |
| `gemini-2.5-pro` | gemini | gemini-2.5-pro | GEMINI_API_KEY |
| `claude-sonnet-4-6` | anthropic | claude-sonnet-4-6 | ANTHROPIC_API_KEY |
| `gpt-4o` | openai_compatible | gpt-4o-2024-11-20 | OPENAI_API_KEY |

Plus the paper alias `paper-llava-1.6-34b` (T2V-CompBench paper VLM).

---

## Three ways to pick a judge

```bash
videvalkit eval --bench worldjen --judge default              # bench's default_judge
videvalkit eval --bench t2vcompbench --judge paper            # bench's paper_judge
videvalkit eval --bench worldjen --judge claude-sonnet-4-6    # any registry name
```

`paper` / `default` are semantic keywords resolved per benchmark. `--judge paper`
on a judge-free benchmark fails fast.

---

## Add your own (no fork)

`~/.config/videvalkit/judges.yaml`:

```yaml
judges:
  my-cluster-qwen3-vl:
    kind: openai_compatible
    endpoint: http://10.20.30.40:8005/v1
    model: Qwen/Qwen3-VL-32B-Instruct
    provider: Qwen
    api_key_env: null
  doubao-pro:
    kind: openai_compatible
    endpoint: https://ark.cn-beijing.volces.com/api/v3
    model: doubao-pro-32k
    provider: bytedance
    api_key_env: ARK_API_KEY
```

Same-name entries override built-ins. Disable user yaml with
`VIDEVALKIT_JUDGE_USER_YAML=0`.

---

## Ad-hoc (no registration)

```bash
videvalkit eval --bench worldjen \
  --judge-endpoint http://10.0.0.5:8003/v1 \
  --judge-model google/gemma-4-31b-it \
  --judge-kind openai_compatible \
  --judge-api-key-env MY_KEY
```

Mutually exclusive with `--judge`.

---

## No judge at all

```bash
videvalkit list benchmarks --no-judge     # 4 judge-free benches
videvalkit list metrics --no-judge        # 17 judge-free metrics
videvalkit eval --bench vbench --no-judge
```

---

## Config precedence

```
builtin → ~/.config/videvalkit/judges.yaml → $CWD/.videvalkit/judges.yaml → env → CLI
```

Later sources override earlier (top-level key replacement, no deep merge).

See [Judge Selection guide](../guides/Judge-Selection.md) for the full workflow.
