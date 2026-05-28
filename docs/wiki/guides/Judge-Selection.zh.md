# 指南:评审选择

[← 首页](../../index.md) · 另见[评审参考](../reference/Judges.md)

---

## 先确定你处在哪条路线

| 你想要…… | 用 |
|---|---|
| 复现论文数字 | `--judge paper` |
| 便宜 / 快速迭代 | `--judge default` |
| 某个特定模型 | `--judge <name>` |
| 你的私有端点 | `judges.yaml` 或 `--judge-endpoint` |
| 完全不用评审 | `--no-judge` |

---

## 复现论文

```bash
videvalkit eval --bench t2vcompbench --profile full --judge paper \
  --videos gen/ --workspace ws/
```

`paper` 解析到每个基准的 `paper_judge`(例如 T2V-CompBench 用 LLaVA-1.6-34B)。
需要论文 VLM 可用;缺失则快速失败。

---

## 使用你自己的端点(持久)

一次性创建 `~/.config/videvalkit/judges.yaml`:

```yaml
judges:
  my-qwen3-vl:
    kind: openai_compatible
    endpoint: http://10.20.30.40:8005/v1
    model: Qwen/Qwen3-VL-32B-Instruct
    provider: Qwen
    api_key_env: null
```

然后:

```bash
videvalkit eval --bench worldjen --judge my-qwen3-vl ...
videvalkit list judges            # 确认它出现了
```

---

## 一次性使用端点(临时)

```bash
videvalkit eval --bench worldjen \
  --judge-endpoint http://10.0.0.5:8003/v1 \
  --judge-model google/gemma-4-31b-it \
  --judge-kind openai_compatible
```

---

## 完全不用评审运行

```bash
videvalkit list benchmarks --no-judge      # vbench, worldscore, physics_iq, v_reasonbench
videvalkit eval --bench vbench --no-judge --videos gen/ --workspace ws/
```

对需要评审的基准使用 `--no-judge` 会报错,并给出无需评审的替代项。

---

## 验证连通性

```bash
videvalkit doctor            # Judges 部分会显示可达性 + 密钥是否存在
```
