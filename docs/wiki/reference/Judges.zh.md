# 评审参考

[← 首页](../../index.md)

*评审(judge)*是给语义维度打分的 VLM/LLM。评审是可插拔的注册表条目;
同一个端点可服务多个基准。

---

## 内置评审(8)

| 名称 | Kind | 模型 | 认证 |
|---|---|---|---|
| `gemma-4-31b-local` | openai_compatible | google/gemma-4-31b-it | 本地 :8003 |
| `qwen3-32b-local` | openai_compatible | Qwen/Qwen3-32B | 本地 :8004 |
| `qwen3-vl-32b-local` | openai_compatible | Qwen/Qwen3-VL-32B-Instruct | 本地 :8005 |
| `local-llava-video-7b` | openai_compatible | lmms-lab/LLaVA-Video-7B-Qwen2 | 本地 :8006 |
| `gemini-3-flash` | gemini | gemini-3-flash-preview | GEMINI_API_KEY |
| `gemini-2.5-pro` | gemini | gemini-2.5-pro | GEMINI_API_KEY |
| `claude-sonnet-4-6` | anthropic | claude-sonnet-4-6 | ANTHROPIC_API_KEY |
| `gpt-4o` | openai_compatible | gpt-4o-2024-11-20 | OPENAI_API_KEY |

外加论文别名 `paper-llava-1.6-34b`(T2V-CompBench 论文 VLM)。

---

## 选择评审的三种方式

```bash
videvalkit eval --bench worldjen --judge default              # 基准的 default_judge
videvalkit eval --bench t2vcompbench --judge paper            # 基准的 paper_judge
videvalkit eval --bench worldjen --judge claude-sonnet-4-6    # 任意注册表名
```

`paper` / `default` 是按基准解析的语义关键字。对无需评审的基准使用
`--judge paper` 会快速失败。

---

## 添加你自己的(无需 fork)

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

同名条目覆盖内置。用 `VIDEVALKIT_JUDGE_USER_YAML=0` 禁用用户 yaml。

---

## 临时(无需注册)

```bash
videvalkit eval --bench worldjen \
  --judge-endpoint http://10.0.0.5:8003/v1 \
  --judge-model google/gemma-4-31b-it \
  --judge-kind openai_compatible \
  --judge-api-key-env MY_KEY
```

与 `--judge` 互斥。

---

## 完全不用评审

```bash
videvalkit list benchmarks --no-judge     # 4 个无需评审的基准
videvalkit list metrics --no-judge        # 17 个无需评审的指标
videvalkit eval --bench vbench --no-judge
```

---

## 配置优先级

```
builtin → ~/.config/videvalkit/judges.yaml → $CWD/.videvalkit/judges.yaml → env → CLI
```

后来的来源覆盖先前的(顶层键替换,无深度合并)。

完整工作流见[评审选择指南](../guides/Judge-Selection.md)。
