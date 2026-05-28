# 基准参考

[← 首页](../../index.md)

10 个已注册的基准适配器。每个都逐字节包裹上游代码。
用 `videvalkit eval --bench <name>` 运行。

---

## 锚定基准(生产就绪,6 个)

| 基准 | 维度 | 评审? | 默认评审 | 论文评审 | 分数 |
|---|---|:-:|---|---|---|
| `vbench` | 16 | ✗ | — | — | 质量 + 语义,加权和 |
| `vbench2` | 18 | ✓ VLM | local-llava-video-7b | local-llava-video-7b | 5 大类含物理 |
| `videobench` | 9 | ✓ VLM | gpt-4o | gpt-4o | 对齐 + 动态质量 |
| `worldjen` | 16 | ✓ VLM | gemma-4-31b-local | gemma-4-31b-local | PHAS 四类 |
| `worldscore` | 10 | ✗ | — | — | SLAM + RAFT + SAM 组合 |
| `t2vcompbench` | 7 | ✓ VLM | local-llava-video-7b | paper-llava-1.6-34b | 组合性 |

## 补充(4 个)

| 基准 | 维度 | 评审? | 备注 |
|---|---|:-:|---|
| `physics_iq` | — | ✗ | 像素级 CV 物理 |
| `vbench_pp` | ⊃ vbench | ✓ VLM | I2V + 可信度 |
| `v_reasonbench` | — | ✗ | 确定性验证器 |
| `semantics_axis` | — | ✓ VLM | VLM 评审的语义轴 |

---

## 无需评审的子集

四个基准**无需** VLM/LLM 评审——可完全离线运行:

```bash
videvalkit list benchmarks --no-judge
# vbench · worldscore · physics_iq · v_reasonbench
```

---

## 选择评审

对使用评审的基准:

```bash
videvalkit eval --bench t2vcompbench --judge paper      # LLaVA-1.6-34B(论文)
videvalkit eval --bench t2vcompbench --judge default    # local-llava-video-7b
videvalkit eval --bench t2vcompbench --judge gpt-4o     # 任意注册表名
```

见[评审选择](../guides/Judge-Selection.md)。

---

## 可复现性

`--profile full --judge paper` 是论文忠实路线。相对官方榜单的已报告平均 |Δ|
(v0.0.1/v0.1.0 验证):VBench v1 0.012、VBench-2.0
0.0055、T2V-CompBench 0.0046。逐维度表见 `docs/TEST_MANUAL.md`。

---

## 添加你自己的

简单基准:一份 `manifest.yaml`(路线 A)。复杂:一个 `BaseBenchmark`
子类(路线 B)。见[扩展](../guides/Extending.md)。
