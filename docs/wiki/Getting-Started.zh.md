# 快速上手

[← 首页](../index.md)

---

## 1. 安装

工具包运行于 Linux + CUDA。可复现环境是一个 `conda-pack` 快照;工具包代码安装在其之上。

```bash
git clone https://github.com/videogenevalkit/videogenevalkit.git
cd videogenevalkit

# 激活共享环境(或克隆它):
conda activate /path/to/videvalkit-env      # python 3.10 + torch 2.3.1+cu121

pip install --no-deps -e .
videvalkit doctor                            # 验证
```

> macOS / 无 GPU:你可以执行 `videvalkit list`、`metric list`、`capabilities list`
> 并进行开发,但基准 / 指标的*实际执行*需要 Linux + CUDA。

---

## 2. 准备你的视频

```
videos/
└── MyModel/
    ├── prompt0001-0.mp4
    ├── prompt0002-0.mp4
    └── ...
```

每个模型一个子目录。文件名遵循 `{prompt_id}-{sample}.mp4`。

---

## 3. 选择一个入口

### 跑一个基准

```bash
videvalkit eval --bench vbench \
  --videos videos/ --workspace ws/ \
  --models MyModel --profile quick
# → ws/results/summary/vbench/MyModel.json
```

### 跑单个指标

```bash
# 分布型指标(需要参考集)
videvalkit metric run --name fvd \
  --gen-videos videos/MyModel/ --refs ucf101-fvd --allow-tiny-sample

# 逐视频指标(无参考、无评审)
videvalkit metric run --name motion-smoothness --videos videos/MyModel/
```

### 评测一个能力

```bash
videvalkit capabilities eval visual_quality --videos videos/MyModel/
```

---

## 4. 没有评审端点?没问题

四个基准和 17 个指标**无需** VLM/LLM 评审:

```bash
videvalkit list benchmarks --no-judge
videvalkit list judges            # 如果你确实想用评审,看看有哪些
videvalkit eval --bench vbench --no-judge --videos videos/ --workspace ws/
```

---

## 5. 下一步

| 目标 | 页面 |
|---|---|
| 理解心智模型 | [核心概念](Concepts.md) |
| 切换 / 配置 VLM 评审 | [评审选择](guides/Judge-Selection.md) |
| 为训练循环提速 | [配置档与快速评测](guides/Profiles-and-Quick-Eval.md) · [训练监控](guides/Training-Monitor.md) |
| 查看每条命令 | [命令行参考](reference/CLI.md) |
| 添加你自己的指标/基准 | [扩展](guides/Extending.md) |
