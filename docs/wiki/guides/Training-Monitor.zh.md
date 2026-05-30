# 指南:训练监控

[← 首页](../../index.md)

跨 checkpoint 跟踪质量趋势——不打扰训练进程。两个问题决定怎么搭:

1. **哪些指标值得用来做监控信号?** 大多数不行。
2. **训练 env 和评测 env 不一样时,怎么衔接?**

---

## 1. 选适合监控的指标

监控指标要 (a) 在小样本下仍有可信号、(b) 跨 checkpoint 平滑变化。VBench 大多数维度满足不了。

| 类别 | 例子 | 为什么 |
|---|---|---|
| 🟢 **推荐做监控** | **fvd**、**vfid**、**kvd**、**clip-score**、**viclip-score**,vbench 的质量轴维度(`subject_consistency`、`background_consistency`、`motion_smoothness`、`imaging_quality`、`aesthetic_quality`) | 连续标量,均值方差按 1/√N 衰减,checkpoint 间平滑。 |
| 🟡 **可选,小 N 噪声大** | `temporal_flickering`、`dynamic_degree` | 偏离散 0–1 分数,只在全量/大 prompt 集下可用。 |
| 🔴 **别用来监控** | `object_class`、`multiple_objects`、`scene`、`color`、`spatial_relationship`、`appearance_style`、`temporal_style`、`human_action`、`overall_consistency` | 离散检测式 + 每维 prompt 池本来就小 + 每 prompt 需 ≥5 样本。留给阶段性的全量 milestone 评测。 |

**经验法则**:指标如果是*受 prompt 条件约束*且*离散*(检测器命中/未中、分类器是/否),**别监控** —— 留给 milestone 全量评测。

---

## 2. 跨 env 现实(训练 vs. 评测)

T2V 训练器(Wan、CogVideoX 等)和 `videvalkit` 跑在**不同的 conda env**,
torch / mmcv / vbench 依赖互相冲突。**训练器无法 `import videvalkit`**。
工具包接受这个事实:

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  prompts.jsonl   │    │ <ws>/samples/    │    │ timeline.jsonl   │
│  (固定 prompt    │ →  │ step_N/<id>.mp4  │ →  │ (按 step 的评测  │
│   集,JSON)     │    │ (训练器写)        │    │  分数,JSON)    │
└──────────────────┘    └──────────────────┘    └──────────────────┘
        ↑                       ↑                         ↑
   评测 env(一次性)        训练 env(每 checkpoint)    评测 env
                                                       (消费者)
```

训练器读 JSON、写 mp4。评测消费者读 mp4、写 JSON。**两端不跨 env 互相 import。**

---

## 3. 异步模式 — `videvalkit watch`(推荐)

在单独的 terminal / tmux 里(评测 env)起 watcher。它轮询 checkpoint glob,
为每个新 checkpoint 评分并写入该 run 的 `timeline.jsonl`。

```bash
videvalkit watch \
  --videos-pattern '/runs/r42/checkpoints/step_*/samples' \
  --bench vbench --dimensions subject_consistency \
                 --dimensions motion_smoothness \
                 --dimensions imaging_quality \
  --workspace /runs/r42/eval \
  --gpus 0,1,2,3
```

`--gpus` 跨 N 卡分维度——原本 50 min 串行的全量 vbench 评测,5 卡能压到 ~10 min(~5x)。

训练循环里只管把视频丢到约定路径:

```python
# 训练器(wan22 env,无 videvalkit 依赖)
import json
prompts = json.load(open("/runs/r42/prompts.jsonl"))
for step in train_steps:
    if step % 5000 == 0:
        out_dir = f"/runs/r42/checkpoints/step_{step}/samples"
        for p in prompts:
            generate(p["prompt_en"], out=f"{out_dir}/{p['id']}.mp4")
```

Watcher 自动拾起来评分。训练器可选地 tail `timeline.jsonl`,送到 W&B / TensorBoard。

---

## 4. 同步模式 — subprocess(训练器要拿到分数才继续)

对稀疏 milestone 评测,训练循环需要分数才能记日志:

```python
# 训练器 subprocess 进入评测 env 的 python —— 不共享 imports
import subprocess, json
EVAL_PY = "/root/miniconda3/envs/video-eval/bin/python"

def eval_checkpoint(samples_dir, step):
    r = subprocess.run([
        EVAL_PY, "-m", "videvalkit.cli", "eval",
        "--bench", "vbench",
        "--videos", samples_dir,
        "--workspace", "/runs/r42/eval",
        "--models", f"step_{step}",
        "--dimensions", "subject_consistency",
        "--dimensions", "motion_smoothness",
        "--dimensions", "imaging_quality",
        "--gpus", "0,1,2,3",
    ], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)
```

训练器每次付出 subprocess 启动开销(几秒),换来同步拿结果。

---

## 5. 分布型指标作为主监控信号

`fvd` / `vfid` / `kvd` 衡量与固定参考分布的距离,是**跨 checkpoint 最稳的趋势信号**。
搭配一个固定参考集(`refs register --name <name> --path <dir>`)。

```bash
# watch 内部每 checkpoint 跑——快、无需评审、连续
videvalkit metric run --name fvd \
  --gen-videos /runs/r42/checkpoints/step_5000/samples \
  --refs my-ref --allow-tiny-sample
videvalkit metric run --name vfid --gen-videos ... --refs my-ref
videvalkit metric run --name clip-score --videos ... --prompts ...
```

未放置论文 I3D 权重时 FVD 默认 **S3D-K400** 骨干——一个有效趋势信号。
要论文版数字(milestone 时)就放上 `i3d_torchscript.pt` 并传 `--backbone i3d-k400`。

---

## 6. Python API

`videvalkit.training.monitor` 是显式 API(非框架回调)——训练器**确实能** import videvalkit 时再用:

```python
from videvalkit.training import monitor, MonitorConfig

cfg = MonitorConfig(
    benches=[],                              # bench 维度太噪就跳过
    metrics=["fvd", "vfid", "clip-score"],   # 可信的监控三件套
    workspace="/runs/r42/eval",
)
for step in range(0, 100_000, 5000):
    videos = generate_videos(load_prompts())
    result = monitor.eval(videos, model_name=f"step_{step}", cfg=cfg, step=step)
    tb.add_scalar("eval/fvd", result.summary["fvd"]["score"], step)
```

训练器不能 import videvalkit 时,用上面的 subprocess 模式。

---

## 7. 为什么不用子集?

早期推荐路径是"用一个小 prompt 子集 + `--profile quick`"。对 vbench 这条路是死的:
每个维度**自己的 prompt 池就不一样**,分层抽 44 prompt 后每维只剩 4 个样本——
对离散维度信号量不够,且各维度的子集互不可比。本 wiki 不再推这条路。保留的部分:

- `Subset` 基础设施(find_subset、SubsetSpec)留着,如果哪天有人 ship 一份做过真 Spearman ρ 校准的、bench 专属的子集,仍能用。
- 对 T2V 训练监控,**正确路径是 FVD/VFID + 几个连续的 vbench 质量维度,全量 prompt 集配 `--gpus` 并行评测**。
