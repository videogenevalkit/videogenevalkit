# Getting Started

[← Home](../index.md)

---

## 1. Install

The toolkit runs on Linux + CUDA. The reproducible environment is a `conda-pack`
snapshot; the toolkit code installs on top.

```bash
git clone https://github.com/videogenevalkit/videogenevalkit.git
cd videogenevalkit

# activate the shared env (or clone it):
conda activate /path/to/videvalkit-env      # python 3.10 + torch 2.3.1+cu121

pip install --no-deps -e .
videvalkit doctor                            # verify
```

> macOS / no-GPU: you can `videvalkit list`, `metric list`, `capabilities list`,
> and develop, but benchmark/metric *execution* needs Linux + CUDA.

---

## 2. Stage your videos

```
videos/
└── MyModel/
    ├── prompt0001-0.mp4
    ├── prompt0002-0.mp4
    └── ...
```

One subdirectory per model. Filenames follow `{prompt_id}-{sample}.mp4`.

---

## 3. Pick an entry point

### Run a benchmark

```bash
videvalkit eval --bench vbench \
  --videos videos/ --workspace ws/ \
  --models MyModel --profile quick
# → ws/results/summary/vbench/MyModel.json
```

### Run a single metric

```bash
# distribution metric (needs a reference set)
videvalkit metric run --name fvd \
  --gen-videos videos/MyModel/ --refs ucf101-fvd --allow-tiny-sample

# per-video metric (no reference, no judge)
videvalkit metric run --name motion-smoothness --videos videos/MyModel/
```

### Evaluate a capability

```bash
videvalkit capabilities eval visual_quality --videos videos/MyModel/
```

---

## 4. No judge endpoint? No problem

Four benchmarks and 17 metrics need **no** VLM/LLM judge:

```bash
videvalkit list benchmarks --no-judge
videvalkit list judges            # see what judges exist if you do want one
videvalkit eval --bench vbench --no-judge --videos videos/ --workspace ws/
```

---

## 5. Next steps

| Goal | Page |
|---|---|
| Understand the mental model | [Core Concepts](Concepts.md) |
| Switch / configure the VLM judge | [Judge Selection](guides/Judge-Selection.md) |
| Speed up for training loops | [Profiles & Quick Eval](guides/Profiles-and-Quick-Eval.md) · [Training Monitor](guides/Training-Monitor.md) |
| See every command | [CLI Reference](reference/CLI.md) |
| Add your own metric/bench | [Extending](guides/Extending.md) |
