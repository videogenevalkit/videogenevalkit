# CLI Reference

[← Home](../Home.md)

All commands are subcommands of `videvalkit`. Run `videvalkit <cmd> --help` for
full flags.

---

## Evaluation

| Command | Purpose |
|---|---|
| `eval --bench X --videos V --workspace W` | Run one benchmark |
| `eval-suite --bench A --bench B ... ` / `--all-anchored` | Run several benchmarks into one workspace |
| `capabilities eval <tag> --videos V` | Run all metrics for a capability, aggregate |
| `metric run --name M ...` | Run a single standalone metric |
| `aggregate --workspace W` | Cross-benchmark z-score report |

### `eval` key flags

| Flag | Meaning |
|---|---|
| `--bench` | benchmark name (required) |
| `--videos` / `--workspace` | input dir / output dir (required) |
| `--models` | model names (repeat) |
| `--dimensions` | subset of dims (repeat) |
| `--judge` | `paper` / `default` / `<registry name>` |
| `--judge-endpoint / --judge-model / --judge-kind / --judge-api-key-env` | ad-hoc judge |
| `--no-judge` | refuse benches that need a judge |
| `--profile` | `quick` / `standard` / `full` |
| `--subset` | path to a subset JSON |
| `--aggregator` | override default aggregator |

---

## Metrics & references

| Command | Purpose |
|---|---|
| `metric list [--kind --no-judge --source]` | List metrics, filterable |
| `metric show <name>` | Show a metric's registry entry |
| `metric run --name M --gen-videos / --videos / --ref-videos / --refs / --prompts` | Run a metric |
| `refs list / show <name> / register --name --path` | Manage reference video sets |

### `metric run` inputs by kind

| Metric kind | Required inputs |
|---|---|
| `distribution_reference` | `--gen-videos` + (`--ref-videos` or `--refs`) |
| `per_prompt_reference_free` | `--videos` + `--prompts` |
| `per_video_reference_free` | `--videos` |

`--allow-tiny-sample` bypasses the small-N guard for distribution metrics.

---

## Capabilities

| Command | Purpose |
|---|---|
| `capabilities list [--show-sub]` | List 10 top-level (and 34 sub) tags + contributor counts |
| `capabilities show <tag>` | Show contributors for a tag |
| `capabilities eval <tag> --videos V [--aggregator mean/max/min]` | Cross-metric capability score |

---

## Planning & monitoring

| Command | Purpose |
|---|---|
| `estimate --bench A --bench B --profile P` | Preview wallclock / GPU-h / judge calls |
| `watch --videos-pattern '...' --bench X --profile quick [--once]` | Poll a checkpoint dir, eval each new model |

Python training-loop API: `videvalkit.training.monitor` — see [Training Monitor](../guides/Training-Monitor.md).

---

## Inspection & setup

| Command | Purpose |
|---|---|
| `doctor [--json]` | Health: devices, benches, metrics, profiles, capability coverage, plugins, judges |
| `list benchmarks [--no-judge]` / `list judges` / `list aggregators` | List registries |
| `fetch-smoke-data` / `fetch-checkpoints` / `fetch-upstream` | Pull data / weights / upstream repos |
| `prepare-workspace --workspace W --videos V` | Bootstrap a workspace |
