# Unified VLM Judge Selection — Design Doc

| 字段 | 内容 |
|---|---|
| 版本 | v0.1 (draft) |
| 状态 | Design — pending review |
| 创建 | 2026-05-20 |
| 作者 | videogenevalkit contributors |
| 影响范围 | `configs/judges.py`, `configs/benchmarks.py`, `cli.py`, `runner.py`, `scorers/vlm_judge/factory.py`, 全部 6 个 anchored adapter |
| 目标读者 | toolkit 核心开发 / benchmark 集成者 / 端到端评测用户 |

---

## 1. 背景与问题陈述

### 1.1 当前状态

`videvalkit` 已经具备 VLM judge 的基础抽象：

- **三种 backend 统一**：`OpenAICompatibleVLMJudge` / `GeminiVLMJudge` / `AnthropicVLMJudge` 共享 `build_judge(cfg)` 工厂，adapter 代码完全 backend-agnostic。
- **注册表机制**：`SUPPORTED_JUDGES` 列了 8 个 entry（4 本地 vLLM + 4 managed API）。
- **CLI 切换**：`videvalkit eval --judge <name>` 可以覆盖每个 benchmark 在 `SUPPORTED_BENCHMARKS[name]["default_judge"]` 里声明的默认 judge。
- **API 日志**：`ApiCallLogger` 已经把每一次 chat call 持久化到 `api_logs/calls/{provider}/{model}/...`，便于回放与审计。

### 1.2 当前的缺口

| 缺口 | 用户痛点 |
|---|---|
| **README 已声明、代码未实现的 `~/.config/videvalkit/judges.yaml`** | 想加自己的 endpoint（私有 vLLM、内网代理、第三方 API）必须改源码 `configs/judges.py` |
| **不能 ad-hoc 传 endpoint** | 临时调试 / 一次性跑某个新 model 也得先注册 |
| **"benchmark paper 原生 judge" 没作为 first-class concept 暴露** | 现在的 `default_judge` 已被改"省事化"——例如 T2V-CompBench paper 用 LLaVA-1.6-34B (68GB) 但 default 是 `local-llava-video-7b`；Video-Bench paper 用 GPT-4o 但 README 自报 validate 时替成了 Gemma。用户既看不到 paper 原生选项，也无法"一键 paper-faithful 复现" |
| **per-benchmark / per-dim judge override 不存在** | 想 vbench2 用 Claude、worldjen 用本地 Gemma、t2vcompbench 用 GPT-4o 这种混合配置，目前只能多次 CLI 调用 |

### 1.3 设计契机

用户原话：「各种 bench 可以提供一个切换和选择不同 vlm api 接口和本地部署的 vlm 来进行，也可以选择 bench 本身默认支持的 vlm」

—— 翻译为产品需求：**任一 benchmark，用户必须能在三档之间一键切换**：

1. **Paper-original**：忠实复现 paper（如 LLaVA-1.6-34B / GPT-4o / paper-Gemma）
2. **Managed API**：用云端 API（Claude / Gemini / GPT-4o / 国产模型）
3. **Self-hosted local**：用自己起的 vLLM/SGLang/Ollama endpoint

并且第二、第三档应该**无需改源码**就能扩展。

---

## 2. 目标 / 非目标

### 2.1 目标 (in scope)

1. **三档显式化**：每个 benchmark 都暴露 `paper / default / <name>` 三档语义，CLI 与 Python 入口对称。
2. **用户配置文件**：`~/.config/videvalkit/judges.yaml` 加载并与内置 registry 合并；优先级 user > built-in。
3. **CLI ad-hoc endpoint**：不注册就能跑：`--judge-endpoint URL --judge-model NAME --judge-kind openai_compatible`。
4. **per-benchmark judge override**：跨多个 benchmark 一次性跑时，每个 bench 用不同 judge：`--judge-for vbench2=claude-sonnet-4-6 --judge-for worldjen=gemma-4-31b-local`（aggregate 流程内复用）。
5. **可发现性**：`videvalkit list judges` / `videvalkit judges test <name>` 列全集 + 端点连通性 ping。
6. **零回归**：现有 `--judge <name>` 用法不变，现有用户脚本无须改动。

### 2.2 非目标 (out of scope)

- ❌ **不做** per-dim judge（同一 benchmark 不同 dim 用不同 judge）—— 价值低，复杂度高，留到 v0.3。
- ❌ **不做** judge 自动 fallback / 自动 retry 链 —— 调度层 (`rate_limit`, `http_pool`) 已经管 retry，judge 选择是用户决策不是系统决策。
- ❌ **不实现新的 backend**（不新增 Anthropic / Gemini 之外的第四种 SDK）—— 所有新 endpoint 必须是 OpenAI-compatible 或 Gemini SDK 或 Anthropic SDK 之一。
- ❌ **不做** judge benchmark/A-B 评估工具（"judge X 与 judge Y 在 dim Z 上谁更准"）—— 留到独立 `videvalkit judge-eval` 子项目。

---

## 3. 三档 Judge 选择模型（核心抽象）

### 3.1 概念定义

每个 benchmark 在注册表里声明 **三个语义槽位**：

```python
SUPPORTED_BENCHMARKS = {
    "t2vcompbench": dict(
        cls=T2VCompBenchBenchmark,
        env=_SHARED_ENV,
        needs_judge=True,
        # ---- 三档 ----
        paper_judge="paper-llava-1.6-34b",      # 复现 paper 的原生 VLM
        default_judge="local-llava-video-7b",   # 省事档（小模型 / 已 validate）
        recommended_judges=[                     # 建议列表，docs 与 CLI help 展示
            "paper-llava-1.6-34b",
            "local-llava-video-7b",
            "gpt-4o",
            "claude-sonnet-4-6",
            "gemini-2.5-pro",
        ],
        # ----
        default_aggregator="weighted_sum",
    ),
    ...
}
```

CLI 解析 `--judge` 的值：

| `--judge` 值 | 解析为 |
|---|---|
| 不传 | `default_judge`（向后兼容） |
| `paper` | `paper_judge`（新） |
| `default` | `default_judge`（显式） |
| `<具体名>`（如 `claude-sonnet-4-6`） | 直接查 `SUPPORTED_JUDGES + user_judges` |
| 配合 `--judge-endpoint / --judge-model / --judge-kind` | ad-hoc，绕过 registry |

### 3.2 每个 benchmark 的三档映射（v0.1 提案）

> 这是基于 paper 原文与 README 自报数据的整理；任何一条都可以在评审时调整。

| Benchmark | `paper_judge` | `default_judge` | 备注 |
|---|---|---|---|
| **vbench** | — (no judge) | — | v1 不用 VLM judge |
| **vbench2** | `paper-vbench2-vlm` (LLaVA-Video-7B + 自训 classifier，按 paper) | `local-llava-video-7b` | paper/default 基本一致；保留两档为接口对称 |
| **videobench** | `paper-gpt-4o` (alias → `gpt-4o`) | `gemma-4-31b-local` | paper 强依赖 GPT-4o；validate 时用 Gemma 替代有偏移，doc 在 TEST_MANUAL §X 标注 |
| **worldjen** | `paper-gemma-it` (alias → `gemma-4-31b-local`) | `gemma-4-31b-local` | paper 即 Gemma；两档等价 |
| **t2vcompbench** | `paper-llava-1.6-34b` (68GB upstream) | `local-llava-video-7b` | paper-faithful 要 70GB GPU 显存 |
| **worldscore** | — (no judge) | — | 纯 CV pipeline |
| **physics_iq** | — | — | 像素级 CV |
| **vbench_pp** | `paper-vbench-pp-vlm` | `local-llava-video-7b` | I2V + Trustworthiness 维度需要 |
| **v_reasonbench** | — | — | 确定性 verifier |

新增 alias entry 写入 `SUPPORTED_JUDGES`（kind=openai_compatible，指向 LLaVA-1.6-34B 的 upstream HF 路径或本地 vLLM 服务）。

---

## 4. 配置合并模型

### 4.1 来源与优先级（从低到高）

```
1. 内置 SUPPORTED_JUDGES (configs/judges.py)
        ↓ 合并
2. user judges.yaml (~/.config/videvalkit/judges.yaml)
        ↓ 合并
3. project judges.yaml ($CWD/.videvalkit/judges.yaml，可选)
        ↓ 覆盖
4. 环境变量覆盖 (VIDEVALKIT_JUDGE_ENDPOINT_<name>, ...)
        ↓ 覆盖
5. CLI flags (--judge / --judge-endpoint / ...)
```

合并语义：**后者覆盖前者，按 entry 名 (top-level key) 合并**。同名 entry 完全替换（不做 field-level deep-merge，避免半套配置混乱）。

### 4.2 `~/.config/videvalkit/judges.yaml` schema

```yaml
# 用户私有 judge endpoints —— 与内置 SUPPORTED_JUDGES 同 schema
judges:
  my-internal-claude:
    kind: anthropic
    model: claude-sonnet-4-6
    provider: anthropic
    api_key_env: MY_INTERNAL_ANTHROPIC_KEY

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

加载实现：`videvalkit.configs.judges.load_user_judges()` —— 单文件、单函数、无 plugin loader。

### 4.3 schema 校验

加载时用 pydantic model 校验：

```python
class JudgeConfig(BaseModel):
    kind: Literal["openai_compatible", "gemini", "anthropic"]
    model: str
    provider: str = "unknown"
    endpoint: str | None = None      # 必填 if kind == openai_compatible
    api_key_env: str | None = None
    sys_prompt: str | None = None    # path
```

非法 entry 不抛 fatal，而是 warn + skip，避免一个错配项让 `videvalkit list` 整体崩。

---

## 5. CLI 接口设计

### 5.1 既有命令（不变）

```bash
videvalkit eval --bench worldjen --videos ... --workspace ... --judge gemma-4-31b-local
videvalkit list judges
videvalkit doctor
```

### 5.2 新增 / 扩展

**`--judge` 接受三种语法**

```bash
# (a) registry name —— 不变
videvalkit eval --bench t2vcompbench --judge gpt-4o ...

# (b) 语义关键字
videvalkit eval --bench t2vcompbench --judge paper ...
videvalkit eval --bench t2vcompbench --judge default ...

# (c) ad-hoc endpoint（绕过 registry）
videvalkit eval --bench worldjen \
  --judge-endpoint http://10.0.0.5:8003/v1 \
  --judge-model google/gemma-4-31b-it \
  --judge-kind openai_compatible \
  --judge-api-key-env MY_KEY
```

**`--judge-for` 多 benchmark 场景**

```bash
# 一次跑多个 benchmark，每个用不同 judge
videvalkit eval-suite \
  --bench vbench2 --bench worldjen --bench videobench \
  --workspace ~/runs/mix \
  --judge-for vbench2=local-llava-video-7b \
  --judge-for worldjen=claude-sonnet-4-6 \
  --judge-for videobench=gpt-4o
```

> 注：`eval-suite` 是新子命令，等价于 for-loop 调 `eval` 但共享一个 workspace；当前 `eval` 单 benchmark 行为不动。

**`judges` 子命令组**

```bash
videvalkit judges list                  # 列内置 + user yaml 合并后的全集
videvalkit judges list --source builtin # 只列内置
videvalkit judges list --source user    # 只列 user
videvalkit judges show <name>           # 展开一个 entry 的完整配置
videvalkit judges test <name>           # 发 1 个 ping 请求测试连通性 + auth
videvalkit judges test --all            # 全部测一遍（用在 CI / doctor）
```

`judges test` 输出格式：

```
gemma-4-31b-local              REACH  ✓  AUTH ✓  LATENCY 142ms
qwen3-32b-local                REACH  ✗  endpoint unreachable
claude-sonnet-4-6              REACH  ✓  AUTH ✓  LATENCY 380ms
my-internal-claude   [user]    REACH  ✓  AUTH ✗  401 unauthorized
```

`doctor` 复用 `judges test` 结果作为其 judges section。

### 5.3 `--no-judge` 模式（用户 2026-05-20 加入）

用户场景：没装 vLLM / 没 API key / 不想花 token / 想纯本地确定性 / 隐私要求。让用户能一键过滤所有需 judge 的 bench/metric。

**注册表字段**（同步加到 metric 注册表，见 `INTEGRATION_FRAMEWORK_DESIGN.md` §5.2）：

```python
SUPPORTED_BENCHMARKS["worldjen"] = dict(..., needs_judge=True, ...)
SUPPORTED_METRICS["fvd"]         = dict(..., needs_judge=False, ...)
SUPPORTED_METRICS["object-binding"] = dict(..., needs_judge=True, compute_kind="needs_vlm")
```

**CLI 行为**：

```bash
# 列表带 judge 列 + 过滤
videvalkit list benchmarks            # 全部带 judge? 列显示
videvalkit list benchmarks --no-judge # 只列 judge-free（v0.2: 4/9 bench）
videvalkit list metrics --no-judge    # 只列 judge-free（v0.2: 17/20 metric）

# eval / eval-suite / metric 拒绝运行需 judge 的项
videvalkit eval --bench worldjen --no-judge
# → ERROR: worldjen requires a judge (VLM).
#          See `list benchmarks --no-judge` for runnable alternatives.

# eval-suite 自动 filter（不报错）
videvalkit eval-suite --all-anchored --no-judge ...
# → Running: vbench / worldscore (skipped: vbench2 / videobench / worldjen / t2vcompbench)

# doctor 显示当前 judge-config 下可跑范围
videvalkit doctor
# → Runnable in judge-free mode: 17 metric / 4 bench
#   Need judge endpoint: 3 metric / 5 bench (listed)
```

**判定规则**：
- `--no-judge` 与显式 `--judge X` 互斥（CLI 同时给两个 → 报错）
- profile (`quick` / `standard` / `full`) 内部声明的 `judge: default` 仅在没有 `--no-judge` 时生效
- 一个 bench/metric 在 `eval-suite --no-judge` 下被过滤 → 写入 result.json 的 `skipped_due_to_no_judge: [...]`

**v0.2 现状清单**：

| 类别 | Judge-free | Need VLM/LLM | 总 |
|---|---:|---:|---:|
| Benchmark | 4 (vbench · worldscore · physics_iq · v_reasonbench) | 5 (vbench2 · videobench · worldjen · t2vcompbench · vbench_pp) | 9 |
| Metric | 17 | 3 (object-binding · spatial-relationship · artifact-diagnostic) | 20 |

> **Messaging**：README / PPT / docs 必须显式宣传 "judge-free path = 17 metric + 4 bench"——给没 judge 资源的用户一个干净入口。

---

## 6. Python 入口

`videvalkit.runner.run(...)` 的签名扩展：

```python
def run(
    benchmark: str,
    videos: str | Path,
    workspace: str | Path,
    models: list[str] | None = None,
    dimensions: list[str] | None = None,
    # ---- judge 选择（互斥优先级：judge_override > judge > default） ----
    judge: str | None = None,                    # registry name | "paper" | "default" | None
    judge_override: dict[str, Any] | None = None, # ad-hoc cfg，等价于 CLI --judge-endpoint 系列
    # ----
    aggregator: str | None = None,
    scheduler_config: SchedulerConfig | None = None,
    **adapter_kwargs: Any,
) -> dict[str, Any]:
```

resolve 顺序在 `runner.run` 内统一：

```python
judge_cfg = resolve_judge(
    benchmark=benchmark,
    judge_name=judge,
    judge_override=judge_override,
    user_judges=load_user_judges(),
)
```

`resolve_judge()` 是新增的 pure function，单元测试覆盖所有分支。

---

## 7. 文件改动清单

### 7.1 新文件

| 路径 | 用途 |
|---|---|
| `src/videvalkit/configs/judge_loader.py` | `load_user_judges()` + `resolve_judge()` + `JudgeConfig` pydantic model |
| `src/videvalkit/cli_judges.py` | `judges list/show/test` 子命令组（独立文件避免 cli.py 进一步膨胀） |
| `tests/test_judge_resolution.py` | resolve_judge 全分支单元测试 |
| `tests/test_judges_yaml.py` | yaml 加载 / schema 校验 / 合并优先级测试 |
| `docs/JUDGE_SELECTION_DESIGN.md` | 本文件 |
| `examples/judges.yaml.example` | 示例 user yaml |

### 7.2 修改文件

| 路径 | 修改点 |
|---|---|
| `src/videvalkit/configs/__init__.py` | `SUPPORTED_JUDGES` 改为 `get_judges()` 函数式访问，内部 lazy-merge user yaml；旧的 dict 名做别名保留 |
| `src/videvalkit/configs/judges.py` | 加 `paper-llava-1.6-34b` / `paper-vbench2-vlm` / `paper-vbench-pp-vlm` 三个 paper alias entry |
| `src/videvalkit/configs/benchmarks.py` | 每个 entry 加 `paper_judge` + `recommended_judges` 字段 |
| `src/videvalkit/cli.py` | `eval` 命令加 `--judge-endpoint / --judge-model / --judge-kind / --judge-api-key-env` 四个 flag；新 `eval-suite` 子命令；接 `judges` 子命令组 |
| `src/videvalkit/runner.py` | 用 `resolve_judge()` 替换现有的 inline 解析；增加 `judge_override` 参数 |
| `src/videvalkit/diagnostics.py` | judges 部分调用 `judges test --all` 的内部 API |
| `docs/USER_MANUAL_en.md` + `_cn.md` | 加一节 §X "Switching VLM Judges"，覆盖三档与 ad-hoc 用法 |
| `docs/DEV_MANUAL.md` | §5.3 Module C 更新；§16 (Planned) 标为 done |
| `README.md` | quickstart 加一条 "want to use your own endpoint? edit `~/.config/videvalkit/judges.yaml`" |

### 7.3 删除/废弃

无。所有改动 backward-compatible。

---

## 8. 兼容性 & 迁移

### 8.1 向后兼容矩阵

| 现有用法 | v0.1 行为 |
|---|---|
| `videvalkit eval --bench X --judge gemma-4-31b-local` | 不变 |
| `videvalkit eval --bench X`（不传 --judge） | 走 `default_judge`，不变 |
| `from videvalkit.configs import SUPPORTED_JUDGES` | 保留，返回合并后的快照（含 user yaml） |
| 在源码改 `configs/judges.py` 加 entry | 仍可用，但推荐迁到 user yaml |

### 8.2 破坏性变更

无。`SUPPORTED_JUDGES` 仍是 importable 名字；新增字段都有 default。

### 8.3 灰度策略

环境变量 `VIDEVALKIT_JUDGE_USER_YAML=0` 关闭 user yaml 加载，便于在排查问题时验证"是否是 user 配置导致的"。

---

## 9. 测试方案

### 9.1 单元测试

- `resolve_judge`：5 个分支（registry name / paper / default / ad-hoc override / 错误 name）
- `load_user_judges`：缺文件 / 空文件 / 错配项 / 与 builtin 同名覆盖 / 多源合并优先级
- `JudgeConfig` schema：每种 kind 的必填校验

### 9.2 集成测试

- `videvalkit list judges` 在有/无 user yaml 时输出对比
- `videvalkit eval --bench worldjen --judge paper` 等价于 `--judge gemma-4-31b-local`（mock endpoint）
- `videvalkit eval --judge-endpoint http://fake/v1 --judge-model x --judge-kind openai_compatible` 走通到 dispatcher 层（用 stdlib fake http server）

### 9.3 不做的测试

- 真实 paper-LLaVA-1.6-34B 跑通（68GB 太重）—— 只 mock。
- 真实云 API 调用（不烧 token）—— 用 record/replay fixture。

---

## 10. 里程碑拆解

| 阶段 | 内容 | 工作量 |
|---|---|---|
| **M1 — Core resolver** | `judge_loader.py` + `resolve_judge` + user yaml 加载 + 单测 | 0.5 day |
| **M2 — CLI surface** | `eval` 新 flag + `judges` 子命令组 + `eval-suite` | 0.5 day |
| **M3 — Paper aliases** | `configs/benchmarks.py` 加 `paper_judge` + `configs/judges.py` 加 paper entry + 集成测试 | 0.5 day |
| **M4 — Docs** | USER/DEV manual + README + judges.yaml.example | 0.25 day |
| **M5 — 验证** | 跑一遍真实 worldjen smoke + 模拟 paper switch | 0.25 day |

**总计 ≈ 2 person-days**。M1/M2/M3 可并行 review，M4/M5 串行。

---

## 11. Open Questions

> 评审时需要决策的点：

1. **paper alias 命名**：`paper-llava-1.6-34b` vs `t2vcompbench-paper` vs `paper:t2vcompbench`？倾向第一种（明确指明模型）。
2. **`--judge paper` 在没有 paper 槽位的 benchmark（如 vbench）上的行为**：报错？还是 fallback 到 default？倾向**报错并给出建议**。
3. **是否给 paper-faithful judge 加 `requires` 字段**（"需要 70GB GPU 显存 / 需要 OPENAI_API_KEY"），让 `judges test paper-llava-1.6-34b` 在缺资源时直接 fail-fast？倾向**做**。
4. **国产模型默认 entry**：是否在 v0.1 就内置 Doubao / Qwen3-Max / Hunyuan-VL 等？倾向**只内置 OpenAI-compatible 通用条目模板**，真实 endpoint 由 user yaml 提供（避免内置过期 URL）。
5. **`eval-suite` 是否在 v0.1 范围**：会引入跨 benchmark workspace 协调复杂度，可能拆到 v0.2。倾向**v0.2**，v0.1 只做单 benchmark `eval` 增强。

---

## 12. 风险

| 风险 | 缓解 |
|---|---|
| user yaml schema 与未来 SDK 升级不匹配 | `JudgeConfig` 用 pydantic + `extra="allow"`，未知字段透传到 backend kwargs |
| paper-LLaVA-1.6-34B 在多数用户机器上跑不起 | `judges test` 提前 fail-fast；doc 明确写"paper 档需要 70GB 显存" |
| 三档语义被用户混淆（"为什么 paper 和 default 数字不一样"） | TEST_MANUAL 每个 benchmark 加一张表 "paper vs default 数值差异 + 原因" |
| ad-hoc endpoint 没经过 schema 校验直接打到 dispatcher | CLI 层走同一个 `JudgeConfig` 校验路径，dispatcher 不变 |

---

## 13. 附：决策快照

> 评审时一次性确认下列默认决策（"同意"即全部采纳）：

- ✅ 不做 per-dim judge（留 v0.3）
- ✅ 不做 judge 自动 fallback（系统层 retry 已经足够）
- ✅ 不新增 backend kind（仅 openai_compatible / gemini / anthropic 三种）
- ✅ user yaml 路径固定 `~/.config/videvalkit/judges.yaml`，不支持 `--judges-yaml-path` 自定义
- ✅ 合并语义 = top-level key 覆盖，无 field-level deep-merge
- ✅ `eval-suite` 推到 v0.2
- ✅ paper alias 命名采用 `paper-<model-short-name>` 形式
- ✅ paper-faithful judge 加 `requires` 资源声明，`judges test` 做 fail-fast
- ✅ 国产模型不内置具体 entry，只在 example yaml 给模板

—— end of design v0.1 ——
