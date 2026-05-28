# videogenevalkit 文档

**面向文本生成视频(T2V)的统一评测工具包。**
一个 CLI · 一个工作区 · 一套 schema。10 个基准 · 8+ 评审 · 20 个指标 · 44 个能力标签。

> 状态:`v0.2-dev` · 384 个测试全绿 · v0.2 接近完成。
> 本 wiki 是主文档。设计理念存档见 [`docs/design/`](design/PRODUCT_DESIGN.md)。

---

## 快速导航

| 板块 | 页面 |
|---|---|
| **从这里开始** | [快速上手](wiki/Getting-Started.md) · [核心概念](wiki/Concepts.md) |
| **指南** | [评审选择](wiki/guides/Judge-Selection.md) · [配置档与快速评测](wiki/guides/Profiles-and-Quick-Eval.md) · [训练监控](wiki/guides/Training-Monitor.md) · [扩展](wiki/guides/Extending.md) |
| **参考** | [命令行](wiki/reference/CLI.md) · [基准](wiki/reference/Benchmarks.md) · [指标](wiki/reference/Metrics.md) · [评审](wiki/reference/Judges.md) · [能力标签](wiki/reference/Capability-Tags.md) |
| **项目** | [架构](wiki/Architecture.md) · [路线图与状态](wiki/Roadmap.md) · [贡献指南](wiki/Contributing.md) |

### 完整文档地图

本 wiki 是当前的、可操作的参考文档。另外还有两套文档:

| 文档集 | 位置 | 状态 | 用途 |
|---|---|---|---|
| **Wiki**(本套) | `docs/wiki/` | ✅ 当前(v0.2) | 日常使用、参考、架构 |
| **手册** | `docs/DEV_MANUAL.md` · `TEST_MANUAL.md` · `USER_MANUAL.md`(en/zh) | ⚠️ v0.0.1 时代 + 提示横幅 | 深度架构理念(DEV)、论文对齐验证表(TEST)、长篇安装(USER) |
| **设计存档** | `docs/design/` | 🔒 冻结 | 每个 v0.2 子系统*为何*这样设计(8 篇设计文档 + 幻灯片) |

> 经验法则:**想*做*某事 → wiki**;**想知道*为什么* → design**;**想要验证数字 → TEST_MANUAL**。

---

## 它是什么

videogenevalkit 把碎片化的 T2V 评测生态——VBench、VBench-2.0、
Video-Bench、WorldJen、WorldScore、T2V-CompBench 等等——统一到单一接口之后。

**三种评测方式**(均已可用):

| 入口 | 命令 | 回答的问题 |
|---|---|---|
| **基准(Benchmark)** | `videvalkit eval --bench vbench` | “我的模型在 VBench 上得分如何?” |
| **指标(Metric)** | `videvalkit metric run --name fvd ...` | “这些视频的 FVD 是多少?” |
| **能力(Capability)** | `videvalkit capabilities eval motion ...` | “跨所有指标看,运动表现如何?” |

**三条设计承诺:**

| 承诺 | 含义 |
|---|---|
| 适配而非重写 | 每个基准逐字节包裹上游代码 |
| 后端可插拔 | 评审 / 指标 / 聚合器都是可替换的注册表条目 |
| 插件优先 | 通过 YAML / pip / 本地目录扩展——无需 fork |

---

## 30 秒速览

```bash
# 健康检查——查看设备、基准、指标、配置档、覆盖率
videvalkit doctor

# 用 quick 配置档跑一个基准,无需评审
videvalkit eval --bench vbench --profile quick --videos gen/ --workspace ws/

# 跑单个指标(FVD 会自动下载其骨干网络)
videvalkit metric run --name fvd --gen-videos gen/ --refs my-ref --allow-tiny-sample

# 跨指标评测整个能力
videvalkit capabilities eval motion --videos gen/
```

完整走查见[快速上手](wiki/Getting-Started.md)。
