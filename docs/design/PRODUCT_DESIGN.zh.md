# 产品设计

> **本文为何存在。** wiki 讲*做什么*、*怎么用*;本设计存档记录*为什么这样设计*。
> 状态、指标计分板、版本时间线见[路线图](../wiki/Roadmap.md);分层架构与调用链见
> [架构](../wiki/Architecture.md)。

---

## 1. 愿景

**videogenevalkit 是文本生成视频(T2V)评测的统一入口**——一行 CLI、一份
workspace、一套 schema——把碎片化的 T2V 生态(VBench、VBench-2.0、
Video-Bench、WorldJen、WorldScore、T2V-CompBench……)整合到单一接口之后,
并保持对扩展开放。

三条价值主张:

1. **不重造轮子。** 每个基准都是上游论文仓库的 thin adapter,逐字节对齐官方榜单。
2. **不绑死后端。** 评审 / 评分器 / 聚合器都是可替换的注册表条目——本地 vLLM、
   Claude/Gemini/GPT,或论文原生模型,一个 CLI flag 切换。
3. **不强制 fork。** 新基准 / 指标 / 评审通过 manifest YAML、pip `entry_points`
   或本地用户目录扩展——不改 toolkit 源码。

### 范围边界——我们刻意**不做**什么

| 不做 | 理由 |
|---|---|
| 视频生成本身 | `BaseT2VModel` 预留但不实现;视频从外部进来 |
| 重写官方 metric 公式 | adapter 调上游包;我们只管 IO / 调度 / 聚合 |
| 跨机分布式 | 调度器单机;接口为未来 Ray/SSH 预留 |
| 评测评审自身(“谁更准”) | 留给独立的 `judge-eval` 子项目 |
| 自动 paper→adapter 代码生成 | 不靠谱,留给上游作者 |

---

## 2. 用户与场景

| 用户 | 核心诉求 | 接口 |
|---|---|---|
| 模型研发者 | 跑新 T2V 模型,跨所有基准拿可比数 | `eval --bench X` ×N + `aggregate` |
| 基准集成者 | 把自己的评测方法接入工具链 | manifest YAML / `BaseBenchmark` 子类 |
| VLM 集成者 | 接入私有 vLLM 或新云 API | `judges.yaml` / `--judge-endpoint` |
| 论文复现者 | 证明对已发布数字的忠实复现 | `--judge paper` |
| 下游部署者 | 在自己机房用不同资源跑 | env tarball + `judges.yaml` + 插件加载器 |

三个高频流程:**(A)** 新模型 → 跑遍每个基准 → 聚合 → 汇报;**(B)** 新评测想法
→ 一份 manifest → 完全进入流程;**(C)** 忠实复现论文 → `--profile full --judge paper`。

---

## 3. 系统全景

七层,每层只依赖其下一层;新增指标/基准/评审只触及 L2(实现)+ L5(一行注册表),
别无其他。完整图与端到端调用链见[架构](../wiki/Architecture.md)。
承重思想:**四个注册表**(`SUPPORTED_BENCHMARKS / _JUDGES / _METRICS /
_AGGREGATORS`)是单一事实来源,由 builtin + 用户配置 + 插件惰性合并。

---

## 4. 四大支柱(+ 一个横切)

| 支柱 | 理念 | 深入 |
|---|---|---|
| **A — 环境与一键安装** | 可复现的 conda-pack 快照 + `pip install -e .`;工具包须在各集群一致运行 | — |
| **B — 评审切换** | 复现与省钱*都*是一等公民:每个用评审的基准声明 `paper_judge` 与 `default_judge`,绝不偷换模型 | [扩展性](EXTENSIBILITY_DESIGN.md) |
| **C — 快速集成 + 指标** | 基准或独立指标应能无需 fork 即可加入;抽出的指标与基准路径逐位一致 | [扩展性](EXTENSIBILITY_DESIGN.md) · [指标](VIDEO_METRICS_DESIGN.md) |
| **D — 快速评测与训练监控** | 训练循环需要快速、已校准的读数,区别于论文忠实运行 | [快速评测](QUICK_EVAL_DESIGN.md) |
| **× — 能力标签** | 跨基准*按能力*评测,而非仅按基准 | [能力标签](CAPABILITY_TAGS_DESIGN.md) |

B、C、D 与能力横切在 v0.2 一起交付:它们共用插件加载器和同一种 YAML schema 风格,
拆开只会割裂设计。

---

## 5. 跨支柱原则

这些原则贯穿每个支柱,决定了工具包的“长相”。

1. **注册表驱动,无插件 DSL。** “加东西” = 在某个注册表加一条。三个惰性合并源
   (builtin dict + 用户 yaml + entry_points),没有动态发现的魔法。
2. **适配优先。** 每个基准 / 论文忠实评审都是 thin adapter:归一化输入路径、
   归一化输出为单一 `RawResult`、编排、聚合。**绝不重写论文的 metric 公式。**
3. **80/20 双轨。** 简单场景用 manifest YAML;复杂场景用 Python 子类。两者进同一
   注册表,runner 看不出区别。
4. **显式的 `paper` / `default` / 自定义评审档。** 基准绝不把论文模型偷偷降级为小模型。
5. **Workspace 即唯一事实。** 一次运行的全部输入 / 输出 / 日志 / 缓存都在一个
   workspace 下;逐 prompt 的 raw JSON 是断点续跑原语,API 日志是回放原语。
6. **最小可信接口面。** `BaseBenchmark` = 4 方法,`BaseScorer` = 1,
   `BaseAggregator` = 1,manifest ≤ 12 个顶层字段。接口扩张需明确动机。
7. **可复现是契约,而非目标。** 官方榜单 JSON 已入库;CI 在容差内与之 diff。
8. **插件优先;fork 是最后选项。** 所有正常扩展路径都不改 toolkit 源码。

---

## 6. 决策快照

- 定位:T2V 评测的**统一入口**——不做生成、不做分布式、不做 judge-eval(独立项目)。
- 支柱 **B + C + D + 能力标签一起交付**(A 处于巩固期;NPU 后移)。
- **重集成轻框架**:独立指标 + 逐位一致 lift 双入口 + 能力标签为核心;
  脚手架/校验器推到 v0.3。
- **强制评审协议**:所有 PR 过 3 层关卡——见[评审协议](REVIEW_PROTOCOL.md)。
- **排除**:PSNR/SSIM/LPIPS/FID-image(不适用于 T2V);NPU(后移)。

---

## 7. 设计存档地图

| 文档 | 内容 |
|---|---|
| **PRODUCT_DESIGN**(本文) | 愿景 · 范围 · 支柱 · 跨支柱原则 |
| [VIDEO_METRICS_DESIGN](VIDEO_METRICS_DESIGN.md) | 两档分类 · 双入口 · 逐位一致 lift · 注册表 schema |
| [EXTENSIBILITY_DESIGN](EXTENSIBILITY_DESIGN.md) | 评审切换 + 基准/指标集成 + 插件模型 |
| [QUICK_EVAL_DESIGN](QUICK_EVAL_DESIGN.md) | 评测配置档 · 子集校准 · 训练监控 |
| [CAPABILITY_TAGS_DESIGN](CAPABILITY_TAGS_DESIGN.md) | 受控标签词表 · 解析器 · 版本管理 |
| [REVIEW_PROTOCOL](REVIEW_PROTOCOL.md) | 三层质量关卡 |
| [NPU_ADAPTATION_DESIGN](NPU_ADAPTATION_DESIGN.md) | 后移的未来计划存根 |

可操作参考是 [wiki](../index.md);这些文档解释*为什么*。
