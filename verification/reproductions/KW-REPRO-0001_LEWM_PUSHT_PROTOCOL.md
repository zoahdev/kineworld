# KW-REPRO-0001 — LeWorldModel Push-T 冻结复现协议

状态：**PRE-REGISTERED / NOT RUN**  
冻结日期：2026-09-02  
执行条件：KW-EXP-0006 完成、GPU 空闲、官方代码可取得。  
资源边界：第一阶段只允许官方代码 + 72.3 MB 权重；不下载 13.1 GB 训练数据。

## 1. 问题

LeWM 官方实现能否在 KineWorld 当前单机上按作者协议完成 Push-T 规划评测，并形成可复现、可审计、许可为 GREEN 的第二模型证据？

这不是“LeWM 是否优于 jepa-wms”的单步问题。两个官方 evaluator 的任务切片、episode 初始化、规划 horizon、执行预算和成功定义不同；未完成协议对齐前，禁止把两个成功率排成排行榜。

## 2. 一手冻结事实

- 论文：arXiv:2603.19312 v3。
- 上游官方代码：`lucas-maes/le-wm`（论文第一作者仓库），MIT；执行时记录精确 commit。`hongqin/leworldmodel` 是该仓库的 fork，不作为上游来源。
- 官方模型：`quentinll/lewm-pusht`，HF 页面标注 MIT，`weights.pt` 72.3 MB。
- 官方数据：`quentinll/lewm-pusht` dataset，13.1 GB；本阶段不下载。
- 官方 eval config：Push-T `seed=42`、`num_eval=50`、`goal_offset_steps=25`、`eval_budget=50`、`horizon=5`、`receding_horizon=5`、`action_block=5`。
- 官方 CEM：`num_samples=300`、`n_steps=30`、`topk=30`。

## 3. 两层复现，禁止混算

### Tier A：作者协议复现

目标：确认官方代码 + 官方权重在本机可运行，输出作者 evaluator 原生指标。

必须保持：

- 官方 config 与 CEM 参数不改。
- seed=42；先 3 个 episode smoke，成功后再 50 episode。
- 记录官方输出原文、逐 episode 明细（若 upstream 未提供则只增加不改变行为的 instrumentation）。
- 记录总 wall time、每次 plan latency、GPU peak memory、软件/驱动环境。

Tier A 只能与 LeWM 作者协议或同协议复现比较，不与当前 jepa-wms 15-episode 缩减 CEM 数字直接排名。

### Tier B：预算披露的任务对照

仅在 Tier A 成功后执行。目标是让 LeWM 与 jepa-wms 在尽可能相同的任务定义下比较。

必须先形成 adapter feasibility note，至少逐项对齐或披露：

1. Push-T 环境实现与版本。
2. 初始状态 / goal 的抽样来源及精确索引。
3. 成功条件、episode 长度与 early termination。
4. 观测尺寸、frame skip、action block。
5. 规划 horizon、replan 频率。
6. CEM samples × iterations × top-k 与总 model forward 数。
7. checkpoint 训练数据是否与 eval 初始状态重叠。

无法对齐的字段标记 `NOT_COMPARABLE`，不做总排名。

## 4. 主要指标

Tier A：

- 官方 `metrics` 完整对象（不只摘最高值）。
- run completion / finite outputs。
- wall time、plan latency 分布、peak VRAM。
- 重跑确定性：同 seed 的 episode 选择与结果是否一致。

Tier B（仅协议允许时）：

- success rate + Wilson 95% CI。
- 相同 episode 的 paired win/loss/tie。
- 平均及 P50/P95 plan latency。
- 每 episode 与每成功 episode 的总 model evaluations。
- peak VRAM。

## 5. 判定门

- `REPRODUCED`：Tier A 全部 episode 完成、无非有限值，config/commit/weight hash/环境/原始输出齐全。
- `PARTIAL`：smoke 成功但 full 因明确资源上限停止；保留实际结果，不外推。
- `NOT_REPRODUCED`：官方流程在一天工程预算内无法运行，或结果不可审计。
- `COMPARABLE`：Tier B 的 7 个协议字段全部对齐或已量化归一化。
- `NOT_COMPARABLE`：任何关键任务定义无法对齐；分别报告，不排名。

## 6. Kill criteria

- clone / install / checkpoint conversion 合计超过 1 个工作日仍未完成：停止，记录 blocker。
- 任何依赖引入未披露的非商业限制：商业候选降级为 YELLOW/RED，研究复现可继续但不得进商业 Core。
- 官方 full CEM 超出本机资源：不偷偷缩参并称“官方复现”；另开 `resource-reduced` run，名称与结果完全分离。
- 为提高结果而改变 seed、episode 或成功定义：实验作废，重新预注册。

## 7. 证据包要求

- source commit 与 dirty status。
- `weights.pt` SHA-256 与大小。
- 完整 resolved config。
- Python / PyTorch / CUDA / driver / GPU 快照。
- stdout/stderr 原文、逐 episode JSON、汇总 JSON。
- instrumentation patch 与行为等价 smoke 对照。
- claim 文件只允许引用本实验实际生成的数据。

## 8. 当前未知

- 官方权重是否能在 Windows 原生环境直接转换/加载。
- `stable-worldmodel` 依赖树的完整许可证。
- LeWM official dataset 是否必须用于 evaluator 的 initial/goal states；若必须，Tier A 将需要 13.1 GB 数据下载，必须先取得 Founder 明确同意。
- jepa-wms 与 LeWM 的 Push-T 环境/数据版本能否形成 paired episodes。

## 9. 允许的对外表述（运行前）

“KineWorld 已预注册 LeWorldModel 的独立复现与公平比较协议。”

禁止表述：已复现、已超过、商业底座已替换、比官方快、世界第一。
