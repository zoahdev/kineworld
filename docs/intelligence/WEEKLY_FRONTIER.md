# WEEKLY_FRONTIER.md — 2026-W36

> 截止：2026-09-02（Asia/Shanghai）。本周只记录一手可核验变化；论文声称与 KineWorld 实测严格分开。

## 本周改变路线的发现

### stable-worldmodel 使“自研通用 Harness”停止

- 一手来源：`galilai-group/stable-worldmodel` 官方 repo、PyPI 0.1.1、官方 `pyproject.toml`。
- 免费能力：MIT；统一数据、环境、模型规划和评测；CEM/MPC；LeWM reference；Push-T 16 个 factors of variation。
- 决策：KineWorld 不继续开发通用执行 harness。已写的 v0 contract 冻结为证据元数据/能力声明薄层；实际环境、solver 和模型 adapter 优先直接复用 stable-worldmodel。
- 产品影响：通用 Evaluation SDK 已有强公开竞品，Primary Wedge 改为**世界模型风险、校准与失败诊断**，而不是基础评测框架。

### LeWorldModel 从 WATCH 升级为 REPRODUCE

- 一手来源：arXiv:2603.19312 v3、论文第一作者的 `lucas-maes/le-wm` 上游官方仓库、Hugging Face 官方 LeWM collection。`hongqin/leworldmodel` 经核验是上游 fork，不作为官方源。
- 公开能力：约 15M 参数；端到端从像素训练；两项 loss；官方提供 Push-T 权重、数据与 eval 命令。
- 许可：代码 MIT；HF Push-T 权重与数据页面均标注 MIT。
- 本机成本：权重 72.3 MB，可低成本先评；训练数据 13.1 GB，属于大下载，当前不下载。
- 论文声称：单 GPU 数小时、最高 48× 规划加速。此两项是作者报告，**尚不是 KineWorld 实测**。
- 决策：不打断 KW-EXP-0006。其结束后先冻结公平协议，再做权重级 Push-T 复现；repo 仅 6 commits，按新项目处理，不能直接作为生产底座。

## 当前最高价值事实

1. 现有 jepa-wms baseline 已真实复现，但代码/权重为 CC BY-NC 4.0，只能用于研究证据，不能直接成为商业 Core。
2. KW-EXP-0001 已否定朴素闭环加频率：成功率 0.467→0.400，成本上升。
3. KW-EXP-0003/0004 已否定 CEM spread 作为可靠 uncertainty proxy。
4. KW-EXP-0005 更正后表明误差是少量点级尖峰，剔除尖峰后随 horizon 仅温和约 1.5× 增长；不是 episode 级退化，也不是指数爆炸。
5. KW-EXP-0006 正在验证最后一个廉价解释：物理交互事件是否能解释尖峰。结果出来前不追加变量。

## 本周优先级

| 顺序 | 动作 | 计算/下载 | 通过门 | 失败即停止 |
|---:|---|---|---|---|
| 1 | 完成 KW-EXP-0006 | 已在运行；不并发 GPU | 预注册统计支持物理事件条件信号 | 关闭 cheap-diagnostics |
| 2 | 冻结 LeWM vs jepa-wms 公平协议 | CPU/文档 | 相同任务、episode、成功定义、预算披露 | 协议无法对齐则只做分别复现，不做排名 |
| 3 | 以 stable-worldmodel + 官方 LeWM 权重复现 | 72.3 MB；单 GPU | 可重复运行、hash、配置、逐 episode 结果齐全 | 安装/格式成本超过 1 天则降级 |
| 4 | 决定是否申请 13.1 GB 数据下载 | 需 Founder 同意 | 权重复现有价值且需要训练消融 | 未满足则不下载 |

## 明确不做

- 不把论文成绩改写成 KineWorld 成绩。
- 不在 KW-EXP-0006 未结束时并发启动 GPU 任务。
- 不继续扫 `num_act_stepped`、L1/L2、spread threshold 或 horizon-conditioned conformal。
- 不重造 stable-worldmodel 已提供的通用环境、CEM/MPC、数据或 evaluator。
- 不用“世界最强”“超过白泽”作未经第三方验证的事实性宣传。

## 下周检查

- LeWM repo / weights 是否更新；依赖树中是否存在非商业组件。
- jepa-wms upstream 是否新增 commit、权重或 TorchHub/HF 入口；只记录，不在活跃实验中 pull。
- KW-EXP-0006 是否满足预注册判据；若未满足，执行关闭线程决定。
