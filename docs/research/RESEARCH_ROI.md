# RESEARCH_ROI.md — 研究投资回报率排序

> 日期：2026-09-02（KW-EXP-0001 后重排）。规则：每次只选 ONE highest-ROI experiment；禁止机械执行旧 roadmap。
> 输入：Global Frontier（GIANTS.md）+ KineWorld Failure Distribution（results/failures/）+ 单卡约束 + 科学价值 + 产品价值。

## 已完成实验：E1 闭环重规划（KW-EXP-0001）

| 项 | 结果 |
|---|---|
| 单一变量 | `num_act_stepped 6→2`（每 episode 1 次规划 → 3 次规划） |
| 主指标 | success_rate 0.4667 → **0.4000** |
| 次指标 | ep_end_dist 83.8867 → **71.0204**；reward 6.8199 → **7.1139** |
| 成本 | planning_calls 15 → 45；wall_time +80.0% |
| 翻转 | recovery=0；regression=1（ep_4） |
| 判定 | **NOT_SUPPORTED（预注册 Kill Criteria 触发）** |
| 归因更新 | PLANNING-frequency 假设降级；当前主瓶颈转向 DYNAMICS / planner-objective mismatch |

**决策：不再做 naive `num_act_stepped` sweep。** 继续扫 nas=1/3/4/5 属于在同一失败假设上机械加码，违反 Kill Fast。

## 当前候选重排

| 排名 | 候选 | 类型 | 为什么现在排这里 | 主要风险 |
|---:|---|---|---|---|
| 1 | **KW-EXP-0006（已完成）：物理状态条件误差** | 诊断 | obj_disp 显著预测误差（连续 ρ=0.42、尖峰 Fisher p=0.0018），但带效率≈0（比例带 8–22× 更宽、分档带 ratio≈0.99）；信号真实但不可部署 | obj_disp 是 frozen JEPA 上最后一个可解释变量，之后诊断技巧已到尽头 |
| 2 | **KW-EXP-0005 多 horizon 误差（已完成）** | 诊断 | n=96 复核：存在温和单调增长（1.56×），非单调是 3 点尖峰假象；horizon 条件化中性（带宽比≈1.01） | 已闭合 |
| 3 | **KW-EXP-0004 校准基线（已完成）** | 诊断/校准 | 确定性模型无原生方差；const 保形带有效，spread 带被严格支配 | 已闭合 |
| 4 | **LeWM Push-T 官方权重同台复现** | 对照复现/商业许可去风险 | 官方权重 MIT、约 15M 参数、72.3 MB；复用 MIT `stable-worldmodel` 的 LeWM reference、Push-T、CEM/MPC 与 evaluator；诊断技巧已走到尽头，转入第二方法对照 | evaluator/protocol 不完全相同；必须先冻结公平比较协议；完整 extras 许可仍需审计 |
| 5 | E2 JEPA uncertainty signal（条件不确定性） | 科学实验 | 已有验收门槛：90% 覆盖下平均上界 < 0.091499 且成功/失败覆盖差 < 10pp | 必须引入 ensemble/probabilistic head，即进入训练 |
| 6 | Physically grounded objective | 科学实验 | obj_disp 揭示误差与物理交互强度相关，为"显式建模接触动力学"提供新依据 | 需要额外状态监督 |
| 7 | E4 Dynamics mitigation | 科学实验 | 未证明动力学唯一主因；且高 latent 误差 ≠ 失败（9/18 尖峰成功） | 单卡训练成本高 |
| 8 | E5 R2-Dreamer 复现 | 对照复现 | MIT、可作第三方法对照 | 当前绑定较弱 |

## 研究线程关闭记录：cheap uncertainty proxy（2026-09-02）

连续三个负面结果，按 Kill Fast 纪律**关闭"在 frozen JEPA 上寻找廉价不确定性代理"这一线程**：

| 实验 | 被测条件变量 | 结果 |
|---|---|---|
| KW-EXP-0003 | CEM `elite_loss_std` | 与真实误差 `r=-0.0737`；成功组 spread 反而更高；弃权不优于随机（p=0.6723） |
| KW-EXP-0004 | 模型原生方差 | 不存在；确定性点预测，仅 const 边缘保形带可用 |
| KW-EXP-0005 | rollout horizon | 分层带严格劣于 pooled（α=0.20 下宽 +11.74%，α=0.10 下 +114.67%）；误差曲线非单调 |

**共同结论（2026-09-02 更正后）**：latent 预测误差**在 episode 级与 horizon 级均无结构**，重尾来自**稀疏的点级尖峰**（90 点中仅 3 个），且模型没有原生不确定性。

支撑证据：
- ICC = 0.0000（MS_between 0.002204 ≈ MS_within 0.002247，F=0.981）→ 无 episode 间方差成分。
- `corr(k1,k2)=0.15`、`corr(k1,k6)=-0.23` → 无 episode 内持续性。
- 剔除 3 个尖峰后误差随 horizon **温和单调上升**（0.0295 → 0.0434，约 1.5×），不存在指数级复合爆炸。

**KW-EXP-0006 终结评估（2026-09-02，96 episodes）**：物理交互强度 `obj_disp` 是**本线程唯一一个显著正结果**——显著预测尖峰率（Fisher p=0.0018）与连续误差（Spearman ρ=+0.42，剔除尖峰后仍 ρ=+0.40，控制 horizon 后每层均显著）。但 **Q3 带效率≈0**：比例归一化带被 pooled 带支配（8–22× 更宽），分档带 ratio≈0.99（no_move 收紧 14% 但 high_move 膨胀抵消）。**信号真实、可解释误差来源，但不足以转化为可部署的计算分配/gating 机制**。

**诊断线程终态**：连续四个候选（CEM spread、原生方差、horizon、物理交互）中，只有物理交互给出统计显著但效率≈0 的信号。**"在 frozen JEPA 上挖可用不确定性"这一诊断技巧已到尽头**——能找的结构都找到了，但没有任何一个能压缩保形上界。可用不确定性必须来自训练/结构改造（ensemble / probabilistic head / IMLE 采样），诊断无法绕过。已冻结决策门照常生效：下一项优先做 LeWM Push-T 官方权重同台复现。

## 不确定性前沿核验（2026-09-02）

- **WIMLE**：一手论文 arXiv:2602.14351 与官方 GitHub 已核实。可复用思想：IMLE 多模态预测、ensemble×latent sampling、inverse-variance weighting；不可直接复用现成 frozen JEPA checkpoint。代码是 JAX/SAC 完整训练系统，需独立环境与控制基准。
- **HAUWM/HCU**：一手 OpenReview PDF 已核实。核心是 probabilistic ensemble + variable-horizon prediction + HCU loss，使预测方差随 horizon 增长；代码与 checkpoint 尚未确认公开可用。
- **决策**：不移植 WIMLE/HAUWM 完整训练；先把 calibration target、horizon-conditioned variance、selective risk 作为协议参考。

## KW-EXP-0004 结果（2026-09-02，无训练）

判定：**`CAPABILITY_GAP_CONFIRMED_MARGINAL_BASELINE_ONLY`**

- **能力审计（代码可核验）**：unroll 取 CEM 最佳候选单次展开；`modelcustom/simu_env_planning/` 全目录 dropout / sampling / 噪声注入 **0 命中**，无 ensemble 头。**确定性点预测，无原生方差。**
- **保形校准基线**（leave-one-episode-out，15 折，上界带）：
  - α=0.10：const 覆盖 **0.9111**、平均上界 **0.091499**；spread 带覆盖 0.9333 但上界 **0.239001**（宽 2.61×）；cemloss 带 0.112227。
  - α=0.20：const 覆盖 **0.8222**、上界 **0.043719**；spread 带同覆盖 0.8222 但上界 **0.180391**（宽 **4.13×**）。
  - 条件覆盖：spread 带成功段 0.6667 / 失败段 0.9744，差 30.8pp；const 带 α=0.20 下 0.8333 / 0.8205 基本平衡。
- **Selective risk**：按 spread 弃权 20% 时残留风险 0.038021，**劣于随机** 0.036962（p=0.6723）；弃权 40% 时 0.035930 vs 随机 0.036982（p=0.3815），与随机不可区分。
- **Horizon 分层 = 结构性缺口**：全部 45 segments 共享同一预测 horizon（2 latent steps = 10 env steps）。按 replan 位置分层为 0.033514 / 0.042226 / 0.035054，CI 全部重叠，无显著效应，且**这不是 horizon 曲线**。
- **可量化价值上限**：const 带 90% 上界 `0.091499` = 池化均值 `0.036931` 的 **2.48 倍**。未来条件不确定性方法的验收门槛：90% 覆盖下平均上界 < `0.091499`，且成功/失败覆盖差 < 10pp。

**决策**：spread 归一化带 / spread gating / spread 弃权 / spread compute allocation 全部禁止。当前唯一可用不确定性是 const 保形带，仅保证边际覆盖。

## KW-EXP-0005 结果（2026-09-02，无训练）

判定：**`HORIZON_CONDITIONING_KILLED_ERROR_NONMONOTONIC_HEAVY_TAILED`**

- **复现对照**：success `0.4667`、`ep_end_dist 83.88669`、15 次规划，与官方 nas=6 baseline **逐位一致**，埋点未改变 planner 行为。
- **索引对齐核验**：实测 `predicted_len=7`、`plan_horizon=6` → `tau=1`，`predicted[k]` 即第 k 个 rollout 步；`aligned == legacy` 在 15/15 段成立。**不存在错位 bug**，EXP-0002～0004 结论不受影响。
- **误差曲线（以中位数为准）**：0.0219 → 0.0236 → 0.0340 → **0.0494** → 0.0360 → 0.0364。k=1→4 上升约 2.3×后回落。**不存在"误差随 horizon 复合爆炸"**。均值曲线完全被离群点支配（k=1 的 mean/median = 2.43）。
- **离群归因**：ep_4 的 k=1 误差 `0.3851`（中位数 17.6 倍）**却成功了**；ep_10（k=4, 0.2269）、ep_8（k=4, 0.1699）、ep_7（k=1, 0.1128）失败。**高 latent 误差 ≠ 任务失败**。
- **Planner 乐观度**：predicted-goal 与 actual-goal 同步下降（0.169→0.073 与 0.195→0.071），乐观度仅 0.013–0.026（约目标距离的 10–15%），仅 k=1/4/5 的 CI 排除 0。objective-mismatch 假设**降级为弱支持**。
- **Horizon 条件校准 Kill**：α=0.10 下 pooled `0.066455` / per-horizon `0.142660`（+114.67%）；α=0.20 下 pooled `0.052124` / per-horizon `0.058245`（+11.74%，覆盖仅高 1.1pp）。机制：每层 n=14，α=0.10 保形水平退化为 `ceil(15×0.9)/14 = 1.0`（样本最大值）。
- **⚠ 事后更正（2026-09-02）**：两项再分析部分推翻上述 Q1/Q1b 表述，详见 `KW-EXP-0005.md` 更正节与 manifest 的 `post_hoc_correction`：
  - **「非单调」是假象**：90 点中只有 **3 个尖峰**（ep_4 k=1 / ep_10 k=4 / ep_8 k=4）。剔除后各 k 均值 **0.0295 → 0.0261 → 0.0363 → 0.0432 → 0.0398 → 0.0434**，约 **1.5×** 温和单调上升。正确表述是"存在温和增长，不存在指数级复合爆炸"。
  - **「episode 特异」不准确**：ICC = **0.0000**（F=0.981, df 14,75），episode 间方差为零；`corr(k1,k2)=0.15`、`corr(k1,k6)=-0.23`。ep_4 的中位误差 `0.03559` 正常，其 0.3851 是**单点尖峰**而非整段退化。应更正为"**点级稀疏尖峰**"。
  - **不受影响**：Q3 的 Kill 与 Q4 条件覆盖结论仍成立。

## 下一轮最高 ROI

**KW-EXP-0006 已完成**，判定 `PHYSICAL_INTERACTION_SIGNAL_FOUND_INEFFICIENT_FOR_BANDS`。物理交互 `obj_disp` 是本线程唯一显著正结果，但带效率≈0，不可部署。

诊断阶段（frozen JEPA 上挖不确定性）**已到终点**：四个候选（spread / 原生方差 / horizon / 物理交互）全部检验完毕，无一个能压缩保形上界。下一轮 ONE highest-ROI 按已冻结决策门，**从诊断转入第二方法对照**：

**LeWM Push-T 官方权重同台复现**——复用 MIT `stable-worldmodel` 的 LeWM reference 与 Push-T/CEM/MPC harness，先冻结公平比较协议，下载 72.3 MB 官方权重（MIT 许可）。这是把项目从"单方法诊断"推进到"跨方法对照"的最低成本路径，同时去风险商业许可问题。训练数据 13.1 GB 未经 Founder 明确同意不下载。

## KW-LEWM-0001L 收口（2026-09-02）

- 官方 72.3 MB 权重已严格加载，哈希和 18,034,478 参数量已冻结。
- runtime smoke 完成，但由于官方 13.1 GB 数据集未下载，无法获得 evaluator 必需的 dataset-fitted `StandardScaler`。
- identity action transform 造成显著越界，单任务 0/1 **不是性能结果**；Stage B/C 已按 kill gate 停止。
- 结论：LeWM **基础设施兼容性通过，公平控制复现未完成**。继续放大 CEM 或样本数的 ROI 为零。

新的 ONE highest-ROI：构建 **KineWorld Failure & Risk Card v0**，把已经完成的 96-episode 官方 JEPA-WM 结果转化为机器可读、可复核、可供第三方复跑的评测产品切片。它不宣称模型更强，而是证明 KineWorld 能发现公开 success rate 掩盖的失败结构、校准缺口和物理交互风险；这是当前无需训练、无需大下载、最接近客户/投资人可理解交付物的路径。

**完成回填**：`KW-RISK-CARD-0001` 已生成，离线完整性验证 PASS，测试总计 12/12 PASS；包含机器卡、人类卡、冻结哈希、Level 1/2/3 验证分级与空白外部签署模板。证据等级仍为 E1，因为没有任何独立机构实际复跑或签字。

下一 ONE highest-ROI：把 Level 3 验证请求做成可发送的最小外联包，并筛选 3–5 个具备机器人/世界模型实验能力、利益冲突低且可公开署名的合肥/高校第三方。目标不是购买背书，而是获得一个可复跑、可否定、可公开的独立技术审查。实际发送前保留 Founder 审批。

**准备状态**：已从高校官方页面筛出首批 4 个潜在技术审查对象，并写好不求背书、允许否定结果的外联邮件与利益冲突边界，见 `company/EXTERNAL_VALIDATION_OUTREACH.md`。尚未发送，尚无专家反馈或第三方复现；必须先补齐公开 Level-3 代码/日志下载地址。

## KW-EXP-0006 后的已冻结决策门

- ~~若物理事件可显著解释尖峰：只做一个预注册的 event-conditioned mitigation 实验。~~ **已触发但不适用**：obj_disp 统计显著（Q2 成立）但带效率≈0，做 event-conditioned mitigation 无效率收益，不值得单独实验。
- 若不可解释：关闭 cheap-diagnostics 线程，下一项优先做 **LeWM Push-T 官方权重同台复现**；底层优先复用 `stable-worldmodel`，不自研第二套 harness。
- LeWM 复现先下载 72.3 MB 权重；13.1 GB 训练数据属于大下载，未经 Founder 明确同意不下载。
- 对外只能说"官方实现待独立复现"；论文的 48×、单卡数小时与任务表现不得写成 KineWorld 自有结果。

## 历史记录：KW-EXP-0005 原始计划（已执行完毕）

**多 horizon rollout error 埋点**（仍不训练）

现有 instrumentation 无法评估 horizon-conditioned 校准。最小成本改造是在 evaluator 中按 horizon = 1/2/4/6 latent steps 分别记录 predicted vs actual latent 误差，然后重跑现有 15 episodes 的闭环评估。完成后才具备验证 HAUWM 式 horizon-calibrated ensemble 的数据基础。

- `cem_elite_loss_std` vs predicted-actual latent MSE：segment `r=-0.0737`，episode `r=0.0499`。
- `cem_elite_loss_std` vs 实际状态改善：segment `r=-0.2604`，episode `r=0.4667`。
- 成功组平均 spread `0.017780`，失败组 `0.010641`，方向与简单 uncertainty threshold 假设相反。
- 判定：**NOT_SUPPORTED_AS_DIRECT_UNCERTAINTY_PROXY**；停止 spread threshold sweep，不直接用于 gating/abstention/compute allocation。
- 下一步回到具备明确 calibration target 的 uncertainty 方法，先做一手论文核实。


- episode-level CEM best loss 与实际状态改善：`r=-0.5307`。
- episode-level predicted goal latent distance 与实际状态改善：`r=0.2853`。
- episode-level predicted goal 与 actual goal latent distance：`r=0.7740`。
- 成功组平均累计状态改善：`119.8687`；失败组：`57.6051`。
- 解释：latent 内部目标排序尚可，但不能可靠排序物理状态改善；CEM loss 反而比 latent goal distance 更能解释本批样本的成功差异，但仍不构成因果证据。
- 决策：KW-EXP-0002c 支持“objective-grounding mismatch”作为最高价值候选，暂不进入 dynamics training。


- 3 episodes / 9 segments，`diagnostics.json` 全部生成且数值有限。
- `predicted_actual_latent_mse` 均值：0.03545；按 replan 位置：0.02754 → 0.05484 → 0.02398。
- 第 2 个 segment 的误差均值最高，但 n=3，不能确认稳定的 horizon 曲线。
- CEM best loss 与实际 segment 误差不同步，支持继续研究 planner-objective mismatch；尚不足以确认 DYNAMICS 主因。
- CEM best loss 与实际状态改善相关：`r=-0.2629`；predicted goal latent distance 与实际改善相关：`r=-0.3177`；但 predicted goal 与 actual goal latent distance 高相关：`r=0.8992`。
- 解释：模型 latent 内部目标距离有一致性，但 latent 目标距离对物理状态改善的预测力弱，**planner-objective 与 physical grounding 错配**成为当前最高价值假设；相关性为探索性统计，不是因果证明。
- 因此 D1 **pilot + full diagnostic 已完成，但结论未闭合**；当前最高 ROI 转为对 objective-mismatch / uncertainty 的诊断，不直接训练模型。


## KW-EXP-0002c L1 objective 对照结果

- 同一 3 episode 集、同 seed、同 CEM/nas 条件：L2 与 L1 均为 `1/3` 成功。
- L1 `ep_end_dist=74.713`，劣于 L2 `66.783`；L1 平均非首段状态改善略高（29.146 vs 26.794），但 episode 结果一胜一负混合。
- L1 predicted-vs-actual 与 actual-goal latent 相关 `0.2416`，低于 L2 的 `0.8460`。
- 判定：**L1 pilot INCONCLUSIVE_DO_NOT_EXTEND**。停止 L1/L2 盲目 sweep。


**目标**：把“DYNAMICS vs PLANNING”的归因置信度从 medium 提升到 high。

**最小设计**：
1. 在 evaluator/harness 记录每次 planning call：planning step、horizon、CEM best loss、elite mean/std、预测终端 latent 与执行后实际 latent 的 L2、per-step end_dist/success。
2. 用同一 checkpoint、同一 seed、同一 episode 集跑 closed-loop nas=2 的小样本诊断（先 3–5 episodes，含 persistent failure + ep_4 regression）。
3. 对比每个 10-step segment：规划时预测改善 vs 实际执行改善。

**判定逻辑**：
- 若 predicted latent 接近目标但实际状态不接近 → DYNAMICS_FAILURE 确认。
- 若 predicted latent 本身不接近但 CEM loss 低 → planner-objective mismatch / representation failure。
- 若 elite std 高且结果波动大 → UNCERTAINTY 方向优先级上升。

**Kill Criteria**：若诊断日志无法区分以上三类失败，停止该方向，转向不确定性论文阅读与第二方法对照。

## 排队规则

1. D1 诊断先行（低成本、直接提高下一轮决策质量）。
2. E2 uncertainty 需要一手论文输入；未读前不开实验。
3. E4 dynamics 只有在 D1 证明“模型预测误差主导”后才进入。
