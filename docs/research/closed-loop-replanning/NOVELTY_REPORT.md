# NOVELTY_REPORT — KW-EXP-0001 闭环重规划（Closed-Loop Replanning on JEPA-WM Push-T）

> 日期：2026-09-02。宪法 §13 ZERO NOVELTY TEST：任何 idea 被称为"原创"前必须执行。

## 0. Paper + GitHub First Survey（§58 STEP 9-10）

| 项 | 状态 |
|---|---|
| Paper | jepa-wms (arXiv 2512.24497) — 官方消融论文 |
| Official GitHub | facebookresearch/jepa-wms ✅（本地 external/jepa-wms, commit 13cf1d9） |
| Checkpoint | `jepa_wm_pusht.pth.tar`（官方，211.6MB）✅ 已就位 |
| Dataset | `pusht_noise`（OSF 镜像）✅ 已就位 |
| Benchmark / Evaluation | 官方 `evals/simu_env_planning` ✅ 已跑通 |
| Planner | 官方 CEM ✅（含原生闭环 while-loop） |
| Encoder | DINOv2 ViT-S/14 frozen ✅ |
| License | CC BY-NC 4.0（研究可用，商业 Core 禁入） |

**复用结论：全部组件复用官方，KineWorld 新增 = 实验设计 + 测量 + 证据，零新算法代码。**

## 1. 哪些部分已有？

- **闭环 MPC / receding horizon 本身**：控制理论数十年的标准实践（MPC 教科书）；TD-MPC2 在线 MPPI 每步重规划；jepa-wms 官方代码已原生支持（`plan_evaluator.py` while-loop + `num_act_stepped` 参数），MetaWorld 官方配置用 nas=3（半闭环）、RoboCasa 用 nas=1（全闭环）。
- **V-JEPA 2-AC 真机规划**：Meta 已展示潜空间规划有效性。

## 2. 有没有同样概念只是换名字？

有。这就是标准 MPC。**我们不声称发明闭环 MPC。**

## 3-6. 是新算法 / 新组合 / 新 benchmark / 新 empirical finding？

- 新算法：**否**
- 新组合：**否**（官方代码已支持）
- 新 benchmark：**否**
- **新 empirical finding：是（候选）**——jepa-wms 官方 Push-T 配置默认 `num_act_stepped=6`（全开环），官方论文未报告闭环/开环消融对 Push-T 成功率的影响；官方代码注释仅对 DROID 说明"cannot replan"。**JEPA-WM 在 Push-T 上闭环重规划的收益/成本曲线，公开记录中未见到量化数据。**本实验产出该量化对比（同模型、同 seed、同 episode 集、同 CEM 预算，唯一变量 = 重规划频率）。

## 7. Prior art

- MPC / receding horizon control（经典控制理论）
- TD-MPC2（nicklashansen/tdmpc2）：MPPI 每步重规划，状态空间
- jepa-wms 官方 MW（nas=3）/RoboCasa（nas=1）配置：闭环机制存在于官方仓库但 Push-T 未启用、未消融
- V-JEPA 2-AC（Meta 2025-06）：潜空间规划，真机

## 8. 真正新增内容（诚实口径）

1. **量化证据**：JEPA-WM Push-T 开环 vs 闭环（nas=6→2）在相同条件下的成功率/延迟/成本对比——公开记录未见此消融。
2. **失败归因因果检验**：用闭环干预验证 Phase 0 失败分布的主归因（PLANNING vs DYNAMICS），产出"失败归因→实验干预→归因修正"的方法论示范。
3. 若 SUPPORTED：closed-loop recovery 数据（recovery rate、extra planning cost、per-episode 翻转记录）作为后续 Failure Recovery Runtime 产品方向的第一块证据。

**禁止表述**：不得声称"KineWorld 发明了闭环规划"。允许表述："KineWorld 量化验证了闭环重规划对官方 JEPA-WM Push-T baseline 的影响（E1 证据）"。

## 9. 实验设计（§57 标准）

- Hypothesis：闭环重规划（每 10 env 步一次）通过观测反馈校正消除开环漂移，在相同 CEM 预算下提升 Push-T 成功率（>0.467@n15）。
- Baseline：ep15_red10x100（nas=6 开环，succ=0.467，seed=1，episodes 0-14）。
- Proposed Change：num_act_stepped 6→2（唯一变量；horizon=6、CEM 10×100×10、seed、episode 集全部不变）。
- Metrics：success_rate（主）、ep_end_dist、emb_l2、planning_calls、planning_latency、wall_time、VRAM、per-episode 翻转记录（开环败→闭环胜 / 开环胜→闭环败）。
- Compute Budget：~25min 单卡（实测首次规划 37.3s，后续随剩余步数递减）。
- Kill Criteria：闭环 succ ≤ 0.467 → NOT_SUPPORTED。
