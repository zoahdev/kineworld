# KW-EXP-0003 — CEM Elite Spread Uncertainty Proxy Audit

> 日期：2026-09-02。状态：COMPLETE。结论：**NOT_SUPPORTED_AS_DIRECT_UNCERTAINTY_PROXY**。证据等级 E1。

## 1. 问题

评估 CEM elite loss standard deviation（`cem_elite_loss_std`）能否作为不确定性代理，用于预测 latent prediction error、实际状态改善或最终成功。

## 2. 数据与方法

- 数据来源：KW-EXP-0002b full diagnostic
- 规模：15 episodes / 45 segments
- checkpoint、seed、环境和 planner：保持原实验条件
- 未运行新模型、未改变 action policy
- Segment-level：使用 30 个非首段 segment 的 `actual_state_dist_delta`；预测误差和目标距离使用全部 45 segments
- Episode-level：每 episode 对 3 个 segment 求 spread/MSE/goal distance，累计状态改善为 3 段 delta 之和

## 3. 结果

| 关系 | Segment-level r | Episode-level r |
|---|---:|---:|
| elite spread vs predicted-actual latent MSE | **-0.0737** | **0.0499** |
| elite spread vs 实际状态改善 | **-0.2604** | **0.4667** |
| elite spread vs predicted goal distance | 0.3040 | 0.0943 |
| elite spread vs actual goal distance | 0.2765 | 0.1792 |

成功/失败 episode 分组：

| 分组 | 平均 elite spread | 平均 latent MSE | 平均累计状态改善 |
|---|---:|---:|---:|
| success（6） | **0.017780** | 0.031523 | 119.8687 |
| failure（9） | 0.010641 | 0.040537 | 57.6051 |

## 4. 判定

1. `cem_elite_loss_std` 与实际 latent prediction error 几乎不相关，不能作为当前动力学误差的不确定性估计。
2. 成功组 spread 反而高于失败组，方向与简单“spread 越高越不可靠”假设相反。
3. Episode-level spread 与状态改善的正相关不能解释为 uncertainty calibration；更可能反映 CEM 搜索在可成功轨迹上的多样性或任务难度混杂。
4. **不支持将 elite spread 直接用于 replan gating、abstention 或 compute allocation。**

## 5. 研究含义

当前 JEPA-WM 缺少可校准 uncertainty 的结论得到强化：CEM 内部候选分布的 spread 不是可直接复用的 epistemic/aleatoric uncertainty。下一步若研究 uncertainty，应回到有明确 calibration target 的方法（WIMLE、horizon-calibrated uncertainty），而不是继续调 spread threshold。

## 6. 下一步

- 停止 `cem_elite_loss_std` threshold sweep。
- 先阅读并核实 WIMLE 与 Horizon-Calibrated Uncertainty 一手材料。
- 若继续做工程实验，优先选择有物理状态标签的 calibration protocol，并预注册 calibration error / AUROC / selective risk 指标。
