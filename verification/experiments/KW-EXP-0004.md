# KW-EXP-0004 — Horizon-Conditioned Uncertainty Calibration Baseline

- 日期：2026-09-02
- 类型：诊断 / 校准基线（**无训练、无 GPU、无新策略执行**）
- 前置：KW-EXP-0003（`elite_loss_std` 作为 direct proxy 已被否定）
- 输入数据：`pt_L2_cem_sourcedset_H6_nas6_ctxt2_ep15_d1objfull` 的 15 episodes / 45 segments
- 可复现脚本：`verification/scripts/kw_exp_0004_calibration.py`（seed=20260902）

## 判定

**`CAPABILITY_GAP_CONFIRMED_MARGINAL_BASELINE_ONLY`**

当前冻结 JEPA 世界模型**没有原生 per-sample 不确定性输出**；唯一被验证可用的不确定性是**边缘（constant）保形带**。`elite_loss_std` 归一化带被严格支配，不得用于 gating / abstention / compute allocation。同时，现有数据**无法**建立 horizon-conditioned 曲线（结构性缺口，非发现）。

---

## Q1 — 能力审计（代码层面，可核验）

| 检查项 | 结果 |
|---|---|
| unroll 路径 | `agent._predicted_best_encs_over_iterations[-1]` → 单次 CEM 最佳候选确定性展开 |
| `dropout` / `.sample()` / `.rsample()` | `app/vjepa_wm/modelcustom/simu_env_planning/` 全目录 **0 命中** |
| `torch.randn` / `normal_(` 噪声注入 | 全目录 **0 命中** |
| ensemble / 多模态采样头 | 不存在 |
| 结论 | **确定性点预测，无方差输出** |

这意味着：任何"uncertainty"都必须外部构造。伪造方差等于伪造证据，因此本实验只评估**可证伪的校准协议**。

## Q2 — 保形校准基线

- 目标量：`e = predicted_actual_latent_mse`（规划终端 latent 与执行后真实 latent 的 MSE）
- 协议：leave-one-episode-out split conformal（15 折，每折留出 3 segments），上界预测带
- 池化误差：mean `0.036931`，std `0.027270`，min `0.010097`，max `0.141065`，n=45

### α = 0.10（目标覆盖 0.90）

| 分数 | 覆盖 | 平均上界（效率） | 成功段覆盖 | 失败段覆盖 |
|---|---:|---:|---:|---:|
| **const（边缘带）** | **0.9111** | **0.091499** | 0.8333 | 0.9231 |
| spread 归一化 | 0.9333 | 0.239001 | 0.6667 | 0.9744 |
| cem_best_loss 归一化 | 0.9111 | 0.112227 | 0.6667 | 0.9487 |

### α = 0.20（目标覆盖 0.80）

| 分数 | 覆盖 | 平均上界 | 成功段覆盖 | 失败段覆盖 |
|---|---:|---:|---:|---:|
| **const（边缘带）** | **0.8222** | **0.043719** | 0.8333 | 0.8205 |
| spread 归一化 | 0.8222 | 0.180391 | 0.6667 | 0.8462 |
| cem_best_loss 归一化 | 0.8222 | 0.068244 | 0.3333 | 0.8974 |

### 解读

1. **const 带是唯一被验证有效的基线**：α=0.10 实测 0.9111、α=0.20 实测 0.8222，均达到目标覆盖。这是本项目第一个带有效性保证的不确定性基线。
2. **spread 归一化被严格支配**：同等覆盖（0.8222）下平均上界 `0.180391`，是 const 带 `0.043719` 的 **4.13 倍**；α=0.10 下为 **2.61 倍**。更宽却无更高覆盖 = 纯效率损失。
3. **条件校准失效**：spread 带在成功段覆盖 0.6667、失败段 0.9744，差 30.8 个百分点；const 带 α=0.20 下为 0.8333 / 0.8205，基本平衡。spread 不是"保守"，是**错配**。
4. **不确定性方法的可量化价值上限**：const 带 90% 上界 `0.091499` = 池化均值 `0.036931` 的 **2.48 倍**。任何未来条件不确定性方法若能收敛到接近 oracle，效率最多可提升约 2.5 倍。这是 WIMLE / HAUWM 类方法的具体验收门槛。

## Q3 — Selective risk / abstention

| 弃权比例 | k | spread 策略残留风险 | 随机策略残留风险 | p 值 |
|---|---:|---:|---:|---:|
| 20% | 9 | 0.038021 | 0.036962 | 0.6723 |
| 40% | 18 | 0.035930 | 0.036982 | 0.3815 |

- 20% 弃权：spread 策略**劣于随机**（67.2% 的随机子集取得更低或相等风险）。
- 40% 弃权：点估计略优，但 p=0.3815，与随机无区分。
- 结论：**基于 spread 的弃权不带来可辨识收益**，不得作为 abstention 机制。

## Q4 — Horizon 分层：结构性缺口

按 replan 位置（segment index）分层的误差：

| 位置 | n | mean | std | bootstrap 95% CI |
|---|---:|---:|---:|---|
| segment_0 | 15 | 0.033514 | 0.031253 | [0.022121, 0.050836] |
| segment_1 | 15 | 0.042226 | 0.025843 | [0.030324, 0.055745] |
| segment_2 | 15 | 0.035054 | 0.025421 | [0.024484, 0.049217] |

- 三组 CI 高度重叠，**无显著位置效应**。
- **关键限制**：这不是 horizon 曲线。全部 45 个 segment 共享同一预测 horizon（`num_act_stepped=2` latent steps = 10 env steps，`pred_time = min(2, T-1) = 2`）。位置只是 episode 内的执行次序。
- 因此 **horizon-conditioned calibration 在当前 instrumentation 下不可实现**，需新增多 horizon 埋点才能评估。

---

## 结论与约束

1. **不做**：spread 归一化带、spread 阈值 gating、spread 弃权、spread compute allocation。
2. **可用**：const 保形带作为边缘不确定性基线，仅保证边际覆盖，不保证条件覆盖。
3. **能力缺口**：JEPA 无原生不确定性。要获得条件不确定性，必须引入 ensemble / probabilistic head / IMLE 采样，即进入训练或结构改造。
4. **数据缺口**：无多 horizon 误差数据，horizon-conditioned 校准评估暂不可做。
5. **验收门槛**：任何新方法至少要在 90% 目标覆盖下把平均上界显著压低到 `0.091499` 以下，并同时把成功/失败段覆盖差收敛到 10 个百分点以内。

## 下一步

1. 优先补齐 **multi-horizon rollout error 埋点**（horizon = 1/2/4/6 latent steps），这是使 horizon-conditioned 校准可评估的最小成本改造，仍不需要训练。
2. 之后才评估是否值得引入 HAUWM 式 horizon-calibrated ensemble。
3. 在此之前不进入动力学训练。

## 证据等级

- E1（探索性统计）。n=45 segments / 15 episodes，单 seed、单 checkpoint、单任务（Push-T）。
- 保形覆盖保证在 **episode 间可交换**假设下成立；segment 内部自相关由 leave-one-episode-out 折缓解，但未完全消除。
- 成功段仅 6 个，条件覆盖的成功组估计方差很大，仅作方向性参考。
