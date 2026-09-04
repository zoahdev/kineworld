# KW-EXP-0002c — Objective-Grounding 对照

> 日期：2026-09-02。结论：**L1 pilot INCONCLUSIVE / 不扩展**。证据等级 E1。

## 1. 设计

在同一 Push-T checkpoint、seed=1、episodes 0-2、CEM 10×100×10、闭环 `num_act_stepped=2` 条件下，仅将规划 objective 从官方 `L2` 改为 `L1`。模型、环境、动作执行和诊断 instrumentation 不变。

- L2：`pt_L2_cem_sourcedset_H6_nas6_ctxt2_ep3_d1obj3`
- L1：`pt_L1_cem_sourcedset_H6_nas6_ctxt2_ep3_objl1pilot_20260901T183615Z`
- 两者均 exit=0，planning calls=9。

## 2. 结果

| 指标 | L2 | L1 | 结论 |
|---|---:|---:|---|
| success | 1/3 | 1/3 | 无提升 |
| ep_end_dist | 66.783 | 74.713 | L1 更差 |
| 平均实际状态改善（非首段） | 26.794 | 29.146 | L1 略高，但样本太小 |
| predicted_actual latent MSE | 0.03545 | 0.03145 | 不足以解释性能 |
| predicted goal latent MSE | 0.10921 | 0.08498 | L1 更低 |
| actual goal latent MSE | 0.11836 | 0.08864 | L1 更低 |
| pred-vs-actual / actual-goal 相关 | 0.8460 | 0.2416 | L1 预测一致性更弱 |

Per-episode 平均 state distance：

| Episode | L2 | L1 |
|---|---:|---:|
| ep_0 | 155.01 | 114.46 |
| ep_1 | 64.54 | 100.25 |
| ep_2 | 140.55 | 155.21 |

L1 在 ep_0 改善、但在 ep_1/ep_2 变差；未显示稳定优势。

## 3. 判定

- 不扩展到 15 episodes：L1 pilot 没有成功率优势，且最终 `ep_end_dist` 更差。
- 不把 L1 的较低 latent objective 值解释为物理性能提升。
- 该对照支持“objective 数值尺度/排序与物理结果可能错配”，但不能证明 L1 是更优 objective。

## 4. 下一步

停止 L1/L2 盲目 objective sweep。下一轮优先建立 uncertainty proxy 与 abstention/gating 诊断，或设计带物理状态 grounding 的 objective；任何性能干预前先预注册单一变量与 kill criteria。
