# KW-CLAIM-0002 — `ep_total_emb_l2=0.0` under closed-loop is a metric artifact

- 日期：2026-09-02
- 关联实验：KW-EXP-0001
- Claim：KW-EXP-0001 closed-loop run 中的 `ep_total_emb_l2=0.0` 不能解释为潜空间预测改善；它是官方指标路径在 `num_act_stepped < horizon` 时的聚合伪影。
- 结论：**SUPPORTED（代码审计 + 运行观测）**
- 证据等级：**E1**
- 因果等级：**C1**

## 证据

1. Baseline（nas=6）`ep_total_emb_l2=0.6196`；closed-loop（nas=2）输出 `0.0`。
2. `plan_evaluator.py` 将每次 replan 的 `predicted_best_encs_over_iterations` 追加到 episode 级列表。
3. `episode_plot_utils.compare_unrolled_plan_expert()` 使用跨 planning call 累积的索引 `s` 切片每次 plan 的 embedding；第二次/第三次 replan 的 unroll 是相对当前观测重新开始，全局 `s` 会导致空切片。
4. 空切片的 `mse_loss` 产生 NaN；上层 `np.nansum` 将全 NaN 聚合为 0。

## 影响

- KW-EXP-0001 不把 closed-loop `ep_total_emb_l2=0.0` 作为任何正向证据。
- 后续若要做闭环潜空间误差分析，必须先修复该指标路径或用独立 harness 重新计算。
