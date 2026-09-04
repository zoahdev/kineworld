# KW-EXP-0001 — Closed-Loop Replanning on JEPA-WM Push-T

> 日期：2026-09-02。状态：**COMPLETE**。结论：**NOT_SUPPORTED（主指标）**。
> 证据等级：**E1（Internal Experiment）**。因果等级：**C2（单变量受控干预；单 seed、n=15，未独立复现）**。

## 1. 实验问题

在 JEPA-WM Push-T 官方评测协议下，保持模型、checkpoint、CEM 预算、seed、episode 集不变，仅提高重规划频率，是否能提升任务成功率？

## 2. Hypothesis

闭环重规划通过观测反馈校正开环漂移，因此在相同模型与相同 CEM 单次预算下，`num_act_stepped=2`（每 2 个 latent 步 = 10 个 env 步重规划）应使 success_rate 高于开环 baseline 0.467。

## 3. 单一科学变量

- **变量**：`planner.num_act_stepped`: `6 → 2`
- **保持不变**：checkpoint、DINOv2 frozen encoder、AdaLN predictor、horizon=6、CEM 10×100×10、seed=1、episodes 0–14、goal_source=dset、评测环境、硬件。
- **机制**：官方 `plan_evaluator.py` 原生 while-loop；无新算法代码。

## 4. Baseline

- Run：`pt_L2_cem_sourcedset_H6_nas6_ctxt2_ep15_red10x100_20260901T154959Z`
- 配置：`external/jepa-wms/configs/dump_online_evals/pt/pt_L2_cem_sourcedset_H6_nas6_ctxt2_ep15_red10x100.yaml`
- `num_act_stepped=6`（=horizon；每 episode 规划 1 次，全开环）
- 结果：success_rate=0.4666667（7/15），ep_end_dist=83.88669，reward=6.819867。

## 5. Treatment

- Run：`pt_L2_cem_sourcedset_H6_nas6_ctxt2_ep15_red10x100_nas2_20260901T162318Z`
- 配置：`external/jepa-wms/configs/dump_online_evals/pt/pt_L2_cem_sourcedset_H6_nas6_ctxt2_ep15_red10x100_nas2.yaml`
- `num_act_stepped=2`；配置中保留 `meta.official_num_act_stepped=6` 供审计。
- 实测：每 episode 规划 3 次（env step 0/10/20），共 45 次规划。

## 6. 预注册 Kill Criteria

来自 `docs/research/RESEARCH_ROI.md`：

> 同 seed 同 15 episodes 下，闭环 succ ≤ 0.467（无提升）→ NOT_SUPPORTED，记录 negative result，主归因修正为 DYNAMICS，回 FRONTIER_GRAPH 重排序。

## 7. 主结果

| 指标 | Baseline nas=6 | Closed-loop nas=2 | Delta | 判定 |
|---|---:|---:|---:|---|
| **success_rate** | **0.4667（7/15）** | **0.4000（6/15）** | **-0.0667（-1 episode）** | **Kill triggered** |
| episode_reward | 6.8199 | 7.1139 | +0.2941 | 次指标改善 |
| ep_end_dist | 83.8867 | 71.0204 | -12.8663（相对 -15.34%） | 次指标改善 |
| planning_calls | 15 | 45 | ×3 | 成本显著增加 |
| planning_latency_mean | 37.581s/call | 23.520s/call | -37.4%/call | horizon 递减导致 |
| planning_latency_p95 | 38.020s | 37.340s | 近似相同 | 首次规划主导 |
| wall_time | 620.80s | 1117.59s | +80.0% | 成本显著增加 |
| ep_time | 40.535s | 73.596s | +81.6% | 成本显著增加 |
| VRAM peak | 7095 MiB | 7343 MiB | +248 MiB（+3.5%） | 可接受 |

**主判定：success_rate 0.400 ≤ 0.467，按预注册标准判定为 NOT_SUPPORTED。**

## 8. Per-episode 翻转矩阵

Baseline 成功：ep_1, ep_4, ep_5, ep_6, ep_9, ep_12, ep_13。
Closed-loop 成功：ep_1, ep_5, ep_6, ep_9, ep_12, ep_13。

| Episode | Baseline | Closed-loop | 翻转类型 |
|---|---|---|---|
| ep_0 | fail | fail | persistent failure |
| ep_1 | success | success | persistent success |
| ep_2 | fail | fail | persistent failure |
| ep_3 | fail | fail | persistent failure |
| ep_4 | **success** | **fail** | **regression（开环胜→闭环败）** |
| ep_5 | success | success | persistent success |
| ep_6 | success | success | persistent success |
| ep_7 | fail | fail | persistent failure |
| ep_8 | fail | fail | persistent failure |
| ep_9 | success | success | persistent success |
| ep_10 | fail | fail | persistent failure |
| ep_11 | fail | fail | persistent failure |
| ep_12 | success | success | persistent success |
| ep_13 | success | success | persistent success |
| ep_14 | fail | fail | persistent failure |

- Recovery（开环败→闭环胜）：**0**
- Regression（开环胜→闭环败）：**1（ep_4）**
- 净效应：**-1 success**

## 9. 失败分析

1. **闭环没有救回任何 baseline 失败 episode**：8 个开环失败在闭环下全部仍失败。
2. **闭环引入 1 个回归 episode**：ep_4 从 success 变为 fail。
3. **end_dist 改善但 success 不改善**：闭环平均让终端状态更接近目标（83.89→71.02），但没有跨过 success threshold。解释空间包括：
   - 重规划能纠正一部分几何误差，但模型动力学误差仍阻止最终精确对齐；
   - 缩减 CEM（10×100×10）下单次规划质量有限，更多重规划只是更频繁地围绕有偏模型重新优化；
   - 后段 horizon 缩短降低每次规划成本，但也缩短可用前瞻，可能放大局部最优。
4. **ep_4 回归**提示闭环并非单调安全干预；在当前操作点，重规划可能破坏原本可行的开环轨迹。

## 10. 指标有效性说明

- `ep_total_emb_l2`：baseline=0.6196，closed-loop=0.0。**0.0 不是模型改善，而是官方指标路径在 nas<horizon 时失效。**
- 代码审计结论：`episode_plot_utils.compare_unrolled_plan_expert()` 用累积索引 `s` 切片每次 replan 的 `predicted_best_encs`；闭环第二次/第三次规划的 unroll 是相对当前观测重新开始，继续用全局 `s` 会导致空切片，`mse_loss(empty)` 产生 NaN；上层 `np.nansum` 将全 NaN 聚合为 0。
- 因此本实验不把 closed-loop `ep_total_emb_l2=0.0` 作为任何正向证据。

## 11. Compute / Hardware

- GPU：NVIDIA GeForce RTX 5070 Ti Laptop GPU，12,227 MiB。
- VRAM peak：7343 MiB（delta 6377 MiB），未触墙。
- Closed-loop 规划延迟结构：首次约 37.3s，第二次约 23.5s，第三次约 9.8s（horizon 随 `plan_steps_left` 缩短）。
- 总规划时间近似：45 × 23.52s ≈ 1058s，占总 wall time 1117.6s 的约 94.6%。

## 12. Reproducibility / Provenance

- jepa-wms commit：`13cf1d9c7e476f53c17714d2e0f1dc239a883ce0`
- checkpoint：`checkpoints/jepa_wm_pusht.pth.tar`（211,639,615 bytes；SHA-256 见 `verification/artifacts/sha256_2026-09-02.txt`）
- metrics：
  - baseline：`results/runs/pt_L2_cem_sourcedset_H6_nas6_ctxt2_ep15_red10x100_20260901T154959Z/metrics.json`
  - closed-loop：`results/runs/pt_L2_cem_sourcedset_H6_nas6_ctxt2_ep15_red10x100_nas2_20260901T162318Z/metrics.json`
- 官方 eval.csv：
  - `results/phase0_jepa_wm_pusht/simu_env_planning/phase0/pt_L2_cem_sourcedset_H6_nas6_ctxt2_ep15_red10x100/eval.csv`
  - `results/phase0_jepa_wm_pusht/simu_env_planning/phase0/pt_L2_cem_sourcedset_H6_nas6_ctxt2_ep15_red10x100_nas2/eval.csv`

## 13. Limitations

- 单 seed、15 episodes，统计功效不足；本结论是该操作点下的工程/科学证据，不是普遍定理。
- CEM 已缩减（10×100×10，而非官方 30×300×10）；结果不能直接外推到完整 CEM。
- 只测试 `num_act_stepped=2`；未扫描 nas=1/3/4/5。
- 没有 per-step latent prediction error 分解；当前动力学归因置信度仍为 medium。

## 14. Conclusion

- **Primary claim：NOT_SUPPORTED。** 在 JEPA-WM Push-T、缩减 CEM、seed=1、episodes 0–14 条件下，将重规划频率从 1 次/episode 提高到 3 次/episode 未提升 success_rate，反而从 0.467 降至 0.400。
- **Secondary observation：PARTIALLY_SUPPORTED。** ep_end_dist 与 reward 改善，说明闭环可能改善平均接近程度，但不转化为成功率。
- **Research implication：** “开环漂移是当前最大失败源”的工作假设被证伪；当前瓶颈应向 DYNAMICS / planner-objective mismatch / uncertainty-aware replanning 迁移。

## 15. Next Action

1. 更新 failure attribution：PLANNING-frequency 假设降级；DYNAMICS/模型误差升为当前主瓶颈。
2. 更新 FRONTIER_GRAPH / RESEARCH_ROI：不再继续 naive replanning sweep。
3. 下一轮最高 ROI：优先建立 per-step replanning diagnostics 或 JEPA uncertainty signal，区分“模型预测错”与“CEM 优化错”。
