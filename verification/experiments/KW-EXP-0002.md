# KW-EXP-0002 — Replanning Error Decomposition

> 日期：2026-09-02。状态：COMPLETE（pilot + full diagnostic）。结论：**DIAGNOSTIC_SIGNAL_OBSERVED；不构成性能支持证据**。
> 证据等级：E1。因果等级：C1（诊断记录，不是性能干预）。

## 1. 目的

KW-EXP-0001 证明 `num_act_stepped=2` 没有恢复失败，但其官方 `ep_total_emb_l2=0.0` 是无效聚合值。KW-EXP-0002 增加旁路 instrumentation，记录每次 replan 的 CEM 收敛统计，以及“执行段末端预测 latent”与“真实观测编码 latent”的 MSE，避免把未执行的 horizon 尾部拿来比较。

## 2. 设计

- Checkpoint：同 `jepa_wm_pusht.pth.tar`
- jepa-wms commit：`13cf1d9c7e476f53c17714d2e0f1dc239a883ce0`
- seed：1
- episodes：0–2（3 episodes）
- planner：CEM，10 iterations × 100 samples × 10 elites
- horizon：6
- 唯一运行配置：`num_act_stepped=2`
- 代码变化：仅诊断旁路；不改变 action、objective、success 判定或环境状态

## 3. Pilot 运行结果

- Run：`pt_L2_cem_sourcedset_H6_nas6_ctxt2_ep3_d1diag2_20260901T172636Z`
- exit code：0
- success：1/3 = 0.3333
- planning calls：9（3/episode）
- mean planning latency：23.477s/call
- VRAM peak：7747 MiB
- diagnostics：每个 episode 生成 3 条记录，共 9 条，全部含有限值

## 4. Full diagnostic 运行结果

- Run：`pt_L2_cem_sourcedset_H6_nas6_ctxt2_ep15_d1full_20260901T173624Z`
- exit code：0
- success：6/15 = 0.4000（与 KW-EXP-0001 完全复现）
- planning calls：45（3/episode）
- mean planning latency：23.442s/call
- VRAM peak：7825 MiB
- diagnostics：15 episodes / 45 segments，全部含有限值

## 5. Full diagnostic 汇总

| replan | env step | horizon | mean predicted-vs-actual latent MSE | mean actual state distance |
|---:|---:|---:|---:|---:|
| 0 | 0 | 6 | 0.033514 | 178.238 |
| 1 | 10 | 4 | 0.042226 | 172.981 |
| 2 | 20 | 2 | 0.035053 | 95.728 |

- 全 45 个 segment 的 MSE 均为有限值，均值为 `0.0370`（按 45 段平均）。
- 成功 episode（ep_1/5/6/9/12/13）MSE 均值：`0.031523`。
- 失败 episode（ep_0/2/3/4/7/8/10/11/14）MSE 均值：`0.040537`。
- 失败组略高于成功组，但当前没有独立 per-episode baseline 诊断，且差异不足以作因果结论。
- `ep_4`（KW-EXP-0001 中唯一 regression）三段 MSE：`0.043761 / 0.029257 / 0.017912`，没有呈现持续升高。

## 6. Pilot 诊断汇总（保留）

| replan | env step | horizon | mean predicted-vs-actual latent MSE | mean actual state distance |
|---:|---:|---:|---:|---:|
| 0 | 0 | 6 | 0.027541 | 136.670 |
| 1 | 10 | 4 | 0.054840 | 140.352 |
| 2 | 20 | 2 | 0.023976 | 83.082 |

全 9 个 pilot segment 的 latent MSE 均值：`0.035452`。


## 7. KW-EXP-0002c Objective-Grounding 对照

基于 full diagnostic 的 15 episode / 45 segment 数据计算：

| 关系 | Segment-level r | Episode-level r |
|---|---:|---:|
| CEM best loss vs 实际状态改善 | -0.2629 | **-0.5307** |
| Predicted goal latent distance vs 实际状态改善 | -0.3177 | **0.2853** |
| Predicted goal vs actual goal latent distance | 0.8992 | **0.7740** |

- 成功组平均累计状态改善：`119.8687`；失败组：`57.6051`。
- latent 内部目标距离保持较高一致性，但其与物理状态改善的 episode-level 相关很弱。
- CEM best loss 与状态改善在本批样本中相关更强，但这仍是单 seed 的探索性统计，不代表泛化或因果关系。
- 当前最合理的工作假设：**latent objective 与物理 grounding 错配**，需要低成本 objective 对照验证。


1. 诊断链路现在有效：每次 replan 都记录了 `plan_index/env_step/steps_left/plan_horizon/num_act_stepped/CEM loss/actual_success/actual_state_dist`，并对齐到实际执行 segment 的末端。
2. Full diagnostic 中第二个 segment 的 MSE 均值（0.042226）高于第一、第三段，但差异很小，且没有持续随 horizon 增长的单调规律。
3. 失败组 MSE 略高于成功组（0.040537 vs 0.031523），但当前没有独立 per-episode baseline 诊断，不能确认动力学因果关系。
4. `ep_4` regression 的 MSE 从 0.043761 降至 0.017912，没有呈现“误差持续升高导致回归”的简单模式。
5. CEM loss 与实际 segment 误差不同步，支持继续研究 planner-objective mismatch；当前不能把该候选信号升级为结论。
6. `ep_total_emb_l2=0.0` 的原官方聚合警示仍成立；D1 独立字段避免了该问题。
7. 新增目标错配诊断：`predicted_goal_latent_mse` 与 `actual_goal_latent_mse` 的 segment-level 相关为 `r=0.8992`，但 `predicted_goal_latent_mse` 与实际状态距离改善相关仅 `r=-0.3177`；CEM best loss 与实际改善相关 `r=-0.2629`。

## 7. 结论

- instrumentation：**SUPPORTED**（技术链路通过）。
- “动力学误差是唯一主因”：**NOT_ESTABLISHED**，full diagnostic 未给出简单单调证据。
- planner-objective mismatch：**候选信号**，CEM loss 与真实执行误差不同步。
- uncertainty：当前 elite std 可记录，但尚未校准，不能直接作为 uncertainty model。

## 8. 下一步

当前最高 ROI 转为 **KW-EXP-0002b：objective-mismatch / uncertainty diagnostics**，优先加入 predicted goal distance、actual state-distance delta 和 segment-level correlation，再决定是否需要动力学训练。暂不进行 expensive dynamics training 或 naive nas sweep。