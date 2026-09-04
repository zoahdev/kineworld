# KW-CLAIM-0001 — Closed-loop replanning improves JEPA-WM Push-T success

- 日期：2026-09-02
- 关联实验：KW-EXP-0001
- Claim：在 JEPA-WM Push-T 上，将官方 evaluator 的重规划频率从每 episode 1 次（`num_act_stepped=6`）提高到每 episode 3 次（`num_act_stepped=2`），可在相同模型、seed、episode 集和 CEM 单次预算下提升 success_rate。
- 结论：**NOT_SUPPORTED**
- 证据等级：**E1**
- 因果等级：**C2**（单变量受控干预；单 seed、n=15，未独立复现）

## 证据

| 指标 | Baseline | Closed-loop |
|---|---:|---:|
| success_rate | 0.4667（7/15） | 0.4000（6/15） |
| ep_end_dist | 83.8867 | 71.0204 |
| episode_reward | 6.8199 | 7.1139 |
| planning_calls | 15 | 45 |
| wall_time | 620.80s | 1117.59s |

- Recovery：0 episode
- Regression：1 episode（ep_4）
- Kill criteria：closed-loop succ ≤ 0.467 → triggered

## 允许表述

“KineWorld 在官方 JEPA-WM Push-T baseline 上量化验证：在当前缩减 CEM 操作点，朴素提高重规划频率没有提升 success_rate；success_rate 0.467→0.400，但 ep_end_dist 83.89→71.02。”

## 禁止表述

- 禁止声称闭环 MPC 是 KineWorld 发明。
- 禁止声称闭环重规划对 JEPA-WM Push-T 普遍无效；本结论只覆盖当前模型、缩减 CEM、seed=1、15 episodes、nas=2 操作点。
