# GAP_MATRIX.md — 能力缺口矩阵

> 版本 2026-09-02。状态：SOLVED / PARTIALLY SOLVED / OPEN / UNKNOWN。
> KineWorld 研究资源优先投：OPEN 与 PARTIALLY SOLVED 且高价值区域。

| # | 能力 | 状态 | best paper | best repo | best checkpoint | best benchmark | 单卡可行 |
|---|---|---|---|---|---|---|---|
| 1 | Perception | SOLVED（复用） | DINOv2, V-JEPA 2 | facebookresearch/dinov2 | ✅ 公开 | ImageNet/视频基准 | ✅（frozen 推理） |
| 2 | Representation | PARTIALLY SOLVED | V-JEPA 2, LeWM, R2-Dreamer | jepa-wms, leworldmodel | ✅ jepa_wm_pusht / LeWM Push-T | jepa-wms evals / LeWM eval | ✅ |
| 3 | Action-conditioned dynamics | PARTIALLY SOLVED | jepa-wms (2512.24497), LeWM (2603.19312), TD-MPC2 | jepa-wms, leworldmodel, tdmpc2 | ✅ | Push-T/MW/Wall | ✅（jepa-wms 已复现；LeWM 待同台） |
| 4 | Persistent belief | OPEN | RSSM（部分） | dreamerv3 | ✅ | — | ⚠️（需训练） |
| 5 | Stochastic belief | PARTIALLY SOLVED | DreamerV3, WIMLE | dreamerv3 | ✅ | DMC | ⚠️ |
| 6 | Uncertainty | PARTIALLY SOLVED | WIMLE, Horizon-Calibrated-Unc (ICLR'26) | dreamerv3（随机隐变量） | ✅ | HumanoidBench | ⚠️（JEPA 上 OPEN） |
| 7 | Object permanence | OPEN | Object-Centric WM (ICLR'26) | — | — | — | UNKNOWN |
| 8 | Partial observability | OPEN | POMDP 文献（经典） | — | — | — | UNKNOWN |
| 9 | Long-horizon prediction | PARTIALLY SOLVED | HaM-World, Dreamer 4 | — | — | Minecraft(D4) | ⚠️ |
| 10 | Planning | PARTIALLY SOLVED | jepa-wms, TD-MPC2, stable-worldmodel | jepa-wms CEM, stable-worldmodel CEM/MPC ✅ | ✅ | jepa-wms / SWM evals | ✅（jepa-wms 实测 37.6s/plan；SWM 待本机） |
| 11 | Active exploration | OPEN | Active Inference 文献 | — | — | — | UNKNOWN |
| 12 | Active experiment design | OPEN | Bayesian Experimental Design | — | — | — | UNKNOWN |
| 13 | Causal intervention | OPEN（C1 级有证据） | — | — | — | — | UNKNOWN |
| 14 | Counterfactual reasoning | OPEN | — | — | — | — | UNKNOWN |
| 15 | Online adaptation | OPEN | SysID（跨领域未吸收） | — | — | — | UNKNOWN |
| 16 | Memory | OPEN | — | — | — | — | UNKNOWN |
| 17 | Temporal abstraction | OPEN | LeCun H-JEPA（未实现@scale） | — | — | — | UNKNOWN |
| 18 | Adaptive compute | OPEN | —（1X 自曝需求） | — | — | — | ⚠️（依赖 #6 信号） |
| 19 | Goal-conditioned representation | PARTIALLY SOLVED | jepa-wms（goal_source=dset/exp） | jepa-wms ✅ | ✅ | pt/mw/wall evals | ✅（已复现） |
| 20 | Self model | UNKNOWN | — | — | — | — | UNKNOWN |
| 21 | Other-agent model | UNKNOWN | — | — | — | — | UNKNOWN |
| 22 | Hybrid representation | OPEN | PAN/GLP（未开源） | — | — | — | ❌（27B 级） |
| 23 | Failure recovery | **OPEN（朴素闭环已证伪）** | MPC 闭环（经典控制） | jepa-wms 原生支持 | ✅ | Push-T（本地 baseline） | ✅ KW-EXP-0001：succ 0.467→0.400，NOT_SUPPORTED |
| 24 | Planner routing | OPEN | — | — | — | — | ⚠️（依赖多 planner 先备） |

## 资源投放结论（2026-09-02，KW-EXP-0001 后）

- **#23 Failure recovery 仍为 OPEN，但朴素闭环路径已被证伪**：nas=2 闭环 recovery=0、regression=1、success_rate 0.467→0.400。
- **当前最高价值且立即可行：Replanning diagnostics** —— 记录 per-plan CEM loss、predicted-vs-actual latent error、per-step end_dist，区分 #3 动力学误差、#10 规划误差、#6 不确定性需求。
- 下一梯队：#6 Uncertainty on JEPA（先读 WIMLE + Horizon-Calibrated-Unc 一手）→ #3 Action-conditioned dynamics（仅当 diagnostics 证明值得训练）。
- 新增低成本对照：LeWM 官方 Push-T 权重仅 72.3 MB、MIT；待 KW-EXP-0006 完成后冻结同台协议并复现。先评权重，不下载 13.1 GB 训练集、不宣称论文结果属于 KineWorld。
- **Research infrastructure 不再视为 KineWorld 核心缺口**：stable-worldmodel 已公开统一 harness、CEM/MPC、环境和 FoV；只补其缺少的证据/风险/失败诊断层。
- 不投：#22（算力不可及）、#20/#21（UNKNOWN，无下手点）、#1（已 SOLVED，纯复用）、naive `num_act_stepped` sweep（已 kill）。
