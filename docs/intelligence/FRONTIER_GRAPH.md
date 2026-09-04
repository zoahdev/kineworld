# FRONTIER_GRAPH.md — 研究问题依赖图

> 版本 2026-09-02。表示研究问题之间的依赖关系，非简单表格。每个节点记录：最强已知方法 / 最佳公开实现 / 证据强度 / 算力成本 / 依赖 / 未解决失败 / KineWorld 实验。

## 主链

```
Perception
  ↓
Representation
  ↓
Belief ──────────────→ Uncertainty
  ↓                        │
Dynamics ────────────→ Causality
  ↓
Imagination ─────────→ Hierarchy
  ↓
Planning ────────────→ Adaptive Compute
  ↓
Action ──────────────→ Investigation
  ↓
Prediction Error
  ↓
Adaptation ──────────→ Memory
```

---

## 节点详情

### N1 Perception
- 最强方法：冻结大规模 SSL 视觉编码器（DINOv2 / V-JEPA 2 / V-JEPA 2.1 dense）
- 最佳公开实现：facebookresearch/dinov2 ✅、V-JEPA 2 权重 ✅
- 证据强度：强（多基准 SOTA）
- 算力成本：推理极低（ViT-S/14 单帧 ms 级）；训练不可及 → 不重训
- 依赖：无
- 未解决失败：单目 3D grounding 弱（1X 自曝）；domain shift 下退化
- KineWorld：Phase 0 已用 DINOv2 frozen ✅

### N2 Representation
- 最强方法：patch 级潜空间（V-JEPA 系）vs 离散随机隐变量（RSSM）vs 冗余约减（R2-Dreamer）
- 最佳公开实现：jepa-wms（AdaLN predictor over DINOv2 patches）✅ 本地已复现
- 证据强度：中强（规划有效，但 collapse 避免仍启发式）
- 算力成本：编码器 frozen 时近零
- 依赖：N1
- 未解决失败：表示崩塌风险；latent-only 缺乏 grounding（PAN/GLP 批评）
- KineWorld：Phase 0 量化（emb_l2=0.620 vs expert）

### N3 Belief
- 最强方法：ctxt_window 滑窗（jepa-wms, T=2）；RSSM 后验递推
- 最佳公开实现：jepa-wms ✅ / dreamerv3 ✅
- 证据强度：中——**当前 JEPA-WM belief 是确定性的、无概率**
- 依赖：N2
- 未解决失败：无原生不确定性；遮挡下信念维持未验证
- KineWorld：候选实验（belief under occlusion）— 待 ROI 排序

### N3b Uncertainty（分支）
- 最强方法：RSSM 随机隐变量；WIMLE（IMLE 集成 + 逆方差加权，ICLR'26）；Horizon-Calibrated Uncertainty 预训练（ICLR'26）
- 最佳公开实现：dreamerv3 ✅；WIMLE 待回溯一手
- 证据强度：中
- 依赖：N3
- 未解决失败：**JEPA 无原生不确定性** —— 前沿公认的洞
- KineWorld：候选实验（uncertainty calibration via ensemble/ dropout on predictor）

### N4 Dynamics
- 最强方法：动作条件化潜空间预测器（V-JEPA 2-AC / jepa-wms AdaLN）；TD-MPC2 单步 MLP
- 最佳公开实现：jepa-wms ✅（本地 checkpoint `jepa_wm_pusht.pth.tar`）
- 证据强度：强（Push-T 规划有效，Phase 0 实测）
- 算力成本：推理 = CEM 每次规划 37.6s（10×100×H6，本机）
- 依赖：N2
- 未解决失败：**长 horizon 误差复合**（Phase 0 实测 0.70→0.86@50 步；HaM-World 用几何约束部分缓解）；KW-EXP-0001 排除“单纯规划频率不足”后成为当前主瓶颈候选
- KineWorld：Phase 0 rollout-error 曲线 ✅（results/rollout_err.json）；下一步需要 per-step replanning diagnostics 区分 DYNAMICS vs planner-objective mismatch

### N4b Causality（分支）
- 现状：全行业 C1（action-conditioned prediction）级别；C2+（干预泛化）少证据
- KineWorld：causal claim 必须标级（C0-C5，见宪法配套规则）

### N5 Imagination
- 最强方法：潜空间开环 unroll（jepa-wms）；Dreamer imagination rollout
- 依赖：N4
- 未解决失败：长程退化（同 N4）；层级想象未实现
- KineWorld：暂不作为独立变量

### N6 Planning
- 最强方法：CEM/MPPI（jepa-wms、TD-MPC2）；闭环 MPC（receding horizon）
- 最佳公开实现：jepa-wms CEM planner ✅；**官方 evaluator 原生支持 num_act_stepped<horizon 闭环**
- 证据强度：强（但 Push-T 官方默认 num_act_stepped=6 全开环——**闭环收益在 JEPA-WM Push-T 上未被官方量化**）
- 算力成本：闭环成本 = 规划次数 × 37.6s
- 依赖：N5
- 未解决失败：开环执行漂移假设已被 KW-EXP-0001 证伪（recovery=0，succ 0.467→0.400）；朴素闭环不能恢复失败
- **KineWorld：KW-EXP-0001（闭环重规划）= NOT_SUPPORTED**；不再做 naive nas sweep，转向 N4/N3b 诊断

### N6b Adaptive Compute（分支）
- 现状：1X 自曝 11s/rollout 需"停下来想"；无成熟公开方案
- 依赖：N6 + N3b
- KineWorld：候选实验（按不确定性调节 CEM 预算）— 需先有不确定性信号

### N7 Action
- 现状：CEM 输出 mean 直接执行；action_skip/frameskip 机制官方已实现
- KineWorld：暂不作为变量

### N8 Prediction Error → N9 Adaptation
- 最强方法：RSSM 在线更新；SysID 在线参数辨识（跨领域，未被吸收）
- 证据强度：弱（JEPA 部署后学习 = 前沿开放问题）
- KineWorld：候选实验（prediction-error-triggered replan/adapt）— 需 N3b 先决

### N9b Memory（分支）
- 现状：ctxt_window=2 短窗；长时记忆在 WM 中基本未解决
- KineWorld：PLANNED，不进入当前实验

---

## 当前 KineWorld 位置（2026-09-02）

```
N1 ✅(复用) → N2 ✅(复现) → N3 ⚠️(确定性) → N4 ⚠️(当前主瓶颈候选) → N5 ✅(附带)
→ N6 ❌ KW-EXP-0001 NOT_SUPPORTED(朴素闭环) → N7 ✅(复用) → N8/N9 ⬜未触及
```

下一步最大杠杆：N4/N3b 诊断 —— 先用 per-step replanning diagnostics 区分“模型预测错”与“CEM/目标空间错”，再决定是否进入 uncertainty 或动力学训练实验。
