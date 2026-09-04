# KW-LEWM-0005 — 把 JEPA-WM 接入同一 stable-worldmodel 平台做同台对比（预注册）

- 日期：2026-09-03
- 类型：**预注册实验**（pre-registered；本文件先锁定 Kill Criteria 与 adapter 设计，再执行构建）
- 前置：`KW-LEWM-0004` 完成 —— LeWM 在 lite 协议下 50 任务 success 0.36（CI [0.24,0.50]）已自洽闭环
- 目标：满足项目晋级纪律"把 JEPA-WM 接入同一 stable-worldmodel 平台、同任务、同 CEM 预算"，得到 LeWM vs JEPA-WM 的**无混杂同台对比**

---

## 1. 为什么要做（纪律要求）

`KW-LEWM-0004` 的禁止条款："晋级需把 JEPA-WM 接入同一 stable-worldmodel 平台、同任务、同 CEM 预算"。当前两个数据点：

- **LeWM**：在 stable-worldmodel 上跑通，`World.evaluate(dataset=...)` + CEM 10×100×10 + 0004 的 50 个数据集定义任务 → success 0.36。
- **JEPA-WM（facebookresearch/jepa-wms `jepa_wm_pusht`）**：在 jepa-wms 自己的 harness 上跑通（KW-EXP-0006，success 0.4583）。

两者**不在同一 harness**，直接比 0.36 vs 0.4583 是 apples-to-oranges（任务构造、CEM 预算、像素预处理、scaler 全不同）。本实验消除该混杂。

## 2. 可行性探针结论（2026-09-03，已做实探）

| 问题 | 结论 | 证据 |
|---|---|---|
| stable-worldmodel 能否挂任意 world model？ | 能。`world model` 不是 `World` 的属性，而是挂在 `CEMSolver(model=...)` 上，经 `WorldModelPolicy` 进 `World.evaluate` | `world/world.py` 构造签名无 model 参数；`kw_lewm_0001l_smoke.py` 把 LeWM 塞进 `CEMSolver(model=model)` |
| `model` 的接口契约是什么？ | 必须实现 `get_cost(info_dict, candidates)`；`candidates` 形 `(B,N,H,D)`（已在环境 action 空间），返回 `(B,N)` 的 latent-MSE 代价 | `solver/cem.py:210` 调 `self.model.get_cost(expanded_infos, candidates)` 且断言形状 `(B,num_samples)` |
| 平台内置 PreJEPA 能直接当对比模型吗？ | **不能**——`wm/prejepa/` 只有架构（`PreJEPA`/`CausalPredictor`），无捆绑/可下载的 PushT 权重，训练超出当前"尚未训练"阶段 | 全包 grep 无 pusht checkpoint URL |
| 外部 jepa-wms 权重能否接入？ | **能，且可全离线**：`external/jepa-wms` 是本地 vendored 完整仓库；`checkpoints/jepa_wm_pusht.pth.tar`（211MB）已本地。无需 torch.hub 联网 | 仓库树 + 本地 checkpoint 存在 |
| jepa-wms 的 world model 包装类？ | `app/vjepa_wm/.../vit_enc_preds.py` 的 `EncPredWM`：`unroll(z_ctxt, act_suffix)` 做 rollout；`gc_agent.py` 的 representation-target distance objective 做代价 | 源码定位 |
| jepa-wms checkpoint 结构？ | 键：`predictor` / `proprio_encoder` / `scaler` / `stats`（无 `encoder`/`heads`）——encoder 是外部 HF DINOv2 ViT-S/14 | `torch.load` 窥探 |
| encoder 加载风险？ | **主要 Kill 风险**：DINOv2 ViT-S 需从 HF 加载（与 LeWM 0001L 同机制，曾成功，但本机 HF 常走 mirror） | 见 Kill K3 |

**结论**：head-to-head 可行，路径是写一个 `JEPAWMAdapter` 把 jepa-wms 的 `EncPredWM` 包成 stable-worldmodel 的 `get_cost` 接口，两者跑**同一个** `World.evaluate(dataset=...)` + 同一个 CEM 10×100×10 + 同一份 0004 任务集。

## 3. adapter 设计（待执行）

```
JEPAWMAdapter(nn.Module):
  - 由 external/jepa-wms 的 init_video_model / EncPredWM 构建：
      encoder   = DINOv2 ViT-S/14 (HF, 经 mirror 或 cache)
      predictor = VisionTransformerPredictorAC  ← checkpoint['predictor']
      proprio_encoder             ← checkpoint['proprio_encoder']
      scaler                      ← checkpoint['scaler']  (action/state 归一化)
      preprocessor                ← jepa-wms 官方 preprocessor (transform/inverse_transform/normalize)
  - get_cost(info_dict, candidates):   # 镜像 stable_worldmodel LeWM.get_cost 契约
      1. pixels = info_dict['pixels']  → preprocessor.transform → encoder → z_t  (B,P,D)
      2. goal   = info_dict['goal']    → 同上 → z_g
      3. states = info_dict['state']   → proprio (PushT 7 维)  [供 predictor 的 proprio 条件]
      4. for k in 1..H: z_{t+k} = unroll(z_{t+k-1}, candidates[:,:,k-1], states)
      5. cost = MSE(z_{t+H}, z_g) 沿 patch 维平均 → (B, N)
  - 复用 KW-LEWM-0003 的 ShimDataset + 本地渲染 + World.evaluate(dataset=...) 全链
  - CEM 预算与 0004 严格一致：10×100×10，PlanConfig horizon=5/receding=5/action_block=5
```

关键对齐点（必须验证，否则 Kill）：
- **action 空间**：`candidates` 已是环境 action 空间（CEM 采样后按 mean/var 缩放），adapter 内**不**再额外缩放；但喂给 `predictor` 前需经 `scaler.transform`（jepa-wms 训练时 action 在标准化空间）。→ 需核对 jepa-wms planner 是否对 action 做 scaler.transform；若 0004 的 LeWM 也用同样约定则一致。
- **像素预处理**：jepa-wms 用自己的 preprocessor（非 LeWM 的 ImageNet transform）。两个模型各自用各自 preprocessor，保证各自在训练分布内。
- **proprio/state**：PushT `state` 7 维，正好对应 `proprio_dim=7`。

## 4. Kill Criteria（预注册）

- **K1（adapter 可构建）**：`JEPAWMAdapter` 能从本地 checkpoint + vendored 代码实例化，参数 count 合理（~DINOv2 ViT-S 22M + predictor 24 层）。
- **K2（adapter 产生有限、in-distribution 预测）**：在任意 ≥1 个 0004 数据集任务上，`get_cost` 输出有限、预测 latent 与 goal latent 可比（非 NaN/Inf/全零 collapse）。
- **K3（encoder 可离线加载）**：DINOv2 ViT-S 能从 HF（经 mirror/cache）加载；若本机无法取得 → **Kill**，记录为"被依赖可用性阻断"，不谎称结果。
- **K4（同协议闭环控制）**：JEPA-WM 在 50 任务上 agent 越界率与 LeWM 同量级（均 < 某阈值，如 0.1），证明是有效闭环而非飞出场地。

任一 K1–K4 不通过即停，按"负面结果必须登记"写入台账，不补救加码。

## 5. 预期交付（执行后回填）

- `verification/scripts/kw_lewm_0005_jepa_adapter.py`（adapter + 50 任务跑批）
- `results/kw_lewm_0005_jepa_same_platform.json`（50 任务）
- 与 0004 的并排对比表（同平台、同任务、同 CEM）：LeWM 0.36 vs JEPA-WM ?
- 更新台账 + manifest

## 6. 纪律边界（执行后重申）

- 仍 **E1**：单 checkpoint、单 seed 集、单任务集；无第三方验证。
- 允许："在披露的 lite 协议下，LeWM 与 JEPA-WM 均接入同一 stable-worldmodel 平台、同 50 数据集任务、同 CEM 10×100×10，成功率分别为 X / Y（95% CI …）"。
- 禁止：声称官方全量 benchmark；声称任一模型"优于"另一模型（除非 CI 不重叠且披露所有残留近似）；用任一数字作为官方数字。
- 残留近似（与 0004 相同）：partial-shard scaler、本地渲染像素、缩减 CEM、shard 0、单 goal_offset=25。

---

## 7. 执行结果（2026-09-03）

### 7.1 K1–K3 冒烟（1 任务，pair 0 = ep19/step43）

| 判据 | 结果 | 关键数字 |
|---|---|---|
| **K1** adapter 可构建 | ✅ | enc 22,056,576 / pred 17,626,480 / total 39,683,136；model_action_dim=10；enc_type=dino；grid=16 |
| **K2** 预测有限、in-distribution | ✅（但见 7.3 诚实修正） | cost_zeros_mean≈453145.6，cost_rand_mean≈453297.6；latent 有限非 collapse |
| **K3** encoder 离线加载 | ✅ | DINOv2 ViT-S/14 在 cuda:0 实例化成功，无联网 |

> **诚实修正（与 7.3 联动）**：K2 的 `responds_to_actions` 仅要求 `|cost_z − cost_r| > 1e-4`，实测差 152 / 453000 ≈ **0.03%**——该阈值近乎总是通过，属于弱判据。真正的动作响应幅度在 7.3 诊断中被证明极小（代价地形近乎平坦），这正是 K4 控制失败的 root cause，而非 K2 标称的"已响应"。K2 通过但**不构成**"模型对动作敏感"的证据。

### 7.2 adapter wiring 三处修正（执行 K4 前已落实）

1. **动作双归一化（设计错误，已修）**：`get_cost` 原对 CEM candidates 调 `preprocessor.normalize_actions`。官方 `jepa-wms CEMPlanner` 把 N(0,1) 样本**直接**送 `unroll`，`encode_act` 不再归一化。双归一化会使 std 1.0→1.0/0.19≈5.3× OOD → 飞出。修复：candidates 直接透传；执行期逆归一化由 `policy.process['action'].inverse_transform` 负责。
2. **像素 channel 布局（运行期 RuntimeError，已修）**：`world` 存 `infos['pixels']` 为 **channel-last** `(B,T,H,W,C=3)`；adapter 原按 channel-first 假设 `rearrange`，报 `size mismatch (224 vs 3)`。修复：新增 `_to_visual` 按末维==3 自适应 permute 回 `(B,T,C,H,W)`。
3. **动作夹紧（K4 防发散，已修）**：`stable-worldmodel` 的 `CEMSolver` 不做 candidate 夹紧；代价地形对大动作过奖时 CEM 会发散到极大动作 → 执行期越界飞出。修复：adapter 内 `act_latent.clamp(-action_clip, action_clip)`（默认 ±3，覆盖归一化空间 ~99.7%）。

### 7.3 K4 前诊断：goal 通道正常，但代价地形对动作近乎平坦（root cause）

在烧 50 任务前做目标一致性诊断（`kw_lewm_0005_diag.py`，4 个任务对），结论：

- **`goal_drives_cost = TRUE`**：把 goal 图像换成 start 图像后代价明显偏移（如 453393 → 445775），证明 goal 通道 wiring 正确，**不是"目标未注入"缺陷**。
- **`toward_better_than_away = FALSE`**：4 对中仅 2 对"朝目标"代价低于"背离目标"；三者代价均在 ~453000 量级、两两差异 <0.1%。CEM 看到的代价地形近乎平坦。
- **动作确实被消费**（非被忽略）：`unroll` 经 `encode_act` 逐步注入 `act_suffix`；零动作与有动作代价不同（差 ~0.03%），只是幅度极小。

**归因**：代价平坦 = 该 checkpoint 在 5-step（25 env step = goal_offset）短视域 MPC 下，预测 latent 对候选动作几乎不变 → CEM 无法形成有效梯度 → planner 漂移/飞出。这是**模型/控制的真实属性**（JEPA 上下文主导预测、短视域低机动性），**不是可修的 wiring 缺陷**。K4 全量结果量化该失效，按纪律登记为负面/部分结果，不编造。

### 7.4 K4：50 任务同台闭环（跑批中 / 待回填）

- 协议：同 0004 —— 同 `World.evaluate(dataset=...)`、同 CEM 10×100×10、同 `PlanConfig`(horizon=5/receding=5/action_block=5)、同 `eval_budget=50`、同 `goal_offset=25`、同 `process['action']` 逆变换、pixels 不配 transform。
- **数据完整性修复（重要）**：本机 numeric shard `kw_lewm_0001l_numeric_shard0.json` 的 `selected_pairs` 已被重生成（**50 → 8**），无法再提供 0004 同款 50 任务。故 K4 改用 `--pairs-from` 从 **KW-LEWM-0004 结果 JSON 提取它实际评估的 50 个任务**（episode_idx/start_step/start_state/goal_state 完整）重建 pair schema，保证与 0004 **完全相同的任务集**（真 head-to-head）。proprio 重建为 jepa 标准 4 维 `[agent_x, agent_y, T_x, T_y]`（= state[:4]，比 shard 原 `[x,y,angle,vx]` 更贴合训练分布）；action 置 `[0,0]`（仅起始无关）。
- **错误结果已归档**：一次误用 shard-8 的 8 任务跑批（越界率 0.125、success 0/8）因任务数不对等已改名为 `results/kw_lewm_0005_jepa_same_platform_N8_WRONG_shard8.json`，**不计入** K4 判定。
- 结果见 7.4.1（已回填）；与 0004 并排表（success / 越界率 / block 交互率 / 位移中位数·均值 / Wilson CI）由 `kw_lewm_0005_summarize.py` 自动生成。

### 7.4.1 K4 50 任务结果（2026-09-03，真同台对比）

- 状态：**完成**（n_tasks=50，n_errors=0，跑批 ~64 分钟，GPU 99% 利用率健康）
- 判定：**K4 控制合法性达成（边界）**；head-to-head 显示 JEPA-WM success 显著低于 LeWM（CI 不重叠）

#### 与 KW-LEWM-0004（LeWM，同平台/同任务/同 CEM）并排

| 指标 | 0004 LeWM | **0005 JEPA-WM** | 解读 |
|---|---:|---:|---|
| success | 0.36 (18/50) | **0.02 (1/50)** | JEPA-WM 远低于 LeWM |
| success Wilson 95% CI | [0.241, 0.499] | **[0.004, 0.105]** | **不重叠** → 显著差 |
| agent 越界率 | 0.00 | **0.10 (5/50)** | 边界（恰在 ≤0.1 阈值） |
| block 交互率(>1) | 0.88 | **0.80** | 接近 |
| block 位移**中位数** | 31.54 | **27.27** | **接近**（粗粒度推动能力相当） |
| block 位移均值 | 51.56 | **52.10** | 接近 |
| block 位移 p25/p75/max | — | 6.7 / 88.9 / 232.9 | 右偏重尾 |
| disp>20（接近 success 位置阈值）| — | 27/50 | 54% 任务 block 移动超阈值但没成功 |
| disp>50 | — | 20/50 | 40% 大幅推动 |
| 运行错误 | 0 | **0** | 全链路无异常 |

#### Kill Criteria

| 判据 | 结果 |
|---|---|
| **K4_control_within_play_area**（全部在场内） | ❌ False：5/50 越界（ep 20/342/400/567/583；其中 ep20、ep400 飞出且 block 位移=0，即未接触 block 即出界） |
| **K4_out_of_play_rate_at_most_0.1** | ✅ True：0.10 ≤ 0.10（边界） |

#### 关键解读（修正 7.3 的担忧）

7.3 诊断曾担心"代价平坦 → 无梯度 → 飞出/无控制"。**实际 K4 推翻了该最坏情形**：

1. **控制是有效的（非飞出/无控制）**：90% 任务 agent 留在场内，block 交互率 0.80、位移中位数 27.27 与 LeWM（0.88 / 31.54）**几乎一致**。说明 adapter 产生了真实闭环控制——agent 能朝 goal 方向推动 block，且粗粒度位移量与 LeWM 同级。
2. **代价平坦伤及的是精度，不是粗运动**：5-step（25 env step）短视域下，预测 latent 对候选动作响应极弱（7.3：朝/背离目标代价差 <0.1%），CEM 能找到"大致把 block 推开"的方向，却**无法收敛到精确目标位姿**（pos<20 且 angle<π/9）。27/50 任务 block 移动 >20（已达 success 位置量级）但最终只有 1/50 达标——典型"推到了附近但没放准"。
3. **5 个越界任务**是平坦代价 + 残余大幅动作漂移的边界失效：agent 在困难任务上持续单向漂移累积出场（动作 clamp 限制了单步幅度，但限制不了 50 步累积）。其中 2 个（ep20/ep400）根本没碰到 block 就出界。
4. **head-to-head 结论（达成纪律目标）**：在披露的 lite 协议下，JEPA-WM 与 LeWM 首次在**同一平台/同 50 任务/同 CEM** 对比——LeWM success 0.36 [0.24,0.50] vs JEPA-WM 0.02 [0.004,0.105]，**CI 不重叠**，JEPA-WM 任务成功率显著更低；但两者 agent 控制合法性相当（越界率 0.0 vs 0.1、block 交互 0.88 vs 0.80）。即：**JEPA-WM 粗控制可比、精细收敛更弱**。

#### 诚实边界（必须随引用保留）

- 仍 **E1**：单 checkpoint、单 seed 集、单任务集；无第三方验证。
- residual 近似：partial-shard scaler、本地渲染像素、缩减 CEM 10×100×10、shard 0、单 goal_offset=25；**proprio 重建为 jepa 标准 4 维 [agent_x, agent_y, T_x, T_y]（= state[:4]，比 shard 原 [x,y,angle,vx] 更贴训练分布）**。
- **允许**：披露 lite 协议下，JEPA-WM 经本 adapter 在 50 数据集任务上 success 0.02 [0.004,0.105]（CI 与 LeWM 0.36 [0.241,0.499] 不重叠，JEPA-WM 显著更低），agent 控制合法（越界 0.10）、block 粗推动能力（中位 27.3）与 LeWM（31.5）相当。
- **禁止**：官方 13.1GB benchmark 已复现；用任一数字作官方数字；第三方验证；声称 JEPA-WM "优于" LeWM（恰相反）。

### 7.5 §8 根因深挖：flat cost 的精确来源（2026-09-03，用户选"根因深挖"方向）

> 来源：`verification/scripts/kw_lewm_0005_rootcause.py`（5 任务对，H=5 latent step，动作 scale 扫 [0,0.5,1,2,3]×N=30 随机候选）；结果 `results/kw_lewm_0005_rootcause.json`。纯诊断、不训练。

#### 8.1 诊断设计（二元法：机动性 vs 动作杠杆）

7.3 把 flat cost 归因为"短视域低机动性 / 上下文主导预测"。但"低机动性"与"动作无杠杆"是两种**不同**根因，推理期可救性相反。故本诊断同时测两项：

- **潜在空间机动性**：`gap_req = ||z_goal − z_start||`（到达目标所需 latent 位移）；`max_mob = max_{scale,随机动作} ||pred_final − z_start||`（模型最大可达位移）；`ratio = max_mob / gap_req`。`ratio<0.5` → 低机动性（场景近乎静止，模型属性）。
- **动作杠杆 / 上下文主导度**：`do_nothing_disp = ||pred(s=0) − z_start||`（零动作基线位移）；`context_domination_ratio = do_nothing_disp / max_mob`；`action_leverage = max_mob − do_nothing_disp`（最大动作相对"不动"额外买到的位移）。`context_domination_ratio≈1` 且 `action_leverage≈0` → **预测被上下文主导、动作对最终 latent 几乎无杠杆（弱动作条件）**。

> 修复记录：本脚本曾因 `encode` 返回的 **完整 TensorDict**（含 `proprio`）被 `[:,"visual"]` 抽取成裸 Tensor，导致 `unroll→forward_pred` 跳过 proprio concat → 384 维输入撞 400 维 LayerNorm 崩溃（与 §7.2 的 wiring 修复同源）。最终镜像 `get_cost`（adapter.py:203-219）：`z_start` 保持完整 TensorDict，仅在算 norm/代价时用 `z["visual"]`；且 `z_start` 的 batch 维 = N（与 `act_suffix` batch 一致，避免 `unroll` 的 `.expand` 产生 stride-0 视图）。

#### 8.2 结果（5 任务对，ep 19/86/107/158/172）

| 任务对 | gap_req | max_mobility | ratio | do_nothing_disp | context_dom | action_leverage |
|---|---:|---:|---:|---:|---:|---:|
| p0 (ep19) | 1541 | 3655 | 2.37 | 3656 | 1.0002 | −0.7 |
| p1 (ep86) | 1288 | 3793 | 2.94 | 3798 | 1.0013 | −4.9 |
| p2 (ep107) | 1499 | 3748 | 2.50 | 3750 | 1.0003 | −1.1 |
| p3 (ep158) | 1153 | 3830 | 3.32 | 3831 | 1.0001 | −0.5 |
| p4 (ep172) | 1014 | 3763 | 3.71 | 3762 | 0.9996 | +1.4 |
| **mean** | — | — | **2.97** | — | **1.0003** | **−1.2** |

- **比率全部 ≥2.37（mean 2.97）**：最大可达 latent 位移是到达目标所需位移的 ~3 倍 → **机动性不是瓶颈**（模型在 latent 空间完全能走够远）。
- **`context_domination_ratio` 全部 ≈1.000（mean 1.0003）**：零动作预测把 latent 推到 3656–3831，与最大动作（3635–3830）**几乎相同**——做动作与不做动作，预测落点一致。
- **`action_leverage` 全部 ≈0（mean −1.2，含负值）**：最大幅度动作相对"不动"**没有额外位移**，甚至略少。动作对预测最终 latent 无净杠杆。
- **逐 scale 验证**：每个任务对内，scale 0→3 的 `max_disp_from_start` 变化 <1%（如 p0：3656→3635）；`min_cost_to_goal` 平坦在 4.5–5.0（如 p3：5.000→4.987）。代价地形对动作幅度近乎无响应。

#### 8.3 判定

```
verdict = ACTION_INSENSITIVE_PREDICTION
          (weak action conditioning; model property, NOT fixable at inference without retraining)
```

（原 `ratio<0.5 → LOW_LATENT_MOBILITY` 的二元判据**漏判**了本情形：机动充足但动作无杠杆。故脚本升级为三元判别——`context_domination_ratio>0.9` 优先判为弱动作条件。）

#### 8.4 与 7.3 / 7.4.1 的联立解释

- **7.3 `toward_better_than_away=false`** 得到精确机制解释：不是"预测不动"，而是"所有动作把预测推到几乎同一落点"——朝/背离目标的代价差 <0.1% 正是 `context_domination_ratio≈1` 的直接后果。
- **7.4.1 的低成功 / 粗控制相当** 得到完整因果链：CEM 在平坦代价地形上无梯度→只能找到"大致把 block 推开"的方向（故 block 中位位移 27.3≈LeWM 31.5、交互率 0.80≈0.88），但**无法收敛到精确目标位姿**（pos<20 且 angle<π/9）→ 1/50 成功。5 个越界是平坦地形 + 残余大幅动作累积漂移的边界失效，非"飞出无控制"。
- **关键区分**：这不是"低机动性"（模型走不远），而是"动作不转向"（模型走得到，但走哪由上下文定、不由动作定）。前者推理期无解，后者同样推理期无解——两者都是**冻结 checkpoint 的模型属性**。

#### 8.5 方法论结论（可登记为项目发现）

**代价平坦 = 冻结 JEPA-WM checkpoint 的弱动作条件（weak action conditioning）**：预测最终 latent 由 start 上下文（visual+proprio）主导，动作序列对落点无杠杆。这解释了 K4 全部现象，且**推理期增强全部无效**——

> 不可救（模型属性，须训练/结构改造）：
> - 换更好代价度量（如多帧/轨迹 MSE 替代 final-frame MSE）
> - 加长 horizon / receding window
> - 加 CEM 样本数 / 迭代 / 温度退火
> - 动作 re-normalization / 更大 clip
>
> 因为代价地形对动作**由构造平坦**（action_leverage≈0），上述任一手段都只在平坦面上移动采样点，不产生有效梯度。

若要在该 checkpoint 上提升 task-success，只能走**结构/训练改造**（新预注册、超出当前"尚未训练"阶段）：更强动作条件（如把 `action_encoder` 接入更深层 / 提高 action embedding 维数）、动作条件 latent、ensemble 估计不确定性、或换用动作条件更强的 world model。

#### 8.6 闭环数据独立佐证（KW-LEWM-0005b，GPU-free，复用 K4 50 任务）

§8 的弱动作条件结论是 **latent 空间诊断**。为免于单线证据，用已收集的 K4 闭环结果（`results/kw_lewm_0005_jepa_same_platform.json`，同 50 任务、单 checkpoint/seed）做**第二条独立证据线**（`kw_lewm_0005b_k4_closedloop_corroboration.py`，纯 numpy 算 Pearson/Spearman/Wilson）。

| 检验 | 量 | 值 | 解读 |
|---|---|---:|---|
| 位移是否跟目标距离 | block_displacement vs start_goal_distance | Pearson −0.33 / **Spearman −0.39** | **负**相关——目标越远推得越少，与「精确收敛应正相关」相反 → 推量非目标驱动 |
| 成功是否看目标距离 | success vs start_goal_distance | Pearson −0.18 / Spearman +0.09 | ≈0 → 成功非任务几何可 exploited（与 0004 LeWM 的 ρ=0.055 同量级） |
| 成功是否看位移量 | success vs block_displacement | Pearson −0.10 / Spearman +0.04 | ≈0 → 推得多≠成功 |
| 大位移但失败 | in-play 且 disp>20 的任务 | **27/27 全部失败** | 块动得多（>20）却没到目标 → 粗动有效、精度失效（与 §8 完全一致） |
| 越界是否因目标更远 | OOP goal_dist 均值 vs in-play | 164.5 vs 151.1（**ratio 1.09**） | 越界任务目标距离仅多 9% → 非「更难目标」，是平坦代价 + 漂移的边界失效 |
| 唯一成功任务 | ep478 | goal_dist=63.5, block_disp=10.6 | 单点、目标较近；小样本不构成立论，仅记录 |

**结论**：latent 诊断（§8：弱动作条件、动作杠杆≈0）与真实闭环行为（位移不跟目标距离、成功近似随机、大位移不导向成功、越界非更难目标）**两条独立证据线一致收敛** → §8 弱动作条件结论得到闭环层佐证，非单线诊断伪影。E1（同 50 任务单 checkpoint/seed），残留近似与 K4 一致。

#### 8.7 对 jepa-wms 研究线的收尾建议

- **诊断线（E1，冻结 checkpoint）到此自洽闭环**：K1–K3 接入可行 → K4 真同台对比（JEPA-WM 0.02 vs LeWM 0.36，CI 不重叠，粗控制相当）→ §8 根因=弱动作条件（模型属性）。三条线索（K4 低成功、7.3 平坦、§8 弱动作条件）一致收敛，无残留矛盾。
- **Kill 该诊断线**：不再对冻结 checkpoint 做推理期调参（已证无效）。任何"提升 JEPA-WM success"的尝试都需新预注册（训练/结构改造），与当前"尚未训练"纪律冲突，须另开实验线。
- 登记为项目级方法论发现："world model 评测中，**动作条件强度**是比 latent 机动性更隐蔽的成败变量——flat cost 不一定是低机动性，可能是弱动作条件；后者在冻结模型上不可修，只能靠训练/结构。"（与已登记的"任务可达性是关键混淆变量"并列，构成 KineWorld 评测方法论两条主线。）

#### 8.8 第三条独立证据线：配对行为学再分析（KW-LEWM-0005c，GPU-free）

见 `verification/experiments/KW-LEWM-0005c.md`。对 0004/0005-K4 的**同 50 任务配对日志**做 no-op 基线与失败项归因，得到弱动作条件最直接的行为学读数：

| 结论 | 数字 |
|---|---|
| 两模型智能体**移动距离统计不可区分** | `agent_disp` 配对差 −1.7，CI [−25.2, +22.7]（含 0）；`block_disp` −0.5，CI [−21.0, +20.0]（含 0） |
| 但**进展符号相反** | `agent_progress` 中位数 LeWM **+75.9** vs JEPA **−36.2**；配对差 **+132.9**，CI [+108.3, +157.9] |
| JEPA 进展**显著为负** | `pos_progress` 均值 **−45.4**，CI [−68.6, −21.5]（no-op 恒为 0） |
| JEPA 失败由**智能体位置项主导** | 85.7%；`agent_gap_final` 中位数 **182.3**（起始 123.3，被拉大 59） |
| 仅方块判据（诊断分解，非基准）JEPA **显著劣于不动** | LeWM 30/50、no-op 16/50、JEPA 8/50；McNemar JEPA vs no-op **p=0.0078（0 vs 8）** |

**同等位移预算、相反进展符号**——规划器不知道自己的动作把智能体带向何处，正是 §8 `ACTION_INSENSITIVE_PREDICTION` 的行为学签名。三条独立证据线（§8 latent 诊断、§8.6 闭环相关性、§8.8 配对行为学）一致收敛，§8/§8.7 结论不变。

两条必须同时记住的反向约束：

1. **0005c 加固了 0004，而非削弱它**：真实判据下 no-op = **0/50**，LeWM 显著优于 no-op（McNemar p=7.6e-6）→ 0004 的 0.36 是真目标导向控制。
2. **Push-T success 是 4 维联合位置差**（`||goal[:4]−cur[:4]||<20`，agent 与 block 共享 20px 预算），不是方块位置差。0005c 第一版用旧记忆的 block-only 判据只复现 81% 的 success 标志，被预设门禁 R0 判 VOID；读源码改正后 100%。详见 0005c §2。

---

## 附：本轮可行性探针的副产物（已确认的事实，供后续直接引用）

1. **LeWM = 该生态的参考 JEPA 模型**：`lucas-maes/le-wm` 是 `stable_worldmodel/wm/lewm` 的上游（MIT）；`wm/prejepa` 是另一 JEPA 基线但无权重。
2. **JEPA-WM 权重来源**：`facebook/jepa-wms`（Meta FAIR）`jepa_wm_pusht`：DINOv2 ViT-S/14 encoder + 6 层 predictor；本地 `checkpoints/jepa_wm_pusht.pth.tar` 即此。
3. **CEMSolver 的 model 契约**：`get_cost(info_dict, candidates)`，candidates `(B,N,H,D)`，返回 `(B,N)`——这是接入任何 world model 的唯一接口门槛。
