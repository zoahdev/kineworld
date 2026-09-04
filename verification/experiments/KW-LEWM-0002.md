# KW-LEWM-0002 — action scaler 是否为 LeWM 闭环控制失效的机制（单变量消融）

- 日期：2026-09-03
- 类型：**单变量消融**（无训练，仅推理）
- 状态：**完成**（8 episodes × 2 条件，2026-09-03）
- 判定：**`ACTION_SCALER_NECESSARY_NOT_SUFFICIENT`**（K1 通过、K2 失败）
- 前置：`KW-LEWM-0001L`（Codex，2026-09-02）判定 `CLOSED — INFRASTRUCTURE PASS / VALID CONTROL NOT SUPPORTED`
- 脚本：`verification/scripts/kw_lewm_0002_scaler_ablation.py`
- 结果：`results/kw_lewm_0002_scaler_ablation.json`

> 协议与 Kill Criteria 在正式批次执行前已写入脚本（`kill_criteria` 字段）。本文档结果节在数字产生后回填，不修改判据。

---

## 1. 它要回答的问题

KW-LEWM-0001L 成功严格加载了官方 LeWM Push-T 权重（18,034,478 参数），但闭环控制结果被判为**无效**：agent 从 `(281.87, 104.94)` 移动到 `(120.21, 1202.36)`（y 远超 512 的合法场地），而 block 完全没动。该实验据此 Kill 掉 Stage B/C，结论是"没有官方数据集拟合的 `StandardScaler`，控制结果不可用"。

但 0001L 只观察到"失效"，**没有验证失效机制**。本实验补上这一步：把失效归因从"缺数据集"精确化到"缺 action 逆变换"这一个变量。

### 1.1 机制假说（来自代码，不是猜测）

`stable_worldmodel/solver/cem.py` 采样候选动作用的是：

```python
candidates = torch.randn(current_bs, self.num_samples, self.horizon, self.action_dim, ...)
```

即 **N(0, 1)**，且**不做裁剪**。`policy.py` 在动作送进环境前会做：

```python
if 'action' in self.process:
    action = self.process['action'].inverse_transform(action)
```

官方 evaluator 用训练集拟合 `StandardScaler`，把 CEM 输出的标准化动作逆变换回环境单位。KW-LEWM-0001L 传了 `process={}`，等于让 CEM 直接把 N(0,1) 的采样值当作环境动作执行。

### 1.2 量化的量级错配

从公开 parquet 分片的 **80,384 行真实数据**估计出的官方动作分布（见 §3）：

| | CEM 默认采样 | 专家动作真实分布 | 倍数 |
|---|---:|---:|---:|
| 均值 | 0 | ≈ -0.036 / 0.007 | — |
| 标准差 | **1.0** | **≈ 0.188 / 0.197** | **5.3×** |

CEM 的探索范围是专家动作分布的 5.3 倍。这就是 agent 累积飞出场地的量化原因。

---

## 2. 实验设计（单变量）

**唯一被操纵的变量**：`process['action']` 变换。

| 条件 | `process['action']` | 含义 |
|---|---|---|
| `identity` | 无（空 `process`） | 复现 KW-LEWM-0001L 的失效条件 |
| `scaler` | 由真实数据统计拟合的 StandardScaler | 修复条件 |

**保持不变**（两个条件逐位相同）：

1. 官方 LeWM Push-T checkpoint（严格加载，18,034,478 参数）
2. 随机环境任务，seed 42..49，**两条件用同一组 seed**
3. 缩减 CEM：`n_steps=10, num_samples=100, topk=10`
4. `PlanConfig`：horizon=5, receding_horizon=5, action_block=5, history_len=1, warm_start=True
5. `max_episode_steps=50`、`image_shape=(224,224)`、ImageNet 归一化

### 2.1 观测指标

| 指标 | 用途 |
|---|---|
| `agent_out_of_play_area` | 终态 agent 坐标是否越出 `[0, 512]²`——**主判据** |
| `block_displacement` | block 质心位移——控制是否产生物理效果 |
| `success` | 官方 `eval_state`：pos_diff < 20 且 angle_diff < π/9 |

### 2.2 Kill Criteria

- **K1（scaler 起作用）**：`scaler` 条件的越界率**严格低于** `identity` 条件。若不满足 → 缺失的 action 逆变换**不是**失效机制，本线停止，作为负面结果登记。
- **K2（scaler 条件下控制合法）**：`scaler` 条件**零越界**。若仍有越界 → lite 协议仍不可用于控制结论。

---

## 3. action 统计的来源与边界

`results/kw_lewm_0001l_numeric_shard0.json`，由 `verification/scripts/kw_lewm_parquet_numeric_probe.py` 经 **HTTP Range** 从公开 parquet 分片读取：

| 项 | 值 |
|---|---:|
| 远端对象 | 502,089,201 bytes（479 MB，含 pixels 列） |
| 实际传输 | **5.68 MB**（只读 numeric 列，从不请求 pixels） |
| 读取行数 | 80,384 |
| content SHA-256 | `a538eb6a20fcaca4189b71bde8d9593b38597003950e199ce587e57c720ab3c8` |

统计量（总体标准差，ddof=0）：

| 字段 | 维 | mean | scale |
|---|---:|---|---|
| **action** | 2 | -0.0359, 0.0071 | **0.1875, 0.1970** |
| proprio | 4 | 267.49, 270.46, -13.36, 2.58 | 117.65, 95.38, 69.70, 73.26 |
| state | 7 | 267.49, 270.46, 267.68, 269.26, 2.14, -13.36, 2.58 | 117.65, 95.38, 92.73, 59.84, 1.89, 69.70, 73.26 |

**边界（必须遵守）**：这是 **partial shard 的 80,384 行**，不是官方完整训练集，因此该 scaler 是**近似**，不等于官方全量 `StandardScaler`。本实验只用它做机制归因（相对比较：identity vs scaler），**不**用它声称官方 benchmark 复现。

---

## 4. 结果（2026-09-03，8 episodes × 2 条件，seeds 42..49 两条件相同）

- 状态：**完成**
- 判定：**`ACTION_SCALER_NECESSARY_NOT_SUFFICIENT`**

### 4.1 汇总

| 指标 | `identity` | `scaler` |
|---|---:|---:|
| agent 越界率 | **8/8 (100%)** | **2/8 (25%)** |
| agent 终态坐标 abs max | 3,960.4 | 595.9 |
| block 位移均值 | 1.81 | 35.00 |
| **block 位移中位数** | **0.00** | **0.00** |
| block 位移 > 0 的 episode 数 | 2/8 | **3/8** |
| success | 0/8 | 0/8 |
| 运行错误 | 0 | 0 |

### 4.2 Kill Criteria

| 判据 | 结果 |
|---|---|
| **K1（scaler 起作用）** | ✅ **通过**：越界率 1.00 → 0.25；越界幅度 3,960 → 596（6.6× 收敛） |
| **K2（scaler 条件下控制合法）** | ❌ **失败**：仍有 2/8 越界，且 block 位移中位数为 0 |

### 4.3 逐 episode（`scaler` 条件）

| ep | agent 起点 → 终点 | 越界 | block 位移 | 起点 goal 距离 |
|---|---|---|---:|---:|
| 0 | (282,105) → (297,349) | 否 | **208.36** | 360.9 |
| 1 | (254,304) → (418,166) | 否 | **16.20** | 213.1 |
| 2 | (127,313) → (-481,457) | **是** | 0.00 | 109.2 |
| 3 | (107,251) → (37,235) | 否 | 0.00 | 115.5 |
| 4 | (294,251) → (216,218) | 否 | **55.46** | 259.5 |
| 5 | (354,304) → (497,168) | 否 | 0.00 | 188.9 |
| 6 | (378,292) → (161,231) | 否 | 0.00 | 266.8 |
| 7 | (270,156) → (-596,50) | **是** | 0.00 | 253.2 |

`identity` 条件全部 8 个 episode 越界，终态坐标量级荒谬（-3,960 / 2,725 / -2,725），block 位移 6/8 为 0。

### 4.4 结论

1. **缺失的 action 逆变换确实是失效机制的一部分**（K1 通过）：修复后 agent 从"必然飞出场地"变成"6/8 留在场内"，且 3/8 episode 产生了真实物理交互（block 位移 208 / 55 / 16）。KW-LEWM-0001L 的失效归因被精确化。
2. **但它不是充分条件**（K2 失败）：仍有 2/8 越界，且 5/8 的 block 完全没动。
3. **不能把 0/8 解读为模型能力差**：任务的 goal 是**随机采样**的，起点 goal 距离高达 360.9，且要求 block 位姿精确对齐（pos_diff<20 且 angle_diff<π/9）。官方协议用的是**同一 episode 内相隔 25 步的真实状态对**，天然可达。这是本实验尚未消除的最后一个混淆项。

### 4.5 与 2-episode 冒烟的差异（自我更正）

先用 2 episodes 冒烟时，`scaler` 条件给出"越界 0/2、block 位移均值 112.28"，看似完全修复。扩展到 8 episodes 后，block 位移**中位数**为 0，冒烟的乐观数字由 ep0 的 208.36 单点拉高。

**教训**：本实验重犯了已在项目 MEMORY.md 登记的统计错误——**用小样本均值描述重尾分布**。`block_displacement` 在多数 episode 为 0、少数 episode 达 200+，是典型零膨胀重尾，必须以中位数为准。已同步在脚本 `summarize()` 中同时输出均值与中位数。

---

## 5. 允许的与禁止的陈述

**允许**（在结果支持时）：

> 在固定任务与规划配置下，使用从公开数据估计的 action 逆变换，可将 LeWM 闭环控制的 agent 越界率从 X 降到 Y，block 位移从 A 提升到 B；缺失该逆变换是 KW-LEWM-0001L 控制失效的机制。

**禁止**：

- 官方 LeWM benchmark 已复现；
- LeWM 优于/劣于 JEPA-WM 或任何竞品（本实验无竞品对照组）；
- 该近似 scaler 等价于官方全量 scaler；
- 第三方验证（当前仍为 E1 内部证据）。

---

## 6. 与 JEPA-WM 对比的前置障碍（尚未解决）

原始目标 KW-LEWM-0001 是"LeWM vs JEPA-WM 同台对比"。本实验**不**达成该目标，且必须指出两个方法当前**不可直接比较**：

| 维度 | LeWM（本线） | JEPA-WM（KW-EXP-0006） |
|---|---|---|
| 环境实现 | `swm/PushT-v1`（stable-worldmodel） | jepa-wms 自带 Push-T |
| 观测分辨率 | 224×224 | 96×96 |
| 规划 horizon | 5 | 6 |
| CEM | 10×100×10（缩减） | 10×100×10 |
| 任务采样 | 随机环境任务 | 数据集 episode |
| episode 数 | 8 | 96 |

只有把 **JEPA-WM 接入同一 `stable-worldmodel` 平台**、用同一组任务与同一 CEM 预算跑，对比才成立。这是后续独立实验，不在本实验范围内。
